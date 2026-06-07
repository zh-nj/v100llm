#pragma once

#include <cstdio>
#include <cstdlib>

#include <math_constants.h>

#include <cutlass/half.h>

#include "sm70/common/mma_884.h"

#include "config.h"
#include "params.h"
#include "utils.h"

namespace sm70::prefill::sparse {

namespace detail {

__device__ __forceinline__ float bf16_to_float(const cutlass::bfloat16_t value) {
    return static_cast<float>(value);
}

__device__ __forceinline__ const cutlass::bfloat16_t* q_ptr_for(
    const SparseAttnFwdParams &params,
    int q_idx,
    int head_idx) {
    return params.q
         + q_idx * params.stride_q_s_q
         + head_idx * params.stride_q_h_q;
}

__device__ __forceinline__ const cutlass::bfloat16_t* kv_ptr_for(
    const SparseAttnFwdParams &params,
    int token_idx,
    int kv_head_idx) {
    return params.kv
         + token_idx * params.stride_kv_s_kv
         + kv_head_idx * params.stride_kv_h_kv;
}

__device__ __forceinline__ const int* indices_ptr_for(
    const SparseAttnFwdParams &params,
    int q_idx,
    int kv_head_idx) {
    return params.indices
         + q_idx * params.stride_indices_s_q
         + kv_head_idx * params.stride_indices_h_kv;
}

__device__ __forceinline__ cutlass::bfloat16_t* out_ptr_for(
    const SparseAttnFwdParams &params,
    int q_idx,
    int head_idx) {
    return params.out
         + (q_idx * params.h_q + head_idx) * params.d_v;
}

__device__ __forceinline__ float* max_logits_ptr_for(
    const SparseAttnFwdParams &params,
    int q_idx,
    int head_idx) {
    return params.max_logits + q_idx * params.h_q + head_idx;
}

__device__ __forceinline__ float* lse_ptr_for(
    const SparseAttnFwdParams &params,
    int q_idx,
    int head_idx) {
    return params.lse + q_idx * params.h_q + head_idx;
}

__device__ __forceinline__ int valid_topk_for_query(
    const SparseAttnFwdParams &params,
    int q_idx) {
    if (params.topk_length == nullptr) {
        return params.topk;
    }
    return max(0, min(params.topk_length[q_idx], params.topk));
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ float shared_k_value(
    const cutlass::half_t *kv_tile,
    int local_token_idx,
    int dim) {
    return static_cast<float>(kv_tile[local_token_idx * HEAD_DIM_QK + dim]);
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ float qk_score_token_from_shared(
    const cutlass::bfloat16_t *q,
    const cutlass::half_t *kv_tile,
    int local_token_idx,
    float sm_scale) {
    float acc = 0.0F;
#pragma unroll 1
    for (int dim = 0; dim < HEAD_DIM_QK; ++dim) {
        acc += bf16_to_float(q[dim]) * shared_k_value<HEAD_DIM_QK>(kv_tile, local_token_idx, dim);
    }
    return acc * sm_scale;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ float qk_score_token_from_shared_warp(
    const cutlass::bfloat16_t *q,
    const cutlass::half_t *kv_tile,
    int local_token_idx,
    int lane,
    float sm_scale) {
    float partial = 0.0F;
#pragma unroll 1
    for (int dim = lane; dim < HEAD_DIM_QK; dim += 32) {
        partial += bf16_to_float(q[dim]) * shared_k_value<HEAD_DIM_QK>(kv_tile, local_token_idx, dim);
    }
    return warp_reduce_sum(partial) * sm_scale;
}

__device__ __forceinline__ half half_from_float(float value) {
    return __float2half_rn(value);
}

__device__ __forceinline__ half probability_half_from_score(
    float score,
    float row_lse,
    float sink_scale) {
    return score != -CUDART_INF_F ? half_from_float(expf(score - row_lse) * sink_scale) : half_from_float(0.0F);
}

__device__ __forceinline__ half probability_half_from_online_score(
    float score,
    float row_max) {
    return score != -CUDART_INF_F ? half_from_float(expf(score - row_max)) : half_from_float(0.0F);
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void load_mma884_q_fragment(
    const cutlass::bfloat16_t *q,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_q) {
    const int q_row = (lane & 8) + (lane & 3) + (lane >> 4) * 4;
    // H13a: vectorized uint2 (64-bit = 4 halves) smem load.
    // `q` is the per-head smem slice after H10, guaranteed 8-byte
    // aligned; `dim_base` is a multiple of 4 from the caller loop.
    if (q_row == 0) {
        const uint2 packed = *reinterpret_cast<const uint2*>(&q[dim_base]);
        // Four bfloat16 values packed into two uint32 halves.
        // Each uint16 word is a bfloat16 that we upcast to half via
        // fp32 (Volta has no native bf16 -> half conversion).
        const uint16_t bf0 = static_cast<uint16_t>(packed.x & 0xFFFFu);
        const uint16_t bf1 = static_cast<uint16_t>(packed.x >> 16);
        const uint16_t bf2 = static_cast<uint16_t>(packed.y & 0xFFFFu);
        const uint16_t bf3 = static_cast<uint16_t>(packed.y >> 16);
        // bf16 -> float: zero-extend 16 bits into fp32 mantissa.
        auto bf16_u16_to_float = [] __device__ (uint16_t w) -> float {
            uint32_t bits = static_cast<uint32_t>(w) << 16;
            float out;
            __builtin_memcpy(&out, &bits, sizeof(out));
            return out;
        };
        frag_q[0] = __float2half_rn(bf16_u16_to_float(bf0));
        frag_q[1] = __float2half_rn(bf16_u16_to_float(bf1));
        frag_q[2] = __float2half_rn(bf16_u16_to_float(bf2));
        frag_q[3] = __float2half_rn(bf16_u16_to_float(bf3));
    } else {
        const half zero = __float2half_rn(0.0F);
        frag_q[0] = zero;
        frag_q[1] = zero;
        frag_q[2] = zero;
        frag_q[3] = zero;
    }
}

// H13c: half-Q variant. Reads fp16 Q directly from smem (Q was
// converted bf16 -> fp16 once at the staging prologue). Same
// predicated lane pattern as the bf16 version.
template<int HEAD_DIM_QK>
__device__ __forceinline__ void load_mma884_q_fragment_half(
    const half *q,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_q) {
    const int q_row = (lane & 8) + (lane & 3) + (lane >> 4) * 4;
    if (q_row == 0) {
        const uint2 packed = *reinterpret_cast<const uint2*>(&q[dim_base]);
        *reinterpret_cast<uint2*>(&frag_q[0]) = packed;
    } else {
        const half zero = __float2half_rn(0.0F);
        frag_q[0] = zero;
        frag_q[1] = zero;
        frag_q[2] = zero;
        frag_q[3] = zero;
    }
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void load_mma884_k_fragment(
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_k) {
    const int token_offset = (lane >> 4) * 4 + (lane & 4) * 2 + (lane & 3);
    const int local_token_idx = token_base + token_offset;
    const bool valid_token = local_token_idx < tile_count && token_refs[local_token_idx] != 0;
    // H13a: vectorized uint2 (64-bit = 4 halves) smem load.
    // `kv_tile` row stride is HEAD_DIM_QK * sizeof(half) bytes, so
    // &kv_tile[local_token_idx * HEAD_DIM_QK + dim_base] is 8-byte
    // aligned when `dim_base` is a multiple of 4 (guaranteed by the
    // outer loop `dim_base += 4`).
    if (valid_token) {
        const cutlass::half_t *row_ptr =
            kv_tile + local_token_idx * HEAD_DIM_QK + dim_base;
        const uint2 packed = *reinterpret_cast<const uint2*>(row_ptr);
        // Four half values packed into two uint32. Store directly.
        *reinterpret_cast<uint2*>(&frag_k[0]) = packed;
    } else {
        const half zero = __float2half_rn(0.0F);
        frag_k[0] = zero;
        frag_k[1] = zero;
        frag_k[2] = zero;
        frag_k[3] = zero;
    }
}

__device__ __forceinline__ void store_mma884_qk_row0_scores(
    const flash_mla::sm70::Array<float, 8> &score_frag,
    float *scores,
    const int *token_refs,
    int tile_start,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
#pragma unroll
    for (int s1 = 0; s1 < 2; ++s1) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int s0 = 0; s0 < 2; ++s0) {
                const int q_row = (lane & 8) + (lane & 1) + (lane >> 4) * 4 + q * 2;
                const int s_col = (lane & 4) * 2 + (lane & 2) + s1 * 4 + s0;
                const int local_token_idx = token_base + s_col;
                if (q_row == 0 && local_token_idx < tile_count) {
                    const int topk_idx = tile_start + local_token_idx;
                    scores[topk_idx] = token_refs[local_token_idx] != 0
                        ? score_frag[s1 * 4 + q * 2 + s0] * sm_scale
                        : -CUDART_INF_F;
                }
            }
        }
    }
}

__device__ __forceinline__ void load_mma884_p_fragment(
    const float *scores,
    int tile_start,
    int token_base,
    int tile_count,
    int lane,
    float row_lse,
    float sink_scale,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    const int p_row = (lane & 8) + (lane & 3) + (lane >> 4) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int local_token_idx = token_base + i;
        const int topk_idx = tile_start + local_token_idx;
        frag_p[i] = p_row == 0 && local_token_idx < tile_count
            ? probability_half_from_score(scores[topk_idx], row_lse, sink_scale)
            : half_from_float(0.0F);
    }
}

__device__ __forceinline__ void load_mma884_online_p_fragment(
    const float *tile_scores,
    int token_base,
    int tile_count,
    int lane,
    float row_max,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    const int p_row = (lane & 8) + (lane & 3) + (lane >> 4) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int local_token_idx = token_base + i;
        frag_p[i] = p_row == 0 && local_token_idx < tile_count
            ? probability_half_from_online_score(tile_scores[local_token_idx], row_max)
            : half_from_float(0.0F);
    }
}

// H11: same interface, but reads pre-computed probabilities from an
// fp16 smem cache populated once per tile. No expf in the hot loop.
__device__ __forceinline__ void load_mma884_online_p_fragment_cached(
    const half *p_cache,          // [tile_count] fp16, already expf-scaled
    int token_base,
    int tile_count,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    // H14b: vectorized uint2 (64-bit = 4 halves) smem load, emits as
    // ld.shared.v2.b32 (LDS.U.64). Broadcast to all 32 lanes of the
    // warp; per-lane predicated select keeps the value only when
    // (p_row == 0 && local_token_idx < tile_count).
    //
    // Alignment: token_base is a multiple of 4 (outer PV loop steps
    // by 4); p_cache is half* -> &p_cache[token_base] is 8-byte
    // aligned. p_cache is sized K_TILE=32 halves so token_base+3 <=
    // 31 -> in-bounds even when token_base+3 >= tile_count.
    const int p_row = (lane & 8) + (lane & 3) + (lane >> 4) * 4;
    const half kZeroH = __float2half_rn(0.0F);
    half loaded[4];
    const uint2 packed = *reinterpret_cast<const uint2*>(&p_cache[token_base]);
    *reinterpret_cast<uint2*>(loaded) = packed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int local_token_idx = token_base + i;
        frag_p[i] = p_row == 0 && local_token_idx < tile_count
            ? loaded[i]
            : kZeroH;
    }
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void load_mma884_v_fragment(
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_v) {
    // H4: read kv_tile directly as half, avoiding a half->float->half
    // round-trip that was hidden behind the `shared_k_value` helper.
    // H14a: vectorized uint2 (64-bit = 4 halves) smem load, emits as
    // ld.shared.v2.b32 (LDS.U.64). After the load, we mask out per-
    // element entries where dim >= HEAD_DIM_V.
    //
    // Alignment: dim_base is a multiple of 16 (outer loop stride);
    // dim_offset is a multiple of 4 ({0,8,16,24}); so base_dim is a
    // multiple of 4 -> 8-byte aligned.
    //
    // Boundary safety: when base_dim + 3 >= HEAD_DIM_V the uint2 read
    // overflows kv_tile's logical bound. Since smem is allocated with
    // q_smem/p_cache_all immediately following kv_tile, the over-read
    // hits those slabs (garbage but allocated memory, no OOM). The
    // per-element mask below discards the garbage.
    const int token_offset = lane & 3;
    const int local_token_idx = token_base + token_offset;
    const int dim_offset = (lane >> 4) * 4 + (lane & 4) * 2;
    const bool valid_token = local_token_idx < tile_count && token_refs[local_token_idx] != 0;
    const half *kv_half = reinterpret_cast<const half*>(kv_tile);
    const half kZeroH = __float2half_rn(0.0F);
    const int base_dim = dim_base + dim_offset;
    if (valid_token) {
        // H14c-A: the dim post-mask 4-iter compare+cmov has been
        // dropped because base_dim+3 is provably < HEAD_DIM_V.
        // Derivation:
        //   dim_base is multiple of 16 and < HEAD_DIM_V=512 -> <= 496
        //   dim_offset in {0,4,8,12}
        //   base_dim = dim_base + dim_offset <= 508
        //   base_dim + 3 <= 511 < 512 = HEAD_DIM_V
        const half *row_ptr = kv_half + local_token_idx * HEAD_DIM_QK + base_dim;
        const uint2 packed = *reinterpret_cast<const uint2*>(row_ptr);
        *reinterpret_cast<uint2*>(&frag_v[0]) = packed;
    } else {
        frag_v[0] = kZeroH;
        frag_v[1] = kZeroH;
        frag_v[2] = kZeroH;
        frag_v[3] = kZeroH;
    }
}

__device__ __forceinline__ void store_mma884_pv_row0_output(
    const flash_mla::sm70::Array<float, 8> &out_frag,
    cutlass::bfloat16_t *out,
    int dim_base,
    int lane) {
    const int row_base = (lane >> 4) * 4 + (lane & 8) + (lane & 1);
    const int col_base = (lane & 4) * 2 + (lane & 2);
#pragma unroll
    for (int d1 = 0; d1 < 2; ++d1) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int d0 = 0; d0 < 2; ++d0) {
                const int q_row = row_base + q * 2;
                const int dim = dim_base + col_base + d1 * 4 + d0;
                if (q_row == 0 && dim < HEAD_DIM_V) {
                    out[dim] = cutlass::bfloat16_t(out_frag[d1 * 4 + q * 2 + d0]);
                }
            }
        }
    }
}

__device__ __forceinline__ void add_mma884_pv_row0_output_to_accum(
    const flash_mla::sm70::Array<float, 8> &out_frag,
    float *output_accum,
    int dim_base,
    int lane) {
    const int row_base = (lane >> 4) * 4 + (lane & 8) + (lane & 1);
    const int col_base = (lane & 4) * 2 + (lane & 2);
#pragma unroll
    for (int d1 = 0; d1 < 2; ++d1) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int d0 = 0; d0 < 2; ++d0) {
                const int q_row = row_base + q * 2;
                const int dim = dim_base + col_base + d1 * 4 + d0;
                if (q_row == 0 && dim < HEAD_DIM_V) {
                    output_accum[dim] += out_frag[d1 * 4 + q * 2 + d0];
                }
            }
        }
    }
}

#ifdef FLASH_MLA_METER_QK_SUB
// H-METER-QK-SUB: 4 slots for compute_mma884_qk_group internals.
// Slots: [loadQ, loadK, mma, store]
__device__ unsigned long long g_qk_sub_cycles[4] = {0, 0, 0, 0};
__device__ unsigned long long g_qk_sub_group_count = 0;
#endif

template<int HEAD_DIM_QK>
__device__ __forceinline__ void compute_mma884_qk_group(
    const cutlass::bfloat16_t *q,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    float *scores,
    int tile_start,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
    flash_mla::sm70::Array<float, 8> score_frag{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_frag[i] = 0.0F;
    }

#ifdef FLASH_MLA_METER_QK_SUB
    const bool qk_meter_lane =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0);
    unsigned long long qk_t0 = 0, qk_t1 = 0;
    unsigned long long qk_sub_local[4] = {0, 0, 0, 0};
#endif

#pragma unroll 1
    for (int dim_base = 0; dim_base < HEAD_DIM_QK; dim_base += 4) {
        flash_mla::sm70::Array<half, 4> frag_q{};
        flash_mla::sm70::Array<half, 4> frag_k{};
        flash_mla::sm70::Array<float, 8> next_frag{};
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) qk_t0 = clock64();
#endif
        load_mma884_q_fragment<HEAD_DIM_QK>(q, dim_base, lane, frag_q);
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[0] += (qk_t1 - qk_t0);
            qk_t0 = qk_t1;
        }
#endif
        load_mma884_k_fragment<HEAD_DIM_QK>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_k);
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[1] += (qk_t1 - qk_t0);
            qk_t0 = qk_t1;
        }
#endif
        flash_mla::sm70::mma_m8n8k4_row_col(next_frag, frag_q, frag_k, score_frag);
        score_frag = next_frag;
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[2] += (qk_t1 - qk_t0);
            qk_t0 = qk_t1;
        }
#endif
    }
#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) qk_t0 = clock64();
#endif
    store_mma884_qk_row0_scores(score_frag, scores, token_refs, tile_start, token_base, tile_count, lane, sm_scale);
#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) {
        qk_t1 = clock64();
        qk_sub_local[3] += (qk_t1 - qk_t0);
        for (int s = 0; s < 4; ++s) {
            g_qk_sub_cycles[s] += qk_sub_local[s];
        }
        g_qk_sub_group_count += 1;
    }
#endif
}

// H13c + H13b: half-Q variant with double-buffered Q/K prefetch
// to hide smem scoreboard latency behind MMA.
//
// Pattern:
//   preload q_a, k_a for iter 0
//   for n in 0..N-2:
//     prefetch q_b, k_b for iter n+1
//     mma on (q_a, k_a)
//     swap a <- b
//   mma on last (q_a, k_a)
template<int HEAD_DIM_QK>
__device__ __forceinline__ void compute_mma884_qk_group_half_q(
    const half *q,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    float *scores,
    int tile_start,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
    flash_mla::sm70::Array<float, 8> score_frag{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_frag[i] = 0.0F;
    }

#ifdef FLASH_MLA_METER_QK_SUB
    const bool qk_meter_lane =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0);
    unsigned long long qk_t0 = 0, qk_t1 = 0;
    unsigned long long qk_sub_local[4] = {0, 0, 0, 0};
#endif

    // HEAD_DIM_QK is 512 or 576; always a multiple of 4 so no tail.
    static_assert(HEAD_DIM_QK % 4 == 0, "HEAD_DIM_QK must be a multiple of 4");
    constexpr int N_ITERS = HEAD_DIM_QK / 4;

    // Preload iter 0 into buffer A.
    flash_mla::sm70::Array<half, 4> frag_q_a{};
    flash_mla::sm70::Array<half, 4> frag_k_a{};
#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) qk_t0 = clock64();
#endif
    load_mma884_q_fragment_half<HEAD_DIM_QK>(q, 0, lane, frag_q_a);
#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) {
        qk_t1 = clock64();
        qk_sub_local[0] += (qk_t1 - qk_t0);
        qk_t0 = qk_t1;
    }
#endif
    load_mma884_k_fragment<HEAD_DIM_QK>(kv_tile, token_refs, token_base, tile_count, 0, lane, frag_k_a);
#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) {
        qk_t1 = clock64();
        qk_sub_local[1] += (qk_t1 - qk_t0);
    }
#endif

    // Main loop: prefetch iter n+1 into buffer B while MMA-ing iter n
    // from buffer A, then swap.
#pragma unroll 1
    for (int n = 0; n < N_ITERS - 1; ++n) {
        const int dim_next = (n + 1) * 4;
        flash_mla::sm70::Array<half, 4> frag_q_b{};
        flash_mla::sm70::Array<half, 4> frag_k_b{};
        flash_mla::sm70::Array<float, 8> next_frag;
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) qk_t0 = clock64();
#endif
        load_mma884_q_fragment_half<HEAD_DIM_QK>(q, dim_next, lane, frag_q_b);
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[0] += (qk_t1 - qk_t0);
            qk_t0 = qk_t1;
        }
#endif
        load_mma884_k_fragment<HEAD_DIM_QK>(kv_tile, token_refs, token_base, tile_count, dim_next, lane, frag_k_b);
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[1] += (qk_t1 - qk_t0);
            qk_t0 = qk_t1;
        }
#endif
        flash_mla::sm70::mma_m8n8k4_row_col(next_frag, frag_q_a, frag_k_a, score_frag);
        score_frag = next_frag;
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[2] += (qk_t1 - qk_t0);
        }
#endif
        // Swap buffers (all register moves, ~free).
        frag_q_a = frag_q_b;
        frag_k_a = frag_k_b;
    }

    // Drain: last MMA uses the final preloaded fragment.
    {
        flash_mla::sm70::Array<float, 8> next_frag;
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) qk_t0 = clock64();
#endif
        flash_mla::sm70::mma_m8n8k4_row_col(next_frag, frag_q_a, frag_k_a, score_frag);
        score_frag = next_frag;
#ifdef FLASH_MLA_METER_QK_SUB
        if (qk_meter_lane) {
            qk_t1 = clock64();
            qk_sub_local[2] += (qk_t1 - qk_t0);
        }
#endif
    }

#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) qk_t0 = clock64();
#endif
    store_mma884_qk_row0_scores(score_frag, scores, token_refs, tile_start, token_base, tile_count, lane, sm_scale);
#ifdef FLASH_MLA_METER_QK_SUB
    if (qk_meter_lane) {
        qk_t1 = clock64();
        qk_sub_local[3] += (qk_t1 - qk_t0);
        for (int s = 0; s < 4; ++s) {
            g_qk_sub_cycles[s] += qk_sub_local[s];
        }
        g_qk_sub_group_count += 1;
    }
#endif
}

template<int HEAD_DIM_QK>
__device__ void stage_kv_tile_to_shared(
    const SparseAttnFwdParams &params,
    const int *indices,
    int tile_start,
    int tile_count,
    int valid_topk,
    int kv_head_idx,
    cutlass::half_t *kv_tile,
    int *token_refs) {
    // H14e: vectorized HBM->smem KV staging. Each thread processes
    // 4 consecutive halves per iter (uint2 load + uint2 store),
    // reducing the loop trip count from (tile_count*HEAD_DIM_QK)/256
    // to (tile_count*HEAD_DIM_QK)/1024. Per-token pointer and
    // validity are hoisted out of the dim loop.
    //
    // Helper: convert uint16 bf16 bits to IEEE half. Volta has no
    // native bf16->half path, so go via fp32 (same pattern as the
    // H13a Q-fragment loader).
    auto bf16_u16_to_half = [] __device__ (uint16_t w) -> half {
        uint32_t bits = static_cast<uint32_t>(w) << 16;
        float f;
        __builtin_memcpy(&f, &bits, sizeof(f));
        return __float2half_rn(f);
    };

    // Token_refs: one thread per local_token_idx writes the ref.
    // Thread (tid < tile_count) handles token refs; all threads
    // participate in the copy.
    if ((int)threadIdx.x < tile_count) {
        const int local_token_idx = threadIdx.x;
        const int topk_idx = tile_start + local_token_idx;
        int token_idx = -1;
        bool valid_token = false;
        if (topk_idx < valid_topk && topk_idx < params.topk) {
            token_idx = indices[topk_idx];
            valid_token = token_idx >= 0 && token_idx < params.s_kv;
        }
        token_refs[local_token_idx] = valid_token ? token_idx + 1 : 0;
    }
    __syncthreads();

    // Vectorized copy: each thread handles 4 halves per iter.
    // linear_idx_4 counts in units of 4 halves (8 bytes).
    const int total_halves = tile_count * HEAD_DIM_QK;
    const int start_idx = threadIdx.x * 4;
    const int step = blockDim.x * 4;
    const half kZeroH = __float2half_rn(0.0F);
    #pragma unroll 1
    for (int linear_idx = start_idx; linear_idx < total_halves; linear_idx += step) {
        const int local_token_idx = linear_idx / HEAD_DIM_QK;
        const int dim = linear_idx - local_token_idx * HEAD_DIM_QK;
        // Re-read token_refs to get validity + token_idx. Cheap:
        // smem broadcast, amortized across 4 halves.
        const int ref = token_refs[local_token_idx];
        const bool valid_token = ref != 0;
        const int token_idx = ref - 1;
        if (valid_token) {
            const cutlass::bfloat16_t *kv_row =
                kv_ptr_for(params, token_idx, kv_head_idx);
            // HBM uint2 read: 4 bf16 = 8 bytes. kv_row is
            // bf16-aligned; dim is a multiple of 4 (start_idx and
            // step are multiples of 4), so &kv_row[dim] is 8-byte
            // aligned when the underlying tensor stride is also a
            // multiple of 4 (always true in this codebase).
            const uint2 packed = *reinterpret_cast<const uint2*>(&kv_row[dim]);
            const uint16_t bf0 = static_cast<uint16_t>(packed.x & 0xFFFFu);
            const uint16_t bf1 = static_cast<uint16_t>(packed.x >> 16);
            const uint16_t bf2 = static_cast<uint16_t>(packed.y & 0xFFFFu);
            const uint16_t bf3 = static_cast<uint16_t>(packed.y >> 16);
            half out4[4];
            out4[0] = bf16_u16_to_half(bf0);
            out4[1] = bf16_u16_to_half(bf1);
            out4[2] = bf16_u16_to_half(bf2);
            out4[3] = bf16_u16_to_half(bf3);
            *reinterpret_cast<uint2*>(&kv_tile[linear_idx]) =
                *reinterpret_cast<const uint2*>(out4);
        } else {
            // Write 4 half zeros via a single uint2 (64-bit zero).
            const uint2 z{0u, 0u};
            *reinterpret_cast<uint2*>(&kv_tile[linear_idx]) = z;
            (void)kZeroH;
        }
    }
    __syncthreads();
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void fill_scores_warp_parallel(
    const SparseAttnFwdParams &params,
    const cutlass::bfloat16_t *q,
    const int *indices,
    int kv_head_idx,
    int valid_topk,
    float *scores,
    cutlass::half_t *kv_tile,
    int *token_refs) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
#pragma unroll 1
    for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
        const int tile_count = min(K_TILE, params.topk - tile_start);
        stage_kv_tile_to_shared<HEAD_DIM_QK>(
            params,
            indices,
            tile_start,
            tile_count,
            valid_topk,
            kv_head_idx,
            kv_tile,
            token_refs);
        for (int tile_base = 0; tile_base < tile_count; tile_base += warps_per_cta) {
            const int tile_idx = tile_base + warp_idx;
            if (tile_idx < tile_count) {
                const int topk_idx = tile_start + tile_idx;
                float score = -CUDART_INF_F;
                if (token_refs[tile_idx] != 0) {
                    score = qk_score_token_from_shared_warp<HEAD_DIM_QK>(
                        q,
                        kv_tile,
                        tile_idx,
                        lane,
                        params.sm_scale);
                }
                if (lane == 0) {
                    scores[topk_idx] = score;
                }
            }
        }
        __syncthreads();
    }
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void compute_mma884_pv_group(
    const float *scores,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int tile_start,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    float row_lse,
    float sink_scale,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
    flash_mla::sm70::Array<float, 8> next_frag{};
    load_mma884_p_fragment(scores, tile_start, token_base, tile_count, lane, row_lse, sink_scale, frag_p);
    load_mma884_v_fragment<HEAD_DIM_QK>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
    flash_mla::sm70::mma_m8n8k4_row_row(next_frag, frag_p, frag_v, out_frag);
    out_frag = next_frag;
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void compute_mma884_online_pv_group(
    const float *tile_scores,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    float row_max,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
    flash_mla::sm70::Array<float, 8> next_frag{};
    load_mma884_online_p_fragment(tile_scores, token_base, tile_count, lane, row_max, frag_p);
    load_mma884_v_fragment<HEAD_DIM_QK>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
    flash_mla::sm70::mma_m8n8k4_row_row(next_frag, frag_p, frag_v, out_frag);
    out_frag = next_frag;
}

// H11: same as compute_mma884_online_pv_group but reads pre-computed
// probabilities from p_cache smem. Avoids per-group expf.
#ifdef FLASH_MLA_METER_S4A_SUB
// H-METER-S4A-SUB: 5-slot clock64 counters for s4a internals.
// [0] loadP  load_mma884_online_p_fragment_cached
// [1] loadV  load_mma884_v_fragment  (token_refs gather)
// [2] mma    mma_m8n8k4_row_row + out_frag assign
// [3] accum  add_mma884_pv_row0_output_to_accum (per dim_group)
// [4] zeros  out_frag init (per dim_group)
__device__ unsigned long long g_s4a_sub_cycles[5] = {0, 0, 0, 0, 0};
// Number of pv_group calls sampled (inner iter count).
__device__ unsigned long long g_s4a_sub_group_count = 0;
// Number of dim_group outer iters sampled (accum calls).
__device__ unsigned long long g_s4a_sub_dim_group_count = 0;
#endif

template<int HEAD_DIM_QK>
__device__ __forceinline__ void compute_mma884_online_pv_group_cached(
    const half *p_cache,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
    flash_mla::sm70::Array<float, 8> next_frag{};
#ifdef FLASH_MLA_METER_S4A_SUB
    const bool s4a_meter_lane =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0);
    unsigned long long s4a_t0 = 0, s4a_t1 = 0;
    if (s4a_meter_lane) s4a_t0 = clock64();
#endif
    load_mma884_online_p_fragment_cached(p_cache, token_base, tile_count, lane, frag_p);
#ifdef FLASH_MLA_METER_S4A_SUB
    if (s4a_meter_lane) {
        s4a_t1 = clock64();
        atomicAdd(&g_s4a_sub_cycles[0], s4a_t1 - s4a_t0);
        s4a_t0 = s4a_t1;
    }
#endif
    load_mma884_v_fragment<HEAD_DIM_QK>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
#ifdef FLASH_MLA_METER_S4A_SUB
    if (s4a_meter_lane) {
        s4a_t1 = clock64();
        atomicAdd(&g_s4a_sub_cycles[1], s4a_t1 - s4a_t0);
        s4a_t0 = s4a_t1;
    }
#endif
    flash_mla::sm70::mma_m8n8k4_row_row(next_frag, frag_p, frag_v, out_frag);
    out_frag = next_frag;
#ifdef FLASH_MLA_METER_S4A_SUB
    if (s4a_meter_lane) {
        s4a_t1 = clock64();
        atomicAdd(&g_s4a_sub_cycles[2], s4a_t1 - s4a_t0);
        atomicAdd(&g_s4a_sub_group_count, 1ULL);
    }
#endif
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void fill_scores_mma884_qk(
    const SparseAttnFwdParams &params,
    const cutlass::bfloat16_t *q,
    const int *indices,
    int kv_head_idx,
    int valid_topk,
    float *scores,
    cutlass::half_t *kv_tile,
    int *token_refs) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
#pragma unroll 1
    for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
        const int tile_count = min(K_TILE, params.topk - tile_start);
        stage_kv_tile_to_shared<HEAD_DIM_QK>(
            params,
            indices,
            tile_start,
            tile_count,
            valid_topk,
            kv_head_idx,
            kv_tile,
            token_refs);
        for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
            compute_mma884_qk_group<HEAD_DIM_QK>(
                q,
                kv_tile,
                token_refs,
                scores,
                tile_start,
                token_base,
                tile_count,
                lane,
                params.sm_scale);
        }
        __syncthreads();
    }
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void fill_tile_scores_mma884_qk(
    const SparseAttnFwdParams &params,
    const cutlass::bfloat16_t *q,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int tile_count,
    float *tile_scores) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
        compute_mma884_qk_group<HEAD_DIM_QK>(
            q,
            kv_tile,
            token_refs,
            tile_scores,
            0,
            token_base,
            tile_count,
            lane,
            params.sm_scale);
    }
}

__device__ __forceinline__ void compute_online_softmax_tile_serial(
    const float *tile_scores,
    int tile_count,
    float *row_max_smem,
    float *row_sum_smem,
    float *online_scale_smem) {
    float tile_max = -CUDART_INF_F;
#pragma unroll 1
    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        tile_max = fmaxf(tile_max, tile_scores[tile_idx]);
    }

    const float old_max = *row_max_smem;
    const float old_sum = *row_sum_smem;
    if (tile_max == -CUDART_INF_F) {
        *online_scale_smem = 1.0F;
        return;
    }

    const float new_max = old_sum > 0.0F ? fmaxf(old_max, tile_max) : tile_max;
    const float old_scale = old_sum > 0.0F ? expf(old_max - new_max) : 0.0F;
    float tile_sum = 0.0F;
#pragma unroll 1
    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const float score = tile_scores[tile_idx];
        if (score != -CUDART_INF_F) {
            tile_sum += expf(score - new_max);
        }
    }

    *row_max_smem = new_max;
    *row_sum_smem = old_sum * old_scale + tile_sum;
    *online_scale_smem = old_scale;
}

// H2: warp-parallel online softmax. Caller now invokes this with
// ALL 32 lanes of the head's lead warp; inside, lane 0 issues the
// final smem writes. Reduction uses __shfl_xor_sync butterfly.
// K_TILE must be <= 32 (current deployed config is K_TILE=32).
__device__ __forceinline__ void compute_online_softmax_tile(
    const float *tile_scores,
    int tile_count,
    float *row_max_smem,
    float *row_sum_smem,
    float *online_scale_smem) {
    static_assert(K_TILE <= 32,
        "H2 warp-parallel softmax requires K_TILE <= 32; "
        "increase K_TILE support in a follow-up round if needed.");
    const int lane = threadIdx.x & 31;

    const float score = (lane < tile_count) ? tile_scores[lane] : -CUDART_INF_F;
    float tile_max = score;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFu, tile_max, offset));
    }

    const float old_max = *row_max_smem;
    const float old_sum = *row_sum_smem;

    if (tile_max == -CUDART_INF_F) {
        if (lane == 0) {
            *online_scale_smem = 1.0F;
        }
        return;
    }

    const float new_max = old_sum > 0.0F ? fmaxf(old_max, tile_max) : tile_max;
    const float old_scale = old_sum > 0.0F ? expf(old_max - new_max) : 0.0F;

    const bool valid = (lane < tile_count) && (score != -CUDART_INF_F);
    float my_term = valid ? expf(score - new_max) : 0.0F;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_term += __shfl_xor_sync(0xFFFFFFFFu, my_term, offset);
    }

    if (lane == 0) {
        *row_max_smem = new_max;
        *row_sum_smem = old_sum * old_scale + my_term;
        *online_scale_smem = old_scale;
    }
}

__device__ __forceinline__ void zero_online_output_accumulator(float *output_accum) {
    for (int dim = threadIdx.x; dim < HEAD_DIM_V; dim += blockDim.x) {
        output_accum[dim] = 0.0F;
    }
}

__device__ __forceinline__ void scale_online_output_accumulator(float *output_accum, float scale) {
    for (int dim = threadIdx.x; dim < HEAD_DIM_V; dim += blockDim.x) {
        output_accum[dim] *= scale;
    }
}

__device__ __forceinline__ void store_online_output(
    float *output_accum,
    cutlass::bfloat16_t *out,
    float row_sum,
    float sink_scale) {
    const bool has_tokens = row_sum > 0.0F;
    for (int dim = threadIdx.x; dim < HEAD_DIM_V; dim += blockDim.x) {
        const float value = has_tokens ? output_accum[dim] / row_sum * sink_scale : 0.0F;
        out[dim] = cutlass::bfloat16_t(value);
    }
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void accumulate_output_mma884_pv(
    const SparseAttnFwdParams &params,
    const float *scores,
    const int *indices,
    int kv_head_idx,
    int valid_topk,
    float row_lse,
    float sink_scale,
    cutlass::bfloat16_t *out,
    cutlass::half_t *kv_tile,
    int *token_refs) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    const bool has_tokens = row_lse != CUDART_INF_F;
    if (!has_tokens) {
        for (int dim = threadIdx.x; dim < HEAD_DIM_V; dim += blockDim.x) {
            out[dim] = cutlass::bfloat16_t(0.0F);
        }
        return;
    }

#pragma unroll 1
    for (int dim_group_base = 0; dim_group_base < HEAD_DIM_V; dim_group_base += warps_per_cta * 16) {
        const int dim_base = dim_group_base + warp_idx * 16;
        const bool dim_valid = dim_base < HEAD_DIM_V;
        flash_mla::sm70::Array<float, 8> out_frag{};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_frag[i] = 0.0F;
        }

#pragma unroll 1
        for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
            const int tile_count = min(K_TILE, params.topk - tile_start);
            stage_kv_tile_to_shared<HEAD_DIM_QK>(
                params,
                indices,
                tile_start,
                tile_count,
                valid_topk,
                kv_head_idx,
                kv_tile,
                token_refs);
            if (dim_valid) {
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_pv_group<HEAD_DIM_QK>(
                        scores,
                        kv_tile,
                        token_refs,
                        tile_start,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_lse,
                        sink_scale,
                        out_frag);
                }
            }
            __syncthreads();
        }

        if (dim_valid) {
            store_mma884_pv_row0_output(out_frag, out, dim_base, lane);
        }
    }
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void accumulate_output_mma884_online(
    const SparseAttnFwdParams &params,
    const cutlass::bfloat16_t *q,
    const int *indices,
    int q_idx,
    int head_idx,
    int kv_head_idx,
    int valid_topk,
    cutlass::bfloat16_t *out,
    float *tile_scores,
    float *online_scalars,
    float *output_accum,
    cutlass::half_t *kv_tile,
    int *token_refs) {
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[SINK_SCALE_SLOT] = 1.0F;
    }
    zero_online_output_accumulator(output_accum);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
#pragma unroll 1
    for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
        const int tile_count = min(K_TILE, params.topk - tile_start);
        stage_kv_tile_to_shared<HEAD_DIM_QK>(
            params,
            indices,
            tile_start,
            tile_count,
            valid_topk,
            kv_head_idx,
            kv_tile,
            token_refs);
        fill_tile_scores_mma884_qk<HEAD_DIM_QK>(params, q, kv_tile, token_refs, tile_count, tile_scores);
        __syncthreads();

        // H2 — warp-parallel softmax: all 32 lanes of warp 0 enter.
        if (threadIdx.x < 32) {
            compute_online_softmax_tile(
                tile_scores,
                tile_count,
                online_scalars + ROW_MAX_SLOT,
                online_scalars + ROW_SUM_SLOT,
                online_scalars + ONLINE_SCALE_SLOT);
        }
        __syncthreads();

        scale_online_output_accumulator(output_accum, online_scalars[ONLINE_SCALE_SLOT]);
        __syncthreads();

        const float row_max = online_scalars[ROW_MAX_SLOT];
#pragma unroll 1
        for (int dim_group_base = 0; dim_group_base < HEAD_DIM_V; dim_group_base += warps_per_cta * 16) {
            const int dim_base = dim_group_base + warp_idx * 16;
            if (dim_base < HEAD_DIM_V) {
                flash_mla::sm70::Array<float, 8> out_frag{};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out_frag[i] = 0.0F;
                }
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_online_pv_group<HEAD_DIM_QK>(
                        tile_scores,
                        kv_tile,
                        token_refs,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_max,
                        out_frag);
                }
                add_mma884_pv_row0_output_to_accum(out_frag, output_accum, dim_base, lane);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float row_max = online_scalars[ROW_MAX_SLOT];
        const float row_sum = online_scalars[ROW_SUM_SLOT];
        const bool has_tokens = row_sum > 0.0F;
        const float row_lse = has_tokens ? logf(row_sum) + row_max : CUDART_INF_F;
        float sink_scale = 1.0F;
        if (params.attn_sink != nullptr && has_tokens) {
            const float sink = params.attn_sink[head_idx];
            sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
        }
        *max_logits_ptr_for(params, q_idx, head_idx) = row_max;
        *lse_ptr_for(params, q_idx, head_idx) = row_lse;
        online_scalars[SINK_SCALE_SLOT] = sink_scale;
    }
    __syncthreads();

    store_online_output(output_accum, out, online_scalars[ROW_SUM_SLOT], online_scalars[SINK_SCALE_SLOT]);
}

template<int HEAD_DIM_QK>
__device__ __forceinline__ void accumulate_output_mma884_online_double_buffer(
    const SparseAttnFwdParams &params,
    const cutlass::bfloat16_t *q,
    const int *indices,
    int q_idx,
    int head_idx,
    int kv_head_idx,
    int valid_topk,
    cutlass::bfloat16_t *out,
    float *tile_scores,
    float *online_scalars,
    float *output_accum,
    cutlass::half_t *kv_tile_0,
    cutlass::half_t *kv_tile_1,
    int *token_refs_0,
    int *token_refs_1) {
    // Double-buffered variant of accumulate_output_mma884_online.
    //
    // Ping-pongs between two KV tile + token_refs shmem buffers so that the
    // next tile's shmem staging (global loads + index resolution, which
    // dominate prefill wall time on V100 — the kernel achieves <1 MB/s HBM
    // without this optimization) can overlap with the current tile's MMA
    // compute. The overall sparse FP8 prefill kernel is heavily memory-bound
    // on V100 (3 MB L2, 900 GB/s peak HBM, but measured 8e-5 GB/s), so any
    // load/compute overlap yields a visible speedup.
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[SINK_SCALE_SLOT] = 1.0F;
    }
    zero_online_output_accumulator(output_accum);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;

    cutlass::half_t *kv_bufs[2] = {kv_tile_0, kv_tile_1};
    int *ref_bufs[2] = {token_refs_0, token_refs_1};

    const int total_topk = params.topk;
    const int num_tiles = (total_topk + K_TILE - 1) / K_TILE;

    // Pre-stage tile 0 into buffer 0 before the main loop so the first
    // iteration can immediately compute while tile 1 is being staged.
    if (num_tiles > 0) {
        const int tile_count_0 = min(K_TILE, total_topk);
        stage_kv_tile_to_shared<HEAD_DIM_QK>(
            params,
            indices,
            /*tile_start=*/0,
            tile_count_0,
            valid_topk,
            kv_head_idx,
            kv_bufs[0],
            ref_bufs[0]);
    }

    int cur_buf = 0;

#pragma unroll 1
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int tile_start = tile_idx * K_TILE;
        const int tile_count = min(K_TILE, total_topk - tile_start);

        cutlass::half_t *cur_kv = kv_bufs[cur_buf];
        int *cur_refs = ref_bufs[cur_buf];

        fill_tile_scores_mma884_qk<HEAD_DIM_QK>(
            params, q, cur_kv, cur_refs, tile_count, tile_scores);
        __syncthreads();

        // H2 — warp-parallel softmax: all 32 lanes of warp 0 enter.
        if (threadIdx.x < 32) {
            compute_online_softmax_tile(
                tile_scores,
                tile_count,
                online_scalars + ROW_MAX_SLOT,
                online_scalars + ROW_SUM_SLOT,
                online_scalars + ONLINE_SCALE_SLOT);
        }
        __syncthreads();

        scale_online_output_accumulator(
            output_accum, online_scalars[ONLINE_SCALE_SLOT]);
        __syncthreads();

        const float row_max = online_scalars[ROW_MAX_SLOT];
#pragma unroll 1
        for (int dim_group_base = 0;
             dim_group_base < HEAD_DIM_V;
             dim_group_base += warps_per_cta * 16) {
            const int dim_base = dim_group_base + warp_idx * 16;
            if (dim_base < HEAD_DIM_V) {
                flash_mla::sm70::Array<float, 8> out_frag{};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out_frag[i] = 0.0F;
                }
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_online_pv_group<HEAD_DIM_QK>(
                        tile_scores,
                        cur_kv,
                        cur_refs,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_max,
                        out_frag);
                }
                add_mma884_pv_row0_output_to_accum(
                    out_frag, output_accum, dim_base, lane);
            }
        }
        __syncthreads();

        // Pre-stage the next tile into the alternate buffer. The
        // __syncthreads() above guarantees all current-tile MMA loads from
        // cur_kv are complete, so overwriting the alternate (unused) buffer
        // is race-free.
        const int next_tile_idx = tile_idx + 1;
        if (next_tile_idx < num_tiles) {
            const int next_start = next_tile_idx * K_TILE;
            const int next_count = min(K_TILE, total_topk - next_start);
            const int next_buf = 1 - cur_buf;
            stage_kv_tile_to_shared<HEAD_DIM_QK>(
                params,
                indices,
                next_start,
                next_count,
                valid_topk,
                kv_head_idx,
                kv_bufs[next_buf],
                ref_bufs[next_buf]);
            cur_buf = next_buf;
        }
    }

    if (threadIdx.x == 0) {
        const float row_max = online_scalars[ROW_MAX_SLOT];
        const float row_sum = online_scalars[ROW_SUM_SLOT];
        const bool has_tokens = row_sum > 0.0F;
        const float row_lse = has_tokens ? logf(row_sum) + row_max : CUDART_INF_F;
        float sink_scale = 1.0F;
        if (params.attn_sink != nullptr && has_tokens) {
            const float sink = params.attn_sink[head_idx];
            sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
        }
        *max_logits_ptr_for(params, q_idx, head_idx) = row_max;
        *lse_ptr_for(params, q_idx, head_idx) = row_lse;
        online_scalars[SINK_SCALE_SLOT] = sink_scale;
    }
    __syncthreads();

    store_online_output(
        output_accum,
        out,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[SINK_SCALE_SLOT]);
}

// ============================================================
// Round 3 of deepseek-v4-flash-prefill-throughput spec:
// HEADS_PER_BLOCK kernel — batch HPB heads of the SAME query into
// one CUDA block so the KV tile is staged once and reused across
// heads. Valid only when HEADS_PER_BLOCK <= q_heads_per_kv
// (= params.h_q / params.h_kv); all heads in a block must share
// the same kv_head_idx and therefore the same top-K KV page set.
// ============================================================

template<int HEAD_DIM_QK, int HPB>
__device__ __forceinline__ void accumulate_output_mma884_online_hpb(
    const SparseAttnFwdParams &params,
    int q_idx,
    int head_base,          // blockIdx.y * HPB
    int kv_head_idx,
    int valid_topk,
    float *tile_scores_all,     // [HPB, K_TILE]
    float *online_scalars_all,  // [HPB, 4]
    float *output_accum_all,    // [HPB, HEAD_DIM_V]
    cutlass::half_t *kv_tile,   // [K_TILE, HEAD_DIM_QK] shared across heads
    int *token_refs) {          // [K_TILE] shared across heads
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;

    // Per-head Q pointers: all heads in this block share the same q_idx
    // and kv_head_idx but differ in head_idx.
    const cutlass::bfloat16_t *q_ptrs[HPB];
    const int *indices = indices_ptr_for(params, q_idx, kv_head_idx);
    #pragma unroll
    for (int h = 0; h < HPB; ++h) {
        q_ptrs[h] = q_ptr_for(params, q_idx, head_base + h);
    }

    // Initialize per-head online scalars and zero per-head output accumulators.
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int h = 0; h < HPB; ++h) {
            float *os = online_scalars_all + h * 4;
            os[ROW_MAX_SLOT] = -CUDART_INF_F;
            os[ROW_SUM_SLOT] = 0.0F;
            os[ONLINE_SCALE_SLOT] = 1.0F;
            os[SINK_SCALE_SLOT] = 1.0F;
        }
    }
    #pragma unroll
    for (int h = 0; h < HPB; ++h) {
        zero_online_output_accumulator(output_accum_all + h * HEAD_DIM_V);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;

    #pragma unroll 1
    for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
        const int tile_count = min(K_TILE, params.topk - tile_start);

        // Stage KV tile ONCE per block — shared across all HPB heads.
        stage_kv_tile_to_shared<HEAD_DIM_QK>(
            params,
            indices,
            tile_start,
            tile_count,
            valid_topk,
            kv_head_idx,
            kv_tile,
            token_refs);

        #pragma unroll
        for (int h = 0; h < HPB; ++h) {
            float *tile_scores = tile_scores_all + h * K_TILE;
            float *online_scalars = online_scalars_all + h * 4;
            float *output_accum = output_accum_all + h * HEAD_DIM_V;

            fill_tile_scores_mma884_qk<HEAD_DIM_QK>(
                params, q_ptrs[h], kv_tile, token_refs, tile_count, tile_scores);
            __syncthreads();

            // H2 — warp-parallel softmax: all 32 lanes of warp 0 enter.
            if (threadIdx.x < 32) {
                compute_online_softmax_tile(
                    tile_scores,
                    tile_count,
                    online_scalars + ROW_MAX_SLOT,
                    online_scalars + ROW_SUM_SLOT,
                    online_scalars + ONLINE_SCALE_SLOT);
            }
            __syncthreads();

            scale_online_output_accumulator(output_accum, online_scalars[ONLINE_SCALE_SLOT]);
            __syncthreads();

            const float row_max = online_scalars[ROW_MAX_SLOT];
            #pragma unroll 1
            for (int dim_group_base = 0; dim_group_base < HEAD_DIM_V;
                 dim_group_base += warps_per_cta * 16) {
                const int dim_base = dim_group_base + warp_idx * 16;
                if (dim_base < HEAD_DIM_V) {
                    flash_mla::sm70::Array<float, 8> out_frag{};
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        out_frag[i] = 0.0F;
                    }
                    #pragma unroll 1
                    for (int token_base = 0; token_base < tile_count; token_base += 4) {
                        compute_mma884_online_pv_group<HEAD_DIM_QK>(
                            tile_scores,
                            kv_tile,
                            token_refs,
                            token_base,
                            tile_count,
                            dim_base,
                            lane,
                            row_max,
                            out_frag);
                    }
                    add_mma884_pv_row0_output_to_accum(out_frag, output_accum, dim_base, lane);
                }
            }
            __syncthreads();
        }
    }

    // Epilogue: per-head LSE + sink_scale + output store.
    #pragma unroll
    for (int h = 0; h < HPB; ++h) {
        const int head_idx = head_base + h;
        float *online_scalars = online_scalars_all + h * 4;
        float *output_accum = output_accum_all + h * HEAD_DIM_V;
        cutlass::bfloat16_t *out = out_ptr_for(params, q_idx, head_idx);

        if (threadIdx.x == 0) {
            const float row_max = online_scalars[ROW_MAX_SLOT];
            const float row_sum = online_scalars[ROW_SUM_SLOT];
            const bool has_tokens = row_sum > 0.0F;
            const float row_lse = has_tokens ? logf(row_sum) + row_max : CUDART_INF_F;
            float sink_scale = 1.0F;
            if (params.attn_sink != nullptr && has_tokens) {
                const float sink = params.attn_sink[head_idx];
                sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
            }
            *max_logits_ptr_for(params, q_idx, head_idx) = row_max;
            *lse_ptr_for(params, q_idx, head_idx) = row_lse;
            online_scalars[SINK_SCALE_SLOT] = sink_scale;
        }
        __syncthreads();

        store_online_output(
            output_accum,
            out,
            online_scalars[ROW_SUM_SLOT],
            online_scalars[SINK_SCALE_SLOT]);
    }
}

template<int HEAD_DIM_QK, int HPB>
__host__ __device__ __forceinline__ size_t sm70_sparse_prefill_shared_bytes_hpb() {
    // Layout per block (H10: q_smem, H11: p_cache_all):
    //   tile_scores_all:   HPB * K_TILE * sizeof(float)
    //   online_scalars_all: HPB * 4 * sizeof(float)
    //   output_accum_all:  HPB * HEAD_DIM_V * sizeof(float)
    //   token_refs:        K_TILE * sizeof(int)
    //   kv_tile:           K_TILE * HEAD_DIM_QK * sizeof(cutlass::half_t)
    //   q_smem (H10+H13c): HPB * HEAD_DIM_QK * sizeof(half)
    //   p_cache_all (H11): HPB * K_TILE * sizeof(half)
    return static_cast<size_t>(HPB) * static_cast<size_t>(K_TILE) * sizeof(float)
         + static_cast<size_t>(HPB) * 4 * sizeof(float)
         + static_cast<size_t>(HPB) * HEAD_DIM_V * sizeof(float)
         + static_cast<size_t>(K_TILE) * sizeof(int)
         + static_cast<size_t>(K_TILE) * static_cast<size_t>(HEAD_DIM_QK) * sizeof(cutlass::half_t)
         + static_cast<size_t>(HPB) * static_cast<size_t>(HEAD_DIM_QK) * sizeof(half)
         + static_cast<size_t>(HPB) * static_cast<size_t>(K_TILE) * sizeof(half);
}

template<int HEAD_DIM_QK, int HPB>
__global__ void __launch_bounds__(NUM_THREADS)
sm70_sparse_prefill_hpb_fwd_kernel(const SparseAttnFwdParams params) {
    static_assert(HEAD_DIM_QK == 512 || HEAD_DIM_QK == 576);
    static_assert(HPB >= 1);

    const int q_idx = blockIdx.x;
    const int head_base = blockIdx.y * HPB;
    if (head_base >= params.h_q) return;
    const int q_heads_per_kv = params.h_q / params.h_kv;
    // All HPB heads in this block must share the same kv_head_idx.
    // The launcher guarantees (q_heads_per_kv % HPB == 0) so a simple
    // integer division is correct across the block.
    const int kv_head_idx = head_base / q_heads_per_kv;

    extern __shared__ float shared[];
    const int valid_topk = valid_topk_for_query(params, q_idx);

    float *tile_scores_all = shared;
    float *online_scalars_all = tile_scores_all + HPB * K_TILE;
    float *output_accum_all = online_scalars_all + HPB * 4;
    int *token_refs = reinterpret_cast<int*>(output_accum_all + HPB * HEAD_DIM_V);
    cutlass::half_t *kv_tile = reinterpret_cast<cutlass::half_t*>(token_refs + K_TILE);

    accumulate_output_mma884_online_hpb<HEAD_DIM_QK, HPB>(
        params,
        q_idx,
        head_base,
        kv_head_idx,
        valid_topk,
        tile_scores_all,
        online_scalars_all,
        output_accum_all,
        kv_tile,
        token_refs);
}



// ============================================================
// Round 4 of deepseek-v4-flash-prefill-throughput spec:
// Warp-specialized HEADS_PER_BLOCK kernel. Each head gets a
// disjoint subset of warps (warps_per_head = warps_per_cta / HPB).
// Heads run in PARALLEL within a block while sharing the KV tile.
// ============================================================

template<int HEAD_DIM_QK>
__device__ __forceinline__ void fill_tile_scores_mma884_qk_subset(
    const SparseAttnFwdParams &params,
    const cutlass::bfloat16_t *q,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int tile_count,
    float *tile_scores,
    int warp_idx_in_head,
    int warps_per_head) {
    const int lane = threadIdx.x & 31;
    for (int token_base = warp_idx_in_head * 16; token_base < tile_count;
         token_base += warps_per_head * 16) {
        compute_mma884_qk_group<HEAD_DIM_QK>(
            q,
            kv_tile,
            token_refs,
            tile_scores,
            0,
            token_base,
            tile_count,
            lane,
            params.sm_scale);
    }
}

// H13c: half-Q variant of fill_tile_scores_mma884_qk_subset.
// Wraps compute_mma884_qk_group_half_q for the WS HPB path.
template<int HEAD_DIM_QK>
__device__ __forceinline__ void fill_tile_scores_mma884_qk_subset_half_q(
    const SparseAttnFwdParams &params,
    const half *q,
    const cutlass::half_t *kv_tile,
    const int *token_refs,
    int tile_count,
    float *tile_scores,
    int warp_idx_in_head,
    int warps_per_head) {
    const int lane = threadIdx.x & 31;
    for (int token_base = warp_idx_in_head * 16; token_base < tile_count;
         token_base += warps_per_head * 16) {
        compute_mma884_qk_group_half_q<HEAD_DIM_QK>(
            q,
            kv_tile,
            token_refs,
            tile_scores,
            0,
            token_base,
            tile_count,
            lane,
            params.sm_scale);
    }
}

__device__ __forceinline__ void zero_online_output_accumulator_subset(
    float *output_accum,
    int thread_idx_in_head,
    int threads_per_head) {
    for (int dim = thread_idx_in_head; dim < HEAD_DIM_V; dim += threads_per_head) {
        output_accum[dim] = 0.0F;
    }
}

__device__ __forceinline__ void scale_online_output_accumulator_subset(
    float *output_accum,
    float scale,
    int thread_idx_in_head,
    int threads_per_head) {
    for (int dim = thread_idx_in_head; dim < HEAD_DIM_V; dim += threads_per_head) {
        output_accum[dim] *= scale;
    }
}

__device__ __forceinline__ void store_online_output_subset(
    float *output_accum,
    cutlass::bfloat16_t *out,
    float row_sum,
    float sink_scale,
    int thread_idx_in_head,
    int threads_per_head) {
    const bool has_tokens = row_sum > 0.0F;
    for (int dim = thread_idx_in_head; dim < HEAD_DIM_V; dim += threads_per_head) {
        const float value = has_tokens ? output_accum[dim] / row_sum * sink_scale : 0.0F;
        out[dim] = cutlass::bfloat16_t(value);
    }
}

#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
// H-METER: 6-stage clock64 counters, written only by block(0,0) thread 0.
// s0 stage_kv, s1 fill_scores, s2 softmax, s3 scale, s4 PV_MMA, s5 epilogue.
__device__ unsigned long long g_stage_cycles[6] = {0, 0, 0, 0, 0, 0};
__device__ unsigned long long g_stage_tile_count = 0;
__device__ unsigned long long g_stage_block_count = 0;
#endif

#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
// H-METER-FINE: 12-stage clock64 counters.
// Slots: [s0, s1a_qk_compute, s1c_sync, s2_softmax, s2s_sync,
//         s3_scale_and_pcache, s3s_sync, s4a_pv_compute, s4s_sync, s5,
//         unused10, unused11]
__device__ unsigned long long g_stage_cycles_fine[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
__device__ unsigned long long g_stage_tile_count_fine = 0;
__device__ unsigned long long g_stage_block_count_fine = 0;
#endif

template<int HEAD_DIM_QK, int HPB>
__device__ __forceinline__ void accumulate_output_mma884_online_hpb_ws(
    const SparseAttnFwdParams &params,
    int q_idx,
    int head_base,
    int kv_head_idx,
    int valid_topk,
    float *tile_scores_all,     // [HPB, K_TILE]
    float *online_scalars_all,  // [HPB, 4]
    float *output_accum_all,    // [HPB, HEAD_DIM_V]
    cutlass::half_t *kv_tile,
    int *token_refs,
    half *q_smem,  // H10+H13c: [HPB, HEAD_DIM_QK] fp16 in smem
    half *p_cache_all) {           // H11: [HPB, K_TILE]
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    constexpr int warps_per_head_c = 0; // placeholder; actual value computed below
    const int warps_per_head = warps_per_cta / HPB;
    const int head_in_block = warp_idx / warps_per_head;
    const int warp_idx_in_head = warp_idx % warps_per_head;
    const int thread_idx_in_head = warp_idx_in_head * 32 + lane;
    const int threads_per_head = warps_per_head * 32;
    const bool head_lead = (warp_idx_in_head == 0 && lane == 0);

    // Per-head local pointers.
    float *tile_scores = tile_scores_all + head_in_block * K_TILE;
    float *online_scalars = online_scalars_all + head_in_block * 4;
    float *output_accum = output_accum_all + head_in_block * HEAD_DIM_V;
    const int head_idx = head_base + head_in_block;
    const cutlass::bfloat16_t *q = q_ptr_for(params, q_idx, head_idx);
    const int *indices = indices_ptr_for(params, q_idx, kv_head_idx);
    cutlass::bfloat16_t *out = out_ptr_for(params, q_idx, head_idx);

    // Initialize per-head online scalars (one thread per head).
    if (head_lead) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[SINK_SCALE_SLOT] = 1.0F;
    }
    // Each head zeroes its own output_accum slab using its subset of threads.
    zero_online_output_accumulator_subset(output_accum, thread_idx_in_head, threads_per_head);

    // H10 + H13c: stage each head's Q vector to smem as half (fp16).
    // Convert bf16 -> fp16 once per element; the QK hot loop then
    // does native fp16 smem loads (no bf16 unpack chain).
    {
        half *q_smem_head = q_smem + head_in_block * HEAD_DIM_QK;
        const cutlass::bfloat16_t *q_head = q;  // per-head bf16 HBM ptr
        for (int dim = thread_idx_in_head; dim < HEAD_DIM_QK; dim += threads_per_head) {
            q_smem_head[dim] = __float2half_rn(bf16_to_float(q_head[dim]));
        }
    }
    __syncthreads();

#if defined(FLASH_MLA_METER_SPARSE_WS_HPB) || defined(FLASH_MLA_METER_SPARSE_WS_HPB_FINE)
    const bool meter_lane =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0);
#endif
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
    unsigned long long meter_t0 = 0, meter_t1 = 0;
    unsigned long long meter_local[6] = {0, 0, 0, 0, 0, 0};
    unsigned long long meter_local_tiles = 0;
    #define METER_TAP(stage_idx) do { \
        if (meter_lane) { \
            meter_t1 = clock64(); \
            meter_local[(stage_idx)] += (meter_t1 - meter_t0); \
            meter_t0 = meter_t1; \
        } \
    } while (0)
    #define METER_START() do { if (meter_lane) meter_t0 = clock64(); } while (0)
#else
    #define METER_TAP(stage_idx) do { } while (0)
    #define METER_START() do { } while (0)
#endif
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
    unsigned long long meter_t0_f = 0, meter_t1_f = 0;
    unsigned long long meter_local_fine[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    unsigned long long meter_local_tiles_fine = 0;
    #define METER_TAP_FINE(stage_idx) do { \
        if (meter_lane) { \
            meter_t1_f = clock64(); \
            meter_local_fine[(stage_idx)] += (meter_t1_f - meter_t0_f); \
            meter_t0_f = meter_t1_f; \
        } \
    } while (0)
    #define METER_START_FINE() do { if (meter_lane) meter_t0_f = clock64(); } while (0)
#else
    #define METER_TAP_FINE(stage_idx) do { } while (0)
    #define METER_START_FINE() do { } while (0)
#endif

    #pragma unroll 1
    for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
        const int tile_count = min(K_TILE, params.topk - tile_start);
        METER_START();
        METER_START_FINE();

        // Stage KV tile ONCE per block using ALL threads (shared across heads).
        // The existing helper uses all of blockDim.x; since kv_tile and
        // token_refs are shared across heads, every head-group participates.
        stage_kv_tile_to_shared<HEAD_DIM_QK>(
            params,
            indices,
            tile_start,
            tile_count,
            valid_topk,
            kv_head_idx,
            kv_tile,
            token_refs);
        // stage_kv_tile_to_shared ends with __syncthreads() — all heads see it.
        METER_TAP(0);
        METER_TAP_FINE(0);  // s0 stage_kv

        // Per-head QK scores using only this head's warps.
        // H10: use per-head smem Q slice instead of HBM q pointer.
        fill_tile_scores_mma884_qk_subset_half_q<HEAD_DIM_QK>(
            params, q_smem + head_in_block * HEAD_DIM_QK,
            kv_tile, token_refs, tile_count, tile_scores,
            warp_idx_in_head, warps_per_head);
        METER_TAP_FINE(1);  // s1a QK compute (NO sync)
        __syncthreads();
        METER_TAP(1);
        METER_TAP_FINE(2);  // s1c sync after QK

        // Per-head online softmax — H2 warp-parallel: all 32 lanes of
        // each head's lead warp run the reduction together, lane 0
        // commits smem. `warp_idx_in_head == 0` picks up each head's
        // lead warp; the HPB heads have disjoint lead warps.
        if (warp_idx_in_head == 0) {
            compute_online_softmax_tile(
                tile_scores,
                tile_count,
                online_scalars + ROW_MAX_SLOT,
                online_scalars + ROW_SUM_SLOT,
                online_scalars + ONLINE_SCALE_SLOT);
        }
        METER_TAP_FINE(3);  // s2 softmax (NO sync)
        __syncthreads();
        METER_TAP(2);
        METER_TAP_FINE(4);  // s2s sync after softmax

        // Per-head output scale using this head's thread subset.
        scale_online_output_accumulator_subset(
            output_accum, online_scalars[ONLINE_SCALE_SLOT],
            thread_idx_in_head, threads_per_head);

        // H11: populate p_cache for this tile ONCE using this head's
        // warp-group (threads_per_head threads), before the
        // dim_base loop. Avoids 16x redundant expf in PV hot loop.
        half *p_cache = p_cache_all + head_in_block * K_TILE;
        {
            const float row_max_pcache = online_scalars[ROW_MAX_SLOT];
            for (int t = thread_idx_in_head; t < K_TILE; t += threads_per_head) {
                const half v = (t < tile_count)
                    ? probability_half_from_online_score(tile_scores[t], row_max_pcache)
                    : __float2half_rn(0.0F);
                p_cache[t] = v;
            }
        }
        METER_TAP_FINE(5);  // s3 scale + p_cache populate (NO sync)
        __syncthreads();
        METER_TAP(3);
        METER_TAP_FINE(6);  // s3s sync after scale+pcache

        // PV MMA over V-dim using this head's warps only.
        #pragma unroll 1
        for (int dim_group_base = 0; dim_group_base < HEAD_DIM_V;
             dim_group_base += warps_per_head * 16) {
            const int dim_base = dim_group_base + warp_idx_in_head * 16;
            if (dim_base < HEAD_DIM_V) {
#ifdef FLASH_MLA_METER_S4A_SUB
                const bool s4a_outer_meter =
                    (blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0);
                unsigned long long s4a_outer_t0 = 0, s4a_outer_t1 = 0;
                if (s4a_outer_meter) s4a_outer_t0 = clock64();
#endif
                flash_mla::sm70::Array<float, 8> out_frag{};
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out_frag[i] = 0.0F;
                }
#ifdef FLASH_MLA_METER_S4A_SUB
                if (s4a_outer_meter) {
                    s4a_outer_t1 = clock64();
                    atomicAdd(&g_s4a_sub_cycles[4], s4a_outer_t1 - s4a_outer_t0);
                }
#endif
                #pragma unroll
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_online_pv_group_cached<HEAD_DIM_QK>(
                        p_cache,
                        kv_tile,
                        token_refs,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        out_frag);
                }
#ifdef FLASH_MLA_METER_S4A_SUB
                if (s4a_outer_meter) s4a_outer_t0 = clock64();
#endif
                add_mma884_pv_row0_output_to_accum(out_frag, output_accum, dim_base, lane);
#ifdef FLASH_MLA_METER_S4A_SUB
                if (s4a_outer_meter) {
                    s4a_outer_t1 = clock64();
                    atomicAdd(&g_s4a_sub_cycles[3], s4a_outer_t1 - s4a_outer_t0);
                    atomicAdd(&g_s4a_sub_dim_group_count, 1ULL);
                }
#endif
            }
        }
        METER_TAP_FINE(7);  // s4a PV compute + add_accum (NO sync)
        __syncthreads();
        METER_TAP(4);
        METER_TAP_FINE(8);  // s4s sync after PV
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
        if (meter_lane) meter_local_tiles += 1;
#endif
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
        if (meter_lane) meter_local_tiles_fine += 1;
#endif
    }

    // Epilogue: per-head LSE + sink_scale + output store (each head's lead thread).
    if (head_lead) {
        const float row_max = online_scalars[ROW_MAX_SLOT];
        const float row_sum = online_scalars[ROW_SUM_SLOT];
        const bool has_tokens = row_sum > 0.0F;
        const float row_lse = has_tokens ? logf(row_sum) + row_max : CUDART_INF_F;
        float sink_scale = 1.0F;
        if (params.attn_sink != nullptr && has_tokens) {
            const float sink = params.attn_sink[head_idx];
            sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
        }
        *max_logits_ptr_for(params, q_idx, head_idx) = row_max;
        *lse_ptr_for(params, q_idx, head_idx) = row_lse;
        online_scalars[SINK_SCALE_SLOT] = sink_scale;
    }
    __syncthreads();

#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
    if (meter_lane) meter_t0 = clock64();
#endif
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
    if (meter_lane) meter_t0_f = clock64();
#endif
    store_online_output_subset(
        output_accum,
        out,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[SINK_SCALE_SLOT],
        thread_idx_in_head,
        threads_per_head);
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
    if (meter_lane) {
        meter_t1 = clock64();
        meter_local[5] += (meter_t1 - meter_t0);
        // Add local totals into the global counters. Atomics not
        // needed because only one lane in block(0,0) writes.
        for (int s = 0; s < 6; ++s) {
            g_stage_cycles[s] += meter_local[s];
        }
        g_stage_tile_count += meter_local_tiles;
        g_stage_block_count += 1;
    }
    #undef METER_TAP
    #undef METER_START
#endif
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
    if (meter_lane) {
        meter_t1_f = clock64();
        meter_local_fine[9] += (meter_t1_f - meter_t0_f);  // s5 epilogue
        for (int s = 0; s < 12; ++s) {
            g_stage_cycles_fine[s] += meter_local_fine[s];
        }
        g_stage_tile_count_fine += meter_local_tiles_fine;
        g_stage_block_count_fine += 1;
    }
    #undef METER_TAP_FINE
    #undef METER_START_FINE
#endif
}

template<int HEAD_DIM_QK, int HPB>
__global__ void __launch_bounds__(NUM_THREADS)
sm70_sparse_prefill_hpb_ws_fwd_kernel(const SparseAttnFwdParams params) {
    static_assert(HEAD_DIM_QK == 512 || HEAD_DIM_QK == 576);
    static_assert(HPB >= 1);
    // Require warps_per_cta divisible by HPB so each head gets >= 1 warp.
    static_assert((NUM_THREADS / 32) % HPB == 0,
                  "warps_per_cta must be divisible by HPB");

    const int q_idx = blockIdx.x;
    const int head_base = blockIdx.y * HPB;
    if (head_base >= params.h_q) return;
    const int q_heads_per_kv = params.h_q / params.h_kv;
    const int kv_head_idx = head_base / q_heads_per_kv;

    extern __shared__ float shared[];
    const int valid_topk = valid_topk_for_query(params, q_idx);

    float *tile_scores_all = shared;
    float *online_scalars_all = tile_scores_all + HPB * K_TILE;
    float *output_accum_all = online_scalars_all + HPB * 4;
    int *token_refs = reinterpret_cast<int*>(output_accum_all + HPB * HEAD_DIM_V);
    cutlass::half_t *kv_tile = reinterpret_cast<cutlass::half_t*>(token_refs + K_TILE);
    // H10 + H13c: Q staged to smem as half (fp16).
    half *q_smem = reinterpret_cast<half*>(
        kv_tile + K_TILE * HEAD_DIM_QK);
    // H11: P probabilities cache — computed once per tile, reused
    // by all dim_base iterations of the PV MMA loop.
    half *p_cache_all = reinterpret_cast<half*>(
        q_smem + HPB * HEAD_DIM_QK);

    accumulate_output_mma884_online_hpb_ws<HEAD_DIM_QK, HPB>(
        params,
        q_idx,
        head_base,
        kv_head_idx,
        valid_topk,
        tile_scores_all,
        online_scalars_all,
        output_accum_all,
        kv_tile,
        token_refs,
        q_smem,
        p_cache_all);
}


template<int HEAD_DIM_QK>
__host__ __device__ __forceinline__ size_t sm70_sparse_prefill_shared_bytes(const SparseAttnFwdParams &params) {
    if constexpr (USE_MMA_884_ONLINE) {
        // Double-buffer: 2x kv_tile + 2x token_refs when enabled
        constexpr size_t kv_buf_count = USE_DOUBLE_BUFFER ? 2 : 1;
        return static_cast<size_t>(K_TILE) * sizeof(float)
             + 4 * sizeof(float)
             + HEAD_DIM_V * sizeof(float)
             + kv_buf_count * static_cast<size_t>(K_TILE) * sizeof(int)
             + kv_buf_count * static_cast<size_t>(K_TILE) * static_cast<size_t>(HEAD_DIM_QK) * sizeof(cutlass::half_t);
    }
    return static_cast<size_t>(params.topk + 2) * sizeof(float)
         + static_cast<size_t>(K_TILE) * sizeof(int)
         + static_cast<size_t>(K_TILE) * static_cast<size_t>(HEAD_DIM_QK) * sizeof(cutlass::half_t);
}

__device__ __forceinline__ void compute_lse_deterministic(
    const SparseAttnFwdParams &params,
    const float *scores,
    int q_idx,
    int head_idx,
    float *row_lse_smem,
    float *sink_scale_smem) {
    float max_score = -CUDART_INF_F;
#pragma unroll 1
    for (int topk_idx = 0; topk_idx < params.topk; ++topk_idx) {
        max_score = fmaxf(max_score, scores[topk_idx]);
    }

    float denom = 0.0F;
    if (max_score != -CUDART_INF_F) {
#pragma unroll 1
        for (int topk_idx = 0; topk_idx < params.topk; ++topk_idx) {
            const float score = scores[topk_idx];
            if (score != -CUDART_INF_F) {
                denom += expf(score - max_score);
            }
        }
    }
    const bool has_tokens = denom > 0.0F;
    const float row_lse = has_tokens ? logf(denom) + max_score : CUDART_INF_F;
    float sink_scale = 1.0F;
    if (params.attn_sink != nullptr && has_tokens) {
        const float sink = params.attn_sink[head_idx];
        sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
    }

    *max_logits_ptr_for(params, q_idx, head_idx) = max_score;
    *lse_ptr_for(params, q_idx, head_idx) = row_lse;
    *row_lse_smem = row_lse;
    *sink_scale_smem = sink_scale;
}

template<int HEAD_DIM_QK>
__global__ void __launch_bounds__(NUM_THREADS)
sm70_sparse_prefill_fast_fwd_kernel(const SparseAttnFwdParams params) {
    static_assert(HEAD_DIM_QK == 512 || HEAD_DIM_QK == 576);

    const int q_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int q_heads_per_kv = params.h_q / params.h_kv;
    const int kv_head_idx = head_idx / q_heads_per_kv;

    extern __shared__ float shared[];
    const cutlass::bfloat16_t *q = q_ptr_for(params, q_idx, head_idx);
    const int *indices = indices_ptr_for(params, q_idx, kv_head_idx);
    const int valid_topk = valid_topk_for_query(params, q_idx);
    cutlass::bfloat16_t *out = out_ptr_for(params, q_idx, head_idx);

    if constexpr (USE_MMA_884_ONLINE) {
        float *tile_scores = shared;
        float *online_scalars = tile_scores + K_TILE;
        float *output_accum = online_scalars + 4;
        if constexpr (USE_DOUBLE_BUFFER) {
            int *online_token_refs_0 = reinterpret_cast<int*>(output_accum + HEAD_DIM_V);
            int *online_token_refs_1 = online_token_refs_0 + K_TILE;
            cutlass::half_t *online_kv_tile_0 = reinterpret_cast<cutlass::half_t*>(online_token_refs_1 + K_TILE);
            cutlass::half_t *online_kv_tile_1 = online_kv_tile_0 + K_TILE * HEAD_DIM_QK;
            accumulate_output_mma884_online_double_buffer<HEAD_DIM_QK>(
                params,
                q,
                indices,
                q_idx,
                head_idx,
                kv_head_idx,
                valid_topk,
                out,
                tile_scores,
                online_scalars,
                output_accum,
                online_kv_tile_0,
                online_kv_tile_1,
                online_token_refs_0,
                online_token_refs_1);
            return;
        }
        int *online_token_refs = reinterpret_cast<int*>(output_accum + HEAD_DIM_V);
        cutlass::half_t *online_kv_tile = reinterpret_cast<cutlass::half_t*>(online_token_refs + K_TILE);
        accumulate_output_mma884_online<HEAD_DIM_QK>(
            params,
            q,
            indices,
            q_idx,
            head_idx,
            kv_head_idx,
            valid_topk,
            out,
            tile_scores,
            online_scalars,
            output_accum,
            online_kv_tile,
            online_token_refs);
        return;
    }

    float *scores = shared;
    float *row_lse_smem = shared + params.topk;
    float *sink_scale_smem = shared + params.topk + 1;
    int *token_refs = reinterpret_cast<int*>(shared + params.topk + 2);
    cutlass::half_t *kv_tile = reinterpret_cast<cutlass::half_t*>(token_refs + K_TILE);

    if constexpr (USE_MMA_884_QK) {
        fill_scores_mma884_qk<HEAD_DIM_QK>(params, q, indices, kv_head_idx, valid_topk, scores, kv_tile, token_refs);
    } else {
        fill_scores_warp_parallel<HEAD_DIM_QK>(params, q, indices, kv_head_idx, valid_topk, scores, kv_tile, token_refs);
    }
    __syncthreads();

    if (tid == 0) {
        compute_lse_deterministic(params, scores, q_idx, head_idx, row_lse_smem, sink_scale_smem);
    }
    __syncthreads();

    const float row_lse = *row_lse_smem;
    const float sink_scale = *sink_scale_smem;
    const bool has_tokens = row_lse != CUDART_INF_F;

    if constexpr (USE_MMA_884_PV) {
        accumulate_output_mma884_pv<HEAD_DIM_QK>(
            params,
            scores,
            indices,
            kv_head_idx,
            valid_topk,
            row_lse,
            sink_scale,
            out,
            kv_tile,
            token_refs);
    } else {
        for (int dim = tid; dim < params.d_v; dim += blockDim.x) {
            float acc = 0.0F;
            if (has_tokens) {
#pragma unroll 1
                for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE) {
                    const int tile_count = min(K_TILE, params.topk - tile_start);
                    stage_kv_tile_to_shared<HEAD_DIM_QK>(
                        params,
                        indices,
                        tile_start,
                        tile_count,
                        valid_topk,
                        kv_head_idx,
                        kv_tile,
                        token_refs);
#pragma unroll 1
                    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                        const int topk_idx = tile_start + tile_idx;
                        const float score = scores[topk_idx];
                        if (score != -CUDART_INF_F) {
                            acc += expf(score - row_lse) * shared_k_value<HEAD_DIM_QK>(kv_tile, tile_idx, dim);
                        }
                    }
                    __syncthreads();
                }
                acc *= sink_scale;
            }
            out[dim] = cutlass::bfloat16_t(acc);
        }
    }
}

}  // namespace detail

template<int HEAD_DIM_QK>
void run_fwd_kernel(const SparseAttnFwdParams &params) {
    static_assert(HEAD_DIM_QK == 512 || HEAD_DIM_QK == 576);
    FLASH_ASSERT(params.h_q == 64 || params.h_q == 128);
    FLASH_ASSERT(params.h_kv > 0);
    FLASH_ASSERT(params.h_q % params.h_kv == 0);
    FLASH_ASSERT(params.d_qk == HEAD_DIM_QK);
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.topk <= MAX_TOPK_FALLBACK);

    // Round 3 HPB path: batch HEADS_PER_BLOCK heads per CUDA block
    // so the KV tile is staged once and reused. Requires:
    //   - USE_MMA_884_ONLINE (the online-softmax path)
    //   - !USE_DOUBLE_BUFFER (double-buffer HPB not yet implemented)
    //   - q_heads_per_kv (= h_q / h_kv) divisible by HEADS_PER_BLOCK
    //     so all HPB heads in a block share the same kv_head_idx.
    //   - h_q divisible by HEADS_PER_BLOCK so no block has idle heads.
    // Falls back to the single-head kernel when HPB == 1 or any
    // precondition fails at runtime.
    const int q_heads_per_kv = params.h_q / params.h_kv;
    const bool hpb_viable =
        HEADS_PER_BLOCK > 1
        && USE_MMA_884_ONLINE
        && !USE_DOUBLE_BUFFER
        && (q_heads_per_kv % HEADS_PER_BLOCK) == 0
        && (params.h_q % HEADS_PER_BLOCK) == 0;

    if constexpr (HEADS_PER_BLOCK > 1 && USE_MMA_884_ONLINE && !USE_DOUBLE_BUFFER) {
        if (hpb_viable) {
            auto hpb_kernel = &detail::sm70_sparse_prefill_hpb_ws_fwd_kernel<HEAD_DIM_QK, HEADS_PER_BLOCK>;
            const size_t hpb_smem_size =
                detail::sm70_sparse_prefill_shared_bytes_hpb<HEAD_DIM_QK, HEADS_PER_BLOCK>();
            CHECK_CUDA(cudaFuncSetAttribute(hpb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, hpb_smem_size));
            hpb_kernel<<<dim3(params.s_q, params.h_q / HEADS_PER_BLOCK), dim3(NUM_THREADS), hpb_smem_size, params.stream>>>(params);
            CHECK_CUDA_KERNEL_LAUNCH();
            return;
        }
    }

    auto kernel = &detail::sm70_sparse_prefill_fast_fwd_kernel<HEAD_DIM_QK>;
    const size_t smem_size = detail::sm70_sparse_prefill_shared_bytes<HEAD_DIM_QK>(params);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.s_q, params.h_q), dim3(NUM_THREADS), smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sm70::prefill::sparse
