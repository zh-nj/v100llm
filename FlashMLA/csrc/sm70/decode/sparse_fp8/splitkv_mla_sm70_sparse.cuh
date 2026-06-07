#pragma once

#include <cstdio>
#include <cstdlib>

#include <math_constants.h>

#include <cutlass/half.h>

#include "sm70/common/mma_884_attention.h"
#include "sm70/common/softmax.h"

#include "params.h"
#include "utils.h"

#include "config.h"
#include "dequant.h"

namespace sm70::decode::sparse_fp8 {

namespace detail {

#ifdef FLASH_MLA_METER_SPARSE_DECODE
// Decode-side 6-stage clock64 counters. Written only by block(0,0) thread 0.
// s0 stage_kv, s1 QK, s2 softmax, s3 scale, s4 PV, s5 epilogue (per tile).
// static __device__: each TU (v32_fp8.cu, model1_fp8.cu) has its own
// device symbol. The dump/reset functions below are defined per-TU and
// access that TU's symbol via namespace-qualified wrappers the host
// layer accumulates across.
static __device__ unsigned long long g_decode_stage_cycles[6] = {0, 0, 0, 0, 0, 0};
static __device__ unsigned long long g_decode_stage_tile_count = 0;
static __device__ unsigned long long g_decode_stage_block_count = 0;

__host__ void decode_stage_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out,
    unsigned long long *block_count_out);
__host__ void decode_stage_meter_dump_tu_v32(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out,
    unsigned long long *block_count_out);
__host__ void decode_stage_meter_reset_tu_model1();
__host__ void decode_stage_meter_reset_tu_v32();
#endif

#ifdef FLASH_MLA_METER_DECODE_QK_SUB
// 4 sub-stage counters inside s1 (the QK MMA hot block).
// s1a load_q, s1b load_k, s1c mma884_accumulate_qk, s1d store scores.
static __device__ unsigned long long g_decode_qk_sub_cycles[4] = {0, 0, 0, 0};
static __device__ unsigned long long g_decode_qk_sub_group_count = 0;
static __device__ unsigned long long g_decode_qk_sub_dim_group_count = 0;

__host__ void decode_qk_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *group_count_out,
    unsigned long long *dim_group_count_out);
__host__ void decode_qk_sub_meter_dump_tu_v32(
    unsigned long long *cycles_out,
    unsigned long long *group_count_out,
    unsigned long long *dim_group_count_out);
__host__ void decode_qk_sub_meter_reset_tu_model1();
__host__ void decode_qk_sub_meter_reset_tu_v32();
#endif

#ifdef FLASH_MLA_METER_DECODE_S0_SUB
// 5 sub-stage counters inside s0 stage_kv_tile_to_shared.
// s0a resolve token_refs, s0b resolve sync, s0c main-cache KV load,
// s0d extra-cache pass, s0e final sync.
static __device__ unsigned long long g_decode_s0_sub_cycles[5] = {0, 0, 0, 0, 0};
static __device__ unsigned long long g_decode_s0_sub_tile_count = 0;

__host__ void decode_s0_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out);
__host__ void decode_s0_sub_meter_dump_tu_v32(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out);
__host__ void decode_s0_sub_meter_reset_tu_model1();
__host__ void decode_s0_sub_meter_reset_tu_v32();
#endif

#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
// 6 sub-stage counters inside s0c (the main-cache KV load pass).
// s0c0 token/row setup, s0c1 invalid/extra zero-fill, s0c2 page address setup,
// s0c3 fp8/scale dequant or rope load, s0c4 half conversion, s0c5 smem store.
static __device__ unsigned long long g_decode_s0c_sub_cycles[6] = {0, 0, 0, 0, 0, 0};
static __device__ unsigned long long g_decode_s0c_sub_tile_count = 0;
static __device__ unsigned long long g_decode_s0c_sub_token_count = 0;
static __device__ unsigned long long g_decode_s0c_sub_dim_count = 0;

__host__ void decode_s0c_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out,
    unsigned long long *token_count_out,
    unsigned long long *dim_count_out);
__host__ void decode_s0c_sub_meter_dump_tu_v32(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out,
    unsigned long long *token_count_out,
    unsigned long long *dim_count_out);
__host__ void decode_s0c_sub_meter_reset_tu_model1();
__host__ void decode_s0c_sub_meter_reset_tu_v32();
#endif

#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
// 6 sub-stage counters inside s4 (the PV MMA path).
// s4a0 out_frag zero init, s4a1 loadP/exp/half, s4a2 loadV from smem,
// s4a3 PV HMMA accumulate, s4a4 add row0 output to accumulator,
// s4a5 sync after the PV dim-group loop.
static __device__ unsigned long long g_decode_s4a_sub_cycles[6] = {0, 0, 0, 0, 0, 0};
static __device__ unsigned long long g_decode_s4a_sub_group_count = 0;
static __device__ unsigned long long g_decode_s4a_sub_dim_group_count = 0;
static __device__ unsigned long long g_decode_s4a_sub_tile_count = 0;

__host__ void decode_s4a_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *group_count_out,
    unsigned long long *dim_group_count_out,
    unsigned long long *tile_count_out);
__host__ void decode_s4a_sub_meter_dump_tu_v32(
    unsigned long long *cycles_out,
    unsigned long long *group_count_out,
    unsigned long long *dim_group_count_out,
    unsigned long long *tile_count_out);
__host__ void decode_s4a_sub_meter_reset_tu_model1();
__host__ void decode_s4a_sub_meter_reset_tu_v32();
#endif

__host__ __device__ __forceinline__ int ceil_to_multiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

__device__ __forceinline__ float bf16_to_float(const cutlass::bfloat16_t value) {
    return static_cast<float>(value);
}

__device__ __forceinline__ const cutlass::bfloat16_t* q_ptr_for(
    const SparseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int head_idx) {
    return params.q
         + batch_idx * params.stride_q_b
         + q_seq_idx * params.stride_q_s_q
         + head_idx * params.stride_q_h_q;
}

__device__ __forceinline__ cutlass::bfloat16_t* out_ptr_for(
    const SparseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int head_idx) {
    return params.out
         + batch_idx * params.stride_o_b
         + q_seq_idx * params.stride_o_s_q
         + head_idx * params.stride_o_h_q;
}

__device__ __forceinline__ float* lse_ptr_for(
    const SparseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int head_idx) {
    return params.lse
         + batch_idx * params.stride_lse_b
         + q_seq_idx * params.stride_lse_s_q
         + head_idx;
}

__device__ __forceinline__ float* oaccum_ptr_for(
    const SparseAttnDecodeParams &params,
    int split_idx,
    int q_seq_idx,
    int head_idx) {
    return params.o_accum
         + split_idx * params.stride_o_accum_split
         + q_seq_idx * params.stride_o_accum_s_q
         + head_idx * params.stride_o_accum_h_q;
}

__device__ __forceinline__ float* lseaccum_ptr_for(
    const SparseAttnDecodeParams &params,
    int split_idx,
    int q_seq_idx,
    int head_idx) {
    return params.lse_accum
         + split_idx * params.stride_lse_accum_split
         + q_seq_idx * params.stride_lse_accum_s_q
         + head_idx;
}

__device__ __forceinline__ const uint8_t* v32_token_ptr_for(
    const SparseAttnDecodeParams &params,
    int token_index,
    bool is_extra) {
    const int page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
    const int block_stride = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    const int row_stride = is_extra ? params.stride_extra_kv_row : params.stride_kv_row;
    const int block_index = static_cast<int>(static_cast<uint32_t>(token_index) /
                                            static_cast<uint32_t>(page_block_size));
    const int row_index = static_cast<int>(static_cast<uint32_t>(token_index) %
                                          static_cast<uint32_t>(page_block_size));
    const auto* kv_bytes = reinterpret_cast<const uint8_t*>(is_extra ? params.extra_kv : params.kv);
    return kv_bytes + block_index * block_stride + row_index * row_stride;
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ float load_k_value(
    const SparseAttnDecodeParams &params,
    int token_index,
    bool is_extra,
    int dim) {
    if constexpr (MODEL_TYPE == ModelType::V32) {
        return v32_load_k_value(v32_token_ptr_for(params, token_index, is_extra), dim);
    } else {
        const int page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
        const int block_stride = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
        const int block_index = static_cast<int>(static_cast<uint32_t>(token_index) /
                                                static_cast<uint32_t>(page_block_size));
        const int row_index = static_cast<int>(static_cast<uint32_t>(token_index) %
                                              static_cast<uint32_t>(page_block_size));
        const auto* kv_bytes = reinterpret_cast<const uint8_t*>(is_extra ? params.extra_kv : params.kv);
        return model1_load_k_value(
            kv_bytes + block_index * block_stride, page_block_size, row_index, dim);
    }
}

#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
template<ModelType MODEL_TYPE>
__device__ __forceinline__ float load_k_value_s0c_metered(
    const SparseAttnDecodeParams &params,
    int token_index,
    bool is_extra,
    int dim,
    unsigned long long &address_cycles,
    unsigned long long &data_cycles) {
    const unsigned long long t0 = clock64();
    if constexpr (MODEL_TYPE == ModelType::V32) {
        const uint8_t* token_base = v32_token_ptr_for(params, token_index, is_extra);
        const float* scales = reinterpret_cast<const float*>(token_base + V32_NOPE_DIM);
        const auto* rope = reinterpret_cast<const cutlass::bfloat16_t*>(
            token_base + V32_NOPE_DIM + V32_NUM_SCALES * static_cast<int>(sizeof(float)));
        const int scale_idx = dim / V32_SCALE_GROUP;
        const unsigned long long t1 = clock64();
        float value;
        if (dim < V32_NOPE_DIM) {
            const float dequant = s0c3_e4m3_to_float(token_base[dim]) *
                                  s0c3_v32_scale_to_float(scales[scale_idx]);
            value = s0c3_round_to_float(dequant);
        } else {
            value = s0c3_rope_to_float(rope, dim - V32_NOPE_DIM);
        }
        const unsigned long long t2 = clock64();
        address_cycles += t1 - t0;
        data_cycles += t2 - t1;
        return value;
    } else {
        const int page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
        const int block_stride = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
        const int block_index = static_cast<int>(static_cast<uint32_t>(token_index) /
                                                static_cast<uint32_t>(page_block_size));
        const int row_index = static_cast<int>(static_cast<uint32_t>(token_index) %
                                              static_cast<uint32_t>(page_block_size));
        const auto* kv_bytes = reinterpret_cast<const uint8_t*>(is_extra ? params.extra_kv : params.kv);
        const uint8_t* block_base = kv_bytes + block_index * block_stride;
        const uint8_t* token_base = block_base + row_index * MODEL1_TOKEN_DATA_BYTES;
        const uint8_t* scales = block_base
            + page_block_size * MODEL1_TOKEN_DATA_BYTES
            + row_index * MODEL1_SCALE_STRIDE;
        const auto* rope = reinterpret_cast<const cutlass::bfloat16_t*>(token_base + MODEL1_NOPE_DIM);
        const int scale_idx = dim / MODEL1_SCALE_GROUP;
        const unsigned long long t1 = clock64();
        float value;
        if (dim < MODEL1_NOPE_DIM) {
            const float dequant = s0c3_e4m3_to_float(token_base[dim])
                                * s0c3_e8m0_to_float(scales[scale_idx]);
            value = s0c3_round_to_float(dequant);
        } else {
            value = s0c3_rope_to_float(rope, dim - MODEL1_NOPE_DIM);
        }
        const unsigned long long t2 = clock64();
        address_cycles += t1 - t0;
        data_cycles += t2 - t1;
        return value;
    }
}
#endif

template<ModelType MODEL_TYPE>
__host__ __device__ __forceinline__ constexpr int head_dim_qk() {
    if constexpr (MODEL_TYPE == ModelType::V32) {
        return HEAD_DIM_QK_V32;
    } else {
        return HEAD_DIM_QK_MODEL1;
    }
}

template<ModelType MODEL_TYPE>
__host__ __device__ __forceinline__ constexpr bool use_mma884_online_register_accumulator() {
    return NUM_THREADS == MMA_884_REGISTER_OUTPUT_CTA_THREADS && MODEL_TYPE == ModelType::MODEL1;
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ float qk_score_token(
    const cutlass::bfloat16_t* q,
    const SparseAttnDecodeParams &params,
    int token_index,
    bool is_extra,
    float sm_scale) {
    float acc = 0.0F;
#pragma unroll 1
    for (int dim = 0; dim < head_dim_qk<MODEL_TYPE>(); ++dim) {
        acc += bf16_to_float(q[dim]) * load_k_value<MODEL_TYPE>(params, token_index, is_extra, dim);
    }
    return acc * sm_scale;
}


template<ModelType MODEL_TYPE>
__device__ __forceinline__ float shared_k_value(
    const cutlass::half_t* kv_tile,
    int local_token_idx,
    int dim) {
    return static_cast<float>(kv_tile[local_token_idx * kv_row_stride(head_dim_qk<MODEL_TYPE>()) + dim]);
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ float qk_score_token_from_shared(
    const cutlass::bfloat16_t* q,
    const cutlass::half_t* kv_tile,
    int local_token_idx,
    float sm_scale) {
    float acc = 0.0F;
#pragma unroll 1
    for (int dim = 0; dim < head_dim_qk<MODEL_TYPE>(); ++dim) {
        acc += bf16_to_float(q[dim]) * shared_k_value<MODEL_TYPE>(kv_tile, local_token_idx, dim);
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

template<ModelType MODEL_TYPE>
__device__ __forceinline__ float qk_score_token_from_shared_warp(
    const cutlass::bfloat16_t* q,
    const cutlass::half_t* kv_tile,
    int local_token_idx,
    int lane,
    float sm_scale) {
    float partial = 0.0F;
#pragma unroll 1
    for (int dim = lane; dim < head_dim_qk<MODEL_TYPE>(); dim += 32) {
        partial += bf16_to_float(q[dim]) * shared_k_value<MODEL_TYPE>(kv_tile, local_token_idx, dim);
    }
    return warp_reduce_sum(partial) * sm_scale;
}

__device__ __forceinline__ half half_from_float(float value) {
    return __float2half_rn(value);
}

__device__ __forceinline__ half probability_half_from_score(
    float score,
    float row_lse) {
    return score != -CUDART_INF_F ? half_from_float(expf(score - row_lse)) : half_from_float(0.0F);
}

__device__ __forceinline__ half probability_half_from_online_score(
    float score,
    float row_max) {
    return score != -CUDART_INF_F ? half_from_float(expf(score - row_max)) : half_from_float(0.0F);
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void load_mma884_q_fragment(
    const cutlass::bfloat16_t* q,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_q) {
    const int q_row = flash_mla::sm70::attention::mma884_q_fragment_row(lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = dim_base + i;
        frag_q[i] = q_row == 0 && dim < head_dim_qk<MODEL_TYPE>()
            ? half_from_float(bf16_to_float(q[dim]))
            : half_from_float(0.0F);
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void load_mma884_q_fragment_half(
    const half* q,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_q) {
    const int q_row = flash_mla::sm70::attention::mma884_q_fragment_row(lane);
    if (q_row == 0 && (dim_base + 3) < head_dim_qk<MODEL_TYPE>()) {
        const uint2 packed = *reinterpret_cast<const uint2*>(&q[dim_base]);
        *reinterpret_cast<uint2*>(&frag_q[0]) = packed;
    } else {
        const half zero_h = __ushort_as_half(0);
        frag_q[0] = zero_h;
        frag_q[1] = zero_h;
        frag_q[2] = zero_h;
        frag_q[3] = zero_h;
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void load_mma884_q_fragment_half_batch2(
    const half* q0,
    const half* q1,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_q) {
    const int q_row = flash_mla::sm70::attention::mma884_q_fragment_row(lane);
    const half* q = q_row == 0 ? q0 : (q_row == 1 ? q1 : nullptr);
    if (q != nullptr && (dim_base + 3) < head_dim_qk<MODEL_TYPE>()) {
        const uint2 packed = *reinterpret_cast<const uint2*>(&q[dim_base]);
        *reinterpret_cast<uint2*>(&frag_q[0]) = packed;
    } else {
        const half zero_h = __ushort_as_half(0);
        frag_q[0] = zero_h;
        frag_q[1] = zero_h;
        frag_q[2] = zero_h;
        frag_q[3] = zero_h;
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void load_mma884_k_fragment(
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_k) {
    const int token_offset = flash_mla::sm70::attention::mma884_qk_token_offset(lane);
    const int local_token_idx = token_base + token_offset;
    const bool valid_token = local_token_idx < tile_count && token_refs[local_token_idx] != 0;
    const int row_base = local_token_idx * kv_row_stride(head_dim_qk<MODEL_TYPE>());
    const half zero_h = __ushort_as_half(0);
    const bool can_vector_load =
        valid_token && (dim_base + 3) < head_dim_qk<MODEL_TYPE>();
    if (can_vector_load) {
        // The QK loop advances dim_base by 4, and each smem row is
        // head_dim_qk half values. Both supported decode layouts have row
        // strides divisible by 8 bytes, so this is an aligned 4-half load.
        const uint2 packed = *reinterpret_cast<const uint2*>(
            &kv_tile[row_base + dim_base]);
        *reinterpret_cast<uint2*>(&frag_k[0]) = packed;
        return;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = dim_base + i;
        const bool valid = valid_token && dim < head_dim_qk<MODEL_TYPE>();
        frag_k[i] = valid
            ? reinterpret_cast<const half &>(kv_tile[row_base + dim])
            : zero_h;
    }
}

__device__ __forceinline__ void store_mma884_batch2_scores(
    const flash_mla::sm70::Array<float, 8> &score_frag,
    float* scores0,
    float* scores1,
    const int* token_refs,
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
                const int q_row = flash_mla::sm70::attention::mma884_qk_score_row(lane, q);
                const int s_col = flash_mla::sm70::attention::mma884_qk_score_col(lane, s1, s0);
                const int local_token_idx = token_base + s_col;
                if ((q_row == 0 || q_row == 1) && local_token_idx < tile_count) {
                    const float score = token_refs[local_token_idx] != 0
                        ? score_frag[s1 * 4 + q * 2 + s0] * sm_scale
                        : -CUDART_INF_F;
                    if (q_row == 0) {
                        scores0[local_token_idx] = score;
                    } else {
                        scores1[local_token_idx] = score;
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void store_mma884_qk_row0_scores(
    const flash_mla::sm70::Array<float, 8> &score_frag,
    float* scores,
    const int* token_refs,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
    flash_mla::sm70::attention::store_mma884_row0_scores<true>(
        score_frag, scores, token_refs, token_base, tile_count, lane, sm_scale);
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_qk_group_batch2_half_q(
    const half* q0,
    const half* q1,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    float* scores0,
    float* scores1,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
    flash_mla::sm70::Array<float, 8> score_frag{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_frag[i] = 0.0F;
    }

#pragma unroll 1
    for (int dim_base = 0; dim_base < head_dim_qk<MODEL_TYPE>(); dim_base += 4) {
        flash_mla::sm70::Array<half, 4> frag_q{};
        flash_mla::sm70::Array<half, 4> frag_k{};
        load_mma884_q_fragment_half_batch2<MODEL_TYPE>(q0, q1, dim_base, lane, frag_q);
        load_mma884_k_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_k);
        flash_mla::sm70::attention::mma884_accumulate_qk(score_frag, frag_q, frag_k);
    }
    store_mma884_batch2_scores(score_frag, scores0, scores1, token_refs, token_base, tile_count, lane, sm_scale);
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_qk_group(
    const cutlass::bfloat16_t* q,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    float* scores,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
    flash_mla::sm70::Array<float, 8> score_frag{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_frag[i] = 0.0F;
    }

#ifdef FLASH_MLA_METER_DECODE_QK_SUB
    const bool QK_SUB_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    unsigned long long QK_SUB_C0 = 0, QK_SUB_C1 = 0, QK_SUB_C2 = 0;
#endif
#pragma unroll 1
    for (int dim_base = 0; dim_base < head_dim_qk<MODEL_TYPE>(); dim_base += 4) {
        flash_mla::sm70::Array<half, 4> frag_q{};
        flash_mla::sm70::Array<half, 4> frag_k{};
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T0 = QK_SUB_ACTIVE ? clock64() : 0;
#endif
        load_mma884_q_fragment<MODEL_TYPE>(q, dim_base, lane, frag_q);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T1 = QK_SUB_ACTIVE ? clock64() : 0;
        if (QK_SUB_ACTIVE) QK_SUB_C0 += (QK_SUB_T1 - QK_SUB_T0);
#endif
        load_mma884_k_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_k);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T2 = QK_SUB_ACTIVE ? clock64() : 0;
        if (QK_SUB_ACTIVE) QK_SUB_C1 += (QK_SUB_T2 - QK_SUB_T1);
#endif
        flash_mla::sm70::attention::mma884_accumulate_qk(score_frag, frag_q, frag_k);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T3 = QK_SUB_ACTIVE ? clock64() : 0;
        if (QK_SUB_ACTIVE) {
            QK_SUB_C2 += (QK_SUB_T3 - QK_SUB_T2);
            atomicAdd(&detail::g_decode_qk_sub_dim_group_count, 1ULL);
        }
#endif
    }
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
    unsigned long long QK_SUB_T4 = QK_SUB_ACTIVE ? clock64() : 0;
#endif
    store_mma884_qk_row0_scores(score_frag, scores, token_refs, token_base, tile_count, lane, sm_scale);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
    if (QK_SUB_ACTIVE) {
        unsigned long long QK_SUB_T5 = clock64();
        atomicAdd(&detail::g_decode_qk_sub_cycles[0], QK_SUB_C0);
        atomicAdd(&detail::g_decode_qk_sub_cycles[1], QK_SUB_C1);
        atomicAdd(&detail::g_decode_qk_sub_cycles[2], QK_SUB_C2);
        atomicAdd(&detail::g_decode_qk_sub_cycles[3], (QK_SUB_T5 - QK_SUB_T4));
        atomicAdd(&detail::g_decode_qk_sub_group_count, 1ULL);
    }
#endif
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_qk_group_half_q(
    const half* q,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    float* scores,
    int token_base,
    int tile_count,
    int lane,
    float sm_scale) {
    flash_mla::sm70::Array<float, 8> score_frag{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_frag[i] = 0.0F;
    }

#ifdef FLASH_MLA_METER_DECODE_QK_SUB
    const bool QK_SUB_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    unsigned long long QK_SUB_C0 = 0, QK_SUB_C1 = 0, QK_SUB_C2 = 0;
#endif
#pragma unroll 1
    for (int dim_base = 0; dim_base < head_dim_qk<MODEL_TYPE>(); dim_base += 4) {
        flash_mla::sm70::Array<half, 4> frag_q{};
        flash_mla::sm70::Array<half, 4> frag_k{};
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T0 = QK_SUB_ACTIVE ? clock64() : 0;
#endif
        load_mma884_q_fragment_half<MODEL_TYPE>(q, dim_base, lane, frag_q);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T1 = QK_SUB_ACTIVE ? clock64() : 0;
        if (QK_SUB_ACTIVE) QK_SUB_C0 += (QK_SUB_T1 - QK_SUB_T0);
#endif
        load_mma884_k_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_k);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T2 = QK_SUB_ACTIVE ? clock64() : 0;
        if (QK_SUB_ACTIVE) QK_SUB_C1 += (QK_SUB_T2 - QK_SUB_T1);
#endif
        flash_mla::sm70::attention::mma884_accumulate_qk(score_frag, frag_q, frag_k);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
        unsigned long long QK_SUB_T3 = QK_SUB_ACTIVE ? clock64() : 0;
        if (QK_SUB_ACTIVE) {
            QK_SUB_C2 += (QK_SUB_T3 - QK_SUB_T2);
            atomicAdd(&detail::g_decode_qk_sub_dim_group_count, 1ULL);
        }
#endif
    }
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
    unsigned long long QK_SUB_T4 = QK_SUB_ACTIVE ? clock64() : 0;
#endif
    store_mma884_qk_row0_scores(score_frag, scores, token_refs, token_base, tile_count, lane, sm_scale);
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
    if (QK_SUB_ACTIVE) {
        unsigned long long QK_SUB_T5 = clock64();
        atomicAdd(&detail::g_decode_qk_sub_cycles[0], QK_SUB_C0);
        atomicAdd(&detail::g_decode_qk_sub_cycles[1], QK_SUB_C1);
        atomicAdd(&detail::g_decode_qk_sub_cycles[2], QK_SUB_C2);
        atomicAdd(&detail::g_decode_qk_sub_cycles[3], (QK_SUB_T5 - QK_SUB_T4));
        atomicAdd(&detail::g_decode_qk_sub_group_count, 1ULL);
    }
#endif
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void stage_q_to_shared_half_batch2(
    const cutlass::bfloat16_t* q0,
    const cutlass::bfloat16_t* q1,
    half* q0_smem,
    half* q1_smem) {
    for (int dim = threadIdx.x; dim < head_dim_qk<MODEL_TYPE>(); dim += blockDim.x) {
        q0_smem[dim] = half_from_float(bf16_to_float(q0[dim]));
        q1_smem[dim] = half_from_float(bf16_to_float(q1[dim]));
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void stage_q_to_shared_half(
    const cutlass::bfloat16_t* q,
    half* q_smem) {
    for (int dim = threadIdx.x; dim < head_dim_qk<MODEL_TYPE>(); dim += blockDim.x) {
        q_smem[dim] = half_from_float(bf16_to_float(q[dim]));
    }
}

__device__ __forceinline__ void load_mma884_online_p_fragment_cached_batch2(
    const half* p_cache0,
    const half* p_cache1,
    int token_base,
    int tile_count,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    const int p_row = flash_mla::sm70::attention::mma884_p_fragment_row(lane);
    const half* p_cache = p_row == 0 ? p_cache0 : (p_row == 1 ? p_cache1 : nullptr);
    const half zero_h = __ushort_as_half(0);
    if (p_cache != nullptr) {
        half loaded[4];
        const uint2 packed = *reinterpret_cast<const uint2*>(&p_cache[token_base]);
        *reinterpret_cast<uint2*>(loaded) = packed;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int local_token_idx = token_base + i;
            frag_p[i] = local_token_idx < tile_count ? loaded[i] : zero_h;
        }
    } else {
        frag_p[0] = zero_h;
        frag_p[1] = zero_h;
        frag_p[2] = zero_h;
        frag_p[3] = zero_h;
    }
}

__device__ __forceinline__ void load_mma884_p_fragment(
    const float* scores,
    int tile_start,
    int token_base,
    int tile_count,
    int lane,
    float row_lse,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    flash_mla::sm70::attention::load_mma884_p_fragment_from_scores<true>(
        scores, tile_start, token_base, tile_count, lane, row_lse, frag_p);
}

__device__ __forceinline__ void load_mma884_online_p_fragment(
    const float* tile_scores,
    int token_base,
    int tile_count,
    int lane,
    float row_max,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    flash_mla::sm70::attention::load_mma884_p_fragment_from_scores<true>(
        tile_scores, 0, token_base, tile_count, lane, row_max, frag_p);
}

template<int ONLINE_K_TILE>
__device__ __forceinline__ void populate_mma884_online_p_cache(
    const float* tile_scores,
    int tile_count,
    float row_max,
    half* p_cache) {
#pragma unroll 1
    for (int t = threadIdx.x; t < ONLINE_K_TILE; t += blockDim.x) {
        p_cache[t] = t < tile_count
            ? probability_half_from_online_score(tile_scores[t], row_max)
            : flash_mla::sm70::attention::half_from_float(0.0F);
    }
}

__device__ __forceinline__ void load_mma884_online_p_fragment_cached(
    const half* p_cache,
    int token_base,
    int tile_count,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    const int p_row = flash_mla::sm70::attention::mma884_p_fragment_row(lane);
    const half zero_h = __ushort_as_half(0);
    half loaded[4];
    const uint2 packed = *reinterpret_cast<const uint2*>(&p_cache[token_base]);
    *reinterpret_cast<uint2*>(loaded) = packed;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int local_token_idx = token_base + i;
        frag_p[i] = p_row == 0 && local_token_idx < tile_count
            ? loaded[i]
            : zero_h;
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void load_mma884_v_fragment(
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_v) {
    const int token_offset = flash_mla::sm70::attention::mma884_pv_token_offset(lane);
    const int local_token_idx = token_base + token_offset;
    const int dim_offset = flash_mla::sm70::attention::mma884_pv_dim_offset(lane);
    const bool valid_token = local_token_idx < tile_count && token_refs[local_token_idx] != 0;
    const int row_base = local_token_idx * kv_row_stride(head_dim_qk<MODEL_TYPE>());
    const half zero_h = __ushort_as_half(0);
    const int base_dim = dim_base + dim_offset;
    if constexpr (USE_V_FRAGMENT_VECTOR) {
        if (valid_token) {
            // base_dim is 8-byte aligned:
            //   dim_base is a multiple of 16 from the outer PV loop
            //   dim_offset is in {0, 4, 8, 12}
            // so the four half values can be fetched with one 64-bit load.
            const half* row_ptr = reinterpret_cast<const half*>(kv_tile) + row_base + base_dim;
            const uint2 packed = *reinterpret_cast<const uint2*>(row_ptr);
            *reinterpret_cast<uint2*>(&frag_v[0]) = packed;
        } else {
            frag_v[0] = zero_h;
            frag_v[1] = zero_h;
            frag_v[2] = zero_h;
            frag_v[3] = zero_h;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dim = base_dim + i;
            frag_v[i] = valid_token && dim < HEAD_DIM_V
                ? reinterpret_cast<const half &>(kv_tile[row_base + dim])
                : zero_h;
        }
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_pv_group(
    const float* scores,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int tile_start,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    float row_lse,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
    load_mma884_p_fragment(scores, tile_start, token_base, tile_count, lane, row_lse, frag_p);
    load_mma884_v_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
    flash_mla::sm70::attention::mma884_accumulate_pv(out_frag, frag_p, frag_v);
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_online_pv_group_cached(
    const half* p_cache,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    const bool S4A_SUB_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    unsigned long long S4A_SUB_T0 = S4A_SUB_ACTIVE ? clock64() : 0;
#endif
    load_mma884_online_p_fragment_cached(p_cache, token_base, tile_count, lane, frag_p);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    unsigned long long S4A_SUB_T1 = S4A_SUB_ACTIVE ? clock64() : 0;
    if (S4A_SUB_ACTIVE) {
        atomicAdd(&detail::g_decode_s4a_sub_cycles[1], S4A_SUB_T1 - S4A_SUB_T0);
        S4A_SUB_T0 = S4A_SUB_T1;
    }
#endif
    load_mma884_v_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    S4A_SUB_T1 = S4A_SUB_ACTIVE ? clock64() : 0;
    if (S4A_SUB_ACTIVE) {
        atomicAdd(&detail::g_decode_s4a_sub_cycles[2], S4A_SUB_T1 - S4A_SUB_T0);
        S4A_SUB_T0 = S4A_SUB_T1;
    }
#endif
    flash_mla::sm70::attention::mma884_accumulate_pv(out_frag, frag_p, frag_v);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    if (S4A_SUB_ACTIVE) {
        S4A_SUB_T1 = clock64();
        atomicAdd(&detail::g_decode_s4a_sub_cycles[3], S4A_SUB_T1 - S4A_SUB_T0);
        atomicAdd(&detail::g_decode_s4a_sub_group_count, 1ULL);
    }
#endif
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_online_pv_group_cached_batch2(
    const half* p_cache0,
    const half* p_cache1,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
    load_mma884_online_p_fragment_cached_batch2(
        p_cache0,
        p_cache1,
        token_base,
        tile_count,
        lane,
        frag_p);
    load_mma884_v_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
    flash_mla::sm70::attention::mma884_accumulate_pv(out_frag, frag_p, frag_v);
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void compute_mma884_online_pv_group(
    const float* tile_scores,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    float row_max,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    const bool S4A_SUB_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    unsigned long long S4A_SUB_T0 = S4A_SUB_ACTIVE ? clock64() : 0;
#endif
    load_mma884_online_p_fragment(tile_scores, token_base, tile_count, lane, row_max, frag_p);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    unsigned long long S4A_SUB_T1 = S4A_SUB_ACTIVE ? clock64() : 0;
    if (S4A_SUB_ACTIVE) {
        atomicAdd(&detail::g_decode_s4a_sub_cycles[1], S4A_SUB_T1 - S4A_SUB_T0);
        S4A_SUB_T0 = S4A_SUB_T1;
    }
#endif
    load_mma884_v_fragment<MODEL_TYPE>(kv_tile, token_refs, token_base, tile_count, dim_base, lane, frag_v);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    S4A_SUB_T1 = S4A_SUB_ACTIVE ? clock64() : 0;
    if (S4A_SUB_ACTIVE) {
        atomicAdd(&detail::g_decode_s4a_sub_cycles[2], S4A_SUB_T1 - S4A_SUB_T0);
        S4A_SUB_T0 = S4A_SUB_T1;
    }
#endif
    flash_mla::sm70::attention::mma884_accumulate_pv(out_frag, frag_p, frag_v);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    if (S4A_SUB_ACTIVE) {
        S4A_SUB_T1 = clock64();
        atomicAdd(&detail::g_decode_s4a_sub_cycles[3], S4A_SUB_T1 - S4A_SUB_T0);
        atomicAdd(&detail::g_decode_s4a_sub_group_count, 1ULL);
    }
#endif
}

__device__ __forceinline__ int valid_topk_for_batch(
    const int* topk_length,
    int batch_idx,
    int topk) {
    if (topk_length == nullptr) {
        return topk;
    }
    return max(0, min(__ldg(topk_length + batch_idx), topk));
}

__device__ __forceinline__ int main_logical_span_for_batch(
    const SparseAttnDecodeParams &params,
    int valid_topk) {
    if (params.extra_topk == 0) {
        return params.topk;
    }
    return max(ceil_to_multiple(max(valid_topk, 1), TOPK_BLOCK_SIZE), TOPK_BLOCK_SIZE);
}

__device__ __forceinline__ int logical_score_capacity_for_batch(
    const SparseAttnDecodeParams &params,
    int valid_topk) {
    return main_logical_span_for_batch(params, valid_topk) + params.extra_topk;
}

__host__ __device__ __forceinline__ int logical_score_capacity_for_launch(
    const SparseAttnDecodeParams &params) {
    if (params.extra_topk == 0) {
        return params.topk;
    }
    return ceil_to_multiple(params.topk, TOPK_BLOCK_SIZE) + params.extra_topk;
}

__device__ __forceinline__ bool score_ref_for_logical_idx(
    const SparseAttnDecodeParams &params,
    const int* indices,
    const int* extra_indices,
    int logical_idx,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    int &token_index,
    bool &is_extra) {
    if (params.extra_topk > 0 && logical_idx >= main_logical_span) {
        const int extra_idx = logical_idx - main_logical_span;
        is_extra = true;
        if (extra_indices == nullptr || extra_idx >= valid_extra_topk) {
            token_index = -1;
            return false;
        }
        token_index = __ldg(extra_indices + extra_idx);
        return token_index >= 0;
    }

    is_extra = false;
    if (logical_idx >= valid_topk || logical_idx >= params.topk) {
        token_index = -1;
        return false;
    }
    token_index = __ldg(indices + logical_idx);
    return token_index >= 0;
}

__device__ __forceinline__ bool topk_rows_match(
    const SparseAttnDecodeParams &params,
    int q_seq_idx,
    int logical_start,
    int local_score_count,
    int valid_topk0,
    int valid_topk1,
    int valid_extra_topk0,
    int valid_extra_topk1,
    int main_logical_span0,
    int main_logical_span1,
    int* match_flag) {
    if (threadIdx.x == 0) {
        *match_flag = (valid_topk0 == valid_topk1) &&
            (valid_extra_topk0 == valid_extra_topk1) &&
            (main_logical_span0 == main_logical_span1);
    }
    __syncthreads();
    if (*match_flag == 0) {
        __syncthreads();
        return false;
    }

    const int* indices0 = params.indices
        + q_seq_idx * params.stride_indices_s_q;
    const int* indices1 = params.indices
        + params.stride_indices_b
        + q_seq_idx * params.stride_indices_s_q;
    const int* extra_indices0 = params.extra_indices == nullptr
        ? nullptr
        : params.extra_indices + q_seq_idx * params.stride_extra_indices_s_q;
    const int* extra_indices1 = params.extra_indices == nullptr
        ? nullptr
        : params.extra_indices + params.stride_extra_indices_b
                               + q_seq_idx * params.stride_extra_indices_s_q;

    for (int local_idx = threadIdx.x; local_idx < local_score_count; local_idx += blockDim.x) {
        const int logical_idx = logical_start + local_idx;
        int token0 = -1;
        int token1 = -1;
        bool is_extra0 = false;
        bool is_extra1 = false;
        const bool valid0 = score_ref_for_logical_idx(
            params,
            indices0,
            extra_indices0,
            logical_idx,
            valid_topk0,
            valid_extra_topk0,
            main_logical_span0,
            token0,
            is_extra0);
        const bool valid1 = score_ref_for_logical_idx(
            params,
            indices1,
            extra_indices1,
            logical_idx,
            valid_topk1,
            valid_extra_topk1,
            main_logical_span1,
            token1,
            is_extra1);
        if (valid0 != valid1 ||
            (valid0 && (token0 != token1 || is_extra0 != is_extra1))) {
            atomicExch(match_flag, 0);
        }
    }
    __syncthreads();
    const bool match = *match_flag != 0;
    __syncthreads();
    return match;
}

__device__ __forceinline__ int encode_token_ref(int token_index, bool is_extra) {
    return is_extra ? -(token_index + 1) : token_index + 1;
}

__device__ __forceinline__ bool decode_token_ref(int encoded, int &token_index, bool &is_extra) {
    if (encoded == 0) {
        token_index = -1;
        is_extra = false;
        return false;
    }
    is_extra = encoded < 0;
    token_index = is_extra ? -encoded - 1 : encoded - 1;
    return true;
}

template<ModelType MODEL_TYPE>
__device__ void stage_kv_tile_to_shared(
    const SparseAttnDecodeParams &params,
    const int* indices,
    const int* extra_indices,
    int logical_tile_start,
    int tile_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    cutlass::half_t* kv_tile,
    int* token_refs) {
    const int head_dim = head_dim_qk<MODEL_TYPE>();
    const int row_stride = kv_row_stride(head_dim);

#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    const bool S0_SUB_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    const unsigned long long S0_SUB_T0 = S0_SUB_ACTIVE ? clock64() : 0;
#endif
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
    const bool S0C_SUB_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    unsigned long long S0C_SUB_C[6] = {0, 0, 0, 0, 0, 0};
    unsigned long long S0C_SUB_TOKEN_COUNT = 0;
    unsigned long long S0C_SUB_DIM_COUNT = 0;
#endif

    // Phase 1: Resolve all indices upfront (coalesced index loading).
    // Each thread resolves one or more token indices, avoiding redundant
    // __ldg index reads that would otherwise occur once per dimension.
    for (int tok = threadIdx.x; tok < tile_count; tok += blockDim.x) {
        const int logical_idx = logical_tile_start + tok;
        int token_index = -1;
        bool is_extra = false;
        const bool valid_score = score_ref_for_logical_idx(
            params,
            indices,
            extra_indices,
            logical_idx,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            token_index,
            is_extra);
        token_refs[tok] = valid_score ? encode_token_ref(token_index, is_extra) : 0;
    }
#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    const unsigned long long S0_SUB_T1 = S0_SUB_ACTIVE ? clock64() : 0;
#endif
    __syncthreads();
#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    const unsigned long long S0_SUB_T2 = S0_SUB_ACTIVE ? clock64() : 0;
#endif

    // Phase 2: Load KV data using precomputed token_refs.
    // Global memory reads for indices are eliminated; each thread only
    // does page-address computation + FP8 dequant from the KV cache.
    //
    // Dual-cache access ordering optimization (SM70 L2 thrashing mitigation):
    // When a tile straddles the SWA/compressed boundary, tokens reference two
    // different paged caches (params.kv vs params.extra_kv) residing in
    // separate memory regions with different page tables. Interleaving reads
    // between these regions causes L2 thrashing on V100 (only 3 MB L2).
    // We split the load into two passes: first all SWA (main) tokens, then
    // all compressed (extra) tokens. This keeps each pass accessing a single
    // contiguous memory region, improving L2 residency and TLB hit rate.
    //
    // For tiles that are entirely SWA or entirely compressed (the common case),
    // the second pass simply exits early, so the overhead is minimal.

    // Warp-cooperative layout: warp_id picks tokens, lane_id picks dims.
    // This keeps 128 consecutive dims inside one warp so the scale pointer
    // is effectively loaded once per scale group instead of once per
    // thread-write.
    {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warps_per_cta = blockDim.x >> 5;
        for (int tok = warp; tok < tile_count; tok += warps_per_cta) {
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
            const unsigned long long S0C_T0 = S0C_SUB_ACTIVE ? clock64() : 0;
#endif
            const int encoded = token_refs[tok];
            int token_index;
            bool is_extra;
            const bool valid = decode_token_ref(encoded, token_index, is_extra);
            const int row_off = tok * row_stride;
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
            if (S0C_SUB_ACTIVE) {
                S0C_SUB_C[0] += clock64() - S0C_T0;
                S0C_SUB_TOKEN_COUNT += 1ULL;
            }
#endif
            if (!valid) {
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
                const unsigned long long S0C_T1 = S0C_SUB_ACTIVE ? clock64() : 0;
#endif
                for (int dim = lane; dim < head_dim; dim += 32) {
                    kv_tile[row_off + dim] = cutlass::half_t(0.0F);
                }
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
                if (S0C_SUB_ACTIVE) S0C_SUB_C[1] += clock64() - S0C_T1;
#endif
            } else if (!is_extra) {
                for (int dim = lane; dim < head_dim; dim += 32) {
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
                    if (S0C_SUB_ACTIVE) {
                        float value = load_k_value_s0c_metered<MODEL_TYPE>(
                            params, token_index, false, dim, S0C_SUB_C[2], S0C_SUB_C[3]);
                        const unsigned long long S0C_T2 = clock64();
                        const cutlass::half_t half_value = cutlass::half_t(value);
                        const unsigned long long S0C_T3 = clock64();
                        kv_tile[row_off + dim] = half_value;
                        const unsigned long long S0C_T4 = clock64();
                        S0C_SUB_C[4] += S0C_T3 - S0C_T2;
                        S0C_SUB_C[5] += S0C_T4 - S0C_T3;
                        S0C_SUB_DIM_COUNT += 1ULL;
                    } else
#endif
                    kv_tile[row_off + dim] = cutlass::half_t(load_k_value<MODEL_TYPE>(params, token_index, false, dim));
                }
            } else {
                // Extra-cache token: zero here; pass 2 fills in.
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
                const unsigned long long S0C_T5 = S0C_SUB_ACTIVE ? clock64() : 0;
#endif
                for (int dim = lane; dim < head_dim; dim += 32) {
                    kv_tile[row_off + dim] = cutlass::half_t(0.0F);
                }
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
                if (S0C_SUB_ACTIVE) S0C_SUB_C[1] += clock64() - S0C_T5;
#endif
            }
        }
    }
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
    if (S0C_SUB_ACTIVE) {
        for (int i = 0; i < 6; ++i) {
            atomicAdd(&detail::g_decode_s0c_sub_cycles[i], S0C_SUB_C[i]);
        }
        atomicAdd(&detail::g_decode_s0c_sub_tile_count, 1ULL);
        atomicAdd(&detail::g_decode_s0c_sub_token_count, S0C_SUB_TOKEN_COUNT);
        atomicAdd(&detail::g_decode_s0c_sub_dim_count, S0C_SUB_DIM_COUNT);
    }
#endif
#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    const unsigned long long S0_SUB_T3 = S0_SUB_ACTIVE ? clock64() : 0;
#endif
    if (params.extra_topk > 0) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warps_per_cta = blockDim.x >> 5;
        for (int tok = warp; tok < tile_count; tok += warps_per_cta) {
            const int encoded = token_refs[tok];
            int token_index;
            bool is_extra;
            const bool valid = decode_token_ref(encoded, token_index, is_extra);
            if (valid && is_extra) {
                const int row_off = tok * row_stride;
                for (int dim = lane; dim < head_dim; dim += 32) {
                    kv_tile[row_off + dim] = cutlass::half_t(load_k_value<MODEL_TYPE>(params, token_index, true, dim));
                }
            }
        }
    }
#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    const unsigned long long S0_SUB_T4 = S0_SUB_ACTIVE ? clock64() : 0;
#endif
    __syncthreads();
#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    if (S0_SUB_ACTIVE) {
        const unsigned long long S0_SUB_T5 = clock64();
        atomicAdd(&detail::g_decode_s0_sub_cycles[0], S0_SUB_T1 - S0_SUB_T0);
        atomicAdd(&detail::g_decode_s0_sub_cycles[1], S0_SUB_T2 - S0_SUB_T1);
        atomicAdd(&detail::g_decode_s0_sub_cycles[2], S0_SUB_T3 - S0_SUB_T2);
        atomicAdd(&detail::g_decode_s0_sub_cycles[3], S0_SUB_T4 - S0_SUB_T3);
        atomicAdd(&detail::g_decode_s0_sub_cycles[4], S0_SUB_T5 - S0_SUB_T4);
        atomicAdd(&detail::g_decode_s0_sub_tile_count, 1ULL);
    }
#endif
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ float fill_scores_mma884_qk(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q,
    const int* indices,
    const int* extra_indices,
    int logical_start,
    int local_score_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    float* scores,
    int* token_refs,
    cutlass::half_t* kv_tile) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;

#pragma unroll 1
    for (int tile_start = 0; tile_start < local_score_count; tile_start += SPARSE_K_TILE) {
        const int tile_count = min(SPARSE_K_TILE, local_score_count - tile_start);
        stage_kv_tile_to_shared<MODEL_TYPE>(
            params,
            indices,
            extra_indices,
            logical_start + tile_start,
            tile_count,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            kv_tile,
            token_refs + tile_start);

        for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
            compute_mma884_qk_group<MODEL_TYPE>(
                q,
                kv_tile,
                token_refs + tile_start,
                scores + tile_start,
                token_base,
                tile_count,
                lane,
                params.sm_scale);
        }
        __syncthreads();
    }

    float local_max = -CUDART_INF_F;
#pragma unroll 1
    for (int local_idx = threadIdx.x; local_idx < local_score_count; local_idx += blockDim.x) {
        local_max = fmaxf(local_max, scores[local_idx]);
    }
    return local_max;
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void fill_tile_scores_mma884_qk(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int tile_count,
    float* tile_scores) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
        compute_mma884_qk_group<MODEL_TYPE>(
            q,
            kv_tile,
            token_refs,
            tile_scores,
            token_base,
            tile_count,
            lane,
            params.sm_scale);
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void fill_tile_scores_mma884_qk_half_q(
    const SparseAttnDecodeParams &params,
    const half* q,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int tile_count,
    float* tile_scores) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
        compute_mma884_qk_group_half_q<MODEL_TYPE>(
            q,
            kv_tile,
            token_refs,
            tile_scores,
            token_base,
            tile_count,
            lane,
            params.sm_scale);
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void fill_tile_scores_mma884_qk_batch2_half_q(
    const SparseAttnDecodeParams &params,
    const half* q0,
    const half* q1,
    const cutlass::half_t* kv_tile,
    const int* token_refs,
    int tile_count,
    float* tile_scores0,
    float* tile_scores1) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
        compute_mma884_qk_group_batch2_half_q<MODEL_TYPE>(
            q0,
            q1,
            kv_tile,
            token_refs,
            tile_scores0,
            tile_scores1,
            token_base,
            tile_count,
            lane,
            params.sm_scale);
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ float fill_scores_warp_parallel(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q,
    const int* indices,
    const int* extra_indices,
    int logical_start,
    int local_score_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    float* scores,
    int* token_refs,
    cutlass::half_t* kv_tile) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    float local_max = -CUDART_INF_F;

#pragma unroll 1
    for (int tile_start = 0; tile_start < local_score_count; tile_start += SPARSE_K_TILE) {
        const int tile_count = min(SPARSE_K_TILE, local_score_count - tile_start);
        stage_kv_tile_to_shared<MODEL_TYPE>(
            params,
            indices,
            extra_indices,
            logical_start + tile_start,
            tile_count,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            kv_tile,
            token_refs + tile_start);

        for (int tile_base = 0; tile_base < tile_count; tile_base += warps_per_cta) {
            const int tile_idx = tile_base + warp_idx;
            if (tile_idx < tile_count) {
                const int local_idx = tile_start + tile_idx;
                float score = -CUDART_INF_F;
                if (token_refs[local_idx] != 0) {
                    score = qk_score_token_from_shared_warp<MODEL_TYPE>(
                        q,
                        kv_tile,
                        tile_idx,
                        lane,
                        params.sm_scale);
                }
                if (lane == 0) {
                    scores[local_idx] = score;
                    local_max = fmaxf(local_max, score);
                }
            }
        }
        __syncthreads();
    }

    return local_max;
}

__device__ __forceinline__ float block_reduce_max(float value, float *shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float *shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ __forceinline__ void write_split_accumulators(
    const SparseAttnDecodeParams &params,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    int dim,
    float raw_acc,
    float row_lse,
    bool has_tokens) {
    if (dim == 0) {
        *lseaccum_ptr_for(params, split_idx, q_seq_idx, head_idx) =
            has_tokens ? row_lse * CUDART_L2E_F : -CUDART_INF_F;
    }
    oaccum_ptr_for(params, split_idx, q_seq_idx, head_idx)[dim] = has_tokens ? raw_acc : 0.0F;
}

template<int D_GROUPS_PER_WARP>
__device__ __forceinline__ void write_mma884_online_register_output_batch2(
    const SparseAttnDecodeParams &params,
    flash_mla::sm70::Array<float, 8> (&out_frags)[D_GROUPS_PER_WARP],
    int split_idx0,
    int split_idx1,
    int q_seq_idx,
    int head_idx,
    float row_sum0,
    float row_sum1,
    float row_lse0,
    float row_lse1,
    int warp_idx,
    int lane,
    int warps_per_cta) {
    const bool has_tokens0 = row_sum0 > 0.0F;
    const bool has_tokens1 = row_sum1 > 0.0F;
    const float inv_row_sum0 = has_tokens0 ? 1.0F / row_sum0 : 0.0F;
    const float inv_row_sum1 = has_tokens1 ? 1.0F / row_sum1 : 0.0F;
    const int row_base = (lane >> 4) * 4 + (lane & 8) + (lane & 1);
    const int col_base = (lane & 4) * 2 + (lane & 2);

#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
        const int dim_base = (group_idx * warps_per_cta + warp_idx) * 16;
        if (dim_base < params.d_v) {
#pragma unroll
            for (int d1 = 0; d1 < 2; ++d1) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
#pragma unroll
                    for (int d0 = 0; d0 < 2; ++d0) {
                        const int q_row = row_base + q * 2;
                        const int dim = dim_base + col_base + d1 * 4 + d0;
                        if (dim < params.d_v) {
                            if (q_row == 0) {
                                const float raw_acc = has_tokens0
                                    ? out_frags[group_idx][d1 * 4 + q * 2 + d0] * inv_row_sum0
                                    : 0.0F;
                                write_split_accumulators(
                                    params,
                                    split_idx0,
                                    q_seq_idx,
                                    head_idx,
                                    dim,
                                    raw_acc,
                                    row_lse0,
                                    has_tokens0);
                            } else if (q_row == 1) {
                                const float raw_acc = has_tokens1
                                    ? out_frags[group_idx][d1 * 4 + q * 2 + d0] * inv_row_sum1
                                    : 0.0F;
                                write_split_accumulators(
                                    params,
                                    split_idx1,
                                    q_seq_idx,
                                    head_idx,
                                    dim,
                                    raw_acc,
                                    row_lse1,
                                    has_tokens1);
                            }
                        }
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void write_no_split_accumulators(
    const SparseAttnDecodeParams &params,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    int dim,
    float raw_acc,
    float row_lse,
    bool has_tokens) {
    write_split_accumulators(params, split_idx, q_seq_idx, head_idx, dim, raw_acc, row_lse, has_tokens);
}

__device__ __forceinline__ void write_mma884_pv_row0_result(
    const SparseAttnDecodeParams &params,
    const flash_mla::sm70::Array<float, 8> &out_frag,
    cutlass::bfloat16_t* out,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    int dim_base,
    int lane,
    float row_lse,
    float sink_scale,
    bool has_tokens,
    bool is_no_split) {
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
                if (q_row == 0 && dim < params.d_v) {
                    const float raw_acc = has_tokens ? out_frag[d1 * 4 + q * 2 + d0] : 0.0F;
                    if (is_no_split) {
                        write_no_split_accumulators(
                            params,
                            split_idx,
                            q_seq_idx,
                            head_idx,
                            dim,
                            raw_acc,
                            row_lse,
                            has_tokens);
                        out[dim] = cutlass::bfloat16_t(raw_acc * sink_scale);
                    } else {
                        write_split_accumulators(
                            params,
                            split_idx,
                            q_seq_idx,
                            head_idx,
                            dim,
                            raw_acc,
                            row_lse,
                            has_tokens);
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void add_mma884_pv_row0_output_to_accum(
    const flash_mla::sm70::Array<float, 8> &out_frag,
    float* output_accum,
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

__device__ __forceinline__ void zero_online_output_accumulator(float* output_accum) {
    for (int dim = threadIdx.x; dim < HEAD_DIM_V; dim += blockDim.x) {
        output_accum[dim] = 0.0F;
    }
}

__device__ __forceinline__ void scale_online_output_accumulator(float* output_accum, float scale) {
    for (int dim = threadIdx.x; dim < HEAD_DIM_V; dim += blockDim.x) {
        output_accum[dim] *= scale;
    }
}

__device__ __forceinline__ void write_online_output(
    const SparseAttnDecodeParams &params,
    float* output_accum,
    cutlass::bfloat16_t* out,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    float row_sum,
    float row_lse,
    float sink_scale,
    bool is_no_split) {
    const bool has_tokens = row_sum > 0.0F;
    for (int dim = threadIdx.x; dim < params.d_v; dim += blockDim.x) {
        const float raw_acc = has_tokens ? output_accum[dim] / row_sum : 0.0F;
        if (is_no_split) {
            write_no_split_accumulators(
                params,
                split_idx,
                q_seq_idx,
                head_idx,
                dim,
                raw_acc,
                row_lse,
                has_tokens);
            out[dim] = cutlass::bfloat16_t(raw_acc * sink_scale);
        } else {
            write_split_accumulators(
                params,
                split_idx,
                q_seq_idx,
                head_idx,
                dim,
                raw_acc,
                row_lse,
                has_tokens);
        }
    }
}

template<int D_GROUPS_PER_WARP>
__device__ __forceinline__ void scale_mma884_online_register_fragments(
    flash_mla::sm70::Array<float, 8> (&out_frags)[D_GROUPS_PER_WARP],
    float scale) {
#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_frags[group_idx][i] *= scale;
        }
    }
}

template<int D_GROUPS_PER_WARP>
__device__ __forceinline__ void scale_mma884_online_register_fragments_batch2(
    flash_mla::sm70::Array<float, 8> (&out_frags)[D_GROUPS_PER_WARP],
    float scale0,
    float scale1,
    int lane) {
    const int row_base = (lane >> 4) * 4 + (lane & 8) + (lane & 1);
#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
#pragma unroll
        for (int d1 = 0; d1 < 2; ++d1) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
#pragma unroll
                for (int d0 = 0; d0 < 2; ++d0) {
                    const int q_row = row_base + q * 2;
                    const int frag_idx = d1 * 4 + q * 2 + d0;
                    if (q_row == 0) {
                        out_frags[group_idx][frag_idx] *= scale0;
                    } else if (q_row == 1) {
                        out_frags[group_idx][frag_idx] *= scale1;
                    }
                }
            }
        }
    }
}

template<int D_GROUPS_PER_WARP>
__device__ __forceinline__ void write_mma884_online_register_output(
    const SparseAttnDecodeParams &params,
    flash_mla::sm70::Array<float, 8> (&out_frags)[D_GROUPS_PER_WARP],
    cutlass::bfloat16_t* out,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    float row_sum,
    float row_lse,
    float sink_scale,
    bool is_no_split,
    int warp_idx,
    int lane,
    int warps_per_cta) {
    const bool has_tokens = row_sum > 0.0F;
    const float inv_row_sum = has_tokens ? 1.0F / row_sum : 0.0F;
    const int row_base = (lane >> 4) * 4 + (lane & 8) + (lane & 1);
    const int col_base = (lane & 4) * 2 + (lane & 2);

#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
        const int dim_base = (group_idx * warps_per_cta + warp_idx) * 16;
        if (dim_base < params.d_v) {
#pragma unroll
            for (int d1 = 0; d1 < 2; ++d1) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
#pragma unroll
                    for (int d0 = 0; d0 < 2; ++d0) {
                        const int q_row = row_base + q * 2;
                        const int dim = dim_base + col_base + d1 * 4 + d0;
                        if (q_row == 0 && dim < params.d_v) {
                            const float raw_acc = has_tokens
                                ? out_frags[group_idx][d1 * 4 + q * 2 + d0] * inv_row_sum
                                : 0.0F;
                            if (is_no_split) {
                                write_no_split_accumulators(
                                    params,
                                    split_idx,
                                    q_seq_idx,
                                    head_idx,
                                    dim,
                                    raw_acc,
                                    row_lse,
                                    has_tokens);
                                out[dim] = cutlass::bfloat16_t(raw_acc * sink_scale);
                            } else {
                                write_split_accumulators(
                                    params,
                                    split_idx,
                                    q_seq_idx,
                                    head_idx,
                                    dim,
                                    raw_acc,
                                    row_lse,
                                    has_tokens);
                            }
                        }
                    }
                }
            }
        }
    }
}

template<ModelType MODEL_TYPE>
__device__ __forceinline__ void accumulate_output_mma884_pv(
    const SparseAttnDecodeParams &params,
    const int* indices,
    const int* extra_indices,
    int logical_start,
    int local_score_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    const float* scores,
    int* token_refs,
    cutlass::half_t* kv_tile,
    bool is_no_split,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    cutlass::bfloat16_t* out,
    float row_lse,
    float sink_scale,
    bool has_tokens) {
    if (!has_tokens) {
        for (int dim = threadIdx.x; dim < params.d_v; dim += blockDim.x) {
            if (is_no_split) {
                write_no_split_accumulators(
                    params,
                    split_idx,
                    q_seq_idx,
                    head_idx,
                    dim,
                    0.0F,
                    row_lse,
                    false);
                out[dim] = cutlass::bfloat16_t(0.0F);
            } else {
                write_split_accumulators(
                    params,
                    split_idx,
                    q_seq_idx,
                    head_idx,
                    dim,
                    0.0F,
                    row_lse,
                    false);
            }
        }
        return;
    }

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;

#pragma unroll 1
    for (int dim_group_base = 0; dim_group_base < HEAD_DIM_V; dim_group_base += warps_per_cta * 16) {
        const int dim_base = dim_group_base + warp_idx * 16;
        const bool dim_valid = dim_base < params.d_v;
        flash_mla::sm70::Array<float, 8> out_frag{};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_frag[i] = 0.0F;
        }

#pragma unroll 1
        for (int tile_start = 0; tile_start < local_score_count; tile_start += SPARSE_K_TILE) {
            const int tile_count = min(SPARSE_K_TILE, local_score_count - tile_start);
            stage_kv_tile_to_shared<MODEL_TYPE>(
                params,
                indices,
                extra_indices,
                logical_start + tile_start,
                tile_count,
                valid_topk,
                valid_extra_topk,
                main_logical_span,
                kv_tile,
                token_refs + tile_start);
            if (dim_valid) {
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_pv_group<MODEL_TYPE>(
                        scores,
                        kv_tile,
                        token_refs + tile_start,
                        tile_start,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_lse,
                        out_frag);
                }
            }
            __syncthreads();
        }

        if (dim_valid) {
            write_mma884_pv_row0_result(
                params,
                out_frag,
                out,
                split_idx,
                q_seq_idx,
                head_idx,
                dim_base,
                lane,
                row_lse,
                sink_scale,
                has_tokens,
                is_no_split);
        }
    }
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__device__ __forceinline__ void accumulate_output_mma884_online(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q,
    const int* indices,
    const int* extra_indices,
    int logical_start,
    int local_score_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    bool is_no_split,
    int batch_idx,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    cutlass::bfloat16_t* out,
    float* tile_scores,
    float* online_scalars,
    float* online_reduce_scratch,
    float* output_accum,
    cutlass::half_t* kv_tile,
    int* token_refs,
    half* q_smem,
    half* p_cache) {
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;
    constexpr int ROW_LSE_SLOT = 4;

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[SINK_SCALE_SLOT] = 1.0F;
        online_scalars[ROW_LSE_SLOT] = CUDART_INF_F;
    }
    zero_online_output_accumulator(output_accum);
    if constexpr (USE_STAGE_Q_HALF) {
        stage_q_to_shared_half<MODEL_TYPE>(q, q_smem);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;

#pragma unroll 1
    for (int tile_start = 0; tile_start < local_score_count; tile_start += ONLINE_K_TILE) {
        const int tile_count = min(ONLINE_K_TILE, local_score_count - tile_start);
#ifdef FLASH_MLA_METER_SPARSE_DECODE
        const bool METER_ACTIVE =
            (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
            (threadIdx.x == 0);
        unsigned long long METER_T0 = METER_ACTIVE ? clock64() : 0;
#endif
        stage_kv_tile_to_shared<MODEL_TYPE>(
            params,
            indices,
            extra_indices,
            logical_start + tile_start,
            tile_count,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            kv_tile,
            token_refs);
#ifdef FLASH_MLA_METER_SPARSE_DECODE
        unsigned long long METER_T1 = METER_ACTIVE ? clock64() : 0;
        if (METER_ACTIVE) g_decode_stage_cycles[0] += (METER_T1 - METER_T0);
#endif
        if constexpr (USE_STAGE_Q_HALF) {
            fill_tile_scores_mma884_qk_half_q<MODEL_TYPE>(params, q_smem, kv_tile, token_refs, tile_count, tile_scores);
        } else {
            fill_tile_scores_mma884_qk<MODEL_TYPE>(params, q, kv_tile, token_refs, tile_count, tile_scores);
        }
        __syncthreads();
#ifdef FLASH_MLA_METER_SPARSE_DECODE
        unsigned long long METER_T2 = METER_ACTIVE ? clock64() : 0;
        if (METER_ACTIVE) g_decode_stage_cycles[1] += (METER_T2 - METER_T1);
#endif

        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<true>(
            tile_scores,
            tile_count,
            online_scalars + ROW_MAX_SLOT,
            online_scalars + ROW_SUM_SLOT,
            online_scalars + ONLINE_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();
#ifdef FLASH_MLA_METER_SPARSE_DECODE
        unsigned long long METER_T3 = METER_ACTIVE ? clock64() : 0;
        if (METER_ACTIVE) g_decode_stage_cycles[2] += (METER_T3 - METER_T2);
#endif

        scale_online_output_accumulator(output_accum, online_scalars[ONLINE_SCALE_SLOT]);
        const float row_max = online_scalars[ROW_MAX_SLOT];
        if constexpr (USE_PV_P_CACHE) {
            populate_mma884_online_p_cache<ONLINE_K_TILE>(
                tile_scores, tile_count, row_max, p_cache);
        }
        __syncthreads();
#ifdef FLASH_MLA_METER_SPARSE_DECODE
        unsigned long long METER_T4 = METER_ACTIVE ? clock64() : 0;
        if (METER_ACTIVE) g_decode_stage_cycles[3] += (METER_T4 - METER_T3);
#endif

#pragma unroll 1
        for (int dim_group_base = 0; dim_group_base < HEAD_DIM_V; dim_group_base += warps_per_cta * 16) {
            const int dim_base = dim_group_base + warp_idx * 16;
            if (dim_base < params.d_v) {
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
                const bool S4A_OUTER_ACTIVE =
                    (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
                    (threadIdx.x == 0);
                unsigned long long S4A_OUTER_T0 = S4A_OUTER_ACTIVE ? clock64() : 0;
#endif
                flash_mla::sm70::Array<float, 8> out_frag{};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out_frag[i] = 0.0F;
                }
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
                if (S4A_OUTER_ACTIVE) {
                    unsigned long long S4A_OUTER_T1 = clock64();
                    atomicAdd(&detail::g_decode_s4a_sub_cycles[0],
                              S4A_OUTER_T1 - S4A_OUTER_T0);
                }
#endif
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    if constexpr (USE_PV_P_CACHE) {
                        compute_mma884_online_pv_group_cached<MODEL_TYPE>(
                            p_cache,
                            kv_tile,
                            token_refs,
                            token_base,
                            tile_count,
                            dim_base,
                            lane,
                            out_frag);
                    } else {
                        compute_mma884_online_pv_group<MODEL_TYPE>(
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
                }
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
                if (S4A_OUTER_ACTIVE) S4A_OUTER_T0 = clock64();
#endif
                add_mma884_pv_row0_output_to_accum(out_frag, output_accum, dim_base, lane);
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
                if (S4A_OUTER_ACTIVE) {
                    unsigned long long S4A_OUTER_T1 = clock64();
                    atomicAdd(&detail::g_decode_s4a_sub_cycles[4],
                              S4A_OUTER_T1 - S4A_OUTER_T0);
                    atomicAdd(&detail::g_decode_s4a_sub_dim_group_count, 1ULL);
                }
#endif
            }
        }
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
        const bool S4A_SYNC_ACTIVE =
            (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
            (threadIdx.x == 0);
        unsigned long long S4A_SYNC_T0 = S4A_SYNC_ACTIVE ? clock64() : 0;
#endif
        __syncthreads();
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
        if (S4A_SYNC_ACTIVE) {
            unsigned long long S4A_SYNC_T1 = clock64();
            atomicAdd(&detail::g_decode_s4a_sub_cycles[5],
                      S4A_SYNC_T1 - S4A_SYNC_T0);
            atomicAdd(&detail::g_decode_s4a_sub_tile_count, 1ULL);
        }
#endif
#ifdef FLASH_MLA_METER_SPARSE_DECODE
        unsigned long long METER_T5 = METER_ACTIVE ? clock64() : 0;
        if (METER_ACTIVE) {
            g_decode_stage_cycles[4] += (METER_T5 - METER_T4);
            g_decode_stage_tile_count += 1;
        }
#endif
    }

    if (threadIdx.x == 0) {
        const float row_max = online_scalars[ROW_MAX_SLOT];
        const float row_sum = online_scalars[ROW_SUM_SLOT];
        const bool has_tokens = row_sum > 0.0F;
        const float row_lse = has_tokens ? logf(row_sum) + row_max : CUDART_INF_F;
        float sink_scale = 1.0F;
        if (is_no_split && params.attn_sink != nullptr && has_tokens) {
            const float sink = __ldg(params.attn_sink + head_idx);
            sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
        }
        if (is_no_split) {
            *lse_ptr_for(params, batch_idx, q_seq_idx, head_idx) = row_lse;
        }
        online_scalars[SINK_SCALE_SLOT] = sink_scale;
        online_scalars[ROW_LSE_SLOT] = row_lse;
    }
    __syncthreads();

#ifdef FLASH_MLA_METER_SPARSE_DECODE
    const bool METER_EPI_ACTIVE =
        (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
        (threadIdx.x == 0);
    unsigned long long METER_EPI_T0 = METER_EPI_ACTIVE ? clock64() : 0;
#endif
    write_online_output(
        params,
        output_accum,
        out,
        split_idx,
        q_seq_idx,
        head_idx,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[ROW_LSE_SLOT],
        online_scalars[SINK_SCALE_SLOT],
        is_no_split);
#ifdef FLASH_MLA_METER_SPARSE_DECODE
    if (METER_EPI_ACTIVE) {
        g_decode_stage_cycles[5] += (clock64() - METER_EPI_T0);
        g_decode_stage_block_count += 1;
    }
#endif
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__device__ __forceinline__ void accumulate_output_mma884_online_register(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q,
    const int* indices,
    const int* extra_indices,
    int logical_start,
    int local_score_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    bool is_no_split,
    int batch_idx,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    cutlass::bfloat16_t* out,
    float* tile_scores,
    float* online_scalars,
    float* online_reduce_scratch,
    cutlass::half_t* kv_tile,
    int* token_refs,
    half* q_smem) {
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;
    constexpr int ROW_LSE_SLOT = 4;
    constexpr int WARPS_PER_CTA = MMA_884_REGISTER_OUTPUT_CTA_THREADS / 32;
    constexpr int D_GROUPS_PER_WARP = (HEAD_DIM_V + WARPS_PER_CTA * 16 - 1) / (WARPS_PER_CTA * 16);

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[SINK_SCALE_SLOT] = 1.0F;
        online_scalars[ROW_LSE_SLOT] = CUDART_INF_F;
    }

    flash_mla::sm70::Array<float, 8> out_frags[D_GROUPS_PER_WARP];
#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_frags[group_idx][i] = 0.0F;
        }
    }

    if constexpr (USE_STAGE_Q_HALF) {
        stage_q_to_shared_half<MODEL_TYPE>(q, q_smem);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    constexpr int warps_per_cta = WARPS_PER_CTA;

#pragma unroll 1
    for (int tile_start = 0; tile_start < local_score_count; tile_start += ONLINE_K_TILE) {
        const int tile_count = min(ONLINE_K_TILE, local_score_count - tile_start);
        stage_kv_tile_to_shared<MODEL_TYPE>(
            params,
            indices,
            extra_indices,
            logical_start + tile_start,
            tile_count,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            kv_tile,
            token_refs);
        if constexpr (USE_STAGE_Q_HALF) {
            fill_tile_scores_mma884_qk_half_q<MODEL_TYPE>(params, q_smem, kv_tile, token_refs, tile_count, tile_scores);
        } else {
            fill_tile_scores_mma884_qk<MODEL_TYPE>(params, q, kv_tile, token_refs, tile_count, tile_scores);
        }
        __syncthreads();

        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<true>(
            tile_scores,
            tile_count,
            online_scalars + ROW_MAX_SLOT,
            online_scalars + ROW_SUM_SLOT,
            online_scalars + ONLINE_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();

        scale_mma884_online_register_fragments(out_frags, online_scalars[ONLINE_SCALE_SLOT]);
        const float row_max = online_scalars[ROW_MAX_SLOT];
#pragma unroll
        for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
            const int dim_base = (group_idx * warps_per_cta + warp_idx) * 16;
            if (dim_base < params.d_v) {
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_online_pv_group<MODEL_TYPE>(
                        tile_scores,
                        kv_tile,
                        token_refs,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_max,
                        out_frags[group_idx]);
                }
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
        if (is_no_split && params.attn_sink != nullptr && has_tokens) {
            const float sink = __ldg(params.attn_sink + head_idx);
            sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
        }
        if (is_no_split) {
            *lse_ptr_for(params, batch_idx, q_seq_idx, head_idx) = row_lse;
        }
        online_scalars[SINK_SCALE_SLOT] = sink_scale;
        online_scalars[ROW_LSE_SLOT] = row_lse;
    }
    __syncthreads();

    write_mma884_online_register_output(
        params,
        out_frags,
        out,
        split_idx,
        q_seq_idx,
        head_idx,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[ROW_LSE_SLOT],
        online_scalars[SINK_SCALE_SLOT],
        is_no_split,
        warp_idx,
        lane,
        warps_per_cta);
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__device__ __forceinline__ void accumulate_output_mma884_online_register_batch2_verify(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q0,
    const cutlass::bfloat16_t* q1,
    const int* indices0,
    const int* extra_indices0,
    int logical_start,
    int local_score_count,
    int valid_topk0,
    int valid_extra_topk0,
    int main_logical_span0,
    int split_idx0,
    int split_idx1,
    int q_seq_idx,
    int head_idx,
    float* tile_scores0,
    float* tile_scores1,
    float* online_scalars,
    float* online_reduce_scratch,
    cutlass::half_t* kv_tile,
    int* token_refs,
    half* q0_smem,
    half* q1_smem,
    half* p_cache0,
    half* p_cache1) {
    constexpr int ROW0_MAX_SLOT = 0;
    constexpr int ROW0_SUM_SLOT = 1;
    constexpr int ROW0_SCALE_SLOT = 2;
    constexpr int ROW1_MAX_SLOT = 3;
    constexpr int ROW1_SUM_SLOT = 4;
    constexpr int ROW1_SCALE_SLOT = 5;
    constexpr int ROW0_LSE_SLOT = 6;
    constexpr int ROW1_LSE_SLOT = 7;
    constexpr int WARPS_PER_CTA = MMA_884_REGISTER_OUTPUT_CTA_THREADS / 32;
    constexpr int D_GROUPS_PER_WARP = (HEAD_DIM_V + WARPS_PER_CTA * 16 - 1) / (WARPS_PER_CTA * 16);

    if (threadIdx.x == 0) {
        online_scalars[ROW0_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW0_SUM_SLOT] = 0.0F;
        online_scalars[ROW0_SCALE_SLOT] = 1.0F;
        online_scalars[ROW1_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW1_SUM_SLOT] = 0.0F;
        online_scalars[ROW1_SCALE_SLOT] = 1.0F;
        online_scalars[ROW0_LSE_SLOT] = CUDART_INF_F;
        online_scalars[ROW1_LSE_SLOT] = CUDART_INF_F;
    }

    flash_mla::sm70::Array<float, 8> out_frags[D_GROUPS_PER_WARP];
#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_frags[group_idx][i] = 0.0F;
        }
    }

    stage_q_to_shared_half_batch2<MODEL_TYPE>(q0, q1, q0_smem, q1_smem);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    constexpr int warps_per_cta = WARPS_PER_CTA;

#pragma unroll 1
    for (int tile_start = 0; tile_start < local_score_count; tile_start += ONLINE_K_TILE) {
        const int tile_count = min(ONLINE_K_TILE, local_score_count - tile_start);
        stage_kv_tile_to_shared<MODEL_TYPE>(
            params,
            indices0,
            extra_indices0,
            logical_start + tile_start,
            tile_count,
            valid_topk0,
            valid_extra_topk0,
            main_logical_span0,
            kv_tile,
            token_refs);
        fill_tile_scores_mma884_qk_batch2_half_q<MODEL_TYPE>(
            params,
            q0_smem,
            q1_smem,
            kv_tile,
            token_refs,
            tile_count,
            tile_scores0,
            tile_scores1);
        __syncthreads();
        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<true>(
            tile_scores0,
            tile_count,
            online_scalars + ROW0_MAX_SLOT,
            online_scalars + ROW0_SUM_SLOT,
            online_scalars + ROW0_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();
        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<true>(
            tile_scores1,
            tile_count,
            online_scalars + ROW1_MAX_SLOT,
            online_scalars + ROW1_SUM_SLOT,
            online_scalars + ROW1_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();
        scale_mma884_online_register_fragments_batch2(
            out_frags,
            online_scalars[ROW0_SCALE_SLOT],
            online_scalars[ROW1_SCALE_SLOT],
            lane);
        populate_mma884_online_p_cache<ONLINE_K_TILE>(
            tile_scores0,
            tile_count,
            online_scalars[ROW0_MAX_SLOT],
            p_cache0);
        populate_mma884_online_p_cache<ONLINE_K_TILE>(
            tile_scores1,
            tile_count,
            online_scalars[ROW1_MAX_SLOT],
            p_cache1);
        __syncthreads();

#pragma unroll
        for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
            const int dim_base = (group_idx * warps_per_cta + warp_idx) * 16;
            if (dim_base < params.d_v) {
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_online_pv_group_cached_batch2<MODEL_TYPE>(
                        p_cache0,
                        p_cache1,
                        kv_tile,
                        token_refs,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        out_frags[group_idx]);
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float row0_sum = online_scalars[ROW0_SUM_SLOT];
        const float row1_sum = online_scalars[ROW1_SUM_SLOT];
        online_scalars[ROW0_LSE_SLOT] = row0_sum > 0.0F
            ? logf(row0_sum) + online_scalars[ROW0_MAX_SLOT]
            : CUDART_INF_F;
        online_scalars[ROW1_LSE_SLOT] = row1_sum > 0.0F
            ? logf(row1_sum) + online_scalars[ROW1_MAX_SLOT]
            : CUDART_INF_F;
        *lseaccum_ptr_for(params, split_idx0, q_seq_idx, head_idx) =
            row0_sum > 0.0F ? online_scalars[ROW0_LSE_SLOT] * CUDART_L2E_F : -CUDART_INF_F;
        *lseaccum_ptr_for(params, split_idx1, q_seq_idx, head_idx) =
            row1_sum > 0.0F ? online_scalars[ROW1_LSE_SLOT] * CUDART_L2E_F : -CUDART_INF_F;
    }
    __syncthreads();

    write_mma884_online_register_output_batch2(
        params,
        out_frags,
        split_idx0,
        split_idx1,
        q_seq_idx,
        head_idx,
        online_scalars[ROW0_SUM_SLOT],
        online_scalars[ROW1_SUM_SLOT],
        online_scalars[ROW0_LSE_SLOT],
        online_scalars[ROW1_LSE_SLOT],
        warp_idx,
        lane,
        warps_per_cta);
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__device__ __forceinline__ void accumulate_output_mma884_online_register_double_buffer(
    const SparseAttnDecodeParams &params,
    const cutlass::bfloat16_t* q,
    const int* indices,
    const int* extra_indices,
    int logical_start,
    int local_score_count,
    int valid_topk,
    int valid_extra_topk,
    int main_logical_span,
    bool is_no_split,
    int batch_idx,
    int split_idx,
    int q_seq_idx,
    int head_idx,
    cutlass::bfloat16_t* out,
    float* tile_scores,
    float* online_scalars,
    float* online_reduce_scratch,
    cutlass::half_t* kv_tile_0,
    cutlass::half_t* kv_tile_1,
    int* token_refs_0,
    int* token_refs_1,
    half* q_smem) {
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int SINK_SCALE_SLOT = 3;
    constexpr int ROW_LSE_SLOT = 4;
    constexpr int WARPS_PER_CTA = MMA_884_REGISTER_OUTPUT_CTA_THREADS / 32;
    constexpr int D_GROUPS_PER_WARP = (HEAD_DIM_V + WARPS_PER_CTA * 16 - 1) / (WARPS_PER_CTA * 16);

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[SINK_SCALE_SLOT] = 1.0F;
        online_scalars[ROW_LSE_SLOT] = CUDART_INF_F;
    }

    flash_mla::sm70::Array<float, 8> out_frags[D_GROUPS_PER_WARP];
#pragma unroll
    for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_frags[group_idx][i] = 0.0F;
        }
    }

    if constexpr (USE_STAGE_Q_HALF) {
        stage_q_to_shared_half<MODEL_TYPE>(q, q_smem);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    constexpr int warps_per_cta = WARPS_PER_CTA;

    // Double-buffer pointers: ping-pong between buffer 0 and buffer 1
    cutlass::half_t* kv_bufs[2] = {kv_tile_0, kv_tile_1};
    int* ref_bufs[2] = {token_refs_0, token_refs_1};

    const int num_tiles = (local_score_count + ONLINE_K_TILE - 1) / ONLINE_K_TILE;

    if (num_tiles == 0) {
        // No tokens -- write zero output
        if (threadIdx.x == 0) {
            online_scalars[SINK_SCALE_SLOT] = 1.0F;
            online_scalars[ROW_LSE_SLOT] = CUDART_INF_F;
        }
        __syncthreads();
        write_mma884_online_register_output(
            params,
            out_frags,
            out,
            split_idx,
            q_seq_idx,
            head_idx,
            0.0F,
            CUDART_INF_F,
            1.0F,
            is_no_split,
            warp_idx,
            lane,
            warps_per_cta);
        return;
    }

    // Pre-stage first tile into buffer 0
    {
        const int tile_count_0 = min(ONLINE_K_TILE, local_score_count);
        stage_kv_tile_to_shared<MODEL_TYPE>(
            params, indices, extra_indices,
            logical_start, tile_count_0,
            valid_topk, valid_extra_topk, main_logical_span,
            kv_bufs[0], ref_bufs[0]);
    }

    int cur_buf = 0;

#pragma unroll 1
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int tile_start = tile_idx * ONLINE_K_TILE;
        const int tile_count = min(ONLINE_K_TILE, local_score_count - tile_start);

        cutlass::half_t* cur_kv = kv_bufs[cur_buf];
        int* cur_refs = ref_bufs[cur_buf];

        // Compute QK scores from current tile (already in shmem)
        if constexpr (USE_STAGE_Q_HALF) {
            fill_tile_scores_mma884_qk_half_q<MODEL_TYPE>(params, q_smem, cur_kv, cur_refs, tile_count, tile_scores);
        } else {
            fill_tile_scores_mma884_qk<MODEL_TYPE>(params, q, cur_kv, cur_refs, tile_count, tile_scores);
        }
        __syncthreads();

        // Online softmax update
        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<true>(
            tile_scores,
            tile_count,
            online_scalars + ROW_MAX_SLOT,
            online_scalars + ROW_SUM_SLOT,
            online_scalars + ONLINE_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();

        // Scale existing accumulation by correction factor
        scale_mma884_online_register_fragments(out_frags, online_scalars[ONLINE_SCALE_SLOT]);

        const float row_max = online_scalars[ROW_MAX_SLOT];

        // PV accumulation from current tile
#pragma unroll
        for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
            const int dim_base = (group_idx * warps_per_cta + warp_idx) * 16;
            if (dim_base < params.d_v) {
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_mma884_online_pv_group<MODEL_TYPE>(
                        tile_scores,
                        cur_kv,
                        cur_refs,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_max,
                        out_frags[group_idx]);
                }
            }
        }
        __syncthreads();

        // Pre-stage NEXT tile into the OTHER buffer (overlapped with the
        // __syncthreads fence above ensuring current-tile MMA is complete
        // before we reuse the alternate buffer).
        const int next_tile_idx = tile_idx + 1;
        if (next_tile_idx < num_tiles) {
            const int next_start = next_tile_idx * ONLINE_K_TILE;
            const int next_count = min(ONLINE_K_TILE, local_score_count - next_start);
            const int next_buf = 1 - cur_buf;
            stage_kv_tile_to_shared<MODEL_TYPE>(
                params, indices, extra_indices,
                logical_start + next_start, next_count,
                valid_topk, valid_extra_topk, main_logical_span,
                kv_bufs[next_buf], ref_bufs[next_buf]);
            cur_buf = next_buf;
        }
    }

    if (threadIdx.x == 0) {
        const float row_max = online_scalars[ROW_MAX_SLOT];
        const float row_sum = online_scalars[ROW_SUM_SLOT];
        const bool has_tokens = row_sum > 0.0F;
        const float row_lse = has_tokens ? logf(row_sum) + row_max : CUDART_INF_F;
        float sink_scale = 1.0F;
        if (is_no_split && params.attn_sink != nullptr && has_tokens) {
            const float sink = __ldg(params.attn_sink + head_idx);
            sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
        }
        if (is_no_split) {
            *lse_ptr_for(params, batch_idx, q_seq_idx, head_idx) = row_lse;
        }
        online_scalars[SINK_SCALE_SLOT] = sink_scale;
        online_scalars[ROW_LSE_SLOT] = row_lse;
    }
    __syncthreads();

    write_mma884_online_register_output(
        params,
        out_frags,
        out,
        split_idx,
        q_seq_idx,
        head_idx,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[ROW_LSE_SLOT],
        online_scalars[SINK_SCALE_SLOT],
        is_no_split,
        warp_idx,
        lane,
        warps_per_cta);
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__host__ __device__ __forceinline__ size_t sparse_decode_shared_bytes(const SparseAttnDecodeParams &params) {
    if constexpr (USE_MMA_884_ONLINE) {
        size_t bytes = static_cast<size_t>(ONLINE_K_TILE) * sizeof(float)
             + static_cast<size_t>(MMA_884_ONLINE_SCALAR_COUNT) * sizeof(float)
             + static_cast<size_t>(MMA_884_ONLINE_REDUCE_SCRATCH) * sizeof(float);
        if constexpr (!use_mma884_online_register_accumulator<MODEL_TYPE>()) {
            bytes += static_cast<size_t>(HEAD_DIM_V) * sizeof(float);
        }
        // Double-buffer: 2x kv_tile + 2x token_refs when enabled and using register accumulator
        const int kv_buf_count = (USE_DOUBLE_BUFFER && use_mma884_online_register_accumulator<MODEL_TYPE>()) ? 2 : 1;
        bytes += static_cast<size_t>(kv_buf_count) * static_cast<size_t>(ONLINE_K_TILE) * sizeof(int)
              + sizeof(int)
              + static_cast<size_t>(kv_buf_count) * static_cast<size_t>(ONLINE_K_TILE) * static_cast<size_t>(kv_row_stride(head_dim_qk<MODEL_TYPE>())) * sizeof(cutlass::half_t);
        if constexpr (USE_STAGE_Q_HALF) {
            bytes += static_cast<size_t>(head_dim_qk<MODEL_TYPE>()) * sizeof(half);
        }
        if constexpr (!use_mma884_online_register_accumulator<MODEL_TYPE>() && USE_PV_P_CACHE) {
            bytes += static_cast<size_t>(ONLINE_K_TILE) * sizeof(half);  // ONLINE_K_TILE * sizeof(half)
        }
        return bytes;
    }
    return static_cast<size_t>(logical_score_capacity_for_launch(params)) * (sizeof(float) + sizeof(int))
         + static_cast<size_t>(NUM_THREADS) * sizeof(float)
         + static_cast<size_t>(SPARSE_K_TILE) * static_cast<size_t>(kv_row_stride(head_dim_qk<MODEL_TYPE>())) * sizeof(cutlass::half_t);
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__host__ __device__ __forceinline__ size_t sparse_decode_batch2_verify_shared_bytes(
    const SparseAttnDecodeParams&) {
    return static_cast<size_t>(ONLINE_K_TILE) * 2 * sizeof(float)
         + static_cast<size_t>(MMA_884_BATCH2_VERIFY_SCALAR_COUNT) * sizeof(float)
         + static_cast<size_t>(MMA_884_ONLINE_REDUCE_SCRATCH) * sizeof(float)
         + static_cast<size_t>(ONLINE_K_TILE) * sizeof(int)
         + static_cast<size_t>(ONLINE_K_TILE) * static_cast<size_t>(kv_row_stride(head_dim_qk<MODEL_TYPE>())) * sizeof(cutlass::half_t)
         + static_cast<size_t>(head_dim_qk<MODEL_TYPE>()) * 2 * sizeof(half)
         + static_cast<size_t>(ONLINE_K_TILE) * 2 * sizeof(half)
         + sizeof(int);
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__device__ void process_sparse_partition(
    const SparseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int head_idx,
    int start_block_idx,
    int end_block_idx,
    bool is_no_split,
    int split_idx,
    float *shared) {
    const int tid = threadIdx.x;
    const cutlass::bfloat16_t* q = q_ptr_for(params, batch_idx, q_seq_idx, head_idx);
    const int* indices = params.indices
        + batch_idx * params.stride_indices_b
        + q_seq_idx * params.stride_indices_s_q;
    const int* extra_indices = params.extra_indices == nullptr
        ? nullptr
        : params.extra_indices + batch_idx * params.stride_extra_indices_b
                         + q_seq_idx * params.stride_extra_indices_s_q;
    const int valid_topk = valid_topk_for_batch(params.topk_length, batch_idx, params.topk);
    const int valid_extra_topk = params.extra_indices == nullptr
        ? 0
        : valid_topk_for_batch(params.extra_topk_length, batch_idx, params.extra_topk);
    const int main_logical_span = main_logical_span_for_batch(params, valid_topk);
    const int logical_capacity = logical_score_capacity_for_batch(params, valid_topk);
    const int logical_start = min(start_block_idx * TOPK_BLOCK_SIZE, logical_capacity);
    const int logical_end = min(end_block_idx * TOPK_BLOCK_SIZE, logical_capacity);
    const int local_score_count = max(logical_end - logical_start, 0);

    cutlass::bfloat16_t* out = is_no_split
        ? out_ptr_for(params, batch_idx, q_seq_idx, head_idx)
        : nullptr;

    if constexpr (USE_MMA_884_ONLINE) {
        float* tile_scores = shared;
        float* online_scalars = tile_scores + ONLINE_K_TILE;
        float* online_reduce_scratch = online_scalars + MMA_884_ONLINE_SCALAR_COUNT;
        if constexpr (use_mma884_online_register_accumulator<MODEL_TYPE>()) {
            if constexpr (USE_DOUBLE_BUFFER) {
                // Double-buffer layout: two kv_tile buffers and two token_refs arrays
                int* online_token_refs_0 = reinterpret_cast<int*>(
                    online_reduce_scratch + MMA_884_ONLINE_REDUCE_SCRATCH);
                int* online_token_refs_1 = online_token_refs_0 + ONLINE_K_TILE;
                // The online prefix has a 4-byte skew due to the odd scalar
                // count, so one int padding makes kv_tile 8-byte aligned for
                // vector shared loads.
                cutlass::half_t* online_kv_tile_0 = reinterpret_cast<cutlass::half_t*>(
                    online_token_refs_1 + ONLINE_K_TILE + 1);
                cutlass::half_t* online_kv_tile_1 = online_kv_tile_0
                    + ONLINE_K_TILE * kv_row_stride(head_dim_qk<MODEL_TYPE>());
                half* q_smem = reinterpret_cast<half*>(
                    online_kv_tile_1 + ONLINE_K_TILE * kv_row_stride(head_dim_qk<MODEL_TYPE>()));
                accumulate_output_mma884_online_register_double_buffer<MODEL_TYPE, ONLINE_K_TILE>(
                    params,
                    q,
                    indices,
                    extra_indices,
                    logical_start,
                    local_score_count,
                    valid_topk,
                    valid_extra_topk,
                    main_logical_span,
                    is_no_split,
                    batch_idx,
                    split_idx,
                    q_seq_idx,
                    head_idx,
                    out,
                    tile_scores,
                    online_scalars,
                    online_reduce_scratch,
                    online_kv_tile_0,
                    online_kv_tile_1,
                    online_token_refs_0,
                    online_token_refs_1,
                    q_smem);
            } else {
                int* online_token_refs = reinterpret_cast<int*>(
                    online_reduce_scratch + MMA_884_ONLINE_REDUCE_SCRATCH);
                // One int padding restores 8-byte alignment for the QK
                // vector load while preserving the shared address-space.
                cutlass::half_t* online_kv_tile = reinterpret_cast<cutlass::half_t*>(
                    online_token_refs + ONLINE_K_TILE + 1);
                half* q_smem = reinterpret_cast<half*>(
                    online_kv_tile + ONLINE_K_TILE * kv_row_stride(head_dim_qk<MODEL_TYPE>()));
                accumulate_output_mma884_online_register<MODEL_TYPE, ONLINE_K_TILE>(
                    params,
                    q,
                    indices,
                    extra_indices,
                    logical_start,
                    local_score_count,
                    valid_topk,
                    valid_extra_topk,
                    main_logical_span,
                    is_no_split,
                    batch_idx,
                    split_idx,
                    q_seq_idx,
                    head_idx,
                    out,
                    tile_scores,
                    online_scalars,
                    online_reduce_scratch,
                    online_kv_tile,
                    online_token_refs,
                    q_smem);
            }
            return;
        }

        float* output_accum = online_reduce_scratch + MMA_884_ONLINE_REDUCE_SCRATCH;
        int* online_token_refs = reinterpret_cast<int*>(output_accum + HEAD_DIM_V);
        cutlass::half_t* online_kv_tile = reinterpret_cast<cutlass::half_t*>(
            online_token_refs + ONLINE_K_TILE + 1);
        half* q_smem = reinterpret_cast<half*>(
            online_kv_tile + ONLINE_K_TILE * kv_row_stride(head_dim_qk<MODEL_TYPE>()));
        half* p_cache = q_smem + (USE_STAGE_Q_HALF ? head_dim_qk<MODEL_TYPE>() : 0);
        accumulate_output_mma884_online<MODEL_TYPE, ONLINE_K_TILE>(
            params,
            q,
            indices,
            extra_indices,
            logical_start,
            local_score_count,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            is_no_split,
            batch_idx,
            split_idx,
            q_seq_idx,
            head_idx,
            out,
            tile_scores,
            online_scalars,
            online_reduce_scratch,
            output_accum,
            online_kv_tile,
            online_token_refs,
            q_smem,
            p_cache);
        return;
    }

    float* scores = shared;
    const int score_capacity = logical_score_capacity_for_launch(params);
    int* token_refs = reinterpret_cast<int*>(scores + score_capacity);
    float* reduce = reinterpret_cast<float*>(token_refs + score_capacity);
    cutlass::half_t* kv_tile = reinterpret_cast<cutlass::half_t*>(reduce + NUM_THREADS);

    const float local_max = [&]() {
        if constexpr (USE_MMA_884_QK) {
            return fill_scores_mma884_qk<MODEL_TYPE>(
                params,
                q,
                indices,
                extra_indices,
                logical_start,
                local_score_count,
                valid_topk,
                valid_extra_topk,
                main_logical_span,
                scores,
                token_refs,
                kv_tile);
        } else {
            return fill_scores_warp_parallel<MODEL_TYPE>(
                params,
                q,
                indices,
                extra_indices,
                logical_start,
                local_score_count,
                valid_topk,
                valid_extra_topk,
                main_logical_span,
                scores,
                token_refs,
                kv_tile);
        }
    }();
    const float max_score = block_reduce_max(local_max, reduce);

    float local_sum = 0.0F;
#pragma unroll 1
    for (int local_idx = tid; local_idx < local_score_count; local_idx += blockDim.x) {
        const float score = scores[local_idx];
        if (score != -CUDART_INF_F) {
            local_sum += expf(score - max_score);
        }
    }
    const float denom = block_reduce_sum(local_sum, reduce);

    const bool has_tokens = denom > 0.0F;
    const float row_lse = has_tokens ? logf(denom) + max_score : CUDART_INF_F;
    if (tid == 0 && is_no_split) {
        *lse_ptr_for(params, batch_idx, q_seq_idx, head_idx) = row_lse;
    }

    float sink_scale = 1.0F;
    if (is_no_split && params.attn_sink != nullptr && has_tokens) {
        const float sink = __ldg(params.attn_sink + head_idx);
        sink_scale = 1.0F / (1.0F + expf(sink - row_lse));
    }

    if constexpr (USE_MMA_884_PV) {
        accumulate_output_mma884_pv<MODEL_TYPE>(
            params,
            indices,
            extra_indices,
            logical_start,
            local_score_count,
            valid_topk,
            valid_extra_topk,
            main_logical_span,
            scores,
            token_refs,
            kv_tile,
            is_no_split,
            split_idx,
            q_seq_idx,
            head_idx,
            out,
            row_lse,
            sink_scale,
            has_tokens);
    } else {

        float out_acc[MAX_DV_PER_THREAD];
        int out_dim[MAX_DV_PER_THREAD];
#pragma unroll 1
        for (int i = 0; i < MAX_DV_PER_THREAD; ++i) {
            out_dim[i] = tid + i * blockDim.x;
            out_acc[i] = 0.0F;
        }

        if (has_tokens) {
#pragma unroll 1
            for (int tile_start = 0; tile_start < local_score_count; tile_start += SPARSE_K_TILE) {
                const int tile_count = min(SPARSE_K_TILE, local_score_count - tile_start);
                stage_kv_tile_to_shared<MODEL_TYPE>(
                    params,
                    indices,
                    extra_indices,
                    logical_start + tile_start,
                    tile_count,
                    valid_topk,
                    valid_extra_topk,
                    main_logical_span,
                    kv_tile,
                    token_refs + tile_start);

#pragma unroll 1
                for (int i = 0; i < MAX_DV_PER_THREAD; ++i) {
                    const int dim = out_dim[i];
                    if (dim < params.d_v) {
                        float acc = out_acc[i];
#pragma unroll 1
                        for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                            const int local_idx = tile_start + tile_idx;
                            const float score = scores[local_idx];
                            if (score != -CUDART_INF_F) {
                                const float prob = expf(score - row_lse);
                                acc += prob * shared_k_value<MODEL_TYPE>(kv_tile, tile_idx, dim);
                            }
                        }
                        out_acc[i] = acc;
                    }
                }
                __syncthreads();
            }
        }

#pragma unroll 1
        for (int i = 0; i < MAX_DV_PER_THREAD; ++i) {
            const int dim = out_dim[i];
            if (dim < params.d_v) {
                const float raw_acc = out_acc[i];
                if (is_no_split) {
                    write_no_split_accumulators(
                        params,
                        split_idx,
                        q_seq_idx,
                        head_idx,
                        dim,
                        raw_acc,
                        row_lse,
                        has_tokens);
                    out[dim] = cutlass::bfloat16_t(raw_acc * sink_scale);
                } else {
                    write_split_accumulators(
                        params,
                        split_idx,
                        q_seq_idx,
                        head_idx,
                        dim,
                        raw_acc,
                        row_lse,
                        has_tokens);
                }
            }
        }
    }
}

template<ModelType MODEL_TYPE, int ONLINE_K_TILE>
__device__ void process_sparse_partition_batch2_verify(
    const SparseAttnDecodeParams &params,
    int q_seq_idx,
    int head_idx,
    int partition_idx,
    const DecodingSchedMeta &sched_meta,
    float *shared) {
    constexpr int BATCH0 = 0;
    constexpr int BATCH1 = 1;
    if (sched_meta.begin_req_idx > BATCH1 || sched_meta.end_req_idx > BATCH1) {
        return;
    }

    if (sched_meta.begin_req_idx != sched_meta.end_req_idx) {
#pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
            const int valid_topk = valid_topk_for_batch(params.topk_length, batch_idx, params.topk);
            const int logical_capacity = logical_score_capacity_for_batch(params, valid_topk);
            const int full_end_block_idx = ceil_to_multiple(logical_capacity, TOPK_BLOCK_SIZE) / TOPK_BLOCK_SIZE;
            const int end_block_idx = batch_idx == sched_meta.end_req_idx
                ? sched_meta.end_block_idx
                : full_end_block_idx;
            const bool is_no_split = batch_idx == sched_meta.begin_req_idx
                ? !sched_meta.is_first_req_splitted
                : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
            const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
            process_sparse_partition<MODEL_TYPE, ONLINE_K_TILE>(
                params,
                batch_idx,
                q_seq_idx,
                head_idx,
                start_block_idx,
                end_block_idx,
                is_no_split,
                split_idx,
                shared);
            __syncthreads();
        }
        return;
    }

    const int batch_idx = sched_meta.begin_req_idx;
    const int n_split_idx = sched_meta.begin_split_idx;
    const int start_block_idx = sched_meta.begin_block_idx;
    const int end_block_idx = sched_meta.end_block_idx;
    const bool is_no_split = !sched_meta.is_first_req_splitted;
    const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
    const int splits0 = __ldg(params.num_splits_ptr + BATCH1) - __ldg(params.num_splits_ptr + BATCH0);
    const int splits1 = __ldg(params.num_splits_ptr + BATCH1 + 1) - __ldg(params.num_splits_ptr + BATCH1);
    bool can_pair_split = splits0 > 1 && splits1 > 1 && n_split_idx < splits0 && n_split_idx < splits1;
    if (can_pair_split) {
        const int peer_partition_idx = batch_idx == BATCH0
            ? partition_idx + splits0
            : partition_idx - splits0;
        can_pair_split = peer_partition_idx >= 0 && peer_partition_idx < params.num_sm_parts;
        if (can_pair_split) {
            DecodingSchedMeta peer_meta = params.tile_scheduler_metadata_ptr[peer_partition_idx];
            const int peer_batch_idx = batch_idx == BATCH0 ? BATCH1 : BATCH0;
            can_pair_split = peer_meta.begin_req_idx == peer_batch_idx &&
                peer_meta.end_req_idx == peer_batch_idx &&
                peer_meta.begin_split_idx == n_split_idx &&
                peer_meta.begin_block_idx == start_block_idx &&
                peer_meta.end_block_idx == end_block_idx &&
                peer_meta.is_first_req_splitted == sched_meta.is_first_req_splitted &&
                peer_meta.is_last_req_splitted == sched_meta.is_last_req_splitted;
        }
    }
    if (!can_pair_split) {
        process_sparse_partition<MODEL_TYPE, ONLINE_K_TILE>(
            params,
            batch_idx,
            q_seq_idx,
            head_idx,
            start_block_idx,
            end_block_idx,
            is_no_split,
            split_idx,
            shared);
        return;
    }

    const int split_idx0 = __ldg(params.num_splits_ptr + BATCH0) + n_split_idx;
    const int split_idx1 = __ldg(params.num_splits_ptr + BATCH1) + n_split_idx;
    const int valid_topk0 = valid_topk_for_batch(params.topk_length, BATCH0, params.topk);
    const int valid_topk1 = valid_topk_for_batch(params.topk_length, BATCH1, params.topk);
    const int valid_extra_topk0 = params.extra_indices == nullptr
        ? 0
        : valid_topk_for_batch(params.extra_topk_length, BATCH0, params.extra_topk);
    const int valid_extra_topk1 = params.extra_indices == nullptr
        ? 0
        : valid_topk_for_batch(params.extra_topk_length, BATCH1, params.extra_topk);
    const int main_logical_span0 = main_logical_span_for_batch(params, valid_topk0);
    const int main_logical_span1 = main_logical_span_for_batch(params, valid_topk1);
    const int logical_capacity0 = logical_score_capacity_for_batch(params, valid_topk0);
    const int logical_start0 = min(start_block_idx * TOPK_BLOCK_SIZE, logical_capacity0);
    const int logical_end0 = min(end_block_idx * TOPK_BLOCK_SIZE, logical_capacity0);
    const int local_score_count0 = max(logical_end0 - logical_start0, 0);

    float* tile_scores0 = shared;
    float* tile_scores1 = tile_scores0 + ONLINE_K_TILE;
    float* online_scalars = tile_scores1 + ONLINE_K_TILE;
    float* online_reduce_scratch = online_scalars + MMA_884_BATCH2_VERIFY_SCALAR_COUNT;
    int* token_refs = reinterpret_cast<int*>(
        online_reduce_scratch + MMA_884_ONLINE_REDUCE_SCRATCH);
    cutlass::half_t* kv_tile = reinterpret_cast<cutlass::half_t*>(
        token_refs + ONLINE_K_TILE);
    half* q0_smem = reinterpret_cast<half*>(
        kv_tile + ONLINE_K_TILE * kv_row_stride(head_dim_qk<MODEL_TYPE>()));
    half* q1_smem = q0_smem + head_dim_qk<MODEL_TYPE>();
    half* p_cache0 = q1_smem + head_dim_qk<MODEL_TYPE>();
    half* p_cache1 = p_cache0 + ONLINE_K_TILE;
    int* match_flag = reinterpret_cast<int*>(p_cache1 + ONLINE_K_TILE);

    const bool rows_match = topk_rows_match(
        params,
        q_seq_idx,
        logical_start0,
        local_score_count0,
        valid_topk0,
        valid_topk1,
        valid_extra_topk0,
        valid_extra_topk1,
        main_logical_span0,
        main_logical_span1,
        match_flag);
    if (!rows_match) {
        process_sparse_partition<MODEL_TYPE, ONLINE_K_TILE>(
            params,
            batch_idx,
            q_seq_idx,
            head_idx,
            start_block_idx,
            end_block_idx,
            is_no_split,
            split_idx,
            shared);
        return;
    }
    if (batch_idx == BATCH1) {
        return;
    }

    const int* indices0 = params.indices + q_seq_idx * params.stride_indices_s_q;
    const int* extra_indices0 = params.extra_indices == nullptr
        ? nullptr
        : params.extra_indices + q_seq_idx * params.stride_extra_indices_s_q;
    const cutlass::bfloat16_t* q0 = q_ptr_for(params, BATCH0, q_seq_idx, head_idx);
    const cutlass::bfloat16_t* q1 = q_ptr_for(params, BATCH1, q_seq_idx, head_idx);

    accumulate_output_mma884_online_register_batch2_verify<MODEL_TYPE, ONLINE_K_TILE>(
        params,
        q0,
        q1,
        indices0,
        extra_indices0,
        logical_start0,
        local_score_count0,
        valid_topk0,
        valid_extra_topk0,
        main_logical_span0,
        split_idx0,
        split_idx1,
        q_seq_idx,
        head_idx,
        tile_scores0,
        tile_scores1,
        online_scalars,
        online_reduce_scratch,
        kv_tile,
        token_refs,
        q0_smem,
        q1_smem,
        p_cache0,
        p_cache1);
}

template<ModelType MODEL_TYPE, int NUM_HEADS, int ONLINE_K_TILE>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_sm70_sparse_kernel(const SparseAttnDecodeParams params) {
    static_assert(MODEL_TYPE == ModelType::V32 || MODEL_TYPE == ModelType::MODEL1);
    static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);

    const int q_seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ float shared[];

    const DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) {
        return;
    }

#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        const int valid_topk = valid_topk_for_batch(params.topk_length, batch_idx, params.topk);
        const int logical_capacity = logical_score_capacity_for_batch(params, valid_topk);
        const int full_end_block_idx = ceil_to_multiple(logical_capacity, TOPK_BLOCK_SIZE) / TOPK_BLOCK_SIZE;
        const int end_block_idx = batch_idx == sched_meta.end_req_idx
            ? sched_meta.end_block_idx
            : full_end_block_idx;
        const bool is_no_split = batch_idx == sched_meta.begin_req_idx
            ? !sched_meta.is_first_req_splitted
            : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
        const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;

        process_sparse_partition<MODEL_TYPE, ONLINE_K_TILE>(
            params,
            batch_idx,
            q_seq_idx,
            head_idx,
            start_block_idx,
            end_block_idx,
            is_no_split,
            split_idx,
            shared);

        __syncthreads();
    }
}

template<ModelType MODEL_TYPE, int NUM_HEADS, int ONLINE_K_TILE>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_splitkv_mla_sm70_sparse_batch2_verify_kernel(const SparseAttnDecodeParams params) {
    static_assert(MODEL_TYPE == ModelType::MODEL1);
    static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);

    const int q_seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ float shared[];

    const DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) {
        return;
    }

    process_sparse_partition_batch2_verify<MODEL_TYPE, ONLINE_K_TILE>(
        params,
        q_seq_idx,
        head_idx,
        partition_idx,
        sched_meta,
        shared);
}

}  // namespace detail

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    static_assert(MODEL_TYPE == ModelType::V32 || MODEL_TYPE == ModelType::MODEL1);
    static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);
    FLASH_ASSERT(params.h_q == NUM_HEADS);
    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.d_qk == detail::head_dim_qk<MODEL_TYPE>());
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
    FLASH_ASSERT(params.page_block_size > 0);
    if constexpr (MODEL_TYPE == ModelType::V32) {
        FLASH_ASSERT(params.stride_kv_row == V32_BYTES_PER_TOKEN);
        FLASH_ASSERT(params.extra_kv == nullptr || params.stride_extra_kv_row == V32_BYTES_PER_TOKEN);
    } else {
        FLASH_ASSERT(params.stride_kv_row == MODEL1_BYTES_PER_TOKEN);
        FLASH_ASSERT(params.extra_kv == nullptr || params.stride_extra_kv_row == MODEL1_BYTES_PER_TOKEN);
        FLASH_ASSERT(params.stride_kv_block % MODEL1_TOKEN_DATA_BYTES == 0);
        FLASH_ASSERT(params.extra_kv == nullptr || params.stride_extra_kv_block % MODEL1_TOKEN_DATA_BYTES == 0);
    }
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.topk + params.extra_topk <= MAX_TOPK_ALPHA);

    if constexpr (
        USE_MMA_884_ONLINE && SPARSE_K_TILE > MMA_884_ONLINE_LONG_K_TILE &&
        MODEL_TYPE == ModelType::V32) {
        if (params.topk + params.extra_topk >= MMA_884_ONLINE_LONG_K_TILE_THRESHOLD) {
            auto kernel =
                &detail::flash_fwd_splitkv_mla_sm70_sparse_kernel<MODEL_TYPE, NUM_HEADS, MMA_884_ONLINE_LONG_K_TILE>;
            const size_t smem_size = detail::sparse_decode_shared_bytes<MODEL_TYPE, MMA_884_ONLINE_LONG_K_TILE>(params);
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            kernel<<<dim3(params.s_q, params.h_q, params.num_sm_parts), dim3(NUM_THREADS), smem_size, params.stream>>>(params);
            CHECK_CUDA_KERNEL_LAUNCH();
            return;
        }
    }

    auto kernel = &detail::flash_fwd_splitkv_mla_sm70_sparse_kernel<MODEL_TYPE, NUM_HEADS, SPARSE_K_TILE>;
    const size_t smem_size = detail::sparse_decode_shared_bytes<MODEL_TYPE, SPARSE_K_TILE>(params);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.s_q, params.h_q, params.num_sm_parts), dim3(NUM_THREADS), smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_batch2_verify_kernel(const SparseAttnDecodeParams &params) {
    static_assert(MODEL_TYPE == ModelType::MODEL1);
    static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);
    FLASH_ASSERT(USE_BATCH2_VERIFY);
    FLASH_ASSERT(params.h_q == NUM_HEADS);
    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.d_qk == HEAD_DIM_QK_MODEL1);
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
    FLASH_ASSERT(params.b == 2);
    FLASH_ASSERT(params.s_q == 1);
    FLASH_ASSERT(params.page_block_size > 0);
    FLASH_ASSERT(params.stride_kv_row == MODEL1_BYTES_PER_TOKEN);
    FLASH_ASSERT(params.extra_kv == nullptr || params.stride_extra_kv_row == MODEL1_BYTES_PER_TOKEN);
    FLASH_ASSERT(params.stride_kv_block % MODEL1_TOKEN_DATA_BYTES == 0);
    FLASH_ASSERT(params.extra_kv == nullptr || params.stride_extra_kv_block % MODEL1_TOKEN_DATA_BYTES == 0);
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.topk + params.extra_topk <= MAX_TOPK_ALPHA);

    auto kernel =
        &detail::flash_fwd_splitkv_mla_sm70_sparse_batch2_verify_kernel<MODEL_TYPE, NUM_HEADS, SPARSE_K_TILE>;
    const size_t smem_size = detail::sparse_decode_batch2_verify_shared_bytes<MODEL_TYPE, SPARSE_K_TILE>(params);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.s_q, params.h_q, params.num_sm_parts), dim3(NUM_THREADS), smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sm70::decode::sparse_fp8
