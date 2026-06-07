#pragma once

#include <cuda_fp16.h>
#include <math_constants.h>

#include <cutlass/half.h>

#include "sm70/common/mma_884_attention.h"
#include "sm70/common/softmax.h"

#include "params.h"
#include "utils.h"
#include "config.h"

namespace sm70 {

namespace detail {

template<typename T>
__device__ __forceinline__ float element_to_float(const T &value) {
    return static_cast<float>(value);
}

template<typename InputT>
__device__ __forceinline__ const InputT* q_ptr_for(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx) {
    return reinterpret_cast<const InputT*>(params.q_ptr)
         + batch_idx * params.q_batch_stride
         + q_seq_idx * params.q_row_stride
         + kv_head_idx * params.q_head_stride;
}

template<typename InputT>
__device__ __forceinline__ const InputT* k_ptr_for(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int token_idx,
    int kv_head_idx) {
    const int block_table_idx = token_idx / params.page_block_size;
    const int token_in_block = token_idx - block_table_idx * params.page_block_size;
    const int physical_block = __ldg(
        params.block_table + batch_idx * params.block_table_batch_stride + block_table_idx);
    return reinterpret_cast<const InputT*>(params.k_ptr)
         + physical_block * params.k_batch_stride
         + token_in_block * params.k_row_stride
         + kv_head_idx * params.k_head_stride;
}

template<typename InputT>
__device__ __forceinline__ InputT* out_ptr_for(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx) {
    return reinterpret_cast<InputT*>(params.o_ptr)
         + batch_idx * params.o_batch_stride
         + kv_head_idx * params.o_head_stride
         + q_seq_idx * params.o_row_stride;
}

__device__ __forceinline__ float* lse_ptr_for(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx) {
    return params.softmax_lse_ptr
         + (batch_idx * params.h_k + kv_head_idx) * params.q_seq_per_hk
         + q_seq_idx;
}

__device__ __forceinline__ float* oaccum_ptr_for(
    const DenseAttnDecodeParams &params,
    int split_idx,
    int q_seq_idx,
    int kv_head_idx) {
    return params.oaccum_ptr
         + ((split_idx * params.h_k + kv_head_idx) * params.q_seq_per_hk + q_seq_idx) * params.d_v;
}

__device__ __forceinline__ float* lseaccum_ptr_for(
    const DenseAttnDecodeParams &params,
    int split_idx,
    int q_seq_idx,
    int kv_head_idx) {
    return params.softmax_lseaccum_ptr
         + (split_idx * params.h_k + kv_head_idx) * params.q_seq_per_hk
         + q_seq_idx;
}

template<typename InputT>
__device__ __forceinline__ float qk_score(
    const DenseAttnDecodeParams &params,
    const InputT *q,
    int batch_idx,
    int token_idx,
    int kv_head_idx) {
    const InputT *k = k_ptr_for<InputT>(params, batch_idx, token_idx, kv_head_idx);
    float acc = 0.0F;
#pragma unroll 1
    for (int d = 0; d < params.d; ++d) {
        acc += element_to_float(q[d]) * element_to_float(k[d]);
    }
    return acc * params.scale_softmax;
}

__device__ __forceinline__ half half_from_float(float value) {
    return __float2half_rn(value);
}

__device__ __forceinline__ float shared_dense_kv_value(
    const cutlass::half_t* kv_tile,
    int local_token_idx,
    int dim,
    int head_dim) {
    return static_cast<float>(kv_tile[local_token_idx * head_dim + dim]);
}

template<typename InputT>
__device__ void stage_dense_kv_tile_to_shared(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int tile_start,
    int tile_count,
    int kv_head_idx,
    cutlass::half_t* kv_tile) {
#pragma unroll 1
    for (int linear = threadIdx.x; linear < tile_count * params.d; linear += blockDim.x) {
        const int local_token_idx = linear / params.d;
        const int dim = linear - local_token_idx * params.d;
        const InputT* k = k_ptr_for<InputT>(params, batch_idx, tile_start + local_token_idx, kv_head_idx);
        kv_tile[linear] = cutlass::half_t(element_to_float(k[dim]));
    }
    __syncthreads();
}

template<typename InputT>
__device__ __forceinline__ void load_dense_mma884_q_fragment(
    const DenseAttnDecodeParams &params,
    const InputT* q,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_q) {
    const int q_row = flash_mla::sm70::attention::mma884_q_fragment_row(lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = dim_base + i;
        frag_q[i] = q_row == 0 && dim < params.d
            ? half_from_float(element_to_float(q[dim]))
            : half_from_float(0.0F);
    }
}

__device__ __forceinline__ void load_dense_mma884_k_fragment(
    const DenseAttnDecodeParams &params,
    const cutlass::half_t* kv_tile,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_k) {
    const int token_offset = flash_mla::sm70::attention::mma884_qk_token_offset(lane);
    const int local_token_idx = token_base + token_offset;
    const bool valid_token = local_token_idx < tile_count;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = dim_base + i;
        frag_k[i] = valid_token && dim < params.d
            ? half_from_float(shared_dense_kv_value(kv_tile, local_token_idx, dim, params.d))
            : half_from_float(0.0F);
    }
}

__device__ __forceinline__ void store_dense_mma884_qk_row0_scores(
    const flash_mla::sm70::Array<float, 8> &score_frag,
    float* tile_scores,
    int token_base,
    int tile_count,
    int lane,
    float scale_softmax) {
    flash_mla::sm70::attention::store_mma884_row0_scores<false>(
        score_frag, tile_scores, nullptr, token_base, tile_count, lane, scale_softmax);
}

template<typename InputT>
__device__ __forceinline__ void compute_dense_mma884_qk_group(
    const DenseAttnDecodeParams &params,
    const InputT* q,
    const cutlass::half_t* kv_tile,
    float* tile_scores,
    int token_base,
    int tile_count,
    int lane) {
    flash_mla::sm70::Array<float, 8> score_frag{};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_frag[i] = 0.0F;
    }

#pragma unroll 1
    for (int dim_base = 0; dim_base < params.d; dim_base += 4) {
        flash_mla::sm70::Array<half, 4> frag_q{};
        flash_mla::sm70::Array<half, 4> frag_k{};
        load_dense_mma884_q_fragment(params, q, dim_base, lane, frag_q);
        load_dense_mma884_k_fragment(params, kv_tile, token_base, tile_count, dim_base, lane, frag_k);
        flash_mla::sm70::attention::mma884_accumulate_qk(score_frag, frag_q, frag_k);
    }
    store_dense_mma884_qk_row0_scores(
        score_frag, tile_scores, token_base, tile_count, lane, params.scale_softmax);
}

template<typename InputT>
__device__ __forceinline__ void fill_dense_tile_scores_mma884_qk(
    const DenseAttnDecodeParams &params,
    const InputT* q,
    const cutlass::half_t* kv_tile,
    int tile_count,
    float* tile_scores) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;
    for (int token_base = warp_idx * 16; token_base < tile_count; token_base += warps_per_cta * 16) {
        compute_dense_mma884_qk_group(params, q, kv_tile, tile_scores, token_base, tile_count, lane);
    }
}

__device__ __forceinline__ half dense_probability_half_from_online_score(
    float score,
    float row_max) {
    return half_from_float(expf(score - row_max));
}

__device__ __forceinline__ void load_dense_mma884_online_p_fragment(
    const float* tile_scores,
    int token_base,
    int tile_count,
    int lane,
    float row_max,
    flash_mla::sm70::Array<half, 4> &frag_p) {
    flash_mla::sm70::attention::load_mma884_p_fragment_from_scores<false>(
        tile_scores, 0, token_base, tile_count, lane, row_max, frag_p);
}

__device__ __forceinline__ void load_dense_mma884_v_fragment(
    const DenseAttnDecodeParams &params,
    const cutlass::half_t* kv_tile,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    flash_mla::sm70::Array<half, 4> &frag_v) {
    const int token_offset = flash_mla::sm70::attention::mma884_pv_token_offset(lane);
    const int local_token_idx = token_base + token_offset;
    const int dim_offset = flash_mla::sm70::attention::mma884_pv_dim_offset(lane);
    const bool valid_token = local_token_idx < tile_count;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = dim_base + dim_offset + i;
        frag_v[i] = valid_token && dim < params.d_v
            ? half_from_float(shared_dense_kv_value(kv_tile, local_token_idx, dim, params.d))
            : half_from_float(0.0F);
    }
}

__device__ __forceinline__ void compute_dense_mma884_online_pv_group(
    const DenseAttnDecodeParams &params,
    const float* tile_scores,
    const cutlass::half_t* kv_tile,
    int token_base,
    int tile_count,
    int dim_base,
    int lane,
    float row_max,
    flash_mla::sm70::Array<float, 8> &out_frag) {
    flash_mla::sm70::Array<half, 4> frag_p{};
    flash_mla::sm70::Array<half, 4> frag_v{};
    load_dense_mma884_online_p_fragment(tile_scores, token_base, tile_count, lane, row_max, frag_p);
    load_dense_mma884_v_fragment(params, kv_tile, token_base, tile_count, dim_base, lane, frag_v);
    flash_mla::sm70::attention::mma884_accumulate_pv(out_frag, frag_p, frag_v);
}

__device__ __forceinline__ void add_dense_mma884_pv_row0_output_to_accum(
    const flash_mla::sm70::Array<float, 8> &out_frag,
    float* output_accum,
    int dim_base,
    int lane) {
    flash_mla::sm70::attention::add_mma884_row0_output_fragment(
        out_frag, output_accum, dim_base, lane, dense::HEAD_DIM_V);
}

template<int D_GROUPS_PER_WARP>
__device__ __forceinline__ void scale_dense_mma884_register_fragments(
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

template<typename InputT, int D_GROUPS_PER_WARP>
__device__ __forceinline__ void write_dense_mma884_register_output(
    const DenseAttnDecodeParams &params,
    flash_mla::sm70::Array<float, 8> (&out_frags)[D_GROUPS_PER_WARP],
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    bool is_no_split,
    int split_idx,
    float row_sum,
    float row_lse,
    int warp_idx,
    int lane,
    int warps_per_cta) {
    if (threadIdx.x == 0) {
        if (is_no_split) {
            *lse_ptr_for(params, batch_idx, q_seq_idx, kv_head_idx) = row_lse;
        } else {
            *lseaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx) = row_lse * CUDART_L2E_F;
        }
    }
    const float inv_row_sum = 1.0F / row_sum;
    InputT* out = is_no_split ? out_ptr_for<InputT>(params, batch_idx, q_seq_idx, kv_head_idx) : nullptr;
    float* out_accum = is_no_split ? nullptr : oaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx);
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
                            const float value = out_frags[group_idx][d1 * 4 + q * 2 + d0] * inv_row_sum;
                            if (is_no_split) {
                                out[dim] = InputT(value);
                            } else {
                                out_accum[dim] = value;
                            }
                        }
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void zero_dense_online_output_accumulator(float* output_accum) {
    for (int dim = threadIdx.x; dim < dense::HEAD_DIM_V; dim += blockDim.x) {
        output_accum[dim] = 0.0F;
    }
}

__device__ __forceinline__ void scale_dense_online_output_accumulator(float* output_accum, float scale) {
    for (int dim = threadIdx.x; dim < dense::HEAD_DIM_V; dim += blockDim.x) {
        output_accum[dim] *= scale;
    }
}

template<typename InputT>
__device__ __forceinline__ void write_dense_online_output(
    const DenseAttnDecodeParams &params,
    float* output_accum,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    bool is_no_split,
    int split_idx,
    float row_sum,
    float row_lse) {
    if (threadIdx.x == 0) {
        if (is_no_split) {
            *lse_ptr_for(params, batch_idx, q_seq_idx, kv_head_idx) = row_lse;
        } else {
            *lseaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx) = row_lse * CUDART_L2E_F;
        }
    }
    const float inv_row_sum = 1.0F / row_sum;
    InputT* out = is_no_split ? out_ptr_for<InputT>(params, batch_idx, q_seq_idx, kv_head_idx) : nullptr;
    float* out_accum = is_no_split ? nullptr : oaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx);
    for (int dim = threadIdx.x; dim < params.d_v; dim += blockDim.x) {
        const float value = output_accum[dim] * inv_row_sum;
        if (is_no_split) {
            out[dim] = InputT(value);
        } else {
            out_accum[dim] = value;
        }
    }
}

template<typename InputT, int ONLINE_K_TILE>
__device__ __forceinline__ void accumulate_dense_output_mma884_online(
    const DenseAttnDecodeParams &params,
    const InputT* q,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    int token_start,
    int token_end,
    bool is_no_split,
    int split_idx,
    float* tile_scores,
    float* online_scalars,
    float* online_reduce_scratch,
    float* output_accum,
    cutlass::half_t* kv_tile) {
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int ROW_LSE_SLOT = 3;

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
        online_scalars[ROW_LSE_SLOT] = CUDART_INF_F;
    }
    zero_dense_online_output_accumulator(output_accum);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    const int warps_per_cta = blockDim.x / 32;

#pragma unroll 1
    for (int tile_start = token_start; tile_start < token_end; tile_start += ONLINE_K_TILE) {
        const int tile_count = min(ONLINE_K_TILE, token_end - tile_start);
        stage_dense_kv_tile_to_shared<InputT>(params, batch_idx, tile_start, tile_count, kv_head_idx, kv_tile);
        fill_dense_tile_scores_mma884_qk(params, q, kv_tile, tile_count, tile_scores);
        __syncthreads();

        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<false>(
            tile_scores,
            tile_count,
            online_scalars + ROW_MAX_SLOT,
            online_scalars + ROW_SUM_SLOT,
            online_scalars + ONLINE_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();

        scale_dense_online_output_accumulator(output_accum, online_scalars[ONLINE_SCALE_SLOT]);
        __syncthreads();

        const float row_max = online_scalars[ROW_MAX_SLOT];
#pragma unroll 1
        for (int dim_group_base = 0; dim_group_base < dense::HEAD_DIM_V; dim_group_base += warps_per_cta * 16) {
            const int dim_base = dim_group_base + warp_idx * 16;
            if (dim_base < params.d_v) {
                flash_mla::sm70::Array<float, 8> out_frag{};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out_frag[i] = 0.0F;
                }
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_dense_mma884_online_pv_group(
                        params,
                        tile_scores,
                        kv_tile,
                        token_base,
                        tile_count,
                        dim_base,
                        lane,
                        row_max,
                        out_frag);
                }
                add_dense_mma884_pv_row0_output_to_accum(out_frag, output_accum, dim_base, lane);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        online_scalars[ROW_LSE_SLOT] = logf(online_scalars[ROW_SUM_SLOT]) + online_scalars[ROW_MAX_SLOT];
    }
    __syncthreads();
    write_dense_online_output<InputT>(
        params,
        output_accum,
        batch_idx,
        q_seq_idx,
        kv_head_idx,
        is_no_split,
        split_idx,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[ROW_LSE_SLOT]);
}

template<typename InputT, int ONLINE_K_TILE>
__device__ __forceinline__ void accumulate_dense_output_mma884_online_register(
    const DenseAttnDecodeParams &params,
    const InputT* q,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    int token_start,
    int token_end,
    bool is_no_split,
    int split_idx,
    float* tile_scores,
    float* online_scalars,
    float* online_reduce_scratch,
    cutlass::half_t* kv_tile) {
    static_assert(
        dense::NUM_THREADS == dense::MMA_884_REGISTER_OUTPUT_CTA_THREADS,
        "register output accumulator is specialized for the 256-thread dense online path");
    constexpr int ROW_MAX_SLOT = 0;
    constexpr int ROW_SUM_SLOT = 1;
    constexpr int ONLINE_SCALE_SLOT = 2;
    constexpr int ROW_LSE_SLOT = 3;
    constexpr int WARPS_PER_CTA = dense::MMA_884_REGISTER_OUTPUT_CTA_THREADS / 32;
    constexpr int D_GROUPS_PER_WARP =
        (dense::HEAD_DIM_V + WARPS_PER_CTA * 16 - 1) / (WARPS_PER_CTA * 16);

    if (threadIdx.x == 0) {
        online_scalars[ROW_MAX_SLOT] = -CUDART_INF_F;
        online_scalars[ROW_SUM_SLOT] = 0.0F;
        online_scalars[ONLINE_SCALE_SLOT] = 1.0F;
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
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x / 32;
    constexpr int warps_per_cta = WARPS_PER_CTA;

#pragma unroll 1
    for (int tile_start = token_start; tile_start < token_end; tile_start += ONLINE_K_TILE) {
        const int tile_count = min(ONLINE_K_TILE, token_end - tile_start);
        stage_dense_kv_tile_to_shared<InputT>(params, batch_idx, tile_start, tile_count, kv_head_idx, kv_tile);
        fill_dense_tile_scores_mma884_qk(params, q, kv_tile, tile_count, tile_scores);
        __syncthreads();

        flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<false>(
            tile_scores,
            tile_count,
            online_scalars + ROW_MAX_SLOT,
            online_scalars + ROW_SUM_SLOT,
            online_scalars + ONLINE_SCALE_SLOT,
            online_reduce_scratch);
        __syncthreads();

        scale_dense_mma884_register_fragments(out_frags, online_scalars[ONLINE_SCALE_SLOT]);
        const float row_max = online_scalars[ROW_MAX_SLOT];
#pragma unroll
        for (int group_idx = 0; group_idx < D_GROUPS_PER_WARP; ++group_idx) {
            const int dim_base = (group_idx * warps_per_cta + warp_idx) * 16;
            if (dim_base < params.d_v) {
#pragma unroll 1
                for (int token_base = 0; token_base < tile_count; token_base += 4) {
                    compute_dense_mma884_online_pv_group(
                        params,
                        tile_scores,
                        kv_tile,
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
        online_scalars[ROW_LSE_SLOT] = logf(online_scalars[ROW_SUM_SLOT]) + online_scalars[ROW_MAX_SLOT];
    }
    __syncthreads();
    write_dense_mma884_register_output<InputT>(
        params,
        out_frags,
        batch_idx,
        q_seq_idx,
        kv_head_idx,
        is_no_split,
        split_idx,
        online_scalars[ROW_SUM_SLOT],
        online_scalars[ROW_LSE_SLOT],
        warp_idx,
        lane,
        warps_per_cta);
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

template<typename InputT>
__device__ __forceinline__ void write_zero_output(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    bool is_no_split,
    int split_idx) {
    const int tid = threadIdx.x;
    if (is_no_split) {
        InputT *out = out_ptr_for<InputT>(params, batch_idx, q_seq_idx, kv_head_idx);
        for (int d = tid; d < params.d_v; d += blockDim.x) {
            out[d] = InputT(0.0F);
        }
        if (tid == 0) {
            *lse_ptr_for(params, batch_idx, q_seq_idx, kv_head_idx) = CUDART_INF_F;
        }
    } else {
        float *out = oaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx);
        for (int d = tid; d < params.d_v; d += blockDim.x) {
            out[d] = 0.0F;
        }
        if (tid == 0) {
            *lseaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx) = -CUDART_INF_F;
        }
    }
}

template<typename InputT>
__device__ __forceinline__ void write_output(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    bool is_no_split,
    int split_idx,
    int d_v,
    float value) {
    if (is_no_split) {
        InputT *out = out_ptr_for<InputT>(params, batch_idx, q_seq_idx, kv_head_idx);
        out[d_v] = InputT(value);
    } else {
        float *out = oaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx);
        out[d_v] = value;
    }
}

__device__ __forceinline__ int causal_token_limit(
    const DenseAttnDecodeParams &params,
    int q_seq_idx,
    int seqlen_k) {
    if (!params.is_causal) {
        return seqlen_k;
    }
    const int q_token_idx = q_seq_idx / params.q_head_per_hk;
    return max(min(seqlen_k, seqlen_k - params.s_q + q_token_idx + 1), 0);
}

template<typename InputT, int ONLINE_K_TILE>
__device__ void process_dense_partition(
    const DenseAttnDecodeParams &params,
    int batch_idx,
    int q_seq_idx,
    int kv_head_idx,
    int start_block_idx,
    int end_block_idx,
    bool is_no_split,
    int split_idx,
    float *shared) {
    const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
    const int token_limit = causal_token_limit(params, q_seq_idx, seqlen_k);
    const int token_start = min(start_block_idx * params.page_block_size, token_limit);
    const int token_end = min(end_block_idx * params.page_block_size, token_limit);

    if (token_start >= token_end) {
        write_zero_output<InputT>(params, batch_idx, q_seq_idx, kv_head_idx, is_no_split, split_idx);
        return;
    }

    const InputT *q = q_ptr_for<InputT>(params, batch_idx, q_seq_idx, kv_head_idx);

    if constexpr (dense::USE_MMA_884_ONLINE && dense::NUM_THREADS == dense::MMA_884_REGISTER_OUTPUT_CTA_THREADS) {
        float* tile_scores = shared;
        float* online_scalars = tile_scores + ONLINE_K_TILE;
        float* online_reduce_scratch = online_scalars + dense::MMA_884_ONLINE_SCALAR_COUNT;
        cutlass::half_t* kv_tile = reinterpret_cast<cutlass::half_t*>(
            online_reduce_scratch + dense::MMA_884_ONLINE_REDUCE_SCRATCH);
        accumulate_dense_output_mma884_online_register<InputT, ONLINE_K_TILE>(
            params,
            q,
            batch_idx,
            q_seq_idx,
            kv_head_idx,
            token_start,
            token_end,
            is_no_split,
            split_idx,
            tile_scores,
            online_scalars,
            online_reduce_scratch,
            kv_tile);
        return;
    }

    if constexpr (dense::USE_MMA_884_ONLINE) {
        float* tile_scores = shared;
        float* online_scalars = tile_scores + ONLINE_K_TILE;
        float* online_reduce_scratch = online_scalars + dense::MMA_884_ONLINE_SCALAR_COUNT;
        float* output_accum = online_reduce_scratch + dense::MMA_884_ONLINE_REDUCE_SCRATCH;
        cutlass::half_t* kv_tile = reinterpret_cast<cutlass::half_t*>(output_accum + dense::HEAD_DIM_V);
        accumulate_dense_output_mma884_online<InputT, ONLINE_K_TILE>(
            params,
            q,
            batch_idx,
            q_seq_idx,
            kv_head_idx,
            token_start,
            token_end,
            is_no_split,
            split_idx,
            tile_scores,
            online_scalars,
            online_reduce_scratch,
            output_accum,
            kv_tile);
        return;
    }

    if constexpr (!dense::USE_MMA_884_ONLINE) {
        const int tid = threadIdx.x;
        float *reduce_smem = shared;
        float *score_smem = shared + dense::NUM_THREADS;
        float out_acc[dense::MAX_DV_PER_THREAD];
        int out_dim[dense::MAX_DV_PER_THREAD];
#pragma unroll
        for (int i = 0; i < dense::MAX_DV_PER_THREAD; ++i) {
            out_dim[i] = tid + i * blockDim.x;
            out_acc[i] = 0.0F;
        }

        float row_max = -CUDART_INF_F;
        float row_sum = 0.0F;
#pragma unroll 1
        for (int tile_start = token_start; tile_start < token_end; tile_start += dense::K_TILE) {
            const int tile_count = min(dense::K_TILE, token_end - tile_start);
            float local_tile_max = -CUDART_INF_F;
            if (tid < tile_count) {
                const float score = qk_score<InputT>(params, q, batch_idx, tile_start + tid, kv_head_idx);
                score_smem[tid] = score;
                local_tile_max = score;
            }
            const float tile_max = block_reduce_max(local_tile_max, reduce_smem);
            const float new_row_max = fmaxf(row_max, tile_max);
            const float old_scale = row_max == -CUDART_INF_F ? 0.0F : expf(row_max - new_row_max);

            float local_tile_sum = 0.0F;
            if (tid < tile_count) {
                local_tile_sum = expf(score_smem[tid] - new_row_max);
            }
            const float tile_sum = block_reduce_sum(local_tile_sum, reduce_smem);

#pragma unroll
            for (int i = 0; i < dense::MAX_DV_PER_THREAD; ++i) {
                const int d_v = out_dim[i];
                if (d_v < params.d_v) {
                    float acc = out_acc[i] * old_scale;
#pragma unroll 1
                    for (int local_token_idx = 0; local_token_idx < tile_count; ++local_token_idx) {
                        const float weight = expf(score_smem[local_token_idx] - new_row_max);
                        const InputT *k = k_ptr_for<InputT>(
                            params, batch_idx, tile_start + local_token_idx, kv_head_idx);
                        acc += weight * element_to_float(k[d_v]);
                    }
                    out_acc[i] = acc;
                }
            }

            row_sum = row_sum * old_scale + tile_sum;
            row_max = new_row_max;
            __syncthreads();
        }

        const float row_lse = logf(row_sum) + row_max;
        if (tid == 0) {
            if (is_no_split) {
                *lse_ptr_for(params, batch_idx, q_seq_idx, kv_head_idx) = row_lse;
            } else {
                *lseaccum_ptr_for(params, split_idx, q_seq_idx, kv_head_idx) = row_lse * CUDART_L2E_F;
            }
        }

        const float inv_row_sum = 1.0F / row_sum;
#pragma unroll
        for (int i = 0; i < dense::MAX_DV_PER_THREAD; ++i) {
            const int d_v = out_dim[i];
            if (d_v < params.d_v) {
                write_output<InputT>(
                    params,
                    batch_idx,
                    q_seq_idx,
                    kv_head_idx,
                    is_no_split,
                    split_idx,
                    d_v,
                    out_acc[i] * inv_row_sum);
            }
        }
    }
}

struct DenseCtaTileCoord {
    int q_seq_per_hk_tile_idx;
    int kv_head_idx;
    int partition_idx;
};

__device__ __forceinline__ DenseCtaTileCoord dense_cta_tile_coord_from_block() {
    return {
        static_cast<int>(blockIdx.x),
        static_cast<int>(blockIdx.y),
        static_cast<int>(blockIdx.z),
    };
}

template<typename InputT, int ONLINE_K_TILE>
__global__ void __launch_bounds__(dense::NUM_THREADS)
flash_fwd_splitkv_mla_sm70_dense_kernel(DenseAttnDecodeParams params) {
    static_assert(sizeof(InputT) == 2, "SM70 dense alpha expects a 16-bit input type");

    const DenseCtaTileCoord coord = dense_cta_tile_coord_from_block();

    extern __shared__ float shared[];

    const DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[coord.partition_idx];
    if (sched_meta.begin_req_idx >= params.b) {
        return;
    }

    for (int local_q_idx = 0; local_q_idx < dense::H_TILE; ++local_q_idx) {
        const int q_seq_idx = coord.q_seq_per_hk_tile_idx * dense::H_TILE + local_q_idx;
        if (q_seq_idx >= params.q_seq_per_hk) {
            return;
        }

#pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            const int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
            const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            const int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
            const int end_block_idx = batch_idx == sched_meta.end_req_idx
                ? sched_meta.end_block_idx
                : (seqlen_k + params.page_block_size - 1) / params.page_block_size;
            const bool is_no_split = batch_idx == sched_meta.begin_req_idx
                ? !sched_meta.is_first_req_splitted
                : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
            const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;

            process_dense_partition<InputT, ONLINE_K_TILE>(
                params,
                batch_idx,
                q_seq_idx,
                coord.kv_head_idx,
                start_block_idx,
                end_block_idx,
                is_no_split,
                split_idx,
                shared);

            __syncthreads();
        }
    }
}

}  // namespace detail

template<int ONLINE_K_TILE>
size_t dense_mma884_online_smem_size(int head_dim) {
    size_t smem_size = static_cast<size_t>(ONLINE_K_TILE) * sizeof(float)
                     + static_cast<size_t>(dense::MMA_884_ONLINE_SCALAR_COUNT) * sizeof(float)
                     + static_cast<size_t>(dense::MMA_884_ONLINE_REDUCE_SCRATCH) * sizeof(float)
                     + static_cast<size_t>(ONLINE_K_TILE) * static_cast<size_t>(head_dim) * sizeof(cutlass::half_t);
    if constexpr (dense::NUM_THREADS != dense::MMA_884_REGISTER_OUTPUT_CTA_THREADS) {
        smem_size += static_cast<size_t>(dense::HEAD_DIM_V) * sizeof(float);
    }
    return smem_size;
}

int select_dense_mma884_online_k_tile(const DenseAttnDecodeParams &params) {
    if constexpr (dense::MMA_884_ONLINE_K_TILE != dense::MMA_884_ONLINE_AUTO_K_TILE) {
        return dense::MMA_884_ONLINE_K_TILE;
    }

    return 64;
}

template<typename InputT, int ONLINE_K_TILE>
void launch_flash_splitkv_mla_sm70_dense_online(DenseAttnDecodeParams &params, dim3 grid, dim3 block) {
    const size_t smem_size = dense_mma884_online_smem_size<ONLINE_K_TILE>(params.d);
    auto kernel = &detail::flash_fwd_splitkv_mla_sm70_dense_kernel<InputT, ONLINE_K_TILE>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params) {
    FLASH_ASSERT(params.d == 512 || params.d == 576);
    FLASH_ASSERT(params.d_v == dense::HEAD_DIM_V);
    FLASH_ASSERT(params.page_block_size == dense::PAGE_BLOCK_SIZE);
    const int q_seq_per_hk_tiles = (params.q_seq_per_hk + dense::H_TILE - 1) / dense::H_TILE;
    dim3 grid(q_seq_per_hk_tiles, params.h_k, params.num_sm_parts);
    dim3 block(dense::NUM_THREADS);
    if constexpr (dense::USE_MMA_884_ONLINE) {
        const int online_k_tile = select_dense_mma884_online_k_tile(params);
        if (online_k_tile == 16) {
            launch_flash_splitkv_mla_sm70_dense_online<InputT, 16>(params, grid, block);
        } else if (online_k_tile == 32) {
            launch_flash_splitkv_mla_sm70_dense_online<InputT, 32>(params, grid, block);
        } else {
            launch_flash_splitkv_mla_sm70_dense_online<InputT, 64>(params, grid, block);
        }
        return;
    }

    const size_t smem_size = (dense::NUM_THREADS + dense::K_TILE) * sizeof(float);
    auto kernel = &detail::flash_fwd_splitkv_mla_sm70_dense_kernel<InputT, dense::MMA_884_ONLINE_K_TILE>;
    kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sm70
