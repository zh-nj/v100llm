#pragma once

#include <cuda_fp16.h>
#include <math_constants.h>

#include "sm70/common/mma_884.h"

namespace flash_mla::sm70::attention {

__device__ __forceinline__ half half_from_float(float value) {
    return __float2half_rn(value);
}

__device__ __forceinline__ int mma884_q_fragment_row(int lane) {
    return (lane & 8) + (lane & 3) + (lane >> 4) * 4;
}

__device__ __forceinline__ int mma884_qk_score_row(int lane, int q_iter) {
    return (lane & 8) + (lane & 1) + (lane >> 4) * 4 + q_iter * 2;
}

__device__ __forceinline__ int mma884_qk_token_offset(int lane) {
    return (lane >> 4) * 4 + (lane & 4) * 2 + (lane & 3);
}

__device__ __forceinline__ int mma884_qk_score_col(int lane, int s1, int s0) {
    return (lane & 4) * 2 + (lane & 2) + s1 * 4 + s0;
}

__device__ __forceinline__ int mma884_p_fragment_row(int lane) {
    return mma884_q_fragment_row(lane);
}

__device__ __forceinline__ int mma884_pv_token_offset(int lane) {
    return lane & 3;
}

__device__ __forceinline__ int mma884_pv_dim_offset(int lane) {
    return (lane >> 4) * 4 + (lane & 4) * 2;
}

__device__ __forceinline__ int mma884_pv_output_row(int lane, int q_iter) {
    return (lane >> 4) * 4 + (lane & 8) + (lane & 1) + q_iter * 2;
}

__device__ __forceinline__ int mma884_pv_output_col(int lane, int d1, int d0) {
    return (lane & 4) * 2 + (lane & 2) + d1 * 4 + d0;
}

__device__ __forceinline__ void mma884_accumulate_qk(
    Array<float, 8> &score_frag,
    const Array<half, 4> &frag_q,
    const Array<half, 4> &frag_k) {
    Array<float, 8> next_frag{};
    mma_m8n8k4_row_col(next_frag, frag_q, frag_k, score_frag);
    score_frag = next_frag;
}

__device__ __forceinline__ void mma884_accumulate_pv(
    Array<float, 8> &out_frag,
    const Array<half, 4> &frag_p,
    const Array<half, 4> &frag_v) {
    Array<float, 8> next_frag{};
    mma_m8n8k4_row_row(next_frag, frag_p, frag_v, out_frag);
    out_frag = next_frag;
}

template<bool UseTokenRefs>
__device__ __forceinline__ void store_mma884_row0_scores(
    const Array<float, 8> &score_frag,
    float* scores,
    const int* token_refs,
    int token_base,
    int tile_count,
    int lane,
    float scale) {
#pragma unroll
    for (int s1 = 0; s1 < 2; ++s1) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int s0 = 0; s0 < 2; ++s0) {
                const int q_row = mma884_qk_score_row(lane, q);
                const int s_col = mma884_qk_score_col(lane, s1, s0);
                const int local_token_idx = token_base + s_col;
                if (q_row == 0 && local_token_idx < tile_count) {
                    const bool valid_token = !UseTokenRefs || token_refs[local_token_idx] != 0;
                    scores[local_token_idx] = valid_token
                        ? score_frag[s1 * 4 + q * 2 + s0] * scale
                        : -CUDART_INF_F;
                }
            }
        }
    }
}

template<bool SkipInvalidScores>
__device__ __forceinline__ void load_mma884_p_fragment_from_scores(
    const float* scores,
    int score_base,
    int token_base,
    int tile_count,
    int lane,
    float row_ref,
    Array<half, 4> &frag_p) {
    const int p_row = mma884_p_fragment_row(lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int local_token_idx = token_base + i;
        const int score_idx = score_base + local_token_idx;
        if (p_row == 0 && local_token_idx < tile_count) {
            const float score = scores[score_idx];
            const bool valid_score = !SkipInvalidScores || score != -CUDART_INF_F;
            frag_p[i] = valid_score
                ? half_from_float(expf(score - row_ref))
                : half_from_float(0.0F);
        } else {
            frag_p[i] = half_from_float(0.0F);
        }
    }
}

__device__ __forceinline__ void add_mma884_row0_output_fragment(
    const Array<float, 8> &out_frag,
    float* output_accum,
    int dim_base,
    int lane,
    int head_dim_v) {
#pragma unroll
    for (int d1 = 0; d1 < 2; ++d1) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int d0 = 0; d0 < 2; ++d0) {
                const int q_row = mma884_pv_output_row(lane, q);
                const int dim = dim_base + mma884_pv_output_col(lane, d1, d0);
                if (q_row == 0 && dim < head_dim_v) {
                    output_accum[dim] += out_frag[d1 * 4 + q * 2 + d0];
                }
            }
        }
    }
}

}  // namespace flash_mla::sm70::attention
