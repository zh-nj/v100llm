#pragma once

#include <math.h>
#include <math_constants.h>

#include "mma_884.h"

namespace flash_mla::sm70::softmax {

__device__ __forceinline__ float warp_reduce_max_float(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum_float(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_max_warp_scratch(float value, float* warp_scratch) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    const float warp_value = warp_reduce_max_float(value);
    if (lane == 0) {
        warp_scratch[warp_idx] = warp_value;
    }
    __syncthreads();

    float block_value = -CUDART_INF_F;
    if (warp_idx == 0) {
        block_value = lane < num_warps ? warp_scratch[lane] : -CUDART_INF_F;
        block_value = warp_reduce_max_float(block_value);
        if (lane == 0) {
            warp_scratch[0] = block_value;
        }
    }
    __syncthreads();
    return warp_scratch[0];
}

__device__ __forceinline__ float block_reduce_sum_warp_scratch(float value, float* warp_scratch) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    const float warp_value = warp_reduce_sum_float(value);
    if (lane == 0) {
        warp_scratch[warp_idx] = warp_value;
    }
    __syncthreads();

    float block_value = 0.0F;
    if (warp_idx == 0) {
        block_value = lane < num_warps ? warp_scratch[lane] : 0.0F;
        block_value = warp_reduce_sum_float(block_value);
        if (lane == 0) {
            warp_scratch[0] = block_value;
        }
    }
    __syncthreads();
    return warp_scratch[0];
}

template<bool IgnoreNegInf>
__device__ __forceinline__ void compute_online_softmax_tile_parallel(
    const float* tile_scores,
    int tile_count,
    float* row_max_smem,
    float* row_sum_smem,
    float* online_scale_smem,
    float* online_reduce_scratch) {
    const int tile_idx = threadIdx.x;
    const float local_score = tile_idx < tile_count ? tile_scores[tile_idx] : -CUDART_INF_F;
    const float tile_max = block_reduce_max_warp_scratch(local_score, online_reduce_scratch);

    const float old_max = *row_max_smem;
    const float old_sum = *row_sum_smem;
    if constexpr (IgnoreNegInf) {
        if (tile_max == -CUDART_INF_F) {
            if (threadIdx.x == 0) {
                *online_scale_smem = 1.0F;
            }
            return;
        }
    }

    const float new_max = old_sum > 0.0F ? fmaxf(old_max, tile_max) : tile_max;
    const float old_scale = old_sum > 0.0F ? expf(old_max - new_max) : 0.0F;
    const bool contributes = tile_idx < tile_count && (!IgnoreNegInf || local_score != -CUDART_INF_F);
    const float local_sum = contributes ? expf(local_score - new_max) : 0.0F;
    const float tile_sum = block_reduce_sum_warp_scratch(local_sum, online_reduce_scratch);

    if (threadIdx.x == 0) {
        *row_max_smem = new_max;
        *row_sum_smem = old_sum * old_scale + tile_sum;
        *online_scale_smem = old_scale;
    }
}

template<int Rows>
struct OnlineSoftmax {
    Array<float, Rows> row_max;
    Array<float, Rows> row_sum;

    __device__ __forceinline__ void reset() {
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            row_max[i] = -3.4028234663852886e38F;
            row_sum[i] = 0.0F;
        }
    }

    __device__ __forceinline__ void update(int row, float score) {
        const float old_max = row_max[row];
        const float new_max = fmaxf(old_max, score);
        const float old_scale = __expf(old_max - new_max);
        const float new_scale = __expf(score - new_max);
        row_sum[row] = row_sum[row] * old_scale + new_scale;
        row_max[row] = new_max;
    }

    __device__ __forceinline__ float lse(int row) const {
        return row_max[row] + logf(row_sum[row]);
    }
};

}  // namespace flash_mla::sm70::softmax
