#pragma once

#include <stdint.h>

#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>

#include "config.h"
#include "sm70/common/fp8_dequant.h"

namespace sm70::decode::sparse_fp8 {

__device__ __forceinline__ float bf16_round_to_float(float value) {
    return static_cast<float>(cutlass::bfloat16_t(value));
}

__device__ __forceinline__ float fast_bf16_round_to_float(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounded = bits + 0x00007FFFu + ((bits >> 16) & 1u);
    return __uint_as_float(rounded & 0xFFFF0000u);
}

__device__ __forceinline__ float e8m0_to_float(uint8_t value) {
    return ldexpf(1.0F, static_cast<int>(value) - 127);
}

__device__ __forceinline__ float fast_e4m3_to_float(uint8_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x80u) << 24;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 3) & 0x0Fu;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x07u;

    if (exponent == 0) {
        if (mantissa == 0) {
            return __uint_as_float(sign);
        }
        const float magnitude = static_cast<float>(mantissa) * 0.001953125F;
        return sign ? -magnitude : magnitude;
    }

    const uint32_t bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float s0c3_e4m3_to_float(uint8_t value) {
#if FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION == 1
    return value == 0 ? 0.0F : 1.0F;
#elif FLASH_MLA_SM70_SPARSE_DECODE_FAST_E4M3
    return fast_e4m3_to_float(value);
#else
    return flash_mla::sm70::fp8_dequant::e4m3_to_float(value);
#endif
}

__device__ __forceinline__ float s0c3_v32_scale_to_float(float value) {
#if FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION == 2
    return value == 0.0F ? 0.0F : 1.0F;
#else
    return value;
#endif
}

__device__ __forceinline__ float s0c3_e8m0_to_float(uint8_t value) {
#if FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION == 2
    return value == 0 ? 0.0F : 1.0F;
#else
    return e8m0_to_float(value);
#endif
}

__device__ __forceinline__ float s0c3_round_to_float(float value) {
#if FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION == 3
    return value;
#elif FLASH_MLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND
    return fast_bf16_round_to_float(value);
#else
    return bf16_round_to_float(value);
#endif
}

__device__ __forceinline__ float s0c3_rope_to_float(
    const cutlass::bfloat16_t* rope,
    int index) {
#if FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION == 4
    return 0.0F;
#else
    return static_cast<float>(rope[index]);
#endif
}

__device__ __forceinline__ float v32_load_k_value(const uint8_t* token_base, int dim) {
    if (dim < V32_NOPE_DIM) {
        const float* scales = reinterpret_cast<const float*>(token_base + V32_NOPE_DIM);
        const int scale_idx = dim / V32_SCALE_GROUP;
        const float dequant = s0c3_e4m3_to_float(token_base[dim])
                            * s0c3_v32_scale_to_float(scales[scale_idx]);
        return s0c3_round_to_float(dequant);
    }

    const auto* rope = reinterpret_cast<const cutlass::bfloat16_t*>(
        token_base + V32_NOPE_DIM + V32_NUM_SCALES * static_cast<int>(sizeof(float)));
    return s0c3_rope_to_float(rope, dim - V32_NOPE_DIM);
}

__device__ __forceinline__ float model1_load_k_value(
    const uint8_t* block_base,
    int page_block_size,
    int row_index,
    int dim) {
    const uint8_t* token_base = block_base + row_index * MODEL1_TOKEN_DATA_BYTES;
    if (dim < MODEL1_NOPE_DIM) {
        const uint8_t* scales = block_base
            + page_block_size * MODEL1_TOKEN_DATA_BYTES
            + row_index * MODEL1_SCALE_STRIDE;
        const int scale_idx = dim / MODEL1_SCALE_GROUP;
        const float dequant = s0c3_e4m3_to_float(token_base[dim])
                            * s0c3_e8m0_to_float(scales[scale_idx]);
        return s0c3_round_to_float(dequant);
    }

    const auto* rope = reinterpret_cast<const cutlass::bfloat16_t*>(token_base + MODEL1_NOPE_DIM);
    return s0c3_rope_to_float(rope, dim - MODEL1_NOPE_DIM);
}

}  // namespace sm70::decode::sparse_fp8
