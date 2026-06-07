#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "mma_884.h"

namespace flash_mla::sm70::fp8_dequant {

static constexpr int V32_NOPE_BYTES = 512;
static constexpr int V32_SCALE_COUNT = 4;
static constexpr int V32_ROPE_BF16_COUNT = 64;
static constexpr int V32_BYTES_PER_TOKEN = 656;

__device__ __forceinline__ float e4m3_to_float(uint8_t value) {
    const int sign = value & 0x80;
    const int exponent = (value >> 3) & 0x0f;
    const int mantissa = value & 0x07;

    float magnitude;
    if (exponent == 0) {
        magnitude = mantissa == 0 ? 0.0F : ldexpf(static_cast<float>(mantissa) / 8.0F, -6);
    } else {
        magnitude = ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    }
    return sign ? -magnitude : magnitude;
}

__device__ __forceinline__ half e4m3_to_half(uint8_t value, float scale) {
    return __float2half_rn(e4m3_to_float(value) * scale);
}

template<int Count>
__device__ __forceinline__ void dequant_e4m3_group(
    Array<half, Count>& out,
    const uint8_t* values,
    float scale) {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
        out[i] = e4m3_to_half(values[i], scale);
    }
}

__device__ __forceinline__ int v32_scale_index(int nope_index) {
    return nope_index / 128;
}

}  // namespace flash_mla::sm70::fp8_dequant
