#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace flash_mla::sm70::bf16_cast {

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    union {
        uint32_t u32;
        float f32;
    } value;
    value.u32 = static_cast<uint32_t>(bits) << 16;
    return value.f32;
}

__device__ __forceinline__ half bf16_bits_to_half(uint16_t bits) {
    return __float2half_rn(bf16_bits_to_float(bits));
}

__device__ __forceinline__ half bf16_to_half(__nv_bfloat16 value) {
    return __float2half_rn(__bfloat162float(value));
}

}  // namespace flash_mla::sm70::bf16_cast
