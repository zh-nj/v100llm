#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace flash_mla::sm70 {

template<typename T, int N>
struct alignas(sizeof(T) * N) Array {
    T values[N];

    __host__ __device__ __forceinline__ T& operator[](int index) {
        return values[index];
    }

    __host__ __device__ __forceinline__ const T& operator[](int index) const {
        return values[index];
    }
};

__device__ __forceinline__ int lane_id() {
    int lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__host__ __device__ __forceinline__ int quadpair_id_from_lane(int lane) {
    return (lane & 0x0f) >> 2;
}

__host__ __device__ __forceinline__ int quadpair_rank_from_lane(int lane) {
    return (lane & 0x03) | ((lane >> 2) & 0x04);
}

// Minimal Volta MMA_884 wrappers for FlashMLA SM70 kernels.
// The instruction shape and operand register layout follow NVIDIA Volta PTX
// documentation and TurboMind's SM70 MMA wrapper in
// src/turbomind/kernels/core/mma.h.
__device__ __forceinline__ void mma_m8n8k4_row_col(
    Array<float, 8>& d,
    const Array<half, 4>& a,
    const Array<half, 4>& b,
    const Array<float, 8>& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const uint32_t* A = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(&b);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11}, "
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        d[i] = c[i];
    }
#endif
}

__device__ __forceinline__ void mma_m8n8k4_row_row(
    Array<float, 8>& d,
    const Array<half, 4>& a,
    const Array<half, 4>& b,
    const Array<float, 8>& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const uint32_t* A = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(&b);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11}, "
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        d[i] = c[i];
    }
#endif
}

}  // namespace flash_mla::sm70
