#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace flash_mla::sm70::load_store {

template<typename T>
__device__ __forceinline__ T load_128b(const void* addr) {
    static_assert(sizeof(T) == 16, "load_128b expects a 16-byte destination type");
    int4 raw;
#if defined(__CUDA_ARCH__)
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
        : "l"(addr));
#else
    raw = *reinterpret_cast<const int4*>(addr);
#endif
    return *reinterpret_cast<T*>(&raw);
}

template<typename T>
__device__ __forceinline__ T load_shared(const T* addr) {
    return *addr;
}

template<typename T>
__device__ __forceinline__ void store_shared(T* addr, const T& value) {
    *addr = value;
}

}  // namespace flash_mla::sm70::load_store
