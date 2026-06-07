#pragma once

namespace flash_mla::sm70::layout {

__host__ __device__ __forceinline__ int padded_stride(int logical_stride, int padding) {
    return logical_stride + padding;
}

__host__ __device__ __forceinline__ int padded_offset(
    int row,
    int col,
    int logical_stride,
    int padding) {
    return row * padded_stride(logical_stride, padding) + col;
}

__host__ __device__ __forceinline__ int swizzle_128b_offset(int offset) {
    offset = ((offset & 8) << 2) ^ offset;
    offset = ((offset & ~20) | (((offset & 16) >> 2) | ((offset & 4) << 2)));
    return ((offset & (0x3 << 6)) >> 3) ^ offset;
}

}  // namespace flash_mla::sm70::layout
