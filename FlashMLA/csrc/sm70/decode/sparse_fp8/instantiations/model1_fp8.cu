#include "sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh"

namespace sm70::decode::sparse_fp8 {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams &params);
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 128>(const SparseAttnDecodeParams &params);
template void run_flash_splitkv_mla_fp8_sparse_batch2_verify_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams &params);
template void run_flash_splitkv_mla_fp8_sparse_batch2_verify_kernel<ModelType::MODEL1, 128>(const SparseAttnDecodeParams &params);

}  // namespace sm70::decode::sparse_fp8


#ifdef FLASH_MLA_METER_SPARSE_DECODE
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

// Per-TU helpers that access this TU's local static __device__ counters.
// The global dump/reset functions below combine counters across TUs
// (v32_fp8.cu + model1_fp8.cu) since prod dispatch hits exactly one TU.
__host__ void decode_stage_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out,
    unsigned long long *block_count_out) {
    cudaMemcpyFromSymbol(cycles_out, g_decode_stage_cycles,
                         6 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, g_decode_stage_tile_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(block_count_out, g_decode_stage_block_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

__host__ void decode_stage_meter_reset_tu_model1() {
    unsigned long long zero = 0;
    unsigned long long zeros6[6] = {0, 0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(g_decode_stage_cycles, zeros6,
                       6 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_stage_tile_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_stage_block_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace detail

void decode_stage_meter_dump(unsigned long long *cycles_out,
                             unsigned long long *tile_count_out,
                             unsigned long long *block_count_out) {
    // Read from each TU and combine (sum). Only one TU writes per decode
    // call depending on model_type dispatch, so sum-across is correct.
    unsigned long long c_m[6] = {0}, c_v[6] = {0};
    unsigned long long t_m = 0, t_v = 0;
    unsigned long long b_m = 0, b_v = 0;
    detail::decode_stage_meter_dump_tu_model1(c_m, &t_m, &b_m);
    detail::decode_stage_meter_dump_tu_v32(c_v, &t_v, &b_v);
    for (int i = 0; i < 6; ++i) cycles_out[i] = c_m[i] + c_v[i];
    *tile_count_out = t_m + t_v;
    *block_count_out = b_m + b_v;
}

void decode_stage_meter_reset() {
    detail::decode_stage_meter_reset_tu_model1();
    detail::decode_stage_meter_reset_tu_v32();
}

}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_S0_SUB
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_s0_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out) {
    cudaMemcpyFromSymbol(cycles_out, g_decode_s0_sub_cycles,
                         5 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, g_decode_s0_sub_tile_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

__host__ void decode_s0_sub_meter_reset_tu_model1() {
    unsigned long long zero = 0;
    unsigned long long zeros5[5] = {0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(g_decode_s0_sub_cycles, zeros5,
                       5 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s0_sub_tile_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace detail
}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_s0c_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out,
    unsigned long long *token_count_out,
    unsigned long long *dim_count_out) {
    cudaMemcpyFromSymbol(cycles_out, g_decode_s0c_sub_cycles,
                         6 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, g_decode_s0c_sub_tile_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(token_count_out, g_decode_s0c_sub_token_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(dim_count_out, g_decode_s0c_sub_dim_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

__host__ void decode_s0c_sub_meter_reset_tu_model1() {
    unsigned long long zero = 0;
    unsigned long long zeros6[6] = {0, 0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(g_decode_s0c_sub_cycles, zeros6,
                       6 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s0c_sub_tile_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s0c_sub_token_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s0c_sub_dim_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace detail
}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_QK_SUB
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_qk_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *group_count_out,
    unsigned long long *dim_group_count_out) {
    cudaMemcpyFromSymbol(cycles_out, g_decode_qk_sub_cycles,
                         4 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(group_count_out, g_decode_qk_sub_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(dim_group_count_out, g_decode_qk_sub_dim_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

__host__ void decode_qk_sub_meter_reset_tu_model1() {
    unsigned long long zero = 0;
    unsigned long long zeros4[4] = {0, 0, 0, 0};
    cudaMemcpyToSymbol(g_decode_qk_sub_cycles, zeros4,
                       4 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_qk_sub_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_qk_sub_dim_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace detail
}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_QK_SUB
namespace sm70::decode::sparse_fp8 {

void decode_qk_sub_meter_dump(unsigned long long *cycles_out,
                              unsigned long long *group_count_out,
                              unsigned long long *dim_group_count_out) {
    unsigned long long c_m[4] = {0}, c_v[4] = {0};
    unsigned long long g_m = 0, g_v = 0;
    unsigned long long dg_m = 0, dg_v = 0;
    detail::decode_qk_sub_meter_dump_tu_model1(c_m, &g_m, &dg_m);
    detail::decode_qk_sub_meter_dump_tu_v32(c_v, &g_v, &dg_v);
    for (int i = 0; i < 4; ++i) cycles_out[i] = c_m[i] + c_v[i];
    *group_count_out = g_m + g_v;
    *dim_group_count_out = dg_m + dg_v;
}

void decode_qk_sub_meter_reset() {
    detail::decode_qk_sub_meter_reset_tu_model1();
    detail::decode_qk_sub_meter_reset_tu_v32();
}

}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_s4a_sub_meter_dump_tu_model1(
    unsigned long long *cycles_out,
    unsigned long long *group_count_out,
    unsigned long long *dim_group_count_out,
    unsigned long long *tile_count_out) {
    cudaMemcpyFromSymbol(cycles_out, g_decode_s4a_sub_cycles,
                         6 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(group_count_out, g_decode_s4a_sub_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(dim_group_count_out, g_decode_s4a_sub_dim_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, g_decode_s4a_sub_tile_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

__host__ void decode_s4a_sub_meter_reset_tu_model1() {
    unsigned long long zero = 0;
    unsigned long long zeros6[6] = {0, 0, 0, 0, 0, 0};
    cudaMemcpyToSymbol(g_decode_s4a_sub_cycles, zeros6,
                       6 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s4a_sub_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s4a_sub_dim_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_decode_s4a_sub_tile_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace detail

void decode_s4a_sub_meter_dump(unsigned long long *cycles_out,
                               unsigned long long *group_count_out,
                               unsigned long long *dim_group_count_out,
                               unsigned long long *tile_count_out) {
    unsigned long long c_m[6] = {0}, c_v[6] = {0};
    unsigned long long g_m = 0, g_v = 0;
    unsigned long long dg_m = 0, dg_v = 0;
    unsigned long long t_m = 0, t_v = 0;
    detail::decode_s4a_sub_meter_dump_tu_model1(c_m, &g_m, &dg_m, &t_m);
    detail::decode_s4a_sub_meter_dump_tu_v32(c_v, &g_v, &dg_v, &t_v);
    for (int i = 0; i < 6; ++i) cycles_out[i] = c_m[i] + c_v[i];
    *group_count_out = g_m + g_v;
    *dim_group_count_out = dg_m + dg_v;
    *tile_count_out = t_m + t_v;
}

void decode_s4a_sub_meter_reset() {
    detail::decode_s4a_sub_meter_reset_tu_model1();
    detail::decode_s4a_sub_meter_reset_tu_v32();
}

}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
namespace sm70::decode::sparse_fp8 {

void decode_s0c_sub_meter_dump(unsigned long long *cycles_out,
                               unsigned long long *tile_count_out,
                               unsigned long long *token_count_out,
                               unsigned long long *dim_count_out) {
    unsigned long long c_m[6] = {0}, c_v[6] = {0};
    unsigned long long t_m = 0, t_v = 0;
    unsigned long long tk_m = 0, tk_v = 0;
    unsigned long long d_m = 0, d_v = 0;
    detail::decode_s0c_sub_meter_dump_tu_model1(c_m, &t_m, &tk_m, &d_m);
    detail::decode_s0c_sub_meter_dump_tu_v32(c_v, &t_v, &tk_v, &d_v);
    for (int i = 0; i < 6; ++i) cycles_out[i] = c_m[i] + c_v[i];
    *tile_count_out = t_m + t_v;
    *token_count_out = tk_m + tk_v;
    *dim_count_out = d_m + d_v;
}

void decode_s0c_sub_meter_reset() {
    detail::decode_s0c_sub_meter_reset_tu_model1();
    detail::decode_s0c_sub_meter_reset_tu_v32();
}

}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_S0_SUB
namespace sm70::decode::sparse_fp8 {

void decode_s0_sub_meter_dump(unsigned long long *cycles_out,
                              unsigned long long *tile_count_out) {
    unsigned long long c_m[5] = {0}, c_v[5] = {0};
    unsigned long long t_m = 0, t_v = 0;
    detail::decode_s0_sub_meter_dump_tu_model1(c_m, &t_m);
    detail::decode_s0_sub_meter_dump_tu_v32(c_v, &t_v);
    for (int i = 0; i < 5; ++i) cycles_out[i] = c_m[i] + c_v[i];
    *tile_count_out = t_m + t_v;
}

void decode_s0_sub_meter_reset() {
    detail::decode_s0_sub_meter_reset_tu_model1();
    detail::decode_s0_sub_meter_reset_tu_v32();
}

}  // namespace sm70::decode::sparse_fp8
#endif
