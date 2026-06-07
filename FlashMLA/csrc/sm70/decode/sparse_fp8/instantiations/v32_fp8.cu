#include "../splitkv_mla_sm70_sparse.cuh"
#include "../splitkv_mla_sm70_sparse.h"

namespace sm70::decode::sparse_fp8 {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(const SparseAttnDecodeParams &params);
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 128>(const SparseAttnDecodeParams &params);

}  // namespace sm70::decode::sparse_fp8

#ifdef FLASH_MLA_METER_SPARSE_DECODE
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_stage_meter_dump_tu_v32(
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

__host__ void decode_stage_meter_reset_tu_v32() {
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
}  // namespace sm70::decode::sparse_fp8
#endif

#ifdef FLASH_MLA_METER_DECODE_S0_SUB
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_s0_sub_meter_dump_tu_v32(
    unsigned long long *cycles_out,
    unsigned long long *tile_count_out) {
    cudaMemcpyFromSymbol(cycles_out, g_decode_s0_sub_cycles,
                         5 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, g_decode_s0_sub_tile_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

__host__ void decode_s0_sub_meter_reset_tu_v32() {
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

__host__ void decode_s0c_sub_meter_dump_tu_v32(
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

__host__ void decode_s0c_sub_meter_reset_tu_v32() {
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

__host__ void decode_qk_sub_meter_dump_tu_v32(
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

__host__ void decode_qk_sub_meter_reset_tu_v32() {
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

#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
#include <cuda_runtime.h>

namespace sm70::decode::sparse_fp8 {
namespace detail {

__host__ void decode_s4a_sub_meter_dump_tu_v32(
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

__host__ void decode_s4a_sub_meter_reset_tu_v32() {
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
}  // namespace sm70::decode::sparse_fp8
#endif
