#include "../fwd.cuh"
#include "../fwd.h"

namespace sm70::prefill::sparse {

template void run_fwd_kernel<512>(const SparseAttnFwdParams &params);
template void run_fwd_kernel<576>(const SparseAttnFwdParams &params);

}  // namespace sm70::prefill::sparse


#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
#include <cuda_runtime.h>

// g_stage_cycles / g_stage_tile_count / g_stage_block_count are
// defined inside `namespace sm70::prefill::sparse::detail` by fwd.cuh
// which bf16.cu already includes — no extern declarations needed.

namespace sm70::prefill::sparse {

void stage_meter_dump(unsigned long long *cycles_out,
                      unsigned long long *tile_count_out,
                      unsigned long long *block_count_out) {
    cudaMemcpyFromSymbol(cycles_out, detail::g_stage_cycles,
                         6 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, detail::g_stage_tile_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(block_count_out, detail::g_stage_block_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

void stage_meter_reset() {
    unsigned long long zero = 0;
    unsigned long long zeros6[6] = {0,0,0,0,0,0};
    cudaMemcpyToSymbol(detail::g_stage_cycles, zeros6,
                       6 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_stage_tile_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_stage_block_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace sm70::prefill::sparse
#endif


#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
#include <cuda_runtime.h>

namespace sm70::prefill::sparse {

void stage_meter_fine_dump(unsigned long long *cycles_out,
                           unsigned long long *tile_count_out,
                           unsigned long long *block_count_out) {
    cudaMemcpyFromSymbol(cycles_out, detail::g_stage_cycles_fine,
                         12 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(tile_count_out, detail::g_stage_tile_count_fine,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(block_count_out, detail::g_stage_block_count_fine,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

void stage_meter_fine_reset() {
    unsigned long long zero = 0;
    unsigned long long zeros12[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    cudaMemcpyToSymbol(detail::g_stage_cycles_fine, zeros12,
                       12 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_stage_tile_count_fine, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_stage_block_count_fine, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace sm70::prefill::sparse
#endif


#ifdef FLASH_MLA_METER_QK_SUB
#include <cuda_runtime.h>

namespace sm70::prefill::sparse {

void stage_meter_qk_sub_dump(unsigned long long *cycles_out,
                             unsigned long long *group_count_out) {
    cudaMemcpyFromSymbol(cycles_out, detail::g_qk_sub_cycles,
                         4 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(group_count_out, detail::g_qk_sub_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

void stage_meter_qk_sub_reset() {
    unsigned long long zero = 0;
    unsigned long long zeros4[4] = {0,0,0,0};
    cudaMemcpyToSymbol(detail::g_qk_sub_cycles, zeros4,
                       4 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_qk_sub_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace sm70::prefill::sparse
#endif


#ifdef FLASH_MLA_METER_S4A_SUB
#include <cuda_runtime.h>

namespace sm70::prefill::sparse {

void stage_meter_s4a_sub_dump(unsigned long long *cycles_out,
                              unsigned long long *group_count_out,
                              unsigned long long *dim_group_count_out) {
    cudaMemcpyFromSymbol(cycles_out, detail::g_s4a_sub_cycles,
                         5 * sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(group_count_out, detail::g_s4a_sub_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(dim_group_count_out, detail::g_s4a_sub_dim_group_count,
                         sizeof(unsigned long long), 0,
                         cudaMemcpyDeviceToHost);
}

void stage_meter_s4a_sub_reset() {
    unsigned long long zero = 0;
    unsigned long long zeros5[5] = {0,0,0,0,0};
    cudaMemcpyToSymbol(detail::g_s4a_sub_cycles, zeros5,
                       5 * sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_s4a_sub_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(detail::g_s4a_sub_dim_group_count, &zero,
                       sizeof(unsigned long long), 0,
                       cudaMemcpyHostToDevice);
}

}  // namespace sm70::prefill::sparse
#endif
