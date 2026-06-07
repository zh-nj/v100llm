#include "../splitkv_mla_sm70.cuh"
#include "../splitkv_mla_sm70.h"

namespace sm70 {

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_splitkv_mla_kernel<cutlass::half_t>(DenseAttnDecodeParams &params);
#endif

}  // namespace sm70
