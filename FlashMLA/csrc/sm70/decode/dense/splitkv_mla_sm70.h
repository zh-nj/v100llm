#pragma once

#include "params.h"

namespace sm70 {

template<typename InputT>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params);

}  // namespace sm70
