#pragma once

#include "config.h"
#include "params.h"

namespace sm70::prefill::sparse {

template<int HEAD_DIM_QK>
void run_fwd_kernel(const SparseAttnFwdParams &params);

}  // namespace sm70::prefill::sparse
