#pragma once

#include "config.h"
#include "params.h"

namespace sm70::decode::sparse_fp8 {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_batch2_verify_kernel(const SparseAttnDecodeParams &params);

}  // namespace sm70::decode::sparse_fp8
