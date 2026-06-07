// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Torch library registration for FlashMLA

#include <Python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "pytorch_shim.h"
#include "api/common.h"
#include "api/dense_decode.h"
#include "api/dense_fwd.h"
#include "api/sparse_decode.h"
#include "api/sparse_fwd.h"

#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
namespace sm70::prefill::sparse {
    void stage_meter_dump(unsigned long long*, unsigned long long*, unsigned long long*);
    void stage_meter_reset();
}
#endif

#ifdef FLASH_MLA_METER_SPARSE_DECODE
namespace sm70::decode::sparse_fp8 {
    void decode_stage_meter_dump(unsigned long long*, unsigned long long*, unsigned long long*);
    void decode_stage_meter_reset();
    void decode_qk_sub_meter_dump(unsigned long long*, unsigned long long*, unsigned long long*);
    void decode_qk_sub_meter_reset();
}
#endif

#ifdef FLASH_MLA_METER_DECODE_S0_SUB
namespace sm70::decode::sparse_fp8 {
    void decode_s0_sub_meter_dump(unsigned long long*, unsigned long long*);
    void decode_s0_sub_meter_reset();
}
#endif

#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
namespace sm70::decode::sparse_fp8 {
    void decode_s0c_sub_meter_dump(unsigned long long*, unsigned long long*,
                                   unsigned long long*, unsigned long long*);
    void decode_s0c_sub_meter_reset();
}
#endif

#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
namespace sm70::decode::sparse_fp8 {
    void decode_s4a_sub_meter_dump(unsigned long long*, unsigned long long*,
                                   unsigned long long*, unsigned long long*);
    void decode_s4a_sub_meter_reset();
}
#endif

#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
namespace sm70::prefill::sparse {
    void stage_meter_fine_dump(unsigned long long*, unsigned long long*, unsigned long long*);
    void stage_meter_fine_reset();
}
#endif

#ifdef FLASH_MLA_METER_QK_SUB
namespace sm70::prefill::sparse {
    void stage_meter_qk_sub_dump(unsigned long long*, unsigned long long*);
    void stage_meter_qk_sub_reset();
}
#endif

#ifdef FLASH_MLA_METER_S4A_SUB
namespace sm70::prefill::sparse {
    void stage_meter_s4a_sub_dump(unsigned long long*, unsigned long long*, unsigned long long*);
    void stage_meter_s4a_sub_reset();
}
#endif




TORCH_LIBRARY(_flashmla_C, m) {
    m.def("sparse_decode_fwd", make_pytorch_shim(&sparse_attn_decode_interface));
    m.impl("sparse_decode_fwd", torch::kCUDA,
           make_pytorch_shim(&sparse_attn_decode_interface));

    m.def("dense_decode_fwd", make_pytorch_shim(&dense_attn_decode_interface));
    m.impl("dense_decode_fwd", torch::kCUDA,
           make_pytorch_shim(&dense_attn_decode_interface));

    m.def("sparse_prefill_fwd", make_pytorch_shim(&sparse_attn_prefill_interface));
    m.impl("sparse_prefill_fwd", torch::kCUDA,
           make_pytorch_shim(&sparse_attn_prefill_interface));

    m.def("dense_prefill_fwd", make_pytorch_shim(&FMHACutlassSM100FwdRun));
    m.impl("dense_prefill_fwd", torch::kCUDA,
           make_pytorch_shim(&FMHACutlassSM100FwdRun));

#ifdef FLASH_MLA_METER_SPARSE_WS_HPB
    m.def("sparse_prefill_stage_meter_dump_impl(Tensor(a!) out, Tensor(b!) tile_count, Tensor(c!) block_count) -> ()");
    m.impl("sparse_prefill_stage_meter_dump_impl", torch::kCPU,
           [](at::Tensor out, at::Tensor tile_count, at::Tensor block_count) {
               TORCH_CHECK(out.numel() == 6 && out.dtype() == at::kLong);
               TORCH_CHECK(tile_count.numel() == 1 && tile_count.dtype() == at::kLong);
               TORCH_CHECK(block_count.numel() == 1 && block_count.dtype() == at::kLong);
               sm70::prefill::sparse::stage_meter_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(tile_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(block_count.data_ptr()));
           });
    m.def("sparse_prefill_stage_meter_reset_impl() -> ()");
    m.impl("sparse_prefill_stage_meter_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::prefill::sparse::stage_meter_reset(); }));
#endif
#ifdef FLASH_MLA_METER_SPARSE_WS_HPB_FINE
    m.def("sparse_prefill_stage_meter_fine_dump_impl(Tensor(a!) out, Tensor(b!) tile_count, Tensor(c!) block_count) -> ()");
    m.impl("sparse_prefill_stage_meter_fine_dump_impl", torch::kCPU,
           [](at::Tensor out, at::Tensor tile_count, at::Tensor block_count) {
               TORCH_CHECK(out.numel() == 12 && out.dtype() == at::kLong);
               TORCH_CHECK(tile_count.numel() == 1 && tile_count.dtype() == at::kLong);
               TORCH_CHECK(block_count.numel() == 1 && block_count.dtype() == at::kLong);
               sm70::prefill::sparse::stage_meter_fine_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(tile_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(block_count.data_ptr()));
           });
    m.def("sparse_prefill_stage_meter_fine_reset_impl() -> ()");
    m.impl("sparse_prefill_stage_meter_fine_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::prefill::sparse::stage_meter_fine_reset(); }));
#endif
#ifdef FLASH_MLA_METER_QK_SUB
    m.def("sparse_prefill_qk_sub_dump_impl(Tensor(a!) out, Tensor(b!) group_count) -> ()");
    m.impl("sparse_prefill_qk_sub_dump_impl", torch::kCPU,
           [](at::Tensor out, at::Tensor group_count) {
               TORCH_CHECK(out.numel() == 4 && out.dtype() == at::kLong);
               TORCH_CHECK(group_count.numel() == 1 && group_count.dtype() == at::kLong);
               sm70::prefill::sparse::stage_meter_qk_sub_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(group_count.data_ptr()));
           });
    // Decode QK sub-stage meter (4 sub-stages: load_q, load_k, mma884, store).
    m.def("decode_qk_sub_meter_dump_impl(Tensor(a!) out, Tensor(b!) group_count, Tensor(c!) dim_group_count) -> ()");
    m.impl("decode_qk_sub_meter_dump_impl", torch::kCPU,
           [](torch::Tensor &out, torch::Tensor &group_count, torch::Tensor &dim_group_count) {
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
               TORCH_CHECK(out.numel() == 4 && out.dtype() == at::kLong);
               TORCH_CHECK(group_count.numel() == 1 && group_count.dtype() == at::kLong);
               TORCH_CHECK(dim_group_count.numel() == 1 && dim_group_count.dtype() == at::kLong);
               sm70::decode::sparse_fp8::decode_qk_sub_meter_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(group_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(dim_group_count.data_ptr()));
#endif
           });

    m.def("decode_qk_sub_meter_reset_impl() -> ()");
    m.impl("decode_qk_sub_meter_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() {
#ifdef FLASH_MLA_METER_DECODE_QK_SUB
                   sm70::decode::sparse_fp8::decode_qk_sub_meter_reset();
#endif
               }));

    m.def("sparse_prefill_qk_sub_reset_impl() -> ()");
    m.impl("sparse_prefill_qk_sub_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::prefill::sparse::stage_meter_qk_sub_reset(); }));
#endif
#ifdef FLASH_MLA_METER_SPARSE_DECODE
    m.def("sparse_decode_stage_meter_dump_impl(Tensor(a!) out, Tensor(b!) tile_count, Tensor(c!) block_count) -> ()");
    m.impl("sparse_decode_stage_meter_dump_impl", torch::kCPU,
           [](torch::Tensor &out, torch::Tensor &tile_count, torch::Tensor &block_count) {
               TORCH_CHECK(out.numel() == 6 && out.dtype() == at::kLong);
               TORCH_CHECK(tile_count.numel() == 1 && tile_count.dtype() == at::kLong);
               TORCH_CHECK(block_count.numel() == 1 && block_count.dtype() == at::kLong);
               sm70::decode::sparse_fp8::decode_stage_meter_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(tile_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(block_count.data_ptr()));
           });

    m.def("sparse_decode_stage_meter_reset_impl() -> ()");
    m.impl("sparse_decode_stage_meter_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::decode::sparse_fp8::decode_stage_meter_reset(); }));
#endif
#ifdef FLASH_MLA_METER_DECODE_S0_SUB
    m.def("decode_s0_sub_meter_dump_impl(Tensor(a!) out, Tensor(b!) tile_count) -> ()");
    m.impl("decode_s0_sub_meter_dump_impl", torch::kCPU,
           [](torch::Tensor &out, torch::Tensor &tile_count) {
               TORCH_CHECK(out.numel() == 5 && out.dtype() == at::kLong);
               TORCH_CHECK(tile_count.numel() == 1 && tile_count.dtype() == at::kLong);
               sm70::decode::sparse_fp8::decode_s0_sub_meter_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(tile_count.data_ptr()));
           });

    m.def("decode_s0_sub_meter_reset_impl() -> ()");
    m.impl("decode_s0_sub_meter_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::decode::sparse_fp8::decode_s0_sub_meter_reset(); }));
#endif
#ifdef FLASH_MLA_METER_DECODE_S0C_SUB
    m.def("decode_s0c_sub_meter_dump_impl(Tensor(a!) out, Tensor(b!) tile_count, Tensor(c!) token_count, Tensor(d!) dim_count) -> ()");
    m.impl("decode_s0c_sub_meter_dump_impl", torch::kCPU,
           [](torch::Tensor &out, torch::Tensor &tile_count,
              torch::Tensor &token_count, torch::Tensor &dim_count) {
               TORCH_CHECK(out.numel() == 6 && out.dtype() == at::kLong);
               TORCH_CHECK(tile_count.numel() == 1 && tile_count.dtype() == at::kLong);
               TORCH_CHECK(token_count.numel() == 1 && token_count.dtype() == at::kLong);
               TORCH_CHECK(dim_count.numel() == 1 && dim_count.dtype() == at::kLong);
               sm70::decode::sparse_fp8::decode_s0c_sub_meter_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(tile_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(token_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(dim_count.data_ptr()));
           });

    m.def("decode_s0c_sub_meter_reset_impl() -> ()");
    m.impl("decode_s0c_sub_meter_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::decode::sparse_fp8::decode_s0c_sub_meter_reset(); }));
#endif
#ifdef FLASH_MLA_METER_DECODE_S4A_SUB
    m.def("decode_s4a_sub_meter_dump_impl(Tensor(a!) out, Tensor(b!) group_count, Tensor(c!) dim_group_count, Tensor(d!) tile_count) -> ()");
    m.impl("decode_s4a_sub_meter_dump_impl", torch::kCPU,
           [](torch::Tensor &out, torch::Tensor &group_count,
              torch::Tensor &dim_group_count, torch::Tensor &tile_count) {
               TORCH_CHECK(out.numel() == 6 && out.dtype() == at::kLong);
               TORCH_CHECK(group_count.numel() == 1 && group_count.dtype() == at::kLong);
               TORCH_CHECK(dim_group_count.numel() == 1 && dim_group_count.dtype() == at::kLong);
               TORCH_CHECK(tile_count.numel() == 1 && tile_count.dtype() == at::kLong);
               sm70::decode::sparse_fp8::decode_s4a_sub_meter_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(group_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(dim_group_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(tile_count.data_ptr()));
           });

    m.def("decode_s4a_sub_meter_reset_impl() -> ()");
    m.impl("decode_s4a_sub_meter_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::decode::sparse_fp8::decode_s4a_sub_meter_reset(); }));
#endif
#ifdef FLASH_MLA_METER_S4A_SUB
    m.def("sparse_prefill_s4a_sub_dump_impl(Tensor(a!) out, Tensor(b!) group_count, Tensor(c!) dim_group_count) -> ()");
    m.impl("sparse_prefill_s4a_sub_dump_impl", torch::kCPU,
           [](at::Tensor out, at::Tensor group_count, at::Tensor dim_group_count) {
               TORCH_CHECK(out.numel() == 5 && out.dtype() == at::kLong);
               TORCH_CHECK(group_count.numel() == 1 && group_count.dtype() == at::kLong);
               TORCH_CHECK(dim_group_count.numel() == 1 && dim_group_count.dtype() == at::kLong);
               sm70::prefill::sparse::stage_meter_s4a_sub_dump(
                   reinterpret_cast<unsigned long long*>(out.data_ptr()),
                   reinterpret_cast<unsigned long long*>(group_count.data_ptr()),
                   reinterpret_cast<unsigned long long*>(dim_group_count.data_ptr()));
           });
    m.def("sparse_prefill_s4a_sub_reset_impl() -> ()");
    m.impl("sparse_prefill_s4a_sub_reset_impl",
           torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd,
               []() { sm70::prefill::sparse::stage_meter_s4a_sub_reset(); }));
#endif
}

PyMODINIT_FUNC PyInit__flashmla_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
