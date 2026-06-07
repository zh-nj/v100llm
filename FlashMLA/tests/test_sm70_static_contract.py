import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SM70_COMMON_HEADERS = [
    "csrc/sm70/common/mma_884.h",
    "csrc/sm70/common/layout.h",
    "csrc/sm70/common/load_store.h",
    "csrc/sm70/common/softmax.h",
    "csrc/sm70/common/bf16_cast.h",
    "csrc/sm70/common/fp8_dequant.h",
]
SM70_DENSE_FILES = [
    "csrc/sm70/decode/dense/config.h",
    "csrc/sm70/decode/dense/splitkv_mla_sm70.h",
    "csrc/sm70/decode/dense/splitkv_mla_sm70.cuh",
    "csrc/sm70/decode/dense/instantiations/fp16.cu",
]
SM70_SPARSE_FILES = [
    "csrc/sm70/decode/sparse_fp8/config.h",
    "csrc/sm70/decode/sparse_fp8/dequant.h",
    "csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.h",
    "csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh",
    "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu",
]
SM70_SPARSE_PREFILL_FILES = [
    "csrc/sm70/prefill/sparse/config.h",
    "csrc/sm70/prefill/sparse/fwd.h",
    "csrc/sm70/prefill/sparse/fwd.cuh",
    "csrc/sm70/prefill/sparse/instantiations/bf16.cu",
]
SM70_VLLM_INTEGRATION_DOC = "docs/sm70-volta-vllm-integration-notes.md"
SM70_VLLM_BENCH = "benchmark/bench_vllm_deepseek_v4_flash_sm70.py"
SM70_SPARSE_DECODE_BENCH = "benchmark/bench_sm70_sparse_decode.py"
SM70_SPARSE_PREFILL_BENCH = "benchmark/bench_sm70_sparse_prefill.py"


def test_setup_exposes_opt_in_sm70_build_flag():
    setup_py = (REPO_ROOT / "setup.py").read_text()

    assert "FLASH_MLA_ENABLE_SM70" in setup_py
    assert "arch=compute_70,code=sm_70" in setup_py


def test_arch_helper_knows_sm70():
    common_h = (REPO_ROOT / "csrc" / "api" / "common.h").read_text()

    assert "is_sm70" in common_h
    assert "major == 7 && minor == 0" in common_h


def test_sm70_common_headers_are_present():
    for relative_path in SM70_COMMON_HEADERS:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_sm70_dense_decode_files_are_present():
    for relative_path in SM70_DENSE_FILES:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_sm70_sparse_decode_files_are_present():
    for relative_path in SM70_SPARSE_FILES:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_sm70_sparse_prefill_bf16_files_are_present():
    for relative_path in SM70_SPARSE_PREFILL_FILES:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_setup_builds_sm70_dense_instantiation():
    setup_py = (REPO_ROOT / "setup.py").read_text()

    assert "csrc/sm70/decode/dense/instantiations/fp16.cu" in setup_py


def test_setup_builds_sm70_sparse_v32_instantiation():
    setup_py = (REPO_ROOT / "setup.py").read_text()

    assert "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu" in setup_py


def test_setup_uses_local_cutlass_checkout_when_present():
    setup_py = (REPO_ROOT / "setup.py").read_text()

    assert "ensure_cutlass_checkout" in setup_py
    assert "csrc/cutlass/include/cutlass/bfloat16.h" in setup_py


def test_sm70_build_allows_legacy_sources_to_skip_sm70_device_pass_features():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    utils_h = (REPO_ROOT / "csrc" / "utils.h").read_text()
    kerutils_common_h = (
        REPO_ROOT / "csrc" / "kerutils" / "include" / "kerutils" / "device" / "common.h"
    ).read_text()

    assert "KERUTILS_ALLOW_SM70_STUB_COMPILE" in setup_py
    assert "FLASH_MLA_LAUNCH_BOUNDS" in utils_h
    assert "KERUTILS_ALLOW_SM70_STUB_COMPILE" in kerutils_common_h


def test_sm70_combine_skips_grid_dependency_instruction_for_volta_device_pass():
    combine_cu = (REPO_ROOT / "csrc" / "smxx" / "decode" / "combine" / "combine.cu").read_text()

    assert "cudaGridDependencySynchronize" in combine_cu
    assert "__CUDA_ARCH__ >= 900" in combine_cu


def test_dense_decode_dispatches_sm70_fp16_and_rejects_bf16():
    dense_decode_h = (REPO_ROOT / "csrc" / "api" / "dense_decode.h").read_text()
    setup_py = (REPO_ROOT / "setup.py").read_text()

    assert "sm70/decode/dense/splitkv_mla_sm70.h" in dense_decode_h
    assert "arch.is_sm70()" in dense_decode_h
    assert "sm70::run_flash_splitkv_mla_kernel<cutlass::half_t>" in dense_decode_h
    assert "sm70::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>" not in dense_decode_h
    assert "SM70 dense decode supports FP16 only" in dense_decode_h
    assert "csrc/sm70/decode/dense/instantiations/bf16_compat.cu" not in setup_py


def test_sparse_decode_dispatches_sm70_fp8_alpha_and_rejects_unfinished_features():
    sparse_decode_h = (REPO_ROOT / "csrc" / "api" / "sparse_decode.h").read_text()

    assert "sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.h" in sparse_decode_h
    assert "Decode_Sm70_Fp8_Dequant_Impl" in sparse_decode_h
    assert "arch.is_sm70()" in sparse_decode_h
    assert "SM70 sparse FP8 alpha" in sparse_decode_h
    sm70_impl_features = sparse_decode_h.split("class Decode_Sm70_Fp8_Dequant_Impl", 1)[1].split("public:", 1)[0]
    assert "DecodeFeatures::HEAD_DIM_512" in sm70_impl_features
    assert "DecodeFeatures::HEAD_DIM_576" in sm70_impl_features
    assert "DecodeFeatures::MODEL1_KVCACHE_FORMAT" in sm70_impl_features
    assert "DecodeFeatures::TOPK_LENGTH" in sm70_impl_features
    assert "DecodeFeatures::EXTRA_KVCACHE" in sm70_impl_features
    assert "DecodeFeatures::EXTRA_TOPK_LENGTH" in sm70_impl_features


def test_feature_gate_error_message_names_missing_features():
    common_h = (REPO_ROOT / "csrc" / "api" / "common.h").read_text()

    assert "missing_features=" in common_h
    assert "required_features=" in common_h
    assert "supported_features=" in common_h
    assert "format_feature_list" in common_h


def test_sm70_sparse_decode_errors_include_arch_model_and_missing_feature_context():
    sparse_decode_h = (REPO_ROOT / "csrc" / "api" / "sparse_decode.h").read_text()

    assert "sm70_sparse_error_context" in sparse_decode_h
    assert "arch=sm70" in sparse_decode_h
    assert "model=" in sparse_decode_h
    assert "missing_feature=" in sparse_decode_h
    assert "TOTAL_TOPK_GT_8192" in sparse_decode_h


def test_sm70_sparse_decode_supports_8192_total_topk_contract():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_decode_h = (REPO_ROOT / "csrc" / "api" / "sparse_decode.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()
    sparse_test = (REPO_ROOT / "tests" / "test_flash_mla_sparse_decoding.py").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    design_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-design.md").read_text()
    task_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md").read_text()

    assert "MAX_TOPK_ALPHA = 8192" in config_h
    assert "params.topk + params.extra_topk <= MAX_TOPK_ALPHA" in sparse_cuh
    assert "TOTAL_TOPK_GT_8192" in sparse_decode_h
    assert "total topk <= 8192" in sparse_decode_h
    assert "TOTAL_TOPK_GT_2048" not in sparse_decode_h
    assert "SM70_SPARSE_DECODE_MAX_TOPK = 8192" in bench
    assert 'elif name == "max_topk"' in bench
    assert "topk=SM70_SPARSE_DECODE_MAX_TOPK" in bench
    assert "extra_topk=4096" in bench
    assert "topk=8192" in sparse_test
    assert "extra_topk=4096" in sparse_test
    assert "topk + extra_topk <= 8192" in readme
    assert "benchmark/bench_sm70_sparse_decode.py --cases max_topk" in readme
    assert "topk + extra_topk <= 8192" in design_doc
    assert "4096+4096" in design_doc
    assert "15/15 correctness" in task_doc
    assert "benchmark/bench_sm70_sparse_decode.py --cases max_topk" in task_doc


def test_dense_decode_launches_combine_for_sm70_after_split_kernel():
    dense_decode_h = (REPO_ROOT / "csrc" / "api" / "dense_decode.h").read_text()

    assert "smxx::decode::run_flash_mla_combine_kernel<cutlass::half_t>" in dense_decode_h
    assert "if (!arch.is_sm70())" not in dense_decode_h


def test_sm70_dense_decode_uses_scheduler_and_split_accumulators():
    dense_decode_h = (REPO_ROOT / "csrc" / "api" / "dense_decode.h").read_text()
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()

    assert "num_sm_parts = 1" not in dense_decode_h
    assert "params.tile_scheduler_metadata_ptr[coord.partition_idx]" in sm70_dense_cuh
    assert "dense::H_TILE" in sm70_dense_cuh
    assert "q_seq_per_hk_tile_idx" in sm70_dense_cuh
    assert "local_q_idx < dense::H_TILE" in sm70_dense_cuh
    assert "(params.q_seq_per_hk + dense::H_TILE - 1) / dense::H_TILE" in sm70_dense_cuh
    assert "params.oaccum_ptr" in sm70_dense_cuh
    assert "params.softmax_lseaccum_ptr" in sm70_dense_cuh


def test_sm70_dense_cta_tile_contract_matches_flattened_q_head_semantics():
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    task_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md").read_text()
    design_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-design.md").read_text()

    assert "struct DenseCtaTileCoord" in sm70_dense_cuh
    assert "dense_cta_tile_coord_from_block" in sm70_dense_cuh
    assert "q_seq_per_hk_tile_idx" in sm70_dense_cuh
    assert "coord.q_seq_per_hk_tile_idx * dense::H_TILE" in sm70_dense_cuh
    assert "coord.kv_head_idx" in sm70_dense_cuh
    assert "coord.partition_idx" in sm70_dense_cuh
    assert "dim3 grid(q_seq_per_hk_tiles, params.h_k, params.num_sm_parts)" in sm70_dense_cuh

    assert "- [x] 每个 CTA 处理一个 `(scheduler partition, kv_head, q_seq_per_hk tile)`" in task_doc
    assert "q_seq_per_hk tile" in design_doc
    assert "scheduler partition" in design_doc


def test_sm70_dense_tile_and_thread_config_are_build_time_tunable():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "config.h").read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "FLASH_MLA_SM70_H_TILE" in setup_py
    assert "FLASH_MLA_SM70_CTA_THREADS" in setup_py
    assert "-DFLASH_MLA_SM70_H_TILE=" in setup_py
    assert "-DFLASH_MLA_SM70_CTA_THREADS=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_H_TILE" in config_h
    assert "#ifndef FLASH_MLA_SM70_CTA_THREADS" in config_h
    assert "H_TILE = FLASH_MLA_SM70_H_TILE" in config_h
    assert "NUM_THREADS = FLASH_MLA_SM70_CTA_THREADS" in config_h
    assert "os.getenv(\"FLASH_MLA_SM70_H_TILE\"" in bench
    assert "os.getenv(\"FLASH_MLA_SM70_CTA_THREADS\"" in bench


def test_sm70_sparse_decode_thread_config_is_build_time_tunable():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_sparse_decode.py").read_text()

    assert "FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS" in config_h
    assert "NUM_THREADS = FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS" in config_h
    assert "os.getenv(\"FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS\"" in bench
    assert "SM70_SPARSE_DECODE_REGISTERS_BY_CTA_THREADS_AND_MODEL" in bench
    assert "SM70_SPARSE_DECODE_SPILLS_BY_CTA_THREADS_AND_MODEL" in bench
    assert "(128, \"model1\"): 64" in bench
    assert "(128, \"model1\"): 0" in bench


def test_sm70_sparse_decode_k_tile_is_build_time_tunable():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_sparse_decode.py").read_text()

    assert "FLASH_MLA_SM70_SPARSE_DECODE_K_TILE" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_DECODE_K_TILE=" in setup_py
    assert "{16, 32, 64}" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_DECODE_K_TILE" in config_h
    assert "SPARSE_K_TILE = FLASH_MLA_SM70_SPARSE_DECODE_K_TILE" in config_h
    assert "os.getenv(\"FLASH_MLA_SM70_SPARSE_DECODE_K_TILE\", \"32\")" in bench


def test_sm70_dense_scalar_path_caches_qk_scores_by_k_tile():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "config.h").read_text()
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "MAX_DV_PER_THREAD" in config_h
    assert "score_smem" in sm70_dense_cuh
    assert "dense::K_TILE" in sm70_dense_cuh
    assert "out_acc[dense::MAX_DV_PER_THREAD]" in sm70_dense_cuh
    assert "(dense::NUM_THREADS + dense::K_TILE) * sizeof(float)" in sm70_dense_cuh
    assert "(SM70_ALPHA_CTA_THREADS + 64) * 4" in bench


def test_sm70_dense_has_opt_in_mma884_online_mainloop():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "config.h").read_text()
    common_attention = (REPO_ROOT / "csrc" / "sm70" / "common" / "mma_884_attention.h").read_text()
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE" in setup_py
    assert "-DFLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE=" in setup_py
    assert "FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE" in setup_py
    assert "-DFLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE=" in setup_py
    dense_online_k_tile_line = next(
        line
        for line in setup_py.splitlines()
        if "dense_mma884_online_k_tile = get_env_int" in line
    )
    assert "{0, 16, 32, 64}" in dense_online_k_tile_line
    assert "#ifndef FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE" in config_h
    assert "#ifndef FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE" in config_h
    assert "USE_MMA_884_ONLINE = FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE != 0" in config_h
    assert "MMA_884_ONLINE_K_TILE = FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE" in config_h
    assert "sm70/common/mma_884.h" in common_attention
    assert "sm70/common/mma_884_attention.h" in sm70_dense_cuh
    assert "stage_dense_kv_tile_to_shared" in sm70_dense_cuh
    assert "compute_dense_mma884_qk_group" in sm70_dense_cuh
    assert "compute_dense_mma884_online_pv_group" in sm70_dense_cuh
    assert "accumulate_dense_output_mma884_online" in sm70_dense_cuh
    assert "if constexpr (dense::USE_MMA_884_ONLINE)" in sm70_dense_cuh
    assert "mma_m8n8k4_row_col" in common_attention
    assert "mma_m8n8k4_row_row" in common_attention
    assert "mma884_accumulate_qk" in sm70_dense_cuh
    assert "mma884_accumulate_pv" in sm70_dense_cuh
    assert "SM70_ALPHA_USE_MMA_884_ONLINE" in bench
    assert "SM70_ALPHA_MMA_884_ONLINE_K_TILE" in bench
    assert "SM70_ALPHA_COMPUTE_PATH" in bench
    assert '"mma884_online": 113' in bench
    assert "mma884_online" in bench
    assert "compute_path" in bench


def test_sm70_dense_mma884_online_has_runtime_k_tile_dispatch():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "config.h").read_text()
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "MMA_884_ONLINE_AUTO_K_TILE = 0" in config_h
    assert "FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE 0" in config_h
    assert "select_dense_mma884_online_k_tile" in sm70_dense_cuh
    assert "launch_flash_splitkv_mla_sm70_dense_online<InputT, 16>" in sm70_dense_cuh
    assert "launch_flash_splitkv_mla_sm70_dense_online<InputT, 32>" in sm70_dense_cuh
    assert "launch_flash_splitkv_mla_sm70_dense_online<InputT, 64>" in sm70_dense_cuh
    assert "runtime_online_k_tile" in bench
    assert "sm70_alpha_runtime_online_k_tile" in bench
    assert "SM70_ALPHA_MMA_884_ONLINE_K_TILE == 0" in bench


def test_sm70_dense_mma884_online_has_register_output_accumulator_path():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "config.h").read_text()
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "MMA_884_REGISTER_OUTPUT_CTA_THREADS = 256" in config_h
    assert "accumulate_dense_output_mma884_online_register" in sm70_dense_cuh
    assert "write_dense_mma884_register_output" in sm70_dense_cuh
    assert "scale_dense_mma884_register_fragments" in sm70_dense_cuh
    assert (
        "if constexpr (dense::USE_MMA_884_ONLINE && "
        "dense::NUM_THREADS == dense::MMA_884_REGISTER_OUTPUT_CTA_THREADS)"
    ) in sm70_dense_cuh
    assert "output_accum + dense::HEAD_DIM_V" in sm70_dense_cuh
    register_path = sm70_dense_cuh.split(
        "__device__ __forceinline__ void accumulate_dense_output_mma884_online_register", 1
    )[1]
    register_path = register_path.split("__device__ __forceinline__ float block_reduce_max", 1)[0]
    assert "output_accum + dense::HEAD_DIM_V" not in register_path
    assert "SM70_ALPHA_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS = 256" in bench
    assert "SM70_ALPHA_CTA_THREADS == SM70_ALPHA_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS" in bench
    assert "+ 512 * 4" not in bench.split("def sm70_alpha_mma884_online_shared_bytes", 1)[1].split("return", 2)[1]


def test_sm70_dense_mma884_online_softmax_uses_parallel_tile_reduction():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "config.h").read_text()
    sm70_dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    common_softmax_h = (REPO_ROOT / "csrc" / "sm70" / "common" / "softmax.h").read_text()
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "MMA_884_ONLINE_SCALAR_COUNT = 5" in config_h
    assert "MMA_884_ONLINE_REDUCE_SCRATCH = NUM_THREADS / 32" in config_h
    assert "block_reduce_max_warp_scratch" in common_softmax_h
    assert "block_reduce_sum_warp_scratch" in common_softmax_h
    assert "compute_online_softmax_tile_parallel" in common_softmax_h
    assert '"sm70/common/softmax.h"' in sm70_dense_cuh
    assert "flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<false>" in sm70_dense_cuh
    assert "__device__ __forceinline__ void compute_dense_online_softmax_tile_parallel" not in sm70_dense_cuh
    assert "__device__ __forceinline__ float block_reduce_max_warp_scratch" not in sm70_dense_cuh
    assert "__device__ __forceinline__ float block_reduce_sum_warp_scratch" not in sm70_dense_cuh
    assert "online_reduce_scratch" in sm70_dense_cuh
    assert "online_scalars + dense::MMA_884_ONLINE_SCALAR_COUNT" in sm70_dense_cuh
    assert "SM70_ALPHA_MMA_884_ONLINE_REDUCE_SCRATCH" in bench
    assert "SM70_ALPHA_MMA_884_ONLINE_REDUCE_SCRATCH * 4" in bench


def test_sm70_dense_mma884_online_docs_record_register_accum_tuning():
    design_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-design.md").read_text()
    task_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md").read_text()

    assert "register output accumulator" in design_doc
    assert "parallel tile softmax reduction" in design_doc
    assert "113 registers、0 spill" in design_doc
    assert "K_TILE=64" in design_doc
    assert "K_TILE=0` runtime-auto" in design_doc
    assert "685.739/652.629 us" in design_doc
    assert "349.525/383.317 us" in design_doc
    assert "默认 scalar score-cache" in design_doc

    assert "register output accumulator" in task_doc
    assert "parallel tile softmax reduction" in task_doc
    assert "113 registers、0 spill" in task_doc
    assert "K_TILE=64" in task_doc
    assert "K_TILE=0` runtime-auto" in task_doc
    assert "633.728/693.504 us" in task_doc
    assert "400.672/476.864 us" in task_doc
    assert "685.739/652.629 us" in task_doc
    assert "349.525/383.317 us" in task_doc
    assert "- [x] 继续推进 dense 的 MMA_884 默认路径选择。" in task_doc


def test_public_python_imports_match_interface_exports():
    init_py = (REPO_ROOT / "flash_mla" / "__init__.py").read_text()
    interface_py = (REPO_ROOT / "flash_mla" / "flash_mla_interface.py").read_text()

    assert "flash_mla_with_kvcache_fp8" in init_py
    assert "def flash_mla_with_kvcache_fp8" in interface_py


def test_dense_decode_test_suite_has_sm70_alpha_subset():
    dense_test = (REPO_ROOT / "tests" / "test_flash_mla_dense_decoding.py").read_text()

    assert "cc_major == 7" in dense_test
    assert "SM70 dense MLA decoding currently requires --dtype fp16" in dense_test
    assert "for d in [512, 576]" in dense_test


def test_sm70_dense_benchmark_script_is_present():
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert '"scalar_online": 64' in bench
    assert '"scalar_online": 0' in bench
    assert "SM70_ALPHA_H_TILE = int(os.getenv(\"FLASH_MLA_SM70_H_TILE\", \"4\"))" in bench
    assert "SM70_ALPHA_CTA_THREADS = int(os.getenv(\"FLASH_MLA_SM70_CTA_THREADS\", \"256\"))" in bench
    assert "SM70_ALPHA_COMPUTE_PATH" in bench
    assert "--cases" in bench
    assert '"tile"' in bench
    assert "h_q = 8" in bench
    assert "torch.float16" in bench


def test_sm70_dense_benchmark_long_cases_cover_acceptance_seqlens():
    bench = (REPO_ROOT / "benchmark" / "bench_sm70_dense_decode.py").read_text()

    assert "seq_lens = [4096, 8192, 16384]" in bench


def test_sm70_vllm_integration_artifacts_capture_current_gate_and_api_contract():
    integration_doc = (REPO_ROOT / SM70_VLLM_INTEGRATION_DOC).read_text()
    bench = (REPO_ROOT / SM70_VLLM_BENCH).read_text()

    assert "/mnt/data/apps/vllm" in integration_doc
    assert "DeepseekV4FlashMLASparseBackend" in integration_doc
    assert "flash_mla_with_kvcache" in integration_doc
    assert "flash_mla_sparse_fwd" in integration_doc
    assert "supports_compute_capability" in integration_doc
    assert "major in [9, 10]" in integration_doc
    assert "is_device_capability_family(90)" in integration_doc
    assert "fp8_ds_mla" in integration_doc
    assert "584B" in integration_doc
    assert "sparse prefill" in integration_doc

    assert "mode" in bench
    assert "inspect" in bench
    assert "openai-stream" in bench
    assert "TTFT" in bench
    assert "finish_reason" in bench
    assert "decode_tokens_per_s" in bench
    assert "DeepseekV4FlashMLASparseBackend" in bench
    assert "sm70_blockers" in bench
    assert "end_to_end_smoke_ready" in bench
    assert "vllm_flashmla_sparse_backend_sm70_gate" in bench
    assert "vllm_flashmla_sparse_runtime_sm70_gate" in bench
    assert "FlashMLA SM70 currently supports sparse decode and sparse prefill" in bench
    assert "- [x] 若 vLLM 还被 FlashInfer、MoE、量化或 scheduler 阻断，将其拆成独立任务" in (
        REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md"
    ).read_text()
    assert "vLLM SM70 gate 放开任务" in integration_doc
    assert "flash_mla_sparse_fwd" in bench


def test_sparse_prefill_dispatches_sm70_bf16_path():
    sparse_fwd_h = (REPO_ROOT / "csrc" / "api" / "sparse_fwd.h").read_text()

    assert "arch.is_sm70()" in sparse_fwd_h
    assert "sm70/prefill/sparse/fwd.h" in sparse_fwd_h
    assert "sm70::prefill::sparse::run_fwd_kernel" in sparse_fwd_h
    assert "SM70 sparse prefill BF16 path" in sparse_fwd_h
    assert "SM70 sparse prefill is not supported" not in sparse_fwd_h
    setup_py = (REPO_ROOT / "setup.py").read_text()
    assert "csrc/sm70/prefill/sparse/instantiations/bf16.cu" in setup_py


def test_sm70_sparse_prefill_uses_parallel_score_fast_path():
    fwd_cuh = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "fwd.cuh").read_text()

    assert "sm70_sparse_prefill_fast_fwd_kernel" in fwd_cuh
    assert "fill_scores_warp_parallel" in fwd_cuh
    assert "for (int tile_start = 0; tile_start < params.topk; tile_start += K_TILE)" in fwd_cuh
    assert "const int warp_idx = threadIdx.x / 32" in fwd_cuh
    assert "for (int tile_base = 0; tile_base < tile_count; tile_base += warps_per_cta)" in fwd_cuh
    assert "compute_lse_deterministic" in fwd_cuh


def test_sm70_sparse_prefill_benchmark_script_is_present():
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert "SM70_SPARSE_PREFILL_REGISTERS_BY_QK_PATH_AND_CTA_THREADS" in bench
    assert "(\"warp_simt_qk\", 128): 32" in bench
    assert "(\"warp_simt_qk\", 256): 29" in bench
    assert "(\"mma884_qk\", 256): 40" in bench
    assert "SM70_SPARSE_PREFILL_SPILLS = 0" in bench
    assert "SM70_SPARSE_PREFILL_CTA_THREADS = int(os.getenv(\"FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS\", \"256\"))" in bench
    assert "SM70_SPARSE_PREFILL_MAX_TOPK = 8192" in bench
    assert "torch.bfloat16" in bench
    assert "--cases" in bench
    assert '"quick"' in bench
    assert '"max_topk"' in bench
    assert 'elif name == "max_topk"' in bench
    assert "SM70_SPARSE_PREFILL_MAX_TOPK," in bench
    assert "have_attn_sink" in bench
    assert "have_topk_length" in bench
    assert "max_abs_max_logits" in bench


def test_sm70_sparse_prefill_thread_config_is_build_time_tunable():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "config.h").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS" in config_h
    assert "NUM_THREADS = FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS" in config_h
    assert "os.getenv(\"FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS\"" in bench


def test_sm70_sparse_prefill_stages_kv_in_fp16_shared_tiles():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "config.h").read_text()
    fwd_cuh = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "fwd.cuh").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_PREFILL_K_TILE=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE" in config_h
    assert "K_TILE = FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE" in config_h
    assert "stage_kv_tile_to_shared" in fwd_cuh
    assert "cutlass::half_t *kv_tile" in fwd_cuh
    assert "shared_k_value" in fwd_cuh
    assert "qk_score_token_from_shared" in fwd_cuh
    assert "sm70_sparse_prefill_shared_bytes" in fwd_cuh
    assert "SM70_SPARSE_PREFILL_K_TILE = int(os.getenv(\"FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE\", \"32\"))" in bench
    assert "SM70_SPARSE_PREFILL_STAGING_BYTES_PER_VALUE = 2" in bench


def test_sm70_sparse_prefill_scores_use_warp_parallel_qk_reduction():
    fwd_cuh = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "fwd.cuh").read_text()

    assert "warp_reduce_sum" in fwd_cuh
    assert "qk_score_token_from_shared_warp" in fwd_cuh
    assert "fill_scores_warp_parallel" in fwd_cuh
    assert "const int warps_per_cta = blockDim.x / 32" in fwd_cuh
    assert "for (int dim = lane; dim < HEAD_DIM_QK; dim += 32)" in fwd_cuh


def test_sm70_sparse_prefill_has_opt_in_mma884_qk_scaffold():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "config.h").read_text()
    fwd_cuh = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "fwd.cuh").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK" in config_h
    assert "USE_MMA_884_QK = FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK != 0" in config_h
    assert "sm70/common/mma_884.h" in fwd_cuh
    assert "fill_scores_mma884_qk" in fwd_cuh
    assert "mma_m8n8k4_row_col" in fwd_cuh
    assert "store_mma884_qk_row0_scores" in fwd_cuh
    assert "if constexpr (USE_MMA_884_QK)" in fwd_cuh
    assert "SM70_SPARSE_PREFILL_USE_MMA_884_QK" in bench
    assert "SM70_SPARSE_PREFILL_QK_PATH" in bench
    assert "mma884_qk" in bench


def test_sm70_sparse_prefill_has_opt_in_mma884_pv_scaffold():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "config.h").read_text()
    fwd_cuh = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "fwd.cuh").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV" in config_h
    assert "USE_MMA_884_PV = FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV != 0" in config_h
    assert "load_mma884_p_fragment" in fwd_cuh
    assert "load_mma884_v_fragment" in fwd_cuh
    assert "store_mma884_pv_row0_output" in fwd_cuh
    assert "accumulate_output_mma884_pv" in fwd_cuh
    assert "mma_m8n8k4_row_row" in fwd_cuh
    assert "if constexpr (USE_MMA_884_PV)" in fwd_cuh
    assert "SM70_SPARSE_PREFILL_USE_MMA_884_PV" in bench
    assert "SM70_SPARSE_PREFILL_PV_PATH" in bench
    assert "mma884_qk+mma884_pv" in bench


def test_sm70_sparse_prefill_has_opt_in_mma884_online_scaffold():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "config.h").read_text()
    fwd_cuh = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "fwd.cuh").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE" in config_h
    assert "USE_MMA_884_ONLINE = FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE != 0" in config_h
    assert "compute_online_softmax_tile" in fwd_cuh
    assert "scale_online_output_accumulator" in fwd_cuh
    assert "accumulate_output_mma884_online" in fwd_cuh
    assert "store_online_output" in fwd_cuh
    assert "if constexpr (USE_MMA_884_ONLINE)" in fwd_cuh
    assert "HEAD_DIM_V * sizeof(float)" in fwd_cuh
    assert "SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE" in bench
    assert "SM70_SPARSE_PREFILL_ONLINE_PATH" in bench
    assert "mma884_online" in bench


def test_sm70_sparse_prefill_mma884_online_is_default_path():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "prefill" / "sparse" / "config.h").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_PREFILL_BENCH).read_text()

    assert 'sparse_prefill_use_mma884_online = get_env_int("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE", 1, {0, 1})' in setup_py
    assert "#define FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE 1" in config_h
    assert (
        'SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE = int(os.getenv("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE", "1")) != 0'
        in bench
    )


def test_sm70_sparse_decode_scores_use_warp_parallel_qk_reduction():
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()

    assert "warp_reduce_sum" in sparse_cuh
    assert "qk_score_token_from_shared_warp" in sparse_cuh
    assert "fill_scores_warp_parallel" in sparse_cuh
    assert "const int warps_per_cta = blockDim.x / 32" in sparse_cuh
    assert "for (int dim = lane; dim < head_dim_qk<MODEL_TYPE>(); dim += 32)" in sparse_cuh


def test_sm70_sparse_decode_has_opt_in_mma884_qk_scaffold():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    common_attention = (REPO_ROOT / "csrc" / "sm70" / "common" / "mma_884_attention.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK" in config_h
    assert "USE_MMA_884_QK = FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK != 0" in config_h
    assert "sm70/common/mma_884.h" in common_attention
    assert "sm70/common/mma_884_attention.h" in sparse_cuh
    assert "fill_scores_mma884_qk" in sparse_cuh
    assert "compute_mma884_qk_group" in sparse_cuh
    assert "store_mma884_qk_row0_scores" in sparse_cuh
    assert "mma_m8n8k4_row_col" in common_attention
    assert "mma884_accumulate_qk" in sparse_cuh
    assert "if constexpr (USE_MMA_884_QK)" in sparse_cuh
    assert "SM70_SPARSE_DECODE_USE_MMA_884_QK" in bench
    assert "SM70_SPARSE_DECODE_QK_PATH" in bench
    assert "SM70_SPARSE_DECODE_MMA884_QK_REGISTERS_BY_CTA_THREADS_AND_MODEL" in bench
    assert '(256, "v32"): 80' in bench
    assert '(256, "model1"): 80' in bench
    assert "qk_path" in bench
    assert "mma884_qk" in bench


def test_sm70_sparse_decode_has_opt_in_mma884_pv_scaffold():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV" in config_h
    assert "USE_MMA_884_PV = FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV != 0" in config_h
    assert "probability_half_from_score" in sparse_cuh
    assert "load_mma884_p_fragment" in sparse_cuh
    assert "load_mma884_v_fragment" in sparse_cuh
    assert "compute_mma884_pv_group" in sparse_cuh
    assert "accumulate_output_mma884_pv" in sparse_cuh
    assert "if constexpr (USE_MMA_884_PV)" in sparse_cuh
    assert "SM70_SPARSE_DECODE_USE_MMA_884_PV" in bench
    assert "SM70_SPARSE_DECODE_PV_PATH" in bench
    assert "SM70_SPARSE_DECODE_MMA884_QK_PV_REGISTERS_BY_CTA_THREADS_AND_MODEL" in bench
    assert "SM70_SPARSE_DECODE_MMA884_QK_PV_SPILLS_BY_CTA_THREADS_AND_MODEL" in bench
    assert '(128, "v32"): 95' in bench
    assert '(128, "model1"): 96' in bench
    assert '(256, "v32"): 64' in bench
    assert '(256, "model1"): 64' in bench
    assert '(256, "v32"): 20' in bench
    assert '(256, "model1"): 28' in bench
    assert "pv_path" in bench
    assert "mma884_pv" in bench


def test_sm70_sparse_decode_has_opt_in_mma884_online_scaffold():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE" in setup_py
    assert "-DFLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE" in config_h
    assert "USE_MMA_884_ONLINE = FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE != 0" in config_h
    assert "probability_half_from_online_score" in sparse_cuh
    assert "load_mma884_online_p_fragment" in sparse_cuh
    assert "compute_online_softmax_tile" in sparse_cuh
    assert "scale_online_output_accumulator" in sparse_cuh
    assert "accumulate_output_mma884_online" in sparse_cuh
    assert "write_online_output" in sparse_cuh
    assert "if constexpr (USE_MMA_884_ONLINE)" in sparse_cuh
    assert "SM70_SPARSE_DECODE_USE_MMA_884_ONLINE" in bench
    assert "SM70_SPARSE_DECODE_ONLINE_PATH" in bench
    assert "SM70_SPARSE_DECODE_COMPUTE_PATH" in bench
    assert "mma884_online" in bench
    assert "online_path" in bench
    assert "compute_path" in bench
    online_registers = bench.split(
        "SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTERS_BY_CTA_THREADS_AND_MODEL = {", 1
    )[1].split("}", 1)[0]
    online_spills = bench.split(
        "SM70_SPARSE_DECODE_MMA884_ONLINE_SPILLS_BY_CTA_THREADS_AND_MODEL = {", 1
    )[1].split("}", 1)[0]
    assert '(128, "v32"): 72' in online_registers
    assert '(128, "model1"): 80' in online_registers
    assert '(256, "v32"): 80' in online_registers
    assert '(256, "model1"): 97' in online_registers
    assert '(128, "v32"): 4' in online_spills
    assert '(128, "model1"): 8' in online_spills
    assert '(256, "v32"): 0' in online_spills
    assert '(256, "model1"): 0' in online_spills


def test_sm70_sparse_decode_mma884_online_uses_register_accumulator_and_parallel_softmax():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    common_softmax_h = (REPO_ROOT / "csrc" / "sm70" / "common" / "softmax.h").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "MMA_884_REGISTER_OUTPUT_CTA_THREADS = 256" in config_h
    assert "MMA_884_ONLINE_SCALAR_COUNT = 5" in config_h
    assert "MMA_884_ONLINE_REDUCE_SCRATCH = NUM_THREADS / 32" in config_h
    assert "accumulate_output_mma884_online_register" in sparse_cuh
    assert "write_mma884_online_register_output" in sparse_cuh
    assert "scale_mma884_online_register_fragments" in sparse_cuh
    assert "block_reduce_max_warp_scratch" in common_softmax_h
    assert "block_reduce_sum_warp_scratch" in common_softmax_h
    assert "compute_online_softmax_tile_parallel" in common_softmax_h
    assert '"sm70/common/softmax.h"' in sparse_cuh
    assert "flash_mla::sm70::softmax::compute_online_softmax_tile_parallel<true>" in sparse_cuh
    assert "__device__ __forceinline__ void compute_online_softmax_tile_parallel" not in sparse_cuh
    assert "__device__ __forceinline__ float block_reduce_max_warp_scratch" not in sparse_cuh
    assert "__device__ __forceinline__ float block_reduce_sum_warp_scratch" not in sparse_cuh
    assert "if constexpr (USE_MMA_884_ONLINE)" in sparse_cuh
    assert "if constexpr (use_mma884_online_register_accumulator<MODEL_TYPE>())" in sparse_cuh
    register_path = sparse_cuh.split(
        "__device__ __forceinline__ void accumulate_output_mma884_online_register", 1
    )[1]
    register_path = register_path.split(
        "__host__ __device__ __forceinline__ size_t sparse_decode_shared_bytes", 1
    )[0]
    assert "output_accum + HEAD_DIM_V" not in register_path
    assert "SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS = 256" in bench
    online_shared = bench.split("def sm70_sparse_decode_shared_bytes", 1)[1].split(
        "extra_topk = case.decode.extra_topk or 0", 1
    )[0]
    register_shared = online_shared.split(
        "if SM70_SPARSE_DECODE_CTA_THREADS != SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS:",
        1,
    )[0]
    assert "+ case.d_v * 4" not in register_shared
    assert "shared_bytes += case.d_v * 4" in online_shared
    assert "SM70_SPARSE_DECODE_MMA_884_ONLINE_REDUCE_SCRATCH * 4" in online_shared


def test_sm70_sparse_decode_mma884_online_register_accumulator_is_model_specific():
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "use_mma884_online_register_accumulator" in sparse_cuh
    assert "MODEL_TYPE == ModelType::MODEL1" in sparse_cuh
    assert "if constexpr (use_mma884_online_register_accumulator<MODEL_TYPE>())" in sparse_cuh
    assert "SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_MODEL = \"model1\"" in bench
    assert "sm70_sparse_decode_uses_register_accumulator" in bench
    assert "model_label(case) == SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_MODEL" in bench

    online_shared = bench.split("def sm70_sparse_decode_shared_bytes", 1)[1].split(
        "extra_topk = case.decode.extra_topk or 0", 1
    )[0]
    assert "if not sm70_sparse_decode_uses_register_accumulator(case):" in online_shared
    assert "shared_bytes += case.d_v * 4" in online_shared


def test_sm70_sparse_decode_mma884_online_uses_runtime_adaptive_k_tile():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()
    task_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md").read_text()

    assert "MMA_884_ONLINE_LONG_K_TILE = 16" in config_h
    assert "MMA_884_ONLINE_LONG_K_TILE_THRESHOLD = 512" in config_h
    assert "template<ModelType MODEL_TYPE, int ONLINE_K_TILE>" in sparse_cuh
    assert "tile_start += ONLINE_K_TILE" in sparse_cuh
    assert "min(ONLINE_K_TILE, local_score_count - tile_start)" in sparse_cuh
    assert "MODEL_TYPE == ModelType::V32" in sparse_cuh
    assert "params.topk + params.extra_topk >= MMA_884_ONLINE_LONG_K_TILE_THRESHOLD" in sparse_cuh
    assert "flash_fwd_splitkv_mla_sm70_sparse_kernel<MODEL_TYPE, NUM_HEADS, MMA_884_ONLINE_LONG_K_TILE>" in sparse_cuh
    assert "sparse_decode_shared_bytes<MODEL_TYPE, MMA_884_ONLINE_LONG_K_TILE>" in sparse_cuh
    assert "sparse_decode_shared_bytes<MODEL_TYPE, SPARSE_K_TILE>" in sparse_cuh
    assert "SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE = 16" in bench
    assert "SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE_THRESHOLD = 512" in bench
    assert "def sm70_sparse_decode_runtime_k_tile" in bench
    assert 'model_label(case) == "v32"' in bench
    assert "runtime_k_tile" in bench
    assert "runtime adaptive K tile" in task_doc


def test_sm70_sparse_decode_mma884_online_is_default_path():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()
    design_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-design.md").read_text()
    task_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md").read_text()

    assert (
        'sparse_decode_use_mma884_online = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE", 1, {0, 1})'
        in setup_py
    )
    assert "#define FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE 1" in config_h
    assert 'os.getenv("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE", "1")' in bench
    assert "默认路径已切到 `256/32/mma884_online`" in design_doc
    assert "默认切到 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` + `K_TILE=32`" in task_doc


def test_sm70_sparse_decode_has_batch2_verify_fast_path_gate():
    setup_py = (REPO_ROOT / "setup.py").read_text()
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_decode_h = (REPO_ROOT / "csrc" / "api" / "sparse_decode.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()

    assert "FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY" in setup_py
    assert (
        'sparse_decode_batch2_verify = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY", 1, {0, 1})'
        in setup_py
    )
    assert "-DFLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY=" in setup_py
    assert "#ifndef FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY" in config_h
    assert "USE_BATCH2_VERIFY = FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY != 0" in config_h
    assert "run_flash_splitkv_mla_fp8_sparse_batch2_verify_kernel" in sparse_decode_h
    assert "can_use_sm70_batch2_verify_fast_path" in sparse_decode_h
    assert "params.b == 2" in sparse_decode_h
    assert "params.s_q == 1" in sparse_decode_h
    assert "params.model_type == ModelType::MODEL1" in sparse_decode_h
    assert "DecodeFeatures::TOPK_LENGTH" in sparse_decode_h
    assert "DecodeFeatures::EXTRA_KVCACHE" in sparse_decode_h
    assert "DecodeFeatures::EXTRA_TOPK_LENGTH" in sparse_decode_h
    assert "topk_rows_match" in sparse_cuh
    assert "flash_fwd_splitkv_mla_sm70_sparse_batch2_verify_kernel" in sparse_cuh
    assert "process_sparse_partition_batch2_verify" in sparse_cuh
    assert "fill_tile_scores_mma884_qk_batch2_half_q<MODEL_TYPE>(" in sparse_cuh
    assert "compute_mma884_online_pv_group_cached_batch2<MODEL_TYPE>(" in sparse_cuh
    assert "scale_mma884_online_register_fragments_batch2(" in sparse_cuh
    assert "write_mma884_online_register_output_batch2(" in sparse_cuh
    batch2_body = sparse_cuh.split(
        "void accumulate_output_mma884_online_register_batch2_verify", 1
    )[1].split(
        "template<ModelType MODEL_TYPE, int ONLINE_K_TILE>\n__device__ __forceinline__ void accumulate_output_mma884_online_register_double_buffer",
        1,
    )[0]
    assert "fill_tile_scores_mma884_qk_batch2_half_q<MODEL_TYPE>(" in batch2_body
    assert "compute_mma884_online_pv_group_cached_batch2<MODEL_TYPE>(" in batch2_body
    assert "scale_mma884_online_register_fragments_batch2(" in batch2_body
    assert "write_mma884_online_register_output_batch2(" in batch2_body
    assert "fill_tile_scores_mma884_qk_half_q<MODEL_TYPE>" not in batch2_body


def test_sm70_decode_mma884_qk_pv_core_is_shared_between_dense_and_sparse():
    common_attention = (REPO_ROOT / "csrc" / "sm70" / "common" / "mma_884_attention.h").read_text()
    dense_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "dense" / "splitkv_mla_sm70.cuh"
    ).read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()

    assert "mma884_accumulate_qk" in common_attention
    assert "mma884_accumulate_pv" in common_attention
    assert "store_mma884_row0_scores" in common_attention
    assert "load_mma884_p_fragment_from_scores" in common_attention
    assert "add_mma884_row0_output_fragment" in common_attention

    assert '"sm70/common/mma_884_attention.h"' in dense_cuh
    assert '"sm70/common/mma_884_attention.h"' in sparse_cuh
    for source in (dense_cuh, sparse_cuh):
        assert "flash_mla::sm70::attention::mma884_accumulate_qk" in source
        assert "flash_mla::sm70::attention::mma884_accumulate_pv" in source
        assert "flash_mla::sm70::attention::store_mma884_row0_scores" in source
        assert "flash_mla::sm70::attention::load_mma884_p_fragment_from_scores" in source


def test_sm70_sparse_decode_docs_record_register_online_accumulator_tuning():
    design_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-design.md").read_text()
    task_doc = (REPO_ROOT / "docs" / "sm70-volta-flashmla-tasks.md").read_text()

    assert "model-specific hybrid accumulator" in design_doc
    assert "register output accumulator" in design_doc
    assert "parallel online softmax reduction" in design_doc
    assert "V32 `80 registers、0 spill`" in design_doc
    assert "97 registers、0 spill" in design_doc
    assert "39220/33076" in design_doc
    assert "233.473/408.952/237.247/858.323 us" in design_doc
    assert "3667.897/7065.823 us" in design_doc

    assert "model-specific hybrid accumulator" in task_doc
    assert "register output accumulator" in task_doc
    assert "parallel online softmax reduction" in task_doc
    assert "V32 `80 registers、0 spill`" in task_doc
    assert "97 registers、0 spill" in task_doc
    assert "39220/33076" in task_doc
    assert "233.473/408.952/237.247/858.323 us" in task_doc
    assert "3667.897/7065.823 us" in task_doc


def test_sm70_benchmarks_report_theoretical_occupancy():
    benchmark_paths = [
        "benchmark/bench_sm70_dense_decode.py",
        SM70_SPARSE_DECODE_BENCH,
        SM70_SPARSE_PREFILL_BENCH,
    ]

    for benchmark_path in benchmark_paths:
        bench = (REPO_ROOT / benchmark_path).read_text()

        assert "SM70_REGISTERS_PER_SM = 65536" in bench
        assert "SM70_MAX_WARPS_PER_SM = 64" in bench
        assert "sm70_theoretical_occupancy" in bench
        assert "active_blocks_per_sm,theoretical_occupancy_pct" in bench


def test_sm70_sparse_decode_benchmark_covers_hq64_and_hq128_paths():
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "RawTestParam" in bench
    assert "h_q=64" in bench
    assert "h_q=128" in bench
    assert "d_qk=576" in bench
    assert "d_qk=512" in bench
    assert "have_topk_length=True" in bench
    assert "extra_s_k=" in bench
    assert "max_abs_lse" in bench
    assert "splitkv_us" in bench
    assert "SM70_SPARSE_DECODE_REGISTERS_BY_CTA_THREADS_AND_MODEL" in bench
    assert '(256, "v32"): 64' in bench
    assert '(256, "model1"): 64' in bench
    assert "SM70_SPARSE_DECODE_SPILLS_BY_CTA_THREADS_AND_MODEL" in bench
    assert '(256, "model1"): 0' in bench
    assert "sm70_sparse_decode_shared_bytes" in bench
    assert "bench_kineto" in bench
    assert "flush=True" in bench


def test_sm70_sparse_decode_mirrors_no_split_rows_to_accumulators():
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()

    assert "oaccum_ptr_for" in sparse_cuh
    assert "lseaccum_ptr_for" in sparse_cuh
    assert "write_no_split_accumulators" in sparse_cuh
    assert "row_lse * CUDART_L2E_F" in sparse_cuh
    assert "raw_acc" in sparse_cuh
    assert "sink_scale" in sparse_cuh


def test_sm70_sparse_decode_stages_kv_in_fp16_shared_tiles():
    config_h = (REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "config.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()
    bench = (REPO_ROOT / SM70_SPARSE_DECODE_BENCH).read_text()

    assert "#define FLASH_MLA_SM70_SPARSE_DECODE_K_TILE 32" in config_h
    assert "SPARSE_K_TILE = FLASH_MLA_SM70_SPARSE_DECODE_K_TILE" in config_h
    assert "stage_kv_tile_to_shared" in sparse_cuh
    assert "cutlass::half_t* kv_tile" in sparse_cuh
    assert "shared_k_value" in sparse_cuh
    assert "static_cast<float>(kv_tile[" in sparse_cuh
    assert "cutlass::half_t(load_k_value" in sparse_cuh
    assert "qk_score_token_from_shared" in sparse_cuh
    assert "detail::sparse_decode_shared_bytes" in sparse_cuh
    assert "os.getenv(\"FLASH_MLA_SM70_SPARSE_DECODE_K_TILE\", \"32\")" in bench
    assert "SM70_SPARSE_DECODE_STAGING_BYTES_PER_VALUE = 2" in bench


def test_sm70_sparse_decode_uses_scheduler_for_splitk_accumulators():
    sparse_decode_h = (REPO_ROOT / "csrc" / "api" / "sparse_decode.h").read_text()
    sparse_cuh = (
        REPO_ROOT / "csrc" / "sm70" / "decode" / "sparse_fp8" / "splitkv_mla_sm70_sparse.cuh"
    ).read_text()

    sm70_impl = sparse_decode_h.split("class Decode_Sm70_Fp8_Dequant_Impl", 1)[1].split(
        "class Decode_Sm90_Impl", 1
    )[0]
    assert "Arch arch = Arch();" in sm70_impl
    assert "arch.num_sms" in sm70_impl
    assert "num_sm_parts" in sm70_impl
    assert "return {\n            1," not in sm70_impl

    assert "partition_idx = blockIdx.z" in sparse_cuh
    assert "tile_scheduler_metadata_ptr[partition_idx]" in sparse_cuh
    assert "process_sparse_partition" in sparse_cuh
    assert "write_split_accumulators" in sparse_cuh
    assert "is_no_split" in sparse_cuh
    assert "dim3(params.s_q, params.h_q, params.num_sm_parts)" in sparse_cuh


def test_readme_documents_sm70_preview_support_matrix():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "SM70 preview" in readme
    assert "FP16 only" in readme
    assert "FLASH_MLA_ENABLE_SM70=1" in readme


def test_sm70_common_headers_compile_for_sm70(tmp_path):
    nvcc = shutil.which("nvcc")
    assert nvcc is not None, "nvcc is required to validate the SM70 header contract"

    source = tmp_path / "sm70_common_contract.cu"
    source.write_text(
        """
#include <cuda_fp16.h>
#include <stdint.h>

#include "csrc/sm70/common/mma_884.h"
#include "csrc/sm70/common/layout.h"
#include "csrc/sm70/common/load_store.h"
#include "csrc/sm70/common/softmax.h"
#include "csrc/sm70/common/bf16_cast.h"
#include "csrc/sm70/common/fp8_dequant.h"

__global__ void sm70_common_contract_kernel(
    half* out,
    const uint8_t* fp8,
    const uint16_t* bf16,
    const float* scales) {
  using namespace flash_mla::sm70;

  Array<half, 4> a{};
  Array<half, 4> b{};
  Array<float, 8> c{};
  Array<float, 8> d{};
  mma_m8n8k4_row_col(d, a, b, c);
  mma_m8n8k4_row_row(c, a, b, d);

  const int lane = lane_id();
  const int qp = quadpair_id_from_lane(lane);
  const int rank = quadpair_rank_from_lane(lane);

  const int offset = layout::padded_offset(qp, rank, 16, 4);
  const int swizzled = layout::swizzle_128b_offset(offset);
  (void)swizzled;

  const int4 raw = load_store::load_128b<int4>(fp8);
  (void)raw;

  softmax::OnlineSoftmax<4> online;
  online.reset();
  online.update(qp, static_cast<float>(rank));

  Array<half, 4> dequant{};
  fp8_dequant::dequant_e4m3_group<4>(dequant, fp8, scales[0]);

  out[0] = bf16_cast::bf16_bits_to_half(bf16[0]);
  out[1] = fp8_dequant::e4m3_to_half(fp8[0], scales[0]);
  out[2] = dequant[0];
  out[3] = __float2half_rn(online.lse(qp));
}
""",
        encoding="utf-8",
    )

    output = tmp_path / "sm70_common_contract.o"
    subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-I",
            str(REPO_ROOT),
            "-gencode",
            "arch=compute_70,code=sm_70",
            "--expt-relaxed-constexpr",
            "-c",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
