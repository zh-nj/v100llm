import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
    CUDA_HOME
)


this_dir = os.path.dirname(os.path.abspath(__file__))


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_env_int(name: str, default: int, allowed: set[int]) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    value = int(raw_value)
    assert value in allowed, f"{name} must be one of {sorted(allowed)}, got {value}"
    return value


def ensure_cutlass_checkout():
    cutlass_sentinel = Path(this_dir) / "csrc/cutlass/include/cutlass/bfloat16.h"
    if cutlass_sentinel.is_file():
        print("Using existing local CUTLASS checkout")
        return
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

def get_features_args():
    features_args = []
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    if is_flag_set("FLASH_MLA_ENABLE_SM70"):
        features_args.append("-DKERUTILS_ALLOW_SM70_STUB_COMPILE")
    return features_args


def get_sm70_tuning_args():
    h_tile = get_env_int("FLASH_MLA_SM70_H_TILE", 4, {4, 8, 16})
    cta_threads = get_env_int("FLASH_MLA_SM70_CTA_THREADS", 256, {128, 256})
    dense_use_mma884_online = get_env_int("FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE", 0, {0, 1})
    dense_mma884_online_k_tile = get_env_int("FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE", 0, {0, 16, 32, 64})
    sparse_decode_cta_threads = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS", 256, {128, 256})
    sparse_decode_k_tile = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_K_TILE", 32, {16, 32, 64})
    sparse_decode_use_mma884_qk = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK", 0, {0, 1})
    sparse_decode_use_mma884_pv = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV", 0, {0, 1})
    sparse_decode_use_mma884_online = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE", 1, {0, 1})
    sparse_decode_kv_row_pad_half = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_KV_ROW_PAD_HALF", 4, {0, 4, 8})
    sparse_decode_s0c3_ablation = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION", 0, {0, 1, 2, 3, 4})
    sparse_decode_fast_e4m3 = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_FAST_E4M3", 1, {0, 1})
    sparse_decode_fast_bf16_round = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND", 1, {0, 1})
    sparse_decode_pv_p_cache = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_PV_P_CACHE", 1, {0, 1})
    sparse_decode_v_fragment_vector = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_V_FRAGMENT_VECTOR", 0, {0, 1})
    sparse_decode_batch2_verify = get_env_int("FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY", 1, {0, 1})
    sparse_prefill_cta_threads = get_env_int("FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS", 256, {128, 256})
    sparse_prefill_k_tile = get_env_int("FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE", 32, {16, 32})
    sparse_prefill_use_mma884_qk = get_env_int("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK", 0, {0, 1})
    sparse_prefill_use_mma884_pv = get_env_int("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV", 0, {0, 1})
    sparse_prefill_use_mma884_online = get_env_int("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE", 1, {0, 1})
    if sparse_decode_use_mma884_online:
        sparse_decode_use_mma884_qk = 1
        sparse_decode_use_mma884_pv = 1
    if sparse_prefill_use_mma884_online:
        sparse_prefill_use_mma884_qk = 1
        sparse_prefill_use_mma884_pv = 1
    return [
        f"-DFLASH_MLA_SM70_H_TILE={h_tile}",
        f"-DFLASH_MLA_SM70_CTA_THREADS={cta_threads}",
        f"-DFLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE={dense_use_mma884_online}",
        f"-DFLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE={dense_mma884_online_k_tile}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS={sparse_decode_cta_threads}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_K_TILE={sparse_decode_k_tile}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK={sparse_decode_use_mma884_qk}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV={sparse_decode_use_mma884_pv}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE={sparse_decode_use_mma884_online}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_KV_ROW_PAD_HALF={sparse_decode_kv_row_pad_half}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION={sparse_decode_s0c3_ablation}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_FAST_E4M3={sparse_decode_fast_e4m3}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND={sparse_decode_fast_bf16_round}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_PV_P_CACHE={sparse_decode_pv_p_cache}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_V_FRAGMENT_VECTOR={sparse_decode_v_fragment_vector}",
        f"-DFLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY={sparse_decode_batch2_verify}",
        f"-DFLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS={sparse_prefill_cta_threads}",
        f"-DFLASH_MLA_SM70_SPARSE_PREFILL_K_TILE={sparse_prefill_k_tile}",
        f"-DFLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK={sparse_prefill_use_mma884_qk}",
        f"-DFLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV={sparse_prefill_use_mma884_pv}",
        f"-DFLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE={sparse_prefill_use_mma884_online}",
    ]


def get_arch_flags():
    # Check NVCC Version
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable. For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), '--version'], stderr=subprocess.STDOUT
    ).decode('utf-8')
    nvcc_version_number = nvcc_version.split('release ')[1].split(',')[0].strip()
    major, minor = map(int, nvcc_version_number.split('.'))
    print(f'Compiling using NVCC {major}.{minor}')

    DISABLE_SM100 = is_flag_set("FLASH_MLA_DISABLE_SM100")
    DISABLE_SM90 = is_flag_set("FLASH_MLA_DISABLE_SM90")
    ENABLE_SM70 = is_flag_set("FLASH_MLA_ENABLE_SM70")
    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, "sm100 compilation for Flash MLA requires NVCC 12.9 or higher. Please set FLASH_MLA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."    # TODO Implement this
    if major >= 13:
        assert not ENABLE_SM70, "sm70 compilation for Flash MLA requires a CUDA 12.x toolchain with Volta offline compilation support."

    arch_flags = []
    if ENABLE_SM70:
        arch_flags.extend(["-gencode", "arch=compute_70,code=sm_70"])
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100f,code=sm_100f"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags

def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]

def get_source_in_ptx_args():
    # SM70 offline compilation can feed non-ASCII comments from existing source
    # files into ptxas when --source-in-ptx is enabled. The flag is only for
    # annotated PTX/debug readability, so make it opt-in.
    if is_flag_set("FLASH_MLA_ENABLE_SOURCE_IN_PTX"):
        return ["--source-in-ptx"]
    return []

ensure_cutlass_checkout()

if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
else:
    cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla.cuda",
        sources=[
            # API
            "csrc/api/api.cpp",

            # Misc kernels for decoding
            "csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
            "csrc/smxx/decode/combine/combine.cu",

            # sm90 dense decode
            "csrc/sm90/decode/dense/instantiations/fp16.cu",
            "csrc/sm90/decode/dense/instantiations/bf16.cu",

            # sm70 dense decode
            "csrc/sm70/decode/dense/instantiations/fp16.cu",

            # sm70 sparse decode
            "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu",
            "csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu",

            # sm70 sparse prefill fallback
            "csrc/sm70/prefill/sparse/instantiations/bf16.cu",

            # sm90 sparse decode
            "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu",
            "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu",

            # sm90 sparse prefill
            "csrc/sm90/prefill/sparse/fwd.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu",
            "csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu",

            # sm100 dense prefill & backward
            "csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
            "csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",

            # sm100 sparse prefill
            "csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu",
            "csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu",
            "csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",
            "csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu",
            "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu",

            # sm100 sparse decode
            "csrc/sm100/decode/head64/instantiations/v32.cu",
            "csrc/sm100/decode/head64/instantiations/model1.cu",
            "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + get_features_args() + get_sm70_tuning_args(),
            "nvcc": [
                "-O3",
                "-std=c++20",
                "-DNDEBUG",
                "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                "-lineinfo",
            ] + get_source_in_ptx_args() + get_features_args() + get_sm70_tuning_args() + get_arch_flags() + get_nvcc_thread_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "kerutils" / "include",   # TODO Remove me
            Path(this_dir) / "csrc" / "sm90",
            Path(this_dir) / "csrc" / "cutlass" / "include",
            Path(this_dir) / "csrc" / "cutlass" / "tools" / "util" / "include",
        ],
    )
)

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
