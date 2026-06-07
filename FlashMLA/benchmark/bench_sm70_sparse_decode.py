#!/usr/bin/env python3
import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for build_dir in sorted(REPO_ROOT.glob("build/lib.*"), reverse=True):
    sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import flash_mla  # noqa: F401,E402
import kernelkit as kk  # noqa: E402
from lib import RawTestParamForDecode as RawTestParam  # noqa: E402
from lib import TestParam, count_flop_and_mem_vol_for_decode, generate_testcase_for_decode, run_flash_mla_decode  # noqa: E402
from ref import ref_sparse_attn_decode  # noqa: E402


SM70_SPARSE_DECODE_REGISTERS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 64,
    (128, "model1"): 64,
    (256, "v32"): 64,
    (256, "model1"): 64,
}
SM70_SPARSE_DECODE_MMA884_QK_REGISTERS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 80,
    (128, "model1"): 80,
    (256, "v32"): 80,
    (256, "model1"): 80,
}
SM70_SPARSE_DECODE_MMA884_QK_PV_REGISTERS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 95,
    (128, "model1"): 96,
    (256, "v32"): 64,
    (256, "model1"): 64,
}
SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTERS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 72,
    (128, "model1"): 80,
    (256, "v32"): 80,
    (256, "model1"): 97,
}
SM70_SPARSE_DECODE_SPILLS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 0,
    (128, "model1"): 0,
    (256, "v32"): 0,
    (256, "model1"): 0,
}
SM70_SPARSE_DECODE_MMA884_QK_PV_SPILLS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 0,
    (128, "model1"): 0,
    (256, "v32"): 20,
    (256, "model1"): 28,
}
SM70_SPARSE_DECODE_MMA884_ONLINE_SPILLS_BY_CTA_THREADS_AND_MODEL = {
    (128, "v32"): 4,
    (128, "model1"): 8,
    (256, "v32"): 0,
    (256, "model1"): 0,
}
SM70_SPARSE_DECODE_CTA_THREADS = int(os.getenv("FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS", "256"))
SM70_SPARSE_DECODE_USE_MMA_884_QK = int(os.getenv("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK", "0")) != 0
SM70_SPARSE_DECODE_USE_MMA_884_PV = int(os.getenv("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV", "0")) != 0
SM70_SPARSE_DECODE_USE_MMA_884_ONLINE = int(os.getenv("FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE", "1")) != 0
if SM70_SPARSE_DECODE_USE_MMA_884_ONLINE:
    SM70_SPARSE_DECODE_USE_MMA_884_QK = True
    SM70_SPARSE_DECODE_USE_MMA_884_PV = True
SM70_SPARSE_DECODE_QK_PATH = "mma884_qk" if SM70_SPARSE_DECODE_USE_MMA_884_QK else "warp_simt_qk"
SM70_SPARSE_DECODE_PV_PATH = "mma884_pv" if SM70_SPARSE_DECODE_USE_MMA_884_PV else "simt_pv"
SM70_SPARSE_DECODE_ONLINE_PATH = "mma884_online" if SM70_SPARSE_DECODE_USE_MMA_884_ONLINE else "two_stage"
SM70_SPARSE_DECODE_COMPUTE_PATH = (
    "mma884_online"
    if SM70_SPARSE_DECODE_USE_MMA_884_ONLINE
    else f"{SM70_SPARSE_DECODE_QK_PATH}+{SM70_SPARSE_DECODE_PV_PATH}"
)
SM70_SPARSE_DECODE_MAX_TOPK = 8192
SM70_SPARSE_DECODE_TOPK_BLOCK = 64
SM70_SPARSE_DECODE_K_TILE = int(os.getenv("FLASH_MLA_SM70_SPARSE_DECODE_K_TILE", "32"))
SM70_SPARSE_DECODE_STAGING_BYTES_PER_VALUE = 2
SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS = 256
SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_MODEL = "model1"
SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE = 16
SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE_THRESHOLD = 512
SM70_SPARSE_DECODE_MMA_884_ONLINE_SCALAR_COUNT = 5
SM70_SPARSE_DECODE_MMA_884_ONLINE_REDUCE_SCRATCH = SM70_SPARSE_DECODE_CTA_THREADS // 32
SM70_REGISTERS_PER_SM = 65536
SM70_SHARED_BYTES_PER_SM = 96 * 1024
SM70_MAX_THREADS_PER_SM = 2048
SM70_MAX_WARPS_PER_SM = 64
SM70_MAX_BLOCKS_PER_SM = 32


def sm70_theoretical_occupancy(registers_per_thread: int, shared_bytes: int, cta_threads: int) -> tuple[int, float]:
    warps_per_block = math.ceil(cta_threads / 32)
    blocks_by_threads = SM70_MAX_THREADS_PER_SM // cta_threads
    blocks_by_warps = SM70_MAX_WARPS_PER_SM // warps_per_block
    blocks_by_registers = SM70_REGISTERS_PER_SM // (registers_per_thread * cta_threads)
    blocks_by_shared = SM70_SHARED_BYTES_PER_SM // shared_bytes if shared_bytes > 0 else SM70_MAX_BLOCKS_PER_SM
    active_blocks = min(
        SM70_MAX_BLOCKS_PER_SM,
        blocks_by_threads,
        blocks_by_warps,
        blocks_by_registers,
        blocks_by_shared,
    )
    occupancy_pct = min(100.0, active_blocks * warps_per_block / SM70_MAX_WARPS_PER_SM * 100.0)
    return active_blocks, occupancy_pct


@dataclass(frozen=True)
class BenchResult:
    case: TestParam
    features: str
    mean_us: float
    splitkv_us: float
    combine_us: float
    tflops: float
    gbps: float
    max_abs_out: float
    max_abs_lse: float


def make_cases(name: str) -> list[TestParam]:
    if name == "quick":
        raw_cases = [
            RawTestParam(
                b=1,
                h_q=64,
                s_q=1,
                h_kv=1,
                s_kv=512,
                is_varlen=False,
                topk=64,
                enable_attn_sink=False,
                block_size=64,
                d_qk=576,
                check_correctness=True,
                num_runs=0,
                seed=0,
            ),
            RawTestParam(
                b=1,
                h_q=128,
                s_q=2,
                h_kv=1,
                s_kv=512,
                is_varlen=False,
                topk=64,
                have_topk_length=True,
                enable_attn_sink=True,
                block_size=64,
                d_qk=576,
                check_correctness=True,
                num_runs=0,
                seed=1,
            ),
            RawTestParam(
                b=1,
                h_q=64,
                s_q=1,
                h_kv=1,
                s_kv=512,
                is_varlen=False,
                topk=64,
                enable_attn_sink=True,
                extra_s_k=512,
                extra_topk=64,
                block_size=64,
                extra_block_size=64,
                d_qk=512,
                check_correctness=True,
                num_runs=0,
                seed=2,
            ),
            RawTestParam(
                b=1,
                h_q=128,
                s_q=2,
                h_kv=1,
                s_kv=512,
                is_varlen=False,
                topk=64,
                have_topk_length=True,
                enable_attn_sink=True,
                extra_s_k=512,
                extra_topk=64,
                block_size=64,
                extra_block_size=64,
                have_extra_topk_length=True,
                d_qk=512,
                check_correctness=True,
                num_runs=0,
                seed=3,
            ),
        ]
    elif name == "matrix":
        raw_cases = [
            RawTestParam(
                b=1,
                h_q=h_q,
                s_q=s_q,
                h_kv=1,
                s_kv=512,
                is_varlen=False,
                topk=64,
                have_topk_length=have_topk_length,
                enable_attn_sink=have_attn_sink,
                extra_s_k=512 if have_extra_kv else None,
                extra_topk=64 if have_extra_kv else None,
                block_size=64,
                extra_block_size=64 if have_extra_kv else None,
                have_extra_topk_length=have_extra_kv and have_topk_length,
                d_qk=d_qk,
                check_correctness=True,
                num_runs=0,
                seed=seed,
            )
            for seed, (d_qk, h_q, s_q, have_attn_sink, have_topk_length, have_extra_kv) in enumerate(
                (
                    (d_qk, h_q, s_q, have_attn_sink, have_topk_length, have_extra_kv)
                    for d_qk in [576, 512]
                    for h_q in [64, 128]
                    for s_q in [1, 2]
                    for have_attn_sink in [False, True]
                    for have_topk_length in [False, True]
                    for have_extra_kv in ([False, True] if d_qk == 512 else [False])
                )
            )
        ]
    elif name == "long":
        raw_cases = [
            RawTestParam(
                b=2,
                h_q=64,
                s_q=2,
                h_kv=1,
                s_kv=2048,
                is_varlen=True,
                topk=512,
                enable_attn_sink=True,
                block_size=64,
                d_qk=576,
                check_correctness=True,
                num_runs=0,
                seed=0,
            ),
            RawTestParam(
                b=2,
                h_q=128,
                s_q=2,
                h_kv=1,
                s_kv=2048,
                is_varlen=True,
                topk=512,
                have_topk_length=True,
                enable_attn_sink=True,
                extra_s_k=2048,
                extra_topk=512,
                block_size=64,
                extra_block_size=64,
                have_extra_topk_length=True,
                d_qk=512,
                check_correctness=True,
                num_runs=0,
                seed=1,
            ),
        ]
    elif name == "max_topk":
        raw_cases = [
            RawTestParam(
                b=1,
                h_q=64,
                s_q=1,
                h_kv=1,
                s_kv=SM70_SPARSE_DECODE_MAX_TOPK,
                is_varlen=False,
                topk=SM70_SPARSE_DECODE_MAX_TOPK,
                enable_attn_sink=True,
                block_size=64,
                d_qk=576,
                check_correctness=True,
                num_runs=0,
                seed=8,
            ),
            RawTestParam(
                b=1,
                h_q=64,
                s_q=1,
                h_kv=1,
                s_kv=4096,
                is_varlen=False,
                topk=4096,
                have_topk_length=True,
                enable_attn_sink=True,
                extra_s_k=4096,
                extra_topk=4096,
                block_size=64,
                extra_block_size=64,
                have_extra_topk_length=True,
                d_qk=512,
                check_correctness=True,
                num_runs=0,
                seed=9,
            ),
        ]
    elif name == "all":
        return make_cases("quick") + make_cases("long") + make_cases("max_topk")
    else:
        raise ValueError(f"unknown case set: {name}")

    cases = [raw.to_test_param() for raw in raw_cases]
    for case in cases:
        assert case.decode is not None
        total_topk = case.topk + (case.decode.extra_topk or 0)
        if total_topk > SM70_SPARSE_DECODE_MAX_TOPK:
            raise ValueError(f"SM70 sparse decode alpha supports topk + extra_topk <= {SM70_SPARSE_DECODE_MAX_TOPK}")
    return cases


def feature_label(case: TestParam) -> str:
    assert case.decode is not None
    features = [model_label(case)]
    if case.decode.is_varlen:
        features.append("varlen")
    if case.have_attn_sink:
        features.append("attn_sink")
    if case.have_topk_length:
        features.append("topk_length")
    if case.decode.extra_topk is not None:
        features.append("extra_kv")
    if case.decode.have_extra_topk_length:
        features.append("extra_topk_length")
    return "+".join(features)


def model_label(case: TestParam) -> str:
    return "model1" if case.d_qk == 512 else "v32"


def sm70_sparse_decode_registers(case: TestParam) -> int:
    if SM70_SPARSE_DECODE_USE_MMA_884_ONLINE:
        table = SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTERS_BY_CTA_THREADS_AND_MODEL
    elif SM70_SPARSE_DECODE_USE_MMA_884_QK and SM70_SPARSE_DECODE_USE_MMA_884_PV:
        table = SM70_SPARSE_DECODE_MMA884_QK_PV_REGISTERS_BY_CTA_THREADS_AND_MODEL
    elif SM70_SPARSE_DECODE_USE_MMA_884_QK:
        table = SM70_SPARSE_DECODE_MMA884_QK_REGISTERS_BY_CTA_THREADS_AND_MODEL
    else:
        table = SM70_SPARSE_DECODE_REGISTERS_BY_CTA_THREADS_AND_MODEL
    return table[
        (SM70_SPARSE_DECODE_CTA_THREADS, model_label(case))
    ]


def sm70_sparse_decode_spills(case: TestParam) -> int:
    if SM70_SPARSE_DECODE_USE_MMA_884_ONLINE:
        table = SM70_SPARSE_DECODE_MMA884_ONLINE_SPILLS_BY_CTA_THREADS_AND_MODEL
    elif SM70_SPARSE_DECODE_USE_MMA_884_QK and SM70_SPARSE_DECODE_USE_MMA_884_PV:
        table = SM70_SPARSE_DECODE_MMA884_QK_PV_SPILLS_BY_CTA_THREADS_AND_MODEL
    else:
        table = SM70_SPARSE_DECODE_SPILLS_BY_CTA_THREADS_AND_MODEL
    return table[
        (SM70_SPARSE_DECODE_CTA_THREADS, model_label(case))
    ]


def sm70_sparse_decode_uses_register_accumulator(case: TestParam) -> bool:
    return (
        SM70_SPARSE_DECODE_USE_MMA_884_ONLINE
        and SM70_SPARSE_DECODE_CTA_THREADS == SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS
        and model_label(case) == SM70_SPARSE_DECODE_MMA884_ONLINE_REGISTER_ACCUM_MODEL
    )


def sm70_sparse_decode_runtime_k_tile(case: TestParam) -> int:
    assert case.decode is not None
    total_topk = case.topk + (case.decode.extra_topk or 0)
    if (
        SM70_SPARSE_DECODE_USE_MMA_884_ONLINE
        and SM70_SPARSE_DECODE_K_TILE > SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE
        and model_label(case) == "v32"
        and total_topk >= SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE_THRESHOLD
    ):
        return SM70_SPARSE_DECODE_MMA_884_ONLINE_LONG_K_TILE
    return SM70_SPARSE_DECODE_K_TILE


def sm70_sparse_decode_shared_bytes(case: TestParam) -> int:
    assert case.decode is not None
    if SM70_SPARSE_DECODE_USE_MMA_884_ONLINE:
        online_k_tile = sm70_sparse_decode_runtime_k_tile(case)
        shared_bytes = (
            online_k_tile * 4
            + SM70_SPARSE_DECODE_MMA_884_ONLINE_SCALAR_COUNT * 4
            + SM70_SPARSE_DECODE_MMA_884_ONLINE_REDUCE_SCRATCH * 4
            + online_k_tile * 4
            + online_k_tile * case.d_qk * SM70_SPARSE_DECODE_STAGING_BYTES_PER_VALUE
        )
        if not sm70_sparse_decode_uses_register_accumulator(case):
            shared_bytes += case.d_v * 4
        return shared_bytes
    extra_topk = case.decode.extra_topk or 0
    if extra_topk == 0:
        score_capacity = case.topk
    else:
        score_capacity = ((case.topk + SM70_SPARSE_DECODE_TOPK_BLOCK - 1)
                          // SM70_SPARSE_DECODE_TOPK_BLOCK
                          * SM70_SPARSE_DECODE_TOPK_BLOCK) + extra_topk
    return (
        score_capacity * (4 + 4)
        + SM70_SPARSE_DECODE_CTA_THREADS * 4
        + SM70_SPARSE_DECODE_K_TILE * case.d_qk * SM70_SPARSE_DECODE_STAGING_BYTES_PER_VALUE
    )


def max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = torch.abs(actual_f - expected_f)
    diff = torch.where(actual_f == expected_f, torch.zeros_like(diff), diff)
    if diff.numel() == 0:
        return 0.0
    return float(diff.max().item())


def time_cuda(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / runs


def benchmark_case(case: TestParam, warmup: int, runs: int, check_correctness: bool) -> BenchResult:
    testcase = generate_testcase_for_decode(case)
    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

    def run_decode():
        return run_flash_mla_decode(case, testcase, tile_scheduler_metadata, None)

    out, lse = run_decode()
    max_abs_out = 0.0
    max_abs_lse = 0.0
    if check_correctness:
        ref_out, ref_lse = ref_sparse_attn_decode(case, testcase)
        max_abs_out = max_abs_diff(out, ref_out)
        max_abs_lse = max_abs_diff(lse, ref_lse)

    for _ in range(warmup):
        run_decode()
    torch.cuda.synchronize()

    kernel_times = kk.bench_kineto(run_decode, runs, flush_l2=False)
    splitkv_kernel_name = "flash_fwd_splitkv_mla_sm70_sparse_kernel"
    combine_kernel_name = "flash_fwd_mla_combine_kernel"
    kernel_names = kernel_times.get_kernel_names()
    have_combine = any(combine_kernel_name in name for name in kernel_names)
    splitkv_us = kernel_times.get_kernel_time(splitkv_kernel_name) * 1e6
    combine_us = kernel_times.get_kernel_time(combine_kernel_name) * 1e6 if have_combine else 0.0
    mean_us = (
        kernel_times.get_e2e_time(splitkv_kernel_name, combine_kernel_name) * 1e6
        if have_combine else splitkv_us
    )
    work = count_flop_and_mem_vol_for_decode(case, testcase)
    seconds = mean_us / 1e6
    return BenchResult(
        case=case,
        features=feature_label(case),
        mean_us=mean_us,
        splitkv_us=splitkv_us,
        combine_us=combine_us,
        tflops=work.flop / seconds / 1e12,
        gbps=work.mem_vol / seconds / 1e9,
        max_abs_out=max_abs_out,
        max_abs_lse=max_abs_lse,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SM70 sparse decode FP8 alpha paths.")
    parser.add_argument("--cases", choices=["quick", "matrix", "long", "max_topk", "all"], default="quick")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-correctness", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda:0")
    torch.cuda.set_device("cuda:0")
    torch.set_float32_matmul_precision("high")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    assert (cc_major, cc_minor) == (7, 0), "This benchmark is intended for SM70/V100."

    print(
        "b,s_q,s_kv,topk,extra_topk,h_q,h_kv,d_qk,d_v,features,mean_us,"
        "splitkv_us,combine_us,tflops,gbps,max_abs_out,max_abs_lse,"
        "registers,spills,shared_bytes,cta_threads,k_tile,runtime_k_tile,qk_path,pv_path,online_path,compute_path,max_topk,"
        "active_blocks_per_sm,theoretical_occupancy_pct",
        flush=True,
    )
    for case in make_cases(args.cases):
        assert case.decode is not None
        result = benchmark_case(case, args.warmup, args.runs, not args.no_correctness)
        extra_topk = case.decode.extra_topk or 0
        registers = sm70_sparse_decode_registers(case)
        spills = sm70_sparse_decode_spills(case)
        shared_bytes = sm70_sparse_decode_shared_bytes(case)
        runtime_k_tile = sm70_sparse_decode_runtime_k_tile(case)
        active_blocks, occupancy_pct = sm70_theoretical_occupancy(
            registers,
            shared_bytes,
            SM70_SPARSE_DECODE_CTA_THREADS,
        )
        print(
            f"{case.decode.b},{case.s_q},{case.s_kv},{case.topk},{extra_topk},"
            f"{case.h_q},{case.h_kv},{case.d_qk},{case.d_v},{result.features},"
            f"{result.mean_us:.3f},{result.splitkv_us:.3f},{result.combine_us:.3f},"
            f"{result.tflops:.6f},{result.gbps:.3f},"
            f"{result.max_abs_out:.6g},{result.max_abs_lse:.6g},"
            f"{registers},{spills},"
            f"{shared_bytes},{SM70_SPARSE_DECODE_CTA_THREADS},{SM70_SPARSE_DECODE_K_TILE},{runtime_k_tile},"
            f"{SM70_SPARSE_DECODE_QK_PATH},{SM70_SPARSE_DECODE_PV_PATH},"
            f"{SM70_SPARSE_DECODE_ONLINE_PATH},{SM70_SPARSE_DECODE_COMPUTE_PATH},"
            f"{SM70_SPARSE_DECODE_MAX_TOPK},"
            f"{active_blocks},{occupancy_pct:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
