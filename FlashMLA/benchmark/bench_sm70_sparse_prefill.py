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
from lib import TestParam, generate_testcase, run_flash_mla_sparse_fwd  # noqa: E402
from ref import ref_sparse_attn_fwd  # noqa: E402


SM70_SPARSE_PREFILL_CTA_THREADS = int(os.getenv("FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS", "256"))
SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE = int(os.getenv("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE", "1")) != 0
SM70_SPARSE_PREFILL_USE_MMA_884_QK = (
    int(os.getenv("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK", "0")) != 0
    or SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE
)
SM70_SPARSE_PREFILL_USE_MMA_884_PV = (
    int(os.getenv("FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV", "0")) != 0
    or SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE
)
SM70_SPARSE_PREFILL_QK_PATH = "mma884_qk" if SM70_SPARSE_PREFILL_USE_MMA_884_QK else "warp_simt_qk"
SM70_SPARSE_PREFILL_PV_PATH = "mma884_pv" if SM70_SPARSE_PREFILL_USE_MMA_884_PV else "simt_pv"
SM70_SPARSE_PREFILL_ONLINE_PATH = "mma884_online" if SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE else "two_stage"
SM70_SPARSE_PREFILL_COMPUTE_PATH = (
    "mma884_online"
    if SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE
    else f"{SM70_SPARSE_PREFILL_QK_PATH}+{SM70_SPARSE_PREFILL_PV_PATH}"
)
SM70_SPARSE_PREFILL_REGISTERS_BY_QK_PATH_AND_CTA_THREADS = {
    ("warp_simt_qk", 128): 32,
    ("warp_simt_qk", 256): 29,
    ("mma884_qk", 256): 40,
}
SM70_SPARSE_PREFILL_REGISTERS_BY_COMPUTE_PATH_AND_CTA_THREADS = {
    ("warp_simt_qk+simt_pv", 128): 32,
    ("warp_simt_qk+simt_pv", 256): 29,
    ("mma884_qk+simt_pv", 256): 40,
    ("mma884_qk+mma884_pv", 256): 48,
    ("mma884_online", 256): 48,
}
SM70_SPARSE_PREFILL_REGISTERS = SM70_SPARSE_PREFILL_REGISTERS_BY_COMPUTE_PATH_AND_CTA_THREADS.get(
    (SM70_SPARSE_PREFILL_COMPUTE_PATH, SM70_SPARSE_PREFILL_CTA_THREADS),
    48 if SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE else (
        48 if SM70_SPARSE_PREFILL_USE_MMA_884_PV else (40 if SM70_SPARSE_PREFILL_USE_MMA_884_QK else 29)
    ),
)
SM70_SPARSE_PREFILL_SPILLS = 0
SM70_SPARSE_PREFILL_MAX_TOPK = 8192
SM70_SPARSE_PREFILL_K_TILE = int(os.getenv("FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE", "32"))
SM70_SPARSE_PREFILL_STAGING_BYTES_PER_VALUE = 2
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


def sm70_sparse_prefill_shared_bytes(case: TestParam) -> int:
    if SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE:
        return (
            SM70_SPARSE_PREFILL_K_TILE * 4
            + 4 * 4
            + case.d_v * 4
            + SM70_SPARSE_PREFILL_K_TILE * 4
            + SM70_SPARSE_PREFILL_K_TILE * case.d_qk * SM70_SPARSE_PREFILL_STAGING_BYTES_PER_VALUE
        )
    return (
        (case.topk + 2) * 4
        + SM70_SPARSE_PREFILL_K_TILE * 4
        + SM70_SPARSE_PREFILL_K_TILE * case.d_qk * SM70_SPARSE_PREFILL_STAGING_BYTES_PER_VALUE
    )


@dataclass(frozen=True)
class BenchResult:
    case: TestParam
    features: str
    mean_us: float
    tflops: float
    gbps: float
    max_abs_out: float
    max_abs_lse: float
    max_abs_max_logits: float


@dataclass(frozen=True)
class WorkEstimate:
    flops: float
    bytes_moved: float


def make_cases(name: str) -> list[TestParam]:
    if name == "quick":
        cases = [
            TestParam(1, 128, 64, h_q=64, d_qk=512, seed=0, num_runs=0),
            TestParam(1, 128, 64, h_q=128, d_qk=576, seed=1, num_runs=0, have_attn_sink=True),
            TestParam(3, 256, 128, h_q=64, d_qk=576, seed=2, num_runs=0, have_topk_length=True),
            TestParam(
                3,
                256,
                128,
                h_q=128,
                d_qk=512,
                seed=3,
                num_runs=0,
                have_attn_sink=True,
                have_topk_length=True,
            ),
        ]
    elif name == "matrix":
        cases = [
            TestParam(
                s_q,
                s_kv,
                topk,
                h_q=h_q,
                d_qk=d_qk,
                seed=seed,
                num_runs=0,
                have_attn_sink=have_attn_sink,
                have_topk_length=have_topk_length,
            )
            for seed, (d_qk, h_q, s_q, s_kv, topk, have_attn_sink, have_topk_length) in enumerate(
                (
                    (d_qk, h_q, s_q, s_kv, topk, have_attn_sink, have_topk_length)
                    for d_qk in [512, 576]
                    for h_q in [64, 128]
                    for s_q, s_kv, topk in [(1, 128, 64), (3, 256, 128)]
                    for have_attn_sink in [False, True]
                    for have_topk_length in [False, True]
                )
            )
        ]
    elif name == "long":
        cases = [
            TestParam(16, 2048, 512, h_q=64, d_qk=576, seed=0, num_runs=0, have_attn_sink=True),
            TestParam(
                16,
                2048,
                512,
                h_q=128,
                d_qk=512,
                seed=1,
                num_runs=0,
                have_attn_sink=True,
                have_topk_length=True,
            ),
        ]
    elif name == "max_topk":
        cases = [
            TestParam(
                1,
                8192,
                SM70_SPARSE_PREFILL_MAX_TOPK,
                h_q=64,
                d_qk=576,
                seed=4,
                num_runs=0,
                have_attn_sink=True,
            ),
        ]
    elif name == "all":
        cases = make_cases("quick") + make_cases("long") + make_cases("max_topk")
    else:
        raise ValueError(f"unknown case set: {name}")

    for case in cases:
        if case.topk > SM70_SPARSE_PREFILL_MAX_TOPK:
            raise ValueError(f"SM70 sparse prefill BF16 fast path supports topk <= {SM70_SPARSE_PREFILL_MAX_TOPK}")
    return cases


def feature_label(case: TestParam) -> str:
    features = []
    if case.have_attn_sink:
        features.append("attn_sink")
    if case.have_topk_length:
        features.append("topk_length")
    return "+".join(features) if features else "-"


def estimate_work(case: TestParam, testcase) -> WorkEstimate:
    valid = (testcase.indices >= 0) & (testcase.indices < case.s_kv)
    if testcase.topk_length is not None:
        topk_range = torch.arange(case.topk, device=testcase.indices.device)
        valid &= topk_range.view(1, 1, case.topk) < testcase.topk_length.view(case.s_q, 1, 1)
        total_topk = int(testcase.topk_length.sum().item()) * case.h_kv
    else:
        total_topk = case.s_q * case.h_kv * case.topk
    valid_indices = int(valid.sum().item())

    flops = 2.0 * total_topk * case.h_q * (case.d_qk + case.d_v)
    bytes_moved = 2.0 * (
        valid_indices * case.h_q * (case.d_qk + case.d_v)
        + case.s_q * case.h_q * (case.d_qk + case.d_v)
    )
    return WorkEstimate(flops=flops, bytes_moved=bytes_moved)


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
    testcase = generate_testcase(case)

    def run_prefill():
        return run_flash_mla_sparse_fwd(case, testcase, False)

    out, max_logits, lse = run_prefill()
    max_abs_out = 0.0
    max_abs_lse = 0.0
    max_abs_max_logits = 0.0
    if check_correctness:
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref_sparse_attn_fwd(case, testcase)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        max_abs_out = max_abs_diff(out.float(), ref_out_fp32)
        max_abs_lse = max_abs_diff(lse, ref_lse)
        max_abs_max_logits = max_abs_diff(max_logits, ref_max_logits)

    mean_us = time_cuda(run_prefill, warmup, runs)
    work = estimate_work(case, testcase)
    seconds = mean_us / 1e6
    return BenchResult(
        case=case,
        features=feature_label(case),
        mean_us=mean_us,
        tflops=work.flops / seconds / 1e12,
        gbps=work.bytes_moved / seconds / 1e9,
        max_abs_out=max_abs_out,
        max_abs_lse=max_abs_lse,
        max_abs_max_logits=max_abs_max_logits,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SM70 sparse prefill BF16 SIMT fast path.")
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
        "s_q,s_kv,topk,h_q,h_kv,d_qk,d_v,features,mean_us,tflops,gbps,"
        "max_abs_out,max_abs_lse,max_abs_max_logits,registers,spills,"
        "shared_bytes,cta_threads,k_tile,qk_path,pv_path,online_path,compute_path,max_topk,"
        "active_blocks_per_sm,theoretical_occupancy_pct",
        flush=True,
    )
    for case in make_cases(args.cases):
        result = benchmark_case(case, args.warmup, args.runs, not args.no_correctness)
        shared_bytes = sm70_sparse_prefill_shared_bytes(case)
        active_blocks, occupancy_pct = sm70_theoretical_occupancy(
            SM70_SPARSE_PREFILL_REGISTERS,
            shared_bytes,
            SM70_SPARSE_PREFILL_CTA_THREADS,
        )
        print(
            f"{case.s_q},{case.s_kv},{case.topk},{case.h_q},{case.h_kv},"
            f"{case.d_qk},{case.d_v},{result.features},{result.mean_us:.3f},"
            f"{result.tflops:.6f},{result.gbps:.3f},"
            f"{result.max_abs_out:.6g},{result.max_abs_lse:.6g},"
            f"{result.max_abs_max_logits:.6g},"
            f"{SM70_SPARSE_PREFILL_REGISTERS},{SM70_SPARSE_PREFILL_SPILLS},"
            f"{shared_bytes},{SM70_SPARSE_PREFILL_CTA_THREADS},"
            f"{SM70_SPARSE_PREFILL_K_TILE},{SM70_SPARSE_PREFILL_QK_PATH},"
            f"{SM70_SPARSE_PREFILL_PV_PATH},{SM70_SPARSE_PREFILL_ONLINE_PATH},"
            f"{SM70_SPARSE_PREFILL_COMPUTE_PATH},"
            f"{SM70_SPARSE_PREFILL_MAX_TOPK},{active_blocks},{occupancy_pct:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
