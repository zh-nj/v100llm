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

import flash_mla  # noqa: E402
from test_flash_mla_dense_decoding import (  # noqa: E402
    TestParam,
    generate_test_data,
    reference_torch,
)


SM70_ALPHA_H_TILE = int(os.getenv("FLASH_MLA_SM70_H_TILE", "4"))
SM70_ALPHA_CTA_THREADS = int(os.getenv("FLASH_MLA_SM70_CTA_THREADS", "256"))
SM70_ALPHA_USE_MMA_884_ONLINE = int(os.getenv("FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE", "0")) != 0
SM70_ALPHA_MMA_884_ONLINE_K_TILE = int(os.getenv("FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE", "0"))
SM70_ALPHA_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS = 256
SM70_ALPHA_MMA_884_ONLINE_REDUCE_SCRATCH = SM70_ALPHA_CTA_THREADS // 32
SM70_ALPHA_COMPUTE_PATH = "mma884_online" if SM70_ALPHA_USE_MMA_884_ONLINE else "scalar_online"
SM70_ALPHA_REGISTERS_BY_COMPUTE_PATH = {
    "scalar_online": 64,
    "mma884_online": 113,
}
SM70_ALPHA_SPILLS_BY_COMPUTE_PATH = {
    "scalar_online": 0,
    "mma884_online": 0,
}
SM70_ALPHA_REGISTERS = SM70_ALPHA_REGISTERS_BY_COMPUTE_PATH[SM70_ALPHA_COMPUTE_PATH]
SM70_ALPHA_SPILLS = SM70_ALPHA_SPILLS_BY_COMPUTE_PATH[SM70_ALPHA_COMPUTE_PATH]
SM70_ALPHA_SHARED_BYTES = (SM70_ALPHA_CTA_THREADS + 64) * 4


def sm70_alpha_runtime_online_k_tile(case: TestParam) -> int:
    if not SM70_ALPHA_USE_MMA_884_ONLINE:
        return SM70_ALPHA_MMA_884_ONLINE_K_TILE
    if SM70_ALPHA_MMA_884_ONLINE_K_TILE == 0:
        return 64
    return SM70_ALPHA_MMA_884_ONLINE_K_TILE


def sm70_alpha_mma884_online_shared_bytes(head_dim: int, online_k_tile: int) -> int:
    base_shared = (
        online_k_tile * 4
        + 5 * 4
        + SM70_ALPHA_MMA_884_ONLINE_REDUCE_SCRATCH * 4
        + online_k_tile * head_dim * 2
    )
    if SM70_ALPHA_CTA_THREADS == SM70_ALPHA_MMA884_ONLINE_REGISTER_ACCUM_CTA_THREADS:
        return base_shared
    return base_shared + 512 * 4


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
    mean_us: float
    tflops: float
    gbps: float
    max_abs_out: float
    max_abs_lse: float


def make_cases(name: str) -> list[TestParam]:
    if name == "quick":
        seq_lens = [20, 140]
        h_q = 1
    elif name == "long":
        seq_lens = [4096, 8192, 16384]
        h_q = 1
    elif name == "tile":
        seq_lens = [140, 4096]
        h_q = 8
    elif name == "all":
        seq_lens = [20, 140, 4096, 8192, 16384]
        h_q = 1
    else:
        raise ValueError(f"unknown case set: {name}")

    return [
        TestParam(
            b=1,
            s_q=1,
            s_k=s_k,
            is_varlen=False,
            is_causal=False,
            test_performance=False,
            block_size=64,
            h_q=h_q,
            h_kv=1,
            d=d,
            dv=512,
        )
        for s_k in seq_lens
        for d in [512, 576]
    ]


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
    cache_seqlens, q, block_table, blocked_k = generate_test_data(case)
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

    def run_flash_mla():
        return flash_mla.flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            case.dv,
            tile_scheduler_metadata,
            num_splits,
            causal=case.is_causal,
        )

    out, lse = run_flash_mla()
    max_abs_out = 0.0
    max_abs_lse = 0.0
    if check_correctness:
        out_ref, lse_ref = reference_torch(cache_seqlens, block_table, q, blocked_k, case.dv, case.is_causal)
        max_abs_out = float((out.float() - out_ref.float()).abs().max().item())
        max_abs_lse = float((lse.float() - lse_ref.float()).abs().max().item())

    mean_us = time_cuda(run_flash_mla, warmup, runs)
    mean_s_k = float(cache_seqlens.float().mean().item())
    flops = case.b * case.h_q * case.s_q * (
        2 * case.d * mean_s_k + 2 * mean_s_k * case.dv
    )
    bytes_moved = case.b * (
        case.s_q * case.h_q * case.d * torch.float16.itemsize
        + mean_s_k * case.h_kv * case.d * torch.float16.itemsize
        + case.s_q * case.h_q * case.dv * torch.float16.itemsize
    )
    seconds = mean_us / 1e6
    return BenchResult(
        case=case,
        mean_us=mean_us,
        tflops=flops / seconds / 1e12,
        gbps=bytes_moved / seconds / 1e9,
        max_abs_out=max_abs_out,
        max_abs_lse=max_abs_lse,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SM70 dense decode alpha.")
    parser.add_argument("--cases", choices=["quick", "long", "tile", "all"], default="quick")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--no-correctness", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda:0")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    assert (cc_major, cc_minor) == (7, 0), "This benchmark is intended for SM70/V100."

    print(
        "b,s_q,s_k,h_q,h_kv,d,dv,mean_us,tflops,gbps,"
        "max_abs_out,max_abs_lse,registers,spills,shared_bytes,h_tile,cta_threads,online_k_tile,runtime_online_k_tile,compute_path,"
        "active_blocks_per_sm,theoretical_occupancy_pct"
    )
    for case in make_cases(args.cases):
        result = benchmark_case(case, args.warmup, args.runs, not args.no_correctness)
        runtime_online_k_tile = sm70_alpha_runtime_online_k_tile(case)
        shared_bytes = (
            sm70_alpha_mma884_online_shared_bytes(case.d, runtime_online_k_tile)
            if SM70_ALPHA_USE_MMA_884_ONLINE
            else SM70_ALPHA_SHARED_BYTES
        )
        active_blocks, occupancy_pct = sm70_theoretical_occupancy(
            SM70_ALPHA_REGISTERS,
            shared_bytes,
            SM70_ALPHA_CTA_THREADS,
        )
        print(
            f"{case.b},{case.s_q},{case.s_k},{case.h_q},{case.h_kv},"
            f"{case.d},{case.dv},{result.mean_us:.3f},"
            f"{result.tflops:.6f},{result.gbps:.3f},"
            f"{result.max_abs_out:.6g},{result.max_abs_lse:.6g},"
            f"{SM70_ALPHA_REGISTERS},{SM70_ALPHA_SPILLS},{shared_bytes},"
            f"{SM70_ALPHA_H_TILE},{SM70_ALPHA_CTA_THREADS},{SM70_ALPHA_MMA_884_ONLINE_K_TILE},{runtime_online_k_tile},"
            f"{SM70_ALPHA_COMPUTE_PATH},"
            f"{active_blocks},{occupancy_pct:.1f}"
        )


if __name__ == "__main__":
    main()
