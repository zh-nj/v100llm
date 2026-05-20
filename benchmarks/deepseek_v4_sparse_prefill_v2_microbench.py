#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch

from vllm import envs
from vllm.v1.attention.ops import tilelang_sparse_prefill
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    dequantize_and_gather_k_cache,
)
from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.v1.attention.ops.tilelang_sparse_prefill_v2 import (
    _flash_mla_sparse_prefill_v2_direct_cache,
    _flash_mla_sparse_prefill_v2_mapped_fused_tilelang,
    _cuda_mapped_mma_attention,
    _cuda_mapped_scalar_attention,
    _tilelang_gather_mapped_fp8_ds_mla_cache,
    _triton_build_direct_cache_row_map,
    _triton_gather_mapped_fp8_ds_mla_cache,
    _triton_gather_selected_fp8_ds_mla_cache,
    _triton_mapped_block_attention,
    _triton_mapped_block_scores,
    _triton_mapped_scalar_attention,
)


@dataclass(frozen=True)
class BenchCase:
    rows: int
    seq_len: int
    top_k: int
    window_size: int
    compress_ratio: int
    block_size: int
    block_i: int
    block_n: int
    block_o: int


@dataclass
class CaseInputs:
    q: torch.Tensor
    compressed_k_cache: torch.Tensor
    swa_k_cache: torch.Tensor
    compressed_block_table: torch.Tensor
    swa_block_table: torch.Tensor
    topk_indices: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    gather_lens: torch.Tensor
    attn_sink: torch.Tensor
    sm_scale: float


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _make_rows(num_rows: int, *, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    base = torch.randn((num_rows, 512), generator=gen, device=device)
    # Keep values small enough that fp8 quantization is stable but non-trivial.
    return (base * 0.125).to(torch.bfloat16)


def _make_case(case: BenchCase, *, device: torch.device) -> CaseInputs:
    if case.seq_len < case.rows:
        raise ValueError("seq_len must be >= rows")
    compressed_tokens = max(1, case.seq_len // case.compress_ratio)
    compressed_blocks = _round_up(compressed_tokens, case.block_size) // case.block_size
    # The first prefill token in the chunk can attend to a full SWA window
    # ending before the chunk, and later query tokens extend that range. Mirror
    # the service path's gathered SWA coverage rather than gathering only the
    # final window.
    swa_tokens = min(case.seq_len, case.window_size + case.rows)
    swa_blocks = _round_up(case.seq_len, case.block_size) // case.block_size

    compressed_cache = torch.empty(
        (compressed_blocks, case.block_size, 584), dtype=torch.uint8, device=device
    )
    swa_cache = torch.empty(
        (swa_blocks, case.block_size, 584), dtype=torch.uint8, device=device
    )
    compressed_rows = _make_rows(compressed_tokens, seed=17, device=device)
    swa_rows = _make_rows(case.seq_len, seed=29, device=device)
    quantize_and_insert_k_cache(
        compressed_rows,
        compressed_cache,
        torch.arange(compressed_tokens, dtype=torch.int64, device=device),
        case.block_size,
    )
    quantize_and_insert_k_cache(
        swa_rows,
        swa_cache,
        torch.arange(case.seq_len, dtype=torch.int64, device=device),
        case.block_size,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(41)
    q = (torch.randn((case.rows, 64, 512), generator=gen, device=device) *
         0.0625).to(torch.float16)

    token_positions = torch.arange(
        case.seq_len - case.rows,
        case.seq_len,
        dtype=torch.int32,
        device=device,
    )
    topk_lens = torch.minimum(
        (token_positions + 1) // case.compress_ratio,
        torch.full_like(token_positions, case.top_k),
    )
    if case.top_k > 0:
        offsets = torch.arange(case.top_k, dtype=torch.int32, device=device)
        topk_indices = (
            token_positions[:, None] // case.compress_ratio
            - offsets[None, :]
            - 1
        )
        topk_indices = torch.remainder(topk_indices, compressed_tokens)
        topk_indices = torch.where(
            offsets[None, :] < topk_lens[:, None],
            topk_indices,
            torch.zeros_like(topk_indices),
        ).to(torch.int32)
    else:
        topk_indices = torch.empty((case.rows, 0), dtype=torch.int32, device=device)

    return CaseInputs(
        q=q,
        compressed_k_cache=compressed_cache,
        swa_k_cache=swa_cache,
        compressed_block_table=torch.arange(
            compressed_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0),
        swa_block_table=torch.arange(
            swa_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0),
        topk_indices=topk_indices.contiguous(),
        query_start_loc=torch.tensor([0, case.rows], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([case.seq_len], dtype=torch.int32, device=device),
        gather_lens=torch.tensor([swa_tokens], dtype=torch.int32, device=device),
        attn_sink=torch.full((64,), -float("inf"), dtype=torch.float32, device=device),
        sm_scale=512.0**-0.5,
    )


def _sync() -> None:
    torch.cuda.synchronize()


def _read_proc_status(pid: int) -> dict[str, str]:
    status = {}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                key, _, value = line.partition(":")
                status[key] = value.strip()
    except FileNotFoundError:
        pass
    return status


def _read_proc_cpu_seconds(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            parts = f.read().split()
    except FileNotFoundError:
        return None
    if len(parts) < 15:
        return None
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return (int(parts[13]) + int(parts[14])) / ticks


def _iter_child_processes(parent_pid: int) -> list[dict[str, str]]:
    children = []
    proc_root = "/proc"
    try:
        entries = os.listdir(proc_root)
    except FileNotFoundError:
        return children
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        status = _read_proc_status(pid)
        if status.get("PPid") != str(parent_pid):
            continue
        try:
            with open(f"{proc_root}/{entry}/cmdline", "rb") as f:
                raw_cmdline = f.read().replace(b"\0", b" ").strip()
        except FileNotFoundError:
            raw_cmdline = b""
        children.append({
            "pid": str(pid),
            "name": status.get("Name", ""),
            "state": status.get("State", ""),
            "rss": status.get("VmRSS", ""),
            "cmdline": raw_cmdline.decode("utf-8", errors="replace"),
        })
    return children


def _latest_tvm_artifacts(limit: int = 8) -> list[dict[str, object]]:
    root = "/tmp/tvm-debug-mode-tempdirs"
    suffixes = (".cu", ".ptx", ".cubin", ".so", ".o")
    artifacts = []
    if not os.path.isdir(root):
        return artifacts
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(suffixes):
                continue
            path = os.path.join(dirpath, filename)
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue
            artifacts.append({
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "path": path,
            })
    artifacts.sort(key=lambda item: item["mtime"], reverse=True)
    return artifacts[:limit]


def _query_gpu_processes(pid: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0] == str(pid):
            rows.append(line.strip())
    return "; ".join(rows) if rows else "none"


def _format_artifact(item: dict[str, object]) -> str:
    mtime = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(float(item["mtime"]))
    )
    return f"{mtime} {item['size']}B {item['path']}"


def _run_under_compile_watchdog(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["_VLLM_SPARSE_PREFILL_V2_WATCHDOG_CHILD"] = "1"
    baseline_artifacts = _latest_tvm_artifacts(limit=64)
    baseline_mtime = max(
        (float(item["mtime"]) for item in baseline_artifacts),
        default=0.0,
    )
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    def forward_output() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            print(line, end="", flush=True)

    output_thread = threading.Thread(target=forward_output, daemon=True)
    output_thread.start()
    start_time = time.time()
    saw_fresh_artifact = False
    saw_compiler_child = False
    print(
        "# compile watchdog: "
        f"pid={child.pid} timeout={args.compile_watchdog_seconds}s "
        f"interval={args.compile_watchdog_interval}s",
        flush=True,
    )
    while True:
        returncode = child.poll()
        if returncode is not None:
            output_thread.join(timeout=2)
            return returncode

        elapsed = time.time() - start_time
        status = _read_proc_status(child.pid)
        cpu_seconds = _read_proc_cpu_seconds(child.pid)
        artifacts = _latest_tvm_artifacts()
        if artifacts:
            saw_fresh_artifact = saw_fresh_artifact or any(
                float(item["mtime"]) > baseline_mtime for item in artifacts
            )
        children = _iter_child_processes(child.pid)
        saw_compiler_child = saw_compiler_child or any(
            "nvcc" in proc["cmdline"]
            or "ptxas" in proc["cmdline"]
            or "gcc" in proc["cmdline"]
            or "g++" in proc["cmdline"]
            for proc in children
        )
        child_summary = (
            "; ".join(
                f"{p['pid']}:{p['name']}:{p['state']}:{p['rss']}:{p['cmdline']}"
                for p in children
            )
            if children
            else "none"
        )
        print(
            "# compile watchdog: "
            f"elapsed={elapsed:.1f}s "
            f"state={status.get('State', 'gone')} "
            f"rss={status.get('VmRSS', 'unknown')} "
            f"cpu={cpu_seconds if cpu_seconds is not None else 'unknown'}s "
            f"children={child_summary} "
            f"gpu={_query_gpu_processes(child.pid)}",
            flush=True,
        )
        if artifacts:
            fresh_artifacts = [
                item for item in artifacts if float(item["mtime"]) > baseline_mtime
            ]
            label = (
                "fresh TVM artifacts"
                if fresh_artifacts
                else "fresh TVM artifacts: none; latest pre-existing artifacts"
            )
            print(f"# compile watchdog: {label}:", flush=True)
            for artifact in (fresh_artifacts or artifacts)[:4]:
                print(
                    f"# compile watchdog:   {_format_artifact(artifact)}",
                    flush=True,
                )
        else:
            print("# compile watchdog: TVM artifacts: none", flush=True)

        if elapsed >= args.compile_watchdog_seconds:
            print(
                "# compile watchdog: timeout reached; terminating child. "
                "If there are no fresh .cu/.ptx/.cubin artifacts and no "
                "nvcc/ptxas child process, this indicates TileLang/TVM "
                "frontend lowering did not reach CUDA codegen in time.",
                flush=True,
            )
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            output_thread.join(timeout=2)
            if not saw_fresh_artifact:
                print(
                    "# compile watchdog: no fresh TVM codegen artifacts were observed.",
                    flush=True,
                )
            if not saw_compiler_child:
                print(
                    "# compile watchdog: no nvcc/ptxas/gcc child process was observed.",
                    flush=True,
                )
            return 124

        time.sleep(max(1, args.compile_watchdog_interval))


def _measure(
    name: str,
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, float | int | str]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    _sync()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    _sync()
    peak = torch.cuda.max_memory_allocated()
    return {
        "name": name,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_allocated_mib": peak / (1024 * 1024),
        "repeat": repeat,
    }


def _run_oracle(case: BenchCase, inputs: CaseInputs) -> tuple[torch.Tensor, ...]:
    compressed_tokens = case.seq_len // case.compress_ratio
    m_tokens = compressed_tokens + case.window_size + case.rows
    kv = torch.empty(
        (1, m_tokens, 512), dtype=torch.bfloat16, device=inputs.q.device
    )
    dequantize_and_gather_k_cache(
        kv,
        inputs.compressed_k_cache,
        seq_lens=inputs.seq_lens // case.compress_ratio,
        gather_lens=None,
        block_table=inputs.compressed_block_table,
        block_size=case.block_size,
        offset=0,
    )
    dequantize_and_gather_k_cache(
        kv,
        inputs.swa_k_cache,
        seq_lens=inputs.seq_lens,
        gather_lens=inputs.gather_lens,
        block_table=inputs.swa_block_table,
        block_size=case.block_size,
        offset=compressed_tokens,
    )
    indices, topk_length = combine_topk_swa_indices(
        inputs.topk_indices,
        inputs.query_start_loc,
        inputs.seq_lens,
        inputs.gather_lens,
        case.window_size,
        case.compress_ratio,
        case.top_k,
        m_tokens,
        compressed_tokens,
    )
    out = torch.empty_like(inputs.q)
    return tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
        q=inputs.q,
        kv=kv.view(-1, 1, 512),
        indices=indices.unsqueeze(1),
        sm_scale=inputs.sm_scale,
        d_v=512,
        attn_sink=inputs.attn_sink,
        topk_length=topk_length,
        out=out,
        output_dtype=torch.float16,
        block_I=case.block_i,
        num_stages=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES,
        heads_per_block=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK,
        threads=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_THREADS,
    )


def _run_selected_v2(case: BenchCase, inputs: CaseInputs) -> tuple[torch.Tensor, ...]:
    out = torch.empty_like(inputs.q)
    return _flash_mla_sparse_prefill_v2_direct_cache(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        compressed_block_table=inputs.compressed_block_table,
        swa_block_table=inputs.swa_block_table,
        topk_indices=inputs.topk_indices,
        query_start_loc=inputs.query_start_loc,
        seq_lens=inputs.seq_lens,
        gather_lens=inputs.gather_lens,
        window_size=case.window_size,
        compress_ratio=case.compress_ratio,
        top_k=case.top_k,
        sm_scale=inputs.sm_scale,
        attn_sink=inputs.attn_sink,
        out=out,
        block_I=case.block_i,
    )


def _run_mapped_fused(case: BenchCase,
                      inputs: CaseInputs) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    out = torch.empty_like(inputs.q)
    return _flash_mla_sparse_prefill_v2_mapped_fused_tilelang(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        topk_length=topk_length,
        sm_scale=inputs.sm_scale,
        attn_sink=inputs.attn_sink,
        out=out,
        block_I=case.block_i,
        num_stages=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES,
        threads=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_THREADS,
    )


def _run_selected_gather_only(case: BenchCase,
                              inputs: CaseInputs) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    return _triton_gather_selected_fp8_ds_mla_cache(
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        compressed_block_table=inputs.compressed_block_table,
        swa_block_table=inputs.swa_block_table,
        topk_indices=inputs.topk_indices,
        query_start_loc=inputs.query_start_loc,
        seq_lens=inputs.seq_lens,
        gather_lens=inputs.gather_lens,
        window_size=case.window_size,
        compress_ratio=case.compress_ratio,
        top_k=case.top_k,
        total_topk=total_topk,
        dim=512,
        output_dtype=torch.float16,
    )


def _run_mapped_loader_tilelang(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    return _tilelang_gather_mapped_fp8_ds_mla_cache(
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        row_topk_length=topk_length,
        dim=512,
        output_dtype=torch.float16,
    )


def _run_mapped_loader_triton(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    return _triton_gather_mapped_fp8_ds_mla_cache(
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        row_topk_length=topk_length,
        dim=512,
        output_dtype=torch.float16,
    )


def _run_mapped_scalar_attention(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    if total_topk > 64:
        raise ValueError(
            "mapped_scalar_attention is a tiny-shape scaffold and requires "
            "top_k + window_size rounded to block_i <= 64"
        )
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    out = torch.empty_like(inputs.q)
    return _triton_mapped_scalar_attention(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        topk_length=topk_length,
        sm_scale=inputs.sm_scale,
        attn_sink=inputs.attn_sink,
        out=out,
    )


def _run_mapped_cuda_scalar_attention(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    out = torch.empty_like(inputs.q)
    return _cuda_mapped_scalar_attention(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        topk_length=topk_length,
        sm_scale=inputs.sm_scale,
        attn_sink=inputs.attn_sink,
        out=out,
    )


def _build_direct_row_map(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    return _triton_build_direct_cache_row_map(
        compressed_block_table=inputs.compressed_block_table,
        swa_block_table=inputs.swa_block_table,
        topk_indices=inputs.topk_indices,
        query_start_loc=inputs.query_start_loc,
        seq_lens=inputs.seq_lens,
        gather_lens=inputs.gather_lens,
        window_size=case.window_size,
        compress_ratio=case.compress_ratio,
        top_k=case.top_k,
        total_topk=total_topk,
        compressed_block_size=case.block_size,
        swa_block_size=case.block_size,
    )


def _run_direct_row_map_only(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _build_direct_row_map(case, inputs)


def _run_mapped_mma_attention_with_row_map(
    inputs: CaseInputs,
    source: torch.Tensor,
    physical_block: torch.Tensor,
    block_offset: torch.Tensor,
    topk_length: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    out = torch.empty_like(inputs.q)
    return _cuda_mapped_mma_attention(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        topk_length=topk_length,
        sm_scale=inputs.sm_scale,
        attn_sink=inputs.attn_sink,
        out=out,
    )


def _run_mapped_mma_attention(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, ...]:
    row_map = _build_direct_row_map(case, inputs)
    return _run_mapped_mma_attention_with_row_map(inputs, *row_map)


def _run_mapped_block_scores(
    case: BenchCase,
    inputs: CaseInputs,
) -> torch.Tensor:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    return _triton_mapped_block_scores(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        topk_length=topk_length,
        block_h=16,
        block_n=case.block_n,
        block_d=64,
    )


def _run_mapped_block_attention(
    case: BenchCase,
    inputs: CaseInputs,
) -> tuple[torch.Tensor, ...]:
    total_topk = _round_up(case.top_k + case.window_size, case.block_i)
    source, physical_block, block_offset, topk_length = (
        _triton_build_direct_cache_row_map(
            compressed_block_table=inputs.compressed_block_table,
            swa_block_table=inputs.swa_block_table,
            topk_indices=inputs.topk_indices,
            query_start_loc=inputs.query_start_loc,
            seq_lens=inputs.seq_lens,
            gather_lens=inputs.gather_lens,
            window_size=case.window_size,
            compress_ratio=case.compress_ratio,
            top_k=case.top_k,
            total_topk=total_topk,
            compressed_block_size=case.block_size,
            swa_block_size=case.block_size,
        )
    )
    out = torch.empty_like(inputs.q)
    return _triton_mapped_block_attention(
        q=inputs.q,
        compressed_k_cache=inputs.compressed_k_cache,
        swa_k_cache=inputs.swa_k_cache,
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        topk_length=topk_length,
        sm_scale=inputs.sm_scale,
        attn_sink=inputs.attn_sink,
        out=out,
        block_h=16,
        block_n=16,
        block_k=64,
        block_o=case.block_o,
    )


def _assert_close(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    torch.testing.assert_close(
        actual[0].float(),
        expected[0].float(),
        atol=2e-2,
        rtol=2e-2,
        msg=lambda msg: f"{label} output mismatch\n{msg}",
    )


def _parse_case(raw: str, defaults: argparse.Namespace) -> BenchCase:
    values = {
        "rows": defaults.rows,
        "seq_len": defaults.seq_len,
        "top_k": defaults.top_k,
        "window_size": defaults.window_size,
        "compress_ratio": defaults.compress_ratio,
        "block_size": defaults.block_size,
        "block_i": defaults.block_i,
        "block_n": defaults.block_n,
        "block_o": defaults.block_o,
    }
    if raw:
        for part in raw.split(","):
            key, value = part.split("=", 1)
            values[key.strip()] = int(value)
    return BenchCase(**values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbench SM70 sparse prefill v2 direct-cache variants."
    )
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-i", type=int, default=16)
    parser.add_argument(
        "--block-n",
        type=int,
        default=16,
        help="Selected-token tile width for mapped_block_attention.",
    )
    parser.add_argument(
        "--block-o",
        type=int,
        default=64,
        help="Output dimension block for mapped_block_attention.",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--paths",
        default="oracle,selected_gather_only,selected_v2,mapped_fused_v2",
        help=(
            "Comma-separated paths to run: oracle, selected_gather_only, "
            "mapped_loader_triton, mapped_loader_tilelang, "
            "mapped_block_scores, mapped_block_attention, "
            "mapped_scalar_attention, mapped_cuda_scalar_attention, "
            "direct_row_map_only, mapped_mma_attention, "
            "mapped_mma_attention_kernel, "
            "selected_v2, mapped_fused_v2"
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--compile-watchdog-seconds",
        type=int,
        default=0,
        help=(
            "Run the benchmark in a child process and report TileLang compile "
            "health every interval seconds. A nonzero value terminates the "
            "child after this many seconds and exits with code 124."
        ),
    )
    parser.add_argument(
        "--compile-watchdog-interval",
        type=int,
        default=30,
        help="Seconds between compile watchdog health reports.",
    )
    args = parser.parse_args()

    if (
        args.compile_watchdog_seconds > 0
        and os.getenv("_VLLM_SPARSE_PREFILL_V2_WATCHDOG_CHILD") != "1"
    ):
        raise SystemExit(_run_under_compile_watchdog(args))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ok, reason = tilelang_sparse_prefill.is_tilelang_available()
    if not ok:
        raise RuntimeError(reason)
    device = torch.device("cuda")
    torch.manual_seed(0)

    cases = [_parse_case(raw, args) for raw in args.case] or [_parse_case("", args)]
    paths = {path.strip() for path in args.paths.split(",") if path.strip()}
    if args.skip_oracle:
        paths.discard("oracle")
    results = []
    for case in cases:
        print(f"# case {case}", flush=True)
        inputs = _make_case(case, device=device)
        _sync()

        oracle = None
        if "oracle" in paths:
            t0 = time.time()
            oracle = _run_oracle(case, inputs)
            _sync()
            print(f"correctness oracle built in {time.time() - t0:.2f}s",
                  flush=True)
        selected = _run_selected_v2(case, inputs) if "selected_v2" in paths else None
        mapped_loader = (
            _run_mapped_loader_tilelang(case, inputs)
            if "mapped_loader_tilelang" in paths else None
        )
        mapped_loader_triton = (
            _run_mapped_loader_triton(case, inputs)
            if "mapped_loader_triton" in paths else None
        )
        mapped_scalar = (
            _run_mapped_scalar_attention(case, inputs)
            if "mapped_scalar_attention" in paths else None
        )
        mapped_cuda_scalar = (
            _run_mapped_cuda_scalar_attention(case, inputs)
            if "mapped_cuda_scalar_attention" in paths else None
        )
        mapped_mma = (
            _run_mapped_mma_attention(case, inputs)
            if "mapped_mma_attention" in paths else None
        )
        mapped_scores = (
            _run_mapped_block_scores(case, inputs)
            if "mapped_block_scores" in paths else None
        )
        mapped_block_attention = (
            _run_mapped_block_attention(case, inputs)
            if "mapped_block_attention" in paths else None
        )
        mapped = (
            _run_mapped_fused(case, inputs)
            if "mapped_fused_v2" in paths else None
        )
        _sync()
        if oracle is not None:
            if selected is not None:
                _assert_close(oracle, selected, label="selected_v2")
            if mapped_loader is not None:
                selected_kv, local_indices, topk_length = mapped_loader
                total_topk = _round_up(
                    case.top_k + case.window_size, case.block_i)
                out = torch.empty_like(inputs.q)
                replay = tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
                    q=inputs.q,
                    kv=selected_kv.view(-1, 1, 512),
                    indices=local_indices,
                    sm_scale=inputs.sm_scale,
                    d_v=512,
                    attn_sink=inputs.attn_sink,
                    topk_length=topk_length,
                    out=out,
                    output_dtype=torch.float16,
                    block_I=case.block_i,
                    num_stages=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES,
                    heads_per_block=(
                        envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK
                    ),
                    threads=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_THREADS,
                )
                del total_topk
                _assert_close(oracle, replay, label="mapped_loader_tilelang")
            if mapped_loader_triton is not None:
                selected_kv, local_indices, topk_length = mapped_loader_triton
                out = torch.empty_like(inputs.q)
                replay = tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
                    q=inputs.q,
                    kv=selected_kv.view(-1, 1, 512),
                    indices=local_indices,
                    sm_scale=inputs.sm_scale,
                    d_v=512,
                    attn_sink=inputs.attn_sink,
                    topk_length=topk_length,
                    out=out,
                    output_dtype=torch.float16,
                    block_I=case.block_i,
                    num_stages=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES,
                    heads_per_block=(
                        envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK
                    ),
                    threads=envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_THREADS,
                )
                _assert_close(oracle, replay, label="mapped_loader_triton")
            if mapped_scalar is not None:
                _assert_close(oracle, mapped_scalar, label="mapped_scalar_attention")
            if mapped_cuda_scalar is not None:
                _assert_close(
                    oracle,
                    mapped_cuda_scalar,
                    label="mapped_cuda_scalar_attention",
                )
            if mapped_mma is not None:
                _assert_close(oracle, mapped_mma, label="mapped_mma_attention")
            if mapped_scores is not None:
                if mapped_loader_triton is not None:
                    selected_kv, _local_indices, lens = mapped_loader_triton
                else:
                    selected_kv, _local_indices, lens = (
                        _run_mapped_loader_triton(case, inputs)
                    )
                expected_scores = torch.einsum(
                    "thd,tnd->thn", inputs.q.float(), selected_kv.float()
                )
                positions = torch.arange(
                    selected_kv.shape[1], device=inputs.q.device
                ).unsqueeze(0)
                valid = positions < lens.unsqueeze(1)
                expected_scores = expected_scores.masked_fill(
                    ~valid.unsqueeze(1), -torch.inf
                )
                torch.testing.assert_close(
                    mapped_scores,
                    expected_scores,
                    atol=3e-2,
                    rtol=3e-2,
                    msg=lambda msg: f"mapped_block_scores mismatch\n{msg}",
                )
            if mapped_block_attention is not None:
                _assert_close(
                    oracle,
                    mapped_block_attention,
                    label="mapped_block_attention",
                )
            if mapped is not None:
                _assert_close(oracle, mapped, label="mapped_fused_v2")
            print("correctness: selected paths match oracle", flush=True)

        fns = []
        if "oracle" in paths:
            fns.append(("oracle_full_gather", lambda c=case, i=inputs:
                        _run_oracle(c, i)))
        if "selected_gather_only" in paths:
            fns.append(("selected_gather_only", lambda c=case, i=inputs:
                        _run_selected_gather_only(c, i)))
        if "mapped_loader_tilelang" in paths:
            fns.append(("mapped_loader_tilelang", lambda c=case, i=inputs:
                        _run_mapped_loader_tilelang(c, i)))
        if "mapped_loader_triton" in paths:
            fns.append(("mapped_loader_triton", lambda c=case, i=inputs:
                        _run_mapped_loader_triton(c, i)))
        if "mapped_block_scores" in paths:
            fns.append(("mapped_block_scores", lambda c=case, i=inputs:
                        _run_mapped_block_scores(c, i)))
        if "mapped_block_attention" in paths:
            fns.append(("mapped_block_attention", lambda c=case, i=inputs:
                        _run_mapped_block_attention(c, i)))
        if "mapped_scalar_attention" in paths:
            fns.append(("mapped_scalar_attention", lambda c=case, i=inputs:
                        _run_mapped_scalar_attention(c, i)))
        if "mapped_cuda_scalar_attention" in paths:
            fns.append(("mapped_cuda_scalar_attention",
                        lambda c=case, i=inputs:
                        _run_mapped_cuda_scalar_attention(c, i)))
        if "direct_row_map_only" in paths:
            fns.append(("direct_row_map_only", lambda c=case, i=inputs:
                        _run_direct_row_map_only(c, i)))
        if "mapped_mma_attention" in paths:
            fns.append(("mapped_mma_attention", lambda c=case, i=inputs:
                        _run_mapped_mma_attention(c, i)))
        if "mapped_mma_attention_kernel" in paths:
            row_map = _build_direct_row_map(case, inputs)
            fns.append(("mapped_mma_attention_kernel",
                        lambda i=inputs, r=row_map:
                        _run_mapped_mma_attention_with_row_map(i, *r)))
        if "selected_v2" in paths:
            fns.append(("selected_v2", lambda c=case, i=inputs:
                        _run_selected_v2(c, i)))
        if "mapped_fused_v2" in paths:
            fns.append(("mapped_fused_v2", lambda c=case, i=inputs:
                        _run_mapped_fused(c, i)))
        for name, fn in fns:
            row = _measure(name, fn, warmup=args.warmup, repeat=args.repeat)
            row.update({
                "rows": case.rows,
                "seq_len": case.seq_len,
                "top_k": case.top_k,
                "window_size": case.window_size,
                "compress_ratio": case.compress_ratio,
                "block_size": case.block_size,
                "block_i": case.block_i,
                "block_n": case.block_n,
                "block_o": case.block_o,
            })
            results.append(row)
            print(
                f"{name}: median={row['median_ms']:.3f} ms "
                f"min={row['min_ms']:.3f} ms "
                f"peak={row['peak_allocated_mib']:.1f} MiB",
                flush=True,
            )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
