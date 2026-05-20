#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.attention.ops import tilelang_sparse_prefill


@dataclass(frozen=True)
class Config:
    heads_per_block: int
    num_stages: int
    threads: int


@dataclass(frozen=True)
class BenchCase:
    s_q: int
    s_kv: int


def _parse_config(raw: str) -> Config:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "config must be hpb,stages,threads, for example 64,1,128")
    return Config(int(parts[0]), int(parts[1]), int(parts[2]))


def _parse_case(raw: str) -> BenchCase:
    values: dict[str, int] = {}
    for part in raw.split(","):
        key, value = part.split("=", 1)
        values[key.strip()] = int(value)
    return BenchCase(s_q=values["s_q"], s_kv=values["s_kv"])


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
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        status = _read_proc_status(pid)
        if status.get("PPid") != str(parent_pid):
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
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


def _format_artifact(item: dict[str, object]) -> str:
    mtime = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(float(item["mtime"])))
    return f"{mtime} {item['size']}B {item['path']}"


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


def _make_inputs(
    *,
    s_q: int,
    s_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    d_qk = dim + tail_dim
    q = (torch.randn((s_q, heads, d_qk), generator=gen, device=device) *
         0.0625).to(torch.float16)
    kv = (torch.randn((s_kv, 1, d_qk), generator=gen, device=device) *
          0.0625).to(torch.float16)
    indices = torch.randint(
        0, s_kv, (s_q, 1, topk), generator=gen, device=device,
        dtype=torch.int32)
    topk_length = torch.full((s_q,), topk, dtype=torch.int32, device=device)
    attn_sink = torch.full((heads,), -float("inf"), dtype=torch.float32,
                           device=device)
    out = torch.empty((s_q, heads, dim), dtype=torch.float16, device=device)
    return q, kv, indices, topk_length, attn_sink, out


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    dim: int,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = kv[:, 0, :].float()[indices[:, 0].long()]
    scores = torch.einsum("shd,skd->shk", q.float(), selected) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("shk,skd->shd", probs, selected[..., :dim])
    max_logits = scores.max(dim=-1).values
    lse = torch.logsumexp(scores, dim=-1)
    return output, max_logits, lse


def _run_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    *,
    dim: int,
    block_i: int,
    config: Config,
    sm_scale: float,
    kernel_output_dtype: torch.dtype,
    pv_gemm_policy: str,
    assume_valid_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=dim,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
        output_dtype=kernel_output_dtype,
        block_I=block_i,
        num_stages=config.num_stages,
        heads_per_block=config.heads_per_block,
        threads=config.threads,
        pv_gemm_policy=pv_gemm_policy,
        assume_valid_indices=assume_valid_indices,
    )


def _measure(fn, *, warmup: int, repeat: int) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
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
    torch.cuda.synchronize()
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        ),
    }


def _child_main(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ok, reason = tilelang_sparse_prefill.is_tilelang_available()
    if not ok:
        raise RuntimeError(reason)
    device = torch.device("cuda")
    config = args.child_config
    sm_scale = (args.dim + args.tail_dim)**-0.5
    kernel_output_dtype = (
        torch.bfloat16 if args.kernel_output_dtype == "bfloat16"
        else torch.float16
    )

    try:
        q, kv, indices, topk_length, attn_sink, out = _make_inputs(
            s_q=args.correctness_s_q,
            s_kv=args.correctness_s_kv,
            heads=args.heads,
            dim=args.dim,
            tail_dim=args.tail_dim,
            topk=args.topk,
            seed=11,
            device=device,
        )
        result = _run_tilelang(
            q, kv, indices, topk_length, attn_sink, out,
            dim=args.dim, block_i=args.block_i, config=config,
            sm_scale=sm_scale, kernel_output_dtype=kernel_output_dtype,
            pv_gemm_policy=args.pv_gemm_policy,
            assume_valid_indices=args.assume_valid_indices)
        ref = _reference(q, kv, indices, dim=args.dim, sm_scale=sm_scale)
        torch.testing.assert_close(
            result[0].float(), ref[0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            result[1], ref[1], atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(
            result[2], ref[2], atol=2e-3, rtol=2e-3)
        print("correctness: ok", flush=True)
    except Exception as exc:
        payload = {
            "config": config.__dict__,
            "status": "failed",
            "phase": "correctness_or_compile",
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0],
        }
        print("RESULT_JSON " + json.dumps(payload, sort_keys=True), flush=True)
        return 1

    case_results = []
    for case in args.case:
        q, kv, indices, topk_length, attn_sink, out = _make_inputs(
            s_q=case.s_q,
            s_kv=case.s_kv,
            heads=args.heads,
            dim=args.dim,
            tail_dim=args.tail_dim,
            topk=args.topk,
            seed=17 + case.s_q,
            device=device,
        )

        def fn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _run_tilelang(
                q, kv, indices, topk_length, attn_sink, out,
                dim=args.dim, block_i=args.block_i, config=config,
                sm_scale=sm_scale, kernel_output_dtype=kernel_output_dtype,
                pv_gemm_policy=args.pv_gemm_policy,
                assume_valid_indices=args.assume_valid_indices)

        row = _measure(fn, warmup=args.warmup, repeat=args.repeat)
        row.update({
            "s_q": case.s_q,
            "s_kv": case.s_kv,
        })
        case_results.append(row)
        print(
            f"bench s_q={case.s_q} s_kv={case.s_kv}: "
            f"median={row['median_ms']:.3f} ms "
            f"min={row['min_ms']:.3f} ms "
            f"peak={row['peak_allocated_mib']:.1f} MiB",
            flush=True,
        )
    payload = {
        "config": config.__dict__,
        "pv_gemm_policy": args.pv_gemm_policy,
        "assume_valid_indices": args.assume_valid_indices,
        "status": "ok",
        "cases": case_results,
    }
    print("RESULT_JSON " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _run_child_with_watchdog(
    args: argparse.Namespace,
    config: Config,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--_child",
        "--child-config",
        f"{config.heads_per_block},{config.num_stages},{config.threads}",
        "--heads",
        str(args.heads),
        "--dim",
        str(args.dim),
        "--tail-dim",
        str(args.tail_dim),
        "--topk",
        str(args.topk),
        "--block-i",
        str(args.block_i),
        "--correctness-s-q",
        str(args.correctness_s_q),
        "--correctness-s-kv",
        str(args.correctness_s_kv),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--kernel-output-dtype",
        args.kernel_output_dtype,
        "--pv-gemm-policy",
        args.pv_gemm_policy,
    ]
    if args.assume_valid_indices:
        cmd.append("--assume-valid-indices")
    for case in args.case:
        cmd.extend(["--case", f"s_q={case.s_q},s_kv={case.s_kv}"])

    baseline_artifacts = _latest_tvm_artifacts(limit=64)
    baseline_mtime = max(
        (float(item["mtime"]) for item in baseline_artifacts),
        default=0.0,
    )
    child = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    output_lines: list[str] = []

    def forward_output() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)

    output_thread = threading.Thread(target=forward_output, daemon=True)
    output_thread.start()
    start_time = time.time()
    saw_fresh_artifact = False
    saw_compiler_child = False
    while True:
        returncode = child.poll()
        if returncode is not None:
            output_thread.join(timeout=2)
            payload = None
            for line in reversed(output_lines):
                if line.startswith("RESULT_JSON "):
                    payload = json.loads(line[len("RESULT_JSON "):])
                    break
            if payload is None:
                payload = {
                    "config": config.__dict__,
                    "status": "failed",
                    "phase": "child_exit",
                    "error_type": "ChildFailed",
                    "error": f"returncode={returncode}",
                }
            payload["returncode"] = returncode
            return payload

        elapsed = time.time() - start_time
        status = _read_proc_status(child.pid)
        cpu_seconds = _read_proc_cpu_seconds(child.pid)
        artifacts = _latest_tvm_artifacts()
        fresh_artifacts = [
            item for item in artifacts if float(item["mtime"]) > baseline_mtime
        ]
        saw_fresh_artifact = saw_fresh_artifact or bool(fresh_artifacts)
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
                f"{p['pid']}:{p['name']}:{p['state']}:{p['rss']}:"
                f"{p['cmdline']}"
                for p in children
            )
            if children else "none"
        )
        print(
            "# watchdog: "
            f"hpb={config.heads_per_block} "
            f"stages={config.num_stages} "
            f"threads={config.threads} "
            f"elapsed={elapsed:.1f}s "
            f"state={status.get('State', 'gone')} "
            f"rss={status.get('VmRSS', 'unknown')} "
            f"cpu={cpu_seconds if cpu_seconds is not None else 'unknown'}s "
            f"children={child_summary} "
            f"gpu={_query_gpu_processes(child.pid)}",
            flush=True,
        )
        if artifacts:
            label = (
                "fresh TVM artifacts"
                if fresh_artifacts
                else "fresh TVM artifacts: none; latest pre-existing artifacts"
            )
            print(f"# watchdog: {label}:", flush=True)
            for artifact in (fresh_artifacts or artifacts)[:4]:
                print(f"# watchdog:   {_format_artifact(artifact)}",
                      flush=True)
        else:
            print("# watchdog: TVM artifacts: none", flush=True)

        if elapsed >= args.compile_timeout_seconds:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            output_thread.join(timeout=2)
            return {
                "config": config.__dict__,
                "status": "timeout",
                "phase": "compile_or_run",
                "error_type": "TimeoutExpired",
                "error": (
                    f"timeout after {args.compile_timeout_seconds}s; "
                    f"saw_fresh_artifact={saw_fresh_artifact}; "
                    f"saw_compiler_child={saw_compiler_child}"
                ),
                "returncode": 124,
            }

        time.sleep(max(1, args.watchdog_interval_seconds))


def _write_markdown(path: str, results: list[dict[str, Any]]) -> None:
    lines = [
        "# TileLang Sparse Prefill HPB/Stage Sweep",
        "",
        "| HPB | stages | threads | s_q | s_kv | median ms | min ms | peak MiB | status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        config = item["config"]
        if item.get("status") == "ok":
            for case in item["cases"]:
                lines.append(
                    f"| {config['heads_per_block']} | "
                    f"{config['num_stages']} | {config['threads']} | "
                    f"{case['s_q']} | {case['s_kv']} | "
                    f"{case['median_ms']:.3f} | {case['min_ms']:.3f} | "
                    f"{case['peak_allocated_mib']:.1f} | "
                    f"ok pv={item.get('pv_gemm_policy', 'full_row')} "
                    f"valid={item.get('assume_valid_indices', False)} |"
                )
        else:
            lines.append(
                f"| {config['heads_per_block']} | "
                f"{config['num_stages']} | {config['threads']} |  |  |  |  |  | "
                f"{item.get('error_type')}: {item.get('error')} |"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the real vLLM TileLang sparse-prefill kernel over "
            "heads_per_block and num_stages on SM70."
        )
    )
    parser.add_argument("--config", action="append", type=_parse_config)
    parser.add_argument("--case", action="append", type=_parse_case)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--tail-dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--block-i", type=int, default=16)
    parser.add_argument("--correctness-s-q", type=int, default=2)
    parser.add_argument("--correctness-s-kv", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--compile-timeout-seconds", type=int, default=900)
    parser.add_argument("--watchdog-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--kernel-output-dtype", choices=("float16", "bfloat16"),
        default="bfloat16")
    parser.add_argument(
        "--pv-gemm-policy",
        choices=("full_row", "full_col", "square"),
        default="full_row")
    parser.add_argument("--assume-valid-indices", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    parser.add_argument("--_child", action="store_true")
    parser.add_argument("--child-config", type=_parse_config)
    args = parser.parse_args()
    if args.case is None:
        args.case = [BenchCase(64, 4096), BenchCase(1024, 4096)]

    if args._child:
        if args.child_config is None:
            raise ValueError("--_child requires --child-config")
        return _child_main(args)

    configs = args.config or [
        Config(64, 1, 128),
        Config(64, 2, 128),
        Config(32, 1, 64),
        Config(32, 2, 64),
        Config(16, 1, 64),
        Config(16, 2, 64),
        Config(16, 1, 128),
        Config(16, 2, 128),
    ]
    results = []
    for config in configs:
        print(
            f"=== hpb={config.heads_per_block} "
            f"stages={config.num_stages} threads={config.threads} ===",
            flush=True,
        )
        result = _run_child_with_watchdog(args, config)
        results.append(result)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.markdown_out:
        _write_markdown(args.markdown_out, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
