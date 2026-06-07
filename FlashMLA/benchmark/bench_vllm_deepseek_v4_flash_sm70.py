#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/mnt/data/apps/vllm"))

CHECKS = [
    {
        "name": "DeepseekV4ForCausalLM registry entry",
        "path": "vllm/model_executor/models/registry.py",
        "needle": "DeepseekV4ForCausalLM",
    },
    {
        "name": "DeepseekV4FlashMLASparseBackend selection",
        "path": "vllm/model_executor/layers/deepseek_v4_attention.py",
        "needle": "DeepseekV4FlashMLASparseBackend",
    },
    {
        "name": "decode calls flash_mla_with_kvcache",
        "path": "vllm/model_executor/layers/deepseek_v4_attention.py",
        "needle": "flash_mla_with_kvcache",
    },
    {
        "name": "prefill calls flash_mla_sparse_fwd",
        "path": "vllm/model_executor/layers/deepseek_v4_attention.py",
        "needle": "flash_mla_sparse_fwd",
    },
    {
        "name": "DeepSeek V4 fp8_ds_mla cache format",
        "path": "vllm/model_executor/layers/deepseek_v4_attention.py",
        "needle": "fp8_ds_mla",
    },
    {
        "name": "MODEL1 584B cache shape",
        "path": "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "needle": "584",
    },
    {
        "name": "vLLM FlashMLA sparse backend gate",
        "path": "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "needle": "major in [9, 10]",
    },
    {
        "name": "vLLM FlashMLA sparse runtime gate",
        "path": "vllm/v1/attention/ops/flashmla.py",
        "needle": "is_device_capability_family(90)",
    },
]

SM70_GATE_BLOCKERS = [
    {
        "id": "vllm_flashmla_sparse_backend_sm70_gate",
        "check_name": "vLLM FlashMLA sparse backend gate",
        "summary": (
            "DeepseekV4FlashMLASparseBackend.supports_compute_capability "
            "still restricts FlashMLA sparse to SM90/SM100."
        ),
        "task": (
            "Patch vLLM FlashMLA sparse backend capability checks to allow "
            "SM70 only when the FlashMLA SM70 sparse decode/prefill support "
            "matrix is satisfied."
        ),
    },
    {
        "id": "vllm_flashmla_sparse_runtime_sm70_gate",
        "check_name": "vLLM FlashMLA sparse runtime gate",
        "summary": (
            "vllm.v1.attention.ops.flashmla.is_flashmla_sparse_supported() "
            "still rejects SM70 at runtime."
        ),
        "task": (
            "Patch the vLLM FlashMLA runtime support probe to include SM70, "
            "then smoke the DeepSeek V4 FlashMLA sparse backend on a V100."
        ),
    },
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text()


def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def inspect_vllm(root: Path) -> dict[str, Any]:
    checks = []
    for item in CHECKS:
        path = root / item["path"]
        exists = path.is_file()
        text = read_text(path) if exists else ""
        checks.append(
            {
                "name": item["name"],
                "path": str(path),
                "needle": item["needle"],
                "ok": exists and item["needle"] in text,
            }
        )
    checks_by_name = {check["name"]: check for check in checks}
    sm70_blockers = [
        {
            "id": blocker["id"],
            "path": checks_by_name[blocker["check_name"]]["path"],
            "summary": blocker["summary"],
            "task": blocker["task"],
        }
        for blocker in SM70_GATE_BLOCKERS
        if checks_by_name.get(blocker["check_name"], {}).get("ok")
    ]

    return {
        "vllm_root": str(root),
        "git_branch": git_value(root, "branch", "--show-current"),
        "git_commit": git_value(root, "rev-parse", "--short", "HEAD"),
        "git_status_short": git_value(root, "status", "--short"),
        "checks": checks,
        "sm70_blockers": sm70_blockers,
        "end_to_end_smoke_ready": all(check["ok"] for check in checks) and not sm70_blockers,
        "contract": {
            "decode_api": (
                "DeepSeek V4 decode uses flash_mla_with_kvcache with "
                "is_fp8_kvcache=True, indices/topk_length, attn_sink, and "
                "optional extra_k_cache/extra_indices_in_kvcache."
            ),
            "prefill_api": (
                "DeepSeek V4 prefill uses flash_mla_sparse_fwd after vLLM "
                "gathers and dequantizes compressed/SWA KV into a BF16 workspace."
            ),
            "sm70_gate": (
                "Current vLLM FlashMLA sparse gates allow SM90/SM100 only; "
                "SM70 must be enabled in vLLM before an end-to-end smoke can "
                "select the FlashMLA SM70 sparse alpha."
            ),
            "flashmla_sm70_boundary": (
                "FlashMLA SM70 currently supports sparse decode and sparse prefill "
                "for the documented DeepSeek V4 FlashMLA shapes; "
                "remaining end-to-end blockers are outside the FlashMLA "
                "kernel dispatch unless the smoke reaches a new kernel error."
            ),
            "smoke_preflight": (
                "Run openai-stream only after end_to_end_smoke_ready is true, "
                "or record the first blocker as a vLLM integration task rather "
                "than a FlashMLA kernel failure."
            ),
        },
    }


def print_inspect(report: dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"vLLM root: {report['vllm_root']}")
        print(f"branch: {report.get('git_branch') or 'unknown'}")
        print(f"commit: {report.get('git_commit') or 'unknown'}")
        if report.get("git_status_short"):
            print("status:")
            print(report["git_status_short"])
        print("\nstatic checks:")
        for check in report["checks"]:
            mark = "ok" if check["ok"] else "missing"
            print(f"- {mark}: {check['name']} ({check['needle']})")
        print("\ncontract:")
        for value in report["contract"].values():
            print(f"- {value}")
        print(f"\nend_to_end_smoke_ready: {report['end_to_end_smoke_ready']}")
        if report["sm70_blockers"]:
            print("sm70 blockers:")
            for blocker in report["sm70_blockers"]:
                print(f"- {blocker['id']}: {blocker['summary']}")

    return 0 if all(check["ok"] for check in report["checks"]) else 2


def parse_sse_line(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return {"done": True}
    return json.loads(payload)


def run_openai_stream(args: argparse.Namespace, inspect_report: dict[str, Any]) -> int:
    if not args.model:
        raise SystemExit("--model is required for --mode openai-stream")

    endpoint = args.endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()
    first_token_time: float | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    fallback_delta_count = 0
    output_parts: list[str] = []

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                chunk = parse_sse_line(raw_line)
                if chunk is None:
                    continue
                if chunk.get("done"):
                    break
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    fallback_delta_count += 1
                    output_parts.append(content)
    except urllib.error.URLError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    end = time.perf_counter()
    generated_units = completion_tokens if completion_tokens is not None else fallback_delta_count
    decode_window_s = None
    decode_tokens_per_s = None
    if first_token_time is not None:
        decode_window_s = max(end - first_token_time, 1e-9)
        decode_tokens_per_s = generated_units / decode_window_s

    result = {
        "mode": "openai-stream",
        "endpoint": endpoint,
        "model": args.model,
        "TTFT": None if first_token_time is None else first_token_time - start,
        "total_s": end - start,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "delta_chunks_when_usage_missing": fallback_delta_count,
        "decode_tokens_per_s": decode_tokens_per_s,
        "decode_window_s": decode_window_s,
        "output_preview": "".join(output_parts)[: args.preview_chars],
        "static_inspect": inspect_report,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or benchmark the vLLM DeepSeek V4 FlashMLA path for SM70. "
            "Use inspect before attempting an end-to-end OpenAI-compatible smoke."
        )
    )
    parser.add_argument("--mode", choices=["inspect", "openai-stream"], default="inspect")
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit JSON for inspect mode.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--preview-chars", type=int, default=400)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inspect_report = inspect_vllm(args.vllm_root)
    if args.mode == "inspect":
        return print_inspect(inspect_report, args.json)
    return run_openai_stream(args, inspect_report)


if __name__ == "__main__":
    raise SystemExit(main())
