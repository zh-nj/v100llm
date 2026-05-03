#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/mnt/data6/models/DeepSeek-V4-Flash")
FLASHMLA_SRC_PATH = Path(
    os.getenv("FLASH_MLA_SRC_DIR", "/mnt/data/apps/FlashMLA")
)


CHECKS = [
    ("registry", "vllm/model_executor/models/registry.py", "DeepseekV4ForCausalLM"),
    (
        "backend",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "DeepseekV4FlashMLASparseBackend",
    ),
    (
        "decode_api",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "flash_mla_with_kvcache",
    ),
    (
        "prefill_api",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "flash_mla_sparse_fwd",
    ),
    (
        "cache_dtype",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "fp8_ds_mla",
    ),
    (
        "model1_cache_shape",
        "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "584",
    ),
    (
        "sm70_backend_gate",
        "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "major in [7, 9, 10]",
    ),
    (
        "sm70_runtime_gate",
        "vllm/v1/attention/ops/flashmla.py",
        "is_device_capability_family(70)",
    ),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _git_output(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def inspect_flashmla_source() -> dict[str, Any]:
    sparse_decode_sources = [
        "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu",
        "csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu",
    ]
    sparse_prefill_sources = [
        "csrc/sm70/prefill/sparse/instantiations/bf16.cu",
    ]

    return {
        "path": str(FLASHMLA_SRC_PATH),
        "branch": _git_output(FLASHMLA_SRC_PATH, "branch", "--show-current"),
        "head": _git_output(FLASHMLA_SRC_PATH, "rev-parse", "--short", "HEAD"),
        "has_sm70_sparse_decode_sources": all(
            (FLASHMLA_SRC_PATH / relpath).is_file()
            for relpath in sparse_decode_sources
        ),
        "has_sm70_sparse_prefill_sources": all(
            (FLASHMLA_SRC_PATH / relpath).is_file()
            for relpath in sparse_prefill_sources
        ),
    }


def inspect_flashmla_runtime() -> dict[str, Any]:
    report: dict[str, Any] = {
        "vllm_file": None,
        "flashmla_core_importable": False,
        "flashmla_core_import_error": None,
        "sparse_supported": None,
    }

    try:
        import vllm

        report["vllm_file"] = getattr(vllm, "__file__", None)
        importlib.import_module("vllm._flashmla_C")
        report["flashmla_core_importable"] = True
    except Exception as exc:  # noqa: BLE001 - inspect must report import failures.
        report["flashmla_core_import_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from vllm.v1.attention.ops import flashmla

        ok, reason = flashmla.is_flashmla_sparse_supported()
        report["sparse_supported"] = [ok, reason]
    except Exception as exc:  # noqa: BLE001 - inspect must keep going.
        report["sparse_supported"] = [False, f"{type(exc).__name__}: {exc}"]

    return report


def inspect_static() -> dict[str, Any]:
    checks = []
    for name, relpath, needle in CHECKS:
        path = ROOT / relpath
        checks.append(
            {
                "name": name,
                "path": str(path),
                "needle": needle,
                "ok": path.is_file() and needle in _read(path),
            }
        )

    model_cfg_path = MODEL_PATH / "config.json"
    model_cfg = json.loads(_read(model_cfg_path)) if model_cfg_path.is_file() else {}
    compress_ratios = sorted(set(model_cfg.get("compress_ratios", [])))
    model_checks = {
        "model_path": str(MODEL_PATH),
        "config_exists": model_cfg_path.is_file(),
        "architecture": model_cfg.get("architectures", [None])[0],
        "model_type": model_cfg.get("model_type"),
        "head_dim": model_cfg.get("head_dim"),
        "qk_rope_head_dim": model_cfg.get("qk_rope_head_dim"),
        "index_topk": model_cfg.get("index_topk"),
        "compress_ratios": compress_ratios,
    }
    model_ok = (
        model_checks["config_exists"]
        and model_checks["architecture"] == "DeepseekV4ForCausalLM"
        and model_checks["model_type"] == "deepseek_v4"
        and model_checks["head_dim"] == 512
        and model_checks["qk_rope_head_dim"] == 64
        and model_checks["index_topk"] is not None
        and model_checks["index_topk"] <= 8192
        and set(compress_ratios).issubset({0, 4, 128})
    )

    return {
        "root": str(ROOT),
        "flashmla_source": inspect_flashmla_source(),
        "flashmla_runtime": inspect_flashmla_runtime(),
        "checks": checks,
        "model": model_checks,
        "model_ready": model_ok,
        "static_ready": all(check["ok"] for check in checks) and model_ok,
    }


def parse_sse_line(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return {"done": True}
    return json.loads(payload)


def choice_contains_generated_delta(choice: dict[str, Any]) -> bool:
    token_ids = choice.get("token_ids")
    if token_ids:
        return True
    delta = choice.get("delta") or {}
    content = delta.get("content") or ""
    if content:
        return True
    return "content" in delta and "role" not in delta


def run_openai_stream(args: argparse.Namespace) -> int:
    endpoint = args.endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    start = time.perf_counter()
    first_token_time = None
    completion_tokens = None
    finish_reason = None
    pieces: list[str] = []
    delta_chunks = 0

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
                content = (choice.get("delta") or {}).get("content") or ""
                if choice_contains_generated_delta(choice):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    delta_chunks += 1
                if content:
                    pieces.append(content)
    except urllib.error.URLError as exc:
        print(json.dumps({"request_failed": str(exc)}, indent=2))
        return 1

    end = time.perf_counter()
    generated = completion_tokens if completion_tokens is not None else delta_chunks
    decode_window_s = None if first_token_time is None else max(end - first_token_time, 1e-9)
    decode_tokens_per_s = None if decode_window_s is None else generated / decode_window_s
    result = {
        "TTFT": None if first_token_time is None else first_token_time - start,
        "total_s": end - start,
        "completion_tokens": completion_tokens,
        "delta_chunks_when_usage_missing": delta_chunks,
        "decode_window_s": decode_window_s,
        "decode_tokens_per_s": decode_tokens_per_s,
        "finish_reason": finish_reason,
        "output_preview": "".join(pieces)[: args.preview_chars],
    }
    if args.bench_mode is not None:
        result["mode"] = args.bench_mode
    if args.prompt_name is not None:
        result["prompt_name"] = args.prompt_name
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "openai-stream"], default="inspect")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--prompt", default="你好，请用一句话说明你是谁。")
    parser.add_argument("--prompt-name")
    parser.add_argument("--bench-mode", choices=["eager", "full_decode_only"])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--preview-chars", type=int, default=400)
    args = parser.parse_args()

    if args.mode == "inspect":
        report = inspect_static()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"root: {report['root']}")
            print(f"static_ready: {report['static_ready']}")
            for check in report["checks"]:
                print(f"- {'ok' if check['ok'] else 'missing'}: {check['name']}")
            print(f"model: {report['model']}")
        return 0 if report["static_ready"] else 2
    return run_openai_stream(args)


if __name__ == "__main__":
    raise SystemExit(main())
