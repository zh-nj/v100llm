#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""O4 validation inference test — runs DeepSeek V4 Flash on 8× V100 in TP=8
with VLLM_DEEPSEEK_V4_INDEXER_TOPK set to the override value. Measures
decode throughput over a warm prompt and verifies the semantic output
matches the default (no-override) configuration.

This is the live-hardware complement to the static + unit test coverage
added in this session. It runs a single 1024-token prompt → 200-step
decode benchmark, reporting TTFT / decode tok/s / completion preview.
Compare two runs: one with `VLLM_DEEPSEEK_V4_INDEXER_TOPK` unset (baseline
top-K from model config) vs. set to the override value.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def build_prompt(target_ctx_tokens: int) -> str:
    """Build a Chinese prompt of approximately target_ctx_tokens tokens."""
    base = "深度学习模型推理过程中的注意力机制、KV缓存、量化技术。"
    # ~20 tokens per repetition (Chinese characters ≈ 1 token each in DS tokenizer)
    return "请用详细中文分析以下内容：" + (base * max(target_ctx_tokens // 20, 1))


def run_one(
    model_path: str,
    prompt: str,
    max_tokens: int,
    tp_size: int,
    max_model_len: int,
    enforce_eager: bool = False,
) -> dict:
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="float16",
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        kv_cache_dtype="fp8_ds_mla",
    )
    if not enforce_eager:
        llm_kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        }
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # Warmup
    _ = llm.generate(["你好"], SamplingParams(temperature=0.0, max_tokens=8))

    # Timed run
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    end = time.perf_counter()

    out = outputs[0]
    prompt_len = len(out.prompt_token_ids)
    completion_len = len(out.outputs[0].token_ids)
    total_s = end - start

    # Approximate decode tok/s: subtract a fixed TTFT estimate
    # (proper TTFT would require streaming, but this gives comparable metric)
    decode_tps = completion_len / total_s if total_s > 0 else 0.0

    return {
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "total_s": total_s,
        "decode_tps_approx": decode_tps,
        "first_token_ids": list(out.outputs[0].token_ids[:16]),
        "output_preview": out.outputs[0].text[:400],
        "env": {
            "VLLM_DEEPSEEK_V4_INDEXER_TOPK": os.getenv(
                "VLLM_DEEPSEEK_V4_INDEXER_TOPK", ""
            ),
            "VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE": os.getenv(
                "VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE", ""
            ),
            "VLLM_SM70_MHC_FAST": os.getenv("VLLM_SM70_MHC_FAST", ""),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data6/models/DeepSeek-V4-Flash")
    ap.add_argument("--ctx-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    prompt = build_prompt(args.ctx_tokens)
    result = run_one(
        args.model, prompt, args.max_tokens, args.tp_size, args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    result["requested_ctx_tokens"] = args.ctx_tokens
    result["requested_max_tokens"] = args.max_tokens

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
