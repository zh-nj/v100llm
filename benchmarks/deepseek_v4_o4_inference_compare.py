#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""O4 validation — back-to-back semantic + throughput comparison of
baseline vs. O4 override (VLLM_DEEPSEEK_V4_INDEXER_TOPK=256) inside a single
process. Each engine is constructed, run, torn down, and the next one is
constructed. This exercises the override code path end-to-end (envs are read
at engine construction time).

Supports FULL_DECODE_ONLY cudagraph mode with inductor combo_kernels disabled
(works around torch inductor decompose_triton_kernel_wrapper_functional
AssertionError).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


def build_prompt(target_ctx_tokens: int) -> str:
    base = "深度学习模型推理过程中的注意力机制、KV缓存、量化技术。"
    return "请用详细中文分析以下内容：" + (base * max(target_ctx_tokens // 20, 1))


def run_with_engine(
    model_path: str,
    prompt: str,
    max_tokens: int,
    tp_size: int,
    max_model_len: int,
    enforce_eager: bool,
    tag: str,
) -> dict:
    import torch
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
        # Use CompilationMode.NONE to skip inductor AOT compile (which on
        # torch 2.9 + our SM70 triton kernels triggers
        # decompose_triton_kernel_wrapper_functional assertion failures and
        # dtype-promotion issues in the piecewise compiled subgraphs). We
        # still get CUDA graph capture via cudagraph_mode=FULL_DECODE_ONLY —
        # the decode-phase launch overhead is the main thing that graphs save.
        llm_kwargs["compilation_config"] = {
            "mode": 0,  # CompilationMode.NONE
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        }
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    # Warmup — avoid compile/jit from polluting the measurement.
    _ = llm.generate(["你好"], SamplingParams(temperature=0.0, max_tokens=8))

    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    end = time.perf_counter()

    out = outputs[0]
    prompt_len = len(out.prompt_token_ids)
    completion_len = len(out.outputs[0].token_ids)
    total_s = end - start

    result = {
        "tag": tag,
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "total_s": total_s,
        "tps_overall": completion_len / total_s if total_s > 0 else 0.0,
        "first_token_ids": list(out.outputs[0].token_ids[:32]),
        "output_preview": out.outputs[0].text[:400],
        "env": {
            "VLLM_DEEPSEEK_V4_INDEXER_TOPK": os.getenv(
                "VLLM_DEEPSEEK_V4_INDEXER_TOPK", ""
            ),
        },
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data6/models/DeepSeek-V4-Flash")
    ap.add_argument("--ctx-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--topk-override", type=int, default=256)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    prompt = build_prompt(args.ctx_tokens)
    results: list[dict] = []

    # Run 1: baseline (unset override → model config default).
    os.environ.pop("VLLM_DEEPSEEK_V4_INDEXER_TOPK", None)
    results.append(run_with_engine(
        args.model, prompt, args.max_tokens, args.tp_size, args.max_model_len,
        args.enforce_eager, tag="baseline_no_override",
    ))

    # Run 2: with override.
    os.environ["VLLM_DEEPSEEK_V4_INDEXER_TOPK"] = str(args.topk_override)
    results.append(run_with_engine(
        args.model, prompt, args.max_tokens, args.tp_size, args.max_model_len,
        args.enforce_eager, tag=f"o4_topk{args.topk_override}",
    ))

    # Derive deltas.
    baseline = results[0]
    fixed = results[1]
    match_prefix = sum(
        1 for a, b in zip(baseline["first_token_ids"], fixed["first_token_ids"])
        if a == b
    )
    delta = {
        "baseline_tag": baseline["tag"],
        "fixed_tag": fixed["tag"],
        "baseline_tps": baseline["tps_overall"],
        "fixed_tps": fixed["tps_overall"],
        "tps_ratio": (fixed["tps_overall"] / baseline["tps_overall"])
        if baseline["tps_overall"] > 0 else None,
        "semantic_prefix_match_len": match_prefix,
        "semantic_full_match": baseline["first_token_ids"] == fixed["first_token_ids"],
    }

    payload = {"runs": results, "delta": delta}
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
