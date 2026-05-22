# SPDX-License-Identifier: Apache-2.0
"""Gemma4 SM70 MTP request-level benchmark helper."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vllm as vllm_pkg
from vllm import LLM, SamplingParams


EXPECTED_CODE = "GEMMA-MTP-42"


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def build_prompt_token_ids(
    tokenizer: Any,
    target_len: int,
    variant: int,
    expected_code: str,
) -> list[int]:
    bos = []
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        bos = [bos_id]

    prefix = (
        "You are a careful retrieval assistant. Read the whole context. "
        "At the end, answer the question. Do not invent facts.\n"
        "Context:\n"
    )
    suffix = (
        f"\nImportant record for benchmark case {variant}: "
        f"the verification code is {expected_code}.\n"
        "Question: What is the verification code? Reply with the code first, "
        "then one short Chinese sentence explaining that you found it in the "
        "context.\n"
        "Answer:"
    )
    filler_unit = (
        f" neutral context line {variant}: alpha beta gamma delta epsilon "
        "zeta eta theta.\n"
    )

    prefix_ids = _encode_no_special(tokenizer, prefix)
    suffix_ids = _encode_no_special(tokenizer, suffix)
    filler_ids = _encode_no_special(tokenizer, filler_unit)
    skeleton = bos + prefix_ids + suffix_ids
    if len(skeleton) > target_len:
        raise ValueError(
            f"target_len={target_len} is smaller than prompt skeleton "
            f"({len(skeleton)} tokens)."
        )

    remaining = target_len - len(skeleton)
    filler = (filler_ids * (remaining // len(filler_ids) + 1))[:remaining]
    prompt_ids = bos + prefix_ids + filler + suffix_ids
    assert len(prompt_ids) == target_len, (len(prompt_ids), target_len)
    return prompt_ids


def semantic_quality(output_text: str, expected_code: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", output_text, flags=re.S).strip()
    has_code = expected_code in stripped
    has_sentence = bool(
        re.search(r"[。.!！]|context|found|上下文|找到|记录", stripped)
    )
    if has_code and has_sentence:
        return "pass"
    if has_code:
        return "partial"
    return "fail"


def _metric_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_value(v) for v in value]
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/data6/models/gemma-4-31B-it-AWQ-4bit",
    )
    parser.add_argument(
        "--draft-model",
        default="/mnt/data6/models/gemma-4-31B-it-assistant",
    )
    parser.add_argument("--input-lens", type=int, nargs="+", required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--expected-code", default=EXPECTED_CODE)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--warmup-len", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    max_model_len = args.max_model_len
    if max_model_len is None:
        max_model_len = max(max(args.input_lens) + args.max_tokens + 128, 2048)
    print(
        json.dumps(
            {
                "event": "bench_start",
                "model": args.model,
                "draft_model": args.draft_model,
                "targets": args.input_lens,
                "max_tokens": args.max_tokens,
                "max_model_len": max_model_len,
                "attention_backend": args.attention_backend,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
                "vllm_file": getattr(vllm_pkg, "__file__", None),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    llm: LLM | None = None
    try:
        llm_kwargs: dict[str, Any] = {}
        if args.attention_backend:
            llm_kwargs["attention_backend"] = args.attention_backend

        start = time.perf_counter()
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=max_model_len,
            enable_prefix_caching=False,
            disable_log_stats=False,
            speculative_config={
                "model": args.draft_model,
                "method": "mtp",
                "num_speculative_tokens": args.num_speculative_tokens,
            },
            **llm_kwargs,
        )
        print(
            json.dumps(
                {"event": "engine_ready", "startup_s": time.perf_counter() - start},
                ensure_ascii=False,
            ),
            flush=True,
        )

        tokenizer = llm.get_tokenizer()
        if args.warmup_len > 0:
            warm_prompt = build_prompt_token_ids(
                tokenizer, args.warmup_len, 0, args.expected_code
            )
            warm_out = llm.generate(
                [{"prompt_token_ids": warm_prompt}],
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=args.warmup_tokens,
                ),
                use_tqdm=False,
            )[0]
            print(
                json.dumps(
                    {
                        "event": "warmup_done",
                        "generated_tokens": len(warm_out.outputs[0].token_ids),
                        "finish_reason": warm_out.outputs[0].finish_reason,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            min_tokens=1,
            max_tokens=args.max_tokens,
        )
        for idx, input_len in enumerate(args.input_lens, start=1):
            prompt_ids = build_prompt_token_ids(
                tokenizer, input_len, idx, args.expected_code
            )
            case_start = time.perf_counter()
            output = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                sampling_params=sampling_params,
                use_tqdm=False,
            )[0]
            case_end = time.perf_counter()

            completion = output.outputs[0]
            metrics = output.metrics
            prompt_tokens = len(output.prompt_token_ids or prompt_ids)
            generated_tokens = len(completion.token_ids)
            first_latency = (
                None
                if metrics is None or metrics.first_token_latency is None
                else float(metrics.first_token_latency)
            )
            first_ts = (
                None
                if metrics is None or metrics.first_token_ts is None
                else float(metrics.first_token_ts)
            )
            last_ts = (
                None
                if metrics is None or metrics.last_token_ts is None
                else float(metrics.last_token_ts)
            )
            decode_tokens = max(generated_tokens - 1, 0)
            decode_window = (
                None
                if first_ts is None or last_ts is None
                else max(last_ts - first_ts, 1e-9)
            )
            decode_tps = (
                None
                if decode_window is None
                else decode_tokens / decode_window if decode_tokens else 0.0
            )
            prefill_tps = (
                None
                if first_latency is None
                else prompt_tokens / max(first_latency, 1e-9)
            )

            print(
                json.dumps(
                    {
                        "input_len": input_len,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "prefill_tokens_per_s": prefill_tps,
                        "decode_tokens_per_s": decode_tps,
                        "TTFT_s": first_latency,
                        "decode_window_s": decode_window,
                        "wall_s": case_end - case_start,
                        "finish_reason": completion.finish_reason,
                        "semantic_quality": semantic_quality(
                            completion.text, args.expected_code
                        ),
                        "output_text": completion.text,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        spec_metrics = []
        for metric in llm.get_metrics():
            if metric.name.startswith("vllm:spec_decode"):
                spec_metrics.append(
                    {
                        "name": metric.name,
                        "value": _metric_value(
                            getattr(metric, "value", getattr(metric, "values", None))
                        ),
                    }
                )
        print(
            json.dumps(
                {"event": "spec_decode_metrics", "metrics": spec_metrics},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "bench_error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1
    finally:
        if llm is not None:
            try:
                llm.llm_engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
