# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import re
from typing import Any, Protocol


class _TokenizerLike(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]: ...


def normalize_output_text(output_text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()


def semantic_quality_conclusion(output_text: str, expected_answer: str) -> str:
    return (
        "pass"
        if normalize_output_text(output_text) == expected_answer.strip()
        else "fail"
    )


def build_result_row(
    *,
    input_len: int,
    prompt_tokens: int,
    generated_tokens: int,
    finish_reason: str | None,
    output_text: str,
    expected_answer: str,
    metrics: Any,
) -> dict[str, Any]:
    ttft = float(metrics.first_token_latency)
    decode_time = max(float(metrics.last_token_ts - metrics.first_token_ts), 1e-6)
    decode_tokens = max(generated_tokens - 1, 0)
    return {
        "input_len": input_len,
        "prefill tokens/s": prompt_tokens / max(ttft, 1e-6),
        "decode tokens/s": decode_tokens / decode_time if decode_tokens else 0.0,
        "TTFT": ttft * 1000.0,
        "finish_reason": finish_reason or "unknown",
        "semantic_quality": semantic_quality_conclusion(
            output_text, expected_answer
        ),
        "output_text": output_text,
    }


def _encode_no_special_tokens(tokenizer: _TokenizerLike, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def build_exact_retrieval_prompt_token_ids(
    tokenizer: _TokenizerLike,
    target_len: int,
    expected_answer: str,
) -> list[int]:
    prefix = (
        "You are doing exact retrieval.\n"
        "Read the context carefully and return only the verification code.\n"
        "Do not add words, punctuation, or explanation.\n"
        "Context begins below.\n"
    )
    needle = f"\nCritical fact: the verification code is {expected_answer}.\n"
    suffix = "\nQuestion: What is the verification code?\nAnswer:"
    prefix_ids = _encode_no_special_tokens(tokenizer, prefix)
    needle_ids = _encode_no_special_tokens(tokenizer, needle)
    suffix_ids = _encode_no_special_tokens(tokenizer, suffix)
    min_len = len(prefix_ids) + len(needle_ids) + len(suffix_ids)
    if min_len > target_len:
        raise ValueError(
            f"target_len={target_len} is smaller than the prompt skeleton "
            f"({min_len} tokens)."
        )
    filler_ids = _encode_no_special_tokens(
        tokenizer,
        " filler filler filler filler filler filler filler filler",
    )
    remaining = target_len - min_len
    filler_prefix = (filler_ids * ((remaining // len(filler_ids)) + 1))[:remaining]
    return prefix_ids + filler_prefix + needle_ids + suffix_ids


def run_benchmark(args) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    def _arg(name: str) -> Any:
        return getattr(args, name, None)

    llm_kwargs = {
        "model": _arg("model"),
        "tokenizer": _arg("tokenizer"),
        "tokenizer_mode": _arg("tokenizer_mode"),
        "skip_tokenizer_init": _arg("skip_tokenizer_init"),
        "trust_remote_code": _arg("trust_remote_code"),
        "tensor_parallel_size": _arg("tensor_parallel_size"),
        "dtype": _arg("dtype"),
        "quantization": _arg("quantization"),
        "revision": _arg("revision"),
        "tokenizer_revision": _arg("tokenizer_revision"),
        "seed": _arg("seed"),
        "gpu_memory_utilization": _arg("gpu_memory_utilization"),
        "swap_space": _arg("swap_space"),
        "cpu_offload_gb": _arg("cpu_offload_gb"),
        "enforce_eager": _arg("enforce_eager"),
        "max_model_len": _arg("max_model_len"),
        "disable_log_stats": False,
    }
    llm = LLM(
        **{key: value for key, value in llm_kwargs.items() if value is not None}
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        min_tokens=1,
        max_tokens=args.max_tokens,
    )
    rows: list[dict[str, Any]] = []
    for input_len in args.input_lens:
        prompt_token_ids = build_exact_retrieval_prompt_token_ids(
            tokenizer,
            input_len,
            args.expected_answer,
        )
        output = llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0]
        completion = output.outputs[0]
        row = build_result_row(
            input_len=input_len,
            prompt_tokens=len(output.prompt_token_ids or prompt_token_ids),
            generated_tokens=len(completion.token_ids),
            finish_reason=completion.finish_reason,
            output_text=completion.text,
            expected_answer=args.expected_answer,
            metrics=output.metrics,
        )
        rows.append(row)
    return rows


def main(args) -> None:
    for row in run_benchmark(args):
        print(json.dumps(row, ensure_ascii=False))


def create_argument_parser():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Benchmark SM70 FP8 exact retrieval quality and latency."
    )
    parser.add_argument("--input-lens", type=int, nargs="+", required=True)
    parser.add_argument("--expected-answer", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    return EngineArgs.add_cli_args(parser)


if __name__ == "__main__":
    parser = create_argument_parser()
    main(parser.parse_args())
