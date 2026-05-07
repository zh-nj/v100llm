#!/usr/bin/env python3
"""Reproduce the 17 tok/s baseline measurement from HEAD commit."""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    from vllm import LLM, SamplingParams

    # Match the 17 tok/s baseline configuration exactly.
    llm = LLM(
        model="/mnt/data6/models/DeepSeek-V4-Flash",
        tensor_parallel_size=8,
        dtype="float16",
        trust_remote_code=True,
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        kv_cache_dtype="fp8_ds_mla",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
    )

    base = "深度学习模型推理过程中的注意力机制、KV缓存、量化技术。"
    prompt = "请用详细中文分析以下内容：" + (base * max(1024 // 20, 1))

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)
    _ = llm.generate(["你好"], SamplingParams(temperature=0.0, max_tokens=8))

    start = time.perf_counter()
    outputs = llm.generate([prompt], sp)
    end = time.perf_counter()

    out = outputs[0]
    print(json.dumps({
        "prompt_tokens": len(out.prompt_token_ids),
        "completion_tokens": len(out.outputs[0].token_ids),
        "total_s": end - start,
        "tps": len(out.outputs[0].token_ids) / (end - start),
        "first_token_ids": list(out.outputs[0].token_ids[:32]),
        "output_preview": out.outputs[0].text[:200],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
