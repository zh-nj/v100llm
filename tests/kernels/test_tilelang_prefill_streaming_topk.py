# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API tests for the streaming prefill top-k wrapper."""

import pytest
import torch


def _make_inputs():
    rows = 2
    heads = 4
    dim = 16
    kv_tokens = 32
    q = torch.empty((rows, heads, dim), dtype=torch.float16)
    k_cache_values = torch.empty((kv_tokens, dim), dtype=torch.float16)
    k_cache_scales = torch.ones((kv_tokens,), dtype=torch.float32)
    weights = torch.ones((rows, heads), dtype=torch.float32)
    row_starts = torch.zeros((rows,), dtype=torch.int32)
    row_ends = torch.full((rows,), kv_tokens, dtype=torch.int32)
    out_indices = torch.empty((rows, 512), dtype=torch.int32)
    return {
        "q": q,
        "k_cache_values": k_cache_values,
        "k_cache_scales": k_cache_scales,
        "weights": weights,
        "row_starts": row_starts,
        "row_ends": row_ends,
        "out_indices": out_indices,
        "topk_tokens": 512,
    }


def test_streaming_topk_rejects_unsupported_q_dtype():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        prefill_streaming_topk_tilelang,
    )

    kwargs = _make_inputs()
    kwargs["q"] = kwargs["q"].float()

    with pytest.raises(ValueError, match="q must be fp16 or bf16"):
        prefill_streaming_topk_tilelang(**kwargs)


def test_streaming_topk_rejects_bad_output_contract():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        prefill_streaming_topk_tilelang,
    )

    kwargs = _make_inputs()
    kwargs["out_indices"] = kwargs["out_indices"].to(torch.int64)

    with pytest.raises(ValueError, match="out_indices must be a 2D int32 tensor"):
        prefill_streaming_topk_tilelang(**kwargs)

    kwargs = _make_inputs()
    kwargs["topk_tokens"] = 256

    with pytest.raises(ValueError, match="topk_tokens must match"):
        prefill_streaming_topk_tilelang(**kwargs)


def test_streaming_topk_rejects_bad_row_ranges_and_tile_size():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        prefill_streaming_topk_tilelang,
    )

    kwargs = _make_inputs()
    kwargs["row_ends"] = kwargs["row_ends"][:1]

    with pytest.raises(ValueError, match="one row range is required"):
        prefill_streaming_topk_tilelang(**kwargs)

    kwargs = _make_inputs()
    kwargs["tile_k"] = 1000

    with pytest.raises(ValueError, match="tile_k must be"):
        prefill_streaming_topk_tilelang(**kwargs)


def test_streaming_topk_rejects_cpu_tensors():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        prefill_streaming_topk_tilelang,
    )

    kwargs = _make_inputs()

    with pytest.raises(ValueError, match="streaming topk inputs must be CUDA tensors"):
        prefill_streaming_topk_tilelang(**kwargs)
