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


def test_streaming_topk_chunked_torch_matches_full_logits_score_sets():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _prefill_streaming_topk_chunked_torch,
    )

    torch.manual_seed(20260518)
    rows = 4
    heads = 3
    dim = 8
    kv_tokens = 96
    topk_tokens = 16
    q = torch.randn((rows, heads, dim), dtype=torch.float16)
    k_cache_values = torch.randn((kv_tokens, dim), dtype=torch.float16)
    k_cache_scales = torch.linspace(0.8, 1.2, kv_tokens, dtype=torch.float32)
    weights = torch.randn((rows, heads), dtype=torch.float32)
    row_starts = torch.tensor([0, 3, 5, 10], dtype=torch.int32)
    row_ends = torch.tensor([64, 80, 96, 50], dtype=torch.int32)
    out_indices = torch.empty((rows, topk_tokens), dtype=torch.int32)

    _prefill_streaming_topk_chunked_torch(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=32,
    )

    k_f32 = k_cache_values.float() * k_cache_scales.reshape(-1).float().view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        start = int(row_starts[row].item())
        end = int(row_ends[row].item())
        expected = logits[row, start:end].topk(topk_tokens).indices.to(torch.int32)
        assert set(out_indices[row].tolist()) == set(expected.tolist())


def test_streaming_topk_chunked_torch_fills_short_rows_with_minus_one():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _prefill_streaming_topk_chunked_torch,
    )

    rows = 2
    heads = 2
    dim = 4
    kv_tokens = 12
    topk_tokens = 8
    q = torch.randn((rows, heads, dim), dtype=torch.float16)
    k_cache_values = torch.randn((kv_tokens, dim), dtype=torch.float16)
    k_cache_scales = torch.ones((kv_tokens,), dtype=torch.float32)
    weights = torch.ones((rows, heads), dtype=torch.float32)
    row_starts = torch.tensor([0, 3], dtype=torch.int32)
    row_ends = torch.tensor([5, 7], dtype=torch.int32)
    out_indices = torch.empty((rows, topk_tokens), dtype=torch.int32)

    _prefill_streaming_topk_chunked_torch(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=8,
    )

    assert out_indices[0, :5].min().item() >= 0
    assert out_indices[0, :5].max().item() < 5
    assert torch.all(out_indices[0, 5:] == -1)
    assert out_indices[1, :4].min().item() >= 0
    assert out_indices[1, :4].max().item() < 4
    assert torch.all(out_indices[1, 4:] == -1)
