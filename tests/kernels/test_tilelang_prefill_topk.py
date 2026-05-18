# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the experimental TileLang prefill top-k kernel."""

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="This test requires CUDA",
)
@pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="TileLang is not installed",
)
@torch.inference_mode()
def test_tilelang_prefill_topk_512_matches_torch_topk_sets() -> None:
    from vllm.v1.attention.ops.tilelang_prefill_topk import prefill_topk_tilelang

    torch.set_default_device("cuda:0")
    torch.manual_seed(20260517)
    rows = 4
    row_len = 1024
    top_k = 512
    row_starts = torch.tensor([0, 3, 5, 7], dtype=torch.int32, device="cuda")
    lengths = torch.full((rows,), row_len, dtype=torch.int32, device="cuda")
    logits = torch.randn(
        (rows, row_len + int(row_starts.max().item())),
        dtype=torch.float32,
        device="cuda",
    )
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")

    prefill_topk_tilelang(
        logits,
        indices,
        lengths,
        row_starts,
        topk_tokens=top_k,
        threads=128,
    )

    for row in range(rows):
        start = int(row_starts[row].item())
        expected = logits[row, start:start + row_len].topk(top_k).indices.to(
            torch.int32
        )
        assert indices[row].min().item() >= 0
        assert indices[row].max().item() < row_len
        assert set(indices[row].tolist()) == set(expected.tolist())


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="This test requires CUDA",
)
@pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="TileLang is not installed",
)
@torch.inference_mode()
def test_tilelang_prefill_topk_512_handles_causal_short_prefix() -> None:
    import vllm._C  # noqa: F401
    from vllm.v1.attention.ops.tilelang_prefill_topk import prefill_topk_tilelang

    torch.set_default_device("cuda:0")
    torch.manual_seed(20260518)
    rows = 768
    max_len = 1024
    top_k = 512
    row_starts = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    lengths = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")
    logits = torch.randn((rows, max_len), dtype=torch.float32, device="cuda")
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")

    prefill_topk_tilelang(
        logits,
        indices,
        lengths,
        row_starts,
        topk_tokens=top_k,
        threads=256,
        causal_row_offset=0,
    )

    for row in (0, 511, 512, rows - 1):
        row_len = int(lengths[row].item())
        expected = logits[row, :row_len].topk(min(top_k, row_len)).indices.to(
            torch.int32
        )
        got = indices[row, :expected.shape[0]]
        assert set(got.tolist()) == set(expected.tolist())
        if row_len < top_k:
            assert torch.all(indices[row, row_len:] == -1)
