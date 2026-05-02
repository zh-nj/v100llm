# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest

from vllm.model_executor.layers.sparse_attn_indexer import (
    _fp8_mqa_logits_torch_fallback,
    _fp8_paged_mqa_logits_torch_fallback,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")


def _pack_fp8_cache(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, heads, head_dim = values.shape
    cache = torch.empty(
        num_blocks,
        block_size,
        heads,
        head_dim + 4,
        device=values.device,
        dtype=torch.uint8,
    )
    cache[..., :head_dim] = values.view(torch.uint8)
    cache[..., head_dim:] = scales.to(torch.float32).view(torch.uint8)
    return cache


def test_sm70_fallback_prefill_logits_matches_reference() -> None:
    _require_cuda()
    torch.manual_seed(0)
    fp8_dtype = torch.float8_e4m3fn
    seq_len, seq_len_kv, heads, head_dim = 5, 9, 3, 16
    q = torch.randn(seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.float16)
    weights = torch.randn(seq_len, heads, device="cuda", dtype=torch.float32)
    k_scale = torch.rand(seq_len_kv, device="cuda", dtype=torch.float32) + 0.5
    cu_ks = torch.tensor([0, 1, 1, 2, 4], device="cuda", dtype=torch.int32)
    cu_ke = torch.tensor([3, 4, 6, 8, 9], device="cuda", dtype=torch.int32)

    q_fp8 = q.to(fp8_dtype)
    k_fp8 = k.to(fp8_dtype)
    logits = _fp8_mqa_logits_torch_fallback(
        q_fp8, (k_fp8, k_scale), weights, cu_ks, cu_ke
    )

    k_dequant = k_fp8.float() * k_scale.view(-1, 1)
    positions = torch.arange(seq_len_kv, device="cuda")
    mask = (positions[None, :] >= cu_ks[:, None]) & (
        positions[None, :] < cu_ke[:, None]
    )
    score = torch.einsum("mhd,nd->hmn", q_fp8.float(), k_dequant)
    expected = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)
    expected = expected.masked_fill(~mask, float("-inf"))

    torch.testing.assert_close(logits, expected)


def test_sm70_fallback_paged_logits_accepts_2d_context_lens() -> None:
    _require_cuda()
    torch.manual_seed(1)
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, heads, head_dim = 2, 2, 3, 16
    max_model_len, block_size, num_blocks = 8, 4, 6
    q = torch.randn(
        batch_size, next_n, heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.float16
    )
    scales = torch.rand(num_blocks, block_size, 1, 1, device="cuda") + 0.5
    kv_cache = _pack_fp8_cache(k.to(fp8_dtype), scales)
    weights = torch.randn(
        batch_size * next_n, heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor([[5, 6], [3, 4]], device="cuda", dtype=torch.int32)
    block_tables = torch.tensor(
        [[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32
    )

    logits = _fp8_paged_mqa_logits_torch_fallback(
        q.to(fp8_dtype),
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )

    k_dequant = k.to(fp8_dtype).float() * scales
    expected = torch.full_like(logits, float("-inf"))
    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            q_row = q.to(fp8_dtype).float()[batch_idx, next_idx]
            row_weights = weights[row].float().unsqueeze(-1)
            for block_rk in range((context_len + block_size - 1) // block_size):
                block_idx = int(block_tables[batch_idx, block_rk].item())
                start = block_rk * block_size
                end = min(start + block_size, max_model_len)
                k_block = k_dequant[block_idx, : end - start, 0, :]
                values = ((q_row @ k_block.transpose(0, 1)).relu() * row_weights).sum(
                    dim=0
                )
                offsets = torch.arange(start, end, device="cuda")
                expected[row, start:end] = torch.where(
                    offsets < context_len, values, float("-inf")
                )

    torch.testing.assert_close(logits, expected)
