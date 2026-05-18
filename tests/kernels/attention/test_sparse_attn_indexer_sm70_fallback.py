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


def test_sm70_paged_logits_uses_1d_grid_for_1m_context() -> None:
    from vllm.model_executor.layers.sm70_mqa_logits import (
        _sm70_paged_mqa_logits_grid,
    )

    grid = _sm70_paged_mqa_logits_grid(num_rows=1, max_model_len=1048576)

    assert grid == (1048576,)


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

    # The default SM70 prefill path uses half GEMM per head and accumulates
    # into fp32 logits, so it can differ from the fp32 eager reference by the
    # final half output rounding of each GEMM.
    torch.testing.assert_close(logits, expected, rtol=2e-2, atol=1e-2)


def test_sm70_gemm_prefill_logits_matches_reference() -> None:
    _require_cuda()
    torch.manual_seed(5)
    from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_mqa_logits_gemm

    fp8_dtype = torch.float8_e4m3fn
    seq_len, seq_len_kv, heads, head_dim = 13, 17, 8, 64
    q = (torch.randn(seq_len, heads, head_dim, device="cuda") * 0.1).to(fp8_dtype)
    k = (torch.randn(seq_len_kv, head_dim, device="cuda") * 0.1).to(fp8_dtype)
    weights = torch.randn(seq_len, heads, device="cuda", dtype=torch.float32) * 0.5
    k_scale = torch.rand(seq_len_kv, device="cuda", dtype=torch.float32) * 0.2 + 0.01
    cu_ks = torch.tensor(
        [0, 0, 1, 1, 3, 4, 4, 5, 6, 8, 8, 9, 10],
        device="cuda",
        dtype=torch.int32,
    )
    cu_ke = torch.tensor(
        [4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 17, 17],
        device="cuda",
        dtype=torch.int32,
    )

    logits = sm70_fp8_mqa_logits_gemm(q, (k, k_scale), weights, cu_ks, cu_ke)

    k_dequant = k.float() * k_scale.view(-1, 1)
    positions = torch.arange(seq_len_kv, device="cuda")
    mask = (positions[None, :] >= cu_ks[:, None]) & (
        positions[None, :] < cu_ke[:, None]
    )
    score = torch.einsum("mhd,nd->hmn", q.float(), k_dequant)
    expected = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)
    expected = expected.masked_fill(~mask, float("-inf"))

    torch.testing.assert_close(logits, expected, rtol=2e-2, atol=1e-2)


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

    # The SM70 paged path uses a Triton kernel with fp8 manual decode and
    # fused accumulation, so it can differ slightly from the eager torch
    # reference while preserving the indexer ordering contract.
    torch.testing.assert_close(logits, expected, rtol=2e-2, atol=1e-2)


def test_sm70_fallback_paged_logits_accepts_1d_context_lens() -> None:
    _require_cuda()
    torch.manual_seed(2)
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, heads, head_dim = 2, 2, 3, 16
    max_model_len, block_size, num_blocks = 8, 4, 6
    q = torch.randn(
        batch_size, next_n, heads, head_dim, device="cuda", dtype=torch.float16
    ).to(fp8_dtype)
    k = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.float16
    ).to(fp8_dtype)
    scales = torch.rand(num_blocks, block_size, 1, 1, device="cuda") + 0.5
    kv_cache = _pack_fp8_cache(k, scales)
    weights = torch.randn(
        batch_size * next_n, heads, device="cuda", dtype=torch.float32
    )
    context_lens_1d = torch.tensor([6, 4], device="cuda", dtype=torch.int32)
    next_n_arange = torch.arange(next_n, device="cuda", dtype=torch.int32)
    context_lens_2d = (
        context_lens_1d.unsqueeze(-1) - next_n + 1 + next_n_arange
    ).contiguous()
    block_tables = torch.tensor(
        [[0, 1, 2], [3, 4, 5]], device="cuda", dtype=torch.int32
    )

    logits_1d = _fp8_paged_mqa_logits_torch_fallback(
        q,
        kv_cache,
        weights,
        context_lens_1d,
        block_tables,
        max_model_len,
    )
    logits_2d = _fp8_paged_mqa_logits_torch_fallback(
        q,
        kv_cache,
        weights,
        context_lens_2d,
        block_tables,
        max_model_len,
    )

    torch.testing.assert_close(logits_1d, logits_2d)


def test_sm70_fallback_paged_logits_is_cudagraph_capture_safe() -> None:
    _require_cuda()
    if torch.cuda.get_device_capability()[0] != 7:
        pytest.skip("SM70 fallback is only selected on compute capability 7.x")

    torch.manual_seed(3)
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, heads, head_dim = 1, 1, 3, 16
    max_model_len, block_size, num_blocks = 8, 4, 2
    q = torch.randn(
        batch_size, next_n, heads, head_dim, device="cuda", dtype=torch.float16
    ).to(fp8_dtype)
    k = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.float16
    ).to(fp8_dtype)
    scales = torch.rand(num_blocks, block_size, 1, 1, device="cuda") + 0.5
    kv_cache = _pack_fp8_cache(k, scales)
    weights = torch.randn(
        batch_size * next_n, heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor([[6]], device="cuda", dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)

    for _ in range(2):
        _fp8_paged_mqa_logits_torch_fallback(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = _fp8_paged_mqa_logits_torch_fallback(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert logits.shape == (batch_size * next_n, max_model_len)


def test_sm70_paged_logits_supports_large_max_model_len_grid() -> None:
    _require_cuda()
    if torch.cuda.get_device_capability()[0] != 7:
        pytest.skip("SM70 paged logits kernel is only validated on compute capability 7.x")

    from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_paged_mqa_logits

    torch.manual_seed(4)
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, heads, head_dim = 1, 1, 2, 16
    max_model_len, block_size, num_blocks = 65536, 4, 1
    q = torch.randn(
        batch_size, next_n, heads, head_dim, device="cuda", dtype=torch.float16
    ).to(fp8_dtype)
    k = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.float16
    ).to(fp8_dtype)
    scales = torch.rand(num_blocks, block_size, 1, 1, device="cuda") + 0.5
    kv_cache = _pack_fp8_cache(k, scales)
    weights = torch.randn(
        batch_size * next_n, heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor([[2]], device="cuda", dtype=torch.int32)
    block_tables = torch.tensor([[0]], device="cuda", dtype=torch.int32)

    logits = sm70_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )
    torch.cuda.synchronize()

    assert logits.shape == (batch_size * next_n, max_model_len)
    assert torch.isfinite(logits[:, :2]).all()
    assert torch.isneginf(logits[:, 2:]).all()
