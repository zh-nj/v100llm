# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDAGraph-safety coverage for the SM70 cascade-GEMM indexer path."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class _NoHostSyncTensor:
    """Tiny tensor-like object used to prove dispatch does not read GPU
    metadata back to the host.  Only the attributes used by the dispatch
    gate are implemented; any attempted host sync raises."""

    def __init__(self, shape, device: str = "cuda"):
        self.shape = tuple(shape)
        self.device = torch.device(device)
        self.dtype = torch.float32

    def dim(self):
        return len(self.shape)

    def reshape(self, *shape):
        return self

    def __getitem__(self, _):
        return self

    def item(self):
        raise AssertionError("dispatch attempted a host .item() sync")

    def cpu(self):
        raise AssertionError("dispatch attempted a host .cpu() sync")

    def tolist(self):
        raise AssertionError("dispatch attempted a host .tolist() sync")


def test_cascade_gemm_dispatch_is_capture_safe_without_host_sync(monkeypatch):
    import vllm.model_executor.layers.sm70_cascade_gemm_indexer as gemm_mod
    import vllm.model_executor.layers.sm70_indexer_snapshot as snap_mod
    import vllm.model_executor.layers.sparse_attn_indexer as indexer_mod

    calls = {}
    sentinel = object()

    monkeypatch.setattr(indexer_mod, "_cascade_gemm_enabled", lambda: True)
    monkeypatch.setattr(
        indexer_mod,
        "_can_use_sm70_torch_indexer_fallback",
        lambda *, use_fp4_cache: True,
    )
    monkeypatch.setattr(indexer_mod, "_cascade_gemm_threshold", lambda: 2048)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def fake_snapshot(**kwargs):
        calls["snapshot_kwargs"] = kwargs
        return _NoHostSyncTensor((4096, 128))

    def fake_gemm(*, q, k_f32_cache, weights, context_lens, max_model_len, out=None):
        calls["gemm_kwargs"] = {
            "q": q,
            "k_f32_cache": k_f32_cache,
            "weights": weights,
            "context_lens": context_lens,
            "max_model_len": max_model_len,
            "out": out,
        }
        return sentinel

    monkeypatch.setattr(snap_mod, "ensure_decode_snapshot_cudagraph", fake_snapshot)
    monkeypatch.setattr(gemm_mod, "sm70_cascade_gemm_indexer_from_snapshot", fake_gemm)

    result = indexer_mod._maybe_cascade_gemm_decode_logits(
        _NoHostSyncTensor((1, 1, 64, 128)),
        None,
        _NoHostSyncTensor((16, 256, 1, 132)),
        _NoHostSyncTensor((1, 64)),
        _NoHostSyncTensor((1, 1)),
        _NoHostSyncTensor((1, 64), device="cuda"),
        max_model_len=4096,
        use_fp4_cache=False,
        k_cache_prefix="model.layers.0.indexer.k_cache",
        cache_block_size=256,
    )

    assert result is sentinel
    assert calls["snapshot_kwargs"]["seq_lens"].shape == (1, 1)
    assert calls["gemm_kwargs"]["context_lens"].shape == (1, 1)


_IS_CUDA = torch.cuda.is_available()
_IS_SM70 = _IS_CUDA and torch.cuda.get_device_capability() == (7, 0)


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
def test_cascade_gemm_from_snapshot_replays_under_cuda_graph():
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        sm70_cascade_gemm_indexer,
        sm70_cascade_gemm_indexer_from_snapshot,
    )

    rng = np.random.default_rng(20260527)
    context_len = 64
    max_model_len = 128
    num_heads = 64
    head_dim = 128
    q = torch.from_numpy(
        rng.integers(0, 120, (1, 1, num_heads, head_dim), dtype=np.uint8)
    ).cuda()
    k_values = torch.from_numpy(
        rng.integers(0, 120, (max_model_len, head_dim), dtype=np.uint8)
    ).cuda()
    k_scales = torch.from_numpy(
        rng.uniform(0.05, 1.5, (max_model_len,)).astype(np.float32)
    ).cuda()
    weights = torch.from_numpy(
        rng.normal(0, 0.02, (1, num_heads)).astype(np.float32)
    ).cuda()

    expected = sm70_cascade_gemm_indexer(
        q,
        k_values,
        k_scales,
        weights,
        context_len,
        max_model_len,
    )

    # Snapshot model: pre-decode K once, then capture only the fixed-shape
    # GEMM + device-context-len epilogue.
    snapshot = sm70_cascade_gemm_indexer(
        q,
        k_values,
        k_scales,
        weights,
        max_model_len,
        max_model_len,
    )
    # Rebuild K_f32 directly so this test does not depend on internal scratch.
    k_f32 = torch.empty((max_model_len, head_dim), device="cuda", dtype=torch.float32)
    # Use the public fallback path once to get the correct dequanted K by
    # passing it as a snapshot through the normal function.
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        _decode_k_to_fp32_kernel,
    )

    _decode_k_to_fp32_kernel[((max_model_len + 31) // 32,)](
        k_values,
        k_values.stride(0),
        k_scales,
        k_f32,
        k_f32.stride(0),
        max_model_len,
        HEAD_DIM=head_dim,
        BLOCK_N=32,
    )
    del snapshot

    context_lens = torch.tensor([[context_len]], device="cuda", dtype=torch.int32)
    out = torch.empty((1, max_model_len), device="cuda", dtype=torch.float32)

    for _ in range(3):
        sm70_cascade_gemm_indexer_from_snapshot(
            q=q,
            k_f32_cache=k_f32,
            weights=weights,
            context_lens=context_lens,
            max_model_len=max_model_len,
            out=out,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sm70_cascade_gemm_indexer_from_snapshot(
            q=q,
            k_f32_cache=k_f32,
            weights=weights,
            context_lens=context_lens,
            max_model_len=max_model_len,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out[:, :context_len],
        expected[:, :context_len],
        rtol=1e-4,
        atol=1e-1,
    )
    assert torch.isinf(out[:, context_len:]).all()


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
def test_cudagraph_snapshot_fill_and_gemm_replay_match_paged_reference():
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        sm70_cascade_gemm_indexer,
        sm70_cascade_gemm_indexer_from_snapshot,
    )
    from vllm.model_executor.layers.sm70_indexer_snapshot import (
        _reset_decode_snapshot_pools_for_tests,
        ensure_decode_snapshot_cudagraph,
        reserve_decode_snapshot_cudagraph,
    )

    _reset_decode_snapshot_pools_for_tests()
    rng = np.random.default_rng(20260528)
    context_len = 96
    max_model_len = 128
    block_size = 32
    num_heads = 64
    head_dim = 128
    num_blocks = (max_model_len + block_size - 1) // block_size

    q = torch.from_numpy(
        rng.integers(0, 120, (1, 1, num_heads, head_dim), dtype=np.uint8)
    ).cuda()
    contig_values = torch.from_numpy(
        rng.integers(0, 120, (max_model_len, head_dim), dtype=np.uint8)
    ).cuda()
    contig_scales = torch.from_numpy(
        rng.uniform(0.05, 1.5, (max_model_len,)).astype(np.float32)
    ).cuda()
    weights = torch.from_numpy(
        rng.normal(0, 0.02, (1, num_heads)).astype(np.float32)
    ).cuda()

    paged_cpu = torch.zeros(
        (num_blocks, block_size, 1, head_dim + 4), dtype=torch.uint8
    )
    values_cpu = contig_values.cpu()
    scales_cpu = contig_scales.cpu()
    for i in range(max_model_len):
        bi, ti = i // block_size, i % block_size
        paged_cpu[bi, ti, 0, :head_dim] = values_cpu[i]
        scale_bytes = np.frombuffer(
            np.float32(float(scales_cpu[i])).tobytes(), dtype=np.uint8
        )
        paged_cpu[bi, ti, 0, head_dim : head_dim + 4] = torch.from_numpy(
            scale_bytes.copy()
        )
    paged = paged_cpu.cuda()
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        1, -1
    )
    context_lens = torch.tensor([[context_len]], dtype=torch.int32, device="cuda")

    expected = sm70_cascade_gemm_indexer(
        q,
        contig_values,
        contig_scales,
        weights,
        context_len,
        max_model_len,
    )

    reserve_decode_snapshot_cudagraph(
        k_cache_prefix="test.layer.k_cache",
        kv_cache=paged,
        head_dim=head_dim,
        max_model_len=max_model_len,
    )
    out = torch.empty((1, max_model_len), device="cuda", dtype=torch.float32)
    for _ in range(3):
        snapshot = ensure_decode_snapshot_cudagraph(
            k_cache_prefix="test.layer.k_cache",
            kv_cache=paged,
            block_table_row=block_table[0],
            seq_lens=context_lens,
            block_size=block_size,
            head_dim=head_dim,
            max_model_len=max_model_len,
        )
        assert snapshot is not None
        sm70_cascade_gemm_indexer_from_snapshot(
            q=q,
            k_f32_cache=snapshot,
            weights=weights,
            context_lens=context_lens,
            max_model_len=max_model_len,
            out=out,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        snapshot = ensure_decode_snapshot_cudagraph(
            k_cache_prefix="test.layer.k_cache",
            kv_cache=paged,
            block_table_row=block_table[0],
            seq_lens=context_lens,
            block_size=block_size,
            head_dim=head_dim,
            max_model_len=max_model_len,
        )
        sm70_cascade_gemm_indexer_from_snapshot(
            q=q,
            k_f32_cache=snapshot,
            weights=weights,
            context_lens=context_lens,
            max_model_len=max_model_len,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out[:, :context_len],
        expected[:, :context_len],
        rtol=1e-4,
        atol=1e-1,
    )
    assert torch.isinf(out[:, context_len:]).all()
