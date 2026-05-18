# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing tests for sparse indexer prefill top-k selection."""

import torch


def _make_prefill_inputs():
    logits = torch.arange(24, dtype=torch.float32).view(2, 12)
    row_starts = torch.tensor([3, 4], dtype=torch.int32)
    row_ends = torch.tensor([11, 12], dtype=torch.int32)
    indices = torch.empty((2, 2048), dtype=torch.int32)
    return logits, row_starts, row_ends, indices


def test_prefill_logits_row_chunks_cap_peak_logits_memory(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.delenv(
        "VLLM_SPARSE_INDEXER_PREFILL_LOGITS_CHUNK_MB", raising=False
    )

    chunks = list(
        sparse_attn_indexer._iter_prefill_logits_row_chunks(
            num_rows=4096,
            num_kv_tokens=10002,
        )
    )

    assert chunks == [(0, 1677), (1677, 3354), (3354, 4096)]


def test_prefill_logits_row_chunks_keep_short_prompts_unsplit(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.delenv(
        "VLLM_SPARSE_INDEXER_PREFILL_LOGITS_CHUNK_MB", raising=False
    )

    chunks = list(
        sparse_attn_indexer._iter_prefill_logits_row_chunks(
            num_rows=1782,
            num_kv_tokens=1782,
        )
    )

    assert chunks == [(0, 1782)]


def test_prefill_logits_row_chunks_honor_env_override(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_PREFILL_LOGITS_CHUNK_MB", "1")

    chunks = list(
        sparse_attn_indexer._iter_prefill_logits_row_chunks(
            num_rows=5,
            num_kv_tokens=262144,
        )
    )

    assert chunks == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_prefill_filtered_topk_experiment_is_default_off(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.delenv("VLLM_SPARSE_INDEXER_PREFILL_FILTERED_TOPK", raising=False)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)

    logits, row_starts, row_ends, _ = _make_prefill_inputs()

    assert not sparse_attn_indexer._should_use_persistent_topk_prefill(
        logits,
        row_starts,
        row_ends,
        topk_tokens=512,
        max_row_len=12345,
        all_row_starts_zero=True,
    )


def test_prefill_filtered_topk_experiment_can_enable_persistent(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_PREFILL_FILTERED_TOPK", "1")
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)

    logits, row_starts, row_ends, _ = _make_prefill_inputs()

    assert sparse_attn_indexer._should_use_persistent_topk_prefill(
        logits,
        row_starts,
        row_ends,
        topk_tokens=512,
        max_row_len=12345,
        all_row_starts_zero=True,
    )


def test_prefill_topk_routes_to_tilelang_topk_first(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.v1.attention.ops import tilelang_prefill_topk

    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_should_use_tilelang_topk_prefill",
        lambda *args, **kwargs: True,
    )

    captured = {}

    def fake_tilelang_topk(
        logits,
        indices,
        lengths,
        row_starts,
        *,
        topk_tokens,
        threads,
        causal_row_offset,
    ):
        captured["lengths"] = lengths.clone()
        captured["row_starts"] = row_starts.clone()
        captured["topk_tokens"] = topk_tokens
        captured["threads"] = threads
        captured["causal_row_offset"] = causal_row_offset
        indices.fill_(11)

    def fail_persistent_topk(*args, **kwargs):
        raise AssertionError("persistent_topk should not run")

    def fail_large_context_topk(*args, **kwargs):
        raise AssertionError("large_context_topk should not run")

    def fail_old_path(*args, **kwargs):
        raise AssertionError("old prefill topk path should not run")

    monkeypatch.setattr(tilelang_prefill_topk, "prefill_topk_tilelang", fake_tilelang_topk)
    monkeypatch.setattr(torch.ops._C, "persistent_topk", fail_persistent_topk, raising=False)
    monkeypatch.setattr(torch.ops._C, "large_context_topk", fail_large_context_topk, raising=False)
    monkeypatch.setattr(torch.ops._C, "top_k_per_row_prefill", fail_old_path, raising=False)

    logits, row_starts, row_ends, indices = _make_prefill_inputs()
    indices = indices[:, :512]

    sparse_attn_indexer._prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        indices,
        topk_tokens=512,
    )

    torch.testing.assert_close(captured["lengths"], row_ends - row_starts)
    torch.testing.assert_close(captured["row_starts"], row_starts)
    assert captured["topk_tokens"] == 512
    assert captured["threads"] == sparse_attn_indexer.TILELANG_TOPK_THREADS
    assert captured["causal_row_offset"] is None
    assert torch.all(indices == 11)


def test_tilelang_prefill_topk_targets_single_request_16k_chunks(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_PREFILL_TILELANG_TOPK", "1")
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)

    monkeypatch.setattr(
        "vllm.v1.attention.ops.tilelang_prefill_topk.is_tilelang_available",
        lambda: (True, None),
    )

    logits = torch.empty((4, 16384), dtype=torch.float32)
    row_starts = torch.zeros((4,), dtype=torch.int32)
    row_ends = torch.tensor([4096, 8192, 12288, 16384], dtype=torch.int32)

    assert sparse_attn_indexer._should_use_tilelang_topk_prefill(
        logits,
        row_starts,
        row_ends,
        topk_tokens=512,
        max_row_len=16384,
        all_row_starts_zero=True,
    )
    assert not sparse_attn_indexer._should_use_tilelang_topk_prefill(
        logits[:, :8192],
        row_starts,
        torch.tensor([2048, 4096, 6144, 8192], dtype=torch.int32),
        topk_tokens=512,
        max_row_len=8192,
        all_row_starts_zero=True,
    )
    assert not sparse_attn_indexer._should_use_tilelang_topk_prefill(
        logits,
        torch.tensor([0, 3, 5, 7], dtype=torch.int32),
        row_ends,
        topk_tokens=512,
        max_row_len=16384,
        all_row_starts_zero=False,
    )


def test_streaming_topk_prefill_default_off(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setattr(
        sparse_attn_indexer.envs,
        "VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK",
        False,
        raising=False,
    )
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sparse_attn_indexer.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 70,
    )

    q = torch.empty((4, 4, 128), dtype=torch.float16)
    kv_cache = torch.empty((16, 128), dtype=torch.float16)

    assert not sparse_attn_indexer._should_use_streaming_topk_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_tokens=512,
        use_fp4_cache=False,
        max_row_len=16384,
    )


def test_streaming_topk_prefill_requires_sm70_fp8_and_long_rows(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setattr(
        sparse_attn_indexer.envs,
        "VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK",
        True,
        raising=False,
    )
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sparse_attn_indexer.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 70,
    )

    q = torch.empty((4, 4, 128), dtype=torch.float16)
    kv_cache = torch.empty((16, 128), dtype=torch.float16)

    assert sparse_attn_indexer._should_use_streaming_topk_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_tokens=512,
        use_fp4_cache=False,
        max_row_len=16384,
    )
    assert not sparse_attn_indexer._should_use_streaming_topk_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_tokens=256,
        use_fp4_cache=False,
        max_row_len=16384,
    )
    assert not sparse_attn_indexer._should_use_streaming_topk_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_tokens=512,
        use_fp4_cache=True,
        max_row_len=16384,
    )
    assert not sparse_attn_indexer._should_use_streaming_topk_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_tokens=512,
        use_fp4_cache=False,
        max_row_len=4096,
    )

    monkeypatch.setattr(
        sparse_attn_indexer.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    assert not sparse_attn_indexer._should_use_streaming_topk_prefill(
        q=q,
        kv_cache=kv_cache,
        topk_tokens=512,
        use_fp4_cache=False,
        max_row_len=16384,
    )


def test_try_streaming_topk_prefill_calls_streaming_wrapper(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.v1.attention.ops import tilelang_prefill_streaming_topk

    monkeypatch.setattr(
        sparse_attn_indexer,
        "_should_use_streaming_topk_prefill",
        lambda **kwargs: True,
    )
    captured = {}

    def fake_streaming_topk(**kwargs):
        captured.update(kwargs)
        kwargs["out_indices"].fill_(23)

    monkeypatch.setattr(
        tilelang_prefill_streaming_topk,
        "prefill_streaming_topk_tilelang",
        fake_streaming_topk,
    )

    q = torch.empty((2, 4, 16), dtype=torch.float32)
    k_cache_values = torch.empty((128, 16), dtype=torch.float16)
    k_cache_scales = torch.ones((128,), dtype=torch.float32)
    weights = torch.ones((2, 4), dtype=torch.float32)
    row_starts = torch.zeros((2,), dtype=torch.int32)
    row_ends = torch.full((2,), 128, dtype=torch.int32)
    out_indices = torch.empty((2, 512), dtype=torch.int32)

    handled = sparse_attn_indexer._try_prefill_streaming_topk_indices(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=512,
        use_fp4_cache=False,
        max_row_len=16384,
    )

    assert handled
    assert captured["q"].dtype == torch.float16
    assert captured["k_cache_values"] is k_cache_values
    assert captured["k_cache_scales"] is k_cache_scales
    assert captured["weights"] is weights
    assert captured["row_starts"] is row_starts
    assert captured["row_ends"] is row_ends
    assert captured["out_indices"] is out_indices
    assert captured["topk_tokens"] == 512
    assert torch.all(out_indices == 23)


def test_try_streaming_topk_prefill_returns_false_when_disabled(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.v1.attention.ops import tilelang_prefill_streaming_topk

    monkeypatch.setattr(
        sparse_attn_indexer,
        "_should_use_streaming_topk_prefill",
        lambda **kwargs: False,
    )

    def fail_streaming_topk(**kwargs):
        raise AssertionError("streaming topk should not run")

    monkeypatch.setattr(
        tilelang_prefill_streaming_topk,
        "prefill_streaming_topk_tilelang",
        fail_streaming_topk,
    )

    q = torch.empty((2, 4, 16), dtype=torch.float16)
    k_cache_values = torch.empty((128, 16), dtype=torch.float16)
    k_cache_scales = torch.ones((128,), dtype=torch.float32)
    weights = torch.ones((2, 4), dtype=torch.float32)
    row_starts = torch.zeros((2,), dtype=torch.int32)
    row_ends = torch.full((2,), 128, dtype=torch.int32)
    out_indices = torch.empty((2, 512), dtype=torch.int32)

    handled = sparse_attn_indexer._try_prefill_streaming_topk_indices(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=512,
        use_fp4_cache=False,
        max_row_len=16384,
    )

    assert not handled


def test_prefill_topk_routes_to_large_context_topk(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_should_use_large_context_topk_prefill",
        lambda *args, **kwargs: True,
    )

    captured = {}

    def fake_large_context_topk(logits, indices, lengths, row_starts):
        captured["lengths"] = lengths.clone()
        captured["row_starts"] = row_starts.clone()
        indices.fill_(7)

    def fail_old_path(*args, **kwargs):
        raise AssertionError("old prefill topk path should not run")

    monkeypatch.setattr(torch.ops._C, "large_context_topk", fake_large_context_topk, raising=False)
    monkeypatch.setattr(torch.ops._C, "top_k_per_row_prefill", fail_old_path, raising=False)

    logits, row_starts, row_ends, indices = _make_prefill_inputs()

    sparse_attn_indexer._prefill_topk_indices(
        logits, row_starts, row_ends, indices, topk_tokens=2048
    )

    torch.testing.assert_close(captured["lengths"], row_ends - row_starts)
    torch.testing.assert_close(captured["row_starts"], row_starts)
    assert torch.all(indices == 7)


def test_prefill_topk_routes_single_request_to_persistent_topk(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_should_use_persistent_topk_prefill",
        lambda *args, **kwargs: True,
    )

    workspace = torch.empty((1024,), dtype=torch.uint8)

    class FakeWorkspaceManager:
        def get_simultaneous(self, *specs):
            assert specs == (((sparse_attn_indexer.RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),)
            return [workspace]

    captured = {}

    def fake_persistent_topk(logits, lengths, indices, topk_workspace, topk_tokens, max_row_len):
        captured["lengths"] = lengths.clone()
        captured["workspace"] = topk_workspace
        captured["topk_tokens"] = topk_tokens
        captured["max_row_len"] = max_row_len
        indices.fill_(5)

    def fail_large_context_topk(*args, **kwargs):
        raise AssertionError("large_context_topk should not run")

    def fail_old_path(*args, **kwargs):
        raise AssertionError("old prefill topk path should not run")

    monkeypatch.setattr(
        sparse_attn_indexer, "current_workspace_manager", lambda: FakeWorkspaceManager()
    )
    monkeypatch.setattr(torch.ops._C, "persistent_topk", fake_persistent_topk, raising=False)
    monkeypatch.setattr(torch.ops._C, "large_context_topk", fail_large_context_topk, raising=False)
    monkeypatch.setattr(torch.ops._C, "top_k_per_row_prefill", fail_old_path, raising=False)

    logits, row_starts, row_ends, indices = _make_prefill_inputs()

    sparse_attn_indexer._prefill_topk_indices(
        logits,
        row_starts,
        row_ends,
        indices,
        topk_tokens=512,
        max_row_len=12345,
        all_row_starts_zero=True,
    )

    torch.testing.assert_close(captured["lengths"], row_ends - row_starts)
    assert captured["workspace"] is workspace
    assert captured["topk_tokens"] == 512
    assert captured["max_row_len"] == 12345
    assert torch.all(indices == 5)


def test_prefill_topk_routes_to_legacy_prefill_kernel(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(sparse_attn_indexer.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_should_use_large_context_topk_prefill",
        lambda *args, **kwargs: False,
    )

    captured = {}

    def fail_large_context_topk(*args, **kwargs):
        raise AssertionError("large_context_topk should not run")

    def fake_old_topk(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, topk_tokens):
        captured["row_starts"] = row_starts.clone()
        captured["row_ends"] = row_ends.clone()
        captured["topk_tokens"] = topk_tokens
        indices.fill_(3)

    monkeypatch.setattr(torch.ops._C, "large_context_topk", fail_large_context_topk, raising=False)
    monkeypatch.setattr(torch.ops._C, "top_k_per_row_prefill", fake_old_topk, raising=False)

    logits, row_starts, row_ends, indices = _make_prefill_inputs()

    sparse_attn_indexer._prefill_topk_indices(
        logits, row_starts, row_ends, indices, topk_tokens=2048
    )

    torch.testing.assert_close(captured["row_starts"], row_starts)
    torch.testing.assert_close(captured["row_ends"], row_ends)
    assert captured["topk_tokens"] == 2048
    assert torch.all(indices == 3)
