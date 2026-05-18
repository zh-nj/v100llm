# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API tests for the streaming prefill top-k wrapper."""

import pytest
import torch


_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


def _has_sm70_cuda() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) == (7, 0)
    )


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

    with pytest.raises(ValueError, match="q must be fp16, bf16, or fp8"):
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


def test_streaming_topk_tile_logits_uses_sm70_fp8_hot_kernel(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    monkeypatch.setattr(
        streaming_topk,
        "_can_use_sm70_fp8_tile_logits",
        lambda q, k_cache_values: True,
    )
    captured = {}

    def fake_sm70_fp8_mqa_logits(q, kv, weights, row_starts, row_ends):
        captured["q"] = q
        captured["kv"] = kv
        captured["weights"] = weights
        captured["row_starts"] = row_starts.clone()
        captured["row_ends"] = row_ends.clone()
        rows = q.shape[0]
        cols = kv[0].shape[0]
        return torch.arange(rows * cols, dtype=torch.float32).view(rows, cols)

    monkeypatch.setattr(
        streaming_topk,
        "sm70_fp8_mqa_logits",
        fake_sm70_fp8_mqa_logits,
    )

    q = torch.empty((2, 3, 8), dtype=torch.float16)
    k_cache_values = torch.empty((64, 8), dtype=torch.float16)
    k_cache_scales = torch.ones((64,), dtype=torch.float32)
    weights = torch.ones((2, 3), dtype=torch.float32)
    row_starts = torch.tensor([4, 17], dtype=torch.int32)
    row_ends = torch.tensor([40, 64], dtype=torch.int32)

    logits = streaming_topk._compute_tile_logits(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        tile_start=16,
        tile_end=48,
    )

    assert captured["q"] is q
    assert captured["kv"][0].data_ptr() == k_cache_values[16:48].data_ptr()
    assert captured["kv"][1].data_ptr() == k_cache_scales[16:48].data_ptr()
    assert captured["weights"] is weights
    torch.testing.assert_close(
        captured["row_starts"], torch.tensor([0, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        captured["row_ends"], torch.tensor([24, 32], dtype=torch.int32)
    )
    torch.testing.assert_close(
        logits, torch.arange(64, dtype=torch.float32).view(2, 32)
    )


def test_streaming_topk_oracle_uses_tilelang_for_benched_topk(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    monkeypatch.setattr(streaming_topk, "is_tilelang_available", lambda: (True, None))
    monkeypatch.setattr(streaming_topk, "_is_sm70_tensor_device", lambda q: True)
    calls = []

    def fake_torch(**kwargs):
        calls.append("torch")

    def fake_tilelang(**kwargs):
        calls.append("tilelang")

    monkeypatch.setattr(
        streaming_topk,
        "_prefill_streaming_topk_chunked_torch",
        fake_torch,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_prefill_streaming_topk_chunked_tilelang",
        fake_tilelang,
    )

    kwargs = _make_inputs()
    kwargs["topk_tokens"] = 512
    kwargs["tile_k"] = 1024
    kwargs["threads"] = 256
    streaming_topk._prefill_streaming_topk_oracle(**kwargs)

    assert calls == ["tilelang"]


def test_streaming_topk_oracle_keeps_unbenched_topk_on_chunked_torch(
    monkeypatch,
):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    monkeypatch.setattr(streaming_topk, "is_tilelang_available", lambda: (True, None))
    monkeypatch.setattr(streaming_topk, "_is_sm70_tensor_device", lambda q: True)
    calls = []

    def fake_torch(**kwargs):
        calls.append("torch")

    def fake_tilelang(**kwargs):
        calls.append("tilelang")

    monkeypatch.setattr(
        streaming_topk,
        "_prefill_streaming_topk_chunked_torch",
        fake_torch,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_prefill_streaming_topk_chunked_tilelang",
        fake_tilelang,
    )

    kwargs = _make_inputs()
    kwargs["out_indices"] = torch.empty((2, 1024), dtype=torch.int32)
    kwargs["topk_tokens"] = 1024
    kwargs["tile_k"] = 1024
    kwargs["threads"] = 256
    streaming_topk._prefill_streaming_topk_oracle(**kwargs)

    assert calls == ["torch"]


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_FP8,
    reason="CUDA with torch.float8_e4m3fn is required",
)
@torch.inference_mode()
def test_streaming_topk_fp8_hot_path_matches_full_logits_score_sets_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        prefill_streaming_topk_tilelang,
    )

    torch.manual_seed(20260519)
    device = torch.device("cuda")
    rows = 3
    heads = 3
    dim = 16
    kv_tokens = 128
    topk_tokens = 16
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    k_cache_values = torch.randn(
        (kv_tokens, dim), dtype=torch.float16, device=device
    ).to(torch.float8_e4m3fn)
    k_cache_scales = torch.ones((kv_tokens,), dtype=torch.float32, device=device)
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([0, 7, 19], dtype=torch.int32, device=device)
    row_ends = torch.tensor([96, 128, 80], dtype=torch.int32, device=device)
    out_indices = torch.empty((rows, topk_tokens), dtype=torch.int32, device=device)

    prefill_streaming_topk_tilelang(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=128,
    )

    k_f32 = k_cache_values.float() * k_cache_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        start = int(row_starts[row].item())
        end = int(row_ends[row].item())
        expected = logits[row, start:end].topk(topk_tokens).indices.to(torch.int32)
        assert set(out_indices[row].tolist()) == set(expected.tolist())


@pytest.mark.skipif(
    not _has_sm70_cuda(),
    reason="SM70 CUDA is required for TileLang candidate top-k",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_candidate_loop_matches_full_logits_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _prefill_streaming_topk_chunked_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260520)
    device = torch.device("cuda")
    rows = 3
    heads = 3
    dim = 16
    kv_tokens = 256
    topk_tokens = 16
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device)
    k_cache_values = torch.randn(
        (kv_tokens, dim), dtype=torch.float16, device=device
    )
    k_cache_scales = torch.linspace(
        0.75, 1.25, kv_tokens, dtype=torch.float32, device=device
    )
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([0, 11, 93], dtype=torch.int32, device=device)
    row_ends = torch.tensor([129, 256, 127], dtype=torch.int32, device=device)
    out_indices = torch.empty((rows, topk_tokens), dtype=torch.int32, device=device)

    _prefill_streaming_topk_chunked_tilelang(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=128,
        threads=256,
    )

    k_f32 = k_cache_values.float() * k_cache_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        start = int(row_starts[row].item())
        end = int(row_ends[row].item())
        expected = logits[row, start:end].topk(topk_tokens).indices.to(torch.int32)
        assert set(out_indices[row].tolist()) == set(expected.tolist())


@pytest.mark.skipif(
    not _has_sm70_cuda(),
    reason="SM70 CUDA is required for TileLang candidate merge",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_candidate_merge_matches_torch_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _update_best_candidates_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    topk_tokens = 4
    best_scores = torch.tensor(
        [[10.0, 5.0, -1.0, -3.0], [6.0, 4.0, 2.0, -2.0]],
        dtype=torch.float32,
        device=device,
    )
    best_indices = torch.tensor(
        [[0, 1, 2, 3], [10, 11, 12, 13]],
        dtype=torch.int32,
        device=device,
    )
    tile_logits = torch.tensor(
        [
            [9.0, 13.0, 1.0, 7.0, 12.0, -4.0, 3.0, 2.0],
            [-5.0, 8.0, 3.0, 7.0, 1.0, 9.0, 0.0, 5.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    tile_offsets = torch.tensor(
        [[1, 4, 3, 6], [3, 1, 5, 7]],
        dtype=torch.int32,
        device=device,
    )
    row_starts = torch.tensor([96, 99], dtype=torch.int32, device=device)
    tile_row_starts = torch.tensor([0, 0], dtype=torch.int32, device=device)
    tile_start = 100

    next_scores, next_indices = _update_best_candidates_tilelang(
        best_scores=best_scores,
        best_indices=best_indices,
        tile_logits=tile_logits,
        tile_offsets=tile_offsets,
        row_starts=row_starts,
        tile_row_starts=tile_row_starts,
        tile_start=tile_start,
        topk_tokens=topk_tokens,
        threads=256,
    )

    tile_cols = tile_offsets + tile_row_starts.view(-1, 1)
    tile_scores = tile_logits.gather(1, tile_cols.to(torch.int64))
    tile_indices = tile_start + tile_cols - row_starts.view(-1, 1)
    merged_scores = torch.cat((best_scores, tile_scores), dim=1)
    merged_indices = torch.cat((best_indices, tile_indices.to(torch.int32)), dim=1)
    expected_scores, keep_pos = merged_scores.topk(topk_tokens, dim=1)
    expected_indices = merged_indices.gather(1, keep_pos)

    for row in range(best_scores.shape[0]):
        assert set(next_scores[row].tolist()) == set(expected_scores[row].tolist())
        assert set(next_indices[row].tolist()) == set(expected_indices[row].tolist())


def test_streaming_topk_candidate_update_uses_single_fused_kernel(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    calls = []

    def fake_get_fused_kernel(topk_tokens, tile_keep, threads):
        calls.append(("get_fused", topk_tokens, tile_keep, threads))

        def fake_kernel(*args):
            calls.append(("run_fused", len(args)))
            next_scores = args[-2]
            next_indices = args[-1]
            next_scores.fill_(1.0)
            next_indices.fill_(2)

        return fake_kernel

    def fail_old_buffer(*args, **kwargs):
        raise AssertionError("candidate buffer kernel should not run")

    def fail_old_gather(*args, **kwargs):
        raise AssertionError("candidate gather kernel should not run")

    def fail_old_topk(*args, **kwargs):
        raise AssertionError("prefill_topk_tilelang should not run in merge")

    monkeypatch.setattr(
        streaming_topk,
        "_get_fused_candidate_update_kernel",
        fake_get_fused_kernel,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_get_candidate_buffer_kernel",
        fail_old_buffer,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_get_candidate_gather_kernel",
        fail_old_gather,
    )
    monkeypatch.setattr(streaming_topk, "prefill_topk_tilelang", fail_old_topk)

    best_scores = torch.zeros((2, 4), dtype=torch.float32)
    best_indices = torch.zeros((2, 4), dtype=torch.int32)
    tile_logits = torch.zeros((2, 8), dtype=torch.float32)
    tile_offsets = torch.zeros((2, 4), dtype=torch.int32)
    row_starts = torch.zeros((2,), dtype=torch.int32)
    tile_row_starts = torch.zeros((2,), dtype=torch.int32)

    next_scores, next_indices = streaming_topk._update_best_candidates_tilelang(
        best_scores=best_scores,
        best_indices=best_indices,
        tile_logits=tile_logits,
        tile_offsets=tile_offsets,
        row_starts=row_starts,
        tile_row_starts=tile_row_starts,
        tile_start=100,
        topk_tokens=4,
        threads=256,
    )

    assert calls == [("get_fused", 4, 4, 256), ("run_fused", 9)]
    assert torch.all(next_scores == 1.0)
    assert torch.all(next_indices == 2)


@pytest.mark.skipif(
    not _has_sm70_cuda(),
    reason="SM70 CUDA is required for TileLang final index copy",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_final_index_copy_masks_short_rows_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _copy_final_indices_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    best_indices = torch.tensor(
        [[8, 3, 1, 7], [5, 6, 9, 10], [2, 4, 11, 12]],
        dtype=torch.int32,
        device=device,
    )
    row_starts = torch.tensor([0, 10, 20], dtype=torch.int32, device=device)
    row_ends = torch.tensor([2, 14, 20], dtype=torch.int32, device=device)
    out_indices = torch.empty_like(best_indices)

    _copy_final_indices_tilelang(
        best_indices=best_indices,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=4,
        threads=256,
    )

    expected = torch.tensor(
        [[8, 3, -1, -1], [5, 6, 9, 10], [-1, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(out_indices, expected)
