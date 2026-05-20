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


def test_streaming_topk_tile_logits_uses_sm70_gemm_for_many_heads(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    monkeypatch.setattr(
        streaming_topk,
        "_can_use_sm70_fp8_tile_logits",
        lambda q, k_cache_values: True,
    )
    calls = []

    def fail_scalar_kernel(*args, **kwargs):
        raise AssertionError("scalar SM70 logits kernel only supports <=8 heads")

    def fake_gemm_kernel(q, kv, weights, row_starts, row_ends):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(kv[0].shape),
                "row_starts": row_starts.clone(),
                "row_ends": row_ends.clone(),
            }
        )
        rows = q.shape[0]
        cols = kv[0].shape[0]
        return torch.full((rows, cols), 7.0, dtype=torch.float32)

    monkeypatch.setattr(streaming_topk, "sm70_fp8_mqa_logits", fail_scalar_kernel)
    monkeypatch.setattr(
        streaming_topk,
        "sm70_fp8_mqa_logits_gemm",
        fake_gemm_kernel,
    )

    q = torch.empty((2, 16, 8), dtype=torch.float16)
    k_cache_values = torch.empty((64, 8), dtype=torch.float16)
    k_cache_scales = torch.ones((64,), dtype=torch.float32)
    weights = torch.ones((2, 16), dtype=torch.float32)
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

    assert calls[0]["q_shape"] == (2, 16, 8)
    assert calls[0]["k_shape"] == (32, 8)
    torch.testing.assert_close(
        calls[0]["row_starts"], torch.tensor([0, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        calls[0]["row_ends"], torch.tensor([24, 32], dtype=torch.int32)
    )
    torch.testing.assert_close(logits, torch.full((2, 32), 7.0))


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


def test_streaming_topk_oracle_uses_fused_tile_path_when_enabled(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    monkeypatch.setattr(streaming_topk, "is_tilelang_available", lambda: (True, None))
    monkeypatch.setattr(streaming_topk, "_is_sm70_tensor_device", lambda q: True)
    monkeypatch.setattr(
        streaming_topk.envs,
        "VLLM_SPARSE_INDEXER_PREFILL_FUSED_TILE_TOPK",
        True,
    )
    monkeypatch.setattr(
        streaming_topk.envs,
        "VLLM_SPARSE_INDEXER_PREFILL_FUSED_TILE_BLOCK_K",
        128,
    )
    calls = []

    def fake_blocked(**kwargs):
        calls.append(("blocked", kwargs["block_k"]))

    def fake_tilelang(**kwargs):
        calls.append(("tilelang", None))

    monkeypatch.setattr(
        streaming_topk,
        "_prefill_streaming_topk_blocked_tilelang",
        fake_blocked,
        raising=False,
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

    assert calls == [("blocked", 128)]


def test_streaming_topk_blocked_path_does_not_materialize_full_tile_logits(
    monkeypatch,
):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    compute_spans = []

    def fake_compute_tile_logits(
        *,
        q,
        k_cache_values,
        k_cache_scales,
        weights,
        row_starts,
        row_ends,
        tile_start,
        tile_end,
    ):
        del q, k_cache_values, k_cache_scales, weights, row_starts, row_ends
        compute_spans.append((tile_start, tile_end))
        assert tile_end - tile_start <= 64
        return torch.zeros((2, tile_end - tile_start), dtype=torch.float32)

    def fake_update_from_scores(
        *,
        best_scores,
        best_indices,
        tile_scores,
        row_starts,
        tile_local_starts,
        tile_abs_starts,
        tile_lengths=None,
        topk_tokens,
        threads,
    ):
        del tile_scores, row_starts, tile_local_starts, tile_abs_starts
        del tile_lengths
        del topk_tokens, threads
        return best_scores, best_indices

    def fake_final_copy(**kwargs):
        kwargs["out_indices"].fill_(-1)

    def fail_tile_offsets(*args, **kwargs):
        raise AssertionError("blocked fused path should not run tile logits top-k")

    monkeypatch.setattr(streaming_topk, "_compute_tile_logits", fake_compute_tile_logits)
    monkeypatch.setattr(
        streaming_topk,
        "_update_best_candidates_from_scores_tilelang",
        fake_update_from_scores,
        raising=False,
    )
    monkeypatch.setattr(streaming_topk, "_copy_final_indices_tilelang", fake_final_copy)
    monkeypatch.setattr(streaming_topk, "_select_tile_offsets_tilelang", fail_tile_offsets)

    kwargs = _make_inputs()
    kwargs["k_cache_values"] = torch.empty((300, 16), dtype=torch.float16)
    kwargs["k_cache_scales"] = torch.ones((300,), dtype=torch.float32)
    kwargs["out_indices"] = torch.empty((2, 16), dtype=torch.int32)
    kwargs["topk_tokens"] = 16

    streaming_topk._prefill_streaming_topk_blocked_tilelang(
        **kwargs,
        tile_k=256,
        block_k=64,
        threads=256,
    )

    assert compute_spans == [
        (0, 64),
        (64, 128),
        (128, 192),
        (192, 256),
        (256, 300),
    ]


def test_streaming_topk_blocked_fp16_path_uses_single_launch_block_update(
    monkeypatch,
):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    calls = []

    def fail_compute_tile_logits(*args, **kwargs):
        raise AssertionError("fp16 fused block path should not materialize logits")

    def fail_update_from_scores(*args, **kwargs):
        raise AssertionError("fp16 fused block path should not merge global logits")

    def fake_block_update(**kwargs):
        calls.append(
            {
                "span": (
                    kwargs["block_start"],
                    kwargs["block_start"] + kwargs["k_tile"].shape[0],
                ),
                "local_starts": kwargs["tile_local_starts"].clone(),
                "lengths": kwargs["tile_lengths"].clone(),
                "threads": kwargs["threads"],
            }
        )
        return kwargs["best_scores"], kwargs["best_indices"]

    def fake_final_copy(**kwargs):
        kwargs["out_indices"].fill_(-1)

    monkeypatch.setattr(
        streaming_topk,
        "_can_use_tilelang_block_candidate_update",
        lambda q, k_cache_values: True,
    )
    monkeypatch.setattr(streaming_topk, "_compute_tile_logits", fail_compute_tile_logits)
    monkeypatch.setattr(
        streaming_topk,
        "_update_best_candidates_from_scores_tilelang",
        fail_update_from_scores,
        raising=False,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_update_best_candidates_from_block_tilelang",
        fake_block_update,
        raising=False,
    )
    monkeypatch.setattr(streaming_topk, "_copy_final_indices_tilelang", fake_final_copy)

    kwargs = _make_inputs()
    kwargs["k_cache_values"] = torch.empty((300, 16), dtype=torch.float16)
    kwargs["k_cache_scales"] = torch.ones((300,), dtype=torch.float32)
    kwargs["row_starts"] = torch.tensor([0, 70], dtype=torch.int32)
    kwargs["row_ends"] = torch.tensor([300, 190], dtype=torch.int32)
    kwargs["out_indices"] = torch.empty((2, 16), dtype=torch.int32)
    kwargs["topk_tokens"] = 16

    streaming_topk._prefill_streaming_topk_blocked_tilelang(
        **kwargs,
        tile_k=256,
        block_k=64,
        threads=256,
    )

    assert [call["span"] for call in calls] == [
        (0, 64),
        (64, 128),
        (128, 192),
        (192, 256),
        (256, 300),
    ]
    assert calls[0]["threads"] == 128
    torch.testing.assert_close(
        calls[1]["local_starts"],
        torch.tensor([0, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(
        calls[1]["lengths"],
        torch.tensor([64, 58], dtype=torch.int32),
    )


def test_streaming_topk_dense_block_skips_tile_local_topk(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    calls = []

    def fail_prefill_topk(*args, **kwargs):
        raise AssertionError("dense block path should not run tile-local top-k")

    def fake_update_kernel(topk_tokens, tile_keep, threads):
        calls.append(("get_update", topk_tokens, tile_keep, threads))

        def fake_kernel(*args):
            calls.append(("run_update", len(args)))
            next_scores = args[-2]
            next_indices = args[-1]
            next_scores.fill_(3.0)
            next_indices.fill_(4)

        return fake_kernel

    def fake_dense_offsets(tile_keep, threads):
        calls.append(("get_dense_offsets", tile_keep, threads))

        def fake_kernel(tile_local_starts, tile_lengths, tile_offsets):
            del tile_local_starts
            calls.append(("run_dense_offsets", tuple(tile_lengths.tolist())))
            tile_offsets.fill_(-1)
            for row in range(tile_offsets.shape[0]):
                length = int(tile_lengths[row].item())
                tile_offsets[row, :length] = torch.arange(
                    length,
                    dtype=tile_offsets.dtype,
                )

        return fake_kernel

    monkeypatch.setattr(streaming_topk, "prefill_topk_tilelang", fail_prefill_topk)
    monkeypatch.setattr(
        streaming_topk,
        "_get_dense_tile_offsets_kernel",
        fake_dense_offsets,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_get_fused_candidate_update_kernel",
        fake_update_kernel,
    )

    best_scores = torch.full((2, 16), -torch.inf, dtype=torch.float32)
    best_indices = torch.full((2, 16), -1, dtype=torch.int32)
    tile_scores = torch.zeros((2, 8), dtype=torch.float32)
    row_starts = torch.tensor([0, 5], dtype=torch.int32)
    tile_local_starts = torch.tensor([0, 3], dtype=torch.int32)
    tile_abs_starts = torch.tensor([100, 103], dtype=torch.int32)
    tile_lengths = torch.tensor([8, 5], dtype=torch.int32)

    next_scores, next_indices = (
        streaming_topk._update_best_candidates_from_scores_tilelang(
            best_scores=best_scores,
            best_indices=best_indices,
            tile_scores=tile_scores,
            row_starts=row_starts,
            tile_local_starts=tile_local_starts,
            tile_abs_starts=tile_abs_starts,
            tile_lengths=tile_lengths,
            topk_tokens=16,
            threads=256,
        )
    )

    assert calls == [
        ("get_dense_offsets", 8, 256),
        ("run_dense_offsets", (8, 5)),
        ("get_update", 16, 8, 256),
        ("run_update", 9),
    ]
    assert torch.all(next_scores == 3.0)
    assert torch.all(next_indices == 4)


@pytest.mark.skipif(not _HAS_FP8, reason="torch.float8_e4m3fn is required")
def test_streaming_topk_fp8_blocked_path_uses_fused_block_update(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    calls = []

    def fail_compute_tile_logits(*args, **kwargs):
        raise AssertionError("fp8 blocked path should not materialize block logits")

    def fake_block_update(**kwargs):
        calls.append(
            (
                kwargs["block_start"],
                kwargs["block_start"] + kwargs["k_tile"].shape[0],
                kwargs["q"].shape[1],
            )
        )
        return kwargs["best_scores"], kwargs["best_indices"]

    def fake_final_copy(**kwargs):
        kwargs["out_indices"].fill_(-1)

    monkeypatch.setattr(streaming_topk, "_compute_tile_logits", fail_compute_tile_logits)
    monkeypatch.setattr(
        streaming_topk,
        "_can_use_tilelang_block_candidate_update",
        lambda q, k_cache_values: True,
    )
    monkeypatch.setattr(
        streaming_topk,
        "_update_best_candidates_from_block_tilelang",
        fake_block_update,
        raising=False,
    )
    monkeypatch.setattr(
        streaming_topk,
        "sm70_fp8_mqa_block_candidate_update",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("TileLang tensorcore block update should handle fp8")
        ),
    )
    monkeypatch.setattr(streaming_topk, "_copy_final_indices_tilelang", fake_final_copy)

    kwargs = _make_inputs()
    kwargs["q"] = torch.empty((2, 16, 16), dtype=torch.float8_e4m3fn)
    kwargs["k_cache_values"] = torch.empty((300, 16), dtype=torch.float8_e4m3fn)
    kwargs["k_cache_scales"] = torch.ones((300,), dtype=torch.float32)
    kwargs["weights"] = torch.ones((2, 16), dtype=torch.float32)
    kwargs["out_indices"] = torch.empty((2, 16), dtype=torch.int32)
    kwargs["topk_tokens"] = 16

    streaming_topk._prefill_streaming_topk_blocked_tilelang(
        **kwargs,
        tile_k=256,
        block_k=64,
        threads=256,
    )

    assert calls == [
        (0, 64, 16),
        (64, 128, 16),
        (128, 192, 16),
        (192, 256, 16),
        (256, 300, 16),
    ]


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
    reason="SM70 CUDA is required for blocked TileLang streaming top-k",
)
@torch.inference_mode()
def test_streaming_topk_blocked_tilelang_matches_full_logits_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _prefill_streaming_topk_blocked_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    rows = 4
    heads = 3
    dim = 16
    kv_tokens = 300
    topk_tokens = 16
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device)
    k_cache_values = torch.randn(
        (kv_tokens, dim), dtype=torch.float16, device=device
    )
    k_cache_scales = torch.linspace(
        0.75, 1.25, kv_tokens, dtype=torch.float32, device=device
    )
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([0, 11, 93, 250], dtype=torch.int32, device=device)
    row_ends = torch.tensor([129, 256, 127, 257], dtype=torch.int32, device=device)
    out_indices = torch.empty((rows, topk_tokens), dtype=torch.int32, device=device)

    _prefill_streaming_topk_blocked_tilelang(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=128,
        block_k=64,
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
        valid_count = min(topk_tokens, end - start)
        if valid_count:
            expected_scores = logits[row, start:end].topk(valid_count).values
            got = out_indices[row, :valid_count]
            assert got.min().item() >= 0
            assert got.max().item() < end - start
            assert torch.unique(got).numel() == valid_count
            got_scores = logits[row, start + got.to(torch.int64)]
            threshold = expected_scores[-1]
            assert torch.all(got_scores >= threshold - 1e-5)
        assert torch.all(out_indices[row, valid_count:] == -1)


@pytest.mark.skipif(
    not _has_sm70_cuda() or not _HAS_FP8,
    reason="SM70 CUDA with torch.float8_e4m3fn is required",
)
@torch.inference_mode()
def test_streaming_topk_fp8_blocked_fused_path_matches_full_logits_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _prefill_streaming_topk_blocked_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260523)
    device = torch.device("cuda")
    rows = 3
    heads = 3
    dim = 16
    kv_tokens = 92
    topk_tokens = 16
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    k_cache_values = torch.randn(
        (kv_tokens, dim), dtype=torch.float16, device=device
    ).to(torch.float8_e4m3fn)
    k_cache_scales = torch.linspace(
        0.75,
        1.25,
        kv_tokens,
        dtype=torch.float32,
        device=device,
    )
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([0, 11, 50], dtype=torch.int32, device=device)
    row_ends = torch.tensor([80, 92, 58], dtype=torch.int32, device=device)
    out_indices = torch.empty((rows, topk_tokens), dtype=torch.int32, device=device)

    _prefill_streaming_topk_blocked_tilelang(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=64,
        block_k=16,
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
        valid_count = min(topk_tokens, end - start)
        if valid_count:
            expected_scores = logits[row, start:end].topk(valid_count).values
            got = out_indices[row, :valid_count]
            assert got.min().item() >= 0
            assert got.max().item() < end - start
            assert torch.unique(got).numel() == valid_count
            got_scores = logits[row, start + got.to(torch.int64)]
            threshold = expected_scores[-1]
            assert torch.all(got_scores >= threshold - 1e-5)
        assert torch.all(out_indices[row, valid_count:] == -1)


@pytest.mark.skipif(
    not _has_sm70_cuda(),
    reason="SM70 CUDA is required for TileLang block GEMM logits",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_block_logits_matches_torch_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _compute_block_logits_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260526)
    device = torch.device("cuda")
    rows = 5
    heads = 4
    dim = 32
    block_n = 16
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device)
    k_tile = torch.randn((block_n, dim), dtype=torch.float16, device=device)
    k_scales = torch.linspace(0.8, 1.2, block_n, dtype=torch.float32, device=device)
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    tile_local_starts = torch.tensor([0, 3, 7, 16, 2], dtype=torch.int32, device=device)
    tile_lengths = torch.tensor([16, 9, 4, 0, 12], dtype=torch.int32, device=device)

    logits = _compute_block_logits_tilelang(
        q=q,
        k_tile=k_tile,
        k_scales=k_scales,
        weights=weights,
        tile_local_starts=tile_local_starts,
        tile_lengths=tile_lengths,
        block_m=8,
        block_d=16,
        threads=128,
    )

    k_f32 = k_tile.float() * k_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    expected = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)
    cols = torch.arange(block_n, device=device).view(1, -1)
    valid = (cols >= tile_local_starts.view(-1, 1)) & (
        cols < (tile_local_starts + tile_lengths).view(-1, 1)
    )
    expected = expected.masked_fill(~valid, -torch.inf)

    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        logits[finite],
        expected[finite],
        rtol=3e-2,
        atol=3e-2,
    )
    assert torch.isneginf(logits[~finite]).all()


@pytest.mark.skipif(
    not _has_sm70_cuda(),
    reason="SM70 CUDA is required for TileLang fused block GEMM candidate update",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_block_gemm_candidate_update_matches_torch_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _update_best_candidates_from_block_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260527)
    device = torch.device("cuda")
    rows = 5
    heads = 4
    dim = 32
    block_n = 16
    topk_tokens = 8
    block_start = 32
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device)
    k_tile = torch.randn((block_n, dim), dtype=torch.float16, device=device)
    k_scales = torch.linspace(0.9, 1.1, block_n, dtype=torch.float32, device=device)
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([24, 30, 31, 40, 34], dtype=torch.int32, device=device)
    tile_local_starts = torch.tensor(
        [0, 0, 0, 8, 2], dtype=torch.int32, device=device
    )
    tile_lengths = torch.tensor(
        [16, 15, 12, 4, 10], dtype=torch.int32, device=device
    )
    best_scores = torch.randn((rows, topk_tokens), dtype=torch.float32, device=device)
    best_indices = torch.arange(
        rows * topk_tokens,
        dtype=torch.int32,
        device=device,
    ).view(rows, topk_tokens)

    next_scores, next_indices = _update_best_candidates_from_block_tilelang(
        q=q,
        k_tile=k_tile,
        k_scales=k_scales,
        weights=weights,
        best_scores=best_scores,
        best_indices=best_indices,
        row_starts=row_starts,
        tile_local_starts=tile_local_starts,
        tile_lengths=tile_lengths,
        block_start=block_start,
        topk_tokens=topk_tokens,
        block_m=8,
        block_d=16,
        threads=128,
    )

    k_f32 = k_tile.float() * k_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    block_logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        start = int(tile_local_starts[row].item())
        end = start + int(tile_lengths[row].item())
        valid_scores = block_logits[row, start:end]
        valid_indices = (
            torch.arange(start, end, dtype=torch.int32, device=device)
            + block_start
            - row_starts[row]
        )
        merged_scores = torch.cat((best_scores[row], valid_scores), dim=0)
        merged_indices = torch.cat((best_indices[row], valid_indices), dim=0)
        expected_scores, keep = merged_scores.topk(topk_tokens)
        expected_indices = merged_indices.gather(0, keep)
        assert torch.all(next_scores[row] >= expected_scores[-1] - 1e-5)
        assert set(next_indices[row].tolist()) == set(expected_indices.tolist())


@pytest.mark.skipif(
    not _has_sm70_cuda(),
    reason="SM70 CUDA is required for TileLang fused block GEMM candidate update",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_block_candidate_update_topk_gt_threads_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _update_best_candidates_from_block_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260530)
    device = torch.device("cuda")
    rows = 2
    heads = 2
    dim = 16
    block_n = 16
    topk_tokens = 512
    block_start = 2048
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device)
    k_tile = torch.randn((block_n, dim), dtype=torch.float16, device=device)
    k_scales = torch.linspace(0.75, 1.25, block_n, dtype=torch.float32, device=device)
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([0, 1536], dtype=torch.int32, device=device)
    tile_local_starts = torch.tensor([0, 4], dtype=torch.int32, device=device)
    tile_lengths = torch.tensor([16, 8], dtype=torch.int32, device=device)
    best_scores = torch.full(
        (rows, topk_tokens),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_indices = torch.full(
        (rows, topk_tokens),
        -1,
        dtype=torch.int32,
        device=device,
    )

    next_scores, next_indices = _update_best_candidates_from_block_tilelang(
        q=q,
        k_tile=k_tile,
        k_scales=k_scales,
        weights=weights,
        best_scores=best_scores,
        best_indices=best_indices,
        row_starts=row_starts,
        tile_local_starts=tile_local_starts,
        tile_lengths=tile_lengths,
        block_start=block_start,
        topk_tokens=topk_tokens,
        block_m=8,
        block_d=16,
        threads=128,
    )

    k_f32 = k_tile.float() * k_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    block_logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        start = int(tile_local_starts[row].item())
        end = start + int(tile_lengths[row].item())
        valid_scores = block_logits[row, start:end]
        valid_indices = (
            torch.arange(start, end, dtype=torch.int32, device=device)
            + block_start
            - row_starts[row]
        )
        valid_count = end - start
        assert set(next_indices[row, :valid_count].tolist()) == set(
            valid_indices.tolist()
        )
        assert torch.all(next_indices[row, valid_count:] == -1)
        got_scores = next_scores[row, :valid_count]
        local_cols = (
            next_indices[row, :valid_count].to(torch.int64)
            - int(block_start - row_starts[row].item())
        )
        expected_scores = valid_scores[local_cols - start]
        torch.testing.assert_close(
            got_scores,
            expected_scores,
            rtol=3e-2,
            atol=3e-2,
        )


@pytest.mark.skipif(
    not _has_sm70_cuda() or not _HAS_FP8,
    reason="SM70 CUDA with torch.float8_e4m3fn is required",
)
@torch.inference_mode()
def test_streaming_topk_tilelang_fp8_block_candidate_update_matches_torch_cuda():
    from vllm.v1.attention.ops.tilelang_prefill_topk import is_tilelang_available
    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        _update_best_candidates_from_block_tilelang,
    )

    ok, reason = is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    torch.manual_seed(20260528)
    device = torch.device("cuda")
    rows = 3
    heads = 12
    dim = 32
    block_n = 16
    topk_tokens = 8
    block_start = 64
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    k_tile = torch.randn((block_n, dim), dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    k_scales = torch.linspace(0.85, 1.15, block_n, dtype=torch.float32, device=device)
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([60, 64, 70], dtype=torch.int32, device=device)
    tile_local_starts = torch.tensor([0, 0, 6], dtype=torch.int32, device=device)
    tile_lengths = torch.tensor([16, 9, 7], dtype=torch.int32, device=device)
    best_scores = torch.randn((rows, topk_tokens), dtype=torch.float32, device=device)
    best_indices = torch.arange(
        rows * topk_tokens,
        dtype=torch.int32,
        device=device,
    ).view(rows, topk_tokens)

    next_scores, next_indices = _update_best_candidates_from_block_tilelang(
        q=q,
        k_tile=k_tile,
        k_scales=k_scales,
        weights=weights,
        best_scores=best_scores,
        best_indices=best_indices,
        row_starts=row_starts,
        tile_local_starts=tile_local_starts,
        tile_lengths=tile_lengths,
        block_start=block_start,
        topk_tokens=topk_tokens,
        block_m=8,
        block_d=16,
        threads=128,
    )

    k_f32 = k_tile.float() * k_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32)
    block_logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        start = int(tile_local_starts[row].item())
        end = start + int(tile_lengths[row].item())
        valid_scores = block_logits[row, start:end]
        valid_indices = (
            torch.arange(start, end, dtype=torch.int32, device=device)
            + block_start
            - row_starts[row]
        )
        merged_scores = torch.cat((best_scores[row], valid_scores), dim=0)
        merged_indices = torch.cat((best_indices[row], valid_indices), dim=0)
        expected_scores, keep = merged_scores.topk(topk_tokens)
        expected_indices = merged_indices.gather(0, keep)
        assert torch.all(next_scores[row] >= expected_scores[-1] - 5e-2)
        assert set(next_indices[row].tolist()) == set(expected_indices.tolist())


@pytest.mark.skipif(
    not _has_sm70_cuda() or not _HAS_FP8,
    reason="SM70 CUDA with torch.float8_e4m3fn is required",
)
@torch.inference_mode()
def test_sm70_fp8_mqa_block_candidate_update_matches_torch_cuda():
    from vllm.model_executor.layers.sm70_mqa_logits import (
        sm70_fp8_mqa_block_candidate_update,
    )

    torch.manual_seed(20260522)
    device = torch.device("cuda")
    rows = 2
    heads = 3
    dim = 16
    kv_tokens = 48
    topk_tokens = 16
    block_start = 17
    block_end = 25
    q = torch.randn((rows, heads, dim), dtype=torch.float16, device=device).to(
        torch.float8_e4m3fn
    )
    k_cache_values = torch.randn(
        (kv_tokens, dim), dtype=torch.float16, device=device
    ).to(torch.float8_e4m3fn)
    k_cache_scales = torch.linspace(
        0.8,
        1.2,
        kv_tokens,
        dtype=torch.float32,
        device=device,
    )
    weights = torch.randn((rows, heads), dtype=torch.float32, device=device)
    row_starts = torch.tensor([0, 19], dtype=torch.int32, device=device)
    row_ends = torch.tensor([40, 24], dtype=torch.int32, device=device)
    best_scores = torch.randn((rows, topk_tokens), dtype=torch.float32, device=device)
    best_indices = torch.arange(
        rows * topk_tokens,
        dtype=torch.int32,
        device=device,
    ).view(rows, topk_tokens)

    next_scores, next_indices = sm70_fp8_mqa_block_candidate_update(
        q=q,
        kv=(k_cache_values, k_cache_scales),
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        best_scores=best_scores,
        best_indices=best_indices,
        block_start=block_start,
        block_end=block_end,
        topk_tokens=topk_tokens,
    )

    k_f32 = k_cache_values.float() * k_cache_scales.view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q.float(), k_f32[block_start:block_end])
    block_logits = (
        score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)
    ).sum(dim=0)

    for row in range(rows):
        local_start = max(0, int(row_starts[row].item()) - block_start)
        local_end = min(block_end - block_start, int(row_ends[row].item()) - block_start)
        valid_block_scores = block_logits[row, local_start:local_end]
        valid_block_indices = (
            torch.arange(local_start, local_end, dtype=torch.int32, device=device)
            + block_start
            - row_starts[row]
        )
        merged_scores = torch.cat((best_scores[row], valid_block_scores), dim=0)
        merged_indices = torch.cat((best_indices[row], valid_block_indices), dim=0)
        expected_scores, keep = merged_scores.topk(topk_tokens)
        expected_indices = merged_indices.gather(0, keep)
        got_scores = next_scores[row]
        got_indices = next_indices[row]
        threshold = expected_scores[-1]
        assert torch.all(got_scores >= threshold - 1e-5)
        assert set(got_indices.tolist()) == set(expected_indices.tolist())


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


def test_streaming_topk_tilelang_pads_tail_tile_to_fixed_topk(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    calls = []

    def fake_prefill_topk_tilelang(
        logits,
        indices,
        lengths,
        row_starts,
        *,
        topk_tokens,
        threads,
    ):
        calls.append(
            {
                "logits_shape": tuple(logits.shape),
                "topk_tokens": topk_tokens,
                "threads": threads,
                "lengths": lengths.clone(),
                "row_starts": row_starts.clone(),
                "tail_pad": logits[:, 452:].clone(),
            }
        )
        indices.fill_(-1)
        for row in range(indices.shape[0]):
            length = int(lengths[row].item())
            indices[row, :length] = torch.arange(length, dtype=torch.int32)

    monkeypatch.setattr(
        streaming_topk,
        "prefill_topk_tilelang",
        fake_prefill_topk_tilelang,
    )

    tile_logits = torch.arange(2 * 452, dtype=torch.float32).view(2, 452)
    row_starts = torch.tensor([9216, 9360], dtype=torch.int32)
    row_ends = torch.tensor([9668, 9480], dtype=torch.int32)

    tile_offsets, tile_row_starts = streaming_topk._select_tile_offsets_tilelang(
        tile_logits=tile_logits,
        row_starts=row_starts,
        row_ends=row_ends,
        tile_start=9216,
        topk_tokens=512,
        threads=256,
    )

    assert calls[0]["logits_shape"] == (2, 512)
    assert calls[0]["topk_tokens"] == 512
    assert calls[0]["threads"] == 256
    torch.testing.assert_close(
        calls[0]["lengths"],
        torch.tensor([452, 120], dtype=torch.int32),
    )
    torch.testing.assert_close(
        calls[0]["row_starts"],
        torch.tensor([0, 144], dtype=torch.int32),
    )
    assert torch.isneginf(calls[0]["tail_pad"]).all()
    assert tile_offsets.shape == (2, 512)
    torch.testing.assert_close(
        tile_row_starts,
        torch.tensor([0, 144], dtype=torch.int32),
    )


def test_streaming_topk_prewarm_compiles_request_path_kernels(monkeypatch):
    import vllm.v1.attention.ops.tilelang_prefill_streaming_topk as streaming_topk

    calls = []

    monkeypatch.setattr(
        streaming_topk,
        "is_tilelang_available",
        lambda: (True, None),
    )
    monkeypatch.setattr(
        streaming_topk,
        "prewarm_prefill_topk_tilelang",
        lambda topk, threads: calls.append(("tile_topk", topk, threads)),
    )
    monkeypatch.setattr(
        streaming_topk,
        "_get_fused_candidate_update_kernel",
        lambda topk, tile_keep, threads: calls.append(
            ("candidate_update", topk, tile_keep, threads)
        ),
    )
    monkeypatch.setattr(
        streaming_topk,
        "_get_final_indices_kernel",
        lambda topk, threads: calls.append(("final_copy", topk, threads)),
    )

    streaming_topk.prewarm_prefill_streaming_topk_tilelang(
        512,
        tile_k=1024,
        threads=256,
    )

    assert calls == [
        ("tile_topk", 512, 256),
        ("candidate_update", 512, 512, 256),
        ("final_copy", 512, 256),
    ]


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
