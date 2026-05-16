# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import types

import pytest
import torch


def test_sparse_flashmla_metadata_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 128
    num_heads_k = 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    topk = 128

    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )
    assert tile_md.dtype == torch.int32
    assert num_splits.dtype == torch.int32


def test_sparse_flashmla_decode_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 64
    head_dim_k = 576
    head_dim_v = 512
    num_heads_k = 1
    page_block_size = 64
    bytes_per_token = 656
    topk = 128

    # Metadata
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    # q_heads_per_hk = num_heads_q // num_heads_k
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )

    # Inputs
    q = torch.zeros(
        (batch_size, seqlen_q, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = torch.zeros(
        (1, page_block_size, num_heads_k, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    indices = torch.zeros(
        (batch_size, seqlen_q, topk), dtype=torch.int32, device=device
    )

    block_table = torch.zeros((batch_size, 128), dtype=torch.int32, device=device)
    out, lse = fm.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_md,
        num_splits,
        indices=indices,
        is_fp8_kvcache=True,
    )
    assert out.shape[0] == batch_size
    assert out.shape[-1] == head_dim_v
    assert lse.shape[0] == batch_size


def test_sparse_flashmla_prefill_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    s_q = 1
    s_kv = 1
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    q = torch.zeros((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.zeros((s_q, h_kv, topk), dtype=torch.int32, device=device)

    out, max_logits, lse = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0, d_v)
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)


def test_sparse_flashmla_prefill_dispatches_to_tilelang_when_cached(
    monkeypatch,
):
    import vllm.third_party.flashmla.flash_mla_interface as fm

    monkeypatch.setenv("VLLM_SM70_USE_TILELANG_SPARSE_PREFILL", "1")

    tilelang_mod = types.ModuleType(
        "vllm.v1.attention.ops.tilelang_sparse_prefill"
    )
    calls: list[str] = []

    def fake_is_tilelang_available():
        return True, None

    def fake_is_tilelang_sparse_fwd_cached(*args, **kwargs):
        calls.append("cached")
        return True

    def fake_flash_mla_sparse_fwd_tilelang(*args, **kwargs):
        calls.append("tilelang")
        return ("tilelang", None, None)

    tilelang_mod.is_tilelang_available = fake_is_tilelang_available
    tilelang_mod.is_tilelang_sparse_fwd_cached = fake_is_tilelang_sparse_fwd_cached
    tilelang_mod.flash_mla_sparse_fwd_tilelang = fake_flash_mla_sparse_fwd_tilelang
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.ops.tilelang_sparse_prefill",
        tilelang_mod,
    )

    flashmla_calls = {}

    def fake_sparse_prefill_fwd(*args, **kwargs):
        flashmla_calls["called"] = True
        return ("flashmla", None, None)

    monkeypatch.setattr(
        fm,
        "flash_mla_cuda",
        types.SimpleNamespace(sparse_prefill_fwd=fake_sparse_prefill_fwd),
    )

    q = torch.zeros((1, 1, 576), dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, 576), dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 128), dtype=torch.int32)

    out = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0)

    assert out[0] == "tilelang"
    assert calls == ["cached", "tilelang"]
    assert "called" not in flashmla_calls


def test_sparse_flashmla_prefill_warns_and_falls_back_on_tilelang_cache_miss(
    monkeypatch,
):
    import vllm.third_party.flashmla.flash_mla_interface as fm

    monkeypatch.setenv("VLLM_SM70_USE_TILELANG_SPARSE_PREFILL", "1")

    tilelang_mod = types.ModuleType(
        "vllm.v1.attention.ops.tilelang_sparse_prefill"
    )
    calls: list[str] = []

    def fake_is_tilelang_available():
        return True, None

    def fake_is_tilelang_sparse_fwd_cached(*args, **kwargs):
        calls.append("cached")
        return False

    def fake_flash_mla_sparse_fwd_tilelang(*args, **kwargs):
        calls.append("tilelang")
        return ("tilelang", None, None)

    tilelang_mod.is_tilelang_available = fake_is_tilelang_available
    tilelang_mod.is_tilelang_sparse_fwd_cached = fake_is_tilelang_sparse_fwd_cached
    tilelang_mod.flash_mla_sparse_fwd_tilelang = fake_flash_mla_sparse_fwd_tilelang
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.ops.tilelang_sparse_prefill",
        tilelang_mod,
    )

    flashmla_calls = {}

    def fake_sparse_prefill_fwd(*args, **kwargs):
        flashmla_calls["called"] = True
        return ("flashmla", None, None)

    monkeypatch.setattr(
        fm,
        "flash_mla_cuda",
        types.SimpleNamespace(sparse_prefill_fwd=fake_sparse_prefill_fwd),
    )

    q = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
    kv = torch.zeros((1, 1, 576), dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 128), dtype=torch.int32)

    with pytest.warns(UserWarning, match="TileLang sparse prefill cache miss"):
        out = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0)

    assert out[0] == "flashmla"
    assert calls == ["cached"]
    assert flashmla_calls["called"] is True


def test_tilelang_prefill_cache_key_uses_out_dtype(monkeypatch):
    import vllm.v1.attention.ops.tilelang_sparse_prefill as tilelang_prefill

    captured = {}

    def fake_is_cached(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(tilelang_prefill, "_is_kernel_cached", fake_is_cached)

    q = torch.zeros((1, 64, 576), dtype=torch.float16)
    kv = torch.zeros((1, 1, 576), dtype=torch.float16)
    indices = torch.zeros((1, 1, 128), dtype=torch.int32)
    out = torch.empty((1, 64, 512), dtype=torch.bfloat16)

    assert tilelang_prefill.is_tilelang_sparse_fwd_cached(
        q, kv, indices, 1.0, 512, out=out
    )
    assert captured["output_dtype_str"] == "bfloat16"


def test_tilelang_prefill_writes_requested_out_dtype(monkeypatch):
    import vllm.v1.attention.ops.tilelang_sparse_prefill as tilelang_prefill

    captured = {}

    def fake_get_kernel(**kwargs):
        captured.update(kwargs)

        def fake_kernel(Q_b, _KV_b, _Indices_b, _Sink, _TopkLen_b):
            s_q, h_q, d_qk = Q_b.shape[1:]
            assert d_qk == 576
            out_dtype = (
                torch.bfloat16
                if kwargs["output_dtype_str"] == "bfloat16"
                else torch.float16
            )
            return (
                torch.full((1, s_q, h_q, 512), 3.0, dtype=out_dtype),
                torch.zeros((1, s_q, h_q), dtype=torch.float32),
                torch.zeros((1, s_q, h_q), dtype=torch.float32),
            )

        return fake_kernel

    monkeypatch.setattr(tilelang_prefill, "_get_kernel", fake_get_kernel)

    q = torch.zeros((1, 64, 576), dtype=torch.float16)
    kv = torch.zeros((1, 1, 576), dtype=torch.float16)
    indices = torch.zeros((1, 1, 128), dtype=torch.int32)
    out = torch.empty((1, 64, 512), dtype=torch.bfloat16)

    result, _max_logits, _lse = tilelang_prefill.flash_mla_sparse_fwd_tilelang(
        q, kv, indices, 1.0, 512, out=out
    )

    assert captured["output_dtype_str"] == "bfloat16"
    assert result.data_ptr() == out.data_ptr()
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, torch.full_like(out, 3.0))


def test_sparse_flashmla_prefill_matches_torch_reference():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    s_q = 1
    h_q = 64
    d = 512
    topk = 128
    sm_scale = 0.25

    q = torch.zeros((s_q, h_q, d), dtype=torch.bfloat16, device=device)
    q[:, :, 0] = 2.0
    q[:, :, 1] = -1.0

    kv = torch.zeros((3, 1, d), dtype=torch.bfloat16, device=device)
    kv[0, 0, 0] = 1.0
    kv[0, 0, 2] = 10.0
    kv[1, 0, 1] = -2.0
    kv[1, 0, 2] = -4.0
    kv[2, 0, 0] = -3.0
    kv[2, 0, 2] = 7.0

    indices = torch.full((s_q, 1, topk), -1, dtype=torch.int32, device=device)
    indices[0, 0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    topk_length = torch.tensor([3], dtype=torch.int32, device=device)

    out, max_logits, lse = fm.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d,
        topk_length=topk_length,
    )

    q_ref = q.float()
    kv_ref = kv[:, 0, :].float()
    scores = torch.einsum("shd,kd->shk", q_ref, kv_ref) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("shk,kd->shd", weights, kv_ref).to(out.dtype)
    expected_max = scores.max(dim=-1).values
    expected_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(max_logits, expected_max, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("magnitude", [448.0, 1024.0])
def test_sparse_flashmla_prefill_matches_reference_for_large_values(
    magnitude: float,
):
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    s_q = 1
    h_q = 64
    d = 512
    topk = 128
    sm_scale = d**-0.5

    q = torch.zeros((s_q, h_q, d), dtype=torch.bfloat16, device=device)
    q[:, :, 0] = magnitude
    q[:, :, 1] = -magnitude / 2

    kv = torch.zeros((3, 1, d), dtype=torch.bfloat16, device=device)
    kv[0, 0, 0] = magnitude
    kv[0, 0, 2] = magnitude
    kv[1, 0, 1] = -magnitude
    kv[1, 0, 2] = -magnitude
    kv[2, 0, 0] = -magnitude
    kv[2, 0, 2] = magnitude / 2

    indices = torch.full((s_q, 1, topk), -1, dtype=torch.int32, device=device)
    indices[0, 0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    topk_length = torch.tensor([3], dtype=torch.int32, device=device)

    out, max_logits, lse = fm.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d,
        topk_length=topk_length,
    )

    q_ref = q.float()
    kv_ref = kv[:, 0, :].float()
    scores = torch.einsum("shd,kd->shk", q_ref, kv_ref) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("shk,kd->shd", weights, kv_ref).to(out.dtype)
    expected_max = scores.max(dim=-1).values
    expected_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(max_logits, expected_max, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-4)
