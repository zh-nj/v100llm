# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers import deepseek_v4_attention as d4a
from vllm.model_executor.layers.deepseek_v4_attention import DeepseekV4MLAAttention


def test_forward_decode_uses_bf16_flashmla_io_for_fp16_model(monkeypatch):
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 1
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(64, dtype=torch.float32)
    attn.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(2, 16, 584, dtype=torch.uint8)
    )

    q = torch.randn(2, 64, 512, dtype=torch.float16)
    output = torch.empty_like(q)
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=torch.zeros(2, 1, 1, dtype=torch.int32),
        decode_swa_lens=torch.ones(2, 1, dtype=torch.int32),
        tile_sched_swaonly=torch.zeros(1, dtype=torch.int32),
    )
    captured = {}

    def fake_flash_mla_with_kvcache(**kwargs):
        captured["q_dtype"] = kwargs["q"].dtype
        captured["out_dtype"] = kwargs["out"].dtype
        kwargs["out"].fill_(1)
        return kwargs["out"], None

    monkeypatch.setattr(d4a, "flash_mla_with_kvcache", fake_flash_mla_with_kvcache)

    attn._forward_decode(
        q=q,
        kv_cache=None,
        swa_metadata=swa_metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert captured == {
        "q_dtype": torch.bfloat16,
        "out_dtype": torch.bfloat16,
    }
    assert output.dtype == torch.float16
    torch.testing.assert_close(output, torch.ones_like(output))


def test_sm70_prefill_kv_staging_preserves_bf16_range():
    kv = torch.tensor([-765952.0, -449.0, 448.0, 765952.0],
                      dtype=torch.bfloat16)

    normalized = d4a._normalize_flashmla_sm70_prefill_kv_(kv)

    assert normalized is kv
    expected = torch.tensor([-765952.0, -449.0, 448.0, 765952.0],
                            dtype=torch.bfloat16)
    torch.testing.assert_close(kv, expected)


def test_sm70_attention_output_clamp_keeps_fp16_values_finite():
    out = torch.tensor(
        [
            -float("inf"),
            -65504.0,
            -1.0,
            1.0,
            65504.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float16,
    )

    clamped = d4a._clamp_sm70_fp16_attention_output_(out)

    assert clamped is out
    expected = torch.tensor(
        [-65504.0, -65504.0, -1.0, 1.0, 65504.0, 65504.0],
        dtype=torch.float16,
    )
    torch.testing.assert_close(out[:-1], expected)
    assert torch.isnan(out[-1])


def test_sm70_fp8_cache_exponents_preserve_upstream_positive_scales():
    exponents = torch.tensor([8.0, 5.0, 1.0, 0.0, -2.0], dtype=torch.float32)

    preserved = d4a._normalize_sm70_fp8_cache_exponents(exponents)

    torch.testing.assert_close(preserved, exponents)
    max_dequant = 448.0 * torch.exp2(preserved.max())
    assert max_dequant.item() == 448.0 * 256.0


def test_sm70_decode_prefill_fallback_builds_local_indices():
    global_indices = torch.tensor(
        [[[10, 11, -1, -1]], [[64, 65, 66, -1]]],
        dtype=torch.int32,
    )
    lens = torch.tensor([2, 3], dtype=torch.int32)

    local_indices, local_lens = d4a._build_decode_prefill_fallback_indices(
        global_indices, lens
    )

    torch.testing.assert_close(local_lens, lens)
    expected = torch.tensor(
        [[[0, 1, -1, -1]], [[4, 5, 6, -1]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(local_indices, expected)


def test_sm70_decode_prefill_fallback_gathers_v4_cache_layout():
    block_size = 64
    k_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8, device="cuda")
    cache_2d = k_cache.reshape(1, -1)

    def store(slot: int, nope_value: float, rope_offset: float) -> None:
        pos = slot % block_size
        token_offset = pos * 576
        scale_offset = block_size * 576 + pos * 8
        nope = torch.full((448,), nope_value, dtype=torch.float32)
        rope = torch.arange(64, dtype=torch.float32) + rope_offset
        cache_2d[0, token_offset:token_offset + 448] = (
            nope.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
        ).cuda()
        cache_2d[0, token_offset + 448:token_offset + 576] = (
            rope.to(torch.bfloat16).contiguous().view(torch.uint8)
        ).cuda()
        cache_2d[0, scale_offset:scale_offset + 7] = 127

    store(1, 2.0, 100.0)
    store(3, -3.0, 200.0)

    out = torch.empty(1, 4, 512, dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([[[1, 3, -1, -1]]], dtype=torch.int32, device="cuda")
    lens = torch.tensor([2], dtype=torch.int32, device="cuda")

    d4a._gather_decode_prefill_fallback_kv_(
        out, k_cache, indices, lens, block_size
    )

    torch.testing.assert_close(
        out[0, 0, :448].cpu(), torch.full((448,), 2.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        out[0, 1, :448].cpu(), torch.full((448,), -3.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        out[0, 0, 448:].cpu(),
        (torch.arange(64, dtype=torch.float32) + 100.0).to(torch.bfloat16),
    )
    torch.testing.assert_close(
        out[0, 1, 448:].cpu(),
        (torch.arange(64, dtype=torch.float32) + 200.0).to(torch.bfloat16),
    )
    torch.testing.assert_close(out[0, 2:].cpu(), torch.zeros(2, 512, dtype=torch.bfloat16))


def test_qnorm_rope_kv_insert_fallback_stores_fp16_rope_tail_as_bf16():
    head_dim = 512
    nope_dim = 448
    rope_dim = 64
    head_bytes = 584
    block_size = 16

    q = torch.zeros(1, 1, head_dim, dtype=torch.float16)
    kv = torch.zeros(1, head_dim, dtype=torch.float16)
    kv[0, nope_dim:] = torch.linspace(-2.0, 2.0, rope_dim).to(torch.float16)
    k_cache = torch.zeros(1, block_size * head_bytes, dtype=torch.uint8)
    slot_mapping = torch.tensor([0], dtype=torch.int64)
    positions = torch.tensor([0], dtype=torch.int64)
    cos_sin_cache = torch.cat(
        (
            torch.ones(1, rope_dim // 2, dtype=torch.float32),
            torch.zeros(1, rope_dim // 2, dtype=torch.float32),
        ),
        dim=-1,
    )

    d4a._torch_qnorm_rope_kv_insert_fallback(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        1e-6,
        block_size,
    )

    stored_rope = k_cache[0, nope_dim:nope_dim + rope_dim * 2]
    expected_rope = (
        kv[0, nope_dim:].to(torch.bfloat16).contiguous().view(torch.uint8)
    )

    torch.testing.assert_close(stored_rope, expected_rope, rtol=0, atol=0)


def test_forward_decode_uses_sparse_prefill_fallback_for_sm70_swa_only(monkeypatch):
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 1
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(64, dtype=torch.float32)
    attn.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(2, 64, 584, dtype=torch.uint8)
    )

    q = torch.randn(2, 64, 512, dtype=torch.float16)
    output = torch.empty_like(q)
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=torch.tensor(
            [[[0, 1, -1]], [[64, -1, -1]]],
            dtype=torch.int32,
        ),
        decode_swa_lens=torch.tensor([2, 1], dtype=torch.int32),
        block_size=64,
        tile_sched_swaonly=torch.zeros(1, dtype=torch.int32),
    )
    captured = {}

    monkeypatch.setattr(
        d4a, "_should_use_sm70_decode_prefill_fallback", lambda q, swa_only: True
    )

    def fail_direct_decode(**_kwargs):
        raise AssertionError("direct FlashMLA decode should not be called")

    def fake_gather_with_indices(out, _k_cache, global_indices, global_lens, _block_size, *, row_stride=None, offset=0):
        out.fill_(2)
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _build_decode_prefill_fallback_indices,
        )
        local_indices, local_lens = _build_decode_prefill_fallback_indices(
            global_indices, global_lens, row_stride=row_stride, offset=offset,
        )
        return out, local_indices, local_lens

    def fake_sparse_prefill(**kwargs):
        captured["q_shape"] = kwargs["q"].shape
        captured["kv_shape"] = kwargs["kv"].shape
        captured["indices"] = kwargs["indices"].clone()
        captured["topk_length"] = kwargs["topk_length"].clone()
        kwargs["out"].fill_(3)
        return kwargs["out"], None, None

    monkeypatch.setattr(d4a, "flash_mla_with_kvcache", fail_direct_decode)
    monkeypatch.setattr(d4a, "_gather_decode_prefill_fallback_kv_with_indices_", fake_gather_with_indices)
    monkeypatch.setattr(d4a, "flash_mla_sparse_fwd", fake_sparse_prefill)

    attn._forward_decode(
        q=q,
        kv_cache=None,
        swa_metadata=swa_metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert captured["q_shape"] == (2, 64, 512)
    assert captured["kv_shape"] == (6, 1, 512)
    torch.testing.assert_close(
        captured["indices"],
        torch.tensor([[[0, 1, -1]], [[3, -1, -1]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["topk_length"], torch.tensor([2, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(output, torch.full_like(output, 3))


def test_forward_decode_uses_sparse_prefill_fallback_for_sm70_compressed(
    monkeypatch,
):
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 4
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(64, dtype=torch.float32)
    attn.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(2, 64, 584, dtype=torch.uint8)
    )
    attn.topk_indices_buffer = torch.zeros(2, 2, dtype=torch.int32)

    q = torch.randn(2, 64, 512, dtype=torch.float16)
    output = torch.empty_like(q)
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=torch.tensor(
            [[[10, 11, -1]], [[12, -1, -1]]],
            dtype=torch.int32,
        ),
        decode_swa_lens=torch.tensor([2, 1], dtype=torch.int32),
        block_size=64,
        is_valid_token=torch.ones(2, dtype=torch.bool),
        token_to_req_indices=torch.zeros(2, dtype=torch.int32),
        tile_sched_c4a=torch.zeros(1, dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(
        block_size=256,
        block_table=torch.zeros(2, 4, dtype=torch.int32),
    )
    kv_cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
    captured = {"gather_shapes": []}

    monkeypatch.setattr(
        d4a, "_should_use_sm70_decode_prefill_fallback", lambda q, swa_only: True
    )
    monkeypatch.setattr(
        d4a,
        "compute_global_topk_indices_and_lens",
        lambda *_args, **_kwargs: (
            torch.tensor([[0, -1], [64, 65]], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),
        ),
    )

    def fail_direct_decode(**_kwargs):
        raise AssertionError("direct FlashMLA decode should not be called")

    def fake_gather_with_indices(out, _k_cache, global_indices, global_lens, _block_size, *, row_stride=None, offset=0):
        captured["gather_shapes"].append(tuple(out.shape))
        out.fill_(2)
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _build_decode_prefill_fallback_indices,
        )
        local_indices, local_lens = _build_decode_prefill_fallback_indices(
            global_indices, global_lens, row_stride=row_stride, offset=offset,
        )
        return out, local_indices, local_lens

    def fake_sparse_prefill(**kwargs):
        captured["indices"] = kwargs["indices"].clone()
        captured["topk_length"] = kwargs["topk_length"]
        captured["kv_shape"] = kwargs["kv"].shape
        kwargs["out"].fill_(3)
        return kwargs["out"], None, None

    monkeypatch.setattr(d4a, "flash_mla_with_kvcache", fail_direct_decode)
    monkeypatch.setattr(d4a, "_gather_decode_prefill_fallback_kv_with_indices_", fake_gather_with_indices)
    monkeypatch.setattr(d4a, "flash_mla_sparse_fwd", fake_sparse_prefill)

    attn._forward_decode(
        q=q,
        kv_cache=kv_cache,
        swa_metadata=swa_metadata,
        attn_metadata=attn_metadata,
        swa_only=False,
        output=output,
    )

    assert captured["gather_shapes"] == [(2, 2, 512), (2, 3, 512)]
    assert captured["kv_shape"] == (10, 1, 512)
    torch.testing.assert_close(
        captured["indices"],
        torch.tensor(
            [[[0, -1, 2, 3, -1]], [[5, 6, 7, -1, -1]]],
            dtype=torch.int32,
        ),
    )
    assert captured["topk_length"] is None
    torch.testing.assert_close(output, torch.full_like(output, 3))
