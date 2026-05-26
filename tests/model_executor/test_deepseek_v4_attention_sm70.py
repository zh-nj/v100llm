# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm.model_executor.layers import deepseek_v4_attention as d4a
from vllm.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4MLAAttention,
    PrefillCaptureSize,
    PrefillGraphDispatcher,
)


def test_copy_flashmla_output_avoids_full_dtype_conversion_temp(monkeypatch):
    flash_output = torch.tensor([[1.5, -2.25]], dtype=torch.bfloat16)
    output = torch.empty_like(flash_output, dtype=torch.float16)
    original_to = torch.Tensor.to

    def fail_flash_output_to(self, *args, **kwargs):
        if self is flash_output:
            raise AssertionError("flash_output.to() would allocate a full temp")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_flash_output_to, raising=False)

    d4a._copy_flashmla_output(flash_output, output)

    expected = torch.tensor([[1.5, -2.25]], dtype=torch.float16)
    torch.testing.assert_close(output, expected)


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


def test_sm70_prefill_chunk_size_env_overrides_default(monkeypatch):
    monkeypatch.delenv("VLLM_DEEPSEEK_V4_PREFILL_CHUNK_SIZE", raising=False)
    assert d4a._get_prefill_chunk_size() == 4

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_PREFILL_CHUNK_SIZE", "16")
    assert d4a._get_prefill_chunk_size() == 16

    monkeypatch.setenv("VLLM_DEEPSEEK_V4_PREFILL_CHUNK_SIZE", "0")
    assert d4a._get_prefill_chunk_size() == 1


def test_sm70_fp8_cache_exponents_preserve_upstream_positive_scales():
    exponents = torch.tensor([8.0, 5.0, 1.0, 0.0, -2.0], dtype=torch.float32)

    preserved = d4a._normalize_sm70_fp8_cache_exponents(exponents)

    torch.testing.assert_close(preserved, exponents)
    max_dequant = 448.0 * torch.exp2(preserved.max())
    assert max_dequant.item() == 448.0 * 256.0


def test_sm70_decode_uses_direct_flashmla_by_default_with_opt_out(
    monkeypatch,
):
    q = SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device=None: (7, 0)
    )
    monkeypatch.delenv("VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE", raising=False)

    assert not d4a._should_use_sm70_decode_prefill_fallback(q, swa_only=False)
    assert not d4a._should_use_sm70_decode_prefill_fallback(q, swa_only=True)

    monkeypatch.setenv("VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE", "0")

    assert d4a._should_use_sm70_decode_prefill_fallback(q, swa_only=False)
    assert d4a._should_use_sm70_decode_prefill_fallback(q, swa_only=True)


def test_sm70_decode_uses_prefill_fallback_when_direct_topk_exceeds_limit(
    monkeypatch,
):
    q = SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device=None: (7, 0)
    )

    assert not d4a._should_use_sm70_decode_prefill_fallback_for_total_topk(
        q, direct_total_topk=8192
    )
    assert d4a._should_use_sm70_decode_prefill_fallback_for_total_topk(
        q, direct_total_topk=8320
    )


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


def test_forward_decode_uses_sparse_prefill_fallback_for_large_c128a_topk(
    monkeypatch,
):
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 128
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(64, dtype=torch.float32)
    attn.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(2, 64, 584, dtype=torch.uint8)
    )

    q = torch.randn(1, 64, 512, dtype=torch.float16)
    output = torch.empty_like(q)
    swa_metadata = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        decode_swa_indices=torch.full((1, 1, 128), -1, dtype=torch.int32),
        decode_swa_lens=torch.tensor([0], dtype=torch.int32),
        block_size=64,
        is_valid_token=torch.ones(1, dtype=torch.bool),
        tile_sched_c128a=torch.zeros(1, dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(
        block_size=256,
        c128a_global_decode_topk_indices=torch.full(
            (1, 1, 8192), -1, dtype=torch.int32
        ),
        c128a_decode_topk_lens=torch.tensor([0], dtype=torch.int32),
    )
    kv_cache = torch.zeros(2, 2, 584, dtype=torch.uint8)
    captured = {}

    monkeypatch.setattr(
        d4a, "_should_use_sm70_decode_prefill_fallback", lambda q, swa_only: False
    )

    def fake_total_topk_guard(_q, direct_total_topk):
        captured["direct_total_topk"] = direct_total_topk
        return True

    def fail_direct_decode(**_kwargs):
        raise AssertionError("direct FlashMLA decode should not be called")

    def fake_gather_with_indices(out, _k_cache, global_indices, global_lens, _block_size, *, row_stride=None, offset=0):
        from vllm.model_executor.layers.deepseek_v4_attention import (
            _build_decode_prefill_fallback_indices,
        )
        local_indices, local_lens = _build_decode_prefill_fallback_indices(
            global_indices, global_lens, row_stride=row_stride, offset=offset,
        )
        return out.zero_(), local_indices, local_lens

    def fake_sparse_prefill(**kwargs):
        captured["kv_shape"] = kwargs["kv"].shape
        captured["indices_shape"] = kwargs["indices"].shape
        kwargs["out"].fill_(5)
        return kwargs["out"], None, None

    monkeypatch.setattr(
        d4a,
        "_should_use_sm70_decode_prefill_fallback_for_total_topk",
        fake_total_topk_guard,
    )
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

    assert captured["direct_total_topk"] == 8320
    assert captured["kv_shape"] == (8320, 1, 512)
    assert captured["indices_shape"] == (1, 1, 8320)
    torch.testing.assert_close(output, torch.full_like(output, 5))


def test_prefill_cudagraph_default_capture_sizes_cover_token_counts():
    sizes = d4a._default_prefill_cudagraph_capture_sizes(
        max_num_batched_tokens=4096,
        max_model_len=4096,
        max_M=8192,
    )

    assert sizes[:4] == [(1, 8192), (2, 8192), (4, 8192), (8, 8192)]
    assert (1024, 8192) in sizes
    assert (4096, 8192) in sizes
    assert sizes[-1] == (4096, 8192)


def test_tilelang_sparse_prefill_prewarm_topks_include_long_c128a_buckets():
    topks = d4a._tilelang_sparse_prefill_prewarm_topks(
        hf_index_topk=512,
        window_size=128,
        max_model_len=524288,
    )

    assert topks == [
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        1152,
        1664,
        2176,
        2688,
        3200,
        3712,
        4224,
    ]


def test_tilelang_sparse_prefill_default_prewarm_context_caps_startup_buckets():
    topks = d4a._tilelang_sparse_prefill_prewarm_topks(
        hf_index_topk=512,
        window_size=128,
        max_model_len=524288,
        prewarm_max_context_len=65536,
    )

    assert topks == [128, 256, 384, 512, 640]


def test_prefill_graph_dispatcher_allocates_private_storage(monkeypatch):
    def fail_workspace_manager():
        raise AssertionError("scratch workspace manager must not be used")

    monkeypatch.setattr(d4a, "current_workspace_manager", fail_workspace_manager)

    dispatcher = PrefillGraphDispatcher(
        capture_sizes=[(2, 4)],
        padded_heads=1,
        head_dim=2,
        scale=1.0,
        attn_sink=torch.zeros(1),
        device=torch.device("cpu"),
    )

    assert dispatcher.enabled
    assert dispatcher.q_padded.shape == (2, 1, 2)
    assert dispatcher.kv_padded.shape == (d4a.PREFILL_CHUNK_SIZE, 4, 2)


def test_prefill_graph_replay_uses_runtime_kv_workspace():
    dispatcher = object.__new__(PrefillGraphDispatcher)
    dispatcher.padded_heads = 1
    dispatcher.head_dim = 2
    dispatcher.scale = 1.0
    dispatcher.attn_sink_workspace = torch.zeros(1)
    dispatcher.device = torch.device("cpu")
    dispatcher.enabled = True
    dispatcher.graph_dispatch_count = 0
    dispatcher.eager_dispatch_count = 0
    dispatcher.sizes = [PrefillCaptureSize(max_tokens=2, max_M=4, graph=None)]
    dispatcher.q_padded = torch.zeros(2, 1, 2, dtype=torch.float16)
    dispatcher.indices_padded = torch.full((2, 1, 4), -1, dtype=torch.int32)
    dispatcher.topk_length_padded = torch.zeros(2, dtype=torch.int32)
    dispatcher.kv_padded = torch.zeros(4, 4, 2, dtype=torch.bfloat16)
    dispatcher.output_padded = torch.zeros(2, 1, 2, dtype=torch.float16)
    dispatcher.flash_output_bf16 = torch.zeros(2, 1, 2, dtype=torch.bfloat16)

    class FakeGraph:
        def replay(self):
            dispatcher.output_padded.fill_(float(dispatcher.kv_padded[0, 0, 0]))

    dispatcher.sizes[0].graph = FakeGraph()
    q = torch.ones(2, 1, 2, dtype=torch.float16)
    kv = torch.full((4, 4, 2), 7.0, dtype=torch.bfloat16)
    output = torch.empty_like(q)

    replayed = dispatcher.try_graph_replay(
        q_chunk=q,
        kv_flat=kv.view(-1, 1, 2),
        combined_indices=torch.tensor([[[0, 1, -1]], [[2, -1, -1]]], dtype=torch.int32),
        combined_lens=torch.tensor([2, 1], dtype=torch.int32),
        output_slice=output,
        num_chunk_tokens=2,
        num_chunk_reqs=2,
        M=4,
        attn_sink=torch.tensor([3.0]),
    )

    assert replayed
    torch.testing.assert_close(
        dispatcher.kv_padded[:2], kv[:2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        dispatcher.kv_padded[2:], torch.zeros_like(dispatcher.kv_padded[2:])
    )
    torch.testing.assert_close(output, torch.full_like(output, 7.0))
    torch.testing.assert_close(dispatcher.attn_sink_workspace, torch.tensor([3.0]))


def test_prefill_graph_replay_copies_per_token_lens_and_request_kv():
    dispatcher = object.__new__(PrefillGraphDispatcher)
    dispatcher.padded_heads = 1
    dispatcher.head_dim = 2
    dispatcher.scale = 1.0
    dispatcher.attn_sink_workspace = torch.zeros(1)
    dispatcher.device = torch.device("cpu")
    dispatcher.enabled = True
    dispatcher.graph_dispatch_count = 0
    dispatcher.eager_dispatch_count = 0
    dispatcher.sizes = [PrefillCaptureSize(max_tokens=4, max_M=4, graph=None)]
    dispatcher.q_padded = torch.zeros(4, 1, 2, dtype=torch.float16)
    dispatcher.indices_padded = torch.full((4, 1, 4), -1, dtype=torch.int32)
    dispatcher.topk_length_padded = torch.zeros(4, dtype=torch.int32)
    dispatcher.kv_padded = torch.zeros(4, 4, 2, dtype=torch.bfloat16)
    dispatcher.output_padded = torch.zeros(4, 1, 2, dtype=torch.float16)
    dispatcher.flash_output_bf16 = torch.zeros(4, 1, 2, dtype=torch.bfloat16)

    class FakeGraph:
        def replay(self):
            torch.testing.assert_close(
                dispatcher.topk_length_padded,
                torch.tensor([2, 3, 4, 0], dtype=torch.int32),
            )
            torch.testing.assert_close(
                dispatcher.kv_padded[1:],
                torch.zeros_like(dispatcher.kv_padded[1:]),
            )
            dispatcher.output_padded.fill_(5)

    dispatcher.sizes[0].graph = FakeGraph()
    q = torch.ones(3, 1, 2, dtype=torch.float16)
    kv = torch.full((4, 4, 2), 7.0, dtype=torch.bfloat16)
    output = torch.empty_like(q)

    replayed = dispatcher.try_graph_replay(
        q_chunk=q,
        kv_flat=kv.view(-1, 1, 2),
        combined_indices=torch.tensor(
            [[[0, 1, -1]], [[2, 3, -1]], [[1, 2, 3]]],
            dtype=torch.int32,
        ),
        combined_lens=torch.tensor([2, 3, 4], dtype=torch.int32),
        output_slice=output,
        num_chunk_tokens=3,
        num_chunk_reqs=1,
        M=4,
    )

    assert replayed
    torch.testing.assert_close(output, torch.full_like(output, 5.0))


def test_forward_prefill_fast_io_skips_q_bf16_trampoline_when_tilelang_cached(
    monkeypatch,
):
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 1
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(1, dtype=torch.float32)
    attn.window_size = 2
    attn.max_num_batched_tokens = 2
    attn.max_model_len = 4
    attn.head_dim = 2
    attn.topk_indices_buffer = torch.zeros(2, 1, dtype=torch.int32)
    attn._prefill_graph_dispatcher = None
    attn.prefix = "layers.0.attn"

    q = torch.randn(2, 1, 2, dtype=torch.float16)
    output = torch.empty_like(q)
    kv_workspace = torch.empty(
        d4a.PREFILL_CHUNK_SIZE,
        attn.window_size + attn.max_num_batched_tokens,
        q.shape[-1],
        dtype=torch.bfloat16,
    )
    swa_metadata = SimpleNamespace(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        prefill_seq_lens=torch.tensor([2], dtype=torch.int32),
        prefill_gather_lens=torch.tensor([2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        block_size=2,
    )
    captured = {}

    class FakeWorkspaceManager:
        def get_simultaneous(self, _specs):
            return [kv_workspace]

    def fake_gather(out, *_args, **_kwargs):
        out.fill_(2)

    def fake_combine(*_args, **_kwargs):
        return (
            torch.tensor([[0, 1], [1, -1]], dtype=torch.int32),
            torch.tensor([2, 1], dtype=torch.int32),
        )

    from vllm.v1.attention.ops import tilelang_sparse_prefill

    def fake_tilelang_prefill(**kwargs):
        captured["q_dtype"] = kwargs["q"].dtype
        captured["output_dtype"] = kwargs["output_dtype"]
        captured["has_out"] = "out" in kwargs
        captured["out_dtype"] = kwargs["out"].dtype
        captured["heads_per_block"] = kwargs["heads_per_block"]
        captured["threads"] = kwargs["threads"]
        captured["pv_gemm_policy"] = kwargs["pv_gemm_policy"]
        captured["assume_valid_indices"] = kwargs["assume_valid_indices"]
        return (
            torch.full(output.shape, 9, dtype=torch.bfloat16),
            None,
            None,
        )

    def fake_copy_flashmla_output(flash_output, output_slice):
        captured["copy_from_dtype"] = flash_output.dtype
        captured["copy_to_dtype"] = output_slice.dtype
        output_slice.copy_(flash_output.to(output_slice.dtype))

    monkeypatch.setattr(
        d4a, "current_workspace_manager", lambda: FakeWorkspaceManager()
    )
    monkeypatch.setattr(d4a, "dequantize_and_gather_k_cache", fake_gather)
    monkeypatch.setattr(d4a, "combine_topk_swa_indices", fake_combine)
    monkeypatch.setattr(
        tilelang_sparse_prefill,
        "flash_mla_sparse_fwd_tilelang",
        fake_tilelang_prefill,
    )
    monkeypatch.setattr(
        d4a,
        "flash_mla_sparse_fwd",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic FlashMLA path should be skipped")
        ),
    )
    monkeypatch.setattr(
        d4a,
        "_should_use_tilelang_sparse_prefill_fast_io",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        d4a.envs,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK",
        32,
    )
    monkeypatch.setattr(
        d4a.envs,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_THREADS",
        64,
    )
    monkeypatch.setattr(
        d4a.envs,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_PV_POLICY",
        "full_col",
    )
    monkeypatch.setattr(
        d4a,
        "_flashmla_bf16_io",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("q bf16 trampoline should be skipped")
        ),
    )
    monkeypatch.setattr(d4a, "_copy_flashmla_output", fake_copy_flashmla_output)

    attn._forward_prefill(
        q=q,
        positions=torch.arange(2, dtype=torch.int64),
        compressed_k_cache=None,
        swa_k_cache=torch.zeros(1, 2, 584, dtype=torch.uint8),
        output=output,
        attn_metadata=None,
        swa_metadata=swa_metadata,
    )

    assert captured == {
        "q_dtype": torch.float16,
        "output_dtype": torch.bfloat16,
        "has_out": True,
        "out_dtype": torch.float16,
        "heads_per_block": 32,
        "threads": 64,
        "pv_gemm_policy": "full_col",
        "assume_valid_indices": False,
        "copy_from_dtype": torch.bfloat16,
        "copy_to_dtype": torch.float16,
    }
    torch.testing.assert_close(output, torch.full_like(output, 9))


def test_forward_prefill_sizes_compressed_workspace_from_runtime_seq_len(
    monkeypatch,
):
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 4
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(1, dtype=torch.float32)
    attn.window_size = 8
    attn.max_num_batched_tokens = 32
    attn.max_model_len = 1024
    attn.head_dim = 2
    attn.topk_indices_buffer = torch.zeros(2, 1, dtype=torch.int32)
    attn._prefill_graph_dispatcher = None
    attn.prefix = "layers.0.attn"

    q = torch.randn(2, 1, 2, dtype=torch.float16)
    output = torch.empty_like(q)
    swa_metadata = SimpleNamespace(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        prefill_seq_lens=torch.tensor([16], dtype=torch.int32),
        prefill_gather_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        block_size=8,
    )
    attn_metadata = SimpleNamespace(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        block_size=16,
    )
    captured = {}

    class FakeWorkspaceManager:
        def get_simultaneous(self, spec):
            shape, dtype = spec
            captured["workspace_shape"] = shape
            return [torch.empty(shape, dtype=dtype)]

    def fake_gather(out, *_args, **_kwargs):
        out.fill_(2)

    def fake_combine(*_args):
        captured["combine_M"] = _args[-2]
        captured["combine_N"] = _args[-1]
        return (
            torch.zeros(2, 128, dtype=torch.int32),
            torch.full((2,), 4, dtype=torch.int32),
        )

    def fake_flash_mla_sparse_fwd(**kwargs):
        kwargs["out"].fill_(7)
        return kwargs["out"], None, None

    monkeypatch.setattr(
        d4a, "current_workspace_manager", lambda: FakeWorkspaceManager()
    )
    monkeypatch.setattr(d4a, "dequantize_and_gather_k_cache", fake_gather)
    monkeypatch.setattr(d4a, "combine_topk_swa_indices", fake_combine)
    monkeypatch.setattr(
        d4a, "_should_use_tilelang_sparse_prefill_fast_io", lambda **_kwargs: False
    )
    monkeypatch.setattr(d4a, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)

    attn._forward_prefill(
        q=q,
        positions=torch.arange(2, dtype=torch.int64),
        compressed_k_cache=torch.zeros(1, 4, 584, dtype=torch.uint8),
        swa_k_cache=torch.zeros(1, 8, 584, dtype=torch.uint8),
        output=output,
        attn_metadata=attn_metadata,
        swa_metadata=swa_metadata,
    )

    expected_n = 16 // attn.compress_ratio
    expected_m = expected_n + attn.window_size + attn.max_num_batched_tokens
    assert captured["workspace_shape"] == (
        d4a.PREFILL_CHUNK_SIZE,
        expected_m,
        q.shape[-1],
    )
    assert captured["combine_N"] == expected_n
    assert captured["combine_M"] == expected_m


def test_tilelang_sparse_prefill_fast_io_default_on(monkeypatch):
    import vllm.envs as env_module

    monkeypatch.delenv("VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO", raising=False)

    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO"
    ]()


def test_tilelang_sparse_prefill_fast_io_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO", False
    )

    q = torch.zeros(1, 1, 2, dtype=torch.float16)

    assert not d4a._should_use_tilelang_sparse_prefill_fast_io(
        q=q,
        kv=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        indices=torch.zeros(1, 1, 2, dtype=torch.int32),
        sm_scale=1.0,
        d_v=2,
        attn_sink=None,
        topk_length=None,
        output=torch.empty_like(q),
    )


def test_tilelang_sparse_prefill_fast_io_requires_cached_shape(monkeypatch):
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO", True
    )
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_USE_TILELANG_SPARSE_PREFILL", True
    )
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_TILELANG_SPARSE_PREFILL_JIT_ON_MISS", False
    )

    import vllm.v1.attention.ops.tilelang_sparse_prefill as tilelang_prefill

    monkeypatch.setattr(
        tilelang_prefill, "is_tilelang_available", lambda: (True, None)
    )
    monkeypatch.setattr(
        tilelang_prefill, "is_tilelang_sparse_fwd_cached", lambda *a, **k: False
    )

    q = torch.zeros(1, 1, 2, dtype=torch.float16)

    assert not d4a._should_use_tilelang_sparse_prefill_fast_io(
        q=q,
        kv=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        indices=torch.zeros(1, 1, 2, dtype=torch.int32),
        sm_scale=1.0,
        d_v=2,
        attn_sink=None,
        topk_length=None,
        output=torch.empty_like(q),
    )


def test_tilelang_sparse_prefill_fast_io_can_jit_uncached_shape(monkeypatch):
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO", True
    )
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_USE_TILELANG_SPARSE_PREFILL", True
    )
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_TILELANG_SPARSE_PREFILL_JIT_ON_MISS", True
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False
    )

    import vllm.v1.attention.ops.tilelang_sparse_prefill as tilelang_prefill

    monkeypatch.setattr(
        tilelang_prefill, "is_tilelang_available", lambda: (True, None)
    )
    monkeypatch.setattr(
        tilelang_prefill, "is_tilelang_sparse_fwd_cached", lambda *a, **k: False
    )

    q = torch.zeros(1, 1, 2, dtype=torch.float16)

    assert d4a._should_use_tilelang_sparse_prefill_fast_io(
        q=q,
        kv=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        indices=torch.zeros(1, 1, 2, dtype=torch.int32),
        sm_scale=1.0,
        d_v=2,
        attn_sink=None,
        topk_length=None,
        output=torch.empty_like(q),
    )


def test_tilelang_sparse_prefill_stage_env_default_and_override(monkeypatch):
    import vllm.envs as env_module

    monkeypatch.delenv("VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES", raising=False)
    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES"
    ]() == 1

    monkeypatch.setenv("VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES", "2")
    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES"
    ]() == 2


def test_tilelang_sparse_prefill_heads_per_block_env_default_and_override(
    monkeypatch,
):
    import vllm.envs as env_module

    monkeypatch.delenv(
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK",
        raising=False,
    )
    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK"
    ]() == 64

    monkeypatch.setenv("VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK", "32")
    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK"
    ]() == 32


def test_tilelang_sparse_prefill_pv_policy_env_default_and_override(
    monkeypatch,
):
    import vllm.envs as env_module

    monkeypatch.delenv("VLLM_SM70_TILELANG_SPARSE_PREFILL_PV_POLICY",
                       raising=False)
    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_PV_POLICY"
    ]() == "full_row"

    monkeypatch.setenv("VLLM_SM70_TILELANG_SPARSE_PREFILL_PV_POLICY",
                       "FULL-COL")
    assert env_module.environment_variables[
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_PV_POLICY"
    ]() == "full_col"


def test_tilelang_sparse_prefill_assume_valid_env_default_and_override(
    monkeypatch,
):
    import vllm.envs as env_module

    monkeypatch.delenv(
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_ASSUME_VALID_INDICES",
        raising=False,
    )
    assert (
        env_module.environment_variables[
            "VLLM_SM70_TILELANG_SPARSE_PREFILL_ASSUME_VALID_INDICES"
        ]()
        is False
    )

    monkeypatch.setenv(
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_ASSUME_VALID_INDICES", "1"
    )
    assert (
        env_module.environment_variables[
            "VLLM_SM70_TILELANG_SPARSE_PREFILL_ASSUME_VALID_INDICES"
        ]()
        is True
    )


def test_sparse_prefill_v2_default_off(monkeypatch):
    monkeypatch.setattr(d4a.envs, "VLLM_SM70_USE_SPARSE_PREFILL_V2", False)
    q = torch.empty(1, 64, 576, dtype=torch.float16)
    out = torch.empty(1, 64, 512, dtype=torch.float16)

    assert not d4a._should_use_sparse_prefill_v2(
        q=q,
        output=out,
        padded_heads=64,
        compress_ratio=128,
        has_attn_metadata=True,
    )


def test_sparse_prefill_v2_retired_even_when_env_enabled(monkeypatch):
    monkeypatch.setattr(d4a.envs, "VLLM_SM70_USE_SPARSE_PREFILL_V2", True)
    q = torch.empty(1, 64, 576, dtype=torch.float16)
    out = torch.empty(1, 64, 512, dtype=torch.float16)

    assert not d4a._should_use_sparse_prefill_v2(
        q=q,
        output=out,
        padded_heads=64,
        compress_ratio=128,
        has_attn_metadata=True,
    )
    assert not d4a._should_use_sparse_prefill_v2(
        q=torch.empty(1, 64, 512, dtype=torch.float16),
        output=out,
        padded_heads=64,
        compress_ratio=4,
        has_attn_metadata=True,
    )
    assert not d4a._should_use_sparse_prefill_v2(
        q=q.to(torch.bfloat16),
        output=out,
        padded_heads=64,
        compress_ratio=128,
        has_attn_metadata=True,
    )
    assert not d4a._should_use_sparse_prefill_v2(
        q=q,
        output=out,
        padded_heads=128,
        compress_ratio=128,
        has_attn_metadata=True,
    )
    assert not d4a._should_use_sparse_prefill_v2(
        q=q,
        output=out,
        padded_heads=64,
        compress_ratio=1,
        has_attn_metadata=True,
    )
    assert not d4a._should_use_sparse_prefill_v2(
        q=q,
        output=out,
        padded_heads=64,
        compress_ratio=128,
        has_attn_metadata=False,
    )


def _make_sparse_prefill_v2_attn():
    attn = object.__new__(DeepseekV4MLAAttention)
    attn.compress_ratio = 4
    attn.scale = 1.0
    attn.attn_sink = torch.zeros(64, dtype=torch.float32)
    attn.window_size = 2
    attn.max_num_batched_tokens = 2
    attn.max_model_len = 16
    attn.head_dim = 512
    attn.padded_heads = 64
    attn.topk_indices_buffer = torch.zeros(2, 2, dtype=torch.int32)
    attn._prefill_graph_dispatcher = None
    attn.prefix = "layers.0.attn"
    return attn


def _make_sparse_prefill_v2_metadata():
    swa_metadata = SimpleNamespace(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        prefill_seq_lens=torch.tensor([8], dtype=torch.int32),
        prefill_gather_lens=torch.tensor([4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        block_table=torch.zeros(1, 2, dtype=torch.int32),
        block_size=4,
    )
    attn_metadata = SimpleNamespace(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        block_size=16,
    )
    return attn_metadata, swa_metadata


def test_forward_prefill_retired_v2_env_uses_v1_workspace(monkeypatch):
    monkeypatch.setattr(d4a.envs, "VLLM_SM70_USE_SPARSE_PREFILL_V2", True)
    monkeypatch.setattr(
        d4a.envs, "VLLM_SM70_SPARSE_PREFILL_V2_DEBUG_COMPARE", False
    )
    attn = _make_sparse_prefill_v2_attn()
    attn_metadata, swa_metadata = _make_sparse_prefill_v2_metadata()
    q = torch.randn(2, 64, 512, dtype=torch.float16)
    output = torch.empty_like(q)
    calls = []
    workspace_kv = torch.empty((1, 12, 512), dtype=torch.bfloat16)

    class FakeWorkspaceManager:
        def get_simultaneous(self, _spec):
            calls.append("workspace")
            return (workspace_kv,)

    def fake_gather(out, *_args, **_kwargs):
        calls.append("gather")
        out.fill_(1)

    def fake_combine(*_args):
        calls.append("combine")
        return (
            torch.zeros(2, 128, dtype=torch.int32),
            torch.full((2,), 4, dtype=torch.int32),
        )

    def fake_flash_mla_sparse_fwd(**kwargs):
        calls.append("v1")
        kwargs["out"].fill_(7)
        return kwargs["out"], None, None

    monkeypatch.setattr(
        d4a, "current_workspace_manager", lambda: FakeWorkspaceManager()
    )
    monkeypatch.setattr(d4a, "dequantize_and_gather_k_cache", fake_gather)
    monkeypatch.setattr(d4a, "combine_topk_swa_indices", fake_combine)
    monkeypatch.setattr(d4a, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)
    monkeypatch.setattr(
        d4a, "_should_use_tilelang_sparse_prefill_fast_io", lambda **_kwargs: False
    )

    attn._forward_prefill(
        q=q,
        positions=torch.arange(2, dtype=torch.int64),
        compressed_k_cache=torch.zeros(1, 4, 584, dtype=torch.uint8),
        swa_k_cache=torch.zeros(2, 4, 584, dtype=torch.uint8),
        output=output,
        attn_metadata=attn_metadata,
        swa_metadata=swa_metadata,
    )

    assert calls.count("workspace") == 1
    assert calls.count("v1") == 1
    assert calls.count("gather") == 2
    assert "combine" in calls
    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_prefill_graph_dispatcher_cache_reuses_shape_and_device(monkeypatch):
    d4a._PREFILL_GRAPH_DISPATCHER_CACHE.clear()
    created = []

    class FakeDispatcher:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def capture_graphs(self, *_args):
            pass

    monkeypatch.setattr(d4a, "PrefillGraphDispatcher", FakeDispatcher)
    attn_sink = torch.zeros(64, dtype=torch.float32)
    capture_sizes = [(1024, 3200)]

    first = d4a._get_or_create_prefill_graph_dispatcher(
        capture_sizes=capture_sizes,
        padded_heads=64,
        head_dim=512,
        scale=1.0,
        attn_sink=attn_sink,
        device=torch.device("cpu"),
        flash_mla_sparse_fwd_fn=lambda **_: None,
        flashmla_bf16_io_fn=lambda q, out: (q, out),
        copy_flashmla_output_fn=lambda flash_out, out: None,
    )
    second = d4a._get_or_create_prefill_graph_dispatcher(
        capture_sizes=list(capture_sizes),
        padded_heads=64,
        head_dim=512,
        scale=1.0,
        attn_sink=attn_sink + 1,
        device=torch.device("cpu"),
        flash_mla_sparse_fwd_fn=lambda **_: None,
        flashmla_bf16_io_fn=lambda q, out: (q, out),
        copy_flashmla_output_fn=lambda flash_out, out: None,
    )

    assert first is second
    assert len(created) == 1


def test_deepseek_v4_phase_profiler_accumulates_and_resets():
    profiler = d4a._DEEPSEEK_V4_PROFILE
    profiler.reset()

    profiler.record("prefill.swa_gather", 100.0)
    profiler.record("prefill.swa_gather", 50.0)
    profiler.record("decode.attn", 25.0)

    snapshot = profiler.snapshot(reset=True)

    assert snapshot["prefill.swa_gather"]["count"] == 2
    assert snapshot["prefill.swa_gather"]["total_us"] == 150.0
    assert snapshot["prefill.swa_gather"]["avg_us"] == 75.0
    assert snapshot["decode.attn"]["count"] == 1
    assert profiler.snapshot() == {}


def test_deepseek_v4_phase_profiler_exports_metric_sink(monkeypatch):
    events = []

    class FakeMetrics:
        def record(self, label, elapsed_us):
            events.append((label, elapsed_us))

    monkeypatch.setattr(
        d4a, "_DEEPSEEK_V4_PROFILE_PROMETHEUS", FakeMetrics(), raising=False
    )
    profiler = d4a._DeepseekV4PhaseProfiler()

    profiler.record("prefill.flashmla_sparse_fwd", 12.5)

    assert events == [("prefill.flashmla_sparse_fwd", 12.5)]


def test_deepseek_v4_phase_profiler_exports_raw_trace(monkeypatch, tmp_path):
    trace_path = tmp_path / "phase.jsonl"
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH", str(trace_path))
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_RAW_TRACE", None, raising=False)

    profiler = d4a._DeepseekV4PhaseProfiler()
    profiler.record("decode.attn.direct_flashmla", 42.0)

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    # Required core fields must match; additional context fields (step_id,
    # layer_idx, compress_ratio, step_kind, step_token_count) are optional
    # and default to None when not recorded inside a _profile_step /
    # _profile_layer context.
    assert row["phase"] == "decode.attn.direct_flashmla"
    assert row["elapsed_us"] == 42.0
    assert "pid" in row
    assert "cuda_device" in row


def test_deepseek_v4_queue_profile_flushes_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_ENABLED", True)
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_MODE", "queue")
    monkeypatch.setattr(d4a.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(d4a, "_is_cuda_stream_capturing", lambda: False)
    monkeypatch.setattr(d4a, "_flush_event_queue", lambda: calls.append("flush"))

    d4a._flush_profile_queue_if_needed()

    assert calls == ["flush"]


def test_deepseek_v4_queue_profile_flush_skips_non_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_ENABLED", True)
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_MODE", "eager")
    monkeypatch.setattr(d4a.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(d4a, "_is_cuda_stream_capturing", lambda: False)
    monkeypatch.setattr(d4a, "_flush_event_queue", lambda: calls.append("flush"))

    d4a._flush_profile_queue_if_needed()

    assert calls == []


def test_deepseek_v4_queue_profile_flush_skips_dynamo_compile(monkeypatch):
    calls = []
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_ENABLED", True)
    monkeypatch.setattr(d4a, "_DEEPSEEK_V4_PROFILE_MODE", "queue")
    monkeypatch.setattr(d4a.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(d4a.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        d4a,
        "_is_cuda_stream_capturing",
        lambda: (_ for _ in ()).throw(AssertionError("should not be traced")),
    )
    monkeypatch.setattr(d4a, "_flush_event_queue", lambda: calls.append("flush"))

    d4a._flush_profile_queue_if_needed()

    assert calls == []


def test_deepseek_v4_indexer_forward_profiles_internal_phases(monkeypatch):
    labels = []

    @contextmanager
    def fake_profile(label, ref, *, extra=None):
        labels.append(label)
        yield

    class FakeLinear:
        def __init__(self, output):
            self.output = output
            self.weight = SimpleNamespace(dtype=output.dtype)

        def __call__(self, _x):
            return self.output, None

    class FakeCompressor:
        def __call__(self, _hidden_states, _positions, _rotary_emb):
            return torch.full((2, 4, 8), 3, dtype=torch.uint8)

    class FakeIndexerOp:
        def __call__(self, _hidden_states, q_quant, _k, weights):
            return q_quant.to(torch.float32).sum() + weights.sum()

    monkeypatch.setattr(d4a, "_profile_or_null", fake_profile)
    monkeypatch.setattr(
        d4a,
        "fused_indexer_q_rope_quant",
        lambda positions, q, cos_sin_cache, weights, *_, **__: (q.to(torch.uint8), weights),
    )

    fake = SimpleNamespace(
        weights_proj=FakeLinear(torch.ones((2, 4), dtype=torch.float16)),
        wq_b=FakeLinear(torch.ones((2, 4 * 8), dtype=torch.float16)),
        n_head=4,
        head_dim=8,
        compressor=FakeCompressor(),
        softmax_scale=1.0,
        use_fp4_kv=False,
        indexer_op=FakeIndexerOp(),
    )
    rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1, dtype=torch.float16))

    d4a.DeepseekV4Indexer.forward(
        fake,
        hidden_states=torch.ones((2, 16), dtype=torch.float16),
        qr=torch.ones((2, 4), dtype=torch.float16),
        positions=torch.arange(2, dtype=torch.int64),
        rotary_emb=rotary_emb,
    )

    assert labels == [
        "indexer.wq_b",
        "indexer.compressor",
        "indexer.weights_proj",
        "indexer.q_rope_quant",
        "indexer.indexer_op",
    ]


def test_deepseek_v4_copy_source_trace_emits_nvtx_during_graph_capture(
    monkeypatch,
):
    from vllm.model_executor.layers import deepseek_v4_copy_source_trace as trace

    calls = []
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_COPY_SOURCE_TRACE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trace, "_torch_compiler_is_compiling", lambda: False)
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda label: calls.append(("push", label)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_pop",
        lambda: calls.append(("pop", None)),
    )

    with trace.copy_source_trace("o_einsum.fp8_a_dequant"):
        calls.append(("body", None))

    assert calls == [
        ("push", "copy_source.o_einsum.fp8_a_dequant"),
        ("body", None),
        ("pop", None),
    ]


def test_deepseek_v4_copy_source_trace_is_default_off(monkeypatch):
    from vllm.model_executor.layers import deepseek_v4_copy_source_trace as trace

    calls = []
    monkeypatch.delenv("VLLM_DEEPSEEK_V4_COPY_SOURCE_TRACE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trace, "_torch_compiler_is_compiling", lambda: False)
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda label: calls.append(("push", label)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_pop",
        lambda: calls.append(("pop", None)),
    )

    with trace.copy_source_trace("mhc_post.comb_contiguous"):
        calls.append(("body", None))

    assert calls == [("body", None)]


def test_deepseek_v4_copy_source_trace_skips_nvtx_while_torch_compiling(
    monkeypatch,
):
    from vllm.model_executor.layers import deepseek_v4_copy_source_trace as trace

    calls = []
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_COPY_SOURCE_TRACE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trace, "_torch_compiler_is_compiling", lambda: True)
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda label: calls.append(("push", label)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_pop",
        lambda: calls.append(("pop", None)),
    )

    with trace.copy_source_trace("attn.boundary_hidden_states"):
        calls.append(("body", None))

    assert calls == [("body", None)]
