# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def _base_inputs():
    from vllm.v1.attention.ops.tilelang_sparse_prefill_v2 import (
        flash_mla_sparse_prefill_v2,
    )

    q = torch.empty(1, 64, 576, dtype=torch.float16)
    out = torch.empty(1, 64, 512, dtype=torch.float16)
    cache = torch.empty(1, 64, 584, dtype=torch.uint8)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    return flash_mla_sparse_prefill_v2, {
        "q": q,
        "compressed_k_cache": cache,
        "swa_k_cache": cache,
        "compressed_block_table": block_table,
        "swa_block_table": block_table,
        "topk_indices": torch.zeros(1, 128, dtype=torch.int32),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "seq_lens": torch.tensor([1], dtype=torch.int32),
        "gather_lens": torch.tensor([1], dtype=torch.int32),
        "window_size": 128,
        "compress_ratio": 128,
        "top_k": 128,
        "sm_scale": 1.0,
        "attn_sink": torch.zeros(64, dtype=torch.float32),
        "out": out,
    }


def test_sparse_prefill_v2_rejects_unsupported_dtype():
    flash_mla_sparse_prefill_v2, kwargs = _base_inputs()
    kwargs["q"] = kwargs["q"].to(torch.bfloat16)

    with pytest.raises(ValueError, match="fp16 q/out"):
        flash_mla_sparse_prefill_v2(**kwargs)


def test_sparse_prefill_v2_rejects_unsupported_head_count():
    flash_mla_sparse_prefill_v2, kwargs = _base_inputs()
    kwargs["q"] = torch.empty(1, 32, 576, dtype=torch.float16)

    with pytest.raises(ValueError, match=r"shape \[tokens, 64, 512 or 576\]"):
        flash_mla_sparse_prefill_v2(**kwargs)


def test_sparse_prefill_v2_rejects_unsupported_compress_ratio():
    flash_mla_sparse_prefill_v2, kwargs = _base_inputs()
    kwargs["compress_ratio"] = 1

    with pytest.raises(ValueError, match="compress_ratio"):
        flash_mla_sparse_prefill_v2(**kwargs)


def test_sparse_prefill_v2_rejects_non_int32_topk_indices():
    flash_mla_sparse_prefill_v2, kwargs = _base_inputs()
    kwargs["topk_indices"] = kwargs["topk_indices"].to(torch.int64)

    with pytest.raises(ValueError, match="topk_indices"):
        flash_mla_sparse_prefill_v2(**kwargs)


def test_sparse_prefill_v2_oracle_uses_existing_gather_tilelang_path(
    monkeypatch,
):
    import vllm.v1.attention.ops.tilelang_sparse_prefill_v2 as v2

    q = torch.empty(2, 64, 576, dtype=torch.float16)
    out = torch.empty(2, 64, 512, dtype=torch.float16)
    cache = torch.empty(3, 64, 584, dtype=torch.uint8)
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    calls = []

    def fake_gather(out_tensor, k_cache, **kwargs):
        calls.append(("gather", out_tensor.shape, k_cache.shape, kwargs))
        out_tensor.fill_(1)

    def fake_combine(*args):
        calls.append(("combine", args[-2], args[-1]))
        return (
            torch.zeros(2, 1, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        )

    def fake_tilelang(**kwargs):
        calls.append((
            "tilelang",
            kwargs["q"].shape,
            kwargs["kv"].shape,
            kwargs["indices"].shape,
            kwargs["topk_length"].shape,
        ))
        return (
            torch.full_like(out, 3, dtype=torch.bfloat16),
            torch.zeros(2, 64, dtype=torch.float32),
            torch.zeros(2, 64, dtype=torch.float32),
        )

    monkeypatch.setattr(v2, "dequantize_and_gather_k_cache", fake_gather,
                        raising=False)
    monkeypatch.setattr(v2, "combine_topk_swa_indices", fake_combine,
                        raising=False)
    monkeypatch.setattr(
        v2.tilelang_sparse_prefill,
        "flash_mla_sparse_fwd_tilelang",
        fake_tilelang,
        raising=False,
    )

    result, max_logits, lse = v2._flash_mla_sparse_prefill_v2_oracle(
        q=q,
        compressed_k_cache=cache,
        swa_k_cache=cache,
        compressed_block_table=block_table,
        swa_block_table=block_table,
        topk_indices=torch.zeros(2, 128, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([256], dtype=torch.int32),
        gather_lens=torch.tensor([2], dtype=torch.int32),
        window_size=128,
        compress_ratio=128,
        top_k=128,
        sm_scale=1.0,
        attn_sink=torch.zeros(64, dtype=torch.float32),
        out=out,
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, torch.full_like(out, 3))
    assert max_logits.shape == (2, 64)
    assert lse.shape == (2, 64)
    assert [call[0] for call in calls] == [
        "gather",
        "gather",
        "combine",
        "tilelang",
    ]
    assert calls[0][3]["offset"] == 0
    assert calls[1][3]["offset"] == 2


def test_sparse_prefill_v2_wrapper_uses_direct_cache_for_512_dim_q(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import vllm.v1.attention.ops.tilelang_sparse_prefill_v2 as v2

    q = torch.empty(2, 64, 512, dtype=torch.float16, device="cuda")
    out = torch.empty(2, 64, 512, dtype=torch.float16, device="cuda")
    cache = torch.empty(2, 4, 584, dtype=torch.uint8, device="cuda")
    block_table = torch.zeros(1, 2, dtype=torch.int32, device="cuda")
    calls = []

    def fake_direct(**kwargs):
        calls.append(kwargs)
        return (
            kwargs["out"],
            torch.zeros(2, 64, dtype=torch.float32, device="cuda"),
            torch.zeros(2, 64, dtype=torch.float32, device="cuda"),
        )

    monkeypatch.setattr(
        v2,
        "_flash_mla_sparse_prefill_v2_direct_cache",
        fake_direct,
    )

    result, max_logits, lse = v2.flash_mla_sparse_prefill_v2(
        q=q,
        compressed_k_cache=cache,
        swa_k_cache=cache,
        compressed_block_table=block_table,
        swa_block_table=block_table,
        topk_indices=torch.zeros(2, 2, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([6], dtype=torch.int32, device="cuda"),
        gather_lens=torch.tensor([4], dtype=torch.int32, device="cuda"),
        window_size=2,
        compress_ratio=4,
        top_k=2,
        sm_scale=1.0,
        attn_sink=torch.zeros(64, dtype=torch.float32, device="cuda"),
        out=out,
    )

    assert calls and calls[0]["q"].data_ptr() == q.data_ptr()
    assert result.data_ptr() == out.data_ptr()
    assert max_logits.shape == (2, 64)
    assert lse.shape == (2, 64)


def test_sparse_prefill_v2_reference_row_map_matches_direct_cache_layout():
    from vllm.v1.attention.ops.tilelang_sparse_prefill_v2 import (
        _reference_direct_cache_row_map,
    )

    row_map = _reference_direct_cache_row_map(
        topk_indices=torch.tensor(
            [
                [0, 1, 5, 99],
                [1, 0, 3, 4],
                [3, 4, 2, 1],
                [0, 1, 2, 3],
                [6, 5, 4, 3],
            ],
            dtype=torch.int32,
        ),
        query_start_loc=torch.tensor([10, 12, 15], dtype=torch.int32),
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        gather_lens=torch.tensor([6, 8], dtype=torch.int32),
        compressed_block_table=torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
        ),
        swa_block_table=torch.tensor(
            [[30, 31, 32, 33], [40, 41, 42, 43]], dtype=torch.int32
        ),
        compressed_block_size=3,
        swa_block_size=5,
        window_size=4,
        compress_ratio=4,
        top_k=4,
    )

    # 0 = invalid/padding, 1 = compressed cache, 2 = SWA cache.
    assert row_map.source[0].tolist() == [1, 1, 2, 2, 2, 2, 0, 0]
    assert row_map.physical_block[0].tolist() == [
        10,
        10,
        31,
        31,
        31,
        31,
        -1,
        -1,
    ]
    assert row_map.block_offset[0].tolist() == [0, 1, 0, 1, 2, 3, -1, -1]
    assert row_map.logical_position[0].tolist() == [0, 1, 5, 6, 7, 8, -1, -1]
    assert row_map.length.tolist() == [6, 6, 8, 8, 8]

    assert row_map.source[2].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert row_map.physical_block[2].tolist() == [21, 21, 20, 20, 42, 43, 43, 43]
    assert row_map.block_offset[2].tolist() == [0, 1, 2, 1, 4, 0, 1, 2]
    assert row_map.logical_position[2].tolist() == [3, 4, 2, 1, 14, 15, 16, 17]


def test_sparse_prefill_v2_reference_load_fp8_ds_mla_token_matches_layout():
    from vllm.v1.attention.ops.tilelang_sparse_prefill_v2 import (
        _reference_load_fp8_ds_mla_token,
    )

    block_size = 4
    physical_block = 1
    block_offset = 2
    cache = torch.zeros(2, block_size, 584, dtype=torch.uint8)
    cache_2d = cache.reshape(cache.shape[0], -1)

    fp8_values = torch.linspace(
        -2.0, 2.0, 448, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    fp8_bytes = fp8_values.view(torch.uint8)
    scales = torch.tensor([127, 128, 126, 127, 129, 125, 127], dtype=torch.uint8)
    rope_tail = (
        torch.arange(64, dtype=torch.float32).to(torch.bfloat16) + 100
    )

    token_data_offset = block_offset * 576
    token_scale_offset = block_size * 576 + block_offset * 8
    cache_2d[
        physical_block, token_data_offset : token_data_offset + 448
    ] = fp8_bytes
    cache_2d[
        physical_block, token_data_offset + 448 : token_data_offset + 576
    ] = rope_tail.view(torch.uint8)
    cache_2d[
        physical_block, token_scale_offset : token_scale_offset + 7
    ] = scales

    token = _reference_load_fp8_ds_mla_token(
        cache,
        physical_block=physical_block,
        block_offset=block_offset,
        block_size=block_size,
        output_dtype=torch.float16,
    )

    expected_scales = torch.exp2(scales.to(torch.float32) - 127.0)
    expected_nope = (
        fp8_values.to(torch.float32)
        * expected_scales.repeat_interleave(64)
    )
    torch.testing.assert_close(token[:448], expected_nope.to(torch.float16))
    torch.testing.assert_close(token[448:], rope_tail.to(torch.float16))


def test_sparse_prefill_v2_tilelang_load_fp8_ds_mla_token_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import vllm.v1.attention.ops.tilelang_sparse_prefill_v2 as v2

    ok, reason = v2.tilelang_sparse_prefill.is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    block_size = 4
    physical_block = 1
    block_offset = 2
    cache = torch.zeros(2, block_size, 584, dtype=torch.uint8)
    cache_2d = cache.reshape(cache.shape[0], -1)

    fp8_values = torch.linspace(
        -2.0, 2.0, 448, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([127, 128, 126, 127, 129, 125, 127], dtype=torch.uint8)
    rope_tail = (
        torch.arange(64, dtype=torch.float32).to(torch.bfloat16) + 100
    )
    token_data_offset = block_offset * 576
    token_scale_offset = block_size * 576 + block_offset * 8
    cache_2d[
        physical_block, token_data_offset : token_data_offset + 448
    ] = fp8_values.view(torch.uint8)
    cache_2d[
        physical_block, token_data_offset + 448 : token_data_offset + 576
    ] = rope_tail.view(torch.uint8)
    cache_2d[
        physical_block, token_scale_offset : token_scale_offset + 7
    ] = scales

    expected = v2._reference_load_fp8_ds_mla_token(
        cache,
        physical_block=physical_block,
        block_offset=block_offset,
        block_size=block_size,
        output_dtype=torch.float16,
    )
    actual = v2._tilelang_debug_load_fp8_ds_mla_tokens(
        cache.cuda(),
        physical_blocks=torch.tensor(
            [physical_block], dtype=torch.int32, device="cuda"
        ),
        block_offsets=torch.tensor(
            [block_offset], dtype=torch.int32, device="cuda"
        ),
        block_size=block_size,
        output_dtype=torch.float16,
    )

    torch.testing.assert_close(actual.cpu()[0], expected, rtol=0, atol=0)


def test_sparse_prefill_v2_direct_cache_attention_matches_gather_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import vllm.v1.attention.ops.tilelang_sparse_prefill as tilelang_prefill
    import vllm.v1.attention.ops.tilelang_sparse_prefill_v2 as v2
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        combine_topk_swa_indices,
        dequantize_and_gather_k_cache,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
        _torch_quantize_and_insert_k_cache,
    )

    ok, reason = tilelang_prefill.is_tilelang_available()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    block_size = 4
    compress_ratio = 2
    top_k = 2
    window_size = 2
    seq_lens = torch.tensor([6], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([4], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=device)
    compressed_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    swa_block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    topk_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device=device)

    compressed_rows = torch.linspace(
        -0.5, 0.5, 3 * 512, dtype=torch.float32
    ).reshape(3, 512).to(torch.bfloat16)
    swa_rows = torch.linspace(
        0.25, -0.25, 6 * 512, dtype=torch.float32
    ).reshape(6, 512).to(torch.bfloat16)
    compressed_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8)
    swa_cache = torch.zeros(2, block_size, 584, dtype=torch.uint8)
    _torch_quantize_and_insert_k_cache(
        compressed_rows,
        compressed_cache,
        torch.arange(3, dtype=torch.int64),
        block_size,
    )
    _torch_quantize_and_insert_k_cache(
        swa_rows,
        swa_cache,
        torch.arange(6, dtype=torch.int64),
        block_size,
    )
    compressed_cache = compressed_cache.to(device)
    swa_cache = swa_cache.to(device)

    q = torch.linspace(
        -0.125, 0.125, 2 * 64 * 512, dtype=torch.float32, device=device
    ).reshape(2, 64, 512).to(torch.float16)
    out_direct = torch.empty(2, 64, 512, dtype=torch.float16, device=device)
    attn_sink = torch.full((64,), -float("inf"), dtype=torch.float32, device=device)
    sm_scale = 0.25

    direct, _direct_max, _direct_lse = v2._flash_mla_sparse_prefill_v2_direct_cache(
        q=q,
        compressed_k_cache=compressed_cache,
        swa_k_cache=swa_cache,
        compressed_block_table=compressed_block_table,
        swa_block_table=swa_block_table,
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=window_size,
        compress_ratio=compress_ratio,
        top_k=top_k,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        out=out_direct,
        block_I=16,
    )

    N = int(seq_lens.max().item()) // compress_ratio
    M = N + int(gather_lens.max().item())
    kv = torch.zeros((1, M, 512), dtype=torch.bfloat16, device=device)
    dequantize_and_gather_k_cache(
        kv,
        compressed_cache,
        seq_lens=seq_lens // compress_ratio,
        gather_lens=None,
        block_table=compressed_block_table,
        block_size=block_size,
        offset=0,
    )
    dequantize_and_gather_k_cache(
        kv,
        swa_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=swa_block_table,
        block_size=block_size,
        offset=N,
    )
    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        top_k,
        M,
        N,
    )
    expected, _expected_max, _expected_lse = (
        tilelang_prefill.flash_mla_sparse_fwd_tilelang(
            q=q,
            kv=kv.view(-1, 1, 512),
            indices=combined_indices.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=512,
            attn_sink=attn_sink,
            topk_length=combined_lens,
            output_dtype=torch.float16,
            block_I=16,
        )
    )

    assert direct.data_ptr() == out_direct.data_ptr()
    torch.testing.assert_close(direct, expected, atol=2e-2, rtol=2e-2)


def test_sparse_prefill_v2_triton_selected_gather_matches_scaffold():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import vllm.v1.attention.ops.tilelang_sparse_prefill_v2 as v2
    from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
        _torch_quantize_and_insert_k_cache,
    )

    device = torch.device("cuda")
    block_size = 4
    compress_ratio = 2
    top_k = 2
    window_size = 2
    total_topk = 16
    seq_lens = torch.tensor([6], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([4], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=device)
    compressed_block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    swa_block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    topk_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device=device)

    compressed_rows = torch.linspace(
        -0.5, 0.5, 3 * 512, dtype=torch.float32
    ).reshape(3, 512).to(torch.bfloat16)
    swa_rows = torch.linspace(
        0.25, -0.25, 6 * 512, dtype=torch.float32
    ).reshape(6, 512).to(torch.bfloat16)
    compressed_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8)
    swa_cache = torch.zeros(2, block_size, 584, dtype=torch.uint8)
    _torch_quantize_and_insert_k_cache(
        compressed_rows,
        compressed_cache,
        torch.arange(3, dtype=torch.int64),
        block_size,
    )
    _torch_quantize_and_insert_k_cache(
        swa_rows,
        swa_cache,
        torch.arange(6, dtype=torch.int64),
        block_size,
    )
    compressed_cache = compressed_cache.to(device)
    swa_cache = swa_cache.to(device)

    expected_kv, expected_indices, expected_lens = (
        v2._tilelang_gather_selected_fp8_ds_mla_cache(
            compressed_k_cache=compressed_cache,
            swa_k_cache=swa_cache,
            compressed_block_table=compressed_block_table,
            swa_block_table=swa_block_table,
            topk_indices=topk_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            window_size=window_size,
            compress_ratio=compress_ratio,
            top_k=top_k,
            total_topk=total_topk,
            dim=512,
            output_dtype=torch.float16,
        )
    )
    actual_kv, actual_indices, actual_lens = (
        v2._triton_gather_selected_fp8_ds_mla_cache(
            compressed_k_cache=compressed_cache,
            swa_k_cache=swa_cache,
            compressed_block_table=compressed_block_table,
            swa_block_table=swa_block_table,
            topk_indices=topk_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            window_size=window_size,
            compress_ratio=compress_ratio,
            top_k=top_k,
            total_topk=total_topk,
            dim=512,
            output_dtype=torch.float16,
        )
    )

    torch.testing.assert_close(actual_kv, expected_kv, rtol=0, atol=0)
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_lens, expected_lens, rtol=0, atol=0)


def test_sparse_prefill_v2_triton_selected_gather_supports_row_slices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import vllm.v1.attention.ops.tilelang_sparse_prefill_v2 as v2
    from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
        _torch_quantize_and_insert_k_cache,
    )

    device = torch.device("cuda")
    block_size = 4
    compress_ratio = 2
    top_k = 2
    window_size = 2
    total_topk = 16
    num_tokens = 4
    seq_lens = torch.tensor([8], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([6], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        [0, num_tokens], dtype=torch.int32, device=device
    )
    compressed_block_table = torch.tensor(
        [[0]], dtype=torch.int32, device=device
    )
    swa_block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    topk_indices = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.int32,
        device=device,
    )

    compressed_rows = torch.linspace(
        -0.5, 0.5, 4 * 512, dtype=torch.float32
    ).reshape(4, 512).to(torch.bfloat16)
    swa_rows = torch.linspace(
        0.25, -0.25, 8 * 512, dtype=torch.float32
    ).reshape(8, 512).to(torch.bfloat16)
    compressed_cache = torch.zeros(1, block_size, 584, dtype=torch.uint8)
    swa_cache = torch.zeros(2, block_size, 584, dtype=torch.uint8)
    _torch_quantize_and_insert_k_cache(
        compressed_rows,
        compressed_cache,
        torch.arange(4, dtype=torch.int64),
        block_size,
    )
    _torch_quantize_and_insert_k_cache(
        swa_rows,
        swa_cache,
        torch.arange(8, dtype=torch.int64),
        block_size,
    )
    compressed_cache = compressed_cache.to(device)
    swa_cache = swa_cache.to(device)

    full_kv, full_indices, full_lens = (
        v2._triton_gather_selected_fp8_ds_mla_cache(
            compressed_k_cache=compressed_cache,
            swa_k_cache=swa_cache,
            compressed_block_table=compressed_block_table,
            swa_block_table=swa_block_table,
            topk_indices=topk_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            window_size=window_size,
            compress_ratio=compress_ratio,
            top_k=top_k,
            total_topk=total_topk,
            dim=512,
            output_dtype=torch.float16,
        )
    )

    chunks = []
    chunk_indices = []
    chunk_lens = []
    for row_start, row_end in ((0, 2), (2, 4)):
        sliced_kv, sliced_indices, sliced_lens = (
            v2._triton_gather_selected_fp8_ds_mla_cache(
                compressed_k_cache=compressed_cache,
                swa_k_cache=swa_cache,
                compressed_block_table=compressed_block_table,
                swa_block_table=swa_block_table,
                topk_indices=topk_indices[row_start:row_end],
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                window_size=window_size,
                compress_ratio=compress_ratio,
                top_k=top_k,
                total_topk=total_topk,
                dim=512,
                output_dtype=torch.float16,
                total_query_tokens=num_tokens,
                query_token_offset=row_start,
            )
        )
        chunks.append(sliced_kv)
        chunk_indices.append(sliced_indices)
        chunk_lens.append(sliced_lens)

    torch.testing.assert_close(
        torch.cat(chunks, dim=0), full_kv, rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.cat(chunk_lens, dim=0), full_lens, rtol=0, atol=0
    )
    assert chunk_indices[0].max().item() < 2 * total_topk
    assert chunk_indices[1].max().item() < 2 * total_topk
