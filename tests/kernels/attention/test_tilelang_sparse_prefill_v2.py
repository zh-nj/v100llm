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

    with pytest.raises(ValueError, match=r"shape \[tokens, 64, 576\]"):
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
