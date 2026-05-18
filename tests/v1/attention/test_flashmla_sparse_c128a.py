# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.flashmla_sparse import (
    _c128a_prefill_max_compressed_tokens,
    build_c128a_topk_metadata,
    c128a_prefill_compressed_topk_buckets,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_c128a_prefill_metadata_width_tracks_request_not_max_model_len():
    positions = torch.tensor([0, 127, 128, 9999], dtype=torch.int64, device="cuda")
    compress_ratio = 128
    max_compressed_tokens = 4096
    num_decode_tokens = 0
    num_tokens = positions.numel()

    req_id_per_token = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    global_decode_buffer = torch.empty(
        (0, max_compressed_tokens), dtype=torch.int32, device="cuda"
    )
    decode_lens_buffer = torch.empty(0, dtype=torch.int32, device="cuda")
    prefill_buffer = torch.empty(
        (num_tokens, max_compressed_tokens), dtype=torch.int32, device="cuda"
    )

    _global_decode, _decode_lens, prefill_local = build_c128a_topk_metadata(
        positions,
        compress_ratio,
        num_decode_tokens,
        req_id_per_token,
        block_table,
        block_size=1,
        slot_mapping=slot_mapping,
        global_decode_buffer=global_decode_buffer,
        decode_lens_buffer=decode_lens_buffer,
        prefill_buffer=prefill_buffer,
        max_compressed_tokens=max_compressed_tokens,
    )

    assert prefill_local.shape == (num_tokens, 128)
    assert prefill_local[-1, 77].item() == 77
    assert prefill_local[-1, 78].item() == -1


def test_c128a_prefill_width_uses_deterministic_buckets():
    assert c128a_prefill_compressed_topk_buckets(4096) == [
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
    ]

    positions = torch.tensor([0, 128 * 1100 - 1], dtype=torch.int64)
    assert _c128a_prefill_max_compressed_tokens(
        positions,
        compress_ratio=128,
        num_decode_tokens=0,
        max_compressed_tokens=4096,
    ) == 1536
