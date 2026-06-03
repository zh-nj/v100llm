# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.model_executor.layers.deepseek_compressor import (
    _normalize_sm70_fp8_cache_exponents,
    _torch_fused_compress_norm_rope_insert_fp8_fallback,
)
from vllm.triton_utils import triton
from vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)

FP8_MAX = 448.0


def test_sm70_compressor_fp8_cache_exponents_preserve_positive_scales() -> None:
    exponents = torch.tensor([4.0, 1.0, 0.0, -3.0], dtype=torch.float32)

    normalized = _normalize_sm70_fp8_cache_exponents(exponents)

    torch.testing.assert_close(normalized, exponents)


def _boundary_reference(
    state_cache: torch.Tensor,
    token_idx: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
) -> torch.Tensor:
    position = int(positions[token_idx].item())
    req_idx = int(token_to_req_indices[token_idx].item())
    window = (1 + int(overlap)) * compress_ratio
    start = position - window + 1

    kv_rows = []
    score_rows = []
    for local_idx in range(window):
        pos = start + local_idx
        if pos < 0:
            kv_rows.append(torch.zeros(head_size, device=state_cache.device))
            score_rows.append(torch.full(
                (head_size, ), float("-inf"), device=state_cache.device))
            continue

        block_number = int(block_table[req_idx, pos // block_size].item())
        block_offset = pos % block_size
        head_offset = int(local_idx >= compress_ratio) * head_size
        row = state_cache[block_number, block_offset]
        kv_rows.append(row[head_offset:head_offset + head_size].float())
        score_rows.append(row[state_width + head_offset:
                              state_width + head_offset + head_size].float())

    score = torch.stack(score_rows).softmax(dim=0)
    kv = torch.stack(kv_rows)
    compressed = (kv * score).sum(dim=0)
    variance = compressed.pow(2).sum() / head_size
    normed = compressed * torch.rsqrt(variance + rms_norm_eps) * norm_weight

    nope_head_dim = head_size - rope_head_dim
    compressed_pos = (position // compress_ratio) * compress_ratio
    cos_sin = cos_sin_cache[compressed_pos].float()
    half = rope_head_dim // 2
    rope = normed[nope_head_dim:].clone()
    even = rope[::2]
    odd = rope[1::2]
    rotated = torch.empty_like(rope)
    rotated[::2] = even * cos_sin[:half] - odd * cos_sin[half:]
    rotated[1::2] = odd * cos_sin[:half] + even * cos_sin[half:]

    result = normed.clone()
    result[nope_head_dim:] = rotated
    return result


def _decode_indexer_cache(
    kv_cache: torch.Tensor,
    slot: int,
    block_size: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_idx = slot // block_size
    pos_in_block = slot % block_size
    cache_2d = kv_cache.reshape(kv_cache.shape[0], -1)
    fp8_offset = pos_in_block * head_size
    scale_offset = block_size * head_size + pos_in_block * 4
    fp8 = cache_2d[block_idx, fp8_offset:fp8_offset + head_size]
    scale = cache_2d[block_idx, scale_offset:scale_offset + 4]
    return fp8.view(torch.float8_e4m3fn).float(), scale.view(torch.float32)


def _decode_sparse_attention_cache(
    kv_cache: torch.Tensor,
    slot: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_idx = slot // block_size
    pos_in_block = slot % block_size
    cache_2d = kv_cache.reshape(kv_cache.shape[0], -1)
    token_stride = 576
    scale_dim = 8
    fp8_offset = pos_in_block * token_stride
    scale_offset = block_size * token_stride + pos_in_block * scale_dim
    fp8 = cache_2d[block_idx, fp8_offset:fp8_offset + 448]
    rope = cache_2d[block_idx, fp8_offset + 448:fp8_offset + 576]
    scales = cache_2d[block_idx, scale_offset:scale_offset + 7].to(torch.float32)
    return (
        fp8.view(torch.float8_e4m3fn).float(),
        rope.view(torch.bfloat16).float(),
        torch.exp2(scales - 127.0),
    )


def _expected_sparse_attention_cache_bytes(
    expected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_nope = expected[:448].to(torch.bfloat16).float()
    expected_rope = (
        expected[448:].to(torch.bfloat16).contiguous().view(torch.uint8).view(-1)
    )
    blocks = expected_nope.view(7, 64)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    exponents = torch.ceil(torch.log2(absmax / FP8_MAX))
    scales = torch.exp2(exponents)
    fp8_bytes = (
        (blocks / scales)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
        .view(-1)
    )
    scale_bytes = (exponents.flatten() + 127.0).clamp(0, 255).to(torch.uint8)
    return fp8_bytes, expected_rope, scale_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm70_compressor_fp8_indexer_fallback_writes_quantized_cache() -> None:
    device = "cuda"
    torch.manual_seed(0)
    head_size = 128
    rope_head_dim = 64
    compress_ratio = 4
    block_size = 4
    state_width = 2 * head_size
    num_tokens = 8
    num_blocks = 3
    token_stride = head_size

    state_cache = torch.randn(
        num_blocks,
        block_size,
        2 * state_width,
        dtype=torch.float32,
        device=device,
    ) * 2048.0
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32,
                               device=device).unsqueeze(0)
    norm_weight = torch.randn(head_size, dtype=torch.float32, device=device)
    cos_sin_cache = torch.randn(32, rope_head_dim, dtype=torch.float32,
                                device=device)
    kv_cache = torch.full(
        (num_blocks, block_size, head_size + 4),
        0x7B,
        dtype=torch.uint8,
        device=device,
    )

    _torch_fused_compress_norm_rope_insert_fp8_fallback(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        norm_weight,
        1e-6,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        block_size,
        head_size,
        state_width,
        compress_ratio,
        True,
        rope_head_dim,
        FP8_MAX,
        head_size,
        token_stride,
        4,
    )

    for token_idx in (3, 7):
        recovered, scale = _decode_indexer_cache(
            kv_cache,
            int(slot_mapping[token_idx].item()),
            block_size,
            head_size,
        )
        expected = _boundary_reference(
            state_cache,
            token_idx,
            token_to_req_indices,
            positions,
            block_table,
            block_size,
            norm_weight,
            1e-6,
            cos_sin_cache,
            head_size,
            state_width,
            compress_ratio,
            True,
            rope_head_dim,
        )
        expected = expected.to(torch.bfloat16).float()
        expected_scale = 2.0**math.ceil(
            math.log2(max(expected.abs().max().item(), 1e-4) / FP8_MAX))

        assert scale.item() == pytest.approx(expected_scale)
        torch.testing.assert_close(
            recovered * scale,
            expected,
            atol=16.0 * expected_scale,
            rtol=0,
        )

    cache_2d = kv_cache.reshape(kv_cache.shape[0], -1)
    assert torch.all(cache_2d[0, :head_size] == 0x7B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm70_compressor_sparse_attention_fallback_writes_flashmla_cache(
) -> None:
    device = "cuda"
    torch.manual_seed(1)
    head_size = 512
    rope_head_dim = 64
    compress_ratio = 4
    block_size = 4
    state_width = 2 * head_size
    num_tokens = 8
    num_blocks = 3
    token_stride = 576

    state_cache = torch.randn(
        num_blocks,
        block_size,
        2 * state_width,
        dtype=torch.float32,
        device=device,
    ) * 2048.0
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32,
                               device=device).unsqueeze(0)
    norm_weight = torch.randn(head_size, dtype=torch.float32, device=device)
    cos_sin_cache = torch.randn(32, rope_head_dim, dtype=torch.float32,
                                device=device)
    kv_cache = torch.full(
        (num_blocks, block_size, 584),
        0x7B,
        dtype=torch.uint8,
        device=device,
    )

    _torch_fused_compress_norm_rope_insert_fp8_fallback(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        norm_weight,
        1e-6,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        block_size,
        head_size,
        state_width,
        compress_ratio,
        True,
        rope_head_dim,
        FP8_MAX,
        64,
        token_stride,
        8,
    )

    for token_idx in (3, 7):
        nope, rope, scales = _decode_sparse_attention_cache(
            kv_cache,
            int(slot_mapping[token_idx].item()),
            block_size,
        )
        expected = _boundary_reference(
            state_cache,
            token_idx,
            token_to_req_indices,
            positions,
            block_table,
            block_size,
            norm_weight,
            1e-6,
            cos_sin_cache,
            head_size,
            state_width,
            compress_ratio,
            True,
            rope_head_dim,
        )
        expected_nope = expected[:448].to(torch.bfloat16).float()
        expected_rope = expected[448:].to(torch.bfloat16).float()

        recovered_nope = nope * scales.repeat_interleave(64)
        max_allowed = 16.0 * scales.max().item()
        assert (recovered_nope - expected_nope).abs().max().item() <= max_allowed
        torch.testing.assert_close(rope, expected_rope, atol=0, rtol=0)

    cache_2d = kv_cache.reshape(kv_cache.shape[0], -1)
    assert torch.all(cache_2d[0, :token_stride] == 0x7B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm70_compressor_sparse_attention_triton_matches_reference_cache_bytes(
) -> None:
    if torch.cuda.get_device_capability()[0] != 7:
        pytest.skip("SM70 parity test")

    device = "cuda"
    torch.manual_seed(123)
    head_size = 512
    rope_head_dim = 64
    compress_ratio = 4
    block_size = 4
    state_width = 2 * head_size
    num_tokens = 8
    num_blocks = 3
    token_stride = 576
    scale_dim = 8

    state_cache = torch.randn(
        num_blocks,
        block_size,
        2 * state_width,
        dtype=torch.float32,
        device=device,
    )
    token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32,
                               device=device).unsqueeze(0)
    norm_weight = torch.randn(head_size, dtype=torch.float32, device=device)
    cos_sin_cache = torch.randn(32, rope_head_dim, dtype=torch.float32,
                                device=device)
    kv_cache = torch.full(
        (num_blocks, block_size, 584),
        0x7B,
        dtype=torch.uint8,
        device=device,
    )

    _fused_kv_compress_norm_rope_insert_sparse_attn[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        norm_weight,
        1e-6,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_size),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=True,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=64,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=4,
        launch_pdl=False,
    )
    torch.cuda.synchronize()

    cache_2d = kv_cache.reshape(kv_cache.shape[0], -1)
    for token_idx in (3, 7):
        expected = _boundary_reference(
            state_cache,
            token_idx,
            token_to_req_indices,
            positions,
            block_table,
            block_size,
            norm_weight,
            1e-6,
            cos_sin_cache,
            head_size,
            state_width,
            compress_ratio,
            True,
            rope_head_dim,
        )
        fp8_bytes, rope_bytes, scale_bytes = _expected_sparse_attention_cache_bytes(
            expected
        )

        slot = int(slot_mapping[token_idx].item())
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        data_offset = pos_in_block * token_stride
        scale_offset = block_size * token_stride + pos_in_block * scale_dim
        actual_fp8 = cache_2d[block_idx, data_offset:data_offset + 448]
        actual_rope = cache_2d[
            block_idx, data_offset + 448:data_offset + token_stride
        ]
        actual_scales = cache_2d[block_idx, scale_offset:scale_offset + 7]

        torch.testing.assert_close(actual_fp8, fp8_bytes)
        torch.testing.assert_close(actual_rope, rope_bytes)
        torch.testing.assert_close(actual_scales, scale_bytes)
