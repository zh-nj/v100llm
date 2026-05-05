# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for SM70 Triton Q-norm/RoPE/KV-insert kernel.

Compares _sm70_triton_qnorm_rope_kv_insert against
_torch_qnorm_rope_kv_insert_fallback for edge cases:
zero-length, single-token, multi-token, padding sentinel,
FP8 quantization accuracy, RoPE correctness, and UE8M0 scale correctness.

Validates: Requirements 11.2, 6.2
"""

import pytest
import torch

from vllm.model_executor.layers.deepseek_v4_attention import (
    _sm70_triton_qnorm_rope_kv_insert,
    _torch_qnorm_rope_kv_insert_fallback,
)

# ── Constants matching the kernel ────────────────────────────────────────────
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK = 64
N_QUANT_BLOCKS = 7
TOKEN_DATA_BYTES = 576  # 448 + 64*2
SCALE_BYTES = 8
HEAD_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES  # 584

# ── SM70 detection ───────────────────────────────────────────────────────────
_has_cuda = torch.cuda.is_available()
_is_sm70 = _has_cuda and torch.cuda.get_device_capability()[0] == 7

pytestmark = [
    pytest.mark.skipif(not _has_cuda, reason="CUDA not available"),
    pytest.mark.skipif(not _is_sm70, reason="SM70 required"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_cos_sin_cache(max_pos: int, rope_dim: int, device: str) -> torch.Tensor:
    """Build a cos||sin cache matching DeepseekV4 layout.
    cos_sin_cache[pos, :rope_dim/2] = cos(theta), [rope_dim/2:] = sin(theta).
    """
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
            / rope_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache  # [max_pos, rope_dim], float32


def make_inputs(
    num_tokens: int,
    n_heads: int,
    block_size: int,
    device: str = "cuda",
    seed: int = 42,
    slot_mapping_override: torch.Tensor | None = None,
):
    """Create matching inputs for both Triton and torch fallback paths."""
    torch.manual_seed(seed)
    max_pos = 4096

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, device)

    if slot_mapping_override is not None:
        slot_mapping = slot_mapping_override
    else:
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    max_slot = int(slot_mapping.max().item()) if slot_mapping.numel() > 0 and (slot_mapping >= 0).any() else 0
    num_blocks = max_slot // block_size + 2

    return q, kv, positions, cos_sin_cache, slot_mapping, num_blocks


def run_both(
    num_tokens: int,
    n_heads: int,
    block_size: int,
    slot_mapping_override: torch.Tensor | None = None,
    seed: int = 42,
):
    """Run both Triton and torch fallback, return (q_triton, k_cache_triton,
    q_torch, k_cache_torch)."""
    device = "cuda"
    q, kv, positions, cos_sin_cache, slot_mapping, num_blocks = make_inputs(
        num_tokens, n_heads, block_size, device, seed, slot_mapping_override
    )
    eps = 1e-6

    # Triton path
    q_triton = q.clone()
    k_cache_triton = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    _sm70_triton_qnorm_rope_kv_insert(
        q_triton, kv, k_cache_triton, slot_mapping,
        positions, cos_sin_cache, eps, block_size,
    )

    # Torch fallback path
    q_torch = q.clone()
    k_cache_torch = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    _torch_qnorm_rope_kv_insert_fallback(
        q_torch, kv, k_cache_torch, slot_mapping,
        positions, cos_sin_cache, eps, block_size,
    )

    return q_triton, k_cache_triton, q_torch, k_cache_torch


# ── Test 1: Zero-length input ────────────────────────────────────────────────


def test_zero_length_input():
    """num_tokens=0 — should return without error."""
    device = "cuda"
    n_heads = 8
    block_size = 16
    num_tokens = 0
    eps = 1e-6

    q = torch.randn(1, n_heads, HEAD_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(1, HEAD_DIM, dtype=torch.float16, device=device)
    positions = torch.zeros(1, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(4096, ROPE_DIM, device)
    slot_mapping = torch.empty(0, dtype=torch.int64, device=device)
    k_cache = torch.zeros(1, block_size * HEAD_BYTES, dtype=torch.uint8, device=device)

    q_orig = q.clone()
    k_cache_orig = k_cache.clone()

    # Both functions should handle zero-length gracefully
    _sm70_triton_qnorm_rope_kv_insert(
        q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size,
    )

    # Q should be unchanged (Triton kernel skips since num_tokens from slot_mapping is 0)
    # k_cache should be unchanged
    assert k_cache.equal(k_cache_orig)


# ── Test 2: Single token ─────────────────────────────────────────────────────


def test_single_token():
    """num_tokens=1 — compare Q output and cache bytes vs torch fallback."""
    n_heads = 8
    block_size = 16

    q_triton, k_cache_triton, q_torch, k_cache_torch = run_both(
        num_tokens=1, n_heads=n_heads, block_size=block_size,
    )

    # Q comparison with fp16 tolerance
    torch.testing.assert_close(q_triton, q_torch, rtol=1e-2, atol=5e-3)

    # Cache comparison: count byte mismatches (expect ≤1% due to FP8 rounding)
    total_bytes = k_cache_triton.numel()
    mismatches = (k_cache_triton != k_cache_torch).sum().item()
    mismatch_pct = mismatches / total_bytes * 100
    assert mismatch_pct <= 1.0, (
        f"Cache byte mismatch: {mismatches}/{total_bytes} ({mismatch_pct:.2f}%)"
    )


# ── Test 3: Multi-token ──────────────────────────────────────────────────────


def test_multi_token():
    """num_tokens=16 — compare Q output and cache bytes."""
    n_heads = 8
    block_size = 16

    q_triton, k_cache_triton, q_torch, k_cache_torch = run_both(
        num_tokens=16, n_heads=n_heads, block_size=block_size,
    )

    torch.testing.assert_close(q_triton, q_torch, rtol=1e-2, atol=5e-3)

    total_bytes = k_cache_triton.numel()
    mismatches = (k_cache_triton != k_cache_torch).sum().item()
    mismatch_pct = mismatches / total_bytes * 100
    assert mismatch_pct <= 1.0, (
        f"Cache byte mismatch: {mismatches}/{total_bytes} ({mismatch_pct:.2f}%)"
    )


# ── Test 4: Padding sentinel ─────────────────────────────────────────────────


def test_padding_sentinel():
    """Mix of valid (slot_mapping >= 0) and invalid (slot_mapping = -1) tokens."""
    device = "cuda"
    n_heads = 8
    block_size = 16
    num_tokens = 8

    # Slots: 0, -1, 2, -1, 4, -1, 6, -1 (alternating valid/invalid)
    slot_mapping = torch.tensor(
        [0, -1, 2, -1, 4, -1, 6, -1], dtype=torch.int64, device=device
    )

    q_triton, k_cache_triton, q_torch, k_cache_torch = run_both(
        num_tokens=num_tokens,
        n_heads=n_heads,
        block_size=block_size,
        slot_mapping_override=slot_mapping,
    )

    # Q should still match (Q processing is independent of slot_mapping)
    torch.testing.assert_close(q_triton, q_torch, rtol=1e-2, atol=5e-3)

    # Cache: valid slots should have data, invalid slots should be zeros
    # Compare overall byte mismatch rate
    total_bytes = k_cache_triton.numel()
    mismatches = (k_cache_triton != k_cache_torch).sum().item()
    mismatch_pct = mismatches / total_bytes * 100
    assert mismatch_pct <= 1.0, (
        f"Cache byte mismatch with padding: {mismatches}/{total_bytes} "
        f"({mismatch_pct:.2f}%)"
    )


# ── Test 5: FP8 quantization accuracy ────────────────────────────────────────


def test_fp8_quantization_accuracy():
    """Verify NoPE FP8 values match torch fallback within ±1 ULP."""
    device = "cuda"
    n_heads = 8
    block_size = 16
    num_tokens = 4

    q_triton, k_cache_triton, q_torch, k_cache_torch = run_both(
        num_tokens=num_tokens, n_heads=n_heads, block_size=block_size, seed=123,
    )

    # Extract NoPE FP8 bytes from cache for each token
    for tok_idx in range(num_tokens):
        pos_in_block = tok_idx % block_size
        token_data_offset = pos_in_block * TOKEN_DATA_BYTES

        # NoPE region: first 448 bytes
        nope_triton = k_cache_triton[0, token_data_offset:token_data_offset + NOPE_DIM]
        nope_torch = k_cache_torch[0, token_data_offset:token_data_offset + NOPE_DIM]

        # Count differences — allow at most 1 ULP difference per value
        # In FP8 e4m3fn, 1 ULP means ±1 in the byte representation
        diff = (nope_triton.to(torch.int16) - nope_torch.to(torch.int16)).abs()
        ulp_violations = (diff > 1).sum().item()
        assert ulp_violations == 0, (
            f"Token {tok_idx}: {ulp_violations}/{NOPE_DIM} NoPE bytes differ "
            f"by more than 1 ULP"
        )


# ── Test 6: RoPE correctness ─────────────────────────────────────────────────


def test_rope_correctness():
    """Verify Q RoPE rotation matches torch fallback."""
    n_heads = 8
    block_size = 16
    num_tokens = 8

    q_triton, _, q_torch, _ = run_both(
        num_tokens=num_tokens, n_heads=n_heads, block_size=block_size, seed=77,
    )

    # Check specifically the RoPE portion of Q (last 64 dims)
    q_rope_triton = q_triton[:, :, NOPE_DIM:]
    q_rope_torch = q_torch[:, :, NOPE_DIM:]

    torch.testing.assert_close(
        q_rope_triton, q_rope_torch, rtol=1e-2, atol=5e-3,
    )

    # Also verify the NoPE portion (first 448 dims) has correct RMSNorm
    q_nope_triton = q_triton[:, :, :NOPE_DIM]
    q_nope_torch = q_torch[:, :, :NOPE_DIM]

    torch.testing.assert_close(
        q_nope_triton, q_nope_torch, rtol=1e-2, atol=5e-3,
    )


# ── Test 7: UE8M0 scale correctness ──────────────────────────────────────────


def test_ue8m0_scale_correctness():
    """Verify scale bytes match exactly between Triton and torch fallback."""
    device = "cuda"
    n_heads = 8
    block_size = 16
    num_tokens = 8

    q_triton, k_cache_triton, q_torch, k_cache_torch = run_both(
        num_tokens=num_tokens, n_heads=n_heads, block_size=block_size, seed=99,
    )

    # Extract scale bytes for each token from the cache
    scale_area_offset = block_size * TOKEN_DATA_BYTES

    for tok_idx in range(num_tokens):
        pos_in_block = tok_idx % block_size
        scale_offset = scale_area_offset + pos_in_block * SCALE_BYTES

        # 7 valid scale bytes (one per quant block)
        scales_triton = k_cache_triton[0, scale_offset:scale_offset + N_QUANT_BLOCKS]
        scales_torch = k_cache_torch[0, scale_offset:scale_offset + N_QUANT_BLOCKS]

        assert scales_triton.equal(scales_torch), (
            f"Token {tok_idx}: scale bytes differ.\n"
            f"  Triton: {scales_triton.tolist()}\n"
            f"  Torch:  {scales_torch.tolist()}"
        )
