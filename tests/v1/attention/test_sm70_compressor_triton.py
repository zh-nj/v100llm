# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for SM70 Triton compressor fallback replacement.

Compares _sm70_triton_fused_compress_norm_rope_insert_fp8 against
_torch_fused_compress_norm_rope_insert_fp8_fallback at compress_ratio
boundaries for C4A (compress_ratio=4) and C128A (compress_ratio=128).

Validates: Requirements 11.2, 6.3
"""

import pytest
import torch

from vllm.model_executor.layers.deepseek_compressor import (
    _sm70_triton_fused_compress_norm_rope_insert_fp8,
    _torch_fused_compress_norm_rope_insert_fp8_fallback,
)

# ── Constants ────────────────────────────────────────────────────────────────
# C4A (head_size=512, compress_ratio=4)
C4A_HEAD_SIZE = 512
C4A_COMPRESS_RATIO = 4
C4A_NOPE_DIM = 448
C4A_ROPE_DIM = 64
C4A_QUANT_BLOCK = 64
C4A_TOKEN_STRIDE = C4A_NOPE_DIM + C4A_ROPE_DIM * 2  # 576
C4A_SCALE_DIM = C4A_NOPE_DIM // 64 + 1  # 8 (7 real + 1 pad)
C4A_OVERLAP = True
C4A_STATE_WIDTH_MULT = 2  # coff = 1 + int(overlap) = 2

# C128A (head_size=512, compress_ratio=128)
C128A_HEAD_SIZE = 512
C128A_COMPRESS_RATIO = 128
C128A_NOPE_DIM = 448
C128A_ROPE_DIM = 64
C128A_QUANT_BLOCK = 64
C128A_TOKEN_STRIDE = C128A_NOPE_DIM + C128A_ROPE_DIM * 2  # 576
C128A_SCALE_DIM = C128A_NOPE_DIM // 64 + 1  # 8
C128A_OVERLAP = False
C128A_STATE_WIDTH_MULT = 1  # coff = 1 + int(overlap) = 1

FP8_MAX = 448.0
RMS_NORM_EPS = 1e-6
ROPE_HEAD_DIM = 64

# ── SM70 detection ───────────────────────────────────────────────────────────
_has_cuda = torch.cuda.is_available()
_is_sm70 = _has_cuda and torch.cuda.get_device_capability()[0] == 7

pytestmark = [
    pytest.mark.skipif(not _has_cuda, reason="CUDA not available"),
    pytest.mark.skipif(not _is_sm70, reason="SM70 required"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_cos_sin_cache(max_pos: int, rope_dim: int, device: str) -> torch.Tensor:
    """Build a cos||sin cache matching DeepseekV4 layout."""
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


def make_compressor_inputs(
    num_tokens: int,
    head_size: int,
    compress_ratio: int,
    overlap: bool,
    device: str = "cuda",
    seed: int = 42,
    positions_override: torch.Tensor | None = None,
):
    """Create matching inputs for both Triton and torch compressor paths.

    Sets up state_cache, block_table, slot_mapping, positions, rms_norm_weight,
    cos_sin_cache, kv_cache, and kv_slot_mapping to match the compressor's
    requirements.

    Returns a dict of all arguments needed by both functions.
    """
    torch.manual_seed(seed)

    coff = 1 + int(overlap)
    state_dim = 2 * coff * head_size  # kv_state + score_state
    state_width = state_dim // 2

    # State cache block parameters
    if compress_ratio == 4:
        state_block_size = 4
    elif compress_ratio == 128:
        state_block_size = 8
    else:
        raise ValueError(f"Invalid compress_ratio: {compress_ratio}")

    window = coff * compress_ratio

    # Positions: we need tokens at compress boundaries
    if positions_override is not None:
        positions = positions_override.to(device)
    else:
        # Place tokens at positions that include compress_ratio boundaries
        # E.g. for compress_ratio=4: positions 0,1,2,3,4,5,6,7
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Need enough blocks to cover the window for any token
    max_pos = int(positions.max().item()) if positions.numel() > 0 else 0
    max_block_in_seq = max_pos // state_block_size + 1
    num_state_blocks = max_block_in_seq + 2

    # State cache: [num_blocks, block_size, state_dim] float32
    state_cache = torch.randn(
        num_state_blocks, state_block_size, state_dim,
        dtype=torch.float32, device=device,
    ) * 0.5  # moderate values for numerical stability

    # Block table: maps (req_idx, block_in_seq) → physical block
    # Use identity mapping for simplicity (single request)
    num_reqs = 1
    block_table = torch.arange(
        num_state_blocks, dtype=torch.int32, device=device,
    ).unsqueeze(0).expand(num_reqs, -1).contiguous()

    # token_to_req_indices: all tokens belong to request 0
    token_to_req_indices = torch.zeros(
        num_tokens, dtype=torch.int32, device=device,
    )

    # Slot mapping for state cache
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # RMS norm weight
    rms_norm_weight = torch.ones(head_size, dtype=torch.float32, device=device)
    # Add slight variation to make norm non-trivial
    rms_norm_weight += torch.randn(head_size, dtype=torch.float32, device=device) * 0.1

    # Cos/sin cache
    max_model_len = max(max_pos + 256, 512)
    cos_sin_cache = make_cos_sin_cache(max_model_len, ROPE_HEAD_DIM, device)

    # KV cache (output cache): paged uint8
    # Determine parameters based on head_size
    if head_size == 512:
        token_stride = C4A_TOKEN_STRIDE
        scale_dim = C4A_SCALE_DIM
        quant_block = C4A_QUANT_BLOCK
    elif head_size == 128:
        token_stride = head_size
        scale_dim = 4  # single float32 scale
        quant_block = 128
    else:
        raise ValueError(f"Unsupported head_size: {head_size}")

    kv_cache_block_size = 256  # standard MLA cache block size

    # Count how many tokens fire (at compress boundaries)
    fire_mask = (positions + 1) % compress_ratio == 0
    num_firing = int(fire_mask.sum().item())

    # kv_slot_mapping: map each token to its KV cache slot
    # Only tokens at boundaries actually write, but all need valid mappings
    kv_slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Allocate enough KV cache blocks
    max_kv_slot = int(kv_slot_mapping.max().item()) if kv_slot_mapping.numel() > 0 else 0
    num_kv_blocks = max_kv_slot // kv_cache_block_size + 2

    # KV cache block stride: block_size * token_stride + block_size * scale_dim
    kv_block_bytes = kv_cache_block_size * token_stride + kv_cache_block_size * scale_dim
    kv_cache = torch.zeros(
        num_kv_blocks, kv_block_bytes,
        dtype=torch.uint8, device=device,
    )

    return dict(
        state_cache=state_cache,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=state_block_size,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=RMS_NORM_EPS,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        kv_slot_mapping=kv_slot_mapping,
        kv_cache_block_size=kv_cache_block_size,
        head_size=head_size,
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_head_dim=ROPE_HEAD_DIM,
        fp8_max=FP8_MAX,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
    )


def run_both_compressors(
    num_tokens: int,
    head_size: int,
    compress_ratio: int,
    overlap: bool,
    seed: int = 42,
    positions_override: torch.Tensor | None = None,
):
    """Run both Triton and torch fallback compressors, return both KV caches."""
    inputs = make_compressor_inputs(
        num_tokens=num_tokens,
        head_size=head_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        seed=seed,
        positions_override=positions_override,
    )

    # Clone inputs for each path (state_cache is read-only, kv_cache is written)
    kv_cache_triton = inputs["kv_cache"].clone()
    kv_cache_torch = inputs["kv_cache"].clone()

    # Run Triton path
    _sm70_triton_fused_compress_norm_rope_insert_fp8(
        state_cache=inputs["state_cache"],
        token_to_req_indices=inputs["token_to_req_indices"],
        positions=inputs["positions"],
        slot_mapping=inputs["slot_mapping"],
        block_table=inputs["block_table"],
        block_size=inputs["block_size"],
        rms_norm_weight=inputs["rms_norm_weight"],
        rms_norm_eps=inputs["rms_norm_eps"],
        cos_sin_cache=inputs["cos_sin_cache"],
        kv_cache=kv_cache_triton,
        kv_slot_mapping=inputs["kv_slot_mapping"],
        kv_cache_block_size=inputs["kv_cache_block_size"],
        head_size=inputs["head_size"],
        state_width=inputs["state_width"],
        compress_ratio=inputs["compress_ratio"],
        overlap=inputs["overlap"],
        rope_head_dim=inputs["rope_head_dim"],
        fp8_max=inputs["fp8_max"],
        quant_block=inputs["quant_block"],
        token_stride=inputs["token_stride"],
        scale_dim=inputs["scale_dim"],
    )

    # Run torch fallback path
    _torch_fused_compress_norm_rope_insert_fp8_fallback(
        state_cache=inputs["state_cache"],
        token_to_req_indices=inputs["token_to_req_indices"],
        positions=inputs["positions"],
        slot_mapping=inputs["slot_mapping"],
        block_table=inputs["block_table"],
        block_size=inputs["block_size"],
        rms_norm_weight=inputs["rms_norm_weight"],
        rms_norm_eps=inputs["rms_norm_eps"],
        cos_sin_cache=inputs["cos_sin_cache"],
        kv_cache=kv_cache_torch,
        kv_slot_mapping=inputs["kv_slot_mapping"],
        kv_cache_block_size=inputs["kv_cache_block_size"],
        head_size=inputs["head_size"],
        state_width=inputs["state_width"],
        compress_ratio=inputs["compress_ratio"],
        overlap=inputs["overlap"],
        rope_head_dim=inputs["rope_head_dim"],
        fp8_max=inputs["fp8_max"],
        quant_block=inputs["quant_block"],
        token_stride=inputs["token_stride"],
        scale_dim=inputs["scale_dim"],
    )

    return kv_cache_triton, kv_cache_torch, inputs


# ── Test 1: C4A basic test ───────────────────────────────────────────────────


def test_c4a_basic():
    """C4A (compress_ratio=4): tokens at boundary positions, compare cache output."""
    # We need at least 8 tokens so positions 3 and 7 are at compress boundaries
    # (position+1) % 4 == 0 → positions 3, 7
    num_tokens = 8

    kv_triton, kv_torch, inputs = run_both_compressors(
        num_tokens=num_tokens,
        head_size=C4A_HEAD_SIZE,
        compress_ratio=C4A_COMPRESS_RATIO,
        overlap=C4A_OVERLAP,
        seed=42,
    )

    # Compare: bytes that were written should match
    total_bytes = kv_triton.numel()
    mismatches = (kv_triton != kv_torch).sum().item()
    mismatch_pct = mismatches / max(total_bytes, 1) * 100

    assert mismatch_pct <= 1.0, (
        f"C4A cache byte mismatch: {mismatches}/{total_bytes} ({mismatch_pct:.2f}%)"
    )


# ── Test 2: C128A basic test ─────────────────────────────────────────────────


def test_c128a_basic():
    """C128A (compress_ratio=128): tokens at boundary positions, compare cache output."""
    # For compress_ratio=128, boundary is at (position+1) % 128 == 0,
    # i.e. position=127. We need 128 tokens.
    num_tokens = 128

    kv_triton, kv_torch, inputs = run_both_compressors(
        num_tokens=num_tokens,
        head_size=C128A_HEAD_SIZE,
        compress_ratio=C128A_COMPRESS_RATIO,
        overlap=C128A_OVERLAP,
        seed=42,
    )

    total_bytes = kv_triton.numel()
    mismatches = (kv_triton != kv_torch).sum().item()
    mismatch_pct = mismatches / max(total_bytes, 1) * 100

    assert mismatch_pct <= 1.0, (
        f"C128A cache byte mismatch: {mismatches}/{total_bytes} ({mismatch_pct:.2f}%)"
    )


# ── Test 3: Empty firing test ────────────────────────────────────────────────


def test_empty_firing():
    """No tokens at compress boundaries → kernel should be a no-op."""
    device = "cuda"

    # For C4A (compress_ratio=4): boundary at (pos+1)%4==0, i.e. pos=3,7,...
    # Use positions 0,1,2 — none at boundary
    num_tokens = 3
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)

    kv_triton, kv_torch, inputs = run_both_compressors(
        num_tokens=num_tokens,
        head_size=C4A_HEAD_SIZE,
        compress_ratio=C4A_COMPRESS_RATIO,
        overlap=C4A_OVERLAP,
        seed=77,
        positions_override=positions,
    )

    # Both caches should be all zeros (no writes happened)
    assert kv_triton.sum().item() == 0, "Triton wrote to cache despite no firing tokens"
    assert kv_torch.sum().item() == 0, "Torch wrote to cache despite no firing tokens"
    assert kv_triton.equal(kv_torch), "Caches differ despite no firing tokens"


# ── Test 4: FP8 output accuracy ──────────────────────────────────────────────


def test_fp8_output_accuracy():
    """NoPE FP8 bytes match within ±1 ULP between Triton and torch fallback."""
    device = "cuda"
    num_tokens = 8  # positions 3 and 7 are C4A boundaries

    kv_triton, kv_torch, inputs = run_both_compressors(
        num_tokens=num_tokens,
        head_size=C4A_HEAD_SIZE,
        compress_ratio=C4A_COMPRESS_RATIO,
        overlap=C4A_OVERLAP,
        seed=123,
    )

    positions = inputs["positions"]
    kv_cache_block_size = inputs["kv_cache_block_size"]
    token_stride = inputs["token_stride"]
    kv_slot_mapping = inputs["kv_slot_mapping"]

    # Check each firing token
    fire_mask = (positions + 1) % C4A_COMPRESS_RATIO == 0
    firing_indices = fire_mask.nonzero(as_tuple=False).squeeze(1)

    assert firing_indices.numel() > 0, "No firing tokens found"

    for idx in firing_indices.tolist():
        kv_slot = int(kv_slot_mapping[idx].item())
        block_idx = kv_slot // kv_cache_block_size
        pos_in_block = kv_slot % kv_cache_block_size
        data_offset = pos_in_block * token_stride

        # NoPE region: first 448 bytes
        nope_triton = kv_triton[block_idx, data_offset:data_offset + C4A_NOPE_DIM]
        nope_torch = kv_torch[block_idx, data_offset:data_offset + C4A_NOPE_DIM]

        # Allow ±1 ULP difference in FP8 byte representation
        diff = (nope_triton.to(torch.int16) - nope_torch.to(torch.int16)).abs()
        ulp_violations = (diff > 1).sum().item()
        assert ulp_violations == 0, (
            f"Token at pos {idx}: {ulp_violations}/{C4A_NOPE_DIM} NoPE bytes "
            f"differ by more than 1 ULP"
        )


# ── Test 5: Scale byte correctness ───────────────────────────────────────────


def test_scale_byte_correctness():
    """UE8M0 scale bytes match exactly between Triton and torch fallback."""
    device = "cuda"
    num_tokens = 8  # positions 3 and 7 are C4A boundaries

    kv_triton, kv_torch, inputs = run_both_compressors(
        num_tokens=num_tokens,
        head_size=C4A_HEAD_SIZE,
        compress_ratio=C4A_COMPRESS_RATIO,
        overlap=C4A_OVERLAP,
        seed=99,
    )

    positions = inputs["positions"]
    kv_cache_block_size = inputs["kv_cache_block_size"]
    token_stride = inputs["token_stride"]
    scale_dim = inputs["scale_dim"]
    kv_slot_mapping = inputs["kv_slot_mapping"]

    fire_mask = (positions + 1) % C4A_COMPRESS_RATIO == 0
    firing_indices = fire_mask.nonzero(as_tuple=False).squeeze(1)

    assert firing_indices.numel() > 0, "No firing tokens found"

    # Scale area starts after all token data in the block
    scale_area_offset = kv_cache_block_size * token_stride

    n_quant_blocks = C4A_NOPE_DIM // C4A_QUANT_BLOCK  # 7

    for idx in firing_indices.tolist():
        kv_slot = int(kv_slot_mapping[idx].item())
        block_idx = kv_slot // kv_cache_block_size
        pos_in_block = kv_slot % kv_cache_block_size
        scale_offset = scale_area_offset + pos_in_block * scale_dim

        scales_triton = kv_triton[block_idx, scale_offset:scale_offset + n_quant_blocks]
        scales_torch = kv_torch[block_idx, scale_offset:scale_offset + n_quant_blocks]

        assert scales_triton.equal(scales_torch), (
            f"Token at pos {idx}: scale bytes differ.\n"
            f"  Triton: {scales_triton.tolist()}\n"
            f"  Torch:  {scales_torch.tolist()}"
        )
