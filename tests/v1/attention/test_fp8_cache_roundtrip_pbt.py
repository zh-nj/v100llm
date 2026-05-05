# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Property-based test for FP8 Cache Encode-Decode Round Trip.

Feature: deepseek-v4-flash-sm70-flashmla, Property 1: FP8 Cache Encode-Decode Round Trip

Validates: Requirements 14.3, 14.4

Generates random float32 KV tensors in [-448, 448] range, encodes through
_torch_qnorm_rope_kv_insert_fallback into the FP8 paged cache, decodes through
_gather_decode_prefill_fallback_kv_, and verifies:
  - NoPE values within 1 ULP of FP8 e4m3fn representation
  - RoPE BF16 values are bitwise identical after round-trip
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.model_executor.layers.deepseek_v4_attention import (
    _gather_decode_prefill_fallback_kv_,
    _torch_qnorm_rope_kv_insert_fallback,
)

# ── Constants matching the KV cache layout ───────────────────────────────────
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
    pytest.mark.skipif(not _is_sm70, reason="SM70 (V100) GPU required"),
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


def fp8_e4m3fn_ulp(value: torch.Tensor) -> torch.Tensor:
    """Compute the ULP (unit in the last place) for FP8 e4m3fn values.

    For a value represented in FP8 e4m3fn, the ULP is the difference
    between adjacent representable values at that magnitude.

    FP8 e4m3fn: sign(1) | exp(4) | mantissa(3), bias=7, max=448.0
    Normal: 2^(exp-7) * 2^-3 = 2^(exp-10)
    Subnormal (exp=0): 2^-9
    """
    abs_val = value.abs().float()
    # Convert to FP8 to get the actual exponent
    fp8_val = abs_val.clamp(max=448.0).to(torch.float8_e4m3fn)
    fp8_bytes = fp8_val.view(torch.uint8).to(torch.int32)
    exp_bits = (fp8_bytes >> 3) & 0xF

    # Normal ULP: 2^(exp - 10), Subnormal ULP: 2^-9
    is_subnorm = exp_bits == 0
    ulp = torch.where(
        is_subnorm,
        torch.tensor(2.0**-9, device=value.device, dtype=torch.float32),
        torch.pow(2.0, (exp_bits.float() - 10.0)),
    )
    return ulp


# ── Hypothesis strategies ────────────────────────────────────────────────────


@st.composite
def fp8_roundtrip_inputs(draw):
    """Generate random inputs for encode-decode round trip testing.

    Varies:
        - num_tokens: 1-16
        - n_heads: 1-4 (for Q, though we focus on KV)
        - tensor magnitudes: near-zero, mid-range, near-max
        - mixed signs
        - sequence positions: 0 to 4095
    """
    num_tokens = draw(st.integers(min_value=1, max_value=16))
    n_heads = draw(st.integers(min_value=1, max_value=4))
    block_size = draw(st.sampled_from([16, 64, 256]))

    # Choose magnitude strategy
    magnitude = draw(st.sampled_from(["near_zero", "mid_range", "near_max", "mixed"]))

    # Choose position strategy
    pos_strategy = draw(st.sampled_from(["sequential", "random", "large"]))

    return {
        "num_tokens": num_tokens,
        "n_heads": n_heads,
        "block_size": block_size,
        "magnitude": magnitude,
        "pos_strategy": pos_strategy,
        "seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
    }


def generate_kv_tensor(
    num_tokens: int,
    magnitude: str,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate KV tensor with specified magnitude strategy."""
    kv = torch.randn(
        num_tokens, HEAD_DIM, dtype=torch.float32, device=device,
        generator=generator,
    )

    if magnitude == "near_zero":
        # Values in [-0.01, 0.01] — exercises subnormal FP8 path
        kv = kv * 0.001
    elif magnitude == "mid_range":
        # Values in roughly [-10, 10]
        kv = kv * 3.0
    elif magnitude == "near_max":
        # Values approaching FP8 max (448)
        kv = kv * 100.0
        kv = kv.clamp(-440.0, 440.0)
    elif magnitude == "mixed":
        # Mix of magnitudes across the token dimension
        scales = torch.tensor(
            [0.001, 1.0, 50.0, 200.0], device=device, dtype=torch.float32
        )
        # Cycle through scales for each token
        for i in range(num_tokens):
            scale = scales[i % len(scales)]
            kv[i] = kv[i] * scale

    # Clamp to valid FP8 range
    kv = kv.clamp(-448.0, 448.0)
    return kv


def generate_positions(
    num_tokens: int,
    pos_strategy: str,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate position indices."""
    if pos_strategy == "sequential":
        return torch.arange(num_tokens, dtype=torch.int64, device=device)
    elif pos_strategy == "random":
        return torch.randint(
            0, 4096, (num_tokens,), dtype=torch.int64, device=device,
            generator=generator,
        )
    elif pos_strategy == "large":
        # Positions near the end of context
        base = 3000
        return torch.arange(
            base, base + num_tokens, dtype=torch.int64, device=device
        )
    else:
        return torch.arange(num_tokens, dtype=torch.int64, device=device)


# ── Test Class ───────────────────────────────────────────────────────────────


class TestFP8CacheRoundTrip:
    """Property 1: FP8 Cache Encode-Decode Round Trip.

    Validates: Requirements 14.3, 14.4

    Verifies that encoding KV data into the FP8 paged cache and decoding it
    back produces values within expected tolerance bounds.
    """

    @given(inputs=fp8_roundtrip_inputs())
    @settings(max_examples=100, deadline=None)
    def test_fp8_cache_encode_decode_roundtrip(self, inputs):
        """**Validates: Requirements 14.3, 14.4**

        For any valid KV tensor (NoPE float32 values in [-448, 448] and BF16
        RoPE values), encoding through _torch_qnorm_rope_kv_insert_fallback
        into the FP8 paged cache and decoding through
        _gather_decode_prefill_fallback_kv_ SHALL produce:
          - NoPE values within 1 ULP of the FP8 e4m3fn representation
          - RoPE values that are bitwise identical after BF16 round-trip
        """
        device = "cuda"
        num_tokens = inputs["num_tokens"]
        n_heads = inputs["n_heads"]
        block_size = inputs["block_size"]
        magnitude = inputs["magnitude"]
        pos_strategy = inputs["pos_strategy"]
        seed = inputs["seed"]

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Generate inputs
        kv = generate_kv_tensor(num_tokens, magnitude, device, generator)
        positions = generate_positions(num_tokens, pos_strategy, device, generator)
        cos_sin_cache = make_cos_sin_cache(4096, ROPE_DIM, device)

        # Q tensor needed for encode function (modified in-place but we
        # don't test Q here — just KV round trip)
        q = torch.randn(
            num_tokens, n_heads, HEAD_DIM, dtype=torch.float16, device=device,
            generator=generator,
        )

        # Slot mapping: sequential valid slots
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

        # Allocate cache
        num_blocks = (num_tokens // block_size) + 2
        k_cache = torch.zeros(
            num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
        )

        eps = 1e-6

        # ── ENCODE: write KV to cache via torch fallback ──
        # Convert KV to fp16 as the function expects fp16 input
        kv_fp16 = kv.to(torch.float16)
        _torch_qnorm_rope_kv_insert_fallback(
            q, kv_fp16, k_cache, slot_mapping, positions, cos_sin_cache,
            eps, block_size,
        )

        # ── DECODE: read back from cache via Triton gather ──
        # Construct indices: each token reads its own slot
        # global_indices shape: [num_tokens, topk] where topk=1
        topk = 1
        global_indices = slot_mapping.to(torch.int32).unsqueeze(1)  # [N, 1]
        global_lens = torch.ones(num_tokens, dtype=torch.int32, device=device)

        # Output workspace: [num_tokens, topk, 512] as bfloat16
        out = torch.zeros(
            num_tokens, topk, HEAD_DIM, dtype=torch.bfloat16, device=device
        )

        _gather_decode_prefill_fallback_kv_(
            out, k_cache, global_indices, global_lens, block_size,
        )

        # ── VERIFY NoPE: within 1 ULP of FP8 e4m3fn representation ──
        # The expected NoPE path is:
        #   original float32 → FP8 e4m3fn (with block quant) → float32 → bf16
        # So we compute the "ideal" decode: quantize original through FP8 path
        # and compare against actual decoded output.

        # Compute expected NoPE output:
        # 1. The encode function applies GPT-J RoPE to the rope portion but
        #    NoPE portion stays as-is (just quantized)
        nope_original = kv_fp16[:, :NOPE_DIM].float()

        # Block quantize the NoPE portion the same way as the encode
        blocks = nope_original.view(-1, N_QUANT_BLOCKS, QUANT_BLOCK)
        absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        exponents = torch.ceil(torch.log2(absmax / 448.0))
        scales = torch.exp2(exponents)
        scaled_vals = (blocks / scales).clamp(-448.0, 448.0)

        # FP8 round-trip: float32 → fp8 → float32
        fp8_vals = scaled_vals.to(torch.float8_e4m3fn)
        fp8_dequant = fp8_vals.float()

        # Apply scale back
        nope_expected_f32 = (fp8_dequant * scales).view(-1, NOPE_DIM)

        # Convert to bf16 (same as decode kernel does)
        nope_expected_bf16 = nope_expected_f32.to(torch.bfloat16)

        # Get actual decoded NoPE (first 448 dims of the 512-dim output)
        nope_actual = out[:, 0, :NOPE_DIM]  # [num_tokens, 448] bf16

        # Compute tolerance: 1 ULP of the FP8 value after scale application
        # The error bound is: scale * ULP_of_fp8_value
        # Since fp8_dequant already has the fp8-quantized value, the error in
        # the dequanted domain is at most scale * 1_ulp_of_fp8
        fp8_ulps = fp8_e4m3fn_ulp(scaled_vals)  # ULP at the scaled level
        tolerance = (fp8_ulps * scales).view(-1, NOPE_DIM).to(torch.float32)

        # Also account for bf16 rounding (which adds ~0.5 ULP of bf16)
        # bf16 has 7 mantissa bits, so relative error is 2^-8
        bf16_eps = nope_expected_f32.abs() * (2.0**-8) + 1e-10

        total_tolerance = tolerance + bf16_eps

        nope_diff = (nope_actual.float() - nope_expected_bf16.float()).abs()

        violations = (nope_diff > total_tolerance).sum().item()
        total_elements = nope_diff.numel()

        assert violations == 0, (
            f"NoPE round-trip violations: {violations}/{total_elements} "
            f"elements exceed 1 ULP tolerance. "
            f"Max diff: {nope_diff.max().item():.6e}, "
            f"Max tolerance: {total_tolerance.max().item():.6e}, "
            f"Magnitude: {magnitude}, Tokens: {num_tokens}"
        )

        # ── VERIFY RoPE: bitwise identical after BF16 round-trip ──
        # The encode path is:
        #   kv (fp16) → _apply_gptj_rope_tail → clone to float32 → rotate
        #   → cast back to fp16 (x.dtype) → then .to(bfloat16) for storage
        # The decode copies the BF16 bytes directly from cache.
        # So expected = rope(kv_fp16_as_f32) → fp16 → bf16

        # Compute expected RoPE output (match exact encode path)
        kv_rope = kv_fp16[:, NOPE_DIM:].float()  # [N, 64]
        cos_sin = cos_sin_cache[positions].float()  # [N, 64]
        cos_vals = cos_sin[:, :32]  # [N, 32]
        sin_vals = cos_sin[:, 32:]  # [N, 32]

        rope_even = kv_rope[:, ::2]  # [N, 32]
        rope_odd = kv_rope[:, 1::2]  # [N, 32]

        rotated_even = rope_even * cos_vals - rope_odd * sin_vals
        rotated_odd = rope_odd * cos_vals + rope_even * sin_vals

        # Interleave back
        rope_rotated = torch.zeros_like(kv_rope)
        rope_rotated[:, ::2] = rotated_even
        rope_rotated[:, 1::2] = rotated_odd

        # Match encode path: float32 → fp16 → bf16
        rope_expected_bf16 = rope_rotated.to(torch.float16).to(torch.bfloat16)

        # Get actual decoded RoPE (last 64 dims of the 512-dim output)
        rope_actual = out[:, 0, NOPE_DIM:]  # [num_tokens, 64] bf16

        # Bitwise comparison: view as uint16 and compare
        rope_expected_u16 = rope_expected_bf16.view(torch.uint16)
        rope_actual_u16 = rope_actual.view(torch.uint16)

        rope_mismatches = (rope_expected_u16 != rope_actual_u16).sum().item()
        total_rope_elements = rope_actual_u16.numel()

        assert rope_mismatches == 0, (
            f"RoPE BF16 bitwise mismatches: {rope_mismatches}/{total_rope_elements}. "
            f"Expected bitwise identical after BF16 round-trip. "
            f"Magnitude: {magnitude}, Tokens: {num_tokens}"
        )
