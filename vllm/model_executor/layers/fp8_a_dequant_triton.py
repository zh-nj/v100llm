# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP8 E4M3 activation dequant to FP16 for SM70 o_einsum.

Replaces
    a_deq = a.float() * a_scale.repeat_interleave(hidden // a_blocks, dim=-1)
    a_deq.half()
with a single Triton kernel, skipping the fp32 materialization and the
`repeat_interleave` scale expansion. Used by `_sm70_fused_o_einsum_wo_b`
and `_sm70_fp8_einsum_bmm`.

Input layout: a is [T, G, D] uint8-view of fp8_e4m3; a_scale is
[T, G, D // block] fp32 with block=128 in the DeepSeek V4 config. Output
is [T, G, D] fp16.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_e4m3_uint8_to_fp32(x_u8):
    """Decode FP8 E4M3 (no NaN, no infinity encoding per DeepSeek) uint8 to
    fp32 in pure Triton. Matches the standard e4m3 bit layout:
      sign(1) | exponent(4) | mantissa(3), bias 7.

    Zero is preserved; denormals (exp=0) yield 0.0 here (accurate enough
    for DeepSeek V4 where denormals are extremely rare and blocked scales
    absorb the tiny-value band).
    """
    sign = (x_u8 >> 7) & 0x1
    exp = (x_u8 >> 3) & 0xF
    mantissa = x_u8 & 0x7

    # Normalized value: (1 + m/8) * 2^(exp - 7)
    # Use bit manipulation to build fp32 bits directly.
    # fp32: sign(1) | exp(8, bias 127) | mantissa(23)
    fp32_exp = (exp.to(tl.int32) + (127 - 7)).to(tl.uint32)
    fp32_mantissa = (mantissa.to(tl.uint32) << 20)
    fp32_bits = (sign.to(tl.uint32) << 31) | (fp32_exp << 23) | fp32_mantissa

    # For exp == 0 (subnormal or zero), force value to 0.
    normalized = tl.where(exp == 0, tl.zeros_like(fp32_bits), fp32_bits)
    return normalized.to(tl.float32, bitcast=True)


@triton.jit
def _fp8_a_dequant_to_fp16_kernel(
    a_ptr,
    a_scale_ptr,
    out_ptr,
    T,
    G,
    D,
    a_stride_t,
    a_stride_g,
    scale_stride_t,
    scale_stride_g,
    out_stride_t,
    out_stride_g,
    BLOCK_D: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
):
    """Dequant a[t,g,:] * a_scale[t,g,:] block-wise to fp16 out[t,g,:].

    Grid: (T, G, D // BLOCK_D). SCALE_GROUP is the stride of the scale
    along the hidden dim (for DeepSeek V4 fp8 this is 128).
    BLOCK_D must divide SCALE_GROUP evenly (use BLOCK_D == SCALE_GROUP
    for simplest implementation).
    """
    t = tl.program_id(0)
    g = tl.program_id(1)
    d_block = tl.program_id(2)

    d_start = d_block * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    a_ptrs = a_ptr + t * a_stride_t + g * a_stride_g + d_offsets
    a_u8 = tl.load(a_ptrs, mask=mask, other=0).to(tl.uint8)
    a_f32 = _fp8_e4m3_uint8_to_fp32(a_u8)

    scale_idx = d_start // SCALE_GROUP
    scale_ptr = a_scale_ptr + t * scale_stride_t + g * scale_stride_g + scale_idx
    scale = tl.load(scale_ptr).to(tl.float32)

    out_vals = (a_f32 * scale).to(tl.float16)
    out_ptrs = out_ptr + t * out_stride_t + g * out_stride_g + d_offsets
    tl.store(out_ptrs, out_vals, mask=mask)


def sm70_fp8_a_dequant_to_fp16(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequant fp8 activation a [T, G, D] with blocked scales a_scale
    [T, G, D // block] into fp16 tensor of the same [T, G, D] shape.

    Args:
        a: fp8_e4m3 activation, shape [T, G, D]. Can be a uint8 view.
        a_scale: fp32 scale, shape [T, G, D // block_size]. ``block_size``
            is inferred from ``D // a_scale.shape[-1]``.
        out: Optional pre-allocated fp16 output tensor. Must have shape
            equal to ``a.shape`` and dtype fp16 if provided.

    Returns:
        fp16 tensor of shape [T, G, D].
    """
    assert a.is_cuda, "sm70_fp8_a_dequant_to_fp16 requires CUDA tensors"
    assert a.ndim == 3, (
        f"Expected a of shape [T, G, D], got {list(a.shape)}"
    )
    T, G, D = a.shape
    # Accept either fp8_e4m3fn or uint8 view; view as uint8 for bit-level decode.
    if a.dtype != torch.uint8:
        a = a.view(torch.uint8)
    assert a_scale.is_cuda and a_scale.ndim == 3, (
        f"Expected a_scale of shape [T, G, D // block], got "
        f"{list(a_scale.shape)}"
    )
    assert a_scale.shape[0] == T and a_scale.shape[1] == G, (
        "a_scale outer dims must match a outer dims"
    )
    block_size = D // a_scale.shape[-1]
    assert D % a_scale.shape[-1] == 0, (
        f"D={D} must be divisible by a_scale.shape[-1]={a_scale.shape[-1]}"
    )
    assert a_scale.dtype == torch.float32, (
        f"a_scale must be float32, got {a_scale.dtype}"
    )

    if out is None:
        out = torch.empty((T, G, D), dtype=torch.float16, device=a.device)
    else:
        assert out.shape == a.shape and out.dtype == torch.float16
    if T == 0:
        return out

    # Use BLOCK_D == block_size for the simple path (one scale per BLOCK_D).
    BLOCK_D = block_size
    grid = (T, G, triton.cdiv(D, BLOCK_D))
    _fp8_a_dequant_to_fp16_kernel[grid](
        a,
        a_scale,
        out,
        T, G, D,
        a.stride(0), a.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        SCALE_GROUP=block_size,
        num_warps=4,
    )
    return out
