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
    """Decode FP8 E4M3FN uint8 to fp32 in pure Triton.

    Matches the standard e4m3 bit layout:
      sign(1) | exponent(4) | mantissa(3), bias 7.

    Matches PyTorch ``torch.float8_e4m3fn`` decode semantics, including
    subnormal values (exp=0, mantissa!=0) and the two NaN encodings
    (exp=15, mantissa=7). DeepSeek's quantizers should not generate NaNs,
    but preserving the dtype contract keeps this helper a semantic drop-in
    replacement for ``a.float()``.
    """
    val32 = x_u8.to(tl.int32)
    sign_bit = (val32 & 0x80) << 24
    low7 = val32 & 0x7F

    # Normalized value: (1 + m/8) * 2^(exp - 7). Build fp32 bits by
    # shifting the packed exp|mantissa low7 field after bias adjustment.
    fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
    fp32_bits = tl.where(low7 == 0, sign_bit, fp32_bits)
    normal_val = fp32_bits.to(tl.float32, bitcast=True)

    # Subnormal E4M3FN: mantissa * 2^-9, with sign preserved.
    subnormal_mag = low7.to(tl.float32) * 1.953125e-3
    subnormal_val = tl.where(
        (val32 & 0x80) != 0, -subnormal_mag, subnormal_mag
    )

    # Preserve signed zero and represent E4M3FN's reserved max-mantissa
    # encodings as quiet fp32 NaNs.
    nan_bits = sign_bit | 0x7FC00000
    nan_val = nan_bits.to(tl.float32, bitcast=True)

    is_subnormal = (low7 < 8) & (low7 != 0)
    decoded = tl.where(is_subnormal, subnormal_val, normal_val)
    return tl.where(low7 == 0x7F, nan_val, decoded)


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
    scale_stride_k,
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

    ``scale_stride_k`` is the stride of ``a_scale`` along the last dim
    (the scale-block-index dim). Production layouts from
    ``fused_inv_rope_fp8_quant`` produce non-contiguous scale tensors
    via ``as_strided`` where ``scale.stride(-1) = tma_aligned_T != 1``,
    so this stride MUST be passed explicitly instead of assuming 1.
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
    scale_ptr = (
        a_scale_ptr
        + t * scale_stride_t
        + g * scale_stride_g
        + scale_idx * scale_stride_k
    )
    scale = tl.load(scale_ptr).to(tl.float32)

    out_vals = (a_f32 * scale).to(tl.float16)
    out_ptrs = out_ptr + t * out_stride_t + g * out_stride_g + d_offsets
    tl.store(out_ptrs, out_vals, mask=mask)


@triton.jit
def _fp8_weight_predequant_to_fp16_kernel(
    b_ptr,
    b_scale_ptr,
    out_ptr,
    rank,
    hidden,
    b_stride_g,
    b_stride_r,
    b_stride_h,
    scale_stride_g,
    scale_stride_r,
    scale_stride_h,
    out_stride_g,
    out_stride_r,
    out_stride_h,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Dequant one [rank, hidden] FP8 weight tile directly to fp16."""
    g = tl.program_id(0)
    r_block = tl.program_id(1)
    h_block = tl.program_id(2)

    r_offsets = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (r_offsets[:, None] < rank) & (h_offsets[None, :] < hidden)

    b_ptrs = (
        b_ptr
        + g * b_stride_g
        + r_offsets[:, None] * b_stride_r
        + h_offsets[None, :] * b_stride_h
    )
    b_u8 = tl.load(b_ptrs, mask=mask, other=0).to(tl.uint8)
    b_f32 = _fp8_e4m3_uint8_to_fp32(b_u8)

    scale_ptr = (
        b_scale_ptr
        + g * scale_stride_g
        + (r_block * BLOCK_R // 128) * scale_stride_r
        + (h_block * BLOCK_H // 128) * scale_stride_h
    )
    scale = tl.load(scale_ptr).to(tl.float32)

    out_vals = (b_f32 * scale).to(tl.float16)
    out_ptrs = (
        out_ptr
        + g * out_stride_g
        + r_offsets[:, None] * out_stride_r
        + h_offsets[None, :] * out_stride_h
    )
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
        a_scale.stride(0), a_scale.stride(1), a_scale.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        SCALE_GROUP=block_size,
        num_warps=4,
    )
    return out


def sm70_fp8_weight_predequant_to_fp16(
    b: torch.Tensor,
    b_scale: torch.Tensor,
    groups: int,
    rank: int,
    hidden: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize FP8 O-projection weights directly to fp16.

    This is the weight-side sibling of ``sm70_fp8_a_dequant_to_fp16``. It
    replaces the old eager expression
    ``(b.float() * scale.repeat_interleave(...)).half().contiguous()`` with a
    single direct fp8->fp16 kernel so CUDA graph capture does not replay the
    intermediate fp32 copy, scale-multiply elementwise, and fp16 copy chain.
    """
    assert b.ndim in (2, 3), f"Expected b rank 2/3, got {b.ndim}"
    assert b_scale.dtype == torch.float32, (
        f"b_scale must be float32, got {b_scale.dtype}"
    )
    assert rank % 128 == 0 and hidden % 128 == 0, (
        f"rank={rank} and hidden={hidden} must be multiples of 128"
    )
    b_3d = b.reshape(groups, rank, hidden)
    scale_3d = b_scale.reshape(groups, rank // 128, hidden // 128)

    if out is None:
        out = torch.empty(
            (groups, rank, hidden), dtype=torch.float16, device=b.device
        )
    else:
        assert out.shape == (groups, rank, hidden)
        assert out.dtype == torch.float16

    if groups == 0 or rank == 0 or hidden == 0:
        return out

    if not b.is_cuda:
        b_ref = (
            b_3d.view(torch.float8_e4m3fn)
            if b_3d.dtype == torch.uint8
            else b_3d
        )
        out.copy_(
            (
                b_ref.float()
                * scale_3d.repeat_interleave(128, dim=1).repeat_interleave(
                    128, dim=2
                )
            ).half()
        )
        return out

    assert b_scale.is_cuda and out.is_cuda, "weight predequant tensors must be CUDA"
    if b_3d.dtype != torch.uint8:
        b_3d = b_3d.view(torch.uint8)

    # Keep each program inside one 128x128 scale tile. BLOCK_R=8 limits
    # per-program elements/registers on Volta while still collapsing the old
    # three eager kernels into one launch.
    BLOCK_R = 8
    BLOCK_H = 128
    grid = (groups, triton.cdiv(rank, BLOCK_R), triton.cdiv(hidden, BLOCK_H))
    _fp8_weight_predequant_to_fp16_kernel[grid](
        b_3d,
        scale_3d,
        out,
        rank,
        hidden,
        b_3d.stride(0), b_3d.stride(1), b_3d.stride(2),
        scale_3d.stride(0), scale_3d.stride(1), scale_3d.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_R=BLOCK_R,
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    return out
