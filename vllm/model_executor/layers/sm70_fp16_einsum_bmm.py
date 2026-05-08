# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 FP16 einsum BMM kernel for DeepSeek V4 ``bhr,hdr->bhd``.

Replaces ``torch.einsum("bhr,hdr->bhd", a_fp16, b_f16)`` in the SM70
software-FP8 o-projection path with a single Triton kernel.  On Volta,
PyTorch dispatches this einsum through multiple reshape+GEMM kernels
with non-trivial launch / layout overhead (~3.2% of total prefill time
per R4 profile).  This kernel folds the whole thing into one launch.

Shapes (production TP=8 deepseek-v4-flash):
    a_fp16: [T, G, R] fp16 (T = num_tokens, G = 1, R = 4096)
    b_f16:  [G, D, R] fp16 (D = 1024 = o_lora_rank, R = 4096)
    out:    [T, G, D] fp16

General form: ``out[t, g, d] = sum_r a[t, g, r] * b[g, d, r]``.

Internally this is a batched GEMM: for each g, ``a[:, g, :] @ b[g, :, :].T``
is a (T, R) x (R, D) fp16 matmul with fp32 accumulate, using Triton
``tl.dot`` which on SM70 issues Volta mma.sync (m8n8k4, fp16 x fp16 -> fp32).

Guaranteed SM70-safe: fp16 operands only, no bf16, no fp8e4nv.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _sm70_fp16_einsum_bhr_hdr_bhd_kernel(
    a_ptr,            # fp16, [T, G, R]
    b_ptr,            # fp16, [G, D, R]
    out_ptr,          # fp16, [T, G, D]
    stride_at, stride_ag, stride_ar,   # strides of a
    stride_bg, stride_bd, stride_br,   # strides of b
    stride_ot, stride_og, stride_od,   # strides of out
    T, D, R,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_G: tl.constexpr,
):
    """One program computes an (BLOCK_T x BLOCK_D) tile of one group.

    Grid: (cdiv(T, BLOCK_T), cdiv(D, BLOCK_D), NUM_G).
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    g = tl.program_id(2)

    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < D

    # Block pointers into a[t, g, :] and b[g, d, :], both contiguous along R.
    a_row_base = a_ptr + t_off[:, None] * stride_at + g * stride_ag
    b_row_base = b_ptr + g * stride_bg + d_off[:, None] * stride_bd

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    for r_start in range(0, R, BLOCK_R):
        r_off = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_off < R
        # a_tile [BLOCK_T, BLOCK_R]
        a_ptrs = a_row_base + r_off[None, :] * stride_ar
        a_tile = tl.load(
            a_ptrs,
            mask=t_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        # b_tile [BLOCK_D, BLOCK_R]
        b_ptrs = b_row_base + r_off[None, :] * stride_br
        b_tile = tl.load(
            b_ptrs,
            mask=d_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        # Matmul: (BLOCK_T, R) x (R, BLOCK_D) via transposed b_tile.
        # Note: allow_tf32=False keeps strict fp16 accumulation on Volta
        # (which has no TF32 anyway; left explicit for safety).
        acc += tl.dot(a_tile, tl.trans(b_tile), allow_tf32=False)

    out_ptrs = (
        out_ptr
        + t_off[:, None] * stride_ot
        + g * stride_og
        + d_off[None, :] * stride_od
    )
    tl.store(out_ptrs, acc.to(tl.float16), mask=t_mask[:, None] & d_mask[None, :])


def _sm70_fp16_einsum_bhr_hdr_bhd_impl(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute ``out[t, g, d] = sum_r a[t, g, r] * b[g, d, r]`` in fp16.

    Args:
        a: fp16 tensor, shape ``[T, G, R]``.
        b: fp16 tensor, shape ``[G, D, R]``.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    assert a.ndim == 3 and b.ndim == 3
    T, G, R = a.shape
    Gb, D, Rb = b.shape
    assert G == Gb, f"group mismatch: a[..., {G}, ...] vs b[{Gb}, ...]"
    assert R == Rb, f"contraction mismatch: a[..., {R}] vs b[..., {Rb}]"

    out = torch.empty((T, G, D), dtype=torch.float16, device=a.device)
    if T == 0:
        return out

    # Tile sizing chosen for V100 / CUDA-core fallback:
    # - BLOCK_T=32 keeps the query-side warp tile inside 16x16 mma requirement;
    # - BLOCK_D=64 matches two consecutive mma tiles on the output side;
    # - BLOCK_R=32 is small enough that (BLOCK_T+BLOCK_D)*BLOCK_R = 48KB/4
    #   stays well inside V100's 96KB smem budget per SM.
    # tl.dot on SM70 requires operands >= 16x16 (m16n16k16), so these are
    # legal; the launcher clamps when T or D is smaller.
    BLOCK_T = 32 if T >= 32 else 16
    BLOCK_D = 64 if D >= 64 else (32 if D >= 32 else 16)
    BLOCK_R = 32

    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D), G)
    _sm70_fp16_einsum_bhr_hdr_bhd_kernel[grid](
        a, b, out,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        T, D, R,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
        NUM_G=G,
        num_warps=4,
        num_stages=2,
    )
    return out


def _sm70_fp16_einsum_bhr_hdr_bhd_fake(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    T, G, _ = a.shape
    _, D, _ = b.shape
    return torch.empty((T, G, D), dtype=torch.float16, device=a.device)


try:
    direct_register_custom_op(
        op_name="sm70_fp16_einsum_bhr_hdr_bhd",
        op_func=_sm70_fp16_einsum_bhr_hdr_bhd_impl,
        mutates_args=[],
        fake_impl=_sm70_fp16_einsum_bhr_hdr_bhd_fake,
    )
    # Export the registered custom op directly — it is AOTAutogradCache-safe
    # (dynamo lowers it to a ``call_function`` node against a known torch op).
    sm70_fp16_einsum_bhr_hdr_bhd = torch.ops.vllm.sm70_fp16_einsum_bhr_hdr_bhd
except (RuntimeError, AttributeError):
    # Fallback (e.g. in tests that stub the custom-op registry): call the
    # bare impl. Inductor will not see this path under the shielded
    # `_sm70_fused_o_einsum_wo_b` wrapper.
    sm70_fp16_einsum_bhr_hdr_bhd = _sm70_fp16_einsum_bhr_hdr_bhd_impl
