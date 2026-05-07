# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SwiGLU-with-limit Triton kernel for SM70.

Replaces the `torch.compile`-based `swiglu_limit_func` on V100 where
`simple_compile_backend='inductor'` yields a 5-op sequence (clamp x2 +
silu + mul + copy_) with measurable per-token overhead. Profiling showed
685 us/call avg for a 43-layer MoE decode step, dominating
`decoder.moe.experts` (86.9% of the scope) at 471 ms aggregate over
688 calls.

The fused kernel does:
    gate = clamp(input[:, :d], max=swiglu_limit) if swiglu_limit > 0 else input[:, :d]
    up   = clamp(input[:, d:], min=-swiglu_limit, max=swiglu_limit) if swiglu_limit > 0 else input[:, d:]
    output[:, :d] = silu(gate) * up

in a single pass with one HBM read of `input` and one HBM write of
`output`, eliminating the intermediate materializations of
`F.silu(gate)` and `gate * up` that torch.compile emits.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_swiglu_limit_kernel(
    out_ptr,
    in_ptr,
    out_stride_m,
    in_stride_m,
    M,
    D,
    swiglu_limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute output[:, :d] = silu(clamp(gate)) * clamp(up).

    gate = input[:, :d]; up = input[:, d:].
    Grid: (ceil(M / BLOCK_M), ceil(D / BLOCK_D)).
    Output and input must both be FP16. Output's column-dim stride must be 1.
    """
    pid_m = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = m_offsets < M
    d_mask = d_offsets < D

    # Gate pointer: input[m, :d] lane d_offsets
    gate_ptrs = in_ptr + m_offsets[:, None] * in_stride_m + d_offsets[None, :]
    up_ptrs = in_ptr + m_offsets[:, None] * in_stride_m + (D + d_offsets[None, :])

    full_mask = m_mask[:, None] & d_mask[None, :]
    gate = tl.load(gate_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=full_mask, other=0.0).to(tl.float32)

    if HAS_LIMIT:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.maximum(tl.minimum(up, swiglu_limit), -swiglu_limit)

    # silu(gate) = gate * sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    out_ptrs = out_ptr + m_offsets[:, None] * out_stride_m + d_offsets[None, :]
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=full_mask)


def sm70_fused_swiglu_limit(
    output: torch.Tensor,
    input: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> None:
    """Fused SwiGLU-with-limit for SM70. Writes result into ``output`` in place.

    Args:
        output: FP16 [M, D] tensor. Overwritten.
        input: FP16 [M, 2*D] tensor, first half gate, second half up.
        swiglu_limit: float; if > 0, clamp gate/up to |swiglu_limit|.
    """
    assert input.is_contiguous() or input.stride(-1) == 1, (
        "sm70_fused_swiglu_limit expects the last dim of `input` to be "
        "contiguous (stride(-1) == 1)."
    )
    assert output.stride(-1) == 1, (
        "sm70_fused_swiglu_limit expects the last dim of `output` to be "
        "contiguous (stride(-1) == 1)."
    )
    assert input.dtype == output.dtype == torch.float16, (
        "sm70_fused_swiglu_limit only supports FP16; got "
        f"input={input.dtype}, output={output.dtype}."
    )
    two_d = input.shape[-1]
    d = output.shape[-1]
    assert two_d == 2 * d, (
        f"sm70_fused_swiglu_limit: expected input shape [M, 2*D]={[input.shape[0], 2 * d]}, "
        f"got {list(input.shape)} with D={d}."
    )
    M = input.shape[0]
    if M == 0 or d == 0:
        return

    BLOCK_M = 16
    # Pick BLOCK_D up to 256 aligned to D or a divisor of D for correctness when
    # D is not a multiple of 256; we rely on the mask in the kernel otherwise.
    BLOCK_D = 256 if d >= 256 else triton.next_power_of_2(max(d, 1))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(d, BLOCK_D))
    _fused_swiglu_limit_kernel[grid](
        output,
        input,
        output.stride(0),
        input.stride(0),
        M,
        d,
        float(swiglu_limit),
        HAS_LIMIT=(swiglu_limit > 0.0),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
