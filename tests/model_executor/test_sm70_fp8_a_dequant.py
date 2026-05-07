# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the SM70 fused FP8-A dequant Triton kernel."""
from __future__ import annotations

import pytest
import torch


pytest.importorskip("triton")

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FP8 dequant test",
)


def _ref_dequant(a_fp8: torch.Tensor, a_scale: torch.Tensor) -> torch.Tensor:
    """Reference: float conversion * repeat_interleave-expanded scale -> fp16."""
    T, G, D = a_fp8.shape
    block_size = D // a_scale.shape[-1]
    a_deq = a_fp8.float() * a_scale.repeat_interleave(block_size, dim=-1)
    return a_deq.half()


@cuda_required
@pytest.mark.parametrize("T,G,D,block_size", [
    (1, 1, 128, 128),
    (1, 8, 1024, 128),
    (32, 4, 512, 128),
    (7, 2, 384, 128),  # non-power-of-2 T
])
def test_sm70_fp8_a_dequant_matches_reference(T, G, D, block_size):
    from vllm.model_executor.layers.fp8_a_dequant_triton import (
        sm70_fp8_a_dequant_to_fp16,
    )

    torch.manual_seed(0)
    a_fp8 = (torch.randn(T, G, D, device="cuda") * 3.0).to(
        dtype=torch.float8_e4m3fn,
    )
    n_scales = D // block_size
    a_scale = (torch.rand(T, G, n_scales, dtype=torch.float32, device="cuda") + 0.1)

    out = sm70_fp8_a_dequant_to_fp16(a_fp8, a_scale)
    ref = _ref_dequant(a_fp8, a_scale)

    # Our kernel treats denormals as zero (DeepSeek V4 blocked-scale design
    # absorbs them); the reference uses PyTorch's full FP8 round-trip which
    # emits small non-zero values for exp=0. Mask those before comparing.
    denormal_mask = (a_fp8.view(torch.uint8) & 0x78) == 0  # exp == 0
    diff = (out - ref).abs()
    # Allow the denormal band (max ~4 * scale_max, tiny) to diverge.
    diff[denormal_mask] = 0

    # Remaining FP8 E4M3 round-trip + fp16 accumulation tolerance.
    # E4M3 has 3-bit mantissa -> ~12.5% relative; tol dominated by abs
    # at values up to ~scale_max * 448.
    scale_max = a_scale.max().item()
    abs_tol = 5e-3 * max(1.0, scale_max * 10)
    torch.testing.assert_close(diff, torch.zeros_like(diff),
                                rtol=0, atol=abs_tol)


@cuda_required
def test_sm70_fp8_a_dequant_zero_tokens():
    from vllm.model_executor.layers.fp8_a_dequant_triton import (
        sm70_fp8_a_dequant_to_fp16,
    )

    a_fp8 = torch.empty(0, 4, 512, dtype=torch.float8_e4m3fn, device="cuda")
    a_scale = torch.empty(0, 4, 4, dtype=torch.float32, device="cuda")
    out = sm70_fp8_a_dequant_to_fp16(a_fp8, a_scale)
    assert out.shape == (0, 4, 512)
