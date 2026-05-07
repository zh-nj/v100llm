# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the SM70 fused SwiGLU-with-limit Triton kernel."""
from __future__ import annotations

import pytest
import torch


pytest.importorskip("triton")

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for SM70 fused SwiGLU test",
)


def _ref_swiglu_limit(input_tensor: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    d = input_tensor.shape[-1] // 2
    gate = input_tensor[:, :d]
    up = input_tensor[:, d:]
    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    return torch.nn.functional.silu(gate) * up


@cuda_required
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0])
@pytest.mark.parametrize("m,d", [(1, 256), (32, 1024), (63, 2048), (4 * 7, 2048)])
def test_sm70_fused_swiglu_limit_matches_reference(swiglu_limit, m, d):
    from vllm.model_executor.layers.fused_moe.swiglu_limit_triton import (
        sm70_fused_swiglu_limit,
    )

    torch.manual_seed(0)
    x = (torch.randn(m, 2 * d, dtype=torch.float16, device="cuda") * 4.0)
    # Include some values outside [-limit, limit] to exercise clamp.
    x[0, :10] = 15.0
    x[0, d : d + 10] = -15.0
    out = torch.zeros(m, d, dtype=torch.float16, device="cuda")

    sm70_fused_swiglu_limit(out, x, swiglu_limit)

    ref = _ref_swiglu_limit(x, swiglu_limit)
    # FP16 round-trip: allow up to 2e-3 abs diff; silu at ~7 can reach ~7.
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@cuda_required
def test_sm70_fused_swiglu_limit_zero_tokens_noop():
    from vllm.model_executor.layers.fused_moe.swiglu_limit_triton import (
        sm70_fused_swiglu_limit,
    )

    x = torch.empty(0, 512, dtype=torch.float16, device="cuda")
    out = torch.empty(0, 256, dtype=torch.float16, device="cuda")
    sm70_fused_swiglu_limit(out, x, 7.0)  # must not raise
