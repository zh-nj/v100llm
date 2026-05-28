# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the fused weights_proj path inside
``fused_indexer_q_rope_quant``.

Confirms that running the kernel with the new ``hidden_states`` /
``weights_proj_weight`` arguments produces the same ``index_weights_out``
as the legacy two-step path (``self.weights_proj(hidden_states)`` followed
by an unfused ``fused_indexer_q_rope_quant``)."""

from __future__ import annotations

import pytest
import torch


_IS_CUDA = torch.cuda.is_available()


def _make_inputs(num_tokens, num_heads, head_dim, hidden, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    positions = torch.randint(0, 4096, (num_tokens,), dtype=torch.int64,
                              generator=g, device=device)
    index_q = torch.randn(
        num_tokens, num_heads, head_dim, generator=g, device=device,
        dtype=torch.float16,
    )
    cos_sin = torch.randn(
        4096, head_dim, generator=g, device=device, dtype=torch.float32,
    )
    hidden_states = torch.randn(
        num_tokens, hidden, generator=g, device=device, dtype=torch.float16,
    )
    weights_proj_w = torch.randn(
        num_heads, hidden, generator=g, device=device, dtype=torch.float16,
    ) * 0.02
    return positions, index_q, cos_sin, hidden_states, weights_proj_w


@pytest.mark.skipif(not _IS_CUDA, reason="CUDA required")
@pytest.mark.parametrize("num_tokens", [1, 4, 17])
@pytest.mark.parametrize("hidden", [4096])
def test_fused_weights_proj_matches_unfused(num_tokens, hidden):
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        fused_indexer_q_rope_quant,
    )

    device = "cuda"
    num_heads = 64
    head_dim = 128
    softmax_scale = head_dim**-0.5
    head_scale = num_heads**-0.5

    positions, index_q, cos_sin, hidden_states, weights_proj_w = _make_inputs(
        num_tokens, num_heads, head_dim, hidden, device,
    )

    # Path A — legacy: do the F.linear ourselves, feed weights into the kernel.
    weights_unfused = (
        hidden_states.float() @ weights_proj_w.float().T
    ).to(torch.float16)
    q_unfused, w_out_unfused = fused_indexer_q_rope_quant(
        positions,
        index_q.clone(),
        cos_sin,
        weights_unfused,
        softmax_scale,
        head_scale,
        use_fp4=False,
    )

    # Path B — fused: pass hidden_states + weights_proj_weight; the placeholder
    # ``index_weights`` only carries shape/dtype so empty_like works.
    placeholder = torch.empty(
        (num_tokens, num_heads), dtype=torch.float16, device=device,
    )
    q_fused, w_out_fused = fused_indexer_q_rope_quant(
        positions,
        index_q.clone(),
        cos_sin,
        placeholder,
        softmax_scale,
        head_scale,
        use_fp4=False,
        hidden_states=hidden_states,
        weights_proj_weight=weights_proj_w,
    )

    # The Q quantization path is identical, so q tensors should match bit-for-bit.
    assert torch.equal(q_unfused.view(torch.int8), q_fused.view(torch.int8)), (
        "q_fp8 mismatch between fused and unfused weights_proj paths"
    )

    # weights_out: fused path replaces an upstream fp16-rounded GEMV with an
    # in-kernel fp32 GEMV before applying q_scale * softmax_scale * head_scale.
    # This removes one fp16 rounding step, so values are NOT bit-identical;
    # they should match to within ~atol of the upstream rounding.  Use a tight
    # tolerance scaled by the magnitude of the unfused weight.
    abs_diff = (w_out_fused - w_out_unfused).abs()
    rel_diff = abs_diff / w_out_unfused.abs().clamp_min(1e-3)
    assert abs_diff.max().item() < 5e-2, (
        f"max abs diff = {abs_diff.max().item():.6f}; "
        f"max rel diff = {rel_diff.max().item():.6f}"
    )
    # Mean rel diff should be tiny (fp16-vs-fp32 GEMV rounding only).
    assert rel_diff.mean().item() < 5e-3, (
        f"mean rel diff = {rel_diff.mean().item():.6f}"
    )
