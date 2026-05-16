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


def test_fused_inv_rope_fp8_quant_fake_preserves_production_strides():
    """The custom-op fake layout feeds stride constants to Inductor.

    If the fake scale output is contiguous, compiled SM70 dequant bakes
    ``scale_stride_k=1`` and reproduces the old garbage-output bug even though
    eager execution passes the correct dynamic stride.
    """
    from vllm.utils.deep_gemm import get_tma_aligned_size
    from vllm.v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant import (
        _fused_inv_rope_fp8_quant_fake,
    )

    T = 7
    n_groups = 2
    heads_per_group = 3
    head_dim = 512
    D = heads_per_group * head_dim
    quant_group_size = 128
    num_scale_blocks = D // quant_group_size
    tma_aligned_T = get_tma_aligned_size(T, 4)

    o = torch.empty(T, n_groups * heads_per_group, head_dim, dtype=torch.float16)
    positions = torch.empty(T, dtype=torch.int64)
    cos_sin_cache = torch.empty(32, 64, dtype=torch.float32)

    fp8_out, scale_out = _fused_inv_rope_fp8_quant_fake(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        448,
        64,
        quant_group_size,
        False,
    )

    assert fp8_out.shape == (T, n_groups, D)
    assert fp8_out.stride() == (D, T * D, 1)
    assert scale_out.shape == (T, n_groups, num_scale_blocks)
    assert scale_out.stride() == (1, num_scale_blocks * tma_aligned_T,
                                  tma_aligned_T)


@cuda_required
def test_sm70_fp8_a_dequant_matches_torch_e4m3fn_byte_semantics():
    """The fused path must preserve PyTorch E4M3FN decode semantics.

    This covers the subnormal byte band as well as the two NaN encodings. The
    O-einsum fallback uses this helper as a drop-in replacement for
    ``a.float().half()`` after scaling, so byte-level decode drift would be a
    real semantic change rather than just an implementation detail.
    """
    from vllm.model_executor.layers.fp8_a_dequant_triton import (
        sm70_fp8_a_dequant_to_fp16,
    )

    a_u8 = torch.arange(0, 256, dtype=torch.uint8, device="cuda").view(
        1, 1, 256
    )
    a_scale = torch.ones(1, 1, 1, dtype=torch.float32, device="cuda")

    out = sm70_fp8_a_dequant_to_fp16(a_u8, a_scale)
    ref = a_u8.view(torch.float8_e4m3fn).float().half()

    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)


@cuda_required
def test_sm70_fp8_weight_predequant_matches_torch_reference():
    """Weight pre-dequant must match the old fp32 expression exactly."""
    from vllm.model_executor.layers.fp8_a_dequant_triton import (
        sm70_fp8_weight_predequant_to_fp16,
    )

    torch.manual_seed(2)
    groups, rank, hidden = 2, 256, 256
    bytes_ = torch.arange(0, 256, dtype=torch.uint8, device="cuda")
    # Avoid NaN inputs; production quantizers should not emit them and the old
    # fp32 path would propagate NaNs through the weight cache.
    bytes_[0x7F] = 0x7E
    bytes_[0xFF] = 0xFE
    b_u8 = bytes_.repeat((groups * rank * hidden + 255) // 256)[
        : groups * rank * hidden
    ].view(groups, rank, hidden)
    b = b_u8.view(torch.float8_e4m3fn)
    b_scale = (
        torch.rand(
            groups,
            rank // 128,
            hidden // 128,
            dtype=torch.float32,
            device="cuda",
        )
        + 0.1
    )

    out = sm70_fp8_weight_predequant_to_fp16(
        b.reshape(groups * rank, hidden),
        b_scale,
        groups,
        rank,
        hidden,
    )
    ref = (
        b.float()
        * b_scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    ).half().contiguous()

    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)


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

    diff = (out - ref).abs()

    # Remaining FP8 E4M3 round-trip + fp16 accumulation tolerance.
    # E4M3 has 3-bit mantissa -> ~12.5% relative; tol dominated by abs
    # at values up to ~scale_max * 448.
    scale_max = a_scale.max().item()
    abs_tol = 5e-3 * max(1.0, scale_max * 10)
    torch.testing.assert_close(diff, torch.zeros_like(diff),
                                rtol=0, atol=abs_tol)


@cuda_required
@pytest.mark.parametrize("T,G,D,block_size", [
    (32, 4, 512, 128),
    (7, 2, 384, 128),
])
def test_sm70_fp8_a_dequant_handles_non_contiguous_scale(T, G, D, block_size):
    """Regression: production scale tensors from fused_inv_rope_fp8_quant
    are built via ``as_strided`` with stride(-1) != 1. The kernel MUST
    read ``a_scale.stride(-1)`` explicitly instead of assuming 1.

    See `.kiro/specs/deepseek-v4-flash-prefill-throughput/` for the full
    bug story; this test reproduces the layout that caused silent
    garbage output on the DeepSeek V4 SM70 fused O einsum + wo_b path.
    """
    from vllm.model_executor.layers.fp8_a_dequant_triton import (
        sm70_fp8_a_dequant_to_fp16,
    )

    torch.manual_seed(1)
    device = "cuda"
    n_scale = D // block_size

    # Match production: fp8_buf [G, T, D] then transposed -> [T, G, D]
    fp8_g_major = (torch.randn(G, T, D, device=device) * 3.0).to(
        dtype=torch.float8_e4m3fn
    )
    a = fp8_g_major.transpose(0, 1)

    # scale_buf via as_strided: [G, T, n_scale] with stride(-1) = tma_aligned_T
    # (simulate tma_aligned_T = T padded)
    tma_aligned_T = max(T, 4)
    scale_numel = G * n_scale * tma_aligned_T
    scale_raw = (torch.rand(scale_numel, device=device) + 0.1).to(torch.float32)
    scale_g_major = scale_raw.as_strided(
        (G, T, n_scale),
        (n_scale * tma_aligned_T, 1, tma_aligned_T),
    )
    a_scale = scale_g_major.transpose(0, 1)
    assert a_scale.stride(-1) != 1, (
        f"test precondition: scale should be non-contiguous on dim -1, "
        f"got stride={a_scale.stride()}"
    )

    out = sm70_fp8_a_dequant_to_fp16(a, a_scale)
    ref = _ref_dequant(a, a_scale)
    diff = (out - ref).abs()
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
