# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and caching tests for the fp32-fallback path in
``cublas_gemm_bf16_bf16_fp32``.

The fallback `(x.float() @ w.float().t())` is hit on V100 (fp16
inputs, no `router_gemm_bf16_fp32` op).  H72 added a per-tensor
``_fp32_view`` cache so the upcast happens at most once per tensor,
not once per call.
"""

from __future__ import annotations

import pytest
import torch


_IS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not _IS_CUDA, reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 1024, 4096),  # decode, C128 compressor (coff=1)
        (1, 2048, 4096),  # decode, C4 compressor (coff=2)
        (4, 2048, 4096),
    ],
)
def test_fallback_matches_fp32_reference(dtype, M, N, K):
    from vllm.model_executor.layers.utils import (
        cublas_gemm_bf16_bf16_fp32,
    )

    g = torch.Generator(device="cuda").manual_seed(20260528)
    x = torch.randn(M, K, generator=g, device="cuda", dtype=dtype)
    w = torch.randn(N, K, generator=g, device="cuda", dtype=dtype) * 0.02

    out = cublas_gemm_bf16_bf16_fp32(x, w)
    ref = x.float() @ w.float().t()
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _IS_CUDA, reason="CUDA required")
def test_weight_fp32_view_is_cached():
    """Repeated calls on the same weight should not re-upcast.

    We probe the side-effect attribute set on the weight tensor; if
    the cache fires the second call's helper sees the same tensor
    pointer.
    """
    from vllm.model_executor.layers.utils import (
        cublas_gemm_bf16_bf16_fp32,
    )

    M, N, K = 1, 2048, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02

    out1 = cublas_gemm_bf16_bf16_fp32(x, w)
    cached_after_first = getattr(w, "_fp32_view", None)
    assert cached_after_first is not None
    assert cached_after_first.dtype == torch.float32
    assert cached_after_first.device == w.device
    assert cached_after_first.shape == w.shape

    out2 = cublas_gemm_bf16_bf16_fp32(x, w)
    cached_after_second = getattr(w, "_fp32_view", None)
    # Same tensor object, not re-allocated.
    assert cached_after_second is cached_after_first

    # Numerics still match.
    torch.testing.assert_close(out1, out2)
