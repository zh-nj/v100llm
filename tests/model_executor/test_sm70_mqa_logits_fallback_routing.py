# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for R5a — SM70 fp8 MQA logits fallback routing.

Ensures _fp8_mqa_logits_torch_fallback dispatches through the SM70
Triton kernel on Volta instead of the pure-PyTorch eager reference
implementation when VLLM_SM70_MQA_LOGITS_TRITON=1.
"""
import os
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(70)
    ),
    reason="SM70 CUDA required",
)


@pytest.fixture(autouse=True)
def _enable_triton_env(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MQA_LOGITS_TRITON", "1")


def test_fallback_routes_to_sm70_triton_kernel():
    """The _fp8_mqa_logits_torch_fallback helper must return the
    byte-identical result of the Triton kernel when called on SM70."""
    from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_mqa_logits
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _fp8_mqa_logits_torch_fallback,
    )

    torch.manual_seed(0)
    M, H, D, N = 17, 8, 64, 128
    q_fp8 = (torch.randn(M, H, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k_fp8 = (torch.randn(N, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.2 + 0.01
    weights = torch.randn(M, H, device="cuda", dtype=torch.float32) * 0.5
    cu_seqlen_ks = torch.zeros(M, device="cuda", dtype=torch.int32)
    cu_seqlen_ke = torch.full((M,), N, device="cuda", dtype=torch.int32)

    direct = sm70_fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke)
    routed = _fp8_mqa_logits_torch_fallback(
        q_fp8, (k_fp8, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke
    )
    # Byte-identical: the fallback must route to the exact same kernel call.
    assert torch.equal(direct, routed), "fallback did not reach the SM70 Triton kernel"


def test_fallback_matches_eager_fp32_reference():
    """Within the fp8 rounding envelope (atol 5e-3)."""
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _fp8_mqa_logits_torch_fallback,
    )

    torch.manual_seed(1)
    M, H, D, N = 11, 4, 64, 96
    q_fp8 = (torch.randn(M, H, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k_fp8 = (torch.randn(N, D, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    k_scale = torch.rand(N, device="cuda", dtype=torch.float32) * 0.2 + 0.01
    weights = torch.randn(M, H, device="cuda", dtype=torch.float32) * 0.5
    cu_seqlen_ks = torch.zeros(M, device="cuda", dtype=torch.int32)
    cu_seqlen_ke = torch.full((M,), N, device="cuda", dtype=torch.int32)

    got = _fp8_mqa_logits_torch_fallback(
        q_fp8, (k_fp8, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke
    )

    q_f32 = q_fp8.float()
    k_f32 = k_fp8.float() * k_scale.reshape(-1).float().view(-1, 1)
    positions = torch.arange(0, N, device=q_fp8.device)
    mask = (positions[None, :] >= cu_seqlen_ks[:, None]) & (
        positions[None, :] < cu_seqlen_ke[:, None]
    )
    score = torch.einsum("mhd,nd->hmn", q_f32, k_f32)
    ref = (score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    ref = ref.masked_fill(~mask, float("-inf"))

    torch.testing.assert_close(got, ref, atol=5e-3, rtol=1e-2)
