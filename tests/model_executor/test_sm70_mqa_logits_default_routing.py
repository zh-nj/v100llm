# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only routing tests for SM70 sparse indexer logits."""

import torch


def _tiny_inputs():
    q = torch.ones(1, 1, 1)
    k = torch.ones(1, 1)
    k_scale = torch.ones(1)
    weights = torch.ones(1, 1)
    cu_seqlen_ks = torch.zeros(1, dtype=torch.int32)
    cu_seqlen_ke = torch.ones(1, dtype=torch.int32)
    return q, (k, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke


def _enable_sm70_fallback(monkeypatch):
    from vllm.model_executor.layers import sparse_attn_indexer

    monkeypatch.delenv("VLLM_SM70_MQA_LOGITS_IMPL", raising=False)
    monkeypatch.delenv("VLLM_SM70_MQA_LOGITS_TRITON", raising=False)
    monkeypatch.setattr(
        sparse_attn_indexer,
        "_can_use_sm70_torch_indexer_fallback",
        lambda use_fp4_cache: True,
    )
    return sparse_attn_indexer


def test_sm70_gemm_is_default_without_env(monkeypatch):
    """SM70 prefill must avoid both fp32 score materialization and scalar Triton by default."""
    sparse_attn_indexer = _enable_sm70_fallback(monkeypatch)
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _fp8_mqa_logits_torch_fallback,
    )

    gemm_sentinel = torch.empty(0)
    triton_sentinel = torch.empty(1)

    def fake_gemm_logits(*args, **kwargs):
        return gemm_sentinel

    def fake_triton_logits(*args, **kwargs):
        return triton_sentinel

    monkeypatch.setattr(
        sparse_attn_indexer,
        "sm70_fp8_mqa_logits_gemm",
        fake_gemm_logits,
        raising=False,
    )
    monkeypatch.setattr(
        sparse_attn_indexer, "sm70_fp8_mqa_logits", fake_triton_logits
    )

    got = _fp8_mqa_logits_torch_fallback(*_tiny_inputs())

    assert got is gemm_sentinel


def test_sm70_triton_impl_routes_to_scalar_kernel(monkeypatch):
    sparse_attn_indexer = _enable_sm70_fallback(monkeypatch)
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _fp8_mqa_logits_torch_fallback,
    )

    monkeypatch.setenv("VLLM_SM70_MQA_LOGITS_IMPL", "triton")
    sentinel = torch.empty(0)

    def fake_sm70_logits(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(
        sparse_attn_indexer, "sm70_fp8_mqa_logits", fake_sm70_logits
    )

    got = _fp8_mqa_logits_torch_fallback(*_tiny_inputs())

    assert got is sentinel


def test_sm70_torch_impl_routes_to_reference(monkeypatch):
    sparse_attn_indexer = _enable_sm70_fallback(monkeypatch)
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _fp8_mqa_logits_torch_fallback,
    )

    monkeypatch.setenv("VLLM_SM70_MQA_LOGITS_IMPL", "torch")

    def fail_sm70_logits(*args, **kwargs):
        raise AssertionError("SM70 kernel should not be used for impl=torch")

    monkeypatch.setattr(
        sparse_attn_indexer,
        "sm70_fp8_mqa_logits_gemm",
        fail_sm70_logits,
        raising=False,
    )
    monkeypatch.setattr(
        sparse_attn_indexer, "sm70_fp8_mqa_logits", fail_sm70_logits
    )

    got = _fp8_mqa_logits_torch_fallback(*_tiny_inputs())

    torch.testing.assert_close(got, torch.ones(1, 1))
