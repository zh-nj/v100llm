# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inductor-shield and direct-dequant tests for SM70 FP8 weight pre-dequant.

Round 1 of the deepseek-v4-flash-prefill-throughput spec shields the
SM70 FP8 weight pre-dequantization path from inductor fusion by
registering `torch.ops.vllm.sm70_fp8_weight_predequant` as an opaque
custom op. Its body now delegates to a direct fp8->fp16 Triton kernel instead
of the old `b.float() * repeat_interleave(scale) -> half()` chain.

Additionally `_sm70_fp8_einsum_bmm` and
`_deepseek_v4_fp8_einsum_torch_fallback` are decorated with
`@torch._dynamo.allow_in_graph` so dynamo treats them as opaque
call_function nodes when traced from a compiled region. Both still
delegate the actual fp8-weight dequant work to the shielded custom op.
"""
from __future__ import annotations

import pytest
import torch


def _is_dynamo_shielded(fn) -> bool:
    """Check if a function is shielded from dynamo tracing (either via
    `torch._dynamo.disable` or `torch._dynamo.allow_in_graph`).
    """
    if fn is None:
        return False
    try:
        from torch._dynamo.trace_rules import is_callable_allowed
        if is_callable_allowed(fn):
            return True
    except Exception:
        pass
    for attr in (
        "_torchdynamo_disable",
        "_torchdynamo_inline",
        "__dynamo_disable_wrap__",
        "_dynamo_disable",
    ):
        if getattr(fn, attr, None):
            return True
    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None and wrapped is not fn:
        return True
    return False


def test_sm70_fp8_weight_predequant_op_registered():
    """The shielding custom op MUST be registered as `torch.ops.vllm.sm70_fp8_weight_predequant`.

    This keeps the weight-cache population opaque to Inductor and gives the
    body a stable place to launch the direct fp8->fp16 predequant kernel.
    """
    # Force the module to import so the custom op gets registered.
    pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    op = getattr(torch.ops.vllm, "sm70_fp8_weight_predequant", None)
    assert op is not None, (
        "torch.ops.vllm.sm70_fp8_weight_predequant must be registered "
        "to shield the fp8 weight pre-dequant expression from inductor "
        "fusion. Without this, SM70 builds with FULL_AND_PIECEWISE "
        "cudagraph mode crash with 'type fp8e4nv not supported'."
    )


def test_sm70_fp8_einsum_bmm_is_dynamo_shielded():
    """_sm70_fp8_einsum_bmm should be allow_in_graph'd (tensor-only args)."""
    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    helper = getattr(dsa, "_sm70_fp8_einsum_bmm", None)
    assert helper is not None
    assert _is_dynamo_shielded(helper), (
        "_sm70_fp8_einsum_bmm must be @torch._dynamo.allow_in_graph'd so "
        "dynamo treats it as opaque when traced from a compiled region."
    )


def test_deepseek_v4_fp8_einsum_torch_fallback_is_dynamo_shielded():
    """_deepseek_v4_fp8_einsum_torch_fallback should also be shielded."""
    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    helper = getattr(dsa, "_deepseek_v4_fp8_einsum_torch_fallback", None)
    assert helper is not None
    assert _is_dynamo_shielded(helper), (
        "_deepseek_v4_fp8_einsum_torch_fallback must be "
        "@torch._dynamo.allow_in_graph'd for the same fp8e4nv shield reason."
    )


def test_sm70_einsum_bmm_triton_defaults_off(monkeypatch):
    """R5b Triton BMM should remain opt-in after the decode regression."""
    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    helper = getattr(dsa, "_sm70_einsum_bmm_triton_enabled", None)
    assert helper is not None

    monkeypatch.delenv("VLLM_SM70_EINSUM_BMM_TRITON", raising=False)
    assert helper() is False

    monkeypatch.setenv("VLLM_SM70_EINSUM_BMM_TRITON", "1")
    assert helper() is True


def test_sm70_predequant_helper_uses_custom_op():
    """The helper that populates `_sm70_predequant_f16` MUST call the custom op."""
    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    helper = getattr(dsa, "_sm70_ensure_predequant_weight", None)
    assert helper is not None, (
        "_sm70_ensure_predequant_weight must exist; it is the single "
        "entry point that routes weight pre-dequant through "
        "torch.ops.vllm.sm70_fp8_weight_predequant."
    )


def test_sm70_weight_predequant_impl_uses_direct_helper(monkeypatch):
    """The custom-op body should avoid the old fp32 weight staging chain."""
    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )

    calls: list[tuple[torch.Tensor, torch.Tensor, int, int, int]] = []

    def fake_direct_predequant(
        b: torch.Tensor,
        b_scale: torch.Tensor,
        groups: int,
        rank: int,
        hidden: int,
    ) -> torch.Tensor:
        calls.append((b, b_scale, groups, rank, hidden))
        return torch.full(
            (groups, rank, hidden),
            3.0,
            dtype=torch.float16,
            device=b.device,
        )

    monkeypatch.setattr(
        dsa,
        "sm70_fp8_weight_predequant_to_fp16",
        fake_direct_predequant,
        raising=False,
    )

    b = torch.ones((128, 128), dtype=torch.uint8)
    b_scale = torch.ones((1, 1, 1), dtype=torch.float32)
    out = dsa._sm70_fp8_weight_predequant_impl(
        b,
        b_scale,
        groups=1,
        rank=128,
        hidden=128,
    )

    assert calls == [(b, b_scale, 1, 128, 128)]
    torch.testing.assert_close(
        out,
        torch.full((1, 128, 128), 3.0, dtype=torch.float16),
        rtol=0,
        atol=0,
    )


def test_sm70_fp8_einsum_bmm_uses_fused_activation_dequant(monkeypatch):
    """The non-fused SM70 O-einsum fallback must avoid fp32 activation staging.

    The fused O-einsum+wo_b path already calls `sm70_fp8_a_dequant_to_fp16`.
    Keep `_sm70_fp8_einsum_bmm` on the same semantic path so any caller of
    `deepseek_v4_fp8_einsum` on SM70 avoids the old
    `a.float() * repeat_interleave(scale) -> half()` round-trip.
    """
    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_dequant(a: torch.Tensor, a_scale: torch.Tensor) -> torch.Tensor:
        calls.append((a, a_scale))
        return torch.ones(a.shape, dtype=torch.float16)

    def fake_predequant_weight(
        b: torch.Tensor,
        b_scale: torch.Tensor,
        groups: int,
        rank: int,
        hidden: int,
    ) -> torch.Tensor:
        del b, b_scale
        return torch.ones((groups, rank, hidden), dtype=torch.float16)

    monkeypatch.setattr(dsa, "sm70_fp8_a_dequant_to_fp16", fake_dequant)
    monkeypatch.setattr(
        dsa, "_sm70_ensure_predequant_weight", fake_predequant_weight
    )
    monkeypatch.setattr(dsa, "_sm70_einsum_bmm_triton_enabled", lambda: False)

    a = torch.ones((1, 2, 4), dtype=torch.uint8)
    a_scale = torch.ones((1, 2, 2), dtype=torch.float32)
    b = torch.ones((2, 3, 4), dtype=torch.uint8)
    b_scale = torch.ones((2, 1, 1), dtype=torch.float32)
    out = torch.empty((1, 2, 3), dtype=torch.float16)

    dsa._sm70_fp8_einsum_bmm(a, a_scale, b, b_scale, out, "bhr,hdr->bhd")

    assert calls == [(a, a_scale)]
    torch.testing.assert_close(
        out,
        torch.full_like(out, 4.0),
        rtol=0,
        atol=0,
    )
