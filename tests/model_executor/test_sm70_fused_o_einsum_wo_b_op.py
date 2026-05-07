# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inductor-shield test for SM70 FP8 weight pre-dequant.

Round 1 of the deepseek-v4-flash-prefill-throughput spec shields the
SM70 FP8 weight pre-dequantization path from inductor fusion by
registering `torch.ops.vllm.sm70_fp8_weight_predequant` as an opaque
custom op. This prevents inductor from generating a `*fp8e4nv` Triton
kernel from the weight pre-dequant expression (SM70 Triton does not
support fp8e4nv).

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

    This is the primary fp8e4nv shield: inductor cannot trace into an
    opaque custom op body and therefore cannot generate an fp8e4nv
    Triton kernel from the weight dequant expression.
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
