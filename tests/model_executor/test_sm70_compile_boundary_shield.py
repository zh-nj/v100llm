# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SM70 compile-path dtype boundary shield.

Backs the bugfix spec
`.kiro/specs/deepseek-v4-flash-compile-path-regression/`.

- Task 1: prove the bug exists (F.linear(fp32, fp16) raises).
- Task 2: validate the shield helper CP1 (no-op on legal input),
  CP2 (semantics preserved), CP3 (idempotent).
"""
from __future__ import annotations

import pytest
import torch


# -- Task 1 — Bug exploration property test ----------------------------------


def test_bug_exploration_fp32_hidden_into_fp16_linear_raises():
    """Reproduces the failure mode exposed by the inductor graph
    partitioner: fp32 hidden_states arriving at an fp16 Linear.

    This test is expected to FAIL on unfixed code (i.e. the error
    is raised), confirming the bug condition C1. A PBT-style
    property framing — if this assertion no longer triggers, either
    the bug has been inadvertently fixed upstream, or the bug
    condition has changed shape.
    """
    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    weight_fp16 = torch.randn(32, 64, dtype=torch.float16)

    with pytest.raises(RuntimeError, match=r"m(at)?1 and m(at)?2"):
        torch.nn.functional.linear(hidden_fp32, weight_fp16)


# -- Task 2 — ensure_boundary_dtype helper properties ------------------------


from vllm.model_executor.layers.sm70_compile_boundary_shield import (  # noqa: E402
    ensure_boundary_dtype,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_cp1_noop_on_matching_dtype(dtype):
    """CP1: shield returns the same tensor object when dtype already matches."""
    x = torch.randn(4, 8, dtype=dtype)
    y = ensure_boundary_dtype(x, dtype)
    assert y is x, "shield must be a no-op when dtype already matches"


def test_cp2_semantics_preserved_up_to_rounding():
    """CP2: shield's output matches an explicit .to() cast."""
    x = torch.randn(8, 64, dtype=torch.float32) * 100.0  # non-trivial range
    y = ensure_boundary_dtype(x, torch.float16)
    ref = x.to(torch.float16)
    assert torch.equal(y, ref), "shield must match torch.Tensor.to() exactly"


def test_cp3_idempotent():
    """CP3: applying the shield twice is the same as once."""
    x = torch.randn(4, 8, dtype=torch.float32)
    once = ensure_boundary_dtype(x, torch.float16)
    twice = ensure_boundary_dtype(once, torch.float16)
    assert twice is once, "second shield pass must be a no-op"


def test_shield_preserves_device():
    """Shield does not move tensors across devices."""
    x = torch.randn(4, 8, dtype=torch.float32)
    y = ensure_boundary_dtype(x, torch.float16)
    assert y.device == x.device


def test_shield_handles_f16_into_f16():
    """The common hot path on SM70 — fp16 hidden, fp16 expected."""
    x = torch.randn(1792, 4096, dtype=torch.float16)
    y = ensure_boundary_dtype(x, torch.float16)
    assert y is x


def test_shield_respects_env_off():
    """CP4: When VLLM_SM70_COMPILE_BOUNDARY_SHIELD=0 the helper is bypassed.

    Verifies that the env flag actually controls the shield so we can
    re-expose the bug for preservation testing.
    """
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        x = torch.randn(4, 8, dtype=torch.float32)
        y = ensure_boundary_dtype(x, torch.float16)
        # Bypass means return the input unchanged even when dtypes differ.
        assert y is x, "shield must be a no-op when env flag is 0"


def test_cp4_bug_reproduces_when_shield_disabled():
    """When the shield is off, the simulated failure path still raises."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
        weight_fp16 = torch.randn(32, 64, dtype=torch.float16)
        guarded = ensure_boundary_dtype(hidden_fp32, torch.float16)
        # Shield was bypassed, so dtype still wrong.
        assert guarded.dtype == torch.float32
        with pytest.raises(RuntimeError, match=r"m(at)?1 and m(at)?2"):
            torch.nn.functional.linear(guarded, weight_fp16)


def test_cp4_bug_masked_when_shield_enabled():
    """With the shield on, the same call succeeds."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
        weight_fp16 = torch.randn(32, 64, dtype=torch.float16)
        guarded = ensure_boundary_dtype(hidden_fp32, torch.float16)
        assert guarded.dtype == torch.float16
        out = torch.nn.functional.linear(guarded, weight_fp16)
        assert out.shape == (8, 32)
        assert out.dtype == torch.float16


def test_moe_forward_shared_fake_uses_model_dtype(monkeypatch):
    """MoE fake impl must mirror the real model-dtype output contract.

    Inductor may feed the custom op the fp32 RMSNorm intermediate, but the
    SM70 DeepSeek V4 MoE implementation returns fp16 buffers. If the fake impl
    keeps the fp32 input dtype, the following pointwise/all-reduce partition is
    generated with fp32 element width and can overrun the real fp16 output.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    fake_layer = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=torch.float16),
    )
    fake_context = SimpleNamespace(
        no_compile_layers={"model.layers.0.ffn.experts": fake_layer},
    )
    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        default_moe_runner, "get_forward_context", lambda: fake_context
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
        hidden_fp32,
        hidden_fp32,
        shared_fp32,
        "model.layers.0.ffn.experts",
        None,
    )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16
    assert fused_out.shape == hidden_fp32.shape
    assert shared_out.shape == shared_fp32.shape


def test_moe_forward_shared_fake_unwraps_modular_quant_method(monkeypatch):
    """The static layer may hold FusedMoEModularMethod around the SM70 method."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    fake_layer = SimpleNamespace(
        quant_method=SimpleNamespace(
            old_quant_method=SimpleNamespace(
                compile_boundary_output_dtype=torch.float16,
            )
        ),
        moe_config=SimpleNamespace(in_dtype=torch.float32),
    )
    fake_context = SimpleNamespace(
        no_compile_layers={"model.layers.0.ffn.experts": fake_layer},
    )
    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        default_moe_runner, "get_forward_context", lambda: fake_context
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
        hidden_fp32,
        hidden_fp32,
        shared_fp32,
        "model.layers.0.ffn.experts",
        None,
    )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16


def test_moe_forward_shared_fake_checks_runner_quant_method(monkeypatch):
    """Compile-time static layers can expose the MoE method through runner."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    fake_layer = SimpleNamespace(
        runner=SimpleNamespace(
            quant_method=SimpleNamespace(
                old_quant_method=SimpleNamespace(
                    compile_boundary_output_dtype=torch.float16,
                )
            )
        ),
        moe_config=SimpleNamespace(in_dtype=torch.float32),
    )
    fake_context = SimpleNamespace(
        no_compile_layers={"model.layers.0.ffn.experts": fake_layer},
    )
    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        default_moe_runner, "get_forward_context", lambda: fake_context
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
        hidden_fp32,
        hidden_fp32,
        shared_fp32,
        "model.layers.0.ffn.experts",
        None,
    )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16


def test_moe_forward_shared_fake_shields_layer_fp32_contract(monkeypatch):
    """An fp32 layer contract at compile time is treated as boundary leakage."""
    import os
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    fake_layer = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=torch.float32),
    )
    fake_context = SimpleNamespace(
        no_compile_layers={"model.layers.0.ffn.experts": fake_layer},
    )
    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        default_moe_runner, "get_forward_context", lambda: fake_context
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
            hidden_fp32,
            hidden_fp32,
            shared_fp32,
            "model.layers.0.ffn.experts",
            None,
        )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16


def test_moe_forward_shared_fake_peeks_forward_context_without_advancing(
    monkeypatch,
):
    """The fake impl may need the static MoE layer list without consuming it."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    fake_layer = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=torch.float16),
    )
    fake_context = SimpleNamespace(
        all_moe_layers=["model.layers.0.ffn.experts"],
        moe_layer_index=0,
        no_compile_layers={"model.layers.0.ffn.experts": fake_layer},
    )
    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        default_moe_runner, "get_forward_context", lambda: fake_context
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
        hidden_fp32,
        hidden_fp32,
        shared_fp32,
        "from_forward_context",
        None,
    )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16
    assert fake_context.moe_layer_index == 0


def test_moe_forward_shared_fake_unresolved_context_uses_shield_dtype(
    monkeypatch,
):
    """AOT fake dispatch can run after forward context has been cleared."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: False
    )
    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
            hidden_fp32,
            hidden_fp32,
            shared_fp32,
            "model.layers.0.ffn.experts",
            None,
        )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16


def test_moe_forward_shared_fake_unresolved_context_preserves_bug_when_disabled(
    monkeypatch,
):
    """The shield env var remains a reproduction switch for the old contract."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    monkeypatch.setattr(
        default_moe_runner, "is_forward_context_available", lambda: False
    )
    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        shared_out, fused_out = default_moe_runner._moe_forward_shared_fake(
            hidden_fp32,
            hidden_fp32,
            shared_fp32,
            "model.layers.0.ffn.experts",
            None,
        )

    assert fused_out.dtype == torch.float32
    assert shared_out.dtype == torch.float32


def test_default_moe_runner_casts_sm70_compile_boundary_outputs():
    """The Python graph carries an explicit fp16 cast after MoE custom op."""
    import os
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
        DefaultMoERunner,
    )

    class Mxfp4SM70MoEMethod:
        compile_boundary_output_dtype = torch.float16

    runner = object.__new__(DefaultMoERunner)
    runner.moe_config = SimpleNamespace(in_dtype=torch.float32)
    runner.quant_method = Mxfp4SM70MoEMethod()

    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)
    fused_fp32 = torch.randn(8, 64, dtype=torch.float32)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        shared_out, fused_out = runner._ensure_compile_boundary_outputs(
            (shared_fp32, fused_fp32)
        )

    assert shared_out.dtype == torch.float16
    assert fused_out.dtype == torch.float16


def test_default_moe_runner_compile_boundary_outputs_can_be_disabled():
    """Disabling the shield preserves the old fp32 graph for repro."""
    import os
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
        DefaultMoERunner,
    )

    class Mxfp4SM70MoEMethod:
        compile_boundary_output_dtype = torch.float16

    runner = object.__new__(DefaultMoERunner)
    runner.moe_config = SimpleNamespace(in_dtype=torch.float32)
    runner.quant_method = Mxfp4SM70MoEMethod()

    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)
    fused_fp32 = torch.randn(8, 64, dtype=torch.float32)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        shared_out, fused_out = runner._ensure_compile_boundary_outputs(
            (shared_fp32, fused_fp32)
        )

    assert shared_out.dtype == torch.float32
    assert fused_out.dtype == torch.float32


def test_deepseek_v4_moe_add_uses_quant_method_boundary_dtype():
    """DeepSeek V4 tuple-output add can discover the SM70 MoE output dtype."""
    from types import SimpleNamespace

    from vllm.model_executor.models.deepseek_v4 import (
        _compile_boundary_output_dtype_from_experts,
    )

    experts = SimpleNamespace(
        quant_method=SimpleNamespace(compile_boundary_output_dtype=torch.float16),
    )

    assert (
        _compile_boundary_output_dtype_from_experts(experts, torch.float32)
        == torch.float16
    )


def test_deepseek_v4_moe_add_unwraps_runner_quant_method():
    """DeepSeek V4 also sees quant methods through the FusedMoE runner."""
    from types import SimpleNamespace

    from vllm.model_executor.models.deepseek_v4 import (
        _compile_boundary_output_dtype_from_experts,
    )

    experts = SimpleNamespace(
        runner=SimpleNamespace(
            quant_method=SimpleNamespace(
                old_quant_method=SimpleNamespace(
                    compile_boundary_output_dtype=torch.float16,
                )
            )
        )
    )

    assert (
        _compile_boundary_output_dtype_from_experts(experts, torch.float32)
        == torch.float16
    )


def test_deepseek_v4_moe_add_uses_opaque_custom_op_boundary():
    """MoE shared+routed add must stay opaque to Inductor fusion.

    The FULL_DECODE_ONLY compile path previously fused the tuple-return add
    from ``moe_forward_shared`` into an add+all_reduce pointwise kernel.  Even
    after dtype shielding, that fusion is not a trustworthy runtime boundary,
    so DeepSeek V4 should combine the two fp16 MoE outputs through a custom op.
    """
    from vllm.model_executor.models import deepseek_v4

    routed = torch.randn(4, 8, dtype=torch.float16)
    shared = torch.randn(4, 8, dtype=torch.float16)

    combined = deepseek_v4._combine_moe_shared_outputs(routed, shared)
    expected = routed + shared

    assert combined.dtype == torch.float16
    assert combined.shape == routed.shape
    assert torch.equal(combined, expected)


def test_moe_forward_shared_runtime_returns_fake_model_dtype(monkeypatch):
    """The runtime custom op must match the fake model-dtype contract."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.runner import default_moe_runner

    class DummyRunner:
        use_dp_chunking = False

        def _sequence_parallel_context(self):
            return nullcontext()

        def forward_impl(
            self,
            layer,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        ):
            del layer, router_logits, input_ids
            assert hidden_states.dtype == torch.float16
            assert shared_experts_input is not None
            assert shared_experts_input.dtype == torch.float16
            return shared_experts_input, hidden_states

    fake_layer = SimpleNamespace(
        runner=DummyRunner(),
        moe_config=SimpleNamespace(in_dtype=torch.float16),
        ensure_moe_quant_config_init=lambda: None,
    )
    monkeypatch.setattr(
        default_moe_runner, "get_layer_from_name", lambda layer_name: fake_layer
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    shared_fp32 = torch.randn(8, 128, dtype=torch.float32)
    router_logits = torch.randn(8, 16, dtype=torch.float32)

    shared_out, fused_out = default_moe_runner._moe_forward_shared(
        hidden_fp32,
        router_logits,
        shared_fp32,
        "model.layers.0.ffn.experts",
        None,
    )

    assert fused_out.dtype == torch.float16
    assert shared_out.dtype == torch.float16


def test_deepseek_v4_attention_output_dtype_uses_qr_dtype_when_shielded():
    """Attention output allocation should not inherit leaked fp32 RMSNorm dtype."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.deepseek_v4_attention import (
        _attention_boundary_output_dtype,
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    qr_fp16 = torch.randn(8, 64, dtype=torch.float16)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        assert _attention_boundary_output_dtype(hidden_fp32, qr_fp16) == torch.float16


def test_deepseek_v4_attention_output_dtype_can_reproduce_old_contract():
    """Disabling the shield keeps the pre-fix fp32 output-buffer contract."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.deepseek_v4_attention import (
        _attention_boundary_output_dtype,
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    qr_fp16 = torch.randn(8, 64, dtype=torch.float16)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        assert _attention_boundary_output_dtype(hidden_fp32, qr_fp16) == torch.float32


def test_deepseek_v4_attention_output_boundary_uses_internal_q_dtype():
    """A pre-allocated fp32 boundary buffer is served by an fp16 internal buffer."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.deepseek_v4_attention import (
        _copy_attention_output_boundary,
        _resolve_attention_output_boundary,
    )

    boundary = torch.empty((4, 8), dtype=torch.float32)
    q = torch.empty((4, 8), dtype=torch.float16)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        internal, boundary_out = _resolve_attention_output_boundary(boundary, q)

    assert internal is not boundary
    assert internal.dtype == torch.float16
    assert boundary_out is boundary

    internal.fill_(1.5)
    _copy_attention_output_boundary(internal, boundary_out)
    assert torch.equal(boundary, internal.to(torch.float32))


def test_deepseek_v4_attention_output_boundary_disabled_is_noop():
    """The env switch keeps the mismatched boundary visible for preservation tests."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.deepseek_v4_attention import (
        _resolve_attention_output_boundary,
    )

    boundary = torch.empty((4, 8), dtype=torch.float32)
    q = torch.empty((4, 8), dtype=torch.float16)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        internal, boundary_out = _resolve_attention_output_boundary(boundary, q)

    assert internal is boundary
    assert boundary_out is None


def test_deepseek_v4_attention_projection_dtype_uses_qr_dtype_when_shielded():
    """O-projection scratch buffers must not inherit leaked fp32 input dtype."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.deepseek_v4_attention import (
        _attention_projection_output_dtype,
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    qr_fp16 = torch.randn(8, 64, dtype=torch.float16)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "1"}):
        assert _attention_projection_output_dtype(hidden_fp32, qr_fp16) == torch.float16


def test_deepseek_v4_attention_projection_dtype_can_reproduce_old_contract():
    """Disabling the shield leaves the old fp32 O-projection scratch dtype."""
    import os
    from unittest.mock import patch

    from vllm.model_executor.layers.deepseek_v4_attention import (
        _attention_projection_output_dtype,
    )

    hidden_fp32 = torch.randn(8, 64, dtype=torch.float32)
    qr_fp16 = torch.randn(8, 64, dtype=torch.float16)

    with patch.dict(os.environ, {"VLLM_SM70_COMPILE_BOUNDARY_SHIELD": "0"}):
        assert _attention_projection_output_dtype(hidden_fp32, qr_fp16) == torch.float32


def test_mhc_pre_fake_matches_torch_fallback_layer_input_dtype():
    """The fake layer_input dtype must match the no-TileLang runtime fallback."""
    from vllm.model_executor.layers.mhc import _mhc_pre_fake

    residual = torch.randn(8, 4, 64, dtype=torch.float16)
    fn = torch.randn(24, 256, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)

    _, _, layer_input = _mhc_pre_fake(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
    )

    assert layer_input.dtype == torch.float16
