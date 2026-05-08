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
