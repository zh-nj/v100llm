# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property test for the SM70 fused MoE platform gate (design Property 6).

Feature: deepgemm-megamoe-sm70-port

The gate :func:`sm70_fused_support` is a pure, side-effect-free decision
function (design §Components "SM70FusedMoEGate"). This module verifies its
single correctness property across the full combination space of
``(device_capability, flag_enabled, dtype, activation, weight_format,
group_size, shape)``.

The fused path directly targets dsv4f's **MXFP4** expert weights, so the gate
enables only when ``weight_format == "mxfp4"``; the placeholder formats
``"fp8"`` / ``"awq_int4"`` (and any other value, e.g. ``"int8"``) are always
rejected this phase.

This test imports only the gate module plus ``torch`` -- no compiled extension
is required, so it runs in a CPU-only environment.

Validates: Requirements 2.4, 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    sm70_fused_support,
)

# --- generation domains ----------------------------------------------------

# SM70 is (7, 0). Include a spread of other capabilities (older Pascal, newer
# Turing/Ampere/Hopper/Blackwell) so the platform branch is exercised in both
# directions.
_CAPABILITIES = [
    (7, 0),  # SM70 / V100 -- the only accepted capability
    (6, 0),
    (6, 1),
    (7, 2),
    (7, 5),
    (8, 0),
    (8, 6),
    (8, 9),
    (9, 0),
    (10, 0),
    (12, 0),
]

# fp16 is the only accepted dtype; bf16/fp32 must be rejected.
_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

# "silu" is the only accepted activation.
_ACTIVATIONS = ["silu", "gelu", "relu", "swiglu", ""]

# "mxfp4" is the only accepted format (dsv4f experts). "fp8" / "awq_int4" are
# recognized placeholders that are always rejected this phase; "int8" stands in
# for an entirely unknown format.
_WEIGHT_FORMATS = ["mxfp4", "fp8", "awq_int4", "int8"]

# None (defaults to the MXFP4 block size, 32) plus a mix of divisor /
# non-divisor group sizes.
_GROUP_SIZES = [None, 32, 64, 128, 96, 48, 17]

# MXFP4 micro-scaling block size; ``group_size=None`` defaults to this.
_MXFP4_GROUP_SIZE = 32


@st.composite
def _shape_dim(draw: st.DrawFn) -> int:
    """A shape dimension biased toward kernel-friendly multiples.

    Mixes values that are multiples of 128 (hence also of 8/32/64) -- so the
    MXFP4 "enabled" branch is reachable -- with arbitrary integers that
    frequently violate the divisibility constraints.
    """
    kind = draw(st.integers(min_value=0, max_value=2))
    if kind == 0:
        return draw(st.integers(min_value=1, max_value=64)) * 128
    if kind == 1:
        return draw(st.integers(min_value=1, max_value=128)) * 8
    return draw(st.integers(min_value=1, max_value=2048))


def _expected_enabled(
    *,
    device_capability: tuple[int, int],
    flag_enabled: bool,
    dtype: torch.dtype,
    activation: str,
    weight_format: str,
    group_size: int | None,
    hidden: int,
    intermediate: int,
) -> bool:
    """Independently derive the expected ``enabled`` value (the IFF spec).

    This mirrors the design's enable condition from first principles and is
    intentionally NOT a call into the module under test, so the property check
    is a genuine cross-validation rather than a tautology.
    """
    if tuple(device_capability) != (7, 0):
        return False
    if not flag_enabled:
        return False
    if dtype is not torch.float16:
        return False
    if activation != "silu":
        return False
    # MXFP4 is the only enabled expert weight format this phase; "fp8" /
    # "awq_int4" are placeholders and anything else is unknown -- all rejected.
    if weight_format != "mxfp4":
        return False
    if hidden % 8 != 0:
        return False
    if (2 * intermediate) % 8 != 0:
        return False
    # MXFP4 block tiling: K (hidden) must be a multiple of group_size; None
    # defaults to the MXFP4 block size (32).
    effective_group_size = (
        _MXFP4_GROUP_SIZE if group_size is None else group_size
    )
    if hidden % effective_group_size != 0:
        return False
    return True


# Feature: deepgemm-megamoe-sm70-port, Property 6: SM70 融合门控纯函数
@settings(max_examples=100, deadline=None)
@given(
    device_capability=st.sampled_from(_CAPABILITIES),
    flag_enabled=st.booleans(),
    dtype=st.sampled_from(_DTYPES),
    activation=st.sampled_from(_ACTIVATIONS),
    weight_format=st.sampled_from(_WEIGHT_FORMATS),
    group_size=st.sampled_from(_GROUP_SIZES),
    hidden=_shape_dim(),
    intermediate=_shape_dim(),
)
def test_prop6_gate_purity(
    device_capability: tuple[int, int],
    flag_enabled: bool,
    dtype: torch.dtype,
    activation: str,
    weight_format: str,
    group_size: int | None,
    hidden: int,
    intermediate: int,
) -> None:
    """Property 6: ``enabled`` is True IFF every gate condition holds.

    For any combination of (capability, flag, dtype, activation, weight_format,
    group_size, shape): ``enabled == True`` if and only if
    ``device_capability == (7, 0)`` AND ``flag_enabled`` AND fp16 AND silu AND
    ``weight_format == "mxfp4"`` AND ``hidden % 8 == 0`` AND
    ``2*intermediate % 8 == 0`` AND ``hidden % group_size == 0`` (where a
    ``None`` ``group_size`` defaults to the MXFP4 block size, 32). Every other
    combination returns ``enabled == False`` with a non-empty ``reason`` (so the
    caller can log the fallback once).
    """
    result = sm70_fused_support(
        device_capability=device_capability,
        flag_enabled=flag_enabled,
        weight_format=weight_format,
        group_size=group_size,
        activation=activation,
        hidden=hidden,
        intermediate=intermediate,
        dtype=dtype,
    )

    expected = _expected_enabled(
        device_capability=device_capability,
        flag_enabled=flag_enabled,
        dtype=dtype,
        activation=activation,
        weight_format=weight_format,
        group_size=group_size,
        hidden=hidden,
        intermediate=intermediate,
    )

    # The IFF relationship: the gate's decision matches the independently
    # derived expectation exactly.
    assert result.enabled == expected, (
        f"gate enabled={result.enabled} but expected {expected} for "
        f"capability={device_capability}, flag={flag_enabled}, dtype={dtype}, "
        f"activation={activation!r}, weight_format={weight_format!r}, "
        f"group_size={group_size}, hidden={hidden}, "
        f"intermediate={intermediate}; reason={result.reason!r}"
    )

    if result.enabled:
        # Accepted: no fallback reason is carried.
        assert result.reason is None
    else:
        # Rejected: a non-empty reason must be present for one-shot logging.
        assert result.reason is not None
        assert isinstance(result.reason, str)
        assert result.reason.strip() != ""
