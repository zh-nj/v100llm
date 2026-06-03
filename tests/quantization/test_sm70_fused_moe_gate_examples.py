# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAMPLE/unit tests for the SM70 fused MoE platform gate (R6.2, R6.4).

Feature: deepgemm-megamoe-sm70-port

These are concrete example tests (NOT property tests) that complement the
combination-space property test in ``test_sm70_fused_moe_prop6_gate.py``. They
pin down two behaviors from the design's Error Handling table:

* **R6.2** — when a model/quant/shape combination is *not* supported (a
  non-MXFP4 expert weight format such as ``"fp8"`` / ``"awq_int4"``, or a shape
  that violates the kernel constraints), the fallback is logged exactly once.
  The gate :func:`sm70_fused_support` is a pure function and performs no logging
  itself; the one-shot logging is the *caller's* job (the
  ``Mxfp4SM70MoEMethod.apply`` integration in task 5.4 will use
  ``logger.warning_once`` with the gate's ``reason``). Until that wiring lands,
  we validate the one-shot semantics on the available surface: the gate returns
  a stable, non-empty ``reason`` that vLLM's ``logger.warning_once`` de-dupes to
  a single emission.
* **R6.4** — under a mocked non-V100 device capability, the gate returns
  ``enabled=False`` with a ``reason`` indicating "not sm70", so the caller takes
  the existing (non-fused) MoE path.

This module imports only the gate, ``torch`` and the vLLM logger/platform shims,
so it runs in a CPU-only environment.

Validates: Requirements 6.2, 6.4
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest
import torch

from vllm.logger import _print_warning_once, init_logger
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusedSupport,
    sm70_fused_support,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

# A fully kernel-friendly, supported shape/quant combo (used as the baseline
# from which individual fields are perturbed to make a combo "unsupported").
# dsv4f experts are MXFP4 (E2M1 + per-32 block scale), so the enabled combo is
# ``weight_format="mxfp4"`` with ``group_size=32``.
_SUPPORTED_KWARGS = dict(
    device_capability=(7, 0),
    flag_enabled=True,
    weight_format="mxfp4",
    group_size=32,  # MXFP4 micro-scaling block size
    activation="silu",
    hidden=512,  # 512 % 8 == 0 and 512 % 32 == 0
    intermediate=256,  # 2*256 % 8 == 0
    dtype=torch.float16,
)


# --- caller-side helpers (stand-ins for the task 5.4 apply() wiring) --------


def _log_fallback_once(
    logger: logging.Logger, support: SM70FusedSupport
) -> bool:
    """Emulate the caller's fallback decision + one-shot logging.

    Mirrors what ``Mxfp4SM70MoEMethod.apply`` will do once the fused path is
    wired in (task 5.4): if the gate rejects the combination, log the reason
    exactly once via ``warning_once`` and fall back to the existing MXFP4
    grouped-GEMM path. Returns ``support.enabled`` so callers can assert which
    path was taken.
    """
    if not support.enabled:
        logger.warning_once(
            "SM70 fused MoE unsupported (%s); falling back to existing "
            "MoE path",
            support.reason,
        )
    return support.enabled


def _capability_from_platform() -> tuple[int, int]:
    """Resolve the current device capability as a ``(major, minor)`` tuple.

    Represents how the caller obtains the capability before consulting the
    gate; patched in tests to simulate running on / off a V100.
    """
    cap = current_platform.get_device_capability()
    if cap is None:
        return (0, 0)
    return (cap.major, cap.minor)


# --- fixtures ---------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Minimal handler that records every emitted ``LogRecord``."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def gate_logger():
    """A vLLM logger with a capturing handler and a clean ``*_once`` cache.

    ``logger.warning_once`` de-dupes via a process-global ``lru_cache``; clear
    it around each test so example assertions about "exactly once" are not
    contaminated by other call sites (or repeated test runs).
    """
    logger = init_logger("vllm.test.sm70_fused_moe_gate_examples")
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)

    _print_warning_once.cache_clear()
    try:
        yield logger, handler
    finally:
        _print_warning_once.cache_clear()
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _warnings(handler: _ListHandler) -> list[logging.LogRecord]:
    return [r for r in handler.records if r.levelno == logging.WARNING]


# --- R6.2: unsupported combination logs the fallback exactly once -----------

# Representative unsupported combinations: each is a V100 with the flag on (so
# only the model/quant/shape field below makes it unsupported), exercising the
# R6.2 "model/quant/shape combination not supported" branch. This covers the
# two non-MXFP4 placeholder weight formats (``fp8`` / ``awq_int4``) plus shape /
# activation constraint violations.
_UNSUPPORTED_COMBOS = {
    "weight_format_fp8": {**_SUPPORTED_KWARGS, "weight_format": "fp8"},
    "weight_format_awq_int4": {
        **_SUPPORTED_KWARGS,
        "weight_format": "awq_int4",
    },
    "group_size_not_dividing_K": {
        **_SUPPORTED_KWARGS,
        "group_size": 96,  # 512 % 96 != 0
    },
    "hidden_not_multiple_of_8": {**_SUPPORTED_KWARGS, "hidden": 100},
    "activation_not_silu": {**_SUPPORTED_KWARGS, "activation": "gelu"},
}


@pytest.mark.parametrize("combo_name", sorted(_UNSUPPORTED_COMBOS))
def test_unsupported_combo_logs_fallback_exactly_once(
    gate_logger, combo_name: str
) -> None:
    """R6.2: an unsupported combo yields a non-empty reason logged once.

    The gate itself is side-effect free, so we drive the caller helper (which
    uses ``warning_once``) repeatedly and assert a single WARNING is emitted —
    even across many forward passes that re-hit the same unsupported combo.
    """
    logger, handler = gate_logger
    kwargs = _UNSUPPORTED_COMBOS[combo_name]

    support = sm70_fused_support(**kwargs)
    # The gate rejects with a stable, non-empty reason (no logging here).
    assert support.enabled is False
    assert support.reason is not None and support.reason.strip() != ""

    # Caller hits the fallback many times (e.g. one per forward pass).
    for _ in range(5):
        took_fused = _log_fallback_once(logger, support)
        assert took_fused is False  # always routes to the existing path

    warnings = _warnings(handler)
    assert len(warnings) == 1, (
        f"expected exactly one fallback warning for {combo_name!r}, got "
        f"{len(warnings)}: {[r.getMessage() for r in warnings]}"
    )
    assert support.reason in warnings[0].getMessage()


def test_distinct_unsupported_reasons_log_separately(gate_logger) -> None:
    """R6.2: one-shot logging is keyed on the reason, not blanket-suppressed.

    Two *different* unsupported reasons must each be logged once (so different
    fallback causes remain observable), while repeats of either are de-duped.
    Here a non-MXFP4 weight format and a non-silu activation produce distinct
    reasons.
    """
    logger, handler = gate_logger

    support_a = sm70_fused_support(**_UNSUPPORTED_COMBOS["weight_format_fp8"])
    support_b = sm70_fused_support(**_UNSUPPORTED_COMBOS["activation_not_silu"])
    assert support_a.reason != support_b.reason

    for _ in range(3):
        _log_fallback_once(logger, support_a)
        _log_fallback_once(logger, support_b)

    warnings = _warnings(handler)
    assert len(warnings) == 2
    logged = {r.getMessage() for r in warnings}
    assert any(support_a.reason in m for m in logged)
    assert any(support_b.reason in m for m in logged)


def test_non_mxfp4_weight_format_reason_is_stable(gate_logger) -> None:
    """R6.2: the non-MXFP4 placeholder formats reject with a stable reason.

    Repeatedly gating the same ``fp8`` / ``awq_int4`` combo yields an identical
    non-empty reason every call (so ``warning_once`` keying is deterministic),
    and the reason references the offending format.
    """
    for fmt in ("fp8", "awq_int4"):
        kwargs = {**_SUPPORTED_KWARGS, "weight_format": fmt}
        reasons = {sm70_fused_support(**kwargs).reason for _ in range(3)}
        assert len(reasons) == 1, f"reason for {fmt!r} not stable: {reasons}"
        (reason,) = reasons
        assert reason is not None and reason.strip() != ""
        assert "mxfp4" in reason and fmt in reason


def test_supported_combo_emits_no_fallback_log(gate_logger) -> None:
    """A fully supported V100 MXFP4 combo enables fusion and logs no fallback."""
    logger, handler = gate_logger

    support = sm70_fused_support(**_SUPPORTED_KWARGS)
    assert support.enabled is True
    assert support.reason is None

    took_fused = _log_fallback_once(logger, support)
    assert took_fused is True
    assert _warnings(handler) == []


# --- R6.4: non-V100 device capability disables the fused path ---------------

_NON_V100_CAPABILITIES = [(6, 1), (7, 2), (7, 5), (8, 0), (8, 6), (9, 0), (10, 0)]


@pytest.mark.parametrize("capability", _NON_V100_CAPABILITIES)
def test_non_v100_capability_disables_and_falls_back(
    gate_logger, capability: tuple[int, int]
) -> None:
    """R6.4: any non-(7,0) capability is rejected with a 'not sm70' reason.

    The caller then takes the existing MoE path (``enabled`` is False),
    leaving non-V100 platform behavior unchanged.
    """
    logger, handler = gate_logger

    kwargs = {**_SUPPORTED_KWARGS, "device_capability": capability}
    support = sm70_fused_support(**kwargs)

    assert support.enabled is False
    assert support.reason is not None
    assert "not sm70" in support.reason

    took_fused = _log_fallback_once(logger, support)
    assert took_fused is False
    assert len(_warnings(handler)) == 1


@pytest.mark.parametrize("capability", [(8, 0), (9, 0)])
def test_non_v100_via_mocked_platform_capability(
    capability: tuple[int, int],
) -> None:
    """R6.4: when ``current_platform`` reports non-V100, the gate disables.

    Mocks ``current_platform.get_device_capability`` (as the design specifies —
    e.g. (8,0)/(9,0)) and feeds it through the caller's capability resolver into
    the gate, asserting the fused path stays off.
    """
    with mock.patch.object(
        current_platform,
        "get_device_capability",
        return_value=DeviceCapability(major=capability[0], minor=capability[1]),
    ):
        resolved = _capability_from_platform()
        assert resolved == capability

        support = sm70_fused_support(
            **{**_SUPPORTED_KWARGS, "device_capability": resolved}
        )

    assert support.enabled is False
    assert support.reason is not None
    assert "not sm70" in support.reason


def test_v100_via_mocked_platform_capability_enables() -> None:
    """R6.4 (control): mocked V100 (7,0) is the only capability that enables.

    Confirms the mock harness is valid and that capability alone gates the
    fused path on exactly (7,0).
    """
    with mock.patch.object(
        current_platform,
        "get_device_capability",
        return_value=DeviceCapability(major=7, minor=0),
    ):
        resolved = _capability_from_platform()
        assert resolved == (7, 0)

        support = sm70_fused_support(
            **{**_SUPPORTED_KWARGS, "device_capability": resolved}
        )

    assert support.enabled is True
    assert support.reason is None
