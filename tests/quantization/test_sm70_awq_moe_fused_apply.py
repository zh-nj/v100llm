# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAMPLE/unit tests for the SM70 fused MoE wiring into ``AWQSM70MoEMethod``.

Feature: deepgemm-megamoe-sm70-port (task 5.4)

These are concrete example tests (NOT property tests) that pin down the task 5.4
integration of the experimental SM70 fused MoE path into
``AWQSM70MoEMethod.apply`` / ``process_weights_after_loading``:

* **R6.3 (default-off safety)** — with ``VLLM_SM70_FUSED_MOE`` off (the default),
  ``process_weights_after_loading`` builds *no* fused weight pack and ``apply``
  never enters the fused branch, so behavior is byte-identical to today.
* **R2.4 / R6.1 / R6.2 (layered fallback)** — when the switch is on, the fused
  branch falls back to the existing path (returns ``None`` from the helper) on:
  a gate rejection, an unsupported kernel shape, a masked/decode layout
  selection, or any runtime error from the fused experts — logging the reason
  once each time.

The kernel-shape pre-check helper mirrors the fused CUDA kernel's compile-time
tile / SMEM constraints; it is validated directly here too.

This module stubs the heavy ``SM70QuantParams`` / ``SM70FusedMoEExperts`` and the
gate so it runs in a CPU-only environment (the real fused kernel needs a V100 +
the rebuilt extension; full numerical validation is task 5.5).

Validates: Requirements 2.4, 6.1, 6.2, 6.3
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.logger import _print_warning_once, init_logger
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusedConfig,
    SM70FusedSupport,
    SM70FusionLevel,
)
from vllm.model_executor.layers.quantization import awq_sm70_moe
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    AWQSM70MoEMethod,
    _sm70_fused_kernel_shape_supported,
)

_MODULE_LOGGER_NAME = awq_sm70_moe.logger.name


# --- helpers ----------------------------------------------------------------


def _make_method() -> AWQSM70MoEMethod:
    """Build an ``AWQSM70MoEMethod`` shell without the heavy ``__init__``.

    Only the attributes the fused branch reads are set (``weight_bits`` /
    ``group_size`` / ``moe.activation``); ``__init__`` would require a full
    ``FusedMoEConfig`` which is irrelevant to the gating/fallback logic here.
    """
    method = AWQSM70MoEMethod.__new__(AWQSM70MoEMethod)
    method.weight_bits = 4
    method.group_size = 128
    method.moe = SimpleNamespace(activation="silu")
    return method


class _StubExperts:
    """Stand-in for ``SM70FusedMoEExperts`` recording the path taken."""

    def __init__(self, layout: str = "contiguous", raise_on_forward: bool = False):
        self._layout = layout
        self._raise = raise_on_forward
        self.forward_called = False

    def select_layout(self, num_tokens: int) -> str:  # noqa: D401 - stub
        return self._layout

    def forward(self, *args, **kwargs) -> torch.Tensor:  # noqa: D401 - stub
        self.forward_called = True
        if self._raise:
            raise RuntimeError("boom: fused kernel unavailable")
        x = args[0]
        return torch.ones(
            (x.shape[0], kwargs["quant"].hidden_logical_size),
            dtype=x.dtype,
            device=x.device,
        )


def _make_layer(experts: _StubExperts, *, enabled: bool, fusion_level):
    quant = SimpleNamespace(
        hidden_K=512, inter_I=256, group_size=128, hidden_logical_size=512
    )
    cfg = SM70FusedConfig(enabled=enabled, fusion_level=fusion_level)
    return SimpleNamespace(
        layer_name="test.layer",
        sm70_fused_quant_params=quant,
        sm70_fused_experts=experts,
        sm70_fused_config=cfg,
    )


@pytest.fixture
def cap_logger():
    """Capture WARNING records from the module logger and reset ``*_once``."""
    logger = init_logger(_MODULE_LOGGER_NAME)
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _H()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    prev = logger.level
    logger.setLevel(logging.DEBUG)
    _print_warning_once.cache_clear()
    try:
        yield records
    finally:
        _print_warning_once.cache_clear()
        logger.removeHandler(handler)
        logger.setLevel(prev)


def _warnings(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if r.levelno == logging.WARNING]


# --- kernel-shape pre-check (mirrors the fused CUDA TORCH_CHECKs, R6.2) ------


def test_shape_precheck_supported() -> None:
    ok, reason = _sm70_fused_kernel_shape_supported(
        hidden_K=512, inter_I=256, m_block=32
    )
    assert ok is True
    assert reason is None


@pytest.mark.parametrize(
    "hidden_K, inter_I, m_block, needle",
    [
        (512, 256, 24, "M_TILE"),  # m_block % 16 != 0
        (512, 200, 32, "I tile"),  # inter_I % 64 != 0
        (500, 256, 32, "K tile"),  # hidden_K % 64 != 0
        (512, 4096, 32, "SMEM"),  # 16*4096*2 + scratch > 96KB
    ],
)
def test_shape_precheck_rejections(hidden_K, inter_I, m_block, needle) -> None:
    ok, reason = _sm70_fused_kernel_shape_supported(
        hidden_K=hidden_K, inter_I=inter_I, m_block=m_block
    )
    assert ok is False
    assert reason is not None and needle in reason


# --- R6.3: default-off safety -----------------------------------------------


def test_default_off_builds_no_fused_pack(cap_logger) -> None:
    """R6.3: flag off => no pack built, no weights touched, nothing logged."""
    method = _make_method()
    # A layer whose AWQ weights would *fail* if accessed — proves the gated
    # build never reads them when the switch is off.
    layer = SimpleNamespace()

    with mock.patch.object(awq_sm70_moe.envs, "VLLM_SM70_FUSED_MOE", False):
        method._maybe_build_sm70_fused_quant_params(layer)

    assert layer.sm70_fused_quant_params is None
    assert layer.sm70_fused_experts is None
    assert _warnings(cap_logger) == []


def test_apply_skips_fused_branch_when_not_built() -> None:
    """R6.3: with no fused experts on the layer, ``apply`` never calls the helper.

    The existing path is reached unchanged; we stub the existing batched path to
    confirm the fused helper is not consulted.
    """
    method = _make_method()
    layer = SimpleNamespace(sm70_batched_ready=True, sm70_fused_experts=None)
    x = torch.zeros((4, 512), dtype=torch.float16)
    tw = torch.zeros((4, 2), dtype=torch.float16)
    ti = torch.zeros((4, 2), dtype=torch.int64)

    sentinel = torch.full((4, 512), 7.0, dtype=torch.float16)
    with (
        mock.patch.object(
            AWQSM70MoEMethod, "_apply_batched", return_value=sentinel
        ) as batched,
        mock.patch.object(
            AWQSM70MoEMethod, "_maybe_apply_sm70_fused"
        ) as fused_helper,
    ):
        out = method.apply(layer, x, tw, ti)

    batched.assert_called_once()
    fused_helper.assert_not_called()
    assert torch.equal(out, sentinel)


# --- R2.4 / R6.1 / R6.2: layered fallback -----------------------------------


def test_fallback_when_gate_rejects(cap_logger) -> None:
    """Gate rejection => helper returns None (fall back) and logs once."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous")
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        awq_sm70_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=False, reason="not sm70 (test)"),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is False
    w = _warnings(cap_logger)
    assert len(w) == 1 and "not sm70 (test)" in w[0].getMessage()


def test_fallback_when_shape_unsupported(cap_logger) -> None:
    """Gate-approved but kernel-shape pre-check fails (L2) => fall back, log once."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous")
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    # inter_I=200 is not a multiple of the kernel's 64-wide I tile.
    layer.sm70_fused_quant_params = SimpleNamespace(
        hidden_K=512, inter_I=200, group_size=128, hidden_logical_size=512
    )
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        awq_sm70_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is False
    w = _warnings(cap_logger)
    assert len(w) == 1 and "unsupported shape" in w[0].getMessage()


def test_fallback_when_masked_layout_selected(cap_logger) -> None:
    """Decode/masked layout selection => fall back (kernel only does contiguous)."""
    method = _make_method()
    experts = _StubExperts(layout="masked")
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((4, 512), dtype=torch.float16)  # small batch -> decode
    tw = torch.zeros((4, 2), dtype=torch.float16)
    ti = torch.zeros((4, 2), dtype=torch.int64)

    with mock.patch.object(
        awq_sm70_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is False
    w = _warnings(cap_logger)
    assert len(w) == 1 and "masked/decode layout" in w[0].getMessage()


def test_fallback_on_runtime_error(cap_logger) -> None:
    """Any runtime error from the fused experts => fall back, log once."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous", raise_on_forward=True)
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        awq_sm70_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is True
    w = _warnings(cap_logger)
    assert len(w) == 1 and "runtime error" in w[0].getMessage()


def test_fused_path_taken_on_success() -> None:
    """Happy path: gate ok + contiguous layout => fused experts.forward used."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous")
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        awq_sm70_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is not None
    assert experts.forward_called is True
    assert out.shape == (32, 512)
