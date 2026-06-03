# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAMPLE/unit tests for the SM70 fused MoE wiring into ``Mxfp4SM70MoEMethod``.

Feature: deepgemm-megamoe-sm70-port (task 5.4)

These are concrete example tests (NOT property tests) that pin down the task 5.4
integration of the experimental SM70 MXFP4 fused MoE path into
``Mxfp4SM70MoEMethod.apply`` / ``process_weights_after_loading``:

* **R6.3 (default-off safety)** — with ``VLLM_SM70_FUSED_MOE`` off (the default),
  ``_maybe_stash_sm70_mxfp4_weights`` stashes *no* MXFP4 pack and ``apply`` never
  enters the fused branch, so behavior is byte-identical to today.
* **R2.4 / R6.1 / R6.2 (layered fallback)** — when the switch is on, the fused
  branch falls back to the existing TurboMind grouped-GEMM path (returns ``None``
  from the helper) on: a gate rejection, an unsupported kernel shape, an active
  CUDA-graph capture (the contiguous fused layout is not replay-safe), or any
  runtime error from the fused experts — logging the reason once each time.
  Eager decode (small M, not capturing) runs the fused path (the contiguous
  mega-kernel is safe there too), not a fallback.
* **stash contract** — when the switch is on the raw MXFP4 weights are stashed
  under the attribute names ``SM70MXFP4QuantParams.from_layer`` reads, so a
  ``SM70MXFP4QuantParams`` can be rebuilt after the TurboMind prep deletes the
  originals.

This module stubs the gate / experts so it runs in a CPU-only environment (the
real fused kernel needs a V100 + the rebuilt extension; full numerical
validation is task 5.5).

Validates: Requirements 2.4, 6.1, 6.2
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.logger import _print_warning_once, init_logger
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusedConfig,
    SM70FusedSupport,
    SM70FusionLevel,
)
from vllm.model_executor.layers.quantization import sm70_mxfp4_moe
from vllm.model_executor.layers.quantization.sm70_mxfp4_moe import (
    Mxfp4SM70MoEMethod,
)

_MODULE_LOGGER_NAME = sm70_mxfp4_moe.logger.name


# --- helpers ----------------------------------------------------------------


def _make_method() -> Mxfp4SM70MoEMethod:
    """Build an ``Mxfp4SM70MoEMethod`` shell without the heavy ``__init__``.

    Only the attributes the fused branch reads are set (``group_size`` /
    ``moe.activation``); ``__init__`` would require a full ``FusedMoEConfig``
    which is irrelevant to the gating / fallback / stash logic here.
    """
    method = Mxfp4SM70MoEMethod.__new__(Mxfp4SM70MoEMethod)
    method.group_size = 32
    method.moe = SimpleNamespace(activation="silu")
    return method


class _StubExperts:
    """Stand-in for ``SM70FusedMoEExperts`` recording the path taken."""

    def __init__(
        self,
        layout: str = "contiguous",
        raise_on_forward: bool = False,
        capturing: bool = False,
    ):
        self._layout = layout
        self._raise = raise_on_forward
        self._capturing = capturing
        self.forward_called = False

    def select_layout(self, num_tokens: int) -> str:  # noqa: D401 - stub
        return self._layout

    def _detect_graph_capturing(self) -> bool:  # noqa: D401 - stub
        return self._capturing

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
        hidden_K=512, inter_I=256, group_size=32, hidden_logical_size=512
    )
    cfg = SM70FusedConfig(enabled=enabled, fusion_level=fusion_level)
    return SimpleNamespace(
        layer_name="test.layer",
        sm70_fused_quant_params=quant,
        sm70_fused_experts=experts,
        sm70_fused_config=cfg,
    )


def _make_mxfp4_layer(num_experts=2, inter_i=64, hidden_k=64, group_size=32):
    """A layer carrying valid raw MXFP4 params (dsv4f create_weights layout)."""
    two_i = 2 * inter_i
    layer = SimpleNamespace(layer_name="test.layer")
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(num_experts, two_i, hidden_k // 2, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.zeros(num_experts, two_i, hidden_k // group_size, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_k, inter_i // 2, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_k, inter_i // group_size, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.sm70_hidden_logical_size = hidden_k
    return layer


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


# --- R6.3: default-off safety -----------------------------------------------


def test_default_off_stashes_no_fused_pack(cap_logger) -> None:
    """R6.3: flag off => no pack built, raw weights untouched, nothing logged."""
    method = _make_method()
    # A layer without any MXFP4 weights — proves the gated stash never reads
    # them when the switch is off.
    layer = SimpleNamespace()

    with mock.patch.object(sm70_mxfp4_moe.envs, "VLLM_SM70_FUSED_MOE", False):
        method._maybe_stash_sm70_mxfp4_weights(layer)

    assert layer.sm70_fused_quant_params is None
    assert layer.sm70_fused_experts is None
    assert not hasattr(layer, "sm70_mxfp4_w13_weight")
    assert _warnings(cap_logger) == []


def test_apply_skips_fused_branch_when_not_built() -> None:
    """R6.3: with no fused experts on the layer, ``apply`` never calls the helper.

    Uses a zero-token batch so the existing path short-circuits cheaply; the
    fused helper must not be consulted.
    """
    method = _make_method()
    out_buf = torch.zeros((0, 512), dtype=torch.float16)
    layer = SimpleNamespace(sm70_batched_ready=True, sm70_fused_experts=None)
    x = torch.zeros((0, 512), dtype=torch.float16)
    tw = torch.zeros((0, 2), dtype=torch.float16)
    ti = torch.zeros((0, 2), dtype=torch.int64)

    with (
        mock.patch.object(
            Mxfp4SM70MoEMethod, "_get_buffers", return_value={"output": out_buf}
        ),
        mock.patch.object(
            Mxfp4SM70MoEMethod, "_maybe_apply_sm70_fused"
        ) as fused_helper,
    ):
        out = method.apply(layer, x, tw, ti, None)

    fused_helper.assert_not_called()
    assert out.shape == (0, 512)


def test_apply_uses_fused_when_built_and_returns_its_result() -> None:
    """When the fused helper returns a tensor, ``apply`` returns it directly."""
    method = _make_method()
    layer = SimpleNamespace(
        sm70_batched_ready=True, sm70_fused_experts=object()
    )
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)
    sentinel = torch.full((32, 512), 3.0, dtype=torch.float16)

    with mock.patch.object(
        Mxfp4SM70MoEMethod, "_maybe_apply_sm70_fused", return_value=sentinel
    ) as fused_helper:
        out = method.apply(layer, x, tw, ti, None)

    fused_helper.assert_called_once()
    assert torch.equal(out, sentinel)


# --- stash contract (switch on) ---------------------------------------------


def test_stash_builds_pack_and_sets_from_layer_attrs(cap_logger) -> None:
    """Switch on: raw MXFP4 weights are stashed and a pack is built (R2.4)."""
    method = _make_method()
    layer = _make_mxfp4_layer()

    with mock.patch.object(sm70_mxfp4_moe.envs, "VLLM_SM70_FUSED_MOE", True):
        method._maybe_stash_sm70_mxfp4_weights(layer)

    # Stash attrs the SM70MXFP4QuantParams.from_layer contract reads.
    assert torch.equal(layer.sm70_mxfp4_w13_weight, layer.w13_weight.data)
    assert torch.equal(
        layer.sm70_mxfp4_w13_weight_scale, layer.w13_weight_scale.data
    )
    assert torch.equal(layer.sm70_mxfp4_w2_weight, layer.w2_weight.data)
    assert torch.equal(
        layer.sm70_mxfp4_w2_weight_scale, layer.w2_weight_scale.data
    )
    assert layer.group_size == 32

    # A valid pack + experts orchestrator were built.
    assert isinstance(layer.sm70_fused_quant_params, SM70MXFP4QuantParams)
    assert isinstance(layer.sm70_fused_experts, SM70FusedMoEExperts)
    assert layer.sm70_fused_quant_params.num_experts == 2
    assert layer.sm70_fused_quant_params.hidden_K == 64
    assert layer.sm70_fused_quant_params.inter_I == 64
    assert layer.sm70_fused_quant_params.group_size == 32


def test_stash_skipped_for_swiglu_limit_layer(cap_logger) -> None:
    """Switch on but layer needs swiglu_limit => no pack built (feature parity).

    The fused kernel implements a plain ``silu(gate)*up`` epilogue with no
    swiglu clamp, so a layer carrying a positive ``swiglu_limit`` must NOT build
    the fused pack (it would be numerically wrong) — and must not hold the extra
    raw-weight copy alive. The TurboMind per-operator path is used instead.
    """
    method = _make_method()
    layer = _make_mxfp4_layer()
    layer.swiglu_limit = 10.0

    with mock.patch.object(sm70_mxfp4_moe.envs, "VLLM_SM70_FUSED_MOE", True):
        method._maybe_stash_sm70_mxfp4_weights(layer)

    assert layer.sm70_fused_quant_params is None
    assert layer.sm70_fused_experts is None
    # No extra raw-weight copy stashed (memory-safety).
    assert not hasattr(layer, "sm70_mxfp4_w13_weight")


def test_stash_skipped_for_biased_layer() -> None:
    """Switch on but the MoE has expert bias => no pack built (feature parity)."""
    method = _make_method()
    method.moe = SimpleNamespace(activation="silu", has_bias=True)
    layer = _make_mxfp4_layer()

    with mock.patch.object(sm70_mxfp4_moe.envs, "VLLM_SM70_FUSED_MOE", True):
        method._maybe_stash_sm70_mxfp4_weights(layer)

    assert layer.sm70_fused_quant_params is None
    assert layer.sm70_fused_experts is None
    assert not hasattr(layer, "sm70_mxfp4_w13_weight")


def test_fallback_when_swiglu_limit_present(cap_logger) -> None:
    """Runtime guard: a swiglu_limit on the layer => fall back, log once."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous")
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    layer.swiglu_limit = 10.0
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        sm70_mxfp4_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is False
    w = _warnings(cap_logger)
    assert len(w) == 1 and "swiglu_limit" in w[0].getMessage()


def test_stash_survives_turbomind_delete_via_from_layer() -> None:
    """The stash lets ``from_layer`` rebuild after the originals are deleted."""
    method = _make_method()
    layer = _make_mxfp4_layer()

    with mock.patch.object(sm70_mxfp4_moe.envs, "VLLM_SM70_FUSED_MOE", True):
        method._maybe_stash_sm70_mxfp4_weights(layer)

    # Simulate the TurboMind prep freeing the originals (as apply() does).
    del layer.w13_weight, layer.w2_weight
    del layer.w13_weight_scale, layer.w2_weight_scale

    # from_layer must still rebuild from the stash attributes alone.
    rebuilt = SM70MXFP4QuantParams.from_layer(layer, group_size=32)
    assert rebuilt.num_experts == 2
    assert rebuilt.hidden_K == 64
    assert rebuilt.inter_I == 64


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
        sm70_mxfp4_moe,
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
        hidden_K=512, inter_I=200, group_size=32, hidden_logical_size=512
    )
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        sm70_mxfp4_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is False
    w = _warnings(cap_logger)
    assert len(w) == 1 and "unsupported shape" in w[0].getMessage()


def test_fallback_when_graph_capturing(cap_logger) -> None:
    """An active CUDA-graph capture => fall back (contiguous layout not replay-safe)."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous", capturing=True)
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((4, 512), dtype=torch.float16)  # small batch -> decode
    tw = torch.zeros((4, 2), dtype=torch.float16)
    ti = torch.zeros((4, 2), dtype=torch.int64)

    with mock.patch.object(
        sm70_mxfp4_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is None
    assert experts.forward_called is False
    w = _warnings(cap_logger)
    assert len(w) == 1 and "CUDA graph capture" in w[0].getMessage()


def test_eager_decode_takes_fused_path() -> None:
    """Eager decode (small M, not capturing) => fused experts.forward is used.

    The fused CUDA mega-kernel consumes the contiguous layout, which is safe for
    eager decode as well as prefill — only an active graph capture is rejected.
    A small (decode-sized) batch with no capture in progress must therefore run
    the fused path, not fall back.
    """
    method = _make_method()
    experts = _StubExperts(layout="masked", capturing=False)  # decode-sized M
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((4, 512), dtype=torch.float16)  # small batch -> decode
    tw = torch.zeros((4, 2), dtype=torch.float16)
    ti = torch.zeros((4, 2), dtype=torch.int64)

    with mock.patch.object(
        sm70_mxfp4_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is not None
    assert experts.forward_called is True
    assert out.shape == (4, 512)


def test_fallback_on_runtime_error(cap_logger) -> None:
    """Any runtime error from the fused experts => fall back, log once."""
    method = _make_method()
    experts = _StubExperts(layout="contiguous", raise_on_forward=True)
    layer = _make_layer(experts, enabled=True, fusion_level=SM70FusionLevel.L2)
    x = torch.zeros((32, 512), dtype=torch.float16)
    tw = torch.zeros((32, 2), dtype=torch.float16)
    ti = torch.zeros((32, 2), dtype=torch.int64)

    with mock.patch.object(
        sm70_mxfp4_moe,
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
        sm70_mxfp4_moe,
        "sm70_fused_support",
        return_value=SM70FusedSupport(enabled=True, reason=None),
    ):
        out = method._maybe_apply_sm70_fused(layer, x, tw, ti)

    assert out is not None
    assert experts.forward_called is True
    assert out.shape == (32, 512)
