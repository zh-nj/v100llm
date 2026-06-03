# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAMPLE/unit test for the SM70 fused MoE benchmark debug logging (R7.2).

Feature: deepgemm-megamoe-sm70-port

This is a concrete example/unit test (NOT a property test). It validates the
observability requirement from the design's Requirements Mapping
(R7 → "SMOKE + EXAMPLE (日志)"):

* **R7.2 — debug log contains layout / reason / shape.** When debug logging is
  enabled, the SM70 fused MoE benchmark (``SM70MoEBenchmark.run_shape`` in
  ``benchmarks/kernels/benchmark_sm70_fused_moe.py``) must emit a single
  ``logger.debug(...)`` record that reports:

    - the selected token **layout** (``contiguous`` / ``masked``),
    - the fused-path gate fallback **reason** (from ``sm70_fused_support`` —
      a string when the path is rejected, ``None`` when it is enabled), and
    - the key **shape** parameters ``(M, K, I, E, topk, group_size)``.

The benchmark logger is namespaced ``vllm.benchmarks.benchmark_sm70_fused_moe``
so it inherits the vLLM logging tree. We mirror the capturing-handler approach
used in ``tests/quantization/test_sm70_fused_moe_gate_examples.py``: attach a
recording :class:`logging.Handler` to that logger and set its level to DEBUG,
then drive a single small benchmark shape and assert on the captured record.

The benchmark emits the R7.2 debug line *before* it builds any inputs or times
any variant, so this test runs on CPU (no V100 / built kernel required): the
masked per-operator path runs on CPU, the contiguous path degrades gracefully,
and either way the log is captured. The non-V100 host naturally exercises the
"reason is a fallback string" branch; a mocked (7, 0) capability exercises the
"reason is ``None`` (fused enabled)" branch.

Validates: Requirements 7.2
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

from vllm.logger import init_logger

# Path to the benchmark script (task 7.1). ``benchmarks/`` is not an importable
# package, so the module is loaded directly from its file path.
_BENCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "kernels"
    / "benchmark_sm70_fused_moe.py"
)
# The logger namespace the benchmark uses (see its ``init_logger`` call).
_BENCH_LOGGER_NAME = "vllm.benchmarks.benchmark_sm70_fused_moe"


# --------------------------------------------------------------------------- #
# Module loading + logging capture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bench_mod() -> ModuleType:
    """Import the benchmark script from its file path (it is not a package)."""
    assert _BENCH_PATH.is_file(), f"benchmark script missing: {_BENCH_PATH}"
    mod_name = "benchmark_sm70_fused_moe"
    spec = importlib.util.spec_from_file_location(mod_name, _BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing so dataclass field resolution (which looks up
    # ``sys.modules[cls.__module__]`` for ``InitVar``/``ClassVar`` types) can
    # find this module while its ``@dataclass`` definitions are being built.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


class _ListHandler(logging.Handler):
    """Minimal handler that records every emitted ``LogRecord``."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def debug_capture():
    """Attach a capturing handler to the benchmark logger at DEBUG level.

    Yields the ``(logger, handler)`` pair and restores the logger's level /
    handlers afterwards so the capture does not leak into other tests.
    """
    logger = init_logger(_BENCH_LOGGER_NAME)
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        yield logger, handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _debug_records(handler: _ListHandler) -> list[logging.LogRecord]:
    return [r for r in handler.records if r.levelno == logging.DEBUG]


def _make_bench(bench_mod: ModuleType):
    """A CPU benchmark driver with a single L0 level and minimal iterations."""
    return bench_mod.SM70MoEBenchmark(
        device="cpu",
        warmup=1,
        iters=1,
        fusion_levels=(bench_mod.SM70FusionLevel.L0,),
    )


# --------------------------------------------------------------------------- #
# R7.2: the debug record contains layout / reason / shape
# --------------------------------------------------------------------------- #

# (label, MoEShape kwargs, expected layout). ``decode_threshold`` defaults to
# 16, so M=8 -> masked (decode) and M=64 -> contiguous (prefill); both exercise
# the same single debug line with a different selected layout.
_SHAPE_CASES = {
    "decode_masked": (dict(M=8, K=64, I=64, E=4, topk=2, group_size=32), "masked"),
    "prefill_contiguous": (
        dict(M=64, K=128, I=64, E=8, topk=2, group_size=64),
        "contiguous",
    ),
}


@pytest.mark.parametrize("case", sorted(_SHAPE_CASES))
def test_debug_log_contains_layout_reason_shape(bench_mod, debug_capture, case) -> None:
    """R7.2: a single debug record reports layout, reason and all shape params.

    On this CPU host the gate rejects the fused path (capability != (7, 0)), so
    ``reason`` is a non-empty fallback string; the test asserts the record
    surfaces the selected layout, that reason, and every shape parameter.
    """
    _logger, handler = debug_capture
    shape_kwargs, expected_layout = _SHAPE_CASES[case]
    shape = bench_mod.MoEShape(**shape_kwargs)

    bench = _make_bench(bench_mod)
    result = bench.run_shape(shape)

    # The benchmark emits exactly one R7.2 debug line per shape.
    records = _debug_records(handler)
    assert len(records) == 1, (
        f"expected exactly one debug record, got {len(records)}: "
        f"{[r.getMessage() for r in records]}"
    )
    msg = records[0].getMessage()

    # Sanity: the benchmark resolved the layout we expect for this M.
    assert result.layout == expected_layout
    # On a non-V100 host the fused gate rejects -> non-empty fallback reason.
    assert result.gate_reason is not None and result.gate_reason.strip() != ""

    # --- layout ---------------------------------------------------------- #
    assert f"layout={expected_layout}" in msg, f"layout missing from log: {msg!r}"

    # --- reason ---------------------------------------------------------- #
    assert "reason=" in msg, f"reason field missing from log: {msg!r}"
    assert result.gate_reason in msg, (
        f"fallback reason {result.gate_reason!r} missing from log: {msg!r}"
    )

    # --- shape (M, K, I, E, topk, group_size) ---------------------------- #
    for token, value in (
        ("M", shape.M),
        ("K", shape.K),
        ("I", shape.I),
        ("E", shape.E),
        ("topk", shape.topk),
        ("group_size", shape.group_size),
    ):
        assert f"{token}={value}" in msg, (
            f"shape param {token}={value} missing from log: {msg!r}"
        )


def test_debug_log_reason_none_when_gate_enabled(bench_mod, debug_capture) -> None:
    """R7.2: with a mocked V100 capability the gate enables and reason is None.

    This exercises the complementary branch of the R7.2 log: when the fused path
    is *accepted* the ``reason`` is ``None`` (rendered ``reason=None``), and the
    layout + shape fields are still present. Capability is mocked to (7, 0) by
    patching the driver's ``_device_capability`` probe (CPU has no real V100).
    """
    _logger, handler = debug_capture
    shape = bench_mod.MoEShape(M=64, K=128, I=64, E=8, topk=2, group_size=64)

    bench = _make_bench(bench_mod)
    # Make the gate see a V100 so it enables the fused path (reason -> None).
    bench._device_capability = lambda: (7, 0)
    result = bench.run_shape(shape)

    assert result.gate_enabled is True
    assert result.gate_reason is None

    records = _debug_records(handler)
    assert len(records) == 1
    msg = records[0].getMessage()

    assert "layout=contiguous" in msg
    # gate enabled -> reason rendered as None, still under the ``reason=`` field.
    assert "gate_enabled=True" in msg
    assert "reason=None" in msg
    # shape params remain present.
    for token, value in (
        ("M", shape.M),
        ("K", shape.K),
        ("I", shape.I),
        ("E", shape.E),
        ("topk", shape.topk),
        ("group_size", shape.group_size),
    ):
        assert f"{token}={value}" in msg, f"{token} missing: {msg!r}"


def test_no_debug_log_when_level_above_debug(bench_mod) -> None:
    """R7.2 (control): the layout/reason/shape line is gated behind DEBUG.

    With the logger at WARNING the R7.2 record is suppressed, confirming the
    line is a genuine ``logger.debug`` emission rather than an always-on print.
    """
    logger = init_logger(_BENCH_LOGGER_NAME)
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)  # handler permissive; logger gates it
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        bench = _make_bench(bench_mod)
        bench.run_shape(bench_mod.MoEShape(M=8, K=64, I=64, E=4, topk=2, group_size=32))
        assert _debug_records(handler) == []
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
