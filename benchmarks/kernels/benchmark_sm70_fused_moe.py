# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SM70 (V100) fused MoE forward vs the per-operator baseline.

This script is part of the ``deepgemm-megamoe-sm70-port`` spec (Requirement 7:
"基准与可观测性"). It reports the latency / throughput of the experimental SM70
fused MoE path (``SM70FusedMoEExperts`` -> ``ops.sm70_fused_moe_out``) against
the existing per-operator baseline (the un-fused L0 path and the fp16
per-operator golden :func:`sm70_moe_reference`) **at identical shapes**, and
wires the debug logging required by R7.2 — it prints, for every shape, the
selected token layout (``contiguous`` / ``masked``), the fused-path gate
fallback ``reason`` (from :func:`sm70_fused_support`), and the key shape
parameters ``(M, K, I, E, topk, group_size)``.

Design context (honest framing, R1.6 / R2.6 / R2.7): fusion saves *overhead*
(eliminating the ``[M, 2*I]`` intermediate-activation HBM round trip in prefill,
and kernel-launch count in decode); it does **not** change the underlying fp16
GEMM throughput ceiling (V100 has no FP8/FP4 tensor cores). The comparison table
should be read with that in mind.

Weights are dsv4f's native **MXFP4** expert weights (E2M1 FP4 nibbles packed
two-per-byte plus a per-32 E8M0 block scale, ``group_size`` = 32), built via
:meth:`SM70MXFP4QuantParams.from_mxfp4_weights`. Every variant decodes them with
the *single shared* :func:`mxfp4_dequant_to_fp16` software dequant (no FP4/FP8
tensor cores on SM70, R2.2), so the per-operator baseline and the fused path
compare like-for-like.

Variants timed (all compute ``float16`` — V100 has no bfloat16):

* ``reference`` — :func:`sm70_moe_reference`, the strictly per-operator fp16
  ``linear1 -> SiluAndMul -> linear2 -> combine`` golden. Used as the baseline.
* ``L0`` — un-fused baseline realised by :class:`SM70FusedMoEExperts` (three
  separate steps over the grouped/contiguous or masked layout).
* ``L1`` — linear1 + SwiGLU epilogue fused (Python-level intermediate dropped).
* ``L2`` / ``L3`` — the hand-written fused CUDA mega-kernel
  ``ops.sm70_fused_moe_out`` (on-chip ``linear1 -> SwiGLU -> linear2`` hand-off).
  Requires a built extension on a real V100; **degrades gracefully** (the
  variant is reported as unavailable with a clear note) when the op is missing
  or the device is not SM70, while the L0/L1/reference variants are still timed.

Graceful degradation summary:

* No CUDA at all -> only the masked-layout per-operator variants + reference run
  (the contiguous layout needs the ``moe_permute`` CUDA op); the fused kernel is
  skipped with a note. The CLI still parses and ``--self-test`` still prints a
  comparison table from synthetic rows.
* CUDA present but ``sm70_fused_moe_out`` not built -> L0/L1/reference run; the
  fused L2/L3 variants are reported unavailable ("op not built").

The script is importable (no work at import time), exposes the comparison data
programmatically (``SM70MoEBenchmark.run`` returns a list of
:class:`BenchmarkResult`; :func:`format_comparison_table` renders it), and has a
``main`` CLI that runs a small default sweep and prints the table — so the smoke
test (task 7.4) and the archiver (task 7.2) can consume it.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

# Ensure the workspace-root ``vllm`` (which carries the SM70 fused-MoE modules
# this benchmark drives) wins over any stale *editable* vllm install that may
# point at a different worktree. When run as a script, ``sys.path[0]`` is this
# file's directory, so ``import vllm`` would otherwise resolve via the editable
# finder to wherever it was installed from — not necessarily this checkout.
# Prepending the repo root makes the workspace copy authoritative; it is a
# no-op when the env already imports vllm from here.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ``Path`` form of the repo root, used by the archiver (R7.3) to resolve a
# relative ``测试结果/`` out-dir against the checkout rather than the process CWD.
_REPO_ROOT_PATH = Path(_REPO_ROOT)

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusionLevel,
    sm70_fused_support,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_reference import (
    sm70_moe_reference,
)

# Namespace the logger under ``vllm.`` so it inherits the vllm logging handler
# and ``VLLM_LOGGING_LEVEL`` (the config only configures the ``vllm`` logger
# tree). Run as a script ``__name__`` is ``"__main__"`` and imported it is the
# bare module name — neither is under ``vllm``, so without this prefix the R7.2
# debug line (layout / fallback reason / shape) would never surface even with
# ``VLLM_LOGGING_LEVEL=DEBUG``.
logger = init_logger(f"vllm.benchmarks.{Path(__file__).stem}")

__all__ = [
    "MoEShape",
    "VariantTiming",
    "BenchmarkResult",
    "SM70MoEBenchmark",
    "format_comparison_table",
    "archive_results",
]

# Default archive root + subject label for the 测试结果/ convention (R7.3).
_ARCHIVE_DIR = "测试结果"
_ARCHIVE_SUBJECT = "sm70-fused-moe"

# MXFP4 packs two E2M1 nibbles per byte (uint8) along the contraction axis.
_MXFP4_PACK_FACTOR = 2
# dsv4f MXFP4 micro-scaling block: one E8M0 (uint8) block scale per 32 elements.
_MXFP4_GROUP_SIZE = 32
# The baseline variant every fused level is compared against.
_BASELINE_VARIANT = "reference"
# SM70 (Tesla V100) compute capability that enables the fused path.
_SM70_CAPABILITY = (7, 0)


@dataclass(frozen=True)
class MoEShape:
    """A single MoE problem shape to benchmark.

    Attributes:
        M: number of tokens (the GEMM ``M`` dim).
        K: hidden size (``linear1`` contraction / ``linear2`` output).
        I: per-expert FFN intermediate size.
        E: number of experts.
        topk: experts routed per token.
        group_size: dsv4f MXFP4 micro-scaling block size (32 — one E8M0 block
            scale per 32 contiguous E2M1 elements along the contraction axis).
    """

    M: int
    K: int
    I: int
    E: int
    topk: int
    group_size: int = _MXFP4_GROUP_SIZE

    def label(self) -> str:
        return (
            f"M={self.M} K={self.K} I={self.I} E={self.E} "
            f"topk={self.topk} gs={self.group_size}"
        )

    def flops(self) -> int:
        """Approximate MoE FFN FLOPs for this shape.

        ``num_slots = M * topk`` token-expert assignments each run linear1
        (``2*K*2I``) and linear2 (``2*I*K``) MACs, i.e. ``6 * num_slots * K * I``
        floating point ops. SwiGLU is negligible and omitted.
        """
        num_slots = self.M * self.topk
        return 6 * num_slots * self.K * self.I


@dataclass
class VariantTiming:
    """Timing outcome for one variant at one shape.

    ``available`` is ``False`` when the variant could not run (e.g. the fused op
    is not built, or the contiguous layout op is missing on a CPU host); in that
    case ``note`` explains why and the latency fields stay ``None``.
    """

    name: str
    available: bool
    latency_ms: float | None = None
    tokens_per_s: float | None = None
    gflops: float | None = None
    note: str = ""


@dataclass
class BenchmarkResult:
    """The full comparison for one shape (programmatic output of ``run``)."""

    shape: MoEShape
    layout: Literal["contiguous", "masked"]
    gate_enabled: bool
    gate_reason: str | None
    device: str
    baseline: str = _BASELINE_VARIANT
    variants: list[VariantTiming] = field(default_factory=list)

    def baseline_latency_ms(self) -> float | None:
        for v in self.variants:
            if v.name == self.baseline and v.available:
                return v.latency_ms
        return None

    def speedup(self, variant: VariantTiming) -> float | None:
        """Speedup of ``variant`` over the baseline (``baseline_ms / ms``)."""
        base = self.baseline_latency_ms()
        if base is None or not variant.available or not variant.latency_ms:
            return None
        return base / variant.latency_ms


# --------------------------------------------------------------------------- #
# Synthetic weight / routing construction
# --------------------------------------------------------------------------- #


def _rand_mxfp4_packed(
    shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """Random uint8 tensor of packed E2M1 nibbles (two FP4 codes per byte).

    The exact bit pattern is irrelevant for a *timing* benchmark — the shared
    :func:`mxfp4_dequant_to_fp16` decode masks each byte into two 4-bit codes,
    so any uint8 yields valid (sign + 3-bit magnitude) E2M1 pairs.
    """
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)


def _rand_mxfp4_scale(
    shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    """Random uint8 E8M0 block-scale exponents kept near the bias (127).

    The decoded multiplier is ``2 ** (raw - 127)``; drawing ``raw`` from
    ``[122, 128)`` keeps it in ``[2**-5, 2**0)`` so the decoded fp16 weights
    stay O(1) and the per-operator reference does not overflow (overflow is
    Property 4's concern, not this timing benchmark's). Mirrors the scale draw
    in ``test_sm70_fused_moe_mxfp4_levels.py``.
    """
    return torch.randint(122, 128, shape, dtype=torch.uint8, device=device)


def build_synthetic_quant_params(
    shape: MoEShape,
    device: torch.device,
) -> SM70MXFP4QuantParams:
    """Build a :class:`SM70MXFP4QuantParams` from random dsv4f MXFP4 weights.

    Mirrors the dsv4f / ``Mxfp4SM70MoEMethod.create_weights`` orientation
    consumed by :meth:`SM70MXFP4QuantParams.from_mxfp4_weights` — packed E2M1
    nibbles (two per uint8 byte along the contraction axis) plus a per-32 E8M0
    (uint8) block scale. Values are random (the timing benchmark only cares
    about shapes / dtypes); the L0/L1 / masked path de-quantizes them to dense
    fp16 via the shared :func:`mxfp4_dequant_to_fp16` decode and the fused L2/L3
    kernel decodes the same bytes on chip.

    MXFP4 tensor shapes (``E`` experts, hidden ``K``, intermediate ``I``,
    ``group_size`` = 32):

    * ``w13_weight``       ``uint8 [E, 2*I, K // 2]``  gate/up (linear1)
    * ``w13_weight_scale`` ``uint8 [E, 2*I, K // gs]``
    * ``w2_weight``        ``uint8 [E, K, I // 2]``    down (linear2)
    * ``w2_weight_scale``  ``uint8 [E, K, I // gs]``

    Unlike the legacy AWQ int4 pack this carries **no** ``StridedPtr`` arrays —
    the MXFP4 op consumes the packed weight/scale tensors directly — so no
    CUDA-only weight-prep step is needed and the pack builds on a CPU host too.

    Args:
        shape: the MoE problem shape.
        device: target device for the weight tensors.
    """
    E, K, I, gs = shape.E, shape.K, shape.I, shape.group_size
    two_i = 2 * I

    w13_weight = _rand_mxfp4_packed((E, two_i, K // _MXFP4_PACK_FACTOR), device)
    w13_weight_scale = _rand_mxfp4_scale((E, two_i, K // gs), device)

    w2_weight = _rand_mxfp4_packed((E, K, I // _MXFP4_PACK_FACTOR), device)
    w2_weight_scale = _rand_mxfp4_scale((E, K, I // gs), device)

    return SM70MXFP4QuantParams.from_mxfp4_weights(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        group_size=gs,
    )


def build_synthetic_routing(
    shape: MoEShape, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build random ``(x, topk_weights, topk_ids)`` for a shape.

    ``x`` is fp16 ``[M, K]``; ``topk_weights`` is fp32 ``[M, topk]`` (the combine
    op / reference expect fp32 weights); ``topk_ids`` is int32 ``[M, topk]``.
    """
    M, K, E, topk = shape.M, shape.K, shape.E, shape.topk
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    logits = torch.randn((M, E), dtype=torch.float32, device=device)
    probs = torch.softmax(logits, dim=-1)
    weights, ids = torch.topk(probs, topk, dim=-1)
    return x, weights.to(torch.float32), ids.to(torch.int32)


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #


def _time_fn(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    """Time ``fn`` and return the mean latency in milliseconds.

    Uses CUDA events on a GPU device (after a warmup + ``synchronize``) and
    ``time.perf_counter`` on CPU. ``fn`` is expected to run the full variant
    once per call.
    """
    for _ in range(max(1, warmup)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters
    start_t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start_t) * 1e3 / iters


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #


class SM70MoEBenchmark:
    """Drive the fused-vs-baseline SM70 MoE comparison (R7.1 / R7.2).

    For each :class:`MoEShape` the benchmark:

    1. builds synthetic dsv4f MXFP4 weights + routing,
    2. resolves the token layout via :meth:`SM70FusedMoEExperts.select_layout`
       (prefill -> ``contiguous``, decode / small M -> ``masked``) and the fused
       gate decision / fallback ``reason`` via :func:`sm70_fused_support`,
    3. emits the R7.2 debug log (layout / reason / shape), and
    4. times the ``reference``, ``L0``, ``L1`` and (if runnable) ``L2`` / ``L3``
       variants, recording latency / throughput.

    The fused CUDA kernel and the contiguous layout op may be unavailable
    (no V100, op not built, CPU host); each variant degrades independently to an
    unavailable :class:`VariantTiming` with an explanatory note rather than
    aborting the run.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        warmup: int = 3,
        iters: int = 10,
        fusion_levels: tuple[SM70FusionLevel, ...] = (
            SM70FusionLevel.L0,
            SM70FusionLevel.L1,
            SM70FusionLevel.L2,
        ),
        m_block: int = 32,
        i_block: int = 64,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.warmup = warmup
        self.iters = iters
        self.fusion_levels = fusion_levels
        self.m_block = m_block
        self.i_block = i_block
        self.seed = seed
        self.experts = SM70FusedMoEExperts()

    # -- introspection ----------------------------------------------------- #

    def _device_capability(self) -> tuple[int, int]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return (0, 0)
        try:
            return tuple(torch.cuda.get_device_capability(self.device))
        except Exception:
            return (0, 0)

    def _device_guard(self):
        """Context manager that makes ``self.device`` the current CUDA device.

        The worktree ``moe_permute`` / ``moe_unpermute`` CUDA ops (used by the
        contiguous layout) operate on the *current* CUDA device and ignore the
        device carried by their input tensors — so timing a non-default device
        (e.g. ``--device cuda:1``) without setting the current device makes
        those ops read/write the wrong GPU and raise an illegal memory access
        (which would poison the CUDA context for the whole run, not just the one
        variant). Setting the current device for the per-shape work keeps the
        ops on the intended GPU. A no-op on CPU.
        """
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.device(self.device)
        return contextlib.nullcontext()

    def _device_label(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return "cpu"
        try:
            return f"{torch.cuda.get_device_name(self.device)} (cap {self._device_capability()})"
        except Exception:
            return str(self.device)

    def _fused_op_available(self) -> bool:
        c = getattr(torch.ops, "_C", None)
        return c is not None and hasattr(c, "sm70_fused_moe_out")

    # -- per-shape run ----------------------------------------------------- #

    def run_shape(self, shape: MoEShape) -> BenchmarkResult:
        """Benchmark a single shape and return its :class:`BenchmarkResult`."""
        torch.manual_seed(self.seed)
        capability = self._device_capability()

        # Layout selection (R2.6 / R2.7): prefill -> contiguous, decode -> masked.
        # Pass an explicit capture flag so the decision is pure / reproducible.
        layout = self.experts.select_layout(shape.M, is_graph_capturing=False)

        # Fused-path gate decision + fallback reason (R7.2). dsv4f MXFP4 path.
        support = sm70_fused_support(
            device_capability=capability,
            flag_enabled=True,
            weight_format="mxfp4",
            group_size=shape.group_size,
            activation="silu",
            hidden=shape.K,
            intermediate=shape.I,
            dtype=torch.float16,
        )

        result = BenchmarkResult(
            shape=shape,
            layout=layout,
            gate_enabled=support.enabled,
            gate_reason=support.reason,
            device=self._device_label(),
        )

        # R7.2 debug log: selected layout, fallback reason, key shape params.
        logger.debug(
            "SM70 MoE bench shape: M=%d K=%d I=%d E=%d topk=%d group_size=%d "
            "| layout=%s | gate_enabled=%s reason=%s | device=%s",
            shape.M,
            shape.K,
            shape.I,
            shape.E,
            shape.topk,
            shape.group_size,
            layout,
            support.enabled,
            support.reason,
            result.device,
        )

        # Build synthetic MXFP4 inputs + time every variant under a device guard:
        # the contiguous-layout moe_permute/moe_unpermute ops act on the *current*
        # CUDA device (not the tensors' device), so a non-default ``--device
        # cuda:N`` must be made current here or those ops raise an illegal memory
        # access that poisons the whole run (see ``_device_guard``). No-op on CPU.
        with self._device_guard():
            # The MXFP4 pack carries no StridedPtr arrays, so it builds on CPU or
            # GPU alike (the fused L2/L3 kernel still needs the compiled op + a
            # V100, handled per-variant below).
            try:
                quant = build_synthetic_quant_params(shape, self.device)
                x, topk_weights, topk_ids = build_synthetic_routing(
                    shape, self.device
                )
            except Exception as e:  # pragma: no cover - setup failure is reported
                for name in ("reference", *(lvl.value for lvl in self.fusion_levels)):
                    result.variants.append(
                        VariantTiming(
                            name=name, available=False, note=f"setup failed: {e}"
                        )
                    )
                return result

            # 1. Reference (per-operator fp16 golden) — the baseline. It consumes
            #    the packed MXFP4 weights directly and decodes them with the
            #    shared ``mxfp4_dequant_to_fp16`` (same numeric source as fused).
            result.variants.append(
                self._time_variant(
                    "reference",
                    lambda: sm70_moe_reference(
                        x,
                        topk_weights,
                        topk_ids,
                        quant.w13_weight,
                        quant.w13_weight_scale,
                        quant.w2_weight,
                        quant.w2_weight_scale,
                        shape.E,
                        activation="silu",
                        group_size=shape.group_size,
                    ),
                    shape,
                )
            )

            # 2. Fusion levels via the orchestrator.
            for level in self.fusion_levels:
                result.variants.append(
                    self._time_fusion_level(
                        level, x, topk_weights, topk_ids, quant, layout
                    )
                )

        return result

    def _time_fusion_level(
        self,
        level: SM70FusionLevel,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant: SM70MXFP4QuantParams,
        layout: Literal["contiguous", "masked"],
    ) -> VariantTiming:
        """Time one fusion level, degrading gracefully when it cannot run."""
        is_fused_kernel = level in (SM70FusionLevel.L2, SM70FusionLevel.L3)

        # The fused CUDA mega-kernel (L2/L3) consumes the contiguous layout only
        # and needs the built op on a real V100. Report unavailable up-front with
        # a precise reason instead of letting the call raise mid-timing.
        if is_fused_kernel and not self._fused_op_available():
            return VariantTiming(
                name=level.value,
                available=False,
                note="fused op sm70_fused_moe_out not built (needs V100 extension)",
            )

        forced_layout: Literal["contiguous", "masked"] = (
            "contiguous" if is_fused_kernel else layout
        )

        def _call() -> object:
            return self.experts.forward(
                x,
                topk_weights,
                topk_ids,
                quant=quant,
                layout=forced_layout,
                m_block=self.m_block,
                i_block=self.i_block,
                fusion_level=level,
            )

        return self._time_variant(level.value, _call, MoEShape(
            M=x.shape[0], K=quant.hidden_K, I=quant.inter_I,
            E=quant.num_experts, topk=topk_ids.shape[1],
            group_size=quant.group_size,
        ))

    def _time_variant(
        self, name: str, fn: Callable[[], object], shape: MoEShape
    ) -> VariantTiming:
        """Run + time ``fn`` once timed, converting failures into a note."""
        try:
            # Smoke the call once so failures surface before the timing loop.
            fn()
            latency_ms = _time_fn(
                fn, device=self.device, warmup=self.warmup, iters=self.iters
            )
        except Exception as e:
            return VariantTiming(name=name, available=False, note=f"{type(e).__name__}: {e}")

        latency_s = latency_ms / 1e3
        tokens_per_s = shape.M / latency_s if latency_s > 0 else None
        gflops = shape.flops() / latency_s / 1e9 if latency_s > 0 else None
        return VariantTiming(
            name=name,
            available=True,
            latency_ms=latency_ms,
            tokens_per_s=tokens_per_s,
            gflops=gflops,
        )

    def run(self, shapes: list[MoEShape]) -> list[BenchmarkResult]:
        """Benchmark every shape and return the list of results."""
        results: list[BenchmarkResult] = []
        for shape in shapes:
            logger.info("Benchmarking %s ...", shape.label())
            results.append(self.run_shape(shape))
        return results


# --------------------------------------------------------------------------- #
# Table formatting (pure — consumed by 7.2 archiver / 7.4 smoke test)
# --------------------------------------------------------------------------- #


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def _sanitize_note(note: str, *, max_len: int = 80) -> str:
    """Collapse a (possibly multi-line) note into one truncated table cell.

    Variant failure notes can carry multi-line exception text (e.g. a CUDA
    error with a stacktrace hint); flatten newlines/extra whitespace and cap the
    length so the rendered table stays aligned and readable.
    """
    flat = " ".join(note.split())
    if len(flat) > max_len:
        flat = flat[: max_len - 1].rstrip() + "…"
    return flat


def format_comparison_table(
    results: list[BenchmarkResult], *, fmt: Literal["plain", "markdown"] = "plain"
) -> str:
    """Render results as a fused-vs-baseline comparison table.

    One row per (shape, variant) with latency (ms), throughput (tokens/s and
    GFLOP/s) and the speedup over the baseline variant. Pure string formatting —
    takes only :class:`BenchmarkResult` data, so it is exercisable on synthetic
    rows without a GPU (task 7.4 smoke test).
    """
    headers = [
        "shape",
        "layout",
        "variant",
        "latency_ms",
        "tokens/s",
        "GFLOP/s",
        f"speedup_vs_{_BASELINE_VARIANT}",
        "note",
    ]
    rows: list[list[str]] = []
    for res in results:
        shape_label = res.shape.label()
        for v in res.variants:
            speedup = res.speedup(v)
            note = v.note
            if not v.available and not note:
                note = "unavailable"
            rows.append(
                [
                    shape_label,
                    res.layout,
                    v.name,
                    _fmt(v.latency_ms, ".3f"),
                    _fmt(v.tokens_per_s, ",.0f"),
                    _fmt(v.gflops, ".1f"),
                    (f"{speedup:.2f}x" if speedup is not None else "-"),
                    _sanitize_note(note),
                ]
            )

    if fmt == "markdown":
        return _render_markdown(headers, rows)
    return _render_plain(headers, rows)


def _render_plain(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "

    def line(cells: list[str]) -> str:
        return sep.join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [line(headers), sep.join("-" * w for w in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def _render_markdown(headers: list[str], rows: list[list[str]]) -> str:
    def esc(cell: str) -> str:
        # Escape pipes so a note never breaks the markdown table columns.
        return cell.replace("|", "\\|")

    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(esc(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Result archiving (R7.3 — write to the repo's 测试结果/ convention)
# --------------------------------------------------------------------------- #


def _sanitize_path_component(text: str) -> str:
    """Make ``text`` safe for the comma-delimited 测试结果/ filename scheme.

    The convention encodes several fields in one filename separated by commas
    (``device,version,subject,timestamp``), so a component must not itself
    contain a comma. Collapse whitespace to ``_`` and drop the field separator
    and path separators so a value never corrupts the field layout or escapes
    its directory.
    """
    flat = "_".join(str(text).split())
    for ch in (",", "/", "\\"):
        flat = flat.replace(ch, "_")
    return flat.strip("_") or "unknown"


def _repo_version_label() -> str:
    """Derive the ``<repo>-<version>`` label used in archive filenames.

    Follows the existing artifacts (e.g. ``1Cat-vLLM-0.0.2``), which match this
    checkout's ``README.md`` H1 heading. Falls back to the installed ``vllm``
    version and finally a static label, so the archiver never hard-fails on a
    missing/edited README.
    """
    try:
        readme = _REPO_ROOT_PATH / "README.md"
        for line in readme.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return _sanitize_path_component(stripped[2:])
    except Exception:
        pass
    try:
        from vllm.version import __version__ as _vllm_version

        return _sanitize_path_component(f"1Cat-vLLM-{_vllm_version}")
    except Exception:
        return "1Cat-vLLM"


def _archive_device_label(
    results: list[BenchmarkResult], device_count: int | None
) -> str:
    """Build the ``<device>_x<count>`` portion of an archive filename.

    Uses the live CUDA device name + visible-device count when available
    (matching ``Tesla_V100-16G_x4``); on a CPU host (e.g. ``--self-test``) it
    falls back to the device recorded on the first result, or ``cpu``.
    """
    name: str | None = None
    count = device_count
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name()
            if count is None:
                count = torch.cuda.device_count()
        except Exception:
            name = None
    if name is None:
        # Fall back to the recorded device label (strip the " (cap ...)" suffix).
        if results and results[0].device:
            name = results[0].device.split(" (")[0]
        else:
            name = "cpu"
    if count is None or count < 1:
        count = 1
    return f"{_sanitize_path_component(name)}_x{count}"


def _render_archive_document(
    results: list[BenchmarkResult],
    *,
    subject: str,
    device_label: str,
    version_label: str,
    timestamp: str,
) -> str:
    """Render the markdown archive document (metadata header + comparison table).

    The benchmark output is a table, not a screenshot, so the artifact is
    markdown text (``.md``) rather than the ``.png`` used for the serving
    dashboards — but it lands under the same ``测试结果/`` tree and naming scheme
    so it archives alongside them.
    """
    lines = [
        f"# SM70 Fused MoE Benchmark — {subject}",
        "",
        f"- Device: `{device_label}`",
        f"- Repo: `{version_label}`",
        f"- Timestamp: `{timestamp}`",
        f"- Shapes benchmarked: {len(results)}",
        "",
        "## Comparison (fused vs per-operator baseline)",
        "",
    ]
    if results:
        lines.append(format_comparison_table(results, fmt="markdown"))
    else:
        lines.append("_No results to report._")
    lines.append("")
    return "\n".join(lines)


def archive_results(
    results: list[BenchmarkResult],
    *,
    out_dir: str | Path = _ARCHIVE_DIR,
    subject: str = _ARCHIVE_SUBJECT,
    device_count: int | None = None,
    timestamp: str | None = None,
) -> Path:
    """Archive benchmark ``results`` under the repo's ``测试结果/`` convention.

    Writes a markdown comparison document to::

        <out_dir>/<subject>/<device>_x<count>,<repo>-<version>,<subject>,<ts>.md

    matching the existing artifacts (e.g.
    ``测试结果/Qwen3.5-27B-AWQ/tp4/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,Qwen3.5-27B-AWQ,20260321_104628.png``)
    but with a ``.md`` extension because the benchmark output is a table rather
    than a screenshot. Reuses :func:`format_comparison_table` for the body.

    Relative ``out_dir`` is resolved against the workspace repo root (not the
    process CWD) so the artifact always lands in ``<repo>/测试结果/`` regardless
    of where the script is invoked from.

    Args:
        results: the benchmark results to archive (may be empty).
        out_dir: archive root directory (default ``测试结果``).
        subject: subject label used both as the sub-directory and in the
            filename (default ``sm70-fused-moe``).
        device_count: number of GPUs to encode as ``_x<count>``; defaults to the
            visible CUDA device count (or 1 on CPU).
        timestamp: ``YYYYMMDD_HHMMSS`` stamp; defaults to ``datetime.now()``.

    Returns:
        The :class:`Path` of the written archive file.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_safe = _sanitize_path_component(subject)
    device_label = _archive_device_label(results, device_count)
    version_label = _repo_version_label()

    base = Path(out_dir)
    if not base.is_absolute():
        base = _REPO_ROOT_PATH / base
    target_dir = base / subject_safe
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{device_label},{version_label},{subject_safe},{ts}.md"
    out_path = target_dir / filename

    document = _render_archive_document(
        results,
        subject=subject_safe,
        device_label=device_label,
        version_label=version_label,
        timestamp=ts,
    )
    out_path.write_text(document, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Default shapes + synthetic demo (for --self-test / smoke)
# --------------------------------------------------------------------------- #


def default_shapes() -> list[MoEShape]:
    """A small default sweep covering decode (masked) + prefill (contiguous).

    Shapes satisfy the fused kernel tile constraints (K % 8, 2*I % 8,
    K % group_size, I % group_size) for the dsv4f MXFP4 group size (32) so the
    L2/L3 variants can run when the op is built.
    """
    gs = _MXFP4_GROUP_SIZE
    return [
        MoEShape(M=8, K=512, I=256, E=8, topk=2, group_size=gs),  # decode
        MoEShape(M=64, K=512, I=256, E=8, topk=2, group_size=gs),  # prefill
        MoEShape(M=256, K=1024, I=512, E=8, topk=2, group_size=gs),  # prefill
    ]


def _demo_results() -> list[BenchmarkResult]:
    """Synthetic results so ``--self-test`` exercises the table without a GPU."""
    gs = _MXFP4_GROUP_SIZE
    demo = [
        BenchmarkResult(
            shape=MoEShape(M=8, K=512, I=256, E=8, topk=2, group_size=gs),
            layout="masked",
            gate_enabled=False,
            gate_reason="not sm70 (device capability (0, 0) != (7, 0))",
            device="cpu",
            variants=[
                VariantTiming("reference", True, 1.234, 6485.0, 31.5),
                VariantTiming("L0", True, 1.180, 6779.0, 32.9),
                VariantTiming("L1", True, 1.050, 7619.0, 37.0),
                VariantTiming(
                    "L2", False, note="fused op sm70_fused_moe_out not built"
                ),
            ],
        ),
        BenchmarkResult(
            shape=MoEShape(M=256, K=1024, I=512, E=8, topk=2, group_size=gs),
            layout="contiguous",
            gate_enabled=True,
            gate_reason=None,
            device="Tesla V100 (cap (7, 0))",
            variants=[
                VariantTiming("reference", True, 4.500, 56888.0, 186.4),
                VariantTiming("L0", True, 4.200, 60952.0, 199.7),
                VariantTiming("L1", True, 3.900, 65641.0, 215.1),
                VariantTiming("L2", True, 2.800, 91428.0, 299.6),
            ],
        ),
    ]
    return demo


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_shape(spec: str) -> MoEShape:
    """Parse a ``M,K,I,E,topk[,group_size]`` shape spec from the CLI."""
    parts = [int(p) for p in spec.split(",")]
    if len(parts) not in (5, 6):
        raise argparse.ArgumentTypeError(
            f"shape must be 'M,K,I,E,topk[,group_size]', got {spec!r}"
        )
    M, K, I, E, topk = parts[:5]
    gs = parts[5] if len(parts) == 6 else _MXFP4_GROUP_SIZE
    return MoEShape(M=M, K=K, I=I, E=E, topk=topk, group_size=gs)


def build_arg_parser() -> argparse.ArgumentParser:
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description=__doc__)
    except Exception:  # pragma: no cover - fallback if vllm util unavailable
        parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to benchmark on ('cuda', 'cuda:N', or 'cpu').",
    )
    parser.add_argument(
        "--shapes",
        type=_parse_shape,
        nargs="*",
        dest="shapes",
        metavar="M,K,I,E,topk[,gs]",
        help="One or more MoE shapes 'M,K,I,E,topk[,group_size]' "
        "(space separated). Defaults to a small built-in sweep.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--levels",
        type=str,
        default="L0,L1,L2",
        help="Comma-separated fusion levels to benchmark (subset of L0,L1,L2,L3).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["plain", "markdown"],
        default="plain",
        help="Comparison-table output format.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Print a synthetic comparison table (no kernels / GPU) and exit. "
        "Used to validate the CLI + table formatting.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive the comparison table to the repo's 测试结果/ directory "
        "(markdown), following the existing device,version,subject,timestamp "
        "naming convention. Also works with --self-test (archives the synthetic "
        "table) so the path/format can be validated without a GPU.",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=_ARCHIVE_DIR,
        help="Archive root directory (default '测试结果', resolved against the "
        "repo root when relative).",
    )
    parser.add_argument(
        "--archive-subject",
        type=str,
        default=_ARCHIVE_SUBJECT,
        help="Subject label used as the archive sub-directory and in the "
        f"filename (default '{_ARCHIVE_SUBJECT}').",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test:
        demo = _demo_results()
        print(format_comparison_table(demo, fmt=args.format))
        if args.archive:
            path = archive_results(
                demo,
                out_dir=args.archive_dir,
                subject=args.archive_subject,
            )
            print(f"\nArchived comparison table to: {path}")
        return 0

    levels = tuple(
        SM70FusionLevel(tok.strip().upper())
        for tok in args.levels.split(",")
        if tok.strip()
    )
    shapes = args.shapes if args.shapes else default_shapes()

    bench = SM70MoEBenchmark(
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        fusion_levels=levels,
        seed=args.seed,
    )
    print(f"Device: {bench._device_label()}")
    print(f"Fused kernel op available: {bench._fused_op_available()}")
    print(f"Benchmarking {len(shapes)} shape(s): "
          f"{', '.join(s.label() for s in shapes)}")

    results = bench.run(shapes)
    print()
    print(format_comparison_table(results, fmt=args.format))
    if args.archive:
        path = archive_results(
            results,
            out_dir=args.archive_dir,
            subject=args.archive_subject,
        )
        print(f"\nArchived comparison table to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
