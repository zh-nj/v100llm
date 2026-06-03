# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (V100) fused MoE platform gating: switch and configuration.

This module hosts the configuration data models for the experimental SM70
(Tesla V100) fused MoE forward path (``linear1 -> SwiGLU -> linear2`` fused with
intermediate activations kept on-chip). It is feature-gated behind the
``VLLM_SM70_FUSED_MOE`` environment variable, which defaults to off so that the
default behavior is identical to the current release and no existing path is
altered.

All compute on V100 is ``float16`` (bfloat16 is unsupported on SM70).

The fused path directly targets dsv4f's **MXFP4** expert weights (E2M1 mantissa
+ a per-32 block scale); the kernel dequantizes MXFP4 to fp16 in software before
the ``mma.sync`` GEMM (no FP4/FP8 tensor cores on V100). FP8 (E4M3) and AWQ int4
expert weights are **placeholders** this phase: the gate recognizes their format
names but always rejects them (``enabled=False``) with a clear reason.

The pure-function gate ``sm70_fused_support`` (and the thin stateless wrapper
``SM70FusedMoEGate``) decide whether the fused path is supported for a given
``(device_capability, flag, dtype, activation, weight_format, group_size,
shape)`` tuple. The decision is side-effect free so it can be exercised by
property tests (design Property 6); the non-empty ``reason`` returned on
rejection is consumed by the caller for one-shot ``logger.*_once`` fallback
logging (R6.2).
"""

from dataclasses import dataclass
from enum import Enum

import torch

import vllm.envs as envs


class SM70FusionLevel(str, Enum):
    """Progressive fusion levels for the SM70 fused MoE path.

    Each level preserves numerical equivalence (R5) and remains fall-back
    safe (R6); higher levels progressively eliminate intermediate-activation
    HBM round trips.

    - ``L0``: baseline, the existing three independent kernels.
    - ``L1``: linear1 + SwiGLU epilogue fused, reusing buffers to drop the
      Python-level intermediate materialization. (default)
    - ``L2``: eliminate the ``[M, 2*I]`` / ``[M, I]`` HBM round trips by
      chaining linear1 -> linear2.
    - ``L3``: single mega-kernel with fully on-chip token-block flow.
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass(frozen=True)
class SM70FusedConfig:
    """Switch and tiling configuration for the SM70 fused MoE path.

    Mirrors the ``VLLM_SM70_FUSED_MOE`` switch plus the kernel tiling
    parameters. When ``enabled`` is ``False`` (the default), behavior is
    identical to the current release (R6.3).

    Attributes:
        enabled: Master switch, mirrors ``VLLM_SM70_FUSED_MOE``. Defaults to
            ``False`` (off); when off, no existing path is changed.
        m_block: Token-block granularity (typically 16 or 32), aligned to the
            grouped-GEMM M-block requirement (R3.2).
        i_block: Intermediate (FFN) tile size, constrained by the V100 shared
            memory budget.
        fusion_level: Progressive fusion level (``L0``/``L1``/``L2``/``L3``),
            defaults to ``L2`` (the fused CUDA mega-kernel that keeps the
            ``gate/up`` / ``h`` activations on chip, eliminating the
            intermediate-activation HBM round trip for the prefill / contiguous
            path). Falls back to the per-operator path when the kernel is
            unavailable or the layout is masked.
    """

    enabled: bool = False
    m_block: int = 32
    i_block: int = 64
    fusion_level: SM70FusionLevel = SM70FusionLevel.L2

    @classmethod
    def from_env(cls) -> "SM70FusedConfig":
        """Build a config from the current environment.

        Reads the ``VLLM_SM70_FUSED_MOE`` switch (default off) and leaves the
        tiling parameters at their defaults.
        """
        return cls(enabled=envs.VLLM_SM70_FUSED_MOE)


@dataclass(frozen=True)
class SM70FusedSupport:
    """Result of an SM70 fused MoE support/gating decision.

    Attributes:
        enabled: Whether the SM70 fused path is supported and should be used.
        reason: When ``enabled`` is ``False``, a non-empty human-readable
            reason for the fallback (used for one-shot logging by the caller);
            ``None`` when ``enabled`` is ``True``.
    """

    enabled: bool
    reason: str | None = None


# SM70 (Tesla V100) compute capability that gates the fused path.
_SM70_CAPABILITY: tuple[int, int] = (7, 0)
# Only SwiGLU (``silu(gate) * up``) is supported by the fused epilogue (R2.3).
_SUPPORTED_ACTIVATION: str = "silu"
# The fused mma.sync kernel requires K and N to be multiples of 8.
_SHAPE_DIVISOR: int = 8
# MXFP4 is the only enabled expert weight format on SM70 (dsv4f experts).
_MXFP4_WEIGHT_FORMAT: str = "mxfp4"
# MXFP4 micro-scaling block size: one E8M0 block scale per 32 E2M1 elements.
_MXFP4_GROUP_SIZE: int = 32
# Recognized-but-unsupported (placeholder) expert weight formats this phase.
# dsv4f experts are MXFP4; FP8 (E4M3) and AWQ int4 are out of scope for now and
# are rejected with a clear reason rather than silently mis-handled.
_PLACEHOLDER_WEIGHT_FORMATS: tuple[str, ...] = ("fp8", "awq_int4")


def sm70_fused_support(
    *,
    device_capability: tuple[int, int],
    flag_enabled: bool,
    weight_format: str,
    group_size: int | None,
    activation: str,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
) -> SM70FusedSupport:
    """Decide whether the SM70 fused MoE path is supported (pure function).

    This is a side-effect-free decision function (design Property 6). It returns
    ``SM70FusedSupport(enabled=True)`` **if and only if** every one of the
    following holds; otherwise it returns ``enabled=False`` with a non-empty,
    specific ``reason`` for the caller's one-shot fallback log (R6.2):

    1. ``device_capability == (7, 0)`` (Tesla V100 / SM70) (R6.1, R6.4).
    2. ``flag_enabled`` is ``True`` (``VLLM_SM70_FUSED_MOE`` on) (R6.3).
    3. ``dtype is torch.float16`` (bfloat16 unsupported on V100) (R6.2).
    4. ``activation == "silu"`` (SwiGLU epilogue) (R2.3, R6.2).
    5. ``weight_format == "mxfp4"`` (dsv4f MXFP4 experts) (R2.4, R6.2). The
       placeholder formats ``"fp8"`` / ``"awq_int4"`` are recognized but always
       rejected this phase; any other value is likewise rejected.
    6. ``hidden`` (the GEMM ``K`` dim) is a multiple of 8 (R6.2).
    7. ``N = 2 * intermediate`` (gate/up out dim) is a multiple of 8 (R6.2).
    8. ``hidden`` (``K``) is a multiple of the MXFP4 ``group_size`` (32): each
       block of 32 contiguous E2M1 elements along ``K`` shares one E8M0 block
       scale, so ``K`` must tile the MXFP4 blocks exactly (R2.2, R6.2). When
       ``group_size`` is ``None`` it defaults to the MXFP4 block size (32).

    Args:
        device_capability: Reported CUDA compute capability ``(major, minor)``.
        flag_enabled: Whether the ``VLLM_SM70_FUSED_MOE`` switch is on.
        weight_format: Expert weight format. ``"mxfp4"`` enables the fused path
            (dsv4f experts); ``"fp8"`` and ``"awq_int4"`` are placeholders that
            are recognized but always rejected this phase; any other value is
            also rejected.
        group_size: MXFP4 micro-scaling block size (one block scale per this
            many contiguous E2M1 elements). dsv4f uses 32. ``None`` defaults to
            the MXFP4 block size (32).
        activation: MoE activation name; only ``"silu"`` is supported.
        hidden: Hidden size, i.e. the GEMM ``K`` dimension.
        intermediate: Per-expert FFN intermediate size; ``N = 2 * intermediate``.
        dtype: Compute dtype; must be ``torch.float16``.

    Returns:
        An :class:`SM70FusedSupport` with ``enabled`` set and, when disabled, a
        non-empty ``reason`` describing the first unmet condition.
    """
    # 1. Platform: only Tesla V100 (SM70). Non-(7,0) keeps existing behavior.
    if tuple(device_capability) != _SM70_CAPABILITY:
        return SM70FusedSupport(
            enabled=False,
            reason=(
                f"not sm70 (device capability {tuple(device_capability)} "
                f"!= {_SM70_CAPABILITY})"
            ),
        )

    # 2. Master switch: default-off so the default release behavior is unchanged.
    if not flag_enabled:
        return SM70FusedSupport(
            enabled=False,
            reason="flag off (VLLM_SM70_FUSED_MOE not enabled)",
        )

    # 3. dtype: fp16 only (bfloat16 is unsupported on V100).
    if dtype is not torch.float16:
        return SM70FusedSupport(
            enabled=False,
            reason=f"dtype not fp16 (got {dtype}, requires torch.float16)",
        )

    # 4. activation: SwiGLU (silu) only.
    if activation != _SUPPORTED_ACTIVATION:
        return SM70FusedSupport(
            enabled=False,
            reason=(
                f"activation not silu (got {activation!r}, "
                f"requires {_SUPPORTED_ACTIVATION!r})"
            ),
        )

    # 5. weight format: MXFP4 only. FP8 / AWQ int4 are recognized placeholders
    # that are explicitly not implemented this phase (dsv4f experts are MXFP4).
    if weight_format != _MXFP4_WEIGHT_FORMAT:
        if weight_format in _PLACEHOLDER_WEIGHT_FORMATS:
            reason = (
                f"weight format not mxfp4 (got {weight_format!r}; placeholder, "
                f"not implemented this phase)"
            )
        else:
            reason = (
                f"weight format not mxfp4 (got {weight_format!r}; "
                f"requires {_MXFP4_WEIGHT_FORMAT!r})"
            )
        return SM70FusedSupport(enabled=False, reason=reason)

    # 6. K (hidden) must be a multiple of 8 for the mma.sync tiling.
    if hidden % _SHAPE_DIVISOR != 0:
        return SM70FusedSupport(
            enabled=False,
            reason=f"K (hidden={hidden}) not divisible by {_SHAPE_DIVISOR}",
        )

    # 7. N = 2 * intermediate (gate/up output dim) must be a multiple of 8.
    n = 2 * intermediate
    if n % _SHAPE_DIVISOR != 0:
        return SM70FusedSupport(
            enabled=False,
            reason=(
                f"N (2*intermediate={n}) not divisible by {_SHAPE_DIVISOR}"
            ),
        )

    # 8. K (hidden) must tile the MXFP4 blocks exactly: one E8M0 block scale per
    # ``group_size`` (32) contiguous E2M1 elements along K. ``None`` defaults to
    # the MXFP4 block size.
    effective_group_size = (
        _MXFP4_GROUP_SIZE if group_size is None else group_size
    )
    if hidden % effective_group_size != 0:
        return SM70FusedSupport(
            enabled=False,
            reason=(
                f"K (hidden={hidden}) not divisible by "
                f"group_size={effective_group_size}"
            ),
        )

    return SM70FusedSupport(enabled=True, reason=None)


class SM70FusedMoEGate:
    """Stateless gating helper for the SM70 fused MoE path (R6).

    Thin wrapper around the pure function :func:`sm70_fused_support`. It holds
    no mutable state and performs no side effects; the master switch is read
    once from :class:`SM70FusedConfig` (which mirrors ``VLLM_SM70_FUSED_MOE``)
    so the platform/shape/quantization decision stays fully deterministic and
    testable (design Property 6).
    """

    @staticmethod
    def supports(
        *,
        device_capability: tuple[int, int],
        flag_enabled: bool,
        weight_format: str,
        group_size: int | None,
        activation: str,
        hidden: int,
        intermediate: int,
        dtype: torch.dtype,
    ) -> SM70FusedSupport:
        """Delegate to :func:`sm70_fused_support` (pure, side-effect free)."""
        return sm70_fused_support(
            device_capability=device_capability,
            flag_enabled=flag_enabled,
            weight_format=weight_format,
            group_size=group_size,
            activation=activation,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
        )

    @classmethod
    def from_config(
        cls,
        config: SM70FusedConfig,
        *,
        device_capability: tuple[int, int],
        weight_format: str,
        group_size: int | None,
        activation: str,
        hidden: int,
        intermediate: int,
        dtype: torch.dtype,
    ) -> SM70FusedSupport:
        """Decide support using the switch carried by ``config.enabled``."""
        return cls.supports(
            device_capability=device_capability,
            flag_enabled=config.enabled,
            weight_format=weight_format,
            group_size=group_size,
            activation=activation,
            hidden=hidden,
            intermediate=intermediate,
            dtype=dtype,
        )
