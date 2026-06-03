# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-operator fp16 golden reference for the SM70 (V100) fused MoE path.

This module implements :func:`sm70_moe_reference` — a strictly per-operator
``linear1 -> SiluAndMul -> linear2 -> weighted-combine`` reference that runs
entirely in ``float16`` (V100 has no bfloat16) and is the *golden* oracle the
fused SM70 kernel is validated against (Requirement 5.1, design Property 1/4).

dsv4f's V100 MoE expert weights are **MXFP4** (E2M1 nibbles packed two-per-byte
plus a per-32 E8M0 block scale).  The reference therefore consumes the *packed*
weights directly and decodes them with :func:`mxfp4_dequant_to_fp16` — the
**single shared numerical source** the fused CUDA kernel must also reproduce
(design §"Components and Interfaces", R5.1).  Sharing the decode is what lets
the equivalence tests (task 5.5) attribute any mismatch to *fusion-order
rounding* rather than to a different dequant.

It additionally provides:

* :func:`sm70_moe_reference_golden` — the exact same computation carried out in
  ``float32`` over the *same* fp16-dequantised weights.  Comparing the fused
  output against *both* the fp16 reference and the fp32 golden lets us separate
  "fusion error" from "fp16 intrinsic accumulation error" (design §Testing
  Strategy, anti-noise criterion).
* The fp16 error-tolerance helpers from design §Data Models — a relative-L2
  *main* criterion, an element-wise *auxiliary* criterion, and an *anti-noise*
  criterion, all of whose thresholds widen linearly with the problem size
  ``(K, I, topk)`` (larger accumulation depth -> looser bound).

MXFP4 weight-shape convention (matches ``Mxfp4SM70MoEMethod.create_weights`` /
``DeepseekV4MegaMoEExperts`` and the contract block at the top of
``csrc/quantization/awq/awq_sm70_fused_moe.cu``):

* ``x``          : ``[num_tokens, K]``                fp16 activations
* ``w13_packed`` : ``uint8 [num_experts, 2 * I, K // 2]``   gate/up (linear1)
* ``w13_scale``  : ``uint8 [num_experts, 2 * I, K // 32]``  per-32 E8M0 scale
* ``w2_packed``  : ``uint8 [num_experts, K, I // 2]``       down    (linear2)
* ``w2_scale``   : ``uint8 [num_experts, K, I // 32]``      per-32 E8M0 scale
* ``topk_ids``     : ``[num_tokens, topk]``           int routing indices
* ``topk_weights`` : ``[num_tokens, topk]``           combine weights

``K`` is the hidden size, ``I`` the intermediate size.  ``w13`` packs gate then
up along its first (output) axis so that ``silu(gate) * up`` matches the
``torch.ops._C.silu_and_mul`` semantics ``silu(t[..., :I]) * t[..., I:]``.

The module only imports ``torch`` at module scope and resolves the optional C
op lazily, so it can be imported and exercised without a compiled vLLM build.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# --- MXFP4 decode constants (mirror the in-kernel software dequant) ----------
# E2M1 (FP4) magnitude lookup table indexed by the 3-bit magnitude code.  This
# is the inverse of the dsv4f quantiser ``_e2m1_nibble`` (whose bucket
# boundaries [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] are exactly the midpoints
# of these representable magnitudes) and matches the marlin / TurboMind E2M1
# decode used elsewhere in the repo.
_E2M1_MAGNITUDE: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# E8M0 block scale: the stored uint8 is a biased power-of-two exponent, so the
# decoded multiplier is ``2 ** (raw - 127)`` (mirrors ``_quantize_mxfp4_pair``'s
# ``ue8m0 = log2(scale) + 127`` and the kernel's e8m0_decode contract).
_E8M0_BIAS: int = 127
# MXFP4 packs two 4-bit nibbles per byte; the block scale covers 32 elements.
_MXFP4_DEFAULT_GROUP_SIZE: int = 32

# --- fp16 error-tolerance base thresholds (design §Data Models) --------------
# Main criterion: relative L2 ||fused - ref||_2 / (||ref||_2 + EPS) <= 2e-2.
_BASE_REL_L2: float = 2e-2
# Auxiliary element-wise criterion.
_BASE_ATOL: float = 1e-2
_BASE_RTOL: float = 1.6e-2
# Anti-noise margin: relL2(fused, golden) <= relL2(ref, golden) + 5e-3.
_BASE_ANTI_NOISE: float = 5e-3
# Numerical floor for the relative-L2 denominator (matches the design formula).
_REL_L2_EPS: float = 1e-6
# Reference accumulation depth (K0 + I0) and topk used to normalise the linear
# relaxation: at this size the bounds equal the base thresholds above.
_REF_DEPTH: float = 1024.0
_REF_TOPK: int = 2


# --- MXFP4 -> fp16 software dequantisation (shared numerical source, R5.1) ----


def mxfp4_dequant_to_fp16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int = _MXFP4_DEFAULT_GROUP_SIZE,
) -> torch.Tensor:
    """Decode MXFP4 packed weights to a dense ``float16`` tensor.

    Reproduces, in pure torch, the exact in-kernel MXFP4 -> fp16 decode the
    fused SM70 kernel performs (and the TurboMind grouped-GEMM baseline relies
    on), so the per-operator reference and the fused path share *one* numeric
    source (design R5.1).  Per element::

        val_fp16 = e2m1_decode(nibble) * e8m0_decode(block_scale_of_group)

    where ``e2m1_decode`` maps a 4-bit code (sign bit + 3-bit magnitude) to its
    signed E2M1 value via :data:`_E2M1_MAGNITUDE`, and ``e8m0_decode`` reads the
    uint8 block scale as a biased power-of-two exponent ``2 ** (raw - 127)``.

    Nibble layout matches ``awq_sm70_gemm.cu::unpack_mxfp4_to_u16``: byte ``j``
    holds the *even* in-feature element ``2 * j`` in its low nibble and the
    *odd* element ``2 * j + 1`` in its high nibble.  The decode operates on the
    final (in-feature ``K``) axis, so any leading dims — e.g. the expert axis of
    ``[E, N, K // 2]`` — are preserved unchanged.

    Args:
        weight_packed: ``uint8`` tensor ``[..., N, K // 2]`` of packed E2M1
            nibbles (two per byte along the in-feature axis).
        weight_scale: ``uint8`` tensor ``[..., N, K // group_size]`` of E8M0
            block-scale exponents (one per ``group_size`` contiguous elements).
        group_size: MXFP4 block size along the in-feature axis (32 for dsv4f).

    Returns:
        Dense ``float16`` tensor ``[..., N, K]`` (``K == 2 * weight_packed.shape[-1]``).
    """
    if weight_packed.dtype != torch.uint8:
        raise ValueError(
            "mxfp4_dequant_to_fp16: weight_packed must be uint8 (packed E2M1), "
            f"got {weight_packed.dtype}."
        )
    if weight_scale.dtype != torch.uint8:
        raise ValueError(
            "mxfp4_dequant_to_fp16: weight_scale must be uint8 (E8M0), "
            f"got {weight_scale.dtype}."
        )
    if group_size <= 0:
        raise ValueError(
            f"mxfp4_dequant_to_fp16: group_size must be positive, got {group_size}."
        )

    packed_in = weight_packed.shape[-1]
    k = packed_in * 2
    if k % group_size != 0:
        raise ValueError(
            "mxfp4_dequant_to_fp16: decoded in-feature size "
            f"K={k} must be divisible by group_size={group_size}."
        )
    num_groups = k // group_size
    if weight_scale.shape[-1] != num_groups:
        raise ValueError(
            "mxfp4_dequant_to_fp16: weight_scale last dim "
            f"{weight_scale.shape[-1]} must equal K // group_size = {num_groups}."
        )
    if weight_scale.shape[:-1] != weight_packed.shape[:-1]:
        raise ValueError(
            "mxfp4_dequant_to_fp16: leading dims of weight_packed "
            f"{tuple(weight_packed.shape[:-1])} and weight_scale "
            f"{tuple(weight_scale.shape[:-1])} must match."
        )

    device = weight_packed.device

    # 1. Unpack nibbles: low nibble -> even in-feature index, high -> odd.
    packed_i16 = weight_packed.to(torch.int16)
    low = packed_i16 & 0x0F  # [..., N, K//2]
    high = (packed_i16 >> 4) & 0x0F  # [..., N, K//2]
    # Interleave so dense[..., 2j] = low[j], dense[..., 2j+1] = high[j].
    nibbles = torch.stack((low, high), dim=-1).reshape(*weight_packed.shape[:-1], k)

    # 2. E2M1 decode: code (low 3 bits) -> magnitude LUT; bit 3 -> sign.
    mag_lut = torch.tensor(_E2M1_MAGNITUDE, dtype=torch.float32, device=device)
    code = (nibbles & 0x07).to(torch.long)
    magnitude = mag_lut[code]  # [..., N, K] fp32
    sign = (nibbles >> 3) & 0x01  # 0 -> +, 1 -> -
    signed = magnitude * (1.0 - 2.0 * sign.to(torch.float32))

    # 3. E8M0 decode: scale = 2 ** (raw - 127), broadcast over each 32-group.
    scale = torch.exp2(weight_scale.to(torch.float32) - float(_E8M0_BIAS))
    scale = scale.repeat_interleave(group_size, dim=-1)  # [..., N, K]

    # val_fp16 = e2m1 * block_scale; computed in fp32 then stored as fp16, which
    # is exactly what the in-kernel dequant materialises before the fp16 mma.
    return (signed * scale).to(torch.float16)


def _silu_and_mul(gate_up: torch.Tensor, swiglu_limit: float = 0.0) -> torch.Tensor:
    """Compute ``silu(gate) * up`` for a ``[..., 2 * I]`` tensor.

    Reuses ``torch.ops._C.silu_and_mul`` (the same op the production path uses)
    when the compiled extension is loaded, and falls back to an equivalent
    torch-native implementation otherwise.  The computation is performed in the
    input dtype (fp16 for the reference path, fp32 for the golden path).

    When ``swiglu_limit > 0`` the DeepSeek-V4 SwiGLU clamp is applied before
    silu/mul (``gate = min(gate, limit)``; ``up = clamp(up, -limit, +limit)``),
    matching ``swiglu_limit_func`` and the fused kernel epilogue.
    """
    assert gate_up.shape[-1] % 2 == 0, (
        f"silu_and_mul expects an even last dim, got {gate_up.shape[-1]}"
    )
    d = gate_up.shape[-1] // 2

    if swiglu_limit and swiglu_limit > 0:
        gate = torch.clamp(gate_up[..., :d], max=swiglu_limit)
        up = torch.clamp(gate_up[..., d:], min=-swiglu_limit, max=swiglu_limit)
        return torch.nn.functional.silu(gate) * up

    c_op = getattr(getattr(torch.ops, "_C", None), "silu_and_mul", None)
    if c_op is not None and gate_up.is_cuda:
        # In-place C op: writes silu(x[..., :d]) * x[..., d:] into `out`.
        gate_up = gate_up.contiguous()
        out = torch.empty(
            (*gate_up.shape[:-1], d), dtype=gate_up.dtype, device=gate_up.device
        )
        c_op(out, gate_up)
        return out

    # Torch-native fallback (CPU, or extension not built).
    gate = gate_up[..., :d]
    up = gate_up[..., d:]
    return torch.nn.functional.silu(gate) * up


def _moe_reference_impl(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    activation: str,
    compute_dtype: torch.dtype,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Shared per-expert ``linear1 -> act -> linear2 -> combine`` core.

    ``w1`` / ``w2`` are *dense* weights (already MXFP4-decoded to fp16 by
    :func:`mxfp4_dequant_to_fp16`).  Every matmul/activation runs in
    ``compute_dtype``.  The output is returned in ``compute_dtype`` and matches
    ``x``'s ``[num_tokens, K]`` shape.
    """
    if activation != "silu":
        raise NotImplementedError(
            f"sm70_moe_reference only supports activation='silu', got {activation!r}"
        )

    num_tokens, hidden_k = x.shape
    topk = topk_ids.shape[1]
    out_k = w2.shape[1]

    xc = x.to(compute_dtype)
    w1c = w1.to(compute_dtype)
    w2c = w2.to(compute_dtype)

    # Expand each token into its `topk` routed slots: [num_tokens * topk, K].
    expanded = xc.view(num_tokens, 1, hidden_k).expand(num_tokens, topk, hidden_k)
    expanded = expanded.reshape(num_tokens * topk, hidden_k)

    flat_ids = topk_ids.reshape(-1)
    slot_out = torch.zeros(
        num_tokens * topk, out_k, dtype=compute_dtype, device=x.device
    )

    # Per-expert grouped GEMM.  Degenerate routing (an expert with 0 tokens, or
    # all tokens funnelled to one expert) is handled by the emptiness check.
    for e in range(num_experts):
        mask = flat_ids == e
        if not bool(mask.any()):
            continue
        inp = expanded[mask]  # [n_e, K]
        # linear1: [n_e, K] @ [K, 2I] -> [n_e, 2I]
        gate_up = inp @ w1c[e].transpose(0, 1)
        # SwiGLU: silu(gate) * up -> [n_e, I]
        h = _silu_and_mul(gate_up, swiglu_limit)
        # linear2: [n_e, I] @ [I, K] -> [n_e, K]
        slot_out[mask] = h @ w2c[e].transpose(0, 1)

    # Weighted combine over the topk slots.
    weighted = slot_out.view(num_tokens, topk, out_k) * topk_weights.view(
        num_tokens, topk, 1
    ).to(compute_dtype)
    return weighted.sum(dim=1)


def sm70_moe_reference(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    group_size: int = _MXFP4_DEFAULT_GROUP_SIZE,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """fp16 per-operator golden reference for the SM70 fused MoE forward.

    Decodes the dsv4f MXFP4 expert weights with :func:`mxfp4_dequant_to_fp16`
    (the shared numeric source, R5.1) and then computes, strictly per operator
    and entirely in ``float16``::

        linear1 -> SiluAndMul -> linear2 -> weighted combine

    Args:
        x: ``[num_tokens, K]`` activations (cast to fp16).
        topk_weights: ``[num_tokens, topk]`` combine weights.
        topk_ids: ``[num_tokens, topk]`` routed expert indices.
        w13_packed: ``uint8 [num_experts, 2 * I, K // 2]`` gate/up MXFP4 weights.
        w13_scale: ``uint8 [num_experts, 2 * I, K // group_size]`` E8M0 scales.
        w2_packed: ``uint8 [num_experts, K, I // 2]`` down MXFP4 weights.
        w2_scale: ``uint8 [num_experts, K, I // group_size]`` E8M0 scales.
        num_experts: total number of experts ``E``.
        activation: gated activation; only ``"silu"`` is supported.
        group_size: MXFP4 block size (32 for dsv4f).
        swiglu_limit: DeepSeek-V4 SwiGLU clamp limit; ``<= 0`` disables it.

    Returns:
        ``[num_tokens, K]`` fp16 output.
    """
    w1 = mxfp4_dequant_to_fp16(w13_packed, w13_scale, group_size)
    w2 = mxfp4_dequant_to_fp16(w2_packed, w2_scale, group_size)
    return _moe_reference_impl(
        x,
        topk_weights,
        topk_ids,
        w1,
        w2,
        num_experts,
        activation,
        compute_dtype=torch.float16,
        swiglu_limit=swiglu_limit,
    )


def sm70_moe_reference_golden(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    group_size: int = _MXFP4_DEFAULT_GROUP_SIZE,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """fp32 "golden" variant of :func:`sm70_moe_reference`.

    Decodes the *same* MXFP4 weights to fp16 (identical numeric source) but
    carries the matmuls/activation out in ``float32``.  Used as the ground truth
    to separate *fusion error* from *fp16 intrinsic accumulation error* via the
    anti-noise tolerance criterion.  Returns a ``[num_tokens, K]`` fp32 tensor.
    """
    w1 = mxfp4_dequant_to_fp16(w13_packed, w13_scale, group_size)
    w2 = mxfp4_dequant_to_fp16(w2_packed, w2_scale, group_size)
    return _moe_reference_impl(
        x,
        topk_weights,
        topk_ids,
        w1,
        w2,
        num_experts,
        activation,
        compute_dtype=torch.float32,
        swiglu_limit=swiglu_limit,
    )


# --- fp16 error-tolerance helpers (design §Data Models) ----------------------


@dataclass(frozen=True)
class Fp16ToleranceBounds:
    """Concrete tolerance thresholds for a given problem size.

    ``scale`` records the linear relaxation factor applied to the base
    thresholds (``>= 1.0``); it is exposed for diagnostics/logging.
    """

    rel_l2: float
    atol: float
    rtol: float
    anti_noise_margin: float
    scale: float


def fp16_tolerance_bounds(K: int, I: int, topk: int) -> Fp16ToleranceBounds:
    """Return fp16 tolerance thresholds scaled by problem size.

    Larger accumulation depth (``K`` feeds linear1, ``I`` feeds linear2) and a
    larger ``topk`` (more terms summed during combine) accumulate more fp16
    rounding error, so the design lets the thresholds widen *linearly* with
    problem size.  At the reference size (``K + I == 1024``, ``topk == 2``) the
    scale is ``1.0`` and the bounds equal the design base thresholds.  The scale
    is clamped to ``>= 1.0`` so the bounds never tighten below the base values.

    Args:
        K: hidden size (linear1 contraction depth).
        I: intermediate size (linear2 contraction depth).
        topk: number of experts combined per token.

    Returns:
        :class:`Fp16ToleranceBounds` with the scaled thresholds.
    """
    depth_scale = (float(K) + float(I)) / _REF_DEPTH
    topk_scale = 1.0 + 0.25 * float(max(0, int(topk) - _REF_TOPK))
    scale = max(1.0, depth_scale * topk_scale)
    return Fp16ToleranceBounds(
        rel_l2=_BASE_REL_L2 * scale,
        atol=_BASE_ATOL * scale,
        rtol=_BASE_RTOL * scale,
        anti_noise_margin=_BASE_ANTI_NOISE * scale,
        scale=scale,
    )


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative L2 norm ``||actual - reference||_2 / (||reference||_2 + eps)``.

    Both tensors are promoted to fp32 before the norm so the metric itself does
    not suffer fp16 rounding.
    """
    a = actual.detach().to(torch.float32)
    r = reference.detach().to(torch.float32)
    diff = torch.linalg.vector_norm(a - r)
    denom = torch.linalg.vector_norm(r) + _REL_L2_EPS
    return float((diff / denom).item())


@dataclass(frozen=True)
class Fp16CheckResult:
    """Outcome of :func:`check_fp16_close`."""

    passed: bool
    rel_l2: float
    rel_l2_bound: float
    elementwise_close: bool
    anti_noise_passed: bool
    fused_golden_rel_l2: float | None
    ref_golden_rel_l2: float | None
    anti_noise_margin: float
    bounds: Fp16ToleranceBounds
    reason: str


def check_fp16_close(
    fused: torch.Tensor,
    ref: torch.Tensor,
    *,
    K: int,
    I: int,
    topk: int,
    golden: torch.Tensor | None = None,
) -> Fp16CheckResult:
    """Check the fused output against the fp16 reference under fp16 tolerances.

    Pass condition (design §Data Models / Property 1):

    * **Primary** — the *main* relative-L2 criterion **or** the *auxiliary*
      element-wise ``allclose`` criterion holds, and
    * **Anti-noise** — when an fp32 ``golden`` is supplied,
      ``relL2(fused, golden) <= relL2(ref, golden) + margin`` (the fused path is
      not meaningfully worse than the per-operator fp16 reference).

    Thresholds are scaled by ``(K, I, topk)`` via :func:`fp16_tolerance_bounds`.

    Args:
        fused: output of the fused SM70 path.
        ref: output of :func:`sm70_moe_reference` (fp16).
        K: hidden size.
        I: intermediate size.
        topk: experts combined per token.
        golden: optional output of :func:`sm70_moe_reference_golden` (fp32).

    Returns:
        :class:`Fp16CheckResult` describing every sub-criterion and the verdict.
    """
    bounds = fp16_tolerance_bounds(K, I, topk)

    rl2 = relative_l2(fused, ref)
    main_pass = rl2 <= bounds.rel_l2

    elementwise_close = bool(
        torch.allclose(
            fused.detach().to(torch.float32),
            ref.detach().to(torch.float32),
            atol=bounds.atol,
            rtol=bounds.rtol,
        )
    )
    primary_pass = main_pass or elementwise_close

    if golden is not None:
        fused_golden = relative_l2(fused, golden)
        ref_golden = relative_l2(ref, golden)
        anti_noise_passed = fused_golden <= ref_golden + bounds.anti_noise_margin
    else:
        fused_golden = None
        ref_golden = None
        anti_noise_passed = True

    passed = primary_pass and anti_noise_passed

    if passed:
        reason = "ok"
    elif not primary_pass:
        reason = (
            f"primary criterion failed: relL2={rl2:.3e} > {bounds.rel_l2:.3e} "
            f"and element-wise allclose(atol={bounds.atol:.3e}, "
            f"rtol={bounds.rtol:.3e}) is False"
        )
    else:
        reason = (
            f"anti-noise criterion failed: relL2(fused,golden)={fused_golden:.3e} "
            f"> relL2(ref,golden)={ref_golden:.3e} + margin="
            f"{bounds.anti_noise_margin:.3e}"
        )

    return Fp16CheckResult(
        passed=passed,
        rel_l2=rl2,
        rel_l2_bound=bounds.rel_l2,
        elementwise_close=elementwise_close,
        anti_noise_passed=anti_noise_passed,
        fused_golden_rel_l2=fused_golden,
        ref_golden_rel_l2=ref_golden,
        anti_noise_margin=bounds.anti_noise_margin,
        bounds=bounds,
        reason=reason,
    )
