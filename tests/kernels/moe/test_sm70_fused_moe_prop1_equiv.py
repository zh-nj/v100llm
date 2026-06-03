# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property test for SM70 fused-MoE vs per-operator numerical equivalence.

Feature: deepgemm-megamoe-sm70-port

This module verifies design **Property 1** ("融合路径与逐算子参考数值等价") of the
SM70 (V100) fused MoE forward (:meth:`SM70FusedMoEExperts.forward`,
``vllm/model_executor/layers/fused_moe/sm70_fused_moe_experts.py``) against the
per-operator fp16 golden reference (:func:`sm70_moe_reference`,
``vllm/model_executor/layers/fused_moe/sm70_moe_reference.py``).

Property 1 (design, Validates Requirements 2.1, 2.2, 2.3, 2.5, 3.3, 5.1, 5.2,
5.3)

    *For any* supported MoE shape ``(M, hidden K, intermediate I, num_experts E,
    topk)``, any fp16 input ``x``, any dsv4f **MXFP4** expert weights
    ``w13/w2`` (E2M1 nibbles packed two-per-byte + per-32 E8M0 block scales),
    any routing ``topk_ids/topk_weights`` — **including** degenerate
    distributions (some expert gets 0 tokens, all-to-one) — and for **both** the
    ``contiguous`` and ``masked`` layouts, the SM70 fused path output SHALL equal
    the per-operator fp16 reference within the agreed fp16 tolerance (main
    criterion relative L2 <= 2e-2, plus the anti-fp16-noise criterion against an
    fp32 golden).

Shared MXFP4 dequant source (R5.1)
----------------------------------
The reference (:func:`sm70_moe_reference`) and every runnable realization of the
fused path decode the **same** MXFP4 pack through the single shared
:func:`mxfp4_dequant_to_fp16` (E2M1 unpack + per-32 E8M0 block scale): the
per-operator L0/L1 levels and the masked path go through
:meth:`SM70MXFP4QuantParams.to_dense_fp16`, and the reference consumes the packed
weights directly. Because they share one numeric source, any mismatch is
attributable to *fusion-order rounding* rather than a divergent dequant.

Tolerance helpers (design §Data Models)
---------------------------------------
The verdict uses :func:`check_fp16_close` (relative-L2 *main* criterion OR
element-wise *auxiliary* ``allclose``, AND — when an fp32 ``golden`` is supplied
— the *anti-noise* criterion ``relL2(fused, golden) <= relL2(ref, golden) +
margin``). Thresholds widen linearly with ``(K, I, topk)`` via
:func:`fp16_tolerance_bounds`. The fp32 golden comes from
:func:`sm70_moe_reference_golden`, so a draw only fails when the fused path is
*meaningfully* worse than the per-operator fp16 reference (not merely when both
share fp16 rounding noise).

Generation domains (design §Testing Strategy, P1 row)
-----------------------------------------------------
* shapes — ``M ∈ [0, 256]`` (incl. the degenerate empty batch), ``K = 32 *
  k_mult`` and ``I = 32 * i_mult`` (multiples of the MXFP4 group size 32, hence
  of 8 — satisfies ``K % 8 == 0`` / ``K % 32 == 0`` / ``2*I % 8 == 0`` and keeps
  ``K``/``I`` even so the two-nibbles-per-byte pack is well-formed), ``E ∈ [1,
  16]``, ``topk ∈ [1, min(E, 8)]``.
* quantization — dsv4f MXFP4 with ``group_size = 32``.
* weights — packed E2M1 nibbles span the full code range; the per-32 E8M0 block
  scales are drawn from two bands ("typical" / "wide") so the de-quantized fp16
  weights cover a moderate and a broader (still overflow-safe) magnitude range.
* routing — uniform / all-to-one / sparse (some experts get 0 tokens), covering
  the degenerate distributions of R5.3.

Scales are kept moderate so the *reference itself* stays finite (fp16 overflow
is Property 4's concern, not Property 1's); a draw whose reference is non-finite
is skipped (equivalence is undefined there).

What this test executes vs. what it defers
------------------------------------------
The hand-written fused CUDA mega-kernel ``ops.sm70_fused_moe_out`` (fusion
levels **L2/L3**) is added by tasks 4.1–4.3 and needs a rebuilt extension on a
real V100; it is not present in the current build. The equivalence is therefore
always exercised on the *runnable* realizations:

* ``contiguous`` + **L0/L1** — the per-operator dense-fp16 ``linear1 -> SwiGLU ->
  linear2`` over the ``moe_permute`` grouped layout, then ``moe_unpermute``
  weighted combine (the fp32-``topk_weights`` contract).
* ``masked`` — the shape-stable ``[E, M, K]`` batched per-operator realization
  (the fused kernel reads the contiguous layout only, so masked is always
  per-operator regardless of fusion level).

When the compiled V100 op *is* available, the **L2/L3** fused-kernel levels are
additionally exercised on the ``contiguous`` layout. ``test_executed_coverage``
prints the executed-vs-skipped matrix for transparency.

The per-operator path (``moe_permute`` / ``moe_unpermute`` + dense fp16 matmul)
needs a CUDA GPU with the compiled ``_moe_C`` op; the module is skipped cleanly
otherwise. Following the repo's V100 target, ``float16`` is the only compute
dtype.

Validates: Requirements 2.1, 2.2, 2.3, 2.5, 3.3, 5.1, 5.2, 5.3
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Force the _moe_C (moe_permute / moe_unpermute) and _C (silu_and_mul / fused)
# op namespaces to register so the support probe and kernels below are present.
from vllm.platforms import current_platform

with contextlib.suppress(ImportError):
    import vllm._C  # noqa: F401
with contextlib.suppress(ImportError):
    import vllm._moe_C  # noqa: F401

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute_unpermute_supported,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusionLevel,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_reference import (
    check_fp16_close,
    sm70_moe_reference,
    sm70_moe_reference_golden,
)

# The per-operator L0/L1 path and the masked path reuse the moe_permute /
# moe_unpermute CUDA ops + dense fp16 matmul; without a GPU + compiled extension
# there is nothing runnable to compare against the reference.
if not torch.cuda.is_available():
    pytest.skip(
        "SM70 fused-equivalence property test requires a CUDA GPU",
        allow_module_level=True,
    )
if current_platform.is_rocm():
    pytest.skip(
        "moe_permute_unpermute is not defined for ROCm",
        allow_module_level=True,
    )
if not moe_permute_unpermute_supported():
    pytest.skip(
        "moe_permute_unpermute is not supported on this platform",
        allow_module_level=True,
    )

# dsv4f MXFP4 block size: one E8M0 block scale per 32 packed E2M1 elements.
_MXFP4_GROUP_SIZE = 32

# The fused CUDA mega-kernel (L2/L3) is only present after the tasks 4.1-4.3
# rebuild on a V100; probe once so we additionally exercise it when available.
_HAS_FUSED_OP = hasattr(getattr(torch.ops, "_C", None), "sm70_fused_moe_out")
# The fused kernel additionally needs a (7,0) device; the per-operator path is
# portable across CUDA GPUs.
_IS_SM70 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 0)

# Routing distributions: uniform spread, all-to-one (every token -> expert 0),
# and sparse (only even experts -> some experts get zero tokens). Covers the
# degenerate distributions called out by R5.3 alongside the regular case.
_ROUTING_MODES = ["uniform", "all_to_one", "sparse"]

# Two E8M0 block-scale bands. Both decode to overflow-safe fp16 weights; "wide"
# simply spans a broader magnitude range than "typical". The raw uint8 stores a
# biased power-of-two exponent (2 ** (raw - 127)), so e.g. raw 125 -> 2**-2.
#   typical: raw in [125, 128) -> scale in {2**-2, 2**-1, 2**0}
#   wide:    raw in [123, 129) -> scale in {2**-4 .. 2**1}
_WEIGHT_REGIMES = {
    "typical": (125, 128),
    "wide": (123, 129),
}


def _make_topk_ids(
    mode: str,
    num_tokens: int,
    num_experts: int,
    topk: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``[num_tokens, topk]`` int32 routing table for a given mode."""
    if mode == "all_to_one":
        return torch.zeros((num_tokens, topk), device=device, dtype=torch.int32)
    if mode == "sparse" and num_experts > 1:
        half = (num_experts + 1) // 2
        ids = torch.randint(
            0, half, (num_tokens, topk), generator=gen, device=device
        )
        return (ids * 2).clamp_(max=num_experts - 1).to(torch.int32)
    return torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        generator=gen,
        device=device,
        dtype=torch.int32,
    )


def _build_case(
    *,
    seed: int,
    k_mult: int,
    i_mult: int,
    num_experts: int,
    topk: int,
    num_tokens: int,
    weight_regime: str,
    routing_mode: str,
    device: torch.device,
    group_size: int = _MXFP4_GROUP_SIZE,
):
    """Construct one supported dsv4f MXFP4 MoE case (+ fp16 inputs).

    Returns ``(quant, x, topk_ids, topk_weights, K, I)`` where ``quant`` is the
    :class:`SM70MXFP4QuantParams` pack the fused path consumes and whose packed
    weights/scales the reference decodes through the *same*
    :func:`mxfp4_dequant_to_fp16` source — so the reference and the fused /
    per-operator paths see identical fp16 weights (equivalence by construction up
    to fp16 rounding order).

    ``topk_weights`` is fp32 because the contiguous ``combine`` reuses the
    ``moe_unpermute`` CUDA op, which requires fp32 router weights.

    MXFP4 weight-shape convention (design §Data Models ``SM70MXFP4QuantParams``):
        w13_weight       uint8 [E, 2*I, K // 2]    (two E2M1 nibbles per byte)
        w13_weight_scale uint8 [E, 2*I, K // 32]   (per-32 E8M0 block scale)
        w2_weight        uint8 [E, K,   I // 2]
        w2_weight_scale  uint8 [E, K,   I // 32]
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    K = group_size * k_mult
    I = group_size * i_mult
    two_i = 2 * I

    # Packed E2M1 nibbles span the entire code range (every byte holds two
    # arbitrary 0..15 nibbles).
    w13_weight = torch.randint(
        0, 256, (num_experts, two_i, K // 2),
        generator=gen, device=device, dtype=torch.uint8,
    )
    w2_weight = torch.randint(
        0, 256, (num_experts, K, I // 2),
        generator=gen, device=device, dtype=torch.uint8,
    )

    # E8M0 block scales drawn from the regime's band -> overflow-safe fp16
    # weights (the reference must stay finite; overflow is Property 4's job).
    lo, hi = _WEIGHT_REGIMES[weight_regime]
    w13_weight_scale = torch.randint(
        lo, hi, (num_experts, two_i, K // group_size),
        generator=gen, device=device, dtype=torch.uint8,
    )
    w2_weight_scale = torch.randint(
        lo, hi, (num_experts, K, I // group_size),
        generator=gen, device=device, dtype=torch.uint8,
    )

    quant = SM70MXFP4QuantParams.from_mxfp4_weights(
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        group_size=group_size,
    )

    # x: ordinary fp16 activations (kept modest so the reference stays finite).
    x = (
        torch.randn(
            (num_tokens, K), generator=gen, device=device, dtype=torch.float16
        )
        * 0.5
    )

    topk_ids = _make_topk_ids(
        routing_mode, num_tokens, num_experts, topk, gen, device
    )

    # Realistic router combine weights in [0, 1] (normalized softmax outputs),
    # in fp32 as the contiguous combine (moe_unpermute) requires.
    topk_weights = torch.rand(
        (num_tokens, topk), generator=gen, device=device, dtype=torch.float32
    )

    return quant, x, topk_ids, topk_weights, K, I


def _runnable_combos() -> list[tuple[str, SM70FusionLevel]]:
    """(layout, fusion_level) combinations exercisable in the current build.

    Always: ``contiguous`` at the per-operator L0/L1 levels and ``masked`` (which
    is per-operator regardless of fusion level). Additionally the fused-kernel
    L2/L3 levels on ``contiguous`` when the compiled V100 op is present.
    """
    combos: list[tuple[str, SM70FusionLevel]] = [
        ("contiguous", SM70FusionLevel.L0),
        ("contiguous", SM70FusionLevel.L1),
        ("masked", SM70FusionLevel.L1),
    ]
    if _HAS_FUSED_OP and _IS_SM70:
        combos += [
            ("contiguous", SM70FusionLevel.L2),
            ("contiguous", SM70FusionLevel.L3),
        ]
    return combos


# Feature: deepgemm-megamoe-sm70-port, Property 1: 融合路径与逐算子参考数值等价
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.differing_executors, HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    k_mult=st.integers(min_value=1, max_value=2),
    i_mult=st.integers(min_value=1, max_value=2),
    num_experts=st.integers(min_value=1, max_value=16),
    topk=st.integers(min_value=1, max_value=8),
    num_tokens=st.integers(min_value=0, max_value=256),
    weight_regime=st.sampled_from(list(_WEIGHT_REGIMES)),
    routing_mode=st.sampled_from(_ROUTING_MODES),
)
def test_prop1_fused_equiv(
    seed: int,
    k_mult: int,
    i_mult: int,
    num_experts: int,
    topk: int,
    num_tokens: int,
    weight_regime: str,
    routing_mode: str,
) -> None:
    """Property 1: the fused path equals the per-operator fp16 reference.

    For any supported shape / dsv4f MXFP4 weights / routing (incl. degenerate
    distributions), every runnable ``(layout, fusion_level)`` realization of
    :meth:`SM70FusedMoEExperts.forward` agrees with :func:`sm70_moe_reference`
    within the design's fp16 tolerance (relative L2 <= 2e-2 plus the anti-noise
    criterion against the fp32 golden). The reference and the fused path share
    the same MXFP4 -> fp16 dequant source (R5.1). A draw whose reference is
    itself non-finite (legitimate fp16 overflow) is out of scope here
    (Property 4) and is skipped.
    """
    topk = min(topk, num_experts)
    device = torch.device("cuda")

    quant, x, topk_ids, topk_weights, K, I = _build_case(
        seed=seed,
        k_mult=k_mult,
        i_mult=i_mult,
        num_experts=num_experts,
        topk=topk,
        num_tokens=num_tokens,
        weight_regime=weight_regime,
        routing_mode=routing_mode,
        device=device,
    )

    # Golden oracles over the SAME MXFP4 pack the fused path decodes: the fp16
    # reference is the equivalence target; the fp32 golden anchors the anti-noise
    # criterion (separates fusion error from fp16 rounding noise). Both decode
    # the packed weights via the shared mxfp4_dequant_to_fp16 (R5.1).
    ref = sm70_moe_reference(
        x,
        topk_weights,
        topk_ids,
        quant.w13_weight,
        quant.w13_weight_scale,
        quant.w2_weight,
        quant.w2_weight_scale,
        num_experts,
        group_size=quant.group_size,
    )
    if not bool(torch.isfinite(ref).all()):
        # Equivalence is undefined when the reference overflows fp16; that
        # regime is Property 4's concern, not Property 1's.
        pytest.skip("reference output is non-finite (fp16 overflow); out of P1 scope")
    golden = sm70_moe_reference_golden(
        x,
        topk_weights,
        topk_ids,
        quant.w13_weight,
        quant.w13_weight_scale,
        quant.w2_weight,
        quant.w2_weight_scale,
        num_experts,
        group_size=quant.group_size,
    )

    experts = SM70FusedMoEExperts()

    for layout, level in _runnable_combos():
        out = experts.forward(
            x,
            topk_weights,
            topk_ids,
            quant=quant,
            layout=layout,
            m_block=32,
            i_block=64,
            fusion_level=level,
        )
        assert out.shape == (num_tokens, quant.hidden_logical_size)
        assert out.dtype == torch.float16

        check = check_fp16_close(out, ref, K=K, I=I, topk=topk, golden=golden)
        assert check.passed, (
            "fused path disagrees with the per-operator fp16 reference: "
            f"{check.reason} | layout={layout}, level={level.value}, "
            f"K={K}, I={I}, E={num_experts}, topk={topk}, "
            f"num_tokens={num_tokens}, gs={quant.group_size}, "
            f"weight_regime={weight_regime}, routing={routing_mode}"
        )


# --- focused equivalence examples (complement the property above) ----------


def test_all_to_one_routing_equivalence() -> None:
    """Degenerate all-to-one routing stays equivalent (R5.3).

    Every token routes to expert 0 in every slot, so all other experts get 0
    tokens. Both the contiguous and masked realizations must still match the
    per-operator reference within fp16 tolerance.
    """
    device = torch.device("cuda")
    quant, x, topk_ids, topk_weights, K, I = _build_case(
        seed=2024,
        k_mult=2,
        i_mult=2,
        num_experts=8,
        topk=2,
        num_tokens=48,
        weight_regime="typical",
        routing_mode="all_to_one",
        device=device,
    )
    ref = sm70_moe_reference(
        x, topk_weights, topk_ids,
        quant.w13_weight, quant.w13_weight_scale,
        quant.w2_weight, quant.w2_weight_scale,
        8, group_size=quant.group_size,
    )
    golden = sm70_moe_reference_golden(
        x, topk_weights, topk_ids,
        quant.w13_weight, quant.w13_weight_scale,
        quant.w2_weight, quant.w2_weight_scale,
        8, group_size=quant.group_size,
    )
    assert torch.isfinite(ref).all()

    experts = SM70FusedMoEExperts()
    for layout in ("contiguous", "masked"):
        out = experts.forward(
            x, topk_weights, topk_ids, quant=quant, layout=layout,
            m_block=32, fusion_level=SM70FusionLevel.L1,
        )
        check = check_fp16_close(out, ref, K=K, I=I, topk=2, golden=golden)
        assert check.passed, f"{layout}: {check.reason}"


def test_empty_batch_returns_empty_output() -> None:
    """A zero-token batch yields an empty, layout-independent output (M=0 edge)."""
    device = torch.device("cuda")
    quant, x, topk_ids, topk_weights, K, I = _build_case(
        seed=5,
        k_mult=1,
        i_mult=1,
        num_experts=4,
        topk=2,
        num_tokens=0,
        weight_regime="typical",
        routing_mode="uniform",
        device=device,
    )
    experts = SM70FusedMoEExperts()
    for layout in ("contiguous", "masked"):
        out = experts.forward(
            x, topk_weights, topk_ids, quant=quant, layout=layout,
            m_block=32, fusion_level=SM70FusionLevel.L1,
        )
        assert out.shape == (0, quant.hidden_logical_size)
        assert out.dtype == torch.float16


def test_executed_coverage(capsys) -> None:
    """Report which (layout, fusion-level) combinations are executed vs skipped.

    Documents the honest coverage of the property test: the per-operator L0/L1
    (contiguous) and masked realizations always run; the fused CUDA mega-kernel
    levels L2/L3 run only when the compiled V100 op is present.
    """
    executed = _runnable_combos()
    all_combos = [
        ("contiguous", SM70FusionLevel.L0),
        ("contiguous", SM70FusionLevel.L1),
        ("masked", SM70FusionLevel.L1),
        ("contiguous", SM70FusionLevel.L2),
        ("contiguous", SM70FusionLevel.L3),
    ]
    executed_set = set(executed)
    lines = ["SM70 Property 1 executed-vs-skipped coverage:"]
    for layout, level in all_combos:
        status = "EXECUTED" if (layout, level) in executed_set else "SKIPPED"
        lines.append(f"  [{status}] layout={layout}, level={level.value}")
    lines.append(
        f"  (fused mega-kernel available={_HAS_FUSED_OP}, is_sm70={_IS_SM70})"
    )
    report = "\n".join(lines)
    with capsys.disabled():
        print("\n" + report)

    # The per-operator realizations must always be exercisable on CUDA.
    assert ("contiguous", SM70FusionLevel.L0) in executed_set
    assert ("contiguous", SM70FusionLevel.L1) in executed_set
    assert ("masked", SM70FusionLevel.L1) in executed_set


def test_swiglu_limit_equivalence() -> None:
    """The DeepSeek-V4 ``swiglu_limit`` clamp matches the reference (R2.3/R2.5).

    With a positive ``swiglu_limit`` the gate is clamped to ``<= limit`` and up
    to ``[-limit, +limit]`` before ``silu*mul``. The reference, the dense L0/L1
    path and (when built) the fused mega-kernel all apply the *same* clamp, so
    every runnable realization must agree within the fp16 tolerance. Inputs are
    scaled up so the clamp actually bites (many gate/up values exceed the
    limit).
    """
    device = torch.device("cuda")
    limit = 10.0
    quant, x, topk_ids, topk_weights, K, I = _build_case(
        seed=99,
        k_mult=2,
        i_mult=2,
        num_experts=8,
        topk=2,
        num_tokens=48,
        weight_regime="wide",
        routing_mode="uniform",
        device=device,
    )
    # Amplify activations so a meaningful fraction of gate/up pre-activations
    # exceed the limit (otherwise the clamp would be a no-op and the test would
    # not actually exercise it).
    x = (x.to(torch.float32) * 6.0).to(torch.float16)

    ref = sm70_moe_reference(
        x, topk_weights, topk_ids,
        quant.w13_weight, quant.w13_weight_scale,
        quant.w2_weight, quant.w2_weight_scale,
        8, group_size=quant.group_size, swiglu_limit=limit,
    )
    if not bool(torch.isfinite(ref).all()):
        pytest.skip("reference output is non-finite (fp16 overflow); out of scope")
    golden = sm70_moe_reference_golden(
        x, topk_weights, topk_ids,
        quant.w13_weight, quant.w13_weight_scale,
        quant.w2_weight, quant.w2_weight_scale,
        8, group_size=quant.group_size, swiglu_limit=limit,
    )

    # Sanity: the clamp is non-trivial here — the unclamped reference differs.
    ref_noclamp = sm70_moe_reference(
        x, topk_weights, topk_ids,
        quant.w13_weight, quant.w13_weight_scale,
        quant.w2_weight, quant.w2_weight_scale,
        8, group_size=quant.group_size, swiglu_limit=0.0,
    )
    assert not torch.allclose(ref, ref_noclamp, rtol=1e-2, atol=1e-2), (
        "swiglu_limit clamp had no effect; amplify the input so it bites"
    )

    experts = SM70FusedMoEExperts()
    for layout, level in _runnable_combos():
        out = experts.forward(
            x, topk_weights, topk_ids, quant=quant, layout=layout,
            m_block=32, i_block=64, fusion_level=level, swiglu_limit=limit,
        )
        check = check_fp16_close(out, ref, K=K, I=I, topk=2, golden=golden)
        assert check.passed, (
            f"swiglu_limit fused path disagrees with the reference: "
            f"{check.reason} | layout={layout}, level={level.value}"
        )
