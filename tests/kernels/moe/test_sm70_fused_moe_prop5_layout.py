# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property test for the SM70 fused MoE scenario-driven layout selection.

Feature: deepgemm-megamoe-sm70-port

This module verifies design **Property 5** ("场景驱动的布局选择") of
:meth:`SM70FusedMoEExperts.select_layout`
(``vllm/model_executor/layers/fused_moe/sm70_fused_moe_experts.py``).

Property 5 (design) has two facets:

1. **Scenario-driven selection.** For any execution context the layout decision
   follows the prefill/decode rule (R2.6 / R2.7):

       layout == "masked"      IFF  is_graph_capturing OR num_tokens <= threshold
       layout == "contiguous"  otherwise

   i.e. a *decode* context — an active CUDA-graph capture (shape must stay
   constant across replays, R2.7) **or** a small batch
   (``num_tokens <= decode_threshold``, the one-token-per-sequence decode step)
   — selects the shape-stable ``masked`` layout, while a *prefill* context
   (large M, not capturing) selects the grouped ``contiguous`` layout (R2.6).

2. **Layout selection does not change the result.** The design states both
   layouts feed the *identical* fp16 MoE math, so picking one over the other is
   purely a scheduling decision and never alters the numerical output.

What this test validates
------------------------
Property 5 has two facets, both covered here:

* **Facet 1 — scenario-driven selection (CPU).**
  ``select_layout(num_tokens, *, is_graph_capturing, decode_threshold)`` is a
  pure Python decision function: its only inputs are the token count and the
  capture/threshold context — it does **not** take the routing tensors
  (``topk_ids`` / ``topk_weights`` / ``x``) at all. ``test_prop5_layout_selection``
  asserts it follows the prefill/decode rule, is deterministic, is independent of
  routing content, and has no side effects. It always passes an explicit
  ``is_graph_capturing`` (so the runtime ``_detect_graph_capturing`` probe is
  never consulted) and runs CPU-only.

* **Facet 2 — layout selection does not change the result (executed, GPU).**
  Since task 5.3 wired ``SM70FusedMoEExperts._forward_masked``, the masked layout
  is now fully executable, so the equivalence can be *run* rather than proxied.
  ``test_prop5_layout_numerical_equivalence`` builds a supported dsv4f **MXFP4**
  MoE (packed E2M1 ``uint8`` weights + per-32 E8M0 ``uint8`` block scales,
  ``group_size=32``) via :meth:`SM70MXFP4QuantParams.from_mxfp4_weights`, runs
  ``forward(..., layout="contiguous")`` and ``forward(..., layout="masked")``
  on identical routing inputs, and asserts the two outputs agree within the
  fp16 tolerance (``check_fp16_close`` from ``sm70_moe_reference``). Both layouts
  decode the *same* MXFP4 pack to dense fp16 (the shared
  :func:`mxfp4_dequant_to_fp16` source) and run the identical per-operator FFN,
  so the scenario-driven layout *choice* never changes the numbers. It is gated
  behind CUDA availability because the contiguous ``combine`` reuses the
  ``moe_unpermute`` CUDA op (which also requires ``float32`` ``topk_weights``).
  Shapes are kept small and the example budget bounded so the executed sweep
  stays fast; the broader equivalence-vs-reference coverage lives in the
  Property 1 test (task 5.5).

Validates: Requirements 2.6, 2.7
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusionLevel,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_reference import (
    check_fp16_close,
)

# --- generation domains ----------------------------------------------------

# num_tokens spans the design's M range [0, 256]: 0 (degenerate empty batch),
# tiny decode-sized batches, and large prefill chunks straddle every plausible
# decode_threshold.
_MAX_NUM_TOKENS = 256
# decode_threshold spans 0 (everything is prefill unless capturing) up past the
# default of 16 (design §SM70FusedConfig m_block / kernel M_TILE granularity).
_MAX_THRESHOLD = 64
# How many distinct routing payloads to build per example to demonstrate that
# the layout decision is invariant to routing *content* (kept small for speed).
_ROUTING_VARIANTS = 3


def _expected_layout(num_tokens: int, capturing: bool, threshold: int) -> str:
    """Independently derive the expected layout (the rule, from first principles).

    Mirrors the design's prefill/decode rule directly rather than calling the
    module under test, so the property check is a genuine cross-validation
    rather than a tautology.
    """
    if capturing or num_tokens <= threshold:
        return "masked"
    return "contiguous"


@st.composite
def _routing_payload(draw: st.DrawFn, num_tokens: int) -> dict:
    """Draw an arbitrary routing payload for a fixed ``num_tokens``.

    The token count is held fixed (it is the only routing-derived input that may
    legitimately influence the layout); everything else — the expert ids, the
    routing weights (including zeros and fp16 extremes), the hidden width and the
    activation values — is free to vary. Used to show the layout decision does
    not depend on this content.
    """
    num_experts = draw(st.integers(min_value=1, max_value=16))
    topk = draw(st.integers(min_value=1, max_value=min(num_experts, 8)))
    k = draw(st.sampled_from([8, 16, 64]))

    n_slots = num_tokens * topk
    flat_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=num_experts - 1),
            min_size=n_slots,
            max_size=n_slots,
        )
    )
    topk_ids = torch.tensor(flat_ids, dtype=torch.int32).reshape(num_tokens, topk)
    # Routing weights cover the full fp16 range plus exact zeros.
    weight_choices = draw(
        st.lists(
            st.sampled_from([0.0, 1.0, -1.0, 1e-4, 65504.0, -65504.0, 0.5]),
            min_size=n_slots,
            max_size=n_slots,
        )
    )
    topk_weights = torch.tensor(weight_choices, dtype=torch.float16).reshape(
        num_tokens, topk
    )
    x = torch.randn((num_tokens, k), dtype=torch.float16)
    return {
        "num_experts": num_experts,
        "topk": topk,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "x": x,
    }


# Feature: deepgemm-megamoe-sm70-port, Property 5: 场景驱动的布局选择
@settings(max_examples=100, deadline=None)
@given(
    num_tokens=st.integers(min_value=0, max_value=_MAX_NUM_TOKENS),
    is_graph_capturing=st.booleans(),
    decode_threshold=st.integers(min_value=0, max_value=_MAX_THRESHOLD),
    data=st.data(),
)
def test_prop5_layout_selection(
    num_tokens: int,
    is_graph_capturing: bool,
    decode_threshold: int,
    data: st.DataObject,
) -> None:
    """Property 5: layout is "masked" for decode (capturing or small M) and
    "contiguous" for prefill, and the decision is a pure function of
    (num_tokens, capture, threshold) — independent of routing content.

    For any context (``num_tokens``, ``is_graph_capturing``, ``decode_threshold``):

    * (selection) ``select_layout`` returns ``"masked"`` iff capturing OR
      ``num_tokens <= decode_threshold``, else ``"contiguous"`` — matching the
      independently-derived rule, whether the threshold is passed per-call or
      carried on the instance.
    * (selection doesn't change the result) the returned layout is invariant
      across arbitrary routing payloads (``topk_ids`` / ``topk_weights`` / ``x``)
      with the same ``num_tokens``, and is deterministic / side-effect free.
    """
    expected = _expected_layout(num_tokens, is_graph_capturing, decode_threshold)

    # Build the orchestrator with the threshold on the instance, and snapshot
    # its mutable-looking state to assert select_layout has no side effects.
    experts = SM70FusedMoEExperts(decode_threshold=decode_threshold)
    fusion_level_before = experts.fusion_level
    threshold_before = experts.decode_threshold

    # --- Facet 1: scenario-driven selection -------------------------------
    # Threshold carried on the instance (decode_threshold defaults to None).
    layout_instance = experts.select_layout(
        num_tokens, is_graph_capturing=is_graph_capturing
    )
    assert layout_instance == expected, (
        f"instance-threshold layout {layout_instance!r} != expected "
        f"{expected!r} for num_tokens={num_tokens}, "
        f"capturing={is_graph_capturing}, threshold={decode_threshold}"
    )
    assert layout_instance in ("masked", "contiguous")

    # Threshold passed explicitly per call (on an orchestrator built with a
    # deliberately different default) must give the same decision — the per-call
    # override wins and the rule depends only on the effective threshold.
    other = SM70FusedMoEExperts(decode_threshold=(_MAX_THRESHOLD - decode_threshold))
    layout_percall = other.select_layout(
        num_tokens,
        is_graph_capturing=is_graph_capturing,
        decode_threshold=decode_threshold,
    )
    assert layout_percall == expected, (
        f"per-call-threshold layout {layout_percall!r} != expected "
        f"{expected!r} for num_tokens={num_tokens}, "
        f"capturing={is_graph_capturing}, threshold={decode_threshold}"
    )

    # --- Facet 2a: determinism (pure function) ----------------------------
    repeat = experts.select_layout(
        num_tokens, is_graph_capturing=is_graph_capturing
    )
    assert repeat == layout_instance, (
        f"select_layout is not deterministic: {repeat!r} != {layout_instance!r}"
    )

    # --- Facet 2b: independent of routing content -------------------------
    # The layout for a fixed (num_tokens, capture, threshold) must not depend on
    # the routing values, so *selecting* a layout cannot change what the MoE
    # computes on those values (runnable proxy for layout numerical equivalence;
    # an executed masked-vs-contiguous equivalence is deferred until the masked
    # kernel lands — see module docstring).
    for _ in range(_ROUTING_VARIANTS):
        payload = data.draw(_routing_payload(num_tokens))
        # num_tokens derived from the routing payload itself; everything else
        # (expert ids, weights, activations, hidden width) varies freely.
        assert payload["topk_ids"].shape[0] == num_tokens
        layout_for_payload = experts.select_layout(
            payload["topk_ids"].shape[0],
            is_graph_capturing=is_graph_capturing,
        )
        assert layout_for_payload == expected, (
            f"layout {layout_for_payload!r} changed with routing content "
            f"(expected {expected!r} for num_tokens={num_tokens}); the decision "
            f"must depend only on token count + capture context"
        )

    # --- Facet 2c: no side effects ----------------------------------------
    assert experts.fusion_level == fusion_level_before
    assert experts.decode_threshold == threshold_before


# --- focused boundary examples (complement the property above) -------------


def test_capture_forces_masked_regardless_of_m() -> None:
    """An active CUDA-graph capture forces ``masked`` even for a huge batch (R2.7)."""
    experts = SM70FusedMoEExperts(decode_threshold=16)
    # Far above any prefill threshold, yet capture pins the shape-stable layout.
    assert experts.select_layout(4096, is_graph_capturing=True) == "masked"


def test_threshold_boundary_is_inclusive() -> None:
    """``num_tokens == threshold`` is decode (masked); one more is prefill."""
    experts = SM70FusedMoEExperts(decode_threshold=16)
    assert experts.select_layout(16, is_graph_capturing=False) == "masked"
    assert experts.select_layout(17, is_graph_capturing=False) == "contiguous"


def test_zero_tokens_is_decode() -> None:
    """A degenerate empty batch (num_tokens == 0) is a decode context (masked)."""
    experts = SM70FusedMoEExperts(decode_threshold=0)
    # 0 <= 0 -> masked even with the smallest possible threshold.
    assert experts.select_layout(0, is_graph_capturing=False) == "masked"


# --- Facet 2: executed contiguous-vs-masked numerical equivalence ----------
# Since task 5.3 wired ``_forward_masked`` the masked layout is now executable,
# so we *run* both layouts and assert they agree (layout choice doesn't change
# the result). Gated on CUDA: the contiguous ``combine`` reuses the
# ``moe_unpermute`` CUDA op (and needs fp32 ``topk_weights``).

# dsv4f MXFP4 facts (design §Data Models ``SM70MXFP4QuantParams``):
#   w13_weight       uint8 [E, 2*I, K // 2]    (two E2M1 nibbles per byte)
#   w13_weight_scale uint8 [E, 2*I, K // 32]   (per-32 E8M0 block scale)
#   w2_weight        uint8 [E, K,   I // 2]
#   w2_weight_scale  uint8 [E, K,   I // 32]
_MXFP4_GROUP_SIZE = 32
# E8M0 block-scale exponents kept in a modest band so the de-quantized fp16
# weights stay small/finite (overflow is Property 4's concern, not Property 5's).
# 127 == 2**0; the +-2 spread gives scales in [2**-2, 2**2].
_E8M0_LO = 125
_E8M0_HI = 129


def _rand_packed_u8(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Random uint8 tensor whose two nibbles are arbitrary E2M1 codes (0..15).

    The exact bit pattern is irrelevant to the equivalence check: both layouts
    decode the *same* :class:`SM70MXFP4QuantParams` via the shared
    :func:`mxfp4_dequant_to_fp16`, so they see byte-for-byte identical dense
    fp16 weights — which is all the contiguous-vs-masked comparison needs.
    """
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)


def _build_mxfp4_quant(
    E: int, K: int, I: int, gs: int, device: torch.device
) -> SM70MXFP4QuantParams:
    """Build a supported dsv4f :class:`SM70MXFP4QuantParams` with random weights.

    Block scales are kept in a narrow E8M0 band (``2**-2 .. 2**2``) so the
    de-quantized fp16 weights stay in a modest, finite range. The packed E2M1
    nibbles are fully random (spanning the whole code range).
    """
    two_i = 2 * I
    return SM70MXFP4QuantParams.from_mxfp4_weights(
        w13_weight=_rand_packed_u8((E, two_i, K // 2), device),
        w13_weight_scale=torch.randint(
            _E8M0_LO, _E8M0_HI, (E, two_i, K // gs), dtype=torch.uint8, device=device
        ),
        w2_weight=_rand_packed_u8((E, K, I // 2), device),
        w2_weight_scale=torch.randint(
            _E8M0_LO, _E8M0_HI, (E, K, I // gs), dtype=torch.uint8, device=device
        ),
        group_size=gs,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="contiguous combine reuses the moe_unpermute CUDA op (needs a GPU)",
)
# Feature: deepgemm-megamoe-sm70-port, Property 5: 场景驱动的布局选择
@settings(max_examples=100, deadline=None)
@given(
    num_tokens=st.integers(min_value=1, max_value=64),
    E=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    data=st.data(),
)
def test_prop5_layout_numerical_equivalence(
    num_tokens: int,
    E: int,
    seed: int,
    data: st.DataObject,
) -> None:
    """Property 5 (facet 2): the contiguous and masked layouts produce
    numerically-equivalent output on identical routing inputs, so the
    scenario-driven layout *choice* never changes the result (R2.6/R2.7).

    Both layouts decode the identical dsv4f MXFP4 pack to dense fp16 (the shared
    :func:`mxfp4_dequant_to_fp16` source) and feed the identical fp16 MoE math
    (same SwiGLU, same per-token combine), so an executed ``layout="contiguous"``
    run and an executed ``layout="masked"`` run agree within the fp16 tolerance.
    """
    device = torch.device("cuda")
    torch.manual_seed(seed)

    gs = _MXFP4_GROUP_SIZE
    topk = data.draw(st.integers(min_value=1, max_value=min(E, 4)))
    # K, I are multiples of the MXFP4 group size (and hence of 8) per the kernel
    # contract (K % 8 == 0 / K % group_size == 0).
    K = gs * data.draw(st.integers(min_value=1, max_value=2))
    I = gs * data.draw(st.integers(min_value=1, max_value=2))

    quant = _build_mxfp4_quant(E, K, I, gs, device)

    x = (torch.randn((num_tokens, K), device=device) * 0.5).to(torch.float16)
    topk_ids = torch.randint(
        0, E, (num_tokens, topk), dtype=torch.int32, device=device
    )
    # moe_unpermute (contiguous combine) expects fp32 router weights; share the
    # exact same weights with the masked path so the comparison is apples-to-apples.
    topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)

    experts = SM70FusedMoEExperts(fusion_level=SM70FusionLevel.L1)

    out_contiguous = experts.forward(
        x, topk_weights, topk_ids, quant=quant, layout="contiguous"
    )
    out_masked = experts.forward(
        x, topk_weights, topk_ids, quant=quant, layout="masked"
    )

    assert out_contiguous.shape == out_masked.shape == (num_tokens, K)
    assert out_contiguous.dtype == out_masked.dtype == torch.float16

    # Layout choice must not change the numbers: compare masked against the
    # contiguous run within the design's fp16 tolerance.
    check = check_fp16_close(out_masked, out_contiguous, K=K, I=I, topk=topk)
    assert check.passed, (
        f"contiguous vs masked layout disagree (layout choice changed the "
        f"result): {check.reason}; E={E}, K={K}, I={I}, topk={topk}, "
        f"num_tokens={num_tokens}, gs={gs}"
    )
