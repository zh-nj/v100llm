# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property test for the SM70 fused MoE contiguous-layout m_block alignment.

Feature: deepgemm-megamoe-sm70-port

This module verifies design **Property 3** ("contiguous 布局每专家段对齐
m_block") of :meth:`SM70MoELayoutAdapter.to_contiguous`
(``vllm/model_executor/layers/fused_moe/sm70_moe_layout.py``). The worktree
``moe_permute`` op uses an *older* signature with **no** ``align_block_size``
kwarg, so (after task 2.1) ``to_contiguous`` performs the ``m_block`` alignment
in **Python**: it calls ``moe_permute`` to densely pack the routed tokens, then
rounds each per-expert segment up to a multiple of ``m_block`` and scatters the
packed rows into the aligned, padded segments of the M axis (R3.2). Each padded
row is tagged with its owning expert in ``m_indices`` (padding rows hold the
``-1`` sentinel) and ``inv_permuted_idx`` is remapped into the padded-row space.

The property has two halves, both checked here for a single ``to_contiguous``
call:

1. *Segment alignment* — every expert segment length, i.e. the adjacent
   difference of ``expert_first_token_offset``, is an integer multiple of
   ``m_block`` (and equals ``round_up(per_expert_slot_count, m_block)``, with
   empty experts contributing a zero-length segment).
2. *Token containment* — every valid routed token slot
   (``token * topk + j``) lands inside the segment of the expert that
   ``topk_ids`` assigned it to: its destination row ``inv_permuted_idx[slot]``
   falls in ``[offset[e], offset[e + 1])`` and ``m_indices`` tags that row with
   expert ``e``.

The generators cover degenerate routing distributions: an ``all-to-one``
collapse onto a single expert, and subset routing (``cap < num_experts``) which
leaves one or more experts with zero tokens.

``moe_permute`` is a CUDA op, so this test is GPU-only and skips when no CUDA
device is present. In this repository the ``_moe_C`` extension is built for
SM70 (V100); run it pinned to a V100 (e.g. ``CUDA_VISIBLE_DEVICES=<v100>``).

Validates: Requirements 3.2
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

if not torch.cuda.is_available():
    pytest.skip(
        "SM70 fused MoE layout property test requires a CUDA GPU "
        "(moe_permute is a CUDA op)",
        allow_module_level=True,
    )

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute_unpermute_supported,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_layout import (
    SM70MoELayoutAdapter,
)

# --- generation domains ----------------------------------------------------

# m_block granularities the SM70 fused kernel aligns expert segments to
# (design §SM70FusedConfig: m_block default 16/32).
_M_BLOCKS = [16, 32]

# Hidden dim K must keep each row 16B-aligned for the permute kernel
# (n_hidden * 2 bytes % 16 == 0  ->  K % 8 == 0). Kept small for speed; K does
# not affect the alignment property.
_HIDDEN = [16, 64]


def _round_up(x: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``x`` (model reference)."""
    return ((x + multiple - 1) // multiple) * multiple


@st.composite
def _routing_case(draw: st.DrawFn) -> dict:
    """Draw a routing scenario for ``to_contiguous``.

    Returns the dimensions plus a flat ``topk_ids`` list (row-major over
    ``(token, slot)``). ``num_tokens >= 1`` because ``moe_permute`` cannot
    launch with an empty token grid.

    Routing modes:
    - ``all_to_one``: every slot routes to a single expert -> all other experts
      get a zero-length (still ``m_block``-aligned) segment.
    - ``spread``: slots are drawn from ``[0, cap)`` with ``cap`` possibly
      smaller than ``num_experts``, so experts ``>= cap`` receive zero tokens.
    """
    num_experts = draw(st.integers(min_value=1, max_value=16))
    num_tokens = draw(st.integers(min_value=1, max_value=48))
    topk = draw(st.integers(min_value=1, max_value=min(num_experts, 8)))
    m_block = draw(st.sampled_from(_M_BLOCKS))
    k = draw(st.sampled_from(_HIDDEN))

    n_slots = num_tokens * topk
    mode = draw(st.sampled_from(["spread", "all_to_one"]))
    if mode == "all_to_one":
        target = draw(st.integers(min_value=0, max_value=num_experts - 1))
        flat_ids = [target] * n_slots
    else:
        # cap < num_experts intentionally leaves trailing experts empty,
        # exercising the degenerate "expert with 0 tokens" distribution.
        cap = draw(st.integers(min_value=1, max_value=num_experts))
        flat_ids = draw(
            st.lists(
                st.integers(min_value=0, max_value=cap - 1),
                min_size=n_slots,
                max_size=n_slots,
            )
        )

    return {
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "topk": topk,
        "m_block": m_block,
        "k": k,
        "flat_ids": flat_ids,
    }


# Feature: deepgemm-megamoe-sm70-port, Property 3: contiguous 布局每专家段对齐 m_block
@settings(max_examples=100, deadline=None)
@given(case=_routing_case())
def test_prop3_mblock_alignment(case: dict) -> None:
    """Property 3: every expert segment is m_block-aligned and contains exactly
    the tokens routed to it.

    For any ``topk_ids`` / ``num_experts`` / ``m_block``, the contiguous layout
    produced by ``SM70MoELayoutAdapter.to_contiguous``:
      * has each ``expert_first_token_offset`` segment length be a multiple of
        ``m_block`` (and equal to ``round_up(slot_count_e, m_block)``), and
      * places every valid routed token slot inside its expert's segment
        (verified via ``inv_permuted_idx`` + ``m_indices``).
    """
    if not moe_permute_unpermute_supported():
        pytest.skip("moe_permute/unpermute is not supported on this platform.")

    num_experts = case["num_experts"]
    num_tokens = case["num_tokens"]
    topk = case["topk"]
    m_block = case["m_block"]
    k = case["k"]
    flat_ids = case["flat_ids"]

    device = "cuda"
    x = torch.randn((num_tokens, k), device=device, dtype=torch.float16)
    topk_ids = torch.tensor(flat_ids, dtype=torch.int32, device=device).reshape(
        num_tokens, topk
    )

    adapter = SM70MoELayoutAdapter()
    layout = adapter.to_contiguous(x, topk_ids, num_experts, m_block)

    offsets = layout.expert_first_token_offset.tolist()
    m_indices = layout.m_indices.tolist()
    inv = layout.inv_permuted_idx.tolist()
    m_padded = layout.permuted_input.shape[0]

    # offsets is the [E + 1] prefix array of per-expert segment starts.
    assert len(offsets) == num_experts + 1, (
        f"expected {num_experts + 1} offsets, got {len(offsets)}"
    )
    assert offsets[0] == 0, f"first offset must be 0, got {offsets[0]}"
    # ``permuted_input`` is allocated with exactly the sum of the aligned
    # per-expert segments (task 2.1 sizes it to ``aligned_offsets[-1]``), so the
    # padded row count equals the last offset.
    assert offsets[-1] == m_padded, (
        f"sum of aligned segments {offsets[-1]} != padded rows {m_padded}"
    )
    assert len(inv) == num_tokens * topk, (
        f"inv_permuted_idx length {len(inv)} != num_tokens*topk "
        f"{num_tokens * topk}"
    )

    # Independent model: per-expert slot counts derived straight from topk_ids.
    expected_counts = [0] * num_experts
    for e in flat_ids:
        expected_counts[e] += 1

    # --- Half 1: segment alignment ----------------------------------------
    for e in range(num_experts):
        seg_len = offsets[e + 1] - offsets[e]
        assert seg_len >= 0, f"expert {e} segment length is negative: {seg_len}"
        assert seg_len % m_block == 0, (
            f"expert {e} segment length {seg_len} is not a multiple of "
            f"m_block={m_block} (offsets={offsets})"
        )
        # Stronger model-based check: the padded length matches exactly the
        # round_up of the number of slots routed to this expert.
        assert seg_len == _round_up(expected_counts[e], m_block), (
            f"expert {e} segment length {seg_len} != round_up("
            f"{expected_counts[e]}, {m_block}) "
            f"= {_round_up(expected_counts[e], m_block)} (offsets={offsets})"
        )

    # --- Half 2: token containment ----------------------------------------
    # Every source slot token*topk+j routed to expert e must map (via
    # inv_permuted_idx) into expert e's [offset[e], offset[e+1]) segment, and
    # the destination row's m_indices tag must be e.
    for slot, e in enumerate(flat_ids):
        dst = inv[slot]
        assert offsets[e] <= dst < offsets[e + 1], (
            f"slot {slot} routed to expert {e} landed at row {dst}, outside "
            f"segment [{offsets[e]}, {offsets[e + 1]}) (offsets={offsets})"
        )
        assert m_indices[dst] == e, (
            f"row {dst} (slot {slot}, expert {e}) has m_indices tag "
            f"{m_indices[dst]}, expected {e}"
        )
