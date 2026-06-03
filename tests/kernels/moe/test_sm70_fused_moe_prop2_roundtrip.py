# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property test for the SM70 fused MoE routing-layout round trip (Property 2).

Feature: deepgemm-megamoe-sm70-port

The :class:`SM70MoELayoutAdapter` (design §Components "SM70MoELayoutAdapter",
Requirement 3) converts the router output into the contiguous (grouped) token
layout by reusing the *worktree* ``moe_permute`` CUDA op and restores the
per-token order with ``combine`` (which reuses ``moe_unpermute``).

Important: the worktree ``moe_permute`` uses an **older signature** — it has no
``align_block_size`` / ``m_indices`` / ``fill_invalid_expert`` kwargs and packs
the routed tokens densely into a ``[num_tokens * topk, K]`` buffer with
*unaligned* per-expert segments (R3.4). ``SM70MoELayoutAdapter.to_contiguous``
performs the GEMM ``m_block`` alignment on top of that, in Python, by padding
each expert segment and remapping the reverse index into the padded row space
(R3.2). This test therefore validates Property 2 against the adapter's *actual*
behavior on this worktree, not against the newer upstream op.

This module verifies the single round-trip / model-equivalence property:

* **Round trip identity** — building the contiguous layout from ``x`` and
  ``topk_ids`` and then calling ``combine`` with *identity* router weights
  (``topk_weights`` all ones, fp32) reconstructs the topk-expanded sum of the
  original token rows, i.e. ``out[token] == topk * x[token]`` (each of a
  token's ``topk`` permuted rows is a copy of that token's input, and
  ``moe_unpermute`` sums them). This is the layout's permute/unpermute
  round-trip recovering the original token order and values.
* **Model equivalence (worktree op as the oracle)** — the adapter is a thin,
  m_block-padding wrapper over the worktree ``moe_permute`` / ``moe_unpermute``.
  So (a) gathering its padded ``permuted_input`` through its remapped
  ``inv_permuted_idx`` must reproduce exactly what a direct ``moe_permute`` call
  reconstructs through *its* (packed) reverse index — i.e. they agree on the
  valid/written rows, ignoring the m_block padding rows the adapter adds; (b)
  each adapter expert-segment length must equal the packed count rounded up to
  ``m_block``; and (c) ``combine`` must match a direct ``moe_unpermute`` call
  elementwise for an arbitrary grouped expert-output tensor and arbitrary
  (fp32) router weights.

The functions under test call the ``_moe_C`` CUDA extension (``moe_permute`` /
``moe_unpermute``), so the test requires a CUDA GPU with the compiled op. It is
guarded accordingly and skipped otherwise.

Validates: Requirements 3.1, 3.4
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Importing current_platform plus the moe extension forces the ``_moe_C`` op
# namespace (moe_permute / moe_unpermute) to register so the support probe and
# the kernels below are available.
from vllm.platforms import current_platform

with contextlib.suppress(ImportError):
    import vllm._moe_C  # noqa: F401

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_layout import (
    SM70MoELayoutAdapter,
)

# The adapter's to_contiguous/combine reuse the moe_permute/moe_unpermute CUDA
# ops; without a GPU + compiled extension there is nothing meaningful to test.
if not torch.cuda.is_available():
    pytest.skip(
        "SM70 layout round-trip property test requires a CUDA GPU",
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

_FILL_INVALID_EXPERT = -1

# Routing distributions to exercise: uniform spread, the all-to-one degenerate
# case (every token to a single expert in every slot), a sparse case that
# leaves some experts with zero tokens, and a "distinct" case mirroring real
# top-k routing (the topk experts of a token are all different).
_ROUTING_MODES = ["uniform", "all_to_one", "sparse", "distinct"]


def _make_topk_ids(
    mode: str,
    num_tokens: int,
    num_experts: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct a deterministic ``[num_tokens, topk]`` int32 routing table.

    ``seed`` makes each Hypothesis example reproducible. The four modes cover
    the degenerate routings called out by the spec (some expert 0 tokens,
    all-to-one) plus ordinary distinct-expert routing.
    """
    torch.manual_seed(seed)
    if mode == "all_to_one":
        # Every token routed to expert 0 in all of its topk slots.
        ids = torch.zeros((num_tokens, topk), device=device, dtype=torch.int32)
    elif mode == "sparse" and num_experts > 1:
        # Route only to even-indexed experts -> odd experts get zero tokens.
        half = (num_experts + 1) // 2
        ids = (
            (torch.randint(0, half, (num_tokens, topk), device=device) * 2)
            .clamp_(max=num_experts - 1)
            .to(torch.int32)
        )
    elif mode == "distinct" and topk <= num_experts:
        # Each token's topk experts are all distinct (real top-k routing).
        ids = (
            torch.rand((num_tokens, num_experts), device=device)
            .argsort(dim=1)[:, :topk]
            .to(torch.int32)
        )
    else:  # "uniform" (and fallbacks for tiny shapes)
        ids = torch.randint(
            0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32
        )
    return ids


# Feature: deepgemm-megamoe-sm70-port, Property 2: 路由布局 permute/unpermute 往返恒等
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.differing_executors],
)
@given(
    num_tokens=st.integers(min_value=1, max_value=64),
    k_mult=st.integers(min_value=1, max_value=32),  # K = k_mult * 8 (fp16 16B align)
    num_experts=st.integers(min_value=1, max_value=16),
    topk=st.integers(min_value=1, max_value=8),
    m_block=st.sampled_from([8, 16, 32, 64, 128]),
    routing_mode=st.sampled_from(_ROUTING_MODES),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_prop2_permute_roundtrip(
    num_tokens: int,
    k_mult: int,
    num_experts: int,
    topk: int,
    m_block: int,
    routing_mode: str,
    seed: int,
) -> None:
    """Property 2: contiguous permute/unpermute round trip is identity.

    For any fp16 ``x`` and any ``topk_ids`` (validated against the *actual*
    worktree ``SM70MoELayoutAdapter``, which m_block-pads on top of the older
    ``moe_permute`` op):

    1. ``SM70MoELayoutAdapter.to_contiguous`` agrees with a direct
       ``moe_permute`` call on the valid/written rows — gathering each layout's
       reverse index reconstructs the same token rows — and its per-expert
       segment lengths are the packed counts rounded up to ``m_block``.
    2. Feeding the permuted input back through ``combine`` with identity
       weights reconstructs ``topk * x`` in the original token order.
    3. ``combine`` matches a direct ``moe_unpermute`` call elementwise for an
       arbitrary grouped expert-output tensor and arbitrary (fp32) router
       weights.
    """
    topk = min(topk, num_experts)
    hidden = k_mult * 8  # K%8==0 keeps (K * 2 bytes) % 16 == 0 (permute kernel req)
    device = torch.device("cuda")

    x = torch.randn((num_tokens, hidden), device=device, dtype=torch.float16)
    topk_ids = _make_topk_ids(
        routing_mode, num_tokens, num_experts, topk, seed, device
    )

    adapter = SM70MoELayoutAdapter(fill_invalid_expert=_FILL_INVALID_EXPERT)
    layout = adapter.to_contiguous(x, topk_ids, num_experts, m_block)

    # --- direct worktree moe_permute (the oracle / model) --------------------
    # The worktree op uses the OLDER signature: no align_block_size /
    # m_indices / fill_invalid_expert. It returns the densely-packed (unaligned)
    # permutation, the packed expert offsets and the flattened reverse index.
    (
        op_packed_input,  # fp16 [num_tokens * topk, K], packed (unaligned)
        _a1q_scale,
        op_packed_offsets,  # int64 [E + 1], packed (unaligned) segment offsets
        op_packed_inv_idx,  # int32 [num_tokens * topk], dest rows in packed space
        _op_permuted_idx,
    ) = moe_permute(
        hidden_states=x,
        a1q_scale=None,
        topk_ids=topk_ids,
        n_expert=num_experts,
        n_local_expert=num_experts,
        expert_map=None,
    )

    # --- (1a) per-expert segment lengths == round_up(packed counts, m_block) --
    # The adapter's only structural change over the op is m_block padding, so
    # each adapter segment length must be the packed count rounded up.
    op_counts = op_packed_offsets[1:] - op_packed_offsets[:-1]
    adapter_counts = (
        layout.expert_first_token_offset[1:]
        - layout.expert_first_token_offset[:-1]
    )
    expected_aligned = ((op_counts + (m_block - 1)) // m_block) * m_block
    torch.testing.assert_close(adapter_counts, expected_aligned, atol=0, rtol=0)
    # All routed (token, slot) pairs are accounted for (no EP dropping here).
    assert int(op_counts.sum().item()) == num_tokens * topk

    # --- (1b) adapter agrees with the op on the valid/written rows -----------
    # Gathering each layout's reverse index over its own permuted buffer
    # reconstructs one row per (token, topk-slot). The adapter buffer is
    # m_block-padded, but inv_permuted_idx only ever references the written
    # rows, so this gather ignores the padding and must match the op exactly.
    adapter_gathered = layout.permuted_input[layout.inv_permuted_idx.long()].view(
        num_tokens, topk, hidden
    )
    op_gathered = op_packed_input[op_packed_inv_idx.long()].view(
        num_tokens, topk, hidden
    )
    torch.testing.assert_close(adapter_gathered, op_gathered, atol=0, rtol=0)
    # ...and both equal the original input broadcast across the topk slots
    # (the permute placed a copy of x[token] at each of its topk rows).
    torch.testing.assert_close(
        adapter_gathered,
        x[:, None, :].expand(num_tokens, topk, hidden),
        atol=0,
        rtol=0,
    )

    # --- (2) round-trip identity with identity (all-ones, fp32) weights ------
    identity_weights = torch.ones(
        (num_tokens, topk), dtype=torch.float32, device=device
    )
    recovered = adapter.combine(
        layout.permuted_input,
        identity_weights,
        layout.inv_permuted_idx,
        layout.expert_first_token_offset,
    )
    # Each of a token's topk permuted rows is a copy of x[token]; with identity
    # weights moe_unpermute sums them -> topk * x[token]. Compare in fp32 with
    # the design's fp16 elementwise tolerance (round trip is exact up to the
    # fp16 reduction of `topk` copies).
    expected = x.to(torch.float32) * float(topk)
    torch.testing.assert_close(
        recovered.to(torch.float32), expected, atol=1e-2, rtol=1.6e-2
    )

    # --- (3) combine == direct moe_unpermute op (the oracle) -----------------
    # Arbitrary grouped expert outputs + arbitrary fp32 router weights: combine
    # is a thin wrapper over moe_unpermute (it only coerces weights to fp32 and
    # forwards the adapter's own padded tensors), so it must agree bit-for-bit
    # with a direct call using the SAME (padded) layout tensors.
    torch.manual_seed(seed + 1)
    sorted_out = torch.randn_like(layout.permuted_input)
    weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)

    adapter_out = adapter.combine(
        sorted_out,
        weights,
        layout.inv_permuted_idx,
        layout.expert_first_token_offset,
    )
    direct_out = torch.empty(
        (num_tokens, hidden), dtype=sorted_out.dtype, device=device
    )
    moe_unpermute(
        out=direct_out,
        permuted_hidden_states=sorted_out,
        topk_weights=weights,
        inv_permuted_idx=layout.inv_permuted_idx,
        expert_first_token_offset=layout.expert_first_token_offset,
    )
    torch.testing.assert_close(adapter_out, direct_out, atol=0, rtol=0)
