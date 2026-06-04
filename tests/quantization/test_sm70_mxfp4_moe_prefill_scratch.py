# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SM70 MXFP4 MoE prefill scratch pool.

Feature: deepgemm-megamoe-sm70-port (prefill buffer reuse)

These tests pin down the grow-only, cross-layer shared scratch pool that
replaced the per-layer ``torch.empty`` churn in the SM70 MXFP4 MoE prefill
path. The churn (large short-lived ``[slots, dim]`` allocations on every MoE
layer of every prefill step) drove CUDA caching-allocator fragmentation; the
pool grows to the largest size seen and is then reused.

Covered invariants:

* **Reuse** — two calls with the same (or smaller) shapes return views backed
  by the *same* underlying storage (no new allocation), which is the whole
  point of the pool.
* **Grow-on-demand** — a larger request grows the backing buffers; smaller
  follow-up requests reuse the grown buffers (no shrink, no churn). This is the
  "big request then a smaller one" sequence that previously fragmented.
* **Shape / slicing correctness** — returned tensors have exactly the requested
  shapes and ``token_expert_indices`` reshapes cleanly to ``[num_tokens, top_k]``.
* **Per-dims keying** — different expert dims get independent pools.

CPU-only: the pool just wraps ``torch.empty`` on the given device.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization import sm70_mxfp4_moe
from vllm.model_executor.layers.quantization.sm70_mxfp4_moe import (
    _get_sm70_moe_prefill_scratch,
)


def _make_layer(
    *,
    hidden: int = 512,
    w13_n_dim: int = 512,
    intermediate: int = 256,
    num_experts: int = 8,
    top_k: int = 6,
) -> SimpleNamespace:
    """A minimal layer shell carrying just the dims the scratch pool reads."""
    return SimpleNamespace(
        sm70_hidden_logical_size=hidden,
        sm70_w13_n_dim=w13_n_dim,
        sm70_intermediate_size=intermediate,
        sm70_num_experts=num_experts,
        _buf_top_k=top_k,
    )


def setup_function(_func) -> None:
    # Isolate each test from the module-global pool cache.
    sm70_mxfp4_moe._SM70_MOE_PREFILL_SCRATCH.clear()


def teardown_function(_func) -> None:
    sm70_mxfp4_moe._SM70_MOE_PREFILL_SCRATCH.clear()


def test_reuse_same_storage_across_calls() -> None:
    """Two calls at the same size reuse the same backing storage (no realloc)."""
    layer = _make_layer()
    device = torch.device("cpu")
    scratch = _get_sm70_moe_prefill_scratch(layer, device)

    top_k = layer._buf_top_k
    num_tokens = 64
    total_slots = num_tokens * top_k

    out1 = torch.empty(num_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16)
    b1 = scratch.get(total_slots, num_tokens, out1)
    ptr_permuted = b1["permuted_input"].data_ptr()
    ptr_sorted = b1["sorted_output"].data_ptr()
    ptr_gate_up = b1["gate_up"].data_ptr()
    ptr_inter = b1["intermediate"].data_ptr()

    out2 = torch.empty(num_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16)
    b2 = scratch.get(total_slots, num_tokens, out2)

    # Same backing storage -> the big buffers were reused, not reallocated.
    assert b2["permuted_input"].data_ptr() == ptr_permuted
    assert b2["sorted_output"].data_ptr() == ptr_sorted
    assert b2["gate_up"].data_ptr() == ptr_gate_up
    assert b2["intermediate"].data_ptr() == ptr_inter


def test_big_then_small_reuses_grown_buffers() -> None:
    """A smaller request after a bigger one reuses the grown buffers (no churn).

    This is exactly the "25k then 9k" sequence that previously fragmented: the
    big call grows the pool, the small call must slice into the SAME storage.
    """
    layer = _make_layer()
    device = torch.device("cpu")
    scratch = _get_sm70_moe_prefill_scratch(layer, device)
    top_k = layer._buf_top_k

    big_tokens = 256
    big_slots = big_tokens * top_k
    out_big = torch.empty(
        big_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16
    )
    b_big = scratch.get(big_slots, big_tokens, out_big)
    big_ptr = b_big["permuted_input"].data_ptr()
    big_capacity = scratch.capacity_slots
    assert big_capacity == big_slots

    small_tokens = 96
    small_slots = small_tokens * top_k
    out_small = torch.empty(
        small_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16
    )
    b_small = scratch.get(small_slots, small_tokens, out_small)

    # No shrink, no realloc: capacity stays at the high-water mark and the
    # smaller view points into the same storage as the big allocation.
    assert scratch.capacity_slots == big_capacity
    assert b_small["permuted_input"].data_ptr() == big_ptr
    assert b_small["permuted_input"].shape == (small_slots, layer.sm70_hidden_logical_size)


def test_grow_allocates_new_storage() -> None:
    """A larger request than seen so far grows (reallocates) the buffers."""
    layer = _make_layer()
    device = torch.device("cpu")
    scratch = _get_sm70_moe_prefill_scratch(layer, device)
    top_k = layer._buf_top_k

    small_tokens = 32
    out_small = torch.empty(
        small_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16
    )
    scratch.get(small_tokens * top_k, small_tokens, out_small)
    small_cap = scratch.capacity_slots

    big_tokens = 512
    out_big = torch.empty(
        big_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16
    )
    scratch.get(big_tokens * top_k, big_tokens, out_big)

    assert scratch.capacity_slots == big_tokens * top_k
    assert scratch.capacity_slots > small_cap


def test_returned_shapes_and_reshape() -> None:
    """Returned buffers have exactly the requested shapes and reshape cleanly."""
    layer = _make_layer()
    device = torch.device("cpu")
    scratch = _get_sm70_moe_prefill_scratch(layer, device)
    top_k = layer._buf_top_k

    num_tokens = 40
    total_slots = num_tokens * top_k
    out = torch.empty(num_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16)
    b = scratch.get(total_slots, num_tokens, out)

    assert b["output"] is out
    assert b["permuted_input"].shape == (total_slots, layer.sm70_hidden_logical_size)
    assert b["sorted_output"].shape == (total_slots, layer.sm70_hidden_logical_size)
    assert b["gate_up"].shape == (total_slots, layer.sm70_w13_n_dim)
    assert b["intermediate"].shape == (total_slots, layer.sm70_intermediate_size)
    assert b["inv_permuted_idx"].shape == (num_tokens, top_k)
    assert b["topk_ids_i32"].shape == (num_tokens, top_k)
    # token_expert_indices must reshape cleanly to [num_tokens, top_k].
    assert b["token_expert_indices"].shape == (num_tokens, top_k)
    assert b["expert_offsets"].shape == (layer.sm70_num_experts + 1,)
    assert b["expert_offsets64"].shape == (layer.sm70_num_experts + 1,)
    assert b["permuted_idx"].shape == (total_slots,)
    assert b["m_indices"].shape == (total_slots,)


def test_distinct_dims_get_distinct_pools() -> None:
    """Layers with different expert dims get independent pools (keyed by dims)."""
    layer_a = _make_layer(hidden=512, intermediate=256)
    layer_b = _make_layer(hidden=1024, intermediate=512)
    device = torch.device("cpu")

    scratch_a = _get_sm70_moe_prefill_scratch(layer_a, device)
    scratch_b = _get_sm70_moe_prefill_scratch(layer_b, device)
    assert scratch_a is not scratch_b

    # Same dims -> same shared pool (the common DeepSeek-V4 case: all layers
    # share one pool).
    layer_a2 = _make_layer(hidden=512, intermediate=256)
    scratch_a2 = _get_sm70_moe_prefill_scratch(layer_a2, device)
    assert scratch_a2 is scratch_a


def test_token_expert_indices_values_are_arange() -> None:
    """token_expert_indices is a clean 0..total_slots-1 arange reshaped."""
    layer = _make_layer()
    device = torch.device("cpu")
    scratch = _get_sm70_moe_prefill_scratch(layer, device)
    top_k = layer._buf_top_k

    num_tokens = 16
    total_slots = num_tokens * top_k
    out = torch.empty(num_tokens, layer.sm70_hidden_logical_size, dtype=torch.float16)
    b = scratch.get(total_slots, num_tokens, out)

    flat = b["token_expert_indices"].reshape(-1)
    assert torch.equal(flat, torch.arange(total_slots, dtype=torch.int32))
