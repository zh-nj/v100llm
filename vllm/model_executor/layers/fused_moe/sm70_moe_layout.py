# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (V100) fused-MoE routing layout adapter.

This module is part of the ``deepgemm-megamoe-sm70-port`` spec (Requirement 3:
"分组/掩码 token 布局与专家路由对接"). It converts the router output
(``topk_ids`` / ``topk_weights``) into the two token layouts that the SM70
fused MoE kernel consumes:

* **contiguous (grouped) layout** — used for *prefill* (large M). Tokens routed
  to the same expert are packed contiguously along the M axis and each expert
  segment is padded up to the GEMM ``m_block`` granularity. This reuses the
  existing ``moe_permute`` CUDA op so the layout stays semantically identical to
  the rest of the vLLM MoE stack (R3.4).
* **masked (batched) layout** — used for *decode* / CUDA graph capture (small
  M, CPU unaware of per-expert token counts). Tokens are scattered into a fixed
  ``[E, max_tokens, K]`` buffer and only the first ``expert_num_tokens[e]`` rows
  of each expert are valid. The shape is constant w.r.t. the actual token
  distribution which keeps it CUDA-graph stable (R2.7 / R3.3).

All compute is ``float16`` (V100 does not support bfloat16). This module
implements both the layout *generation* (``to_contiguous`` / ``to_masked``) and
the ``combine`` (unpermute + weighted reduction) path that restores the per-token
output order, reusing the existing ``moe_unpermute`` op.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.utils.math_utils import round_up

__all__ = [
    "ContiguousLayout",
    "MaskedLayout",
    "SM70MoELayoutAdapter",
]


@dataclass
class ContiguousLayout:
    """Grouped (contiguous) token layout for the SM70 fused MoE kernel.

    Produced by :meth:`SM70MoELayoutAdapter.to_contiguous`. Mirrors the output
    of the existing ``moe_permute`` op so it can feed grouped-GEMM style kernels
    where tokens for one expert occupy a contiguous, ``m_block``-aligned slice
    of the M axis.

    Fields (see design §Data Models):
    - ``permuted_input``: ``fp16 [M_padded, K]`` — input rows gathered per
      expert and padded so each expert segment is a multiple of ``m_block``.
    - ``expert_first_token_offset``: ``int64 [E + 1]`` — start offset of each
      expert segment (aligned to ``m_block``); ``[E]`` is the padded total.
    - ``inv_permuted_idx``: ``int32 [num_tokens * topk]`` — reverse index map
      used by ``moe_unpermute`` during ``combine``. Values index into the
      *padded* row space of ``permuted_input`` (remapped from the worktree
      ``moe_permute`` packed output, see :meth:`SM70MoELayoutAdapter.to_contiguous`).
    - ``m_indices``: ``int32 [M_padded]`` — expert id owning each permuted row;
      invalid/padding rows hold ``fill_invalid_expert``.
    """

    permuted_input: torch.Tensor
    expert_first_token_offset: torch.Tensor
    inv_permuted_idx: torch.Tensor
    m_indices: torch.Tensor


@dataclass
class MaskedLayout:
    """Masked (batched) token layout for the SM70 fused MoE kernel.

    Produced by :meth:`SM70MoELayoutAdapter.to_masked`. The buffer shape is
    fixed (``[E, max_tokens, K]``) regardless of the actual routing
    distribution, which keeps it stable across CUDA graph replays (R2.7).

    Fields (see design §Data Models):
    - ``batched_input``: ``fp16 [E, max_tokens, K]`` — per-expert fixed-length
      batch; only the first ``expert_num_tokens[e]`` rows are valid.
    - ``expert_num_tokens``: ``int32 [E]`` — number of valid tokens per expert
      (the mask). Values are clamped to ``max_tokens``.
    """

    batched_input: torch.Tensor
    expert_num_tokens: torch.Tensor


class SM70MoELayoutAdapter:
    """Convert router output into SM70 fused-MoE token layouts.

    The adapter is stateless; instances exist only to group the layout
    conversion methods together. ``fill_invalid_expert`` controls the sentinel
    expert id written into ``m_indices`` for padding rows of the contiguous
    layout (kept ``-1`` by default, matching the existing vLLM convention).
    """

    def __init__(self, fill_invalid_expert: int = -1) -> None:
        self.fill_invalid_expert = fill_invalid_expert

    def to_contiguous(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        m_block: int,
    ) -> ContiguousLayout:
        """Build the grouped/contiguous layout from the worktree ``moe_permute``.

        The worktree ``moe_permute`` uses an *older* signature: it has **no**
        ``align_block_size`` / ``m_indices`` / ``fill_invalid_expert`` kwargs and
        packs the routed tokens densely into a ``[num_tokens * topk, K]`` buffer
        (one row per (token, topk-slot)) with *unaligned* per-expert segments
        (R3.4). The m_block alignment that the SM70 fused kernel needs (R3.2) is
        therefore performed here in **Python**:

        1. Call ``moe_permute`` to get the densely-packed permutation plus the
           packed ``expert_first_token_offset`` and ``inv_permuted_idx``.
        2. Round each per-expert segment length up to a multiple of ``m_block``
           and build the padded ``expert_first_token_offset`` (R3.2).
        3. Scatter the packed rows into the padded ``permuted_input`` so every
           expert segment starts on its aligned offset; padding rows are
           zero-filled.
        4. Build ``m_indices`` tagging each padded row with its expert id and
           ``fill_invalid_expert`` (``-1`` by default) for padding rows.
        5. Remap ``inv_permuted_idx`` from packed-row space into the padded-row
           space so ``combine`` / ``moe_unpermute`` gathers the right rows.

        Parameters:
        - ``x``: ``fp16 [num_tokens, K]`` input hidden states.
        - ``topk_ids``: ``[num_tokens, topk]`` expert ids from the router.
        - ``num_experts``: total number of experts ``E``.
        - ``m_block``: GEMM M-block granularity to align each expert segment to.
        """
        if x.dim() != 2:
            raise ValueError(
                f"to_contiguous expects 2D hidden states, got shape {tuple(x.shape)}"
            )
        if m_block <= 0:
            raise ValueError(f"m_block must be positive, got {m_block}")

        device = x.device
        hidden_dim = x.size(-1)
        num_tokens = x.size(0)
        topk = topk_ids.size(1) if topk_ids.dim() == 2 else 0

        # Alignment precondition of the worktree ``moe_permute`` op: it asserts
        # ``(n_hidden * element_size) % 16 == 0`` (rows must be 16B-aligned). For
        # fp16 that means K % 8 == 0. Surface it here with an actionable message
        # rather than letting the lower-level op assert fire cryptically.
        elem_size = x.element_size()
        if (hidden_dim * elem_size) % 16 != 0:
            raise ValueError(
                "to_contiguous requires the hidden dim to be 16B-aligned for "
                f"moe_permute: (K={hidden_dim} * {elem_size} bytes) must be a "
                f"multiple of 16 (e.g. K % {16 // elem_size} == 0 for "
                f"{x.dtype})."
            )

        # Degenerate case: no routed rows (num_tokens == 0). The worktree
        # moe_permute launches a grid of size ``num_tokens * topk`` and cannot
        # launch with a zero-sized grid (CUDA "invalid configuration argument"),
        # mirroring the ``total_slots == 0`` guard in AWQSM70MoEMethod. Return
        # empty, well-typed tensors with an all-zero offset vector.
        if num_tokens == 0 or topk == 0:
            return ContiguousLayout(
                permuted_input=torch.zeros(
                    (0, hidden_dim), dtype=x.dtype, device=device
                ),
                expert_first_token_offset=torch.zeros(
                    (num_experts + 1,), dtype=torch.int64, device=device
                ),
                inv_permuted_idx=torch.empty(
                    (0,), dtype=torch.int32, device=device
                ),
                m_indices=torch.empty((0,), dtype=torch.int32, device=device),
            )

        # Step 1: dense (unaligned) permutation via the worktree moe_permute.
        # NOTE: worktree signature has no align_block_size/m_indices/
        # fill_invalid_expert. It returns the flattened inv_permuted_idx and
        # the packed (unaligned) expert_first_token_offset.
        (
            packed_input,
            _a1q_scale,
            packed_offsets,  # int64 [E + 1], packed (unaligned) segment offsets
            packed_inv_idx,  # int32 [num_tokens * topk], dest rows in packed space
            _permuted_idx,
        ) = moe_permute(
            hidden_states=x,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=num_experts,
            n_local_expert=num_experts,
            expert_map=None,
        )

        total_packed = packed_input.size(0)

        # Defensive: should not happen once num_tokens>0, but keep the layout
        # well-typed if the op ever returns an empty permutation.
        if total_packed == 0:
            return ContiguousLayout(
                permuted_input=torch.zeros(
                    (0, hidden_dim), dtype=x.dtype, device=device
                ),
                expert_first_token_offset=torch.zeros(
                    (num_experts + 1,), dtype=torch.int64, device=device
                ),
                inv_permuted_idx=packed_inv_idx,
                m_indices=torch.empty((0,), dtype=torch.int32, device=device),
            )

        # Step 2: per-expert counts -> m_block-aligned segment offsets (R3.2).
        packed_offsets = packed_offsets.to(torch.int64)
        counts = packed_offsets[1:] - packed_offsets[:-1]  # int64 [E]
        aligned_counts = round_up_tensor(counts, m_block)  # int64 [E]
        aligned_offsets = torch.zeros(
            num_experts + 1, dtype=torch.int64, device=device
        )
        torch.cumsum(aligned_counts, dim=0, out=aligned_offsets[1:])
        m_padded = int(aligned_offsets[-1].item())

        # Step 3: build packed-row -> padded-row map. Within each expert segment
        # rows keep their relative order, so each packed row r in expert e is
        # shifted by (aligned_offset[e] - packed_offset[e]).
        row_expert = torch.repeat_interleave(
            torch.arange(num_experts, device=device, dtype=torch.int64),
            counts,
        )  # int64 [total_packed]
        shift_per_expert = aligned_offsets[:-1] - packed_offsets[:-1]  # int64 [E]
        packed_to_padded = (
            torch.arange(total_packed, device=device, dtype=torch.int64)
            + shift_per_expert[row_expert]
        )  # int64 [total_packed]

        # Scatter the packed rows into the padded buffer; padding rows stay 0.
        permuted_input = torch.zeros(
            (m_padded, hidden_dim), dtype=x.dtype, device=device
        )
        permuted_input[packed_to_padded] = packed_input

        # Step 4: m_indices — expert id per padded row, sentinel for padding.
        m_indices = torch.full(
            (m_padded,),
            self.fill_invalid_expert,
            dtype=torch.int32,
            device=device,
        )
        m_indices[packed_to_padded] = row_expert.to(torch.int32)

        # Step 5: remap inv_permuted_idx (dest rows) into padded-row space.
        # packed_inv_idx values index packed rows; map them through the same
        # packed->padded permutation. Guard out-of-range sentinels (EP skips).
        inv_idx_long = packed_inv_idx.to(torch.int64)
        in_range = inv_idx_long < total_packed
        remapped = torch.where(
            in_range,
            packed_to_padded[inv_idx_long.clamp(max=total_packed - 1)],
            inv_idx_long,
        )
        inv_permuted_idx = remapped.to(torch.int32)

        return ContiguousLayout(
            permuted_input=permuted_input,
            expert_first_token_offset=aligned_offsets,
            inv_permuted_idx=inv_permuted_idx,
            m_indices=m_indices,
        )

    def to_masked(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        max_tokens: int,
    ) -> MaskedLayout:
        """Build the masked/batched layout with a CUDA-graph-stable shape.

        Follows the ``batched_moe_align_block_size`` idea: the per-expert token
        count (``expert_num_tokens``) acts as a mask while the surrounding
        buffer keeps a fixed ``[E, max_tokens, K]`` shape, constant w.r.t. the
        routing distribution (R2.7 / R3.3).

        The scatter and the mask are produced from the *same*
        ``torch.any(topk_ids == e)`` pass, mirroring the batched ``prepare()``
        convention in ``fused_batched_moe.py`` (``tokens_per_expert``). This
        guarantees ``expert_num_tokens[e]`` equals exactly the number of valid
        rows written into ``batched_input[e]`` so the kernel never reads past
        the valid region (R3.3 / R3.4). Tokens beyond ``max_tokens`` for a given
        expert are dropped and the (clamped) count reflects that.

        Parameters:
        - ``x``: ``fp16 [num_tokens, K]`` input hidden states.
        - ``topk_ids``: ``[num_tokens, topk]`` expert ids from the router.
        - ``num_experts``: total number of experts ``E``.
        - ``max_tokens``: fixed per-expert capacity of the batched buffer.
        """
        if x.dim() != 2:
            raise ValueError(
                f"to_masked expects 2D hidden states, got shape {tuple(x.shape)}"
            )
        if topk_ids.dim() != 2:
            raise ValueError(
                f"to_masked expects 2D topk_ids, got shape {tuple(topk_ids.shape)}"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        num_tokens, hidden_dim = x.shape

        batched_input = torch.zeros(
            (num_experts, max_tokens, hidden_dim),
            dtype=x.dtype,
            device=x.device,
        )
        expert_num_tokens = torch.zeros(
            (num_experts,), dtype=torch.int32, device=x.device
        )

        # Scatter rows into the per-expert fixed-length batch and record the
        # per-expert count in the same pass. Mirrors the batched prepare()
        # pattern in fused_batched_moe.py: a token routed to expert e (via any
        # of its topk slots) contributes one row to batch e. Counting from the
        # scatter keeps the mask exactly consistent with the populated rows.
        if num_tokens > 0:
            for expert_id in range(num_experts):
                token_mask = torch.any(topk_ids == expert_id, dim=1).flatten()
                rows = int(torch.count_nonzero(token_mask).item())
                if rows == 0:
                    continue
                rows = min(rows, max_tokens)
                batched_input[expert_id, :rows, :] = x[token_mask][:rows]
                expert_num_tokens[expert_id] = rows

        return MaskedLayout(
            batched_input=batched_input,
            expert_num_tokens=expert_num_tokens,
        )

    def combine(
        self,
        sorted_out: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Unpermute + weighted reduce the expert outputs back to token order.

        Inverse of :meth:`to_contiguous`: takes the per-expert grouped outputs
        produced by the fused kernel and scatters each row back to the token it
        came from, scaling by the router weight and summing the ``topk``
        contributions for each token. This reuses the existing ``moe_unpermute``
        CUDA op (weighting + reduction are fused inside the kernel) so the
        result stays semantically identical to the rest of the vLLM MoE stack
        and to ``moe_align_block_size`` / ``moe_permute_unpermute`` (R3.4).

        The mapping onto the ``moe_unpermute`` wrapper is:

        * ``sorted_out`` → ``permuted_hidden_states`` (the grouped expert
          outputs, one row per ``inv_permuted_idx`` slot).
        * ``expert_offsets`` → ``expert_first_token_offset`` (the ``int64
          [E + 1]`` segment offsets from :class:`ContiguousLayout`).

        Parameters:
        - ``sorted_out``: ``fp16 [M_padded, K]`` per-expert grouped outputs in
          the permuted order produced by :meth:`to_contiguous`.
        - ``topk_weights``: ``[num_tokens, topk]`` router weights. The worktree
          ``moe_unpermute`` kernel reads the weights through a ``float*`` (it
          calls ``get_ptr<float>(topk_weights)`` with no dtype dispatch/check),
          so they MUST be ``float32``. Weights are converted here defensively;
          passing fp16 weights would otherwise reinterpret the raw bytes as
          fp32 and silently corrupt the reduction.
        - ``inv_permuted_idx``: ``int32 [num_tokens * topk]`` reverse index map
          from :class:`ContiguousLayout` (remapped into the m_block-padded row
          space by :meth:`to_contiguous`).
        - ``expert_offsets``: ``int64 [E + 1]`` first-token offset of each
          expert segment (``ContiguousLayout.expert_first_token_offset``). Its
          last element is the padded row count ``M_padded``; ``moe_unpermute``
          uses it as ``num_valid`` so the m_block padding rows (whose indices
          are never referenced by ``inv_permuted_idx``) are never gathered —
          this keeps ``combine`` robust to the padded layout from
          :meth:`to_contiguous`.

        Returns the reduced, per-token output ``fp16 [num_tokens, K]``.
        """
        if sorted_out.dim() != 2:
            raise ValueError(
                "combine expects 2D sorted expert outputs, got shape "
                f"{tuple(sorted_out.shape)}"
            )
        if topk_weights.dim() != 2:
            raise ValueError(
                f"combine expects 2D topk_weights, got shape {tuple(topk_weights.shape)}"
            )

        num_tokens = topk_weights.size(0)
        hidden_dim = sorted_out.size(-1)

        out = torch.empty(
            (num_tokens, hidden_dim),
            dtype=sorted_out.dtype,
            device=sorted_out.device,
        )

        # Degenerate case: no tokens to combine. The underlying moe_unpermute
        # CUDA kernel cannot launch with a zero-sized grid (mirrors the
        # ``total_slots == 0`` guard in AWQSM70MoEMethod), so short-circuit and
        # return the empty output directly.
        if num_tokens == 0:
            return out

        # moe_unpermute fuses the topk weighting and the cross-topk reduction:
        # it scatters each permuted row back to its source token (via
        # inv_permuted_idx), multiplies by topk_weights, and sums the topk
        # contributions per token into `out`.
        #
        # The worktree moe_unpermute kernel reads the weights as float32
        # (get_ptr<float>(topk_weights)) without any dtype check, so coerce to
        # fp32 here. This is a no-op when the caller already passes fp32 and
        # prevents silently reinterpreting fp16 bytes as fp32.
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)

        moe_unpermute(
            out=out,
            permuted_hidden_states=sorted_out,
            topk_weights=topk_weights,
            inv_permuted_idx=inv_permuted_idx,
            expert_first_token_offset=expert_offsets,
        )
        return out


def round_up_tensor(values: torch.Tensor, multiple: int) -> torch.Tensor:
    """Round each element of an int64 tensor up to a multiple of ``multiple``.

    Vectorised equivalent of ``round_up`` used to align per-expert segment
    lengths to the kernel ``m_block`` granularity (R3.2) without a CPU sync.
    """
    return ((values + (multiple - 1)) // multiple) * multiple


def _padded_contiguous_rows(
    counts: list[int] | torch.Tensor, m_block: int
) -> int:
    """Helper: padded M dimension after per-expert m_block alignment.

    Mirrors the Python-side alignment done in
    :meth:`SM70MoELayoutAdapter.to_contiguous`: each expert's token count is
    rounded up to a multiple of ``m_block`` and summed. Useful for tests/sizing.
    """
    if isinstance(counts, torch.Tensor):
        counts = counts.tolist()
    return sum(round_up(int(c), m_block) for c in counts)
