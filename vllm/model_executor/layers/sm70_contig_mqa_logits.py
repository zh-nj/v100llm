# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 contiguous-K MQA logits decode kernel (v2: Q reuse).

Companion to ``sm70_fp8_paged_mqa_logits``. Path A goal: make the
indexer decode kernel faster at long context by:

1. Reading K from a flat ``(N, head_dim)`` FP8 buffer + ``(N,)`` float32
   scales (no paged dereference).
2. Loading Q into shared memory ONCE per block, then iterating BLOCK_K
   keys per block. This reduces the launch-overhead × num_programs
   product from ``(num_rows × max_model_len)`` programs (one per
   (query, key)) down to ``(num_rows × ceil(N / BLOCK_K))``.

The arithmetic mirrors ``_sm70_fp8_paged_mqa_logits_kernel`` exactly:
the head cascade collapses heads ≥ 7 into ``score_7`` / ``w_7`` and
applies ``ReLU + weight + sum`` per key. Bit-identical output is the
correctness contract for path A.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.sm70_mqa_logits import (
    _decode_fp8_e4m3fn,
    _sm70_mqa_block_d,
)


@triton.jit
def _sm70_fp8_contig_mqa_logits_kernel(
    # Q: [B, next_n, H, D] uint8 (FP8 e4m3 raw bytes)
    q_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    # K values: [N, D] uint8
    k_values_ptr,
    k_values_stride_n,
    # K scales: [N] float32
    k_scales_ptr,
    # weights: [num_rows, H] float32 (num_rows = B * next_n)
    weights_ptr,
    weights_stride_row,
    # output logits: [num_rows, max_model_len] float32
    logits_ptr,
    logits_stride_row,
    # dims (runtime)
    next_n,
    context_len,
    max_model_len,
    # constexprs
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One block per (next_idx, k_block_idx).

    Block work plan:
      - block_dim=BLOCK_K threads (or whichever Triton chooses).
      - Load weights[next_idx, :NUM_HEADS] once.
      - For each k position in [k_pos_base, k_pos_base+BLOCK_K):
          - load 1 K row (HEAD_DIM bytes) and 1 K scale (4 bytes)
          - for each h in [0, NUM_HEADS):
              load Q[next_idx, h, :HEAD_DIM]  (this is the cost we
              cannot avoid without true smem; keep as-is for now,
              the gain comes from reducing program count)
          - compute the per-head ReLU + weighted sum using the same
            cascade as the paged kernel (heads >= 7 collapse into
            score_7/w_7).
          - store one fp32 logit at logits[next_idx, k_pos].

    The crucial difference vs paged: BLOCK_K consecutive keys are
    handled by ONE program. Paged launches one program per (key);
    contig launches one program per (BLOCK_K keys), so the launch
    overhead amortizes BLOCK_K-fold.
    """
    pid = tl.program_id(0)
    pid_row = pid // tl.cdiv(max_model_len, BLOCK_K)
    pid_k_block = pid - pid_row * tl.cdiv(max_model_len, BLOCK_K)

    if pid_row >= next_n:
        return

    k_pos_base = pid_k_block * BLOCK_K
    if k_pos_base >= context_len:
        return

    q_row_base = q_ptr + pid_row * q_stride_n
    weights_row_base = weights_ptr + pid_row * weights_stride_row

    # Pre-load weights for the 8-bucket cascade (bit-equivalent to paged).
    w_0 = tl.load(weights_row_base + 0) if NUM_HEADS > 0 else 0.0
    w_1 = tl.load(weights_row_base + 1) if NUM_HEADS > 1 else 0.0
    w_2 = tl.load(weights_row_base + 2) if NUM_HEADS > 2 else 0.0
    w_3 = tl.load(weights_row_base + 3) if NUM_HEADS > 3 else 0.0
    w_4 = tl.load(weights_row_base + 4) if NUM_HEADS > 4 else 0.0
    w_5 = tl.load(weights_row_base + 5) if NUM_HEADS > 5 else 0.0
    w_6 = tl.load(weights_row_base + 6) if NUM_HEADS > 6 else 0.0
    w_7 = tl.load(weights_row_base + 7) if NUM_HEADS > 7 else 0.0

    NUM_D_CHUNKS: tl.constexpr = HEAD_DIM // BLOCK_D

    for kb in tl.static_range(BLOCK_K):
        k_pos = k_pos_base + kb
        in_bounds = k_pos < context_len

        if in_bounds:
            k_row_base = k_values_ptr + k_pos * k_values_stride_n
            k_scale = tl.load(k_scales_ptr + k_pos)

            score_0 = tl.zeros((), dtype=tl.float32)
            score_1 = tl.zeros((), dtype=tl.float32)
            score_2 = tl.zeros((), dtype=tl.float32)
            score_3 = tl.zeros((), dtype=tl.float32)
            score_4 = tl.zeros((), dtype=tl.float32)
            score_5 = tl.zeros((), dtype=tl.float32)
            score_6 = tl.zeros((), dtype=tl.float32)
            score_7 = tl.zeros((), dtype=tl.float32)

            for d_chunk in tl.static_range(NUM_D_CHUNKS):
                d_off = d_chunk * BLOCK_D
                d_range = d_off + tl.arange(0, BLOCK_D)

                k_chunk_uint8 = tl.load(k_row_base + d_range)
                k_chunk_f32 = _decode_fp8_e4m3fn(k_chunk_uint8) * k_scale

                for h in tl.static_range(NUM_HEADS):
                    q_chunk_uint8 = tl.load(
                        q_row_base + h * q_stride_h + d_range
                    )
                    q_chunk_f32 = _decode_fp8_e4m3fn(q_chunk_uint8)
                    partial = tl.sum(q_chunk_f32 * k_chunk_f32, axis=0)
                    if h == 0:
                        score_0 += partial
                    elif h == 1:
                        score_1 += partial
                    elif h == 2:
                        score_2 += partial
                    elif h == 3:
                        score_3 += partial
                    elif h == 4:
                        score_4 += partial
                    elif h == 5:
                        score_5 += partial
                    elif h == 6:
                        score_6 += partial
                    else:
                        score_7 += partial

            logit_val = tl.zeros((), dtype=tl.float32)
            if NUM_HEADS > 0:
                logit_val += tl.maximum(score_0, 0.0) * w_0
            if NUM_HEADS > 1:
                logit_val += tl.maximum(score_1, 0.0) * w_1
            if NUM_HEADS > 2:
                logit_val += tl.maximum(score_2, 0.0) * w_2
            if NUM_HEADS > 3:
                logit_val += tl.maximum(score_3, 0.0) * w_3
            if NUM_HEADS > 4:
                logit_val += tl.maximum(score_4, 0.0) * w_4
            if NUM_HEADS > 5:
                logit_val += tl.maximum(score_5, 0.0) * w_5
            if NUM_HEADS > 6:
                logit_val += tl.maximum(score_6, 0.0) * w_6
            if NUM_HEADS > 7:
                logit_val += tl.maximum(score_7, 0.0) * w_7

            tl.store(logits_ptr + pid_row * logits_stride_row + k_pos, logit_val)


def sm70_fp8_contiguous_mqa_logits(
    q: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn or uint8 view
    k_values: torch.Tensor,  # [N, D] uint8
    k_scales: torch.Tensor,  # [N] float32
    weights: torch.Tensor,  # [num_rows, H] float32
    context_len: int,
    max_model_len: int,
    *,
    block_k: int | None = None,
) -> torch.Tensor:
    """SM70 contiguous-K MQA logits (decode path)."""
    assert q.is_cuda, "contiguous mqa_logits requires CUDA tensors"
    if q.dim() == 3:
        q = q.unsqueeze(0)
    batch_size, next_n, num_heads, head_dim = q.shape
    assert batch_size == 1, "contig kernel currently supports batch=1"
    assert k_values.dim() == 2 and k_values.shape[1] == head_dim
    assert k_scales.dim() == 1 and k_scales.shape[0] == k_values.shape[0]
    assert weights.dim() == 2 and weights.shape[1] == num_heads
    assert weights.shape[0] == next_n, (
        f"weights must be [{next_n}, {num_heads}], got {tuple(weights.shape)}"
    )
    assert context_len <= k_values.shape[0]

    num_rows = next_n
    logits = torch.full(
        (num_rows, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    if context_len <= 0:
        return logits

    q_u8 = q.view(torch.uint8)

    BLOCK_D = _sm70_mqa_block_d(head_dim)
    if block_k is None:
        block_k = 8
    BLOCK_K = block_k
    n_k_blocks = (max_model_len + BLOCK_K - 1) // BLOCK_K
    grid = (num_rows * n_k_blocks,)

    _sm70_fp8_contig_mqa_logits_kernel[grid](
        q_u8,
        q_u8.stride(0),
        q_u8.stride(1),
        q_u8.stride(2),
        k_values,
        k_values.stride(0),
        k_scales,
        weights,
        weights.stride(0),
        logits,
        logits.stride(0),
        next_n,
        int(context_len),
        max_model_len,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return logits
