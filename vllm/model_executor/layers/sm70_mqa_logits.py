# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM70-optimized MQA logits kernels for the Sparse Attention Indexer.

Replaces the Python-loop torch fallback with Triton kernels that:
- Manually decode FP8 e4m3fn (no tl.float8e4nv)
- Use fp16 dot products with fp32 accumulation
- Fuse ReLU + weighted sum across heads

Two kernels:
  - sm70_fp8_mqa_logits: non-paged (prefill), q=[M,H,D] k=[N,D]
  - sm70_fp8_paged_mqa_logits: paged (decode), q=[B,next_n,H,D] kv_cache paged
"""

import torch

from vllm.triton_utils import tl, triton


def _sm70_mqa_block_d(head_dim: int) -> int:
    for block_d in (64, 32, 16, 8, 4, 2, 1):
        if head_dim >= block_d and head_dim % block_d == 0:
            return block_d
    raise ValueError(f"Unsupported SM70 MQA head_dim={head_dim}")


@triton.jit
def _decode_fp8_e4m3fn(uint8_val):
    """Decode FP8 e4m3fn uint8 vector to fp32.

    Manual bit extraction: sign(1) | exp(4) | mantissa(3).
    Exponent bias adjustment: FP8 bias=7, FP32 bias=127, delta=120.
    Optimized: builds FP32 bits with minimal intermediates by shifting
    the lower 7 bits (exp|mant) directly and OR-ing with the sign bit.
    """
    val32 = uint8_val.to(tl.int32)
    sign_bit = (val32 & 0x80) << 24
    low7 = val32 & 0x7F
    fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
    fp32_bits = tl.where(low7 == 0, 0, fp32_bits)
    return fp32_bits.to(tl.float32, bitcast=True)


@triton.jit
def _fp32_to_ordered_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    mask = tl.where(
        (bits & 0x80000000) != 0,
        0xFFFFFFFF,
        0x80000000,
    )
    return bits ^ mask


@triton.jit
def _ordered_key_to_fp32(key):
    bits = key ^ tl.where(
        (key & 0x80000000) == 0,
        0xFFFFFFFF,
        0x80000000,
    )
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _sm70_fp8_paged_mqa_logits_kernel(
    # Q: [B, next_n, H, D] as uint8 (fp8 e4m3fn)
    q_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    # KV cache: [num_blocks, block_size, D+4] as uint8 (flattened dim2)
    kv_cache_ptr,
    kv_cache_stride0,
    kv_cache_stride1,
    # weights: [B*next_n, H] float32
    weights_ptr,
    weights_stride0,
    # context_lens: [B, next_n] int32
    context_lens_ptr,
    context_lens_stride0,
    # block_tables: [B, max_blocks] int32
    block_tables_ptr,
    block_tables_stride0,
    # output logits: [B*next_n, max_model_len] float32
    logits_ptr,
    logits_stride0,
    # dims
    batch_size,
    next_n,
    max_model_len,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (row, k_pos).

    pid_row = batch_idx * next_n + next_idx
    pid_k   = k position index

    Optimization: chunks the HEAD_DIM dot product into BLOCK_D-sized pieces,
    processing all heads per chunk to keep K loaded and reduce register
    pressure from full-vector Q reloads.
    """
    pid = tl.program_id(0)
    pid_row = pid // max_model_len
    k_pos = pid - pid_row * max_model_len

    batch_idx = pid_row // next_n
    next_idx = pid_row % next_n

    if batch_idx >= batch_size:
        return

    # Get context length for this query
    context_len = tl.load(
        context_lens_ptr + batch_idx * context_lens_stride0 + next_idx
    )

    if k_pos >= context_len:
        return

    # Find physical block and position within block
    cache_block_idx = k_pos // BLOCK_SIZE
    pos_in_block = k_pos % BLOCK_SIZE

    physical_block = tl.load(
        block_tables_ptr + batch_idx * block_tables_stride0 + cache_block_idx
    )

    # Base address for K vector in paged cache
    kv_base = (
        kv_cache_ptr
        + physical_block.to(tl.int64) * kv_cache_stride0
        + pos_in_block * kv_cache_stride1
    )

    # Preload weights for all heads before the main loop to avoid
    # scattered loads in the epilogue and enable register reuse.
    weights_base = weights_ptr + pid_row * weights_stride0
    w_0 = tl.load(weights_base + 0) if NUM_HEADS > 0 else 0.0
    w_1 = tl.load(weights_base + 1) if NUM_HEADS > 1 else 0.0
    w_2 = tl.load(weights_base + 2) if NUM_HEADS > 2 else 0.0
    w_3 = tl.load(weights_base + 3) if NUM_HEADS > 3 else 0.0
    w_4 = tl.load(weights_base + 4) if NUM_HEADS > 4 else 0.0
    w_5 = tl.load(weights_base + 5) if NUM_HEADS > 5 else 0.0
    w_6 = tl.load(weights_base + 6) if NUM_HEADS > 6 else 0.0
    w_7 = tl.load(weights_base + 7) if NUM_HEADS > 7 else 0.0

    # Q base address for this batch/next position
    q_base_bn = (
        q_ptr
        + batch_idx * q_stride_b
        + next_idx * q_stride_n
    )

    # Chunk the HEAD_DIM dot product into BLOCK_D-sized pieces.
    # For each chunk, load K once and process all heads, reducing
    # peak register pressure from full-vector Q reloads.
    NUM_D_CHUNKS: tl.constexpr = HEAD_DIM // BLOCK_D

    # Per-head score accumulators across D chunks.
    # After all chunks, apply ReLU + weight + sum.
    score_0 = tl.zeros((), dtype=tl.float32)
    score_1 = tl.zeros((), dtype=tl.float32)
    score_2 = tl.zeros((), dtype=tl.float32)
    score_3 = tl.zeros((), dtype=tl.float32)
    score_4 = tl.zeros((), dtype=tl.float32)
    score_5 = tl.zeros((), dtype=tl.float32)
    score_6 = tl.zeros((), dtype=tl.float32)
    score_7 = tl.zeros((), dtype=tl.float32)

    # Scale is stored as float32 at byte offset HEAD_DIM
    k_scale_ptr_typed = (kv_base + HEAD_DIM).to(tl.pointer_type(tl.float32))
    k_scale = tl.load(k_scale_ptr_typed)

    for d_chunk in tl.static_range(NUM_D_CHUNKS):
        d_off = d_chunk * BLOCK_D
        d_range = d_off + tl.arange(0, BLOCK_D)

        # Load and decode K chunk: fp8 e4m3fn → fp32, apply scale
        k_chunk_uint8 = tl.load(kv_base + d_range)
        k_chunk_f32 = _decode_fp8_e4m3fn(k_chunk_uint8) * k_scale

        # Process each head with this K chunk
        for h in tl.static_range(NUM_HEADS):
            # Load and decode Q chunk for head h
            q_chunk_uint8 = tl.load(q_base_bn + h * q_stride_h + d_range)
            q_chunk_f32 = _decode_fp8_e4m3fn(q_chunk_uint8)

            # Partial dot product for this chunk
            partial = tl.sum(q_chunk_f32 * k_chunk_f32, axis=0)

            # Accumulate into per-head score
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

    # Apply ReLU + weighted accumulation using preloaded weights
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

    # Store logit
    out_addr = logits_ptr + pid_row * logits_stride0 + k_pos
    tl.store(out_addr, logit_val)


def sm70_fp8_paged_mqa_logits(
    q: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 1, D+scale_bytes]
    weights: torch.Tensor,  # [B*next_n, H] float32
    context_lens: torch.Tensor,  # [B] or [B, next_n] int32
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    max_model_len: int,
) -> torch.Tensor:
    """SM70 Triton kernel for paged MQA logits (decode path)."""
    batch_size, next_n, num_heads, head_dim = q.shape
    block_size = kv_cache.shape[1]

    # Ensure context_lens is 2D [B, next_n]. For native spec decode, a 1D
    # length means the final generated token's length; earlier speculative
    # tokens use progressively shorter effective context lengths.
    if context_lens.ndim == 1:
        next_n_arange = torch.arange(
            next_n, device=context_lens.device, dtype=torch.int32
        )
        context_lens_2d = (
            context_lens.unsqueeze(-1) - next_n + 1 + next_n_arange
        ).contiguous()
    elif context_lens.ndim == 2:
        context_lens_2d = context_lens.contiguous()
    else:
        raise ValueError(f"context_lens must be 1D or 2D, got {context_lens.ndim}D")

    # Output
    num_rows = batch_size * next_n
    logits = torch.full(
        (num_rows, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    # Pass Q as uint8 view
    q_u8 = q.view(torch.uint8)

    # KV cache: [num_blocks, block_size, 1, D+4] → flatten dim2
    kv_flat = kv_cache.reshape(kv_cache.shape[0], block_size, -1)

    # Use 64 for the real DeepSeek dimensions, but keep small synthetic
    # contract tests valid by choosing a divisor of head_dim.
    BLOCK_D = _sm70_mqa_block_d(head_dim)

    grid = _sm70_paged_mqa_logits_grid(num_rows, max_model_len)
    _sm70_fp8_paged_mqa_logits_kernel[grid](
        q_u8,
        q_u8.stride(0),
        q_u8.stride(1),
        q_u8.stride(2),
        kv_flat,
        kv_flat.stride(0),
        kv_flat.stride(1),
        weights,
        weights.stride(0),
        context_lens_2d,
        context_lens_2d.stride(0),
        block_tables,
        block_tables.stride(0),
        logits,
        logits.stride(0),
        batch_size,
        next_n,
        max_model_len,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
    )
    return logits


@triton.jit
def _sm70_fp8_mqa_block_candidate_update_kernel(
    # Q: [M, H, D] as uint8
    q_ptr,
    q_stride_m,
    q_stride_h,
    # K: [N, D] as uint8
    k_ptr,
    k_stride_n,
    # K scale: [N] float32
    k_scale_ptr,
    # weights: [M, H] float32
    weights_ptr,
    weights_stride_m,
    # row ranges: [M] int32
    row_starts_ptr,
    row_ends_ptr,
    # previous/final candidates: [M, TOPK]
    best_scores_ptr,
    best_indices_ptr,
    next_scores_ptr,
    next_indices_ptr,
    block_start,
    M,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    CANDIDATE_PAD: tl.constexpr,
):
    """Fused fp8 MQA block logits + candidate merge.

    This is the first single-launch building block for streaming prefill top-k:
    one program computes all candidate scores for one query row and one KV block,
    merges them with the existing per-row best candidates, and writes the next
    candidate state.  It avoids materializing [rows, block_k] logits.
    """
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    cand = tl.arange(0, CANDIDATE_PAD)
    is_best = cand < TOPK
    block_offset_raw = cand - TOPK
    in_block = (cand >= TOPK) & (block_offset_raw < BLOCK_N)
    block_offset = tl.where(in_block, block_offset_raw, 0)
    k_pos = block_start + block_offset

    row_start = tl.load(row_starts_ptr + pid_m)
    row_end = tl.load(row_ends_ptr + pid_m)
    valid_block = in_block & (k_pos >= row_start) & (k_pos < row_end)

    q_base_m = q_ptr + pid_m * q_stride_m
    weights_base = weights_ptr + pid_m * weights_stride_m
    block_logits = tl.zeros((CANDIDATE_PAD,), dtype=tl.float32)

    NUM_D_CHUNKS: tl.constexpr = HEAD_DIM // BLOCK_D
    for h in tl.static_range(NUM_HEADS):
        h_score = tl.zeros((CANDIDATE_PAD,), dtype=tl.float32)
        for d_chunk in tl.static_range(NUM_D_CHUNKS):
            d_off = d_chunk * BLOCK_D
            d_range = d_off + tl.arange(0, BLOCK_D)

            k_u8 = tl.load(
                k_ptr + k_pos[:, None] * k_stride_n + d_range[None, :],
                mask=valid_block[:, None],
                other=0,
            )
            k_scale = tl.load(k_scale_ptr + k_pos, mask=valid_block, other=0.0)
            k_f32 = _decode_fp8_e4m3fn(k_u8) * k_scale[:, None]

            q_u8 = tl.load(q_base_m + h * q_stride_h + d_range)
            q_f32 = _decode_fp8_e4m3fn(q_u8)
            h_score += tl.sum(k_f32 * q_f32[None, :], axis=1)

        weight = tl.load(weights_base + h)
        block_logits += tl.maximum(h_score, 0.0) * weight

    block_logits = tl.where(valid_block, block_logits, float("-inf"))
    best_scores = tl.load(
        best_scores_ptr + pid_m * TOPK + cand,
        mask=is_best,
        other=float("-inf"),
    )
    candidate_scores = tl.where(is_best, best_scores, block_logits)

    score_key = _fp32_to_ordered_key(candidate_scores)
    # Tie-break toward lower candidate positions for deterministic tests. The
    # low 16 bits are enough for the supported candidate widths.
    pos_key = (CANDIDATE_PAD - cand).to(tl.uint64)
    packed = (score_key.to(tl.uint64) << 16) | pos_key
    sorted_packed = tl.sort(packed, descending=True)

    rank = cand
    selected_pos = CANDIDATE_PAD - (sorted_packed & 0xFFFF).to(tl.int32)
    selected_score_key = (sorted_packed >> 16).to(tl.uint32)
    selected_score = _ordered_key_to_fp32(selected_score_key)
    selected_is_best = selected_pos < TOPK
    selected_block_offset = selected_pos - TOPK
    best_index = tl.load(
        best_indices_ptr + pid_m * TOPK + selected_pos,
        mask=selected_is_best,
        other=-1,
    )
    block_index = block_start + selected_block_offset - row_start
    selected_index = tl.where(selected_is_best, best_index, block_index)

    tl.store(
        next_scores_ptr + pid_m * TOPK + rank,
        selected_score,
        mask=rank < TOPK,
    )
    tl.store(
        next_indices_ptr + pid_m * TOPK + rank,
        selected_index,
        mask=rank < TOPK,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def sm70_fp8_mqa_block_candidate_update(
    *,
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    best_scores: torch.Tensor,
    best_indices: torch.Tensor,
    block_start: int,
    block_end: int,
    topk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-launch fp8 MQA block logits + candidate merge for SM70."""
    k_fp8, k_scale = kv
    if not (
        q.is_cuda
        and k_fp8.is_cuda
        and k_scale.is_cuda
        and weights.is_cuda
        and row_starts.is_cuda
        and row_ends.is_cuda
        and best_scores.is_cuda
        and best_indices.is_cuda
    ):
        raise ValueError("sm70 fp8 block candidate update expects CUDA tensors")
    if q.dtype not in (torch.float8_e4m3fn, getattr(torch, "float8_e4m3fnuz", None)):
        raise ValueError("q must be fp8")
    if k_fp8.dtype != q.dtype:
        raise ValueError("k cache values must use the same fp8 dtype as q")
    if best_scores.dtype is not torch.float32 or best_indices.dtype is not torch.int32:
        raise ValueError("best candidates must be fp32 scores and int32 indices")
    if topk_tokens != best_scores.shape[1] or topk_tokens != best_indices.shape[1]:
        raise ValueError("topk_tokens must match best candidate width")
    if block_end <= block_start:
        raise ValueError("block_end must be greater than block_start")

    rows, num_heads, head_dim = q.shape
    block_n = block_end - block_start
    candidate_pad = _next_power_of_2(topk_tokens + block_n)
    block_d = min(_sm70_mqa_block_d(head_dim), 16)
    next_scores = torch.empty_like(best_scores)
    next_indices = torch.empty_like(best_indices)
    q_u8 = q.view(torch.uint8)
    k_u8 = k_fp8.view(torch.uint8)
    _sm70_fp8_mqa_block_candidate_update_kernel[(rows,)](
        q_u8,
        q_u8.stride(0),
        q_u8.stride(1),
        k_u8,
        k_u8.stride(0),
        k_scale.reshape(-1),
        weights,
        weights.stride(0),
        row_starts,
        row_ends,
        best_scores,
        best_indices,
        next_scores,
        next_indices,
        block_start,
        rows,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        TOPK=topk_tokens,
        CANDIDATE_PAD=candidate_pad,
        num_warps=8,
    )
    return next_scores, next_indices


def _sm70_paged_mqa_logits_grid(num_rows: int, max_model_len: int) -> tuple[int]:
    return (num_rows * max_model_len,)


@triton.jit
def _sm70_fp8_mqa_logits_kernel(
    # Q: [M, H, D] as uint8
    q_ptr,
    q_stride_m,
    q_stride_h,
    # K: [N, D] as uint8
    k_ptr,
    k_stride_n,
    # K scale: [N] float32
    k_scale_ptr,
    # weights: [M, H] float32
    weights_ptr,
    weights_stride_m,
    # cu_seqlen_ks: [M] int32
    cu_seqlen_ks_ptr,
    # cu_seqlen_ke: [M] int32
    cu_seqlen_ke_ptr,
    # output: [M, N] float32
    out_ptr,
    out_stride_m,
    # dims
    M,
    N,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (m, n) pair.

    Optimization: chunks the HEAD_DIM dot product into BLOCK_D-sized pieces,
    processing all heads per chunk to reuse K and reduce register pressure.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= M or pid_n >= N:
        return

    # Check bounds
    ks = tl.load(cu_seqlen_ks_ptr + pid_m)
    ke = tl.load(cu_seqlen_ke_ptr + pid_m)
    if pid_n < ks or pid_n >= ke:
        tl.store(out_ptr + pid_m * out_stride_m + pid_n, float("-inf"))
        return

    # K base and scale
    k_base = k_ptr + pid_n * k_stride_n
    k_scale = tl.load(k_scale_ptr + pid_n)

    # Q base for this query token
    q_base_m = q_ptr + pid_m * q_stride_m

    # Preload weights for all heads before the main loop
    weights_base = weights_ptr + pid_m * weights_stride_m
    w_0 = tl.load(weights_base + 0) if NUM_HEADS > 0 else 0.0
    w_1 = tl.load(weights_base + 1) if NUM_HEADS > 1 else 0.0
    w_2 = tl.load(weights_base + 2) if NUM_HEADS > 2 else 0.0
    w_3 = tl.load(weights_base + 3) if NUM_HEADS > 3 else 0.0
    w_4 = tl.load(weights_base + 4) if NUM_HEADS > 4 else 0.0
    w_5 = tl.load(weights_base + 5) if NUM_HEADS > 5 else 0.0
    w_6 = tl.load(weights_base + 6) if NUM_HEADS > 6 else 0.0
    w_7 = tl.load(weights_base + 7) if NUM_HEADS > 7 else 0.0

    NUM_D_CHUNKS: tl.constexpr = HEAD_DIM // BLOCK_D

    # Per-head score accumulators (partial dot products across D chunks)
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

        # Load and decode K chunk
        k_chunk_uint8 = tl.load(k_base + d_range)
        k_chunk_f32 = _decode_fp8_e4m3fn(k_chunk_uint8) * k_scale

        # Process each head with this K chunk
        for h in tl.static_range(NUM_HEADS):
            q_chunk_uint8 = tl.load(q_base_m + h * q_stride_h + d_range)
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

    # Apply ReLU + weighted accumulation using preloaded weights
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

    tl.store(out_ptr + pid_m * out_stride_m + pid_n, logit_val)


def sm70_fp8_mqa_logits(
    q: torch.Tensor,  # [M, H, D] float8_e4m3fn
    kv: tuple[torch.Tensor, torch.Tensor],  # (k [N, D] fp8, k_scale [N] fp32)
    weights: torch.Tensor,  # [M, H] float32
    cu_seqlen_ks: torch.Tensor,  # [M] int32
    cu_seqlen_ke: torch.Tensor,  # [M] int32
) -> torch.Tensor:
    """SM70 Triton kernel for non-paged MQA logits (prefill path)."""
    k_fp8, k_scale = kv
    M, num_heads, head_dim = q.shape
    N = k_fp8.shape[0]

    logits = torch.full(
        (M, N), float("-inf"), device=q.device, dtype=torch.float32
    )

    q_u8 = q.view(torch.uint8)
    k_u8 = k_fp8.view(torch.uint8)

    # Use 64 for the real DeepSeek dimensions, but keep small synthetic
    # contract tests valid by choosing a divisor of head_dim.
    BLOCK_D = _sm70_mqa_block_d(head_dim)

    grid = (M, N)
    _sm70_fp8_mqa_logits_kernel[grid](
        q_u8,
        q_u8.stride(0),
        q_u8.stride(1),
        k_u8,
        k_u8.stride(0),
        k_scale,
        weights,
        weights.stride(0),
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        logits.stride(0),
        M,
        N,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        BLOCK_D=BLOCK_D,
    )
    return logits


def sm70_fp8_mqa_logits_gemm(
    q: torch.Tensor,  # [M, H, D] float8_e4m3fn
    kv: tuple[torch.Tensor, torch.Tensor],  # (k [N, D] fp8, k_scale [N] fp32)
    weights: torch.Tensor,  # [M, H] float32
    cu_seqlen_ks: torch.Tensor,  # [M] int32
    cu_seqlen_ke: torch.Tensor,  # [M] int32
) -> torch.Tensor:
    """Headwise GEMM MQA logits for SM70 prefill.

    The eager reference builds a full fp32 score tensor [H, M, N].  That is
    fast for small prompts but becomes the memory cliff for long-context
    prefill.  This path keeps the final [M, N] fp32 logits only, computes one
    head at a time through cuBLAS half GEMM, and accumulates the weighted
    ReLU scores into the final logits.
    """
    k_fp8, k_scale = kv
    M, num_heads, _ = q.shape
    N = k_fp8.shape[0]

    compute_dtype = torch.float16
    q_heads = q.to(compute_dtype).permute(1, 0, 2).contiguous()
    k_half = k_fp8.to(compute_dtype)
    k_half = k_half * k_scale.reshape(-1, 1).to(compute_dtype)
    k_t = k_half.transpose(0, 1)
    weights_half = weights.to(compute_dtype)

    logits = torch.zeros((M, N), device=q.device, dtype=torch.float32)
    for head_idx in range(num_heads):
        scores = torch.mm(q_heads[head_idx], k_t)
        scores.relu_()
        scores.mul_(weights_half[:, head_idx].unsqueeze(1))
        logits.add_(scores)

    positions = torch.arange(0, N, device=q.device)
    mask = (positions[None, :] >= cu_seqlen_ks[:, None]) & (
        positions[None, :] < cu_seqlen_ke[:, None]
    )
    return logits.masked_fill_(~mask, float("-inf"))
