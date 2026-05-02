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
):
    """One program per (row, k_pos).

    pid_row = batch_idx * next_n + next_idx
    pid_k   = k position index
    """
    pid_row = tl.program_id(0)
    k_pos = tl.program_id(1)

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

    # Load K vector (fp8 as uint8) and scale
    kv_base = (
        kv_cache_ptr
        + physical_block.to(tl.int64) * kv_cache_stride0
        + pos_in_block * kv_cache_stride1
    )
    k_uint8 = tl.load(kv_base + tl.arange(0, HEAD_DIM))
    # Scale is stored as float32 at byte offset HEAD_DIM
    k_scale_ptr_typed = (kv_base + HEAD_DIM).to(tl.pointer_type(tl.float32))
    k_scale = tl.load(k_scale_ptr_typed)

    # Decode K: fp8 e4m3fn → fp32, apply scale
    k_sign = ((k_uint8 >> 7) & 1).to(tl.int32)
    k_exp = ((k_uint8 >> 3) & 0xF).to(tl.int32)
    k_mant = (k_uint8 & 0x7).to(tl.int32)
    k_fp32_bits = (k_sign << 31) | ((k_exp + 120) << 23) | (k_mant << 20)
    k_is_zero = (k_exp == 0) & (k_mant == 0)
    k_fp32_bits = tl.where(k_is_zero, 0, k_fp32_bits)
    k_f32 = k_fp32_bits.to(tl.float32, bitcast=True) * k_scale  # [D]

    # Compute dot product for each head and accumulate weighted ReLU
    weights_base = weights_ptr + pid_row * weights_stride0
    logit_val = tl.zeros((), dtype=tl.float32)
    for h in range(NUM_HEADS):
        # Load weight for this head
        wh = tl.load(weights_base + h)

        # Load Q for this head
        q_base = (
            q_ptr
            + batch_idx * q_stride_b
            + next_idx * q_stride_n
            + h * q_stride_h
        )
        q_uint8 = tl.load(q_base + tl.arange(0, HEAD_DIM))

        # Decode Q: fp8 e4m3fn → fp32
        q_sign = ((q_uint8 >> 7) & 1).to(tl.int32)
        q_exp = ((q_uint8 >> 3) & 0xF).to(tl.int32)
        q_mant = (q_uint8 & 0x7).to(tl.int32)
        q_fp32_bits = (q_sign << 31) | ((q_exp + 120) << 23) | (q_mant << 20)
        q_is_zero = (q_exp == 0) & (q_mant == 0)
        q_fp32_bits = tl.where(q_is_zero, 0, q_fp32_bits)
        q_f32 = q_fp32_bits.to(tl.float32, bitcast=True)  # [D]

        # Dot product + ReLU + weighted accumulation
        score = tl.sum(q_f32 * k_f32, axis=0)
        score = tl.maximum(score, 0.0)
        logit_val += score * wh

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

    # Ensure context_lens is 2D [B, next_n]
    if context_lens.ndim == 1:
        context_lens_2d = context_lens.unsqueeze(-1).expand(-1, next_n).contiguous()
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

    grid = (num_rows, max_model_len)
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
    )
    return logits


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
):
    """One program per (m, n) pair."""
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

    # Load K[n] and decode fp8 e4m3fn → fp32
    k_base = k_ptr + pid_n * k_stride_n
    k_uint8 = tl.load(k_base + tl.arange(0, HEAD_DIM))
    k_scale = tl.load(k_scale_ptr + pid_n)

    k_sign = ((k_uint8 >> 7) & 1).to(tl.int32)
    k_exp = ((k_uint8 >> 3) & 0xF).to(tl.int32)
    k_mant = (k_uint8 & 0x7).to(tl.int32)
    k_fp32_bits = (k_sign << 31) | ((k_exp + 120) << 23) | (k_mant << 20)
    k_is_zero = (k_exp == 0) & (k_mant == 0)
    k_fp32_bits = tl.where(k_is_zero, 0, k_fp32_bits)
    k_f32 = k_fp32_bits.to(tl.float32, bitcast=True) * k_scale

    # Compute weighted ReLU dot across heads
    logit_val = tl.zeros((), dtype=tl.float32)
    for h in range(NUM_HEADS):
        # Load weight for this head
        wh = tl.load(weights_ptr + pid_m * weights_stride_m + h)

        q_base = q_ptr + pid_m * q_stride_m + h * q_stride_h
        q_uint8 = tl.load(q_base + tl.arange(0, HEAD_DIM))

        # Decode Q: fp8 e4m3fn → fp32
        q_sign = ((q_uint8 >> 7) & 1).to(tl.int32)
        q_exp = ((q_uint8 >> 3) & 0xF).to(tl.int32)
        q_mant = (q_uint8 & 0x7).to(tl.int32)
        q_fp32_bits = (q_sign << 31) | ((q_exp + 120) << 23) | (q_mant << 20)
        q_is_zero = (q_exp == 0) & (q_mant == 0)
        q_fp32_bits = tl.where(q_is_zero, 0, q_fp32_bits)
        q_f32 = q_fp32_bits.to(tl.float32, bitcast=True)

        score = tl.sum(q_f32 * k_f32, axis=0)
        score = tl.maximum(score, 0.0)
        logit_val += score * wh

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
    )
    return logits
