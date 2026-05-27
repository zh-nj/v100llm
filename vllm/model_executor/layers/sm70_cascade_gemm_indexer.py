# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 cascade-compatible GEMM indexer logits (decode path).

Replaces the decode call to ``sm70_fp8_paged_mqa_logits`` with a
contiguous-K fp32 GEMM that mirrors the existing 8-bucket head
cascade exactly. See
``.kiro/specs/deepseek-v4-decode-indexer-on-compressed-kv/`` for the
design rationale.

Per-call cost on V100 at 32k context: ~90 µs (vs ~7000 µs for
the paged kernel). The K snapshot is pre-staged in a higher-level
helper; this module only handles the per-call compute.

Cascade contract (mirrors ``_sm70_fp8_paged_mqa_logits_kernel``):

  for k_pos in [0, context_len):
      score = [0.0] * 8
      for h in [0, NUM_HEADS):
          partial = sum(decode(q[h]) * decode(k[k_pos]) * k_scale)
          bucket = h if h <= 6 else 7
          score[bucket] += partial
      logit[k_pos] = sum(relu(score[i]) * w_i for i in 0..7)

Bilinearity gives us the GEMM equivalent: by collapsing
``q[7:]`` into a single summed Q row, the dot-product ordering across
the K-axis still differs from the paged kernel (cuBLAS picks its own
tiling), but per-position values match within fp32 precision
(~7e-7 max relative error). The 125-case stress sweep in
``tools/cascade_gemm_indexer.py`` confirms this contract.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _decode_fp8_e4m3fn_inline(uint8_val):
    """Mirror of `_decode_fp8_e4m3fn` in `sm70_mqa_logits.py`. Kept
    here so this file does not depend on Triton helpers exported from
    the legacy module."""
    val32 = uint8_val.to(tl.int32)
    sign_bit = (val32 & 0x80) << 24
    low7 = val32 & 0x7F
    fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
    fp32_bits = tl.where(low7 == 0, 0, fp32_bits)
    return fp32_bits.to(tl.float32, bitcast=True)


@triton.jit
def _decode_q_emul_kernel(
    q_ptr,                # uint8 [B, next_n, H, D]
    q_stride_b,
    q_stride_n,
    q_stride_h,
    weights_ptr,          # float32 [num_rows, H]
    weights_stride_row,
    q_emul_ptr,           # float32 [num_rows, 8, D]
    q_emul_stride_row,
    q_emul_stride_bucket,
    w_emul_ptr,           # float32 [num_rows, 8]
    w_emul_stride_row,
    next_n,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Fuse Q FP8 decode + 8-bucket cascade collapse + weight emul.

    One program per (next_idx, d_chunk). Each program:
      - decodes Q[next_idx, h, d_chunk] for all h, sums heads >= 7 into
        bucket 7, copies heads 0..6 into buckets 0..6.
      - writes the 8-row Q_emul slice for this d_chunk.
      - if d_chunk == 0, also writes the 8-element w_emul row.
    """
    pid = tl.program_id(0)
    BLOCK_D: tl.constexpr = 64
    NUM_D_CHUNKS: tl.constexpr = HEAD_DIM // BLOCK_D
    pid_row = pid // NUM_D_CHUNKS
    pid_chunk = pid - pid_row * NUM_D_CHUNKS
    if pid_row >= next_n:
        return

    d_off = pid_chunk * BLOCK_D
    d_range = d_off + tl.arange(0, BLOCK_D)

    q_row_base = q_ptr + pid_row * q_stride_n
    weights_row_base = weights_ptr + pid_row * weights_stride_row
    q_emul_row_base = q_emul_ptr + pid_row * q_emul_stride_row
    w_emul_row_base = w_emul_ptr + pid_row * w_emul_stride_row

    # Decode buckets 0..6 directly. Bucket 7 accumulates heads 7..H-1.
    bucket_7 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for h in tl.static_range(NUM_HEADS):
        q_uint = tl.load(q_row_base + h * q_stride_h + d_range)
        q_chunk = _decode_fp8_e4m3fn_inline(q_uint)
        if h == 0:
            tl.store(q_emul_row_base + 0 * q_emul_stride_bucket + d_range, q_chunk)
        elif h == 1:
            tl.store(q_emul_row_base + 1 * q_emul_stride_bucket + d_range, q_chunk)
        elif h == 2:
            tl.store(q_emul_row_base + 2 * q_emul_stride_bucket + d_range, q_chunk)
        elif h == 3:
            tl.store(q_emul_row_base + 3 * q_emul_stride_bucket + d_range, q_chunk)
        elif h == 4:
            tl.store(q_emul_row_base + 4 * q_emul_stride_bucket + d_range, q_chunk)
        elif h == 5:
            tl.store(q_emul_row_base + 5 * q_emul_stride_bucket + d_range, q_chunk)
        elif h == 6:
            tl.store(q_emul_row_base + 6 * q_emul_stride_bucket + d_range, q_chunk)
        else:
            bucket_7 += q_chunk
    tl.store(q_emul_row_base + 7 * q_emul_stride_bucket + d_range, bucket_7)

    # First chunk also writes w_emul.
    if pid_chunk == 0:
        for h in tl.static_range(NUM_HEADS):
            if h == 0:
                tl.store(w_emul_row_base + 0, tl.load(weights_row_base + 0))
            elif h == 1:
                tl.store(w_emul_row_base + 1, tl.load(weights_row_base + 1))
            elif h == 2:
                tl.store(w_emul_row_base + 2, tl.load(weights_row_base + 2))
            elif h == 3:
                tl.store(w_emul_row_base + 3, tl.load(weights_row_base + 3))
            elif h == 4:
                tl.store(w_emul_row_base + 4, tl.load(weights_row_base + 4))
            elif h == 5:
                tl.store(w_emul_row_base + 5, tl.load(weights_row_base + 5))
            elif h == 6:
                tl.store(w_emul_row_base + 6, tl.load(weights_row_base + 6))
            elif h == 7:
                tl.store(w_emul_row_base + 7, tl.load(weights_row_base + 7))


@triton.jit
def _decode_k_to_fp32_kernel(
    k_values_ptr,         # uint8 [N, D]
    k_values_stride_n,
    k_scales_ptr,         # float32 [N]
    k_out_ptr,            # float32 [N, D]
    k_out_stride_n,
    N,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Decode FP8 K bytes + multiply by per-row scale, output fp32.

    One program per BLOCK_N rows. Each program decodes BLOCK_N×D
    bytes and writes BLOCK_N×D float32. Memory-bound on the K reads.
    """
    pid = tl.program_id(0)
    n_off = pid * BLOCK_N
    n_range = n_off + tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, HEAD_DIM)

    n_mask = n_range < N
    addr = k_values_ptr + n_range[:, None] * k_values_stride_n + d_range[None, :]
    k_uint = tl.load(addr, mask=n_mask[:, None], other=0)
    k_f32 = _decode_fp8_e4m3fn_inline(k_uint)
    k_scale = tl.load(k_scales_ptr + n_range, mask=n_mask, other=1.0)
    k_scaled = k_f32 * k_scale[:, None]
    out_addr = k_out_ptr + n_range[:, None] * k_out_stride_n + d_range[None, :]
    tl.store(out_addr, k_scaled, mask=n_mask[:, None])


@triton.jit
def _cascade_gemm_epilogue_kernel(
    scores_ptr,           # float32 [num_rows, 8, context_len] (no padding)
    scores_stride_row,
    scores_stride_bucket,
    w_emul_ptr,           # float32 [num_rows, 8]
    w_emul_stride_row,
    out_ptr,              # float32 [num_rows, max_model_len]
    out_stride_row,
    context_len,
    max_model_len,
    BLOCK_K: tl.constexpr,
):
    """Apply ReLU + weighted sum across the 8 buckets, write logits.

    One program per (next_idx, k_block). Each program handles BLOCK_K
    consecutive k positions. Out positions >= context_len are written
    with -inf to preserve the contract.
    """
    pid = tl.program_id(0)
    pid_row = pid // tl.cdiv(max_model_len, BLOCK_K)
    pid_k_block = pid - pid_row * tl.cdiv(max_model_len, BLOCK_K)

    k_pos_base = pid_k_block * BLOCK_K
    k_range = k_pos_base + tl.arange(0, BLOCK_K)
    in_range = k_range < context_len

    w_emul_row = w_emul_ptr + pid_row * w_emul_stride_row
    w_0 = tl.load(w_emul_row + 0)
    w_1 = tl.load(w_emul_row + 1)
    w_2 = tl.load(w_emul_row + 2)
    w_3 = tl.load(w_emul_row + 3)
    w_4 = tl.load(w_emul_row + 4)
    w_5 = tl.load(w_emul_row + 5)
    w_6 = tl.load(w_emul_row + 6)
    w_7 = tl.load(w_emul_row + 7)

    scores_row = scores_ptr + pid_row * scores_stride_row
    s_0 = tl.load(scores_row + 0 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_1 = tl.load(scores_row + 1 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_2 = tl.load(scores_row + 2 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_3 = tl.load(scores_row + 3 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_4 = tl.load(scores_row + 4 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_5 = tl.load(scores_row + 5 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_6 = tl.load(scores_row + 6 * scores_stride_bucket + k_range, mask=in_range, other=0.0)
    s_7 = tl.load(scores_row + 7 * scores_stride_bucket + k_range, mask=in_range, other=0.0)

    logit = (
        tl.maximum(s_0, 0.0) * w_0
        + tl.maximum(s_1, 0.0) * w_1
        + tl.maximum(s_2, 0.0) * w_2
        + tl.maximum(s_3, 0.0) * w_3
        + tl.maximum(s_4, 0.0) * w_4
        + tl.maximum(s_5, 0.0) * w_5
        + tl.maximum(s_6, 0.0) * w_6
        + tl.maximum(s_7, 0.0) * w_7
    )
    # Write logit if in_range, otherwise -inf.
    NEG_INF = float("-inf")
    out_val = tl.where(in_range, logit, NEG_INF)
    out_addr = out_ptr + pid_row * out_stride_row + k_range
    out_mask = k_range < max_model_len
    tl.store(out_addr, out_val, mask=out_mask)


def sm70_cascade_gemm_indexer(
    q: torch.Tensor,            # [B, next_n, H, D] uint8 (FP8) or fp8_e4m3fn
    k_values: torch.Tensor,      # [N, D] uint8 (FP8 raw bytes)
    k_scales: torch.Tensor,      # [N] float32
    weights: torch.Tensor,       # [num_rows, H] float32
    context_len: int,
    max_model_len: int,
    *,
    k_f32_cache: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """SM70 cascade-compatible GEMM indexer logits (decode path).

    The expensive K dequant ``decode_fp8(k_values) * k_scales`` is
    pulled out of the per-step path because in the decode loop the
    snapshot only grows by one row every ``compress_ratio`` steps.
    Callers pass a pre-built fp32 ``k_f32_cache`` (shape ``[N, D]``)
    to avoid redoing the dequant; if None, we build it here (slow
    fallback; correctness only).

    Returns: ``[num_rows, max_model_len]`` fp32, -inf-padded past
    ``context_len``.
    """
    assert q.is_cuda
    if q.dim() == 3:
        q = q.unsqueeze(0)
    batch_size, next_n, num_heads, head_dim = q.shape
    assert batch_size == 1, "cascade-gemm currently supports batch=1"
    assert k_values.dim() == 2 and k_values.shape[1] == head_dim
    assert k_scales.dim() == 1 and k_scales.shape[0] == k_values.shape[0]
    assert weights.dim() == 2 and weights.shape[1] == num_heads
    assert weights.shape[0] == next_n
    assert context_len <= k_values.shape[0]
    device = q.device

    if out is None:
        out = torch.full(
            (next_n, max_model_len),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
    else:
        assert out.shape == (next_n, max_model_len)
        assert out.dtype == torch.float32
        out.fill_(float("-inf"))

    if context_len <= 0:
        return out

    q_u8 = q.view(torch.uint8)
    NUM_BUCKETS = 8

    # 1. Q dequant + 8-bucket collapse + w_emul (one fused Triton kernel).
    q_emul = torch.empty((next_n, NUM_BUCKETS, head_dim), dtype=torch.float32, device=device)
    w_emul = torch.zeros((next_n, NUM_BUCKETS), dtype=torch.float32, device=device)
    NUM_D_CHUNKS = head_dim // 64
    grid_q = (next_n * NUM_D_CHUNKS,)
    _decode_q_emul_kernel[grid_q](
        q_u8,
        q_u8.stride(0),
        q_u8.stride(1),
        q_u8.stride(2),
        weights,
        weights.stride(0),
        q_emul,
        q_emul.stride(0),
        q_emul.stride(1),
        w_emul,
        w_emul.stride(0),
        next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )

    # 2. K dequant + scale (skipped if k_f32_cache provided + complete).
    if k_f32_cache is not None and k_f32_cache.shape[0] >= context_len:
        k_f32 = k_f32_cache[:context_len]
    else:
        k_f32 = torch.empty((context_len, head_dim), dtype=torch.float32, device=device)
        BLOCK_N = 32
        n_blocks = (context_len + BLOCK_N - 1) // BLOCK_N
        _decode_k_to_fp32_kernel[(n_blocks,)](
            k_values,
            k_values.stride(0),
            k_scales,
            k_f32,
            k_f32.stride(0),
            context_len,
            HEAD_DIM=head_dim,
            BLOCK_N=BLOCK_N,
        )

    # 3. fp32 GEMM: q_emul [next_n, 8, D] @ k_f32.T [D, N] -> [next_n, 8, N]
    if next_n == 1:
        scores = torch.matmul(q_emul[0], k_f32.T)  # [8, N]
        scores = scores.unsqueeze(0)  # [1, 8, N]
    else:
        scores = torch.matmul(q_emul, k_f32.T)  # [next_n, 8, N]

    # 4. ReLU + weighted-sum epilogue (fused Triton kernel) writes
    # logits + -inf padding directly into ``out``. Scores tensor is
    # [next_n, 8, context_len] only (no max_model_len padding).
    BLOCK_K = 32
    n_k_blocks = (max_model_len + BLOCK_K - 1) // BLOCK_K
    grid_e = (next_n * n_k_blocks,)
    _cascade_gemm_epilogue_kernel[grid_e](
        scores,
        scores.stride(0),
        scores.stride(1),
        w_emul,
        w_emul.stride(0),
        out,
        out.stride(0),
        int(context_len),
        max_model_len,
        BLOCK_K=BLOCK_K,
    )
    return out
