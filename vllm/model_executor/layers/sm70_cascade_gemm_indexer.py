# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 cascade-compatible GEMM indexer logits (decode path).

Replaces the decode call to ``sm70_fp8_paged_mqa_logits`` with a
contiguous-K fp32 GEMM. See
``.kiro/specs/deepseek-v4-decode-indexer-on-compressed-kv/`` for the
original throughput rationale and
``.kiro/specs/deepseek-v4-indexer-head-reduction-correctness/`` for the
head-reduction correctness fix.

The K snapshot is pre-staged in a higher-level helper; this module only
handles the per-call compute.

Indexer contract (matches ``_sm70_fp8_paged_mqa_logits_kernel`` and the
model's definition — TRUE per-head reduction):

  for k_pos in [0, context_len):
      logit[k_pos] = sum_{h=0..NUM_HEADS-1}
                         relu( sum_d decode(q[h,d]) * decode(k[k_pos,d]) * k_scale )
                       * w_h

  Each head gets its OWN ReLU and OWN weight w_h. The GEMM computes
  per-head scores ``q_emul [next_n, H, D] @ k_f32.T -> [next_n, H, N]``
  and the epilogue applies ``relu(s_h) * w_h`` and sums over heads.

HISTORY: an earlier version collapsed heads 7..H-1 into a single summed
Q row ("8-bucket cascade"). That is only correct for NUM_HEADS <= 8;
for the real indexer (H=64) ``relu(sum) != sum(relu)`` and weights
8..63 were dropped, corrupting top-k KV selection (the long-context
haystack root cause). The per-position cascade-vs-paged stress sweep in
``tests/v1/attention/test_sm70_cascade_gemm_indexer.py`` still holds
(both paths now use true per-head reduction; values agree within fp32
GEMM precision, ~1e-7).
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
    q_emul_ptr,           # float32 [num_rows, H, D]
    q_emul_stride_row,
    q_emul_stride_head,
    next_n,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Fuse Q FP8 decode into a per-head fp32 [num_rows, H, D] tensor.

    One program per (next_idx, d_chunk). Each program decodes
    Q[next_idx, h, d_chunk] for ALL heads h and writes them to their own
    row in q_emul (true 64-head layout; no bucket collapse).

    NOTE: an earlier implementation collapsed heads 7..H-1 into a single
    summed "bucket 7" row, which corrupts the indexer score for H>8
    (relu(sum) != sum(relu); weights 8..H-1 dropped). See
    .kiro/specs/deepseek-v4-indexer-head-reduction-correctness/.
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
    q_emul_row_base = q_emul_ptr + pid_row * q_emul_stride_row

    # Decode every head into its own row (no bucketing).
    for h in tl.static_range(NUM_HEADS):
        q_uint = tl.load(q_row_base + h * q_stride_h + d_range)
        q_chunk = _decode_fp8_e4m3fn_inline(q_uint)
        tl.store(q_emul_row_base + h * q_emul_stride_head + d_range, q_chunk)


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
    scores_ptr,           # float32 [num_rows, H, context_len] (no padding)
    scores_stride_row,
    scores_stride_head,
    weights_ptr,          # float32 [num_rows, H]
    weights_stride_row,
    out_ptr,              # float32 [num_rows, max_model_len]
    out_stride_row,
    context_len,
    max_model_len,
    NUM_HEADS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Apply per-head ReLU + weighted sum across ALL NUM_HEADS heads.

    One program per (next_idx, k_block). Each program handles BLOCK_K
    consecutive k positions, accumulating sum_h relu(s_h) * w_h over the
    full head range (true 64-head semantics, no 8-bucket collapse). Out
    positions >= context_len are written with -inf to preserve the
    contract.
    """
    pid = tl.program_id(0)
    pid_row = pid // tl.cdiv(max_model_len, BLOCK_K)
    pid_k_block = pid - pid_row * tl.cdiv(max_model_len, BLOCK_K)

    k_pos_base = pid_k_block * BLOCK_K
    k_range = k_pos_base + tl.arange(0, BLOCK_K)
    in_range = k_range < context_len

    weights_row = weights_ptr + pid_row * weights_stride_row
    scores_row = scores_ptr + pid_row * scores_stride_row

    logit = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for h in tl.static_range(NUM_HEADS):
        w_h = tl.load(weights_row + h)
        s_h = tl.load(
            scores_row + h * scores_stride_head + k_range,
            mask=in_range, other=0.0,
        )
        logit += tl.maximum(s_h, 0.0) * w_h

    # Write logit if in_range, otherwise -inf.
    NEG_INF = float("-inf")
    out_val = tl.where(in_range, logit, NEG_INF)
    out_addr = out_ptr + pid_row * out_stride_row + k_range
    out_mask = k_range < max_model_len
    tl.store(out_addr, out_val, mask=out_mask)


@triton.jit
def _cascade_gemm_epilogue_tensor_len_kernel(
    scores_ptr,           # float32 [num_rows, H, static_context_len]
    scores_stride_row,
    scores_stride_head,
    weights_ptr,          # float32 [num_rows, H]
    weights_stride_row,
    context_lens_ptr,     # int32 flattened, one length per decode row
    out_ptr,              # float32 [num_rows, max_model_len]
    out_stride_row,
    static_context_len,
    max_model_len,
    NUM_HEADS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Capture-safe epilogue variant (true 64-head reduction).

    Unlike `_cascade_gemm_epilogue_kernel`, the true context length is
    read from a device tensor so the host never calls `.item()` while a
    CUDA graph is being captured.  The GEMM is fixed-shape over
    `static_context_len`; positions beyond the runtime context are
    masked to -inf here.
    """
    pid = tl.program_id(0)
    pid_row = pid // tl.cdiv(max_model_len, BLOCK_K)
    pid_k_block = pid - pid_row * tl.cdiv(max_model_len, BLOCK_K)

    context_len = tl.load(context_lens_ptr + pid_row)
    context_len = tl.minimum(context_len, static_context_len)

    k_pos_base = pid_k_block * BLOCK_K
    k_range = k_pos_base + tl.arange(0, BLOCK_K)
    in_range = (k_range < context_len) & (k_range < static_context_len)

    weights_row = weights_ptr + pid_row * weights_stride_row
    scores_row = scores_ptr + pid_row * scores_stride_row

    logit = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for h in tl.static_range(NUM_HEADS):
        w_h = tl.load(weights_row + h)
        s_h = tl.load(
            scores_row + h * scores_stride_head + k_range,
            mask=in_range, other=0.0,
        )
        logit += tl.maximum(s_h, 0.0) * w_h

    out_val = tl.where(in_range, logit, float("-inf"))
    out_addr = out_ptr + pid_row * out_stride_row + k_range
    tl.store(out_addr, out_val, mask=k_range < max_model_len)


def _decode_q_and_weights(
    q: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Decode Q into per-head fp32 rows and return the head weights.

    Returns ``(q_emul [next_n, H, D] fp32, weights [next_n, H] fp32,
    next_n, num_heads, head_dim)``. ``weights`` is returned as-is (the
    true per-head weights); the previous 8-bucket ``w_emul`` collapse is
    gone.
    """
    if q.dim() == 3:
        q = q.unsqueeze(0)
    batch_size, next_n, num_heads, head_dim = q.shape
    assert batch_size == 1, "cascade-gemm currently supports batch=1"
    assert weights.dim() == 2 and weights.shape[1] == num_heads
    assert weights.shape[0] == next_n

    q_u8 = q.view(torch.uint8)
    device = q.device
    q_emul = torch.empty(
        (next_n, num_heads, head_dim), dtype=torch.float32, device=device
    )
    num_d_chunks = head_dim // 64
    _decode_q_emul_kernel[(next_n * num_d_chunks,)](
        q_u8,
        q_u8.stride(0),
        q_u8.stride(1),
        q_u8.stride(2),
        q_emul,
        q_emul.stride(0),
        q_emul.stride(1),
        next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )
    return q_emul, weights.contiguous(), next_n, num_heads, head_dim


def sm70_cascade_gemm_indexer_from_snapshot(
    *,
    q: torch.Tensor,
    k_f32_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Capture-safe cascade-compatible GEMM indexer from a fixed snapshot.

    `k_f32_cache` is a preallocated contiguous snapshot.  The GEMM shape is
    fixed to `min(k_f32_cache.shape[0], max_model_len)`, while runtime
    masking uses `context_lens` on device.  This avoids host-side `.item()`
    during FULL cudagraph capture.
    """
    assert q.is_cuda
    assert k_f32_cache.is_cuda and k_f32_cache.dtype == torch.float32
    static_context_len = min(int(k_f32_cache.shape[0]), int(max_model_len))
    assert static_context_len > 0

    q_emul, w_emul, next_n, _, head_dim = _decode_q_and_weights(q, weights)
    num_heads = q_emul.shape[1]
    assert k_f32_cache.dim() == 2 and k_f32_cache.shape[1] == head_dim

    if out is None:
        out = torch.empty(
            (next_n, max_model_len), device=q.device, dtype=torch.float32
        )
    else:
        assert out.shape == (next_n, max_model_len)
        assert out.dtype == torch.float32

    k_f32 = k_f32_cache[:static_context_len]
    if next_n == 1:
        scores = torch.matmul(q_emul[0], k_f32.T).unsqueeze(0)
    else:
        scores = torch.matmul(q_emul, k_f32.T)

    import os as _os
    if _os.environ.get("VLLM_HAYSTACK_NEEDLE_RAW") and not torch.cuda.is_current_stream_capturing():
        try:
            from vllm.logger import init_logger as _il
            _lg = _il(__name__)
            kf = k_f32
            kfin = kf[kf.isfinite()]
            qfin = q_emul[q_emul.isfinite()]
            wfin = w_emul[w_emul.isfinite()]
            # which K rows are huge (>1e6)?
            row_absmax = kf.abs().amax(dim=1)
            n_big_rows = int((row_absmax > 1e6).sum().item())
            n_nan_rows = int(kf.isnan().any(dim=1).sum().item())
            _lg.info(
                "CASCADE_STATS ctx=%d k_absmax=%.3g k_nan=%d q_absmax=%.3g "
                "q_nan=%d w_absmax=%.3g big_k_rows(>1e6)=%d nan_k_rows=%d",
                static_context_len,
                kfin.abs().max().item() if kfin.numel() else float("nan"),
                int(kf.isnan().sum().item()),
                qfin.abs().max().item() if qfin.numel() else float("nan"),
                int(q_emul.isnan().sum().item()),
                wfin.abs().max().item() if wfin.numel() else float("nan"),
                n_big_rows, n_nan_rows,
            )
        except Exception:
            pass

    context_lens_flat = context_lens.reshape(-1)
    BLOCK_K = 32
    n_k_blocks = (max_model_len + BLOCK_K - 1) // BLOCK_K
    _cascade_gemm_epilogue_tensor_len_kernel[(next_n * n_k_blocks,)](
        scores,
        scores.stride(0),
        scores.stride(1),
        w_emul,
        w_emul.stride(0),
        context_lens_flat,
        out,
        out.stride(0),
        static_context_len,
        max_model_len,
        NUM_HEADS=num_heads,
        BLOCK_K=BLOCK_K,
    )
    return out


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

    # 1. Q dequant into per-head fp32 rows (one fused Triton kernel).
    q_emul, w_emul, _, _, _ = _decode_q_and_weights(q, weights)
    num_heads_local = q_emul.shape[1]

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

    # 3. fp32 GEMM: q_emul [next_n, H, D] @ k_f32.T [D, N] -> [next_n, H, N]
    if next_n == 1:
        scores = torch.matmul(q_emul[0], k_f32.T)  # [H, N]
        scores = scores.unsqueeze(0)  # [1, H, N]
    else:
        scores = torch.matmul(q_emul, k_f32.T)  # [next_n, H, N]

    # 4. Per-head ReLU + weighted-sum epilogue (fused Triton kernel)
    # writes logits + -inf padding directly into ``out``. Scores tensor
    # is [next_n, H, context_len] only (no max_model_len padding).
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
        NUM_HEADS=num_heads_local,
        BLOCK_K=BLOCK_K,
    )
    return out
