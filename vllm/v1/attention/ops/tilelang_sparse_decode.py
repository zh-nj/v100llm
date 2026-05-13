# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang sparse MLA DECODE kernel for DSv4F on Volta (SM70).

Decode is structurally different from prefill on two axes:

  1. s_q = 1 per request (no query-sequence batching).
  2. All h_q=64 heads of a query attend to the same KV rows, so we
     batch all 64 heads per CUDA block. This is also required by
     tilelang's SM70 MMA emitter: a [16, 16] gemm with 128 threads
     fails warp_col_tiles >= 16; [64, 16] works.

Grid: ``(s_q, batch, num_splits)``. Split-KV on grid.z is what
saturates the SMs at s_q=1; tuned per-shape via autotune.

Per-split outputs go to ``(o_accum, lse_accum)``; ``_combine_torch``
reduces across splits. Promotable to a fused Triton combine if the
partial kernel wins big enough to absorb the combine's launch cost.

Activated by ``VLLM_SM70_USE_TILELANG_SPARSE_DECODE=1``. Gated
independently of the prefill switch.
"""

from typing import Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TILELANG_AVAILABLE: Optional[bool] = None


def is_tilelang_available() -> Tuple[bool, Optional[str]]:
    global _TILELANG_AVAILABLE
    if _TILELANG_AVAILABLE is not None:
        return _TILELANG_AVAILABLE, (
            None if _TILELANG_AVAILABLE else "tilelang import failed"
        )
    try:
        import tilelang  # noqa: F401
        _TILELANG_AVAILABLE = True
        return True, None
    except Exception as exc:
        _TILELANG_AVAILABLE = False
        return False, f"tilelang not available: {exc}"


def _build_partial_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        out_idx=[-2, -1],
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_decode_partial(
        heads: int,
        dim: int,
        tail_dim: int,
        topk: int,
        num_splits: int,
        sm_scale: float,
        heads_per_block: int = 64,
        block_I: int = 32,
        num_stages: int = 1,
        threads: int = 128,
        has_sink: bool = True,
        has_topk_length: bool = True,
    ):
        assert dim == tilelang.math.next_power_of_2(dim)
        assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(tail_dim)
        assert topk % block_I == 0, f"topk={topk} must be divisible by block_I={block_I}"
        assert heads % heads_per_block == 0, \
            f"heads={heads} must be divisible by heads_per_block={heads_per_block}"
        assert heads_per_block in (16, 32, 64, 128), \
            "heads_per_block must be 16/32/64/128 for SM70 MMA layout"

        LOG2E = 1.4426950408889634
        sm_scale_log2 = sm_scale * LOG2E
        has_tail = tail_dim > 0
        BI = block_I
        H = heads_per_block
        REPLICATE_H = heads // heads_per_block

        per_split_I_tiles = (topk // num_splits + BI - 1) // BI
        split_tile_size = per_split_I_tiles * BI
        # Logical split size (without BI rounding). Used for end-cap so
        # split[i] doesn't steal rows from split[i+1].
        logical_split_size = topk // num_splits
        assert topk % num_splits == 0, \
            f"topk={topk} must be divisible by num_splits={num_splits}"

        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        q_shape = [batch, seq_len, heads, dim + tail_dim]
        kv_shape = [batch, seq_len_kv, 1, dim + tail_dim]
        indices_shape = [batch, seq_len, 1, topk]
        sink_shape = [heads]
        topk_len_shape = [batch, seq_len]
        o_accum_shape = [num_splits, batch, seq_len, heads, dim]
        lse_accum_shape = [num_splits, batch, seq_len, heads]

        dtype = T.float16
        accum = T.float32
        i32 = T.int32

        D = dim
        D_tail = tail_dim

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, i32),
            Sink: T.Tensor(sink_shape, accum),
            TopkLen: T.Tensor(topk_len_shape, i32),
            OAccum: T.Tensor(o_accum_shape, accum),
            LseAccum: T.Tensor(lse_accum_shape, accum),
        ):
            with T.Kernel(
                seq_len * REPLICATE_H, batch, num_splits, threads=threads,
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([H, D], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                if has_tail:
                    Q_tail_shared = T.alloc_shared([H, D_tail], dtype)
                    K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                S_shared = T.alloc_shared([H, BI], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H, D], accum)
                acc_s = T.alloc_fragment([H, BI], accum)
                sumexp = T.alloc_fragment([H], accum)
                sumexp_i = T.alloc_fragment([H], accum)
                alpha = T.alloc_fragment([H], accum)
                m_i = T.alloc_fragment([H], accum)
                m_i_prev = T.alloc_fragment([H], accum)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2 ** 30))

                b_i = by
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                h_grp = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H)
                H0 = h_grp * H
                H1 = H0 + H
                split_i = bz

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                if has_tail:
                    T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                tl_cur = T.alloc_fragment([1], i32)
                if has_topk_length:
                    tl_cur[0] = TopkLen[b_i, s_i]
                else:
                    tl_cur[0] = topk

                split_start = split_i * logical_split_size
                split_end_cap = T.min(split_start + logical_split_size, topk)

                for i_local in T.Pipelined(per_split_I_tiles, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        pos = split_start + i_local * BI + bi_i
                        idx_raw = T.if_then_else(
                            pos < topk,
                            Indices[b_i, s_i, 0, pos],
                            -1,
                        )
                        valid = (
                            (pos < tl_cur[0])
                            & (pos < split_end_cap)
                            & (idx_raw >= 0)
                            & (idx_raw < seq_len_kv)
                        )
                        mask[bi_i] = valid

                    for bi_i, d_i in T.Parallel(BI, D):
                        pos = split_start + i_local * BI + bi_i
                        idx_raw = T.if_then_else(
                            (pos < topk),
                            Indices[b_i, s_i, 0, pos],
                            0,
                        )
                        idx_clamped = T.if_then_else(mask[bi_i], idx_raw, 0)
                        KV_shared[bi_i, d_i] = KV[b_i, idx_clamped, 0, d_i]
                    if has_tail:
                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            pos = split_start + i_local * BI + bi_i
                            idx_raw = T.if_then_else(
                                (pos < topk),
                                Indices[b_i, s_i, 0, pos],
                                0,
                            )
                            idx_clamped = T.if_then_else(mask[bi_i], idx_raw, 0)
                            K_tail_shared[bi_i, d_i] = KV[
                                b_i, idx_clamped, 0, D + d_i
                            ]

                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        Q_shared, KV_shared, acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    if has_tail:
                        T.gemm(
                            Q_tail_shared, K_tail_shared, acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H):
                        alpha[h_i] = T.exp2(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale_log2)
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale_log2
                            - m_i[h_i] * sm_scale_log2)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(
                        S_shared, KV_shared, acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for h_i in T.Parallel(H):
                    # Guard against empty split: sumexp may stay 0 if every
                    # row in this split was masked out. Write -inf lse so
                    # the combine op correctly ignores this partial.
                    LseAccum[split_i, b_i, s_i, H0 + h_i] = T.if_then_else(
                        sumexp[h_i] > 0,
                        T.log2(sumexp[h_i]) / LOG2E + m_i[h_i] * sm_scale,
                        -T.infinity(accum),
                    )
                # Store per-split conditional expectation (acc_o / sumexp)
                # so the Python combine op just weights by softmax(lse_s).
                # Empty-split guard: divide only when sumexp > 0; else write 0.
                for h_i, d_i in T.Parallel(H, D):
                    OAccum[split_i, b_i, s_i, H0 + h_i, d_i] = T.if_then_else(
                        sumexp[h_i] > 0,
                        acc_o[h_i, d_i] / sumexp[h_i],
                        T.float32(0.0),
                    )

        return main

    return build_decode_partial


_PARTIAL_CACHE: dict = {}
_PARTIAL_FACTORY = None


def _get_partial_kernel(
    heads: int, dim: int, tail_dim: int, topk: int, num_splits: int,
    sm_scale: float, has_sink: bool, has_topk_length: bool,
    heads_per_block: int, block_I: int, num_stages: int, threads: int,
):
    global _PARTIAL_FACTORY
    if _PARTIAL_FACTORY is None:
        _PARTIAL_FACTORY = _build_partial_kernel_factory()
    key = (heads, dim, tail_dim, topk, num_splits, sm_scale,
           has_sink, has_topk_length,
           heads_per_block, block_I, num_stages, threads)
    if key not in _PARTIAL_CACHE:
        logger.info(
            "TileLang decode partial: JIT compiling heads=%d dim=%d "
            "tail=%d topk=%d n_splits=%d hpb=%d BI=%d stages=%d threads=%d",
            heads, dim, tail_dim, topk, num_splits,
            heads_per_block, block_I, num_stages, threads,
        )
        _PARTIAL_CACHE[key] = _PARTIAL_FACTORY(
            heads=heads, dim=dim, tail_dim=tail_dim, topk=topk,
            num_splits=num_splits, sm_scale=sm_scale,
            heads_per_block=heads_per_block,
            block_I=block_I, num_stages=num_stages, threads=threads,
            has_sink=has_sink, has_topk_length=has_topk_length,
        )
        logger.info("TileLang decode partial: compile complete")
    return _PARTIAL_CACHE[key]


def _combine_torch(
    o_accum: torch.Tensor,
    lse_accum: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch split-combine. o_accum is the per-split conditional
    expectation ``E[v | tokens in split]`` (already divided by sumexp_s
    inside the kernel); lse_accum is natural-base log-sum-exp per split.
    """
    # Global max over splits for stability. Clamp -inf upward to avoid
    # NaN when a split masked out all rows.
    m = lse_accum.amax(dim=0)                        # [B,sq,H]
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    # exp(lse_s - m). Masked-out splits get exp(-inf) = 0.
    exp_lse = torch.exp(lse_accum - m_safe.unsqueeze(0))
    exp_lse = torch.where(
        torch.isfinite(lse_accum),
        exp_lse,
        torch.zeros_like(exp_lse),
    )
    sum_exp = exp_lse.sum(dim=0)                     # [B,sq,H]
    if attn_sink is not None:
        sink_exp = torch.exp(attn_sink - m_safe)      # [B,sq,H] via broadcast
        denom_sink = sum_exp + sink_exp
    else:
        denom_sink = sum_exp
    # Weights across splits (ignoring sink for the weighted-avg; sink
    # shrinks the output uniformly below via sum_exp/denom_sink).
    w = (exp_lse / sum_exp.unsqueeze(0).clamp_min(1e-30)).unsqueeze(-1)
    o = (o_accum * w).sum(dim=0)
    if attn_sink is not None:
        o = o * (sum_exp / denom_sink.clamp_min(1e-30)).unsqueeze(-1)
    final_lse = m_safe + torch.log(sum_exp.clamp_min(1e-30))
    return o.to(out_dtype), final_lse


def flash_mla_sparse_decode_tilelang(
    q: torch.Tensor,
    kv_dense: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    *,
    heads_per_block: int = 64,
    num_splits: int = 8,
    block_I: int = 32,
    num_stages: int = 1,
    threads: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, s_q, heads, d_qk = q.shape
    topk = indices.shape[-1]
    tail_dim = d_qk - d_v
    assert tail_dim in (0, 64)

    in_dtype = q.dtype
    q_fp16 = q.to(torch.float16).contiguous() if q.dtype != torch.float16 else q.contiguous()
    kv_fp16 = kv_dense.to(torch.float16).contiguous() if kv_dense.dtype != torch.float16 else kv_dense.contiguous()
    indices_i32 = indices.to(torch.int32).contiguous()

    has_sink = attn_sink is not None
    has_topk_length = topk_length is not None
    sink_f32 = (
        attn_sink.to(torch.float32).contiguous() if has_sink
        else torch.zeros(heads, dtype=torch.float32, device=q.device)
    )
    if has_topk_length:
        tl_i32 = topk_length.to(torch.int32).contiguous().view(B, s_q)
    else:
        tl_i32 = torch.full((B, s_q), topk, dtype=torch.int32, device=q.device)

    kernel = _get_partial_kernel(
        heads=heads, dim=d_v, tail_dim=tail_dim, topk=topk,
        num_splits=num_splits, sm_scale=sm_scale,
        has_sink=has_sink, has_topk_length=has_topk_length,
        heads_per_block=heads_per_block,
        block_I=block_I, num_stages=num_stages, threads=threads,
    )
    o_accum, lse_accum = kernel(q_fp16, kv_fp16, indices_i32, sink_f32, tl_i32)

    out_tensor, lse = _combine_torch(
        o_accum, lse_accum,
        attn_sink=sink_f32 if has_sink else None,
        out_dtype=in_dtype,
    )
    if out is not None:
        out.copy_(out_tensor)
        out_tensor = out
    return out_tensor, lse
