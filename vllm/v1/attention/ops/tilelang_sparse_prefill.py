# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang sparse MLA forward kernel for DSv4F on Volta (SM70).

Drop-in replacement for `flash_mla_sparse_fwd` that batches all 64
query heads per CUDA block (vs FlashMLA's HEADS_PER_BLOCK=4), cutting
KV memory traffic by 4x at the problem level. On V100, per-call time
drops ~3x at the DSv4F production shape (s_q=1803, h_q=64, topk=256,
d_qk=576, d_v=512).

This module is activated by `VLLM_SM70_USE_TILELANG_SPARSE_PREFILL=1`.
On other paths it is not imported.

Adapted from THUDM/slime's DSv3.2 sparse MLA reference, itself adapted
from tile-ai/tilelang's deepseek_v32/sparse_mla_fwd.py.
"""

from typing import Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TILELANG_AVAILABLE: Optional[bool] = None


def is_tilelang_available() -> Tuple[bool, Optional[str]]:
    """Check if tilelang is importable in this environment."""
    global _TILELANG_AVAILABLE
    if _TILELANG_AVAILABLE is not None:
        return _TILELANG_AVAILABLE, (
            None if _TILELANG_AVAILABLE else "tilelang import failed"
        )
    try:
        import tilelang  # noqa: F401

        _TILELANG_AVAILABLE = True
        return True, None
    except Exception as exc:  # pragma: no cover
        _TILELANG_AVAILABLE = False
        return False, f"tilelang not available: {exc}"


# Build the kernel lazily — tilelang decorators must run at import
# time of the inner `build_sparse_mla_fwd_kernel`, but we want to
# defer tilelang import until the caller opts in.


def _build_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_sparse_mla_fwd_kernel(
        heads: int,
        dim: int,
        tail_dim: int,
        topk: int,
        kv_group: int = 1,
        sm_scale: Optional[float] = None,
        block_I: int = 16,
        num_stages: int = 1,
        threads: int = 128,
        has_sink: bool = True,
        has_topk_length: bool = True,
        output_dtype_str: str = "float16",
    ):
        """Emit a fresh JIT-compiled TileLang kernel for the given config.

        Argument order matches the @tilelang.jit outputs contract:
          Q, KV, Indices, Sink, TopkLen  ->  Output, MaxLogits, Lse

        `output_dtype_str` ∈ {"float16", "bfloat16"}:
          - fp16: kernel emits fp16 Output; wrapper casts to bf16
            externally (extra ~330 ms torch copy kernel per prefill
            in the DSv4F e2e workload).
          - bf16: kernel emits bf16 Output directly, saving the extra
            copy. Q/KV inputs are still fp16 (that conversion stays in
            the wrapper since inline Cast on the hot KV-gather loop
            breaks tilelang's layout optimization — tested, 3x slower).
        """
        assert dim == tilelang.math.next_power_of_2(dim)
        assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(tail_dim)
        assert topk % block_I == 0
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        LOG2E = 1.4426950408889634
        sm_scale_log2 = sm_scale * LOG2E

        has_tail = tail_dim > 0

        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        head_kv = heads // kv_group
        q_shape = [batch, seq_len, heads, dim + tail_dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
        o_shape = [batch, seq_len, heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]
        sink_shape = [heads]
        topk_len_shape = [batch, seq_len]
        max_shape = [batch, seq_len, heads]
        lse_shape = [batch, seq_len, heads]
        indices_dtype = T.int32
        topk_len_dtype = T.int32
        dtype = T.float16  # Q/KV I/O and smem compute dtype
        out_dtype = T.bfloat16 if output_dtype_str == "bfloat16" else T.float16
        accum_dtype = T.float32
        _shape_refs = (
            q_shape, kv_shape, indices_shape, sink_shape,
            topk_len_shape, o_shape, max_shape, lse_shape,
        )

        H = head_kv
        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != H:
            assert kv_group == 1

        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim
        D_tail = tail_dim

        if head_kv > 64:
            assert head_kv % 64 == 0
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, indices_dtype),
            Sink: T.Tensor(sink_shape, accum_dtype),
            TopkLen: T.Tensor(topk_len_shape, topk_len_dtype),
            Output: T.Tensor(o_shape, out_dtype),
            MaxLogits: T.Tensor(max_shape, accum_dtype),
            Lse: T.Tensor(lse_shape, accum_dtype),
        ):
            with T.Kernel(
                seq_len * REPLICATE_H, batch, kv_group, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                if has_tail:
                    Q_tail_shared = T.alloc_shared(
                        [H_per_block, D_tail], dtype)
                    K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2 ** 30))

                b_i, g_i = by, bz
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

                H0 = g_i * padded_H + (
                    0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
                )
                H1 = H0 + H_per_block

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                if has_tail:
                    T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                tl_cur = T.alloc_fragment([1], topk_len_dtype)
                if has_topk_length:
                    tl_cur[0] = TopkLen[b_i, s_i]
                else:
                    tl_cur[0] = topk

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        pos = i_i * BI + bi_i
                        idx = Indices[b_i, s_i, g_i, pos]
                        valid = ((idx >= 0) & (idx < seq_len_kv)
                                 & (pos < tl_cur[0]))
                        mask[bi_i] = valid

                    for bi_i, d_i in T.Parallel(BI, D):
                        idx_raw = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        idx_clamped = T.if_then_else(
                            mask[bi_i], idx_raw, 0)
                        KV_shared[bi_i, d_i] = KV[b_i, idx_clamped, g_i, d_i]
                    if has_tail:
                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            idx_raw = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                            idx_clamped = T.if_then_else(
                                mask[bi_i], idx_raw, 0)
                            K_tail_shared[bi_i, d_i] = KV[
                                b_i, idx_clamped, g_i, D + d_i
                            ]

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
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
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp2(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale_log2)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale_log2
                            - m_i[h_i] * sm_scale_log2)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = (
                            sumexp[h_i] * alpha[h_i] + sumexp_i[h_i])
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(
                        S_shared, KV_shared, acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                # max_logits = sm_scale * max(Q·K^T) in natural base.
                for h_i in T.Parallel(H_per_block):
                    MaxLogits[b_i, s_i, H0 + h_i] = m_i[h_i] * sm_scale

                # LSE natural: log(sumexp) + m_i*sm_scale
                for h_i in T.Parallel(H_per_block):
                    Lse[b_i, s_i, H0 + h_i] = (
                        T.log2(sumexp[h_i]) / LOG2E
                        + m_i[h_i] * sm_scale
                    )

                # attn_sink: scale output by exp(lse) / (exp(lse)+exp(sink)).
                # Equivalent to dividing by (sumexp + exp(sink - m_i*sm_scale))
                # where the offset makes denom numerically stable.
                if has_sink:
                    for h_i in T.Parallel(H_per_block):
                        extra = T.exp2(
                            (Sink[H0 + h_i] - m_i[h_i] * sm_scale)
                            * LOG2E)
                        sumexp[h_i] = sumexp[h_i] + extra

                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] / sumexp[h_i]

                # Output store: fp32 acc_o -> out_dtype HBM directly
                # (fold bf16 cast into the kernel when out_dtype=bf16).
                if output_dtype_str == "bfloat16":
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        Output[b_i, s_i, H0 + h_i, d_i] = T.Cast(
                            out_dtype, acc_o[h_i, d_i])
                else:
                    T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

        return main

    return build_sparse_mla_fwd_kernel


_KERNEL_CACHE: dict = {}
_KERNEL_FACTORY = None


def _get_kernel(
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    sm_scale: float,
    has_sink: bool,
    has_topk_length: bool,
    block_I: int,
    num_stages: int,
    threads: int,
    output_dtype_str: str = "float16",
):
    global _KERNEL_FACTORY
    if _KERNEL_FACTORY is None:
        _KERNEL_FACTORY = _build_kernel_factory()
    key = (heads, dim, tail_dim, topk, sm_scale, has_sink,
           has_topk_length, block_I, num_stages, threads,
           output_dtype_str)
    if key not in _KERNEL_CACHE:
        logger.info(
            "TileLang sparse MLA: JIT compiling for "
            "heads=%d dim=%d tail=%d topk=%d sink=%s topk_len=%s "
            "out=%s (BI=%d stages=%d threads=%d) — takes ~45s",
            heads, dim, tail_dim, topk, has_sink, has_topk_length,
            output_dtype_str, block_I, num_stages, threads,
        )
        _KERNEL_CACHE[key] = _KERNEL_FACTORY(
            heads=heads, dim=dim, tail_dim=tail_dim, topk=topk,
            sm_scale=sm_scale, block_I=block_I, num_stages=num_stages,
            threads=threads, has_sink=has_sink,
            has_topk_length=has_topk_length,
            output_dtype_str=output_dtype_str,
        )
        logger.info("TileLang sparse MLA: compile complete")
    return _KERNEL_CACHE[key]


def flash_mla_sparse_fwd_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    block_I: int = 16,
    num_stages: int = 1,
    threads: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for FlashMLA's `flash_mla_sparse_fwd`.

    See FlashMLA's flash_mla_interface.py for the full contract.
    """
    s_q, h_q, d_qk = q.shape
    s_kv, _h_kv, _d_qk2 = kv.shape
    _s_q2, _h_kv2, topk = indices.shape
    assert d_v == 512
    dim = d_v
    tail_dim = d_qk - d_v
    assert tail_dim in (0, 64), f"unsupported tail_dim {tail_dim}"

    input_dtype = q.dtype
    # Q/KV go in as fp16 (V100 MMA requirement). Output goes out as
    # the input dtype directly when bf16 — fold the bf16 cast into
    # the kernel's final store (saves ~330 ms torch copy kernel per
    # prefill at the DSv4F production shape).
    output_dtype_str = "bfloat16" if input_dtype == torch.bfloat16 else "float16"
    q_fp16 = q.to(torch.float16) if q.dtype != torch.float16 else q
    kv_fp16 = kv.to(torch.float16) if kv.dtype != torch.float16 else kv

    Q_b = q_fp16.unsqueeze(0).contiguous()
    KV_b = kv_fp16.unsqueeze(0).contiguous()
    Indices_b = indices.unsqueeze(0).contiguous()

    has_sink = attn_sink is not None
    has_topk_length = topk_length is not None

    if has_sink:
        Sink = attn_sink.to(torch.float32).contiguous()
    else:
        Sink = torch.zeros(h_q, dtype=torch.float32, device=q.device)

    if has_topk_length:
        TopkLen_b = topk_length.to(torch.int32).unsqueeze(0).contiguous()
    else:
        TopkLen_b = torch.full(
            (1, s_q), topk, dtype=torch.int32, device=q.device)

    kernel = _get_kernel(
        heads=h_q, dim=dim, tail_dim=tail_dim, topk=topk,
        sm_scale=sm_scale, has_sink=has_sink,
        has_topk_length=has_topk_length,
        block_I=block_I, num_stages=num_stages, threads=threads,
        output_dtype_str=output_dtype_str,
    )

    out_tl, max_tl, lse_tl = kernel(
        Q_b, KV_b, Indices_b, Sink, TopkLen_b)

    output = out_tl.squeeze(0)
    max_logits = max_tl.squeeze(0)
    lse = lse_tl.squeeze(0)

    # If output_dtype_str="bfloat16", output is already bf16 from the
    # kernel; no cast needed. Only cast when we asked for fp16 output
    # but caller wanted something else.
    if output_dtype_str == "float16" and input_dtype != torch.float16:
        output = output.to(input_dtype)

    if out is not None:
        out.copy_(output)
        output = out

    return output, max_logits, lse


def prewarm_tilelang_sparse_fwd(
    heads: int,
    d_qk: int,
    d_v: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    block_I: int = 16,
    threads: int = 128,
) -> None:
    """Trigger JIT compile + first-call allocation of the TileLang kernel
    before it enters the prefill hot path.

    Call this once from model-load to avoid a ~45s stall on the first
    prefill. Emits both sink-on and sink-off variants because the
    kernel is compiled per-feature-flag combination.
    """
    logger.info(
        "TileLang sparse MLA: prewarming kernels for "
        "heads=%d d_qk=%d d_v=%d topk=%d",
        heads, d_qk, d_v, topk,
    )
    sm_scale = d_qk ** -0.5

    # Tiny fake inputs (s_q=1) - prewarm JIT compile cost only.
    Q = torch.zeros(1, heads, d_qk, dtype=dtype, device=device)
    KV = torch.zeros(1, 1, d_qk, dtype=dtype, device=device)
    Indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
    attn_sink = torch.zeros(heads, dtype=torch.float32, device=device)
    topk_length = torch.ones(1, dtype=torch.int32, device=device)

    # Compile all 4 variants we may encounter in production: with/without sink,
    # with/without topk_length.
    for sink, tl in [(None, None), (attn_sink, None),
                     (None, topk_length), (attn_sink, topk_length)]:
        flash_mla_sparse_fwd_tilelang(
            Q, KV, Indices, sm_scale, d_v,
            attn_sink=sink, topk_length=tl,
            block_I=block_I, threads=threads,
        )
    torch.cuda.synchronize()
    logger.info("TileLang sparse MLA: prewarm complete")
