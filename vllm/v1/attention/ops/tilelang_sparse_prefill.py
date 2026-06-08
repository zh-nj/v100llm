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

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_TILELANG_AVAILABLE: Optional[bool] = None
_SM70_STAGED_GEMM_REGION_PATCHED = False
_PV_GEMM_POLICIES = ("full_row", "full_col", "square")


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _output_chunk_rows(
    *,
    s_q: int,
    h_q: int,
    d_v: int,
    output_dtype: torch.dtype,
) -> int:
    chunk_mb = envs.VLLM_SM70_TILELANG_SPARSE_PREFILL_OUTPUT_CHUNK_MB
    if chunk_mb <= 0:
        return s_q
    bytes_per_row = h_q * d_v * _dtype_element_size(output_dtype)
    if bytes_per_row <= 0:
        return s_q
    budget_bytes = chunk_mb * 1024 * 1024
    return max(1, min(s_q, budget_bytes // bytes_per_row))


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


def _validate_heads_per_block(heads: int, heads_per_block: int) -> None:
    if heads_per_block not in (16, 32, 64):
        raise ValueError(
            "heads_per_block must be one of (16, 32, 64), "
            f"got {heads_per_block}"
        )
    if heads > heads_per_block and heads % heads_per_block != 0:
        raise ValueError(
            "heads_per_block must divide the query head count when it "
            f"splits heads, got heads={heads} "
            f"heads_per_block={heads_per_block}"
        )


def _validate_kernel_launch_config(
    heads: int,
    heads_per_block: int,
    threads: int,
) -> None:
    _validate_heads_per_block(heads, heads_per_block)
    # On SM70, the Volta MMA macro selected by TileLang's FullRow policy
    # requires a smaller thread partition when the head tile is split to 32.
    # With 128 threads, hpb=32 lowers to warp_col_tiles=8 and fails before
    # codegen. Keep this explicit so users do not get a late JIT traceback.
    if heads_per_block == 32 and threads != 64:
        raise ValueError(
            "heads_per_block=32 requires threads=64 for the SM70 "
            f"TileLang sparse prefill kernel, got threads={threads}"
        )


def _normalize_pv_gemm_policy(policy: str) -> str:
    normalized = policy.strip().lower().replace("-", "_")
    if normalized not in _PV_GEMM_POLICIES:
        raise ValueError(
            "pv_gemm_policy must be one of "
            f"{_PV_GEMM_POLICIES}, got {policy!r}"
        )
    return normalized


def _patch_tilelang_sm70_staged_gemm_region() -> None:
    """Teach TileLang's SM70 GEMM macro to index staged BufferRegions.

    TileLang's software pipeline pass multi-versions producer shared buffers by
    inserting a leading stage dimension, e.g. `[BI, D] -> [2, BI, D]`. The
    RegionOp passed to `T.gemm` correctly carries a unit leading extent, but
    the Volta `ldmatrix_a/b` Python macro indexes the raw buffer with only two
    coordinates. That makes `num_stages=2` fail in LowerTileOp with:

      Buffer KV_shared is 3-dimensional, cannot be indexed with 2 dimensions.

    The patch preserves the existing 2D behavior and additionally prepends any
    leading region coordinates when the shared buffer has been multi-versioned.
    """
    global _SM70_STAGED_GEMM_REGION_PATCHED
    if _SM70_STAGED_GEMM_REGION_PATCHED:
        return

    import tilelang.language as T
    from tilelang.intrinsics import mma_sm70_macro_generator as sm70_mma

    emitter_cls = sm70_mma.TensorCoreIntrinEmitter
    if getattr(emitter_cls, "_vllm_sm70_staged_region_patch", False):
        _SM70_STAGED_GEMM_REGION_PATCHED = True
        return

    def buffer_load_with_region(buffer, region, row, col):
        coords = [rng.min for rng in region.region[:-2]]
        coords.append(region.region[-2].min + row)
        coords.append(region.region[-1].min + col)
        return buffer[coords]

    def patched_ldmatrix_a(self, A_local_buf, A_shared_buf, ki, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        thread_binding = self.get_thread_binding()

        assert not self.a_transposed, "A must be not transposed"

        mma_load_layout = sm70_mma.mma_load_a_32x4_to_shared_16x4_layout
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)

            for i in T.serial(warp_rows):
                wi = warp_m * warp_row_tiles + i * micro_size_x
                wk = rk * chunk + ki * micro_size_k
                for j in T.vectorized(local_size_a):
                    mi, mk = mma_load_layout(tx, j)
                    A_local_buf[i * local_size_a + j] = (
                        buffer_load_with_region(
                            A_buf, A_region, wi + mi, wk + mk)
                    )

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def patched_ldmatrix_b(self, B_local_buf, B_shared_buf, ki, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()

        mma_load_layout = (
            sm70_mma.mma_load_b_32x4_to_shared_16x4_layout_trans
            if b_transposed else sm70_mma.mma_load_b_32x4_to_shared_4x16_layout
        )
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)

            for i in T.serial(warp_cols):
                wi = warp_n * warp_col_tiles + i * micro_size_y
                wk = rk * chunk + ki * micro_size_k
                for j in T.vectorized(local_size_b):
                    if b_transposed:
                        mi, mk = mma_load_layout(tx, j)
                        B_local_buf[i * local_size_b + j] = (
                            buffer_load_with_region(
                                B_buf, B_region, wi + mi, wk + mk)
                        )
                    else:
                        mk, mi = mma_load_layout(tx, j)
                        B_local_buf[i * local_size_b + j] = (
                            buffer_load_with_region(
                                B_buf, B_region, wk + mk, wi + mi)
                        )

        return _warp_ldmatrix_b(B_local_buf, B_region, ki, thread_binding, rk)

    emitter_cls._vllm_original_ldmatrix_a = emitter_cls.ldmatrix_a
    emitter_cls._vllm_original_ldmatrix_b = emitter_cls.ldmatrix_b
    emitter_cls.ldmatrix_a = patched_ldmatrix_a
    emitter_cls.ldmatrix_b = patched_ldmatrix_b
    emitter_cls._vllm_sm70_staged_region_patch = True
    _SM70_STAGED_GEMM_REGION_PATCHED = True


# Build the kernel lazily — tilelang decorators must run at import
# time of the inner `build_sparse_mla_fwd_kernel`, but we want to
# defer tilelang import until the caller opts in.


def _build_kernel_factory():
    import tilelang
    from tilelang import language as T

    _patch_tilelang_sm70_staged_gemm_region()

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
        heads_per_block: int = 64,
        threads: int = 128,
        pv_gemm_policy: str = "full_row",
        assume_valid_indices: bool = False,
        has_sink: bool = True,
        has_topk_length: bool = True,
        output_dtype_str: str = "float16",
        q_dchunk: int = 0,
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
        assert heads_per_block in (16, 32, 64)
        assert pv_gemm_policy in _PV_GEMM_POLICIES
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

        # Q D-tiling: when q_dchunk in (0, D), stage only [H, DC] of Q at a
        # time and loop the QK GEMM over D-chunks. Shrinks Q_shared SMEM so
        # larger BI / threads fit on V100. DC must divide D.
        DC = q_dchunk if (0 < q_dchunk < D) else D
        assert D % DC == 0, f"q_dchunk={q_dchunk} must divide dim={D}"
        N_DCHUNK = D // DC
        Q_TILE_COLS = DC

        if head_kv > heads_per_block:
            assert head_kv % heads_per_block == 0
            REPLICATE_H = head_kv // heads_per_block
        else:
            REPLICATE_H = 1
        H_per_block = padded_H if REPLICATE_H == 1 else heads_per_block

        if pv_gemm_policy == "full_col":
            PV_GEMM_POLICY = T.GemmWarpPolicy.FullCol
        elif pv_gemm_policy == "square":
            PV_GEMM_POLICY = T.GemmWarpPolicy.Square
        else:
            PV_GEMM_POLICY = T.GemmWarpPolicy.FullRow

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
                Q_shared = T.alloc_shared([H_per_block, Q_TILE_COLS], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                # When Q is D-tiled, the QK GEMM consumes a [BI, DC] slice of
                # KV per chunk; copy it into a dedicated exact-shape buffer
                # because the SM70 ldmatrix macro rejects sliced GEMM operands.
                if DC != D:
                    KV_qk_chunk = T.alloc_shared([BI, DC], dtype)
                if has_tail:
                    Q_tail_shared = T.alloc_shared(
                        [H_per_block, D_tail], dtype)
                    K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                if not assume_valid_indices:
                    mask = T.alloc_shared([BI], topk_len_dtype)

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                if pv_gemm_policy == "full_row":
                    sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                    alpha = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                else:
                    sumexp = T.alloc_shared([H_per_block], accum_dtype)
                    sumexp_i = T.alloc_shared([H_per_block], accum_dtype)
                    alpha = T.alloc_shared([H_per_block], accum_dtype)
                    m_i = T.alloc_shared([H_per_block], accum_dtype)
                    m_i_prev = T.alloc_shared([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2 ** 30))

                b_i, g_i = by, bz
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

                H0 = g_i * padded_H + (
                    0 if REPLICATE_H == 1
                    else (bx % REPLICATE_H) * heads_per_block
                )
                H1 = H0 + H_per_block

                if has_tail:
                    T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
                if DC == D:
                    T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)

                if not assume_valid_indices:
                    tl_cur = T.alloc_fragment([1], topk_len_dtype)
                    if has_topk_length:
                        tl_cur[0] = TopkLen[b_i, s_i]
                    else:
                        tl_cur[0] = topk

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    if assume_valid_indices:
                        for bi_i, d_i in T.Parallel(BI, D):
                            idx_raw = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                            KV_shared[bi_i, d_i] = KV[
                                b_i, idx_raw, g_i, d_i
                            ]
                        if has_tail:
                            for bi_i, d_i in T.Parallel(BI, D_tail):
                                idx_raw = Indices[
                                    b_i, s_i, g_i, i_i * BI + bi_i
                                ]
                                K_tail_shared[bi_i, d_i] = KV[
                                    b_i, idx_raw, g_i, D + d_i
                                ]
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = 0
                    else:
                        for bi_i in T.Parallel(BI):
                            pos = i_i * BI + bi_i
                            idx = Indices[b_i, s_i, g_i, pos]
                            valid = ((idx >= 0) & (idx < seq_len_kv)
                                     & (pos < tl_cur[0]))
                            mask[bi_i] = T.if_then_else(valid, 1, 0)

                        for bi_i, d_i in T.Parallel(BI, D):
                            idx_raw = Indices[
                                b_i, s_i, g_i, i_i * BI + bi_i
                            ]
                            idx_clamped = T.if_then_else(
                                mask[bi_i] != 0, idx_raw, 0)
                            KV_shared[bi_i, d_i] = KV[
                                b_i, idx_clamped, g_i, d_i
                            ]
                        if has_tail:
                            for bi_i, d_i in T.Parallel(BI, D_tail):
                                idx_raw = Indices[
                                    b_i, s_i, g_i, i_i * BI + bi_i
                                ]
                                idx_clamped = T.if_then_else(
                                    mask[bi_i] != 0, idx_raw, 0)
                                K_tail_shared[bi_i, d_i] = KV[
                                    b_i, idx_clamped, g_i, D + d_i
                                ]

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i] != 0, 0, -T.infinity(acc_s.dtype))
                    if DC == D:
                        T.gemm(
                            Q_shared, KV_shared, acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    else:
                        # D-tile QK: re-stage each Q D-chunk and the matching
                        # KV D-slice, accumulate into acc_s. KV_shared stays
                        # full-D for the PV GEMM below.
                        for dc in T.serial(N_DCHUNK):
                            T.copy(
                                Q[b_i, s_i, H0:H1, dc * DC:(dc + 1) * DC],
                                Q_shared)
                            for bi_i, d_i in T.Parallel(BI, DC):
                                KV_qk_chunk[bi_i, d_i] = KV_shared[
                                    bi_i, dc * DC + d_i]
                            T.gemm(
                                Q_shared, KV_qk_chunk, acc_s,
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
                        policy=PV_GEMM_POLICY,
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
    heads_per_block: int,
    threads: int,
    pv_gemm_policy: str = "full_row",
    assume_valid_indices: bool = False,
    output_dtype_str: str = "float16",
    q_dchunk: int = 0,
):
    _validate_kernel_launch_config(heads, heads_per_block, threads)
    pv_gemm_policy = _normalize_pv_gemm_policy(pv_gemm_policy)
    global _KERNEL_FACTORY
    if _KERNEL_FACTORY is None:
        _KERNEL_FACTORY = _build_kernel_factory()
    key = (heads, dim, tail_dim, topk, sm_scale, has_sink,
           has_topk_length, block_I, num_stages, heads_per_block, threads,
           pv_gemm_policy, assume_valid_indices, output_dtype_str, q_dchunk)
    if key not in _KERNEL_CACHE:
        logger.info(
            "TileLang sparse MLA: JIT compiling for "
            "heads=%d dim=%d tail=%d topk=%d sink=%s topk_len=%s "
            "out=%s (BI=%d stages=%d hpb=%d threads=%d pv=%s valid=%s dc=%d) — takes ~45s",
            heads, dim, tail_dim, topk, has_sink, has_topk_length,
            output_dtype_str, block_I, num_stages, heads_per_block, threads,
            pv_gemm_policy, assume_valid_indices, q_dchunk,
        )
        _KERNEL_CACHE[key] = _KERNEL_FACTORY(
            heads=heads, dim=dim, tail_dim=tail_dim, topk=topk,
            sm_scale=sm_scale, block_I=block_I, num_stages=num_stages,
            heads_per_block=heads_per_block, threads=threads,
            pv_gemm_policy=pv_gemm_policy,
            assume_valid_indices=assume_valid_indices,
            has_sink=has_sink,
            has_topk_length=has_topk_length,
            output_dtype_str=output_dtype_str,
            q_dchunk=q_dchunk,
        )
        logger.info("TileLang sparse MLA: compile complete")
    return _KERNEL_CACHE[key]


def _is_kernel_cached(
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    sm_scale: float,
    has_sink: bool,
    has_topk_length: bool,
    block_I: int,
    num_stages: int,
    heads_per_block: int,
    threads: int,
    pv_gemm_policy: str = "full_row",
    assume_valid_indices: bool = False,
    output_dtype_str: str = "float16",
    q_dchunk: int = 0,
) -> bool:
    """Return True if a compiled kernel for this config is in the cache.

    Used by the dispatcher to detect cache misses without triggering
    an expensive JIT compile inside a CUDA-graph-captured frame.
    """
    try:
        _validate_kernel_launch_config(heads, heads_per_block, threads)
        pv_gemm_policy = _normalize_pv_gemm_policy(pv_gemm_policy)
    except ValueError:
        return False
    key = (heads, dim, tail_dim, topk, sm_scale, has_sink,
           has_topk_length, block_I, num_stages, heads_per_block, threads,
           pv_gemm_policy, assume_valid_indices, output_dtype_str, q_dchunk)
    return key in _KERNEL_CACHE


def is_tilelang_sparse_fwd_cached(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    block_I: int = 16,
    num_stages: int = 1,
    heads_per_block: int = 64,
    threads: int = 128,
    pv_gemm_policy: str = "full_row",
    assume_valid_indices: bool = False,
    q_dchunk: int = 0,
) -> bool:
    """Public check: would `flash_mla_sparse_fwd_tilelang(q, kv, ...)` hit
    the kernel cache, or would it trigger a JIT compile?

    The dispatcher uses this to fall back to FlashMLA on cache-miss
    shapes instead of stalling the request for ~45 s (or failing
    inside a CUDA-graph-captured frame).
    """
    _s_q, h_q, d_qk = q.shape
    try:
        _validate_kernel_launch_config(h_q, heads_per_block, threads)
        pv_gemm_policy = _normalize_pv_gemm_policy(pv_gemm_policy)
    except ValueError:
        return False
    _s_q2, _h_kv, topk = indices.shape
    dim = d_v
    tail_dim = d_qk - d_v
    if tail_dim not in (0, 64):
        return False

    requested_output_dtype = (
        out.dtype if out is not None
        else output_dtype if output_dtype is not None
        else q.dtype
    )
    output_dtype_str = (
        "bfloat16" if requested_output_dtype == torch.bfloat16 else "float16"
    )
    has_sink = attn_sink is not None
    has_topk_length = topk_length is not None and not assume_valid_indices

    return _is_kernel_cached(
        heads=h_q, dim=dim, tail_dim=tail_dim, topk=topk,
        sm_scale=sm_scale, has_sink=has_sink,
        has_topk_length=has_topk_length,
        block_I=block_I, num_stages=num_stages,
        heads_per_block=heads_per_block, threads=threads,
        pv_gemm_policy=pv_gemm_policy,
        assume_valid_indices=assume_valid_indices,
        output_dtype_str=output_dtype_str,
        q_dchunk=q_dchunk,
    )


def flash_mla_sparse_fwd_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    block_I: int = 16,
    num_stages: int = 1,
    heads_per_block: int = 64,
    threads: int = 128,
    pv_gemm_policy: str = "full_row",
    assume_valid_indices: bool = False,
    q_dchunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for FlashMLA's `flash_mla_sparse_fwd`.

    See FlashMLA's flash_mla_interface.py for the full contract.
    """
    s_q, h_q, d_qk = q.shape
    _validate_kernel_launch_config(h_q, heads_per_block, threads)
    pv_gemm_policy = _normalize_pv_gemm_policy(pv_gemm_policy)
    s_kv, _h_kv, _d_qk2 = kv.shape
    _s_q2, _h_kv2, topk = indices.shape
    assert d_v == 512
    dim = d_v
    tail_dim = d_qk - d_v
    assert tail_dim in (0, 64), f"unsupported tail_dim {tail_dim}"

    input_dtype = q.dtype
    requested_output_dtype = (
        output_dtype if output_dtype is not None
        else out.dtype if out is not None
        else input_dtype
    )
    # Q/KV go in as fp16 (V100 MMA requirement). `output_dtype` controls the
    # TileLang temporary dtype when supplied; `out` is only the final destination
    # buffer. This lets DSv4F request bf16 kernel output while copying each small
    # chunk into the fp16 model output buffer, instead of materializing one large
    # bf16 prefill tensor.
    output_dtype_str = (
        "bfloat16" if requested_output_dtype == torch.bfloat16 else "float16"
    )
    rows_per_chunk = _output_chunk_rows(
        s_q=s_q, h_q=h_q, d_v=dim, output_dtype=requested_output_dtype
    )
    if out is not None and rows_per_chunk < s_q:
        max_logits_full = torch.empty(
            (s_q, h_q), dtype=torch.float32, device=q.device)
        lse_full = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
        for row_start in range(0, s_q, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, s_q)
            _out, max_part, lse_part = flash_mla_sparse_fwd_tilelang(
                q=q[row_start:row_end],
                kv=kv,
                indices=indices[row_start:row_end],
                sm_scale=sm_scale,
                d_v=d_v,
                attn_sink=attn_sink,
                topk_length=(
                    topk_length[row_start:row_end]
                    if topk_length is not None else None
                ),
                out=out[row_start:row_end],
                output_dtype=requested_output_dtype,
                block_I=block_I,
                num_stages=num_stages,
                heads_per_block=heads_per_block,
                threads=threads,
                pv_gemm_policy=pv_gemm_policy,
                assume_valid_indices=assume_valid_indices,
                q_dchunk=q_dchunk,
            )
            max_logits_full[row_start:row_end].copy_(max_part)
            lse_full[row_start:row_end].copy_(lse_part)
        return out, max_logits_full, lse_full

    q_fp16 = q.to(torch.float16) if q.dtype != torch.float16 else q
    kv_fp16 = kv.to(torch.float16) if kv.dtype != torch.float16 else kv

    Q_b = q_fp16.unsqueeze(0).contiguous()
    KV_b = kv_fp16.unsqueeze(0).contiguous()
    Indices_b = indices.unsqueeze(0).contiguous()

    has_sink = attn_sink is not None
    has_topk_length = topk_length is not None and not assume_valid_indices

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
        block_I=block_I, num_stages=num_stages,
        heads_per_block=heads_per_block, threads=threads,
        pv_gemm_policy=pv_gemm_policy,
        assume_valid_indices=assume_valid_indices,
        output_dtype_str=output_dtype_str,
        q_dchunk=q_dchunk,
    )

    out_tl, max_tl, lse_tl = kernel(
        Q_b, KV_b, Indices_b, Sink, TopkLen_b)

    output = out_tl.squeeze(0)
    max_logits = max_tl.squeeze(0)
    lse = lse_tl.squeeze(0)

    # If output_dtype_str="bfloat16", output is already bf16 from the
    # kernel; no cast needed. Only cast when no explicit out buffer was
    # supplied and the historical return dtype should match q.
    if (
        out is None
        and output_dtype is None
        and output_dtype_str == "float16"
        and input_dtype != torch.float16
    ):
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
    num_stages: int = 1,
    heads_per_block: int = 64,
    threads: int = 128,
    pv_gemm_policy: str = "full_row",
    assume_valid_indices: bool = False,
    all_feature_variants: bool = False,
    q_dchunk: int = 0,
) -> None:
    """Trigger JIT compile + first-call allocation of the TileLang kernel
    before it enters the prefill hot path.

    Call this once from model-load to avoid a ~45s stall on common prefill
    buckets. Production DSv4F prefill uses both attn_sink and topk_length, so
    compile that variant by default; pass all_feature_variants=True for
    exhaustive debugging coverage.
    """
    logger.info(
        "TileLang sparse MLA: prewarming kernels for "
        "heads=%d d_qk=%d d_v=%d topk=%d",
        heads, d_qk, d_v, topk,
    )
    sm_scale = d_qk ** -0.5
    _validate_kernel_launch_config(heads, heads_per_block, threads)
    pv_gemm_policy = _normalize_pv_gemm_policy(pv_gemm_policy)

    # Tiny fake inputs (s_q=1) - prewarm JIT compile cost only.
    Q = torch.zeros(1, heads, d_qk, dtype=dtype, device=device)
    KV = torch.zeros(1, 1, d_qk, dtype=dtype, device=device)
    Indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
    attn_sink = torch.zeros(heads, dtype=torch.float32, device=device)
    topk_length = torch.ones(1, dtype=torch.int32, device=device)

    variants = (
        [(None, None), (attn_sink, None), (None, topk_length),
         (attn_sink, topk_length)]
        if all_feature_variants else [(attn_sink, topk_length)]
    )
    for sink, tl in variants:
        flash_mla_sparse_fwd_tilelang(
            Q, KV, Indices, sm_scale, d_v,
            attn_sink=sink, topk_length=tl,
            block_I=block_I, num_stages=num_stages,
            heads_per_block=heads_per_block, threads=threads,
            pv_gemm_policy=pv_gemm_policy,
            assume_valid_indices=assume_valid_indices,
            q_dchunk=q_dchunk,
        )
    torch.cuda.synchronize()
    logger.info("TileLang sparse MLA: prewarm complete")
