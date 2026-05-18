# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental TileLang prefill top-k kernel for DSv4F sparse indexer.

This module intentionally owns the TileLang call boundary even while the first
kernel is simple.  The Python interface should remain stable as the kernel
body evolves from the current correctness-first implementation into a blocked
filtered top-k pipeline.
"""

from typing import Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TILELANG_AVAILABLE: Optional[bool] = None
_KERNEL_FACTORY = None
_KERNEL_CACHE: dict[tuple[int, int], object] = {}
_FILL_FACTORY = None
_FILL_CACHE: dict[tuple[int, int], object] = {}


def is_tilelang_available() -> Tuple[bool, Optional[str]]:
    """Check whether TileLang can be imported in this environment."""
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


def _build_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_prefill_topk_kernel(topk: int, threads: int = 256):
        rows = T.dynamic("rows")
        cols = T.dynamic("cols")
        RADIX = 256

        @T.prim_func
        def main(
            Logits: T.Tensor((rows, cols), T.float32),
            RowStarts: T.Tensor((rows,), T.int32),
            Lengths: T.Tensor((rows,), T.int32),
            Output: T.Tensor((rows, topk), T.int32),
        ):
            with T.Kernel(rows, threads=threads) as row:
                hist = T.alloc_shared((RADIX,), T.int32)
                threshold_bin = T.alloc_shared((1,), T.int32)
                greater_count = T.alloc_shared((1,), T.int32)
                remaining = T.alloc_shared((1,), T.int32)
                output_count = T.alloc_shared((1,), T.int32)
                tb0 = T.alloc_shared((1,), T.int32)
                tb1 = T.alloc_shared((1,), T.int32)
                tb2 = T.alloc_shared((1,), T.int32)
                tb3 = T.alloc_shared((1,), T.int32)
                tx = T.get_thread_binding()
                start = RowStarts[row]
                length = Lengths[row]
                length_rounded = T.ceildiv(length, threads) * threads

                if length <= topk:
                    for k in T.Parallel(topk):
                        Output[row, k] = T.if_then_else(k < length, k, -1)
                else:
                    if tx == 0:
                        remaining[0] = topk
                        output_count[0] = 0
                    T.sync_threads()

                    # Pass 0: bits [31:24].
                    for b in T.Parallel(RADIX):
                        hist[b] = 0
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            bin_idx = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            T.atomic_add(hist[bin_idx], 1)
                    T.sync_threads()
                    if tx == 0:
                        seen = T.alloc_fragment((1,), T.int32)
                        found = T.alloc_fragment((1,), T.int32)
                        seen[0] = 0
                        found[0] = 0
                        for i in T.serial(RADIX):
                            b = RADIX - 1 - i
                            cnt = hist[b]
                            if (found[0] == 0) & ((seen[0] + cnt) >= remaining[0]):
                                threshold_bin[0] = b
                                greater_count[0] = seen[0]
                                found[0] = 1
                            seen[0] += cnt
                        tb0[0] = threshold_bin[0]
                        remaining[0] -= greater_count[0]
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            bin_idx = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            if bin_idx > tb0[0]:
                                out_pos = T.atomic_add(
                                    output_count[0], 1, return_prev=True
                                )
                                if out_pos < topk:
                                    Output[row, out_pos] = pos
                    T.sync_threads()

                    # Pass 1: bits [23:16] among pass-0 threshold bin.
                    for b in T.Parallel(RADIX):
                        hist[b] = 0
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            hi0 = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            if hi0 == tb0[0]:
                                bin_idx = ((key >> T.uint32(16)) & T.uint32(0xFF)).astype(T.int32)
                                T.atomic_add(hist[bin_idx], 1)
                    T.sync_threads()
                    if tx == 0:
                        seen = T.alloc_fragment((1,), T.int32)
                        found = T.alloc_fragment((1,), T.int32)
                        seen[0] = 0
                        found[0] = 0
                        for i in T.serial(RADIX):
                            b = RADIX - 1 - i
                            cnt = hist[b]
                            if (found[0] == 0) & ((seen[0] + cnt) >= remaining[0]):
                                threshold_bin[0] = b
                                greater_count[0] = seen[0]
                                found[0] = 1
                            seen[0] += cnt
                        tb1[0] = threshold_bin[0]
                        remaining[0] -= greater_count[0]
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            hi0 = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            bin_idx = ((key >> T.uint32(16)) & T.uint32(0xFF)).astype(T.int32)
                            if (hi0 == tb0[0]) & (bin_idx > tb1[0]):
                                out_pos = T.atomic_add(
                                    output_count[0], 1, return_prev=True
                                )
                                if out_pos < topk:
                                    Output[row, out_pos] = pos
                    T.sync_threads()

                    # Pass 2: bits [15:8].
                    for b in T.Parallel(RADIX):
                        hist[b] = 0
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            hi0 = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            hi1 = ((key >> T.uint32(16)) & T.uint32(0xFF)).astype(T.int32)
                            if (hi0 == tb0[0]) & (hi1 == tb1[0]):
                                bin_idx = ((key >> T.uint32(8)) & T.uint32(0xFF)).astype(T.int32)
                                T.atomic_add(hist[bin_idx], 1)
                    T.sync_threads()
                    if tx == 0:
                        seen = T.alloc_fragment((1,), T.int32)
                        found = T.alloc_fragment((1,), T.int32)
                        seen[0] = 0
                        found[0] = 0
                        for i in T.serial(RADIX):
                            b = RADIX - 1 - i
                            cnt = hist[b]
                            if (found[0] == 0) & ((seen[0] + cnt) >= remaining[0]):
                                threshold_bin[0] = b
                                greater_count[0] = seen[0]
                                found[0] = 1
                            seen[0] += cnt
                        tb2[0] = threshold_bin[0]
                        remaining[0] -= greater_count[0]
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            hi0 = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            hi1 = ((key >> T.uint32(16)) & T.uint32(0xFF)).astype(T.int32)
                            bin_idx = ((key >> T.uint32(8)) & T.uint32(0xFF)).astype(T.int32)
                            if (hi0 == tb0[0]) & (hi1 == tb1[0]) & (bin_idx > tb2[0]):
                                out_pos = T.atomic_add(
                                    output_count[0], 1, return_prev=True
                                )
                                if out_pos < topk:
                                    Output[row, out_pos] = pos
                    T.sync_threads()

                    # Pass 3: bits [7:0]. Fill threshold equals until topk.
                    for b in T.Parallel(RADIX):
                        hist[b] = 0
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            hi0 = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            hi1 = ((key >> T.uint32(16)) & T.uint32(0xFF)).astype(T.int32)
                            hi2 = ((key >> T.uint32(8)) & T.uint32(0xFF)).astype(T.int32)
                            if (hi0 == tb0[0]) & (hi1 == tb1[0]) & (hi2 == tb2[0]):
                                bin_idx = (key & T.uint32(0xFF)).astype(T.int32)
                                T.atomic_add(hist[bin_idx], 1)
                    T.sync_threads()
                    if tx == 0:
                        seen = T.alloc_fragment((1,), T.int32)
                        found = T.alloc_fragment((1,), T.int32)
                        seen[0] = 0
                        found[0] = 0
                        for i in T.serial(RADIX):
                            b = RADIX - 1 - i
                            cnt = hist[b]
                            if (found[0] == 0) & ((seen[0] + cnt) >= remaining[0]):
                                threshold_bin[0] = b
                                greater_count[0] = seen[0]
                                found[0] = 1
                            seen[0] += cnt
                        tb3[0] = threshold_bin[0]
                    T.sync_threads()
                    for pos in T.serial(tx, length_rounded, threads):
                        if pos < length:
                            v = Logits[row, start + pos]
                            bits = T.reinterpret(T.uint32, v)
                            mask = T.if_then_else(
                                (bits & T.uint32(0x80000000)) != T.uint32(0),
                                T.uint32(0xFFFFFFFF),
                                T.uint32(0x80000000),
                            )
                            key = bits ^ mask
                            hi0 = ((key >> T.uint32(24)) & T.uint32(0xFF)).astype(T.int32)
                            hi1 = ((key >> T.uint32(16)) & T.uint32(0xFF)).astype(T.int32)
                            hi2 = ((key >> T.uint32(8)) & T.uint32(0xFF)).astype(T.int32)
                            bin_idx = (key & T.uint32(0xFF)).astype(T.int32)
                            if (
                                (hi0 == tb0[0])
                                & (hi1 == tb1[0])
                                & (hi2 == tb2[0])
                                & (bin_idx >= tb3[0])
                            ):
                                out_pos = T.atomic_add(
                                    output_count[0], 1, return_prev=True
                                )
                                if out_pos < topk:
                                    Output[row, out_pos] = pos

        return main

    return build_prefill_topk_kernel


def _build_fill_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_fill_short_rows_kernel(topk: int, threads: int = 256):
        rows = T.dynamic("rows")

        @T.prim_func
        def main(
            Lengths: T.Tensor((rows,), T.int32),
            Output: T.Tensor((rows, topk), T.int32),
        ):
            with T.Kernel(rows, threads=threads) as row:
                length = Lengths[row]
                for k in T.Parallel(topk):
                    Output[row, k] = T.if_then_else(k < length, k, -1)

        return main

    return build_fill_short_rows_kernel


def _get_kernel(topk: int, threads: int):
    global _KERNEL_FACTORY
    if _KERNEL_FACTORY is None:
        _KERNEL_FACTORY = _build_kernel_factory()
    key = (topk, threads)
    if key not in _KERNEL_CACHE:
        logger.info(
            "TileLang prefill top-k: JIT compiling topk=%d threads=%d",
            topk,
            threads,
        )
        _KERNEL_CACHE[key] = _KERNEL_FACTORY(topk=topk, threads=threads)
        logger.info("TileLang prefill top-k: compile complete")
    return _KERNEL_CACHE[key]


def _get_fill_kernel(topk: int, threads: int):
    global _FILL_FACTORY
    if _FILL_FACTORY is None:
        _FILL_FACTORY = _build_fill_kernel_factory()
    key = (topk, threads)
    if key not in _FILL_CACHE:
        logger.info(
            "TileLang prefill top-k short-fill: JIT compiling topk=%d threads=%d",
            topk,
            threads,
        )
        _FILL_CACHE[key] = _FILL_FACTORY(topk=topk, threads=threads)
        logger.info("TileLang prefill top-k short-fill: compile complete")
    return _FILL_CACHE[key]


def is_tilelang_prefill_topk_cached(topk: int, threads: int = 256) -> bool:
    return (topk, threads) in _KERNEL_CACHE


def _prefill_topk_legacy(
    logits: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    topk_tokens: int,
) -> None:
    try:
        torch.ops._C.top_k_per_row_prefill
    except AttributeError:  # pragma: no cover - depends on import order.
        import vllm._C  # noqa: F401

    row_ends = (row_starts + lengths).contiguous()
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )


def prefill_topk_tilelang(
    logits: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    *,
    topk_tokens: int,
    threads: int = 256,
    causal_row_offset: Optional[int] = None,
) -> None:
    """Fill ``indices`` with local top-k positions for each prefill row."""
    ok, reason = is_tilelang_available()
    if not ok:
        raise RuntimeError(reason)
    if topk_tokens != indices.shape[1]:
        raise ValueError(
            f"topk_tokens={topk_tokens} must match indices.shape[1]={indices.shape[1]}"
        )
    if logits.dtype != torch.float32 or logits.ndim != 2 or logits.stride(1) != 1:
        raise ValueError("logits must be a 2D row-contiguous float32 CUDA tensor")
    if indices.dtype != torch.int32 or indices.ndim != 2 or not indices.is_contiguous():
        raise ValueError("indices must be a 2D contiguous int32 CUDA tensor")
    if lengths.dtype != torch.int32 or lengths.ndim != 1 or not lengths.is_contiguous():
        raise ValueError("lengths must be a 1D contiguous int32 CUDA tensor")
    if row_starts.dtype != torch.int32 or row_starts.ndim != 1:
        raise ValueError("row_starts must be a 1D int32 CUDA tensor")
    if logits.shape[0] != indices.shape[0] or lengths.shape[0] != logits.shape[0]:
        raise ValueError("batch dimensions must match")

    lengths_i32 = lengths.contiguous()
    row_starts_i32 = row_starts.contiguous()

    tail_start = 0
    if causal_row_offset is not None:
        short_prefix_rows = max(
            0, min(logits.shape[0], topk_tokens - int(causal_row_offset))
        )
        if short_prefix_rows > 0:
            fill_kernel = _get_fill_kernel(topk_tokens, threads)
            fill_kernel(
                lengths_i32[:short_prefix_rows],
                indices[:short_prefix_rows],
            )
        # TileLang's sync-heavy radix body is only safe and useful once all
        # rows are comfortably longer than top-k.  The causal prefix just above
        # top-k is small, and the existing C++ kernel handles it quickly.
        legacy_prefix_rows = max(
            0, min(logits.shape[0], 1024 - int(causal_row_offset))
        )
        if legacy_prefix_rows > short_prefix_rows:
            _prefill_topk_legacy(
                logits[short_prefix_rows:legacy_prefix_rows],
                indices[short_prefix_rows:legacy_prefix_rows],
                lengths_i32[short_prefix_rows:legacy_prefix_rows],
                row_starts_i32[short_prefix_rows:legacy_prefix_rows],
                topk_tokens,
            )
        tail_start = legacy_prefix_rows

    if tail_start >= logits.shape[0]:
        return

    kernel = _get_kernel(topk_tokens, threads)
    kernel(
        logits[tail_start:],
        row_starts_i32[tail_start:],
        lengths_i32[tail_start:],
        indices[tail_start:],
    )
