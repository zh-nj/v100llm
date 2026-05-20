# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming prefill top-k wrapper for the DSv4F sparse indexer.

The public call boundary is intentionally shaped like the final implementation:
query rows + gathered K cache + row ranges -> local top-k indices.  FP8 tile
logits use the SM70 CUDA/Triton logits kernel; non-FP8 shapes retain the torch
reference path.  Tile-local top-k and cross-tile candidate merge use the
existing TileLang radix top-k kernel while the tile loop remains the next
fusion target.
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.sm70_mqa_logits import (
    sm70_fp8_mqa_block_candidate_update,
    sm70_fp8_mqa_logits,
    sm70_fp8_mqa_logits_gemm,
)
from vllm.v1.attention.ops.tilelang_prefill_topk import (
    is_tilelang_available,
    prewarm_prefill_topk_tilelang,
    prefill_topk_tilelang,
)


_FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz")
    if hasattr(torch, name)
)
_CANDIDATE_BUFFER_FACTORY = None
_CANDIDATE_BUFFER_CACHE: dict[tuple[int, int, int], object] = {}
_CANDIDATE_GATHER_FACTORY = None
_CANDIDATE_GATHER_CACHE: dict[tuple[int, int, int], object] = {}
_FUSED_CANDIDATE_UPDATE_FACTORY = None
_FUSED_CANDIDATE_UPDATE_CACHE: dict[tuple[int, int, int], object] = {}
_DENSE_TILE_OFFSETS_FACTORY = None
_DENSE_TILE_OFFSETS_CACHE: dict[tuple[int, int], object] = {}
_FINAL_INDICES_FACTORY = None
_FINAL_INDICES_CACHE: dict[tuple[int, int], object] = {}
_BLOCK_LOGITS_FACTORY = None
_BLOCK_LOGITS_CACHE: dict[tuple[int, int, int, int, int, int], object] = {}
_BLOCK_CANDIDATE_UPDATE_FACTORY = None
_BLOCK_CANDIDATE_UPDATE_CACHE: dict[
    tuple[int, int, int, int, int, int, int],
    object,
] = {}
_FP8_BLOCK_CANDIDATE_UPDATE_FACTORY = None
_FP8_BLOCK_CANDIDATE_UPDATE_CACHE: dict[
    tuple[int, int, int, int, int, int, int],
    object,
] = {}
# Keep the TileLang candidate path to shapes with a measured win.  The fused
# candidate-update kernel makes topk=512 faster than the chunked torch loop on
# the SM70 benchmark shape; larger top-k widths still need their own sweep.
_MAX_BENCHED_TILELANG_CANDIDATE_TOPK = 512


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors):
        raise ValueError("streaming topk inputs must be CUDA tensors")


def _is_supported_q_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16, *_FP8_DTYPES)


def _is_sm70_tensor_device(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    device = tensor.device
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return torch.cuda.get_device_capability(device) == (7, 0)


def _can_use_sm70_fp8_tile_logits(
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
) -> bool:
    return (
        q.is_cuda
        and k_cache_values.is_cuda
        and q.dtype in _FP8_DTYPES
        and k_cache_values.dtype in _FP8_DTYPES
        and q.ndim == 3
    )


def _can_use_sm70_fp8_fused_block_update(
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
) -> bool:
    # The first fused block-update kernel is one Triton program per query row
    # and loops over heads internally. It is a correctness stepping stone, not
    # the DSV4F production kernel; keep 64-head DSV4F on the measured path until
    # the tensorcore BLOCK_M x BLOCK_N fused GEMM kernel lands.
    return _can_use_sm70_fp8_tile_logits(q, k_cache_values) and q.shape[1] <= 8


def _can_use_tilelang_block_candidate_update(
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
) -> bool:
    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    return (
        q.is_cuda
        and k_cache_values.is_cuda
        and (
            (
                q.dtype is torch.float16
                and k_cache_values.dtype is torch.float16
            )
            or (
                fp8_e4m3fn is not None
                and q.dtype is fp8_e4m3fn
                and k_cache_values.dtype is fp8_e4m3fn
            )
        )
        and q.ndim == 3
    )


def _build_candidate_buffer_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_candidate_buffer_kernel(
        topk: int,
        tile_keep: int,
        threads: int = 256,
    ):
        rows = T.dynamic("rows")
        tile_len = T.dynamic("tile_len")
        candidate_width = topk + tile_keep

        @T.prim_func
        def main(
            BestScores: T.Tensor((rows, topk), T.float32),
            BestIndices: T.Tensor((rows, topk), T.int32),
            TileLogits: T.Tensor((rows, tile_len), T.float32),
            TileOffsets: T.Tensor((rows, tile_keep), T.int32),
            RowStarts: T.Tensor((rows,), T.int32),
            TileLocalStarts: T.Tensor((rows,), T.int32),
            TileAbsStarts: T.Tensor((rows,), T.int32),
            CandidateScores: T.Tensor((rows, candidate_width), T.float32),
            CandidateIndices: T.Tensor((rows, candidate_width), T.int32),
        ):
            with T.Kernel(rows, threads=threads) as row:
                for col in T.Parallel(candidate_width):
                    if col < topk:
                        CandidateScores[row, col] = BestScores[row, col]
                        CandidateIndices[row, col] = BestIndices[row, col]
                    else:
                        k = col - topk
                        offset = TileOffsets[row, k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        CandidateScores[row, col] = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            -T.infinity(T.float32),
                        )
                        CandidateIndices[row, col] = T.if_then_else(
                            valid,
                            TileAbsStarts[row] + offset - RowStarts[row],
                            -1,
                        )

        return main

    return build_candidate_buffer_kernel


def _build_candidate_gather_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_candidate_gather_kernel(
        topk: int,
        candidate_width: int,
        threads: int = 256,
    ):
        rows = T.dynamic("rows")
        neg_inf = -3.4028234663852886e38

        @T.prim_func
        def main(
            CandidateScores: T.Tensor((rows, candidate_width), T.float32),
            CandidateIndices: T.Tensor((rows, candidate_width), T.int32),
            MergePositions: T.Tensor((rows, topk), T.int32),
            NextScores: T.Tensor((rows, topk), T.float32),
            NextIndices: T.Tensor((rows, topk), T.int32),
        ):
            with T.Kernel(rows, threads=threads) as row:
                for col in T.Parallel(topk):
                    pos = MergePositions[row, col]
                    valid = pos >= 0
                    safe_pos = T.if_then_else(valid, pos, 0)
                    NextScores[row, col] = T.if_then_else(
                        valid,
                        CandidateScores[row, safe_pos],
                        neg_inf,
                    )
                    NextIndices[row, col] = T.if_then_else(
                        valid,
                        CandidateIndices[row, safe_pos],
                        -1,
                    )

        return main

    return build_candidate_gather_kernel


def _build_fused_candidate_update_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_fused_candidate_update_kernel(
        topk: int,
        tile_keep: int,
        threads: int = 256,
    ):
        rows = T.dynamic("rows")
        tile_len = T.dynamic("tile_len")
        candidate_width = topk + tile_keep
        RADIX = 256
        neg_inf = -3.4028234663852886e38

        @T.prim_func
        def main(
            BestScores: T.Tensor((rows, topk), T.float32),
            BestIndices: T.Tensor((rows, topk), T.int32),
            TileLogits: T.Tensor((rows, tile_len), T.float32),
            TileOffsets: T.Tensor((rows, tile_keep), T.int32),
            RowStarts: T.Tensor((rows,), T.int32),
            TileLocalStarts: T.Tensor((rows,), T.int32),
            TileAbsStarts: T.Tensor((rows,), T.int32),
            NextScores: T.Tensor((rows, topk), T.float32),
            NextIndices: T.Tensor((rows, topk), T.int32),
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
                rounded = T.ceildiv(candidate_width, threads) * threads

                if tx == 0:
                    remaining[0] = topk
                    output_count[0] = 0
                T.sync_threads()

                # Pass 0: bits [31:24].
                for b in T.Parallel(RADIX):
                    hist[b] = 0
                T.sync_threads()
                for pos in T.serial(tx, rounded, threads):
                    if pos < candidate_width:
                        is_best = pos < topk
                        best_pos = T.if_then_else(is_best, pos, 0)
                        tile_k = T.if_then_else(is_best, 0, pos - topk)
                        offset = TileOffsets[row, tile_k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        tile_v = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            neg_inf,
                        )
                        v = T.if_then_else(
                            is_best,
                            BestScores[row, best_pos],
                            tile_v,
                        )
                        bits = T.reinterpret(T.uint32, v)
                        mask = T.if_then_else(
                            (bits & T.uint32(0x80000000)) != T.uint32(0),
                            T.uint32(0xFFFFFFFF),
                            T.uint32(0x80000000),
                        )
                        key = bits ^ mask
                        bin_idx = (
                            (key >> T.uint32(24)) & T.uint32(0xFF)
                        ).astype(T.int32)
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

                # Pass 1: bits [23:16] among pass-0 threshold bin.
                for b in T.Parallel(RADIX):
                    hist[b] = 0
                T.sync_threads()
                for pos in T.serial(tx, rounded, threads):
                    if pos < candidate_width:
                        is_best = pos < topk
                        best_pos = T.if_then_else(is_best, pos, 0)
                        tile_k = T.if_then_else(is_best, 0, pos - topk)
                        offset = TileOffsets[row, tile_k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        tile_v = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            neg_inf,
                        )
                        v = T.if_then_else(
                            is_best,
                            BestScores[row, best_pos],
                            tile_v,
                        )
                        bits = T.reinterpret(T.uint32, v)
                        mask = T.if_then_else(
                            (bits & T.uint32(0x80000000)) != T.uint32(0),
                            T.uint32(0xFFFFFFFF),
                            T.uint32(0x80000000),
                        )
                        key = bits ^ mask
                        hi0 = (
                            (key >> T.uint32(24)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        if hi0 == tb0[0]:
                            bin_idx = (
                                (key >> T.uint32(16)) & T.uint32(0xFF)
                            ).astype(T.int32)
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

                # Pass 2: bits [15:8].
                for b in T.Parallel(RADIX):
                    hist[b] = 0
                T.sync_threads()
                for pos in T.serial(tx, rounded, threads):
                    if pos < candidate_width:
                        is_best = pos < topk
                        best_pos = T.if_then_else(is_best, pos, 0)
                        tile_k = T.if_then_else(is_best, 0, pos - topk)
                        offset = TileOffsets[row, tile_k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        tile_v = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            neg_inf,
                        )
                        v = T.if_then_else(
                            is_best,
                            BestScores[row, best_pos],
                            tile_v,
                        )
                        bits = T.reinterpret(T.uint32, v)
                        mask = T.if_then_else(
                            (bits & T.uint32(0x80000000)) != T.uint32(0),
                            T.uint32(0xFFFFFFFF),
                            T.uint32(0x80000000),
                        )
                        key = bits ^ mask
                        hi0 = (
                            (key >> T.uint32(24)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi1 = (
                            (key >> T.uint32(16)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        if (hi0 == tb0[0]) & (hi1 == tb1[0]):
                            bin_idx = (
                                (key >> T.uint32(8)) & T.uint32(0xFF)
                            ).astype(T.int32)
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

                # Pass 3: bits [7:0].
                for b in T.Parallel(RADIX):
                    hist[b] = 0
                T.sync_threads()
                for pos in T.serial(tx, rounded, threads):
                    if pos < candidate_width:
                        is_best = pos < topk
                        best_pos = T.if_then_else(is_best, pos, 0)
                        tile_k = T.if_then_else(is_best, 0, pos - topk)
                        offset = TileOffsets[row, tile_k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        tile_v = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            neg_inf,
                        )
                        v = T.if_then_else(
                            is_best,
                            BestScores[row, best_pos],
                            tile_v,
                        )
                        bits = T.reinterpret(T.uint32, v)
                        mask = T.if_then_else(
                            (bits & T.uint32(0x80000000)) != T.uint32(0),
                            T.uint32(0xFFFFFFFF),
                            T.uint32(0x80000000),
                        )
                        key = bits ^ mask
                        hi0 = (
                            (key >> T.uint32(24)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi1 = (
                            (key >> T.uint32(16)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi2 = (
                            (key >> T.uint32(8)) & T.uint32(0xFF)
                        ).astype(T.int32)
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
                            found[0] = 1
                        seen[0] += cnt
                    tb3[0] = threshold_bin[0]
                T.sync_threads()

                # Emit all candidates strictly above the threshold key.
                for pos in T.serial(tx, rounded, threads):
                    if pos < candidate_width:
                        is_best = pos < topk
                        best_pos = T.if_then_else(is_best, pos, 0)
                        tile_k = T.if_then_else(is_best, 0, pos - topk)
                        offset = TileOffsets[row, tile_k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        tile_v = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            neg_inf,
                        )
                        tile_index = T.if_then_else(
                            valid,
                            TileAbsStarts[row] + offset - RowStarts[row],
                            -1,
                        )
                        v = T.if_then_else(
                            is_best,
                            BestScores[row, best_pos],
                            tile_v,
                        )
                        out_index = T.if_then_else(
                            is_best,
                            BestIndices[row, best_pos],
                            tile_index,
                        )
                        bits = T.reinterpret(T.uint32, v)
                        mask = T.if_then_else(
                            (bits & T.uint32(0x80000000)) != T.uint32(0),
                            T.uint32(0xFFFFFFFF),
                            T.uint32(0x80000000),
                        )
                        key = bits ^ mask
                        hi0 = (
                            (key >> T.uint32(24)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi1 = (
                            (key >> T.uint32(16)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi2 = (
                            (key >> T.uint32(8)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi3 = (key & T.uint32(0xFF)).astype(T.int32)
                        is_greater = (hi0 > tb0[0]) | (
                            (hi0 == tb0[0])
                            & (
                                (hi1 > tb1[0])
                                | (
                                    (hi1 == tb1[0])
                                    & (
                                        (hi2 > tb2[0])
                                        | ((hi2 == tb2[0]) & (hi3 > tb3[0]))
                                    )
                                )
                            )
                        )
                        if is_greater:
                            out_pos = T.atomic_add(
                                output_count[0], 1, return_prev=True
                            )
                            if out_pos < topk:
                                NextScores[row, out_pos] = v
                                NextIndices[row, out_pos] = out_index
                T.sync_threads()

                # Fill threshold-equal candidates until top-k is complete.
                for pos in T.serial(tx, rounded, threads):
                    if pos < candidate_width:
                        is_best = pos < topk
                        best_pos = T.if_then_else(is_best, pos, 0)
                        tile_k = T.if_then_else(is_best, 0, pos - topk)
                        offset = TileOffsets[row, tile_k]
                        valid = offset >= 0
                        tile_col = TileLocalStarts[row] + offset
                        safe_tile_col = T.if_then_else(valid, tile_col, 0)
                        tile_v = T.if_then_else(
                            valid,
                            TileLogits[row, safe_tile_col],
                            neg_inf,
                        )
                        tile_index = T.if_then_else(
                            valid,
                            TileAbsStarts[row] + offset - RowStarts[row],
                            -1,
                        )
                        v = T.if_then_else(
                            is_best,
                            BestScores[row, best_pos],
                            tile_v,
                        )
                        out_index = T.if_then_else(
                            is_best,
                            BestIndices[row, best_pos],
                            tile_index,
                        )
                        bits = T.reinterpret(T.uint32, v)
                        mask = T.if_then_else(
                            (bits & T.uint32(0x80000000)) != T.uint32(0),
                            T.uint32(0xFFFFFFFF),
                            T.uint32(0x80000000),
                        )
                        key = bits ^ mask
                        hi0 = (
                            (key >> T.uint32(24)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi1 = (
                            (key >> T.uint32(16)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi2 = (
                            (key >> T.uint32(8)) & T.uint32(0xFF)
                        ).astype(T.int32)
                        hi3 = (key & T.uint32(0xFF)).astype(T.int32)
                        if (
                            (hi0 == tb0[0])
                            & (hi1 == tb1[0])
                            & (hi2 == tb2[0])
                            & (hi3 == tb3[0])
                        ):
                            out_pos = T.atomic_add(
                                output_count[0], 1, return_prev=True
                            )
                            if out_pos < topk:
                                NextScores[row, out_pos] = v
                                NextIndices[row, out_pos] = out_index

        return main

    return build_fused_candidate_update_kernel


def _build_dense_tile_offsets_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_dense_tile_offsets_kernel(tile_keep: int, threads: int = 256):
        rows = T.dynamic("rows")

        @T.prim_func
        def main(
            TileLocalStarts: T.Tensor((rows,), T.int32),
            TileLengths: T.Tensor((rows,), T.int32),
            TileOffsets: T.Tensor((rows, tile_keep), T.int32),
        ):
            with T.Kernel(rows, threads=threads) as row:
                start = TileLocalStarts[row]
                length = TileLengths[row]
                for col in T.Parallel(tile_keep):
                    valid = col < length
                    TileOffsets[row, col] = T.if_then_else(valid, col, -1)

        return main

    return build_dense_tile_offsets_kernel


def _build_final_indices_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_final_indices_kernel(topk: int, threads: int = 256):
        rows = T.dynamic("rows")

        @T.prim_func
        def main(
            BestIndices: T.Tensor((rows, topk), T.int32),
            RowStarts: T.Tensor((rows,), T.int32),
            RowEnds: T.Tensor((rows,), T.int32),
            Output: T.Tensor((rows, topk), T.int32),
        ):
            with T.Kernel(rows, threads=threads) as row:
                length = RowEnds[row] - RowStarts[row]
                for col in T.Parallel(topk):
                    Output[row, col] = T.if_then_else(
                        col < length,
                        BestIndices[row, col],
                        -1,
                    )

        return main

    return build_final_indices_kernel


def _build_block_logits_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_block_logits_kernel(
        heads: int,
        dim: int,
        block_n: int,
        block_m: int = 16,
        block_d: int = 32,
        threads: int = 128,
    ):
        assert dim % block_d == 0

        rows = T.dynamic("rows")
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            Q: T.Tensor((rows, heads, dim), dtype),
            KTile: T.Tensor((block_n, dim), dtype),
            KScales: T.Tensor((block_n,), accum_dtype),
            Weights: T.Tensor((rows, heads), accum_dtype),
            TileLocalStarts: T.Tensor((rows,), T.int32),
            TileLengths: T.Tensor((rows,), T.int32),
            Output: T.Tensor((rows, block_n), accum_dtype),
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as row_block:
                Q_shared = T.alloc_shared((block_m, block_d), dtype)
                K_shared = T.alloc_shared((block_n, block_d), dtype)
                score = T.alloc_fragment((block_m, block_n), accum_dtype)
                logits = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.clear(logits)
                for h in T.serial(heads):
                    T.clear(score)
                    for d_block in T.Pipelined(dim // block_d, num_stages=1):
                        for m_i, d_i in T.Parallel(block_m, block_d):
                            row = row_block * block_m + m_i
                            Q_shared[m_i, d_i] = T.if_then_else(
                                row < rows,
                                Q[row, h, d_block * block_d + d_i],
                                T.cast(0.0, dtype),
                            )
                        for n_i, d_i in T.Parallel(block_n, block_d):
                            K_shared[n_i, d_i] = KTile[
                                n_i, d_block * block_d + d_i
                            ]
                        T.gemm(
                            Q_shared,
                            K_shared,
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    for m_i, n_i in T.Parallel(block_m, block_n):
                        row = row_block * block_m + m_i
                        if row < rows:
                            scaled = score[m_i, n_i] * KScales[n_i]
                            logits[m_i, n_i] = logits[m_i, n_i] + (
                                T.max(scaled, T.cast(0.0, accum_dtype))
                                * Weights[row, h]
                            )

                for m_i, n_i in T.Parallel(block_m, block_n):
                    row = row_block * block_m + m_i
                    if row < rows:
                        start = TileLocalStarts[row]
                        length = TileLengths[row]
                        valid = (n_i >= start) & (n_i < start + length)
                        Output[row, n_i] = T.if_then_else(
                            valid,
                            logits[m_i, n_i],
                            -T.infinity(accum_dtype),
                        )

        return main

    return build_block_logits_kernel


def _build_block_candidate_update_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_block_candidate_update_kernel(
        heads: int,
        dim: int,
        block_n: int,
        topk: int,
        block_m: int = 64,
        block_d: int = 32,
        threads: int = 128,
    ):
        assert dim % block_d == 0

        rows = T.dynamic("rows")
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            Q: T.Tensor((rows, heads, dim), dtype),
            KTile: T.Tensor((block_n, dim), dtype),
            KScales: T.Tensor((block_n,), accum_dtype),
            Weights: T.Tensor((rows, heads), accum_dtype),
            BestScores: T.Tensor((rows, topk), accum_dtype),
            BestIndices: T.Tensor((rows, topk), T.int32),
            RowStarts: T.Tensor((rows,), T.int32),
            TileLocalStarts: T.Tensor((rows,), T.int32),
            TileAbsStarts: T.Tensor((rows,), T.int32),
            TileLengths: T.Tensor((rows,), T.int32),
            NextScores: T.Tensor((rows, topk), accum_dtype),
            NextIndices: T.Tensor((rows, topk), T.int32),
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as row_block:
                Q_shared = T.alloc_shared((block_m, block_d), dtype)
                K_shared = T.alloc_shared((block_n, block_d), dtype)
                block_logits = T.alloc_shared((block_m, block_n), accum_dtype)
                row_scores_shared = T.alloc_shared((topk,), accum_dtype)
                row_indices_shared = T.alloc_shared((topk,), T.int32)
                thread_min_scores = T.alloc_shared((threads,), accum_dtype)
                thread_min_pos = T.alloc_shared((threads,), T.int32)
                rescan_needed = T.alloc_shared((1,), T.int32)
                filled_count = T.alloc_shared((1,), T.int32)
                score = T.alloc_fragment((block_m, block_n), accum_dtype)
                logits = T.alloc_fragment((block_m, block_n), accum_dtype)
                min_score = T.alloc_fragment((1,), accum_dtype)
                min_pos = T.alloc_fragment((1,), T.int32)
                local_min_score = T.alloc_fragment((1,), accum_dtype)
                local_min_pos = T.alloc_fragment((1,), T.int32)

                T.clear(logits)
                for h in T.serial(heads):
                    T.clear(score)
                    for d_block in T.Pipelined(dim // block_d, num_stages=1):
                        for m_i, d_i in T.Parallel(block_m, block_d):
                            row = row_block * block_m + m_i
                            Q_shared[m_i, d_i] = T.if_then_else(
                                row < rows,
                                Q[row, h, d_block * block_d + d_i],
                                T.cast(0.0, dtype),
                            )
                        for n_i, d_i in T.Parallel(block_n, block_d):
                            K_shared[n_i, d_i] = KTile[
                                n_i, d_block * block_d + d_i
                            ]
                        T.gemm(
                            Q_shared,
                            K_shared,
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    for m_i, n_i in T.Parallel(block_m, block_n):
                        row = row_block * block_m + m_i
                        if row < rows:
                            scaled = score[m_i, n_i] * KScales[n_i]
                            logits[m_i, n_i] = logits[m_i, n_i] + (
                                T.max(scaled, T.cast(0.0, accum_dtype))
                                * Weights[row, h]
                            )

                for m_i, n_i in T.Parallel(block_m, block_n):
                    block_logits[m_i, n_i] = logits[m_i, n_i]
                T.sync_threads()

                tx = T.get_thread_binding()
                rounded_topk = T.ceildiv(topk, threads) * threads
                for m_i in T.serial(block_m):
                    row = row_block * block_m + m_i
                    if row < rows:
                        for k_i in T.Parallel(topk):
                            row_scores_shared[k_i] = BestScores[row, k_i]
                            row_indices_shared[k_i] = BestIndices[row, k_i]
                    T.sync_threads()

                    if row < rows:
                        if tx == 0:
                            filled_count[0] = 0
                            for k_i in T.serial(topk):
                                if row_indices_shared[k_i] >= 0:
                                    filled_count[0] = filled_count[0] + 1
                        T.sync_threads()

                        if filled_count[0] == topk:
                            local_min_score[0] = T.infinity(accum_dtype)
                            local_min_pos[0] = 0
                            for k_i in T.serial(tx, rounded_topk, threads):
                                if k_i < topk:
                                    candidate = row_scores_shared[k_i]
                                    if candidate < local_min_score[0]:
                                        local_min_score[0] = candidate
                                        local_min_pos[0] = k_i
                            thread_min_scores[tx] = local_min_score[0]
                            thread_min_pos[tx] = local_min_pos[0]
                            T.sync_threads()

                            if tx == 0:
                                min_score[0] = thread_min_scores[0]
                                min_pos[0] = thread_min_pos[0]
                                for t_i in T.serial(1, threads):
                                    if thread_min_scores[t_i] < min_score[0]:
                                        min_score[0] = thread_min_scores[t_i]
                                        min_pos[0] = thread_min_pos[t_i]
                            T.sync_threads()

                        start = TileLocalStarts[row]
                        end = start + TileLengths[row]
                        for n_i in T.serial(block_n):
                            if (n_i >= start) & (n_i < end):
                                if tx == 0:
                                    v = block_logits[m_i, n_i]
                                    if filled_count[0] < topk:
                                        row_scores_shared[filled_count[0]] = v
                                        row_indices_shared[filled_count[0]] = (
                                            TileAbsStarts[row]
                                            + n_i
                                            - start
                                            - RowStarts[row]
                                        )
                                        filled_count[0] = filled_count[0] + 1
                                        rescan_needed[0] = T.if_then_else(
                                            filled_count[0] == topk,
                                            1,
                                            0,
                                        )
                                    elif v > min_score[0]:
                                        row_scores_shared[min_pos[0]] = v
                                        row_indices_shared[min_pos[0]] = (
                                            TileAbsStarts[row]
                                            + n_i
                                            - start
                                            - RowStarts[row]
                                        )
                                        rescan_needed[0] = 1
                                    else:
                                        rescan_needed[0] = 0
                                T.sync_threads()

                                if rescan_needed[0] != 0:
                                    local_min_score[0] = T.infinity(accum_dtype)
                                    local_min_pos[0] = 0
                                    for k_i in T.serial(tx, rounded_topk, threads):
                                        if k_i < topk:
                                            candidate = row_scores_shared[k_i]
                                            if candidate < local_min_score[0]:
                                                local_min_score[0] = candidate
                                                local_min_pos[0] = k_i
                                    thread_min_scores[tx] = local_min_score[0]
                                    thread_min_pos[tx] = local_min_pos[0]
                                    T.sync_threads()

                                    if tx == 0:
                                        min_score[0] = thread_min_scores[0]
                                        min_pos[0] = thread_min_pos[0]
                                        for t_i in T.serial(1, threads):
                                            if thread_min_scores[t_i] < min_score[0]:
                                                min_score[0] = thread_min_scores[t_i]
                                                min_pos[0] = thread_min_pos[t_i]
                                    T.sync_threads()

                        for k_i in T.Parallel(topk):
                            NextScores[row, k_i] = row_scores_shared[k_i]
                            NextIndices[row, k_i] = row_indices_shared[k_i]
                    T.sync_threads()

        return main

    return build_block_candidate_update_kernel


def _build_fp8_block_candidate_update_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_fp8_block_candidate_update_kernel(
        heads: int,
        dim: int,
        block_n: int,
        topk: int,
        block_m: int = 64,
        block_d: int = 32,
        threads: int = 128,
    ):
        assert dim % block_d == 0

        rows = T.dynamic("rows")
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            Q: T.Tensor((rows, heads, dim), T.uint8),
            KTile: T.Tensor((block_n, dim), T.uint8),
            KScales: T.Tensor((block_n,), accum_dtype),
            Weights: T.Tensor((rows, heads), accum_dtype),
            BestScores: T.Tensor((rows, topk), accum_dtype),
            BestIndices: T.Tensor((rows, topk), T.int32),
            RowStarts: T.Tensor((rows,), T.int32),
            TileLocalStarts: T.Tensor((rows,), T.int32),
            TileAbsStarts: T.Tensor((rows,), T.int32),
            TileLengths: T.Tensor((rows,), T.int32),
            NextScores: T.Tensor((rows, topk), accum_dtype),
            NextIndices: T.Tensor((rows, topk), T.int32),
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as row_block:
                Q_shared = T.alloc_shared((block_m, block_d), dtype)
                K_shared = T.alloc_shared((block_n, block_d), dtype)
                block_logits = T.alloc_shared((block_m, block_n), accum_dtype)
                row_scores_shared = T.alloc_shared((topk,), accum_dtype)
                row_indices_shared = T.alloc_shared((topk,), T.int32)
                thread_min_scores = T.alloc_shared((threads,), accum_dtype)
                thread_min_pos = T.alloc_shared((threads,), T.int32)
                rescan_needed = T.alloc_shared((1,), T.int32)
                filled_count = T.alloc_shared((1,), T.int32)
                score = T.alloc_fragment((block_m, block_n), accum_dtype)
                logits = T.alloc_fragment((block_m, block_n), accum_dtype)
                min_score = T.alloc_fragment((1,), accum_dtype)
                min_pos = T.alloc_fragment((1,), T.int32)
                local_min_score = T.alloc_fragment((1,), accum_dtype)
                local_min_pos = T.alloc_fragment((1,), T.int32)

                T.clear(logits)
                for h in T.serial(heads):
                    T.clear(score)
                    for d_block in T.Pipelined(dim // block_d, num_stages=1):
                        for m_i, d_i in T.Parallel(block_m, block_d):
                            row = row_block * block_m + m_i
                            x_uint8 = T.if_then_else(
                                row < rows,
                                Q[row, h, d_block * block_d + d_i],
                                T.Cast(T.uint8, 0),
                            )
                            val32 = T.Cast(T.int32, x_uint8)
                            sign_bit = (val32 & 0x80) << 24
                            low7 = val32 & 0x7F
                            fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
                            fp32_bits = T.if_then_else(low7 == 0, 0, fp32_bits)
                            normal_val = T.reinterpret(fp32_bits, "float32")
                            is_subnorm = (low7 < 8) & (low7 != 0)
                            subnorm_val = T.Cast(T.float32, low7) * 1.953125e-3
                            sign_mask = (val32 >> 7) & 1
                            subnorm_val = T.if_then_else(
                                sign_mask == 1,
                                -subnorm_val,
                                subnorm_val,
                            )
                            q_float = T.if_then_else(
                                is_subnorm,
                                subnorm_val,
                                normal_val,
                            )
                            Q_shared[m_i, d_i] = T.if_then_else(
                                row < rows,
                                T.Cast(dtype, q_float),
                                T.cast(0.0, dtype),
                            )
                        for n_i, d_i in T.Parallel(block_n, block_d):
                            x_uint8 = KTile[n_i, d_block * block_d + d_i]
                            val32 = T.Cast(T.int32, x_uint8)
                            sign_bit = (val32 & 0x80) << 24
                            low7 = val32 & 0x7F
                            fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
                            fp32_bits = T.if_then_else(low7 == 0, 0, fp32_bits)
                            normal_val = T.reinterpret(fp32_bits, "float32")
                            is_subnorm = (low7 < 8) & (low7 != 0)
                            subnorm_val = T.Cast(T.float32, low7) * 1.953125e-3
                            sign_mask = (val32 >> 7) & 1
                            subnorm_val = T.if_then_else(
                                sign_mask == 1,
                                -subnorm_val,
                                subnorm_val,
                            )
                            k_float = T.if_then_else(
                                is_subnorm,
                                subnorm_val,
                                normal_val,
                            )
                            K_shared[n_i, d_i] = T.Cast(
                                dtype,
                                k_float * KScales[n_i],
                            )
                        T.gemm(
                            Q_shared,
                            K_shared,
                            score,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    for m_i, n_i in T.Parallel(block_m, block_n):
                        row = row_block * block_m + m_i
                        if row < rows:
                            logits[m_i, n_i] = logits[m_i, n_i] + (
                                T.max(
                                    score[m_i, n_i],
                                    T.cast(0.0, accum_dtype),
                                )
                                * Weights[row, h]
                            )

                for m_i, n_i in T.Parallel(block_m, block_n):
                    block_logits[m_i, n_i] = logits[m_i, n_i]
                T.sync_threads()

                tx = T.get_thread_binding()
                rounded_topk = T.ceildiv(topk, threads) * threads
                for m_i in T.serial(block_m):
                    row = row_block * block_m + m_i
                    if row < rows:
                        for k_i in T.Parallel(topk):
                            row_scores_shared[k_i] = BestScores[row, k_i]
                            row_indices_shared[k_i] = BestIndices[row, k_i]
                    T.sync_threads()

                    if row < rows:
                        if tx == 0:
                            filled_count[0] = 0
                            for k_i in T.serial(topk):
                                if row_indices_shared[k_i] >= 0:
                                    filled_count[0] = filled_count[0] + 1
                        T.sync_threads()

                        if filled_count[0] == topk:
                            local_min_score[0] = T.infinity(accum_dtype)
                            local_min_pos[0] = 0
                            for k_i in T.serial(tx, rounded_topk, threads):
                                if k_i < topk:
                                    candidate = row_scores_shared[k_i]
                                    if candidate < local_min_score[0]:
                                        local_min_score[0] = candidate
                                        local_min_pos[0] = k_i
                            thread_min_scores[tx] = local_min_score[0]
                            thread_min_pos[tx] = local_min_pos[0]
                            T.sync_threads()

                            if tx == 0:
                                min_score[0] = thread_min_scores[0]
                                min_pos[0] = thread_min_pos[0]
                                for t_i in T.serial(1, threads):
                                    if thread_min_scores[t_i] < min_score[0]:
                                        min_score[0] = thread_min_scores[t_i]
                                        min_pos[0] = thread_min_pos[t_i]
                            T.sync_threads()

                        start = TileLocalStarts[row]
                        end = start + TileLengths[row]
                        for n_i in T.serial(block_n):
                            if (n_i >= start) & (n_i < end):
                                if tx == 0:
                                    v = block_logits[m_i, n_i]
                                    if filled_count[0] < topk:
                                        row_scores_shared[filled_count[0]] = v
                                        row_indices_shared[filled_count[0]] = (
                                            TileAbsStarts[row]
                                            + n_i
                                            - start
                                            - RowStarts[row]
                                        )
                                        filled_count[0] = filled_count[0] + 1
                                        rescan_needed[0] = T.if_then_else(
                                            filled_count[0] == topk,
                                            1,
                                            0,
                                        )
                                    elif v > min_score[0]:
                                        row_scores_shared[min_pos[0]] = v
                                        row_indices_shared[min_pos[0]] = (
                                            TileAbsStarts[row]
                                            + n_i
                                            - start
                                            - RowStarts[row]
                                        )
                                        rescan_needed[0] = 1
                                    else:
                                        rescan_needed[0] = 0
                                T.sync_threads()

                                if rescan_needed[0] != 0:
                                    local_min_score[0] = T.infinity(accum_dtype)
                                    local_min_pos[0] = 0
                                    for k_i in T.serial(tx, rounded_topk, threads):
                                        if k_i < topk:
                                            candidate = row_scores_shared[k_i]
                                            if candidate < local_min_score[0]:
                                                local_min_score[0] = candidate
                                                local_min_pos[0] = k_i
                                    thread_min_scores[tx] = local_min_score[0]
                                    thread_min_pos[tx] = local_min_pos[0]
                                    T.sync_threads()

                                    if tx == 0:
                                        min_score[0] = thread_min_scores[0]
                                        min_pos[0] = thread_min_pos[0]
                                        for t_i in T.serial(1, threads):
                                            if thread_min_scores[t_i] < min_score[0]:
                                                min_score[0] = thread_min_scores[t_i]
                                                min_pos[0] = thread_min_pos[t_i]
                                    T.sync_threads()

                        for k_i in T.Parallel(topk):
                            NextScores[row, k_i] = row_scores_shared[k_i]
                            NextIndices[row, k_i] = row_indices_shared[k_i]
                    T.sync_threads()

        return main

    return build_fp8_block_candidate_update_kernel


def _get_candidate_buffer_kernel(topk: int, tile_keep: int, threads: int):
    global _CANDIDATE_BUFFER_FACTORY
    if _CANDIDATE_BUFFER_FACTORY is None:
        _CANDIDATE_BUFFER_FACTORY = _build_candidate_buffer_kernel_factory()
    key = (topk, tile_keep, threads)
    if key not in _CANDIDATE_BUFFER_CACHE:
        _CANDIDATE_BUFFER_CACHE[key] = _CANDIDATE_BUFFER_FACTORY(
            topk=topk,
            tile_keep=tile_keep,
            threads=threads,
        )
    return _CANDIDATE_BUFFER_CACHE[key]


def _get_candidate_gather_kernel(topk: int, candidate_width: int, threads: int):
    global _CANDIDATE_GATHER_FACTORY
    if _CANDIDATE_GATHER_FACTORY is None:
        _CANDIDATE_GATHER_FACTORY = _build_candidate_gather_kernel_factory()
    key = (topk, candidate_width, threads)
    if key not in _CANDIDATE_GATHER_CACHE:
        _CANDIDATE_GATHER_CACHE[key] = _CANDIDATE_GATHER_FACTORY(
            topk=topk,
            candidate_width=candidate_width,
            threads=threads,
        )
    return _CANDIDATE_GATHER_CACHE[key]


def _get_fused_candidate_update_kernel(topk: int, tile_keep: int, threads: int):
    global _FUSED_CANDIDATE_UPDATE_FACTORY
    if _FUSED_CANDIDATE_UPDATE_FACTORY is None:
        _FUSED_CANDIDATE_UPDATE_FACTORY = (
            _build_fused_candidate_update_kernel_factory()
        )
    key = (topk, tile_keep, threads)
    if key not in _FUSED_CANDIDATE_UPDATE_CACHE:
        _FUSED_CANDIDATE_UPDATE_CACHE[key] = _FUSED_CANDIDATE_UPDATE_FACTORY(
            topk=topk,
            tile_keep=tile_keep,
            threads=threads,
        )
    return _FUSED_CANDIDATE_UPDATE_CACHE[key]


def _get_dense_tile_offsets_kernel(tile_keep: int, threads: int):
    global _DENSE_TILE_OFFSETS_FACTORY
    if _DENSE_TILE_OFFSETS_FACTORY is None:
        _DENSE_TILE_OFFSETS_FACTORY = _build_dense_tile_offsets_kernel_factory()
    key = (tile_keep, threads)
    if key not in _DENSE_TILE_OFFSETS_CACHE:
        _DENSE_TILE_OFFSETS_CACHE[key] = _DENSE_TILE_OFFSETS_FACTORY(
            tile_keep=tile_keep,
            threads=threads,
        )
    return _DENSE_TILE_OFFSETS_CACHE[key]


def _get_final_indices_kernel(topk: int, threads: int):
    global _FINAL_INDICES_FACTORY
    if _FINAL_INDICES_FACTORY is None:
        _FINAL_INDICES_FACTORY = _build_final_indices_kernel_factory()
    key = (topk, threads)
    if key not in _FINAL_INDICES_CACHE:
        _FINAL_INDICES_CACHE[key] = _FINAL_INDICES_FACTORY(
            topk=topk,
            threads=threads,
        )
    return _FINAL_INDICES_CACHE[key]


def _get_block_logits_kernel(
    heads: int,
    dim: int,
    block_n: int,
    block_m: int,
    block_d: int,
    threads: int,
):
    global _BLOCK_LOGITS_FACTORY
    if _BLOCK_LOGITS_FACTORY is None:
        _BLOCK_LOGITS_FACTORY = _build_block_logits_kernel_factory()
    key = (heads, dim, block_n, block_m, block_d, threads)
    if key not in _BLOCK_LOGITS_CACHE:
        _BLOCK_LOGITS_CACHE[key] = _BLOCK_LOGITS_FACTORY(
            heads=heads,
            dim=dim,
            block_n=block_n,
            block_m=block_m,
            block_d=block_d,
            threads=threads,
        )
    return _BLOCK_LOGITS_CACHE[key]


def _get_block_candidate_update_kernel(
    heads: int,
    dim: int,
    block_n: int,
    topk: int,
    block_m: int,
    block_d: int,
    threads: int,
):
    global _BLOCK_CANDIDATE_UPDATE_FACTORY
    if _BLOCK_CANDIDATE_UPDATE_FACTORY is None:
        _BLOCK_CANDIDATE_UPDATE_FACTORY = (
            _build_block_candidate_update_kernel_factory()
        )
    key = (heads, dim, block_n, topk, block_m, block_d, threads)
    if key not in _BLOCK_CANDIDATE_UPDATE_CACHE:
        _BLOCK_CANDIDATE_UPDATE_CACHE[key] = _BLOCK_CANDIDATE_UPDATE_FACTORY(
            heads=heads,
            dim=dim,
            block_n=block_n,
            topk=topk,
            block_m=block_m,
            block_d=block_d,
            threads=threads,
        )
    return _BLOCK_CANDIDATE_UPDATE_CACHE[key]


def _get_fp8_block_candidate_update_kernel(
    heads: int,
    dim: int,
    block_n: int,
    topk: int,
    block_m: int,
    block_d: int,
    threads: int,
):
    global _FP8_BLOCK_CANDIDATE_UPDATE_FACTORY
    if _FP8_BLOCK_CANDIDATE_UPDATE_FACTORY is None:
        _FP8_BLOCK_CANDIDATE_UPDATE_FACTORY = (
            _build_fp8_block_candidate_update_kernel_factory()
        )
    key = (heads, dim, block_n, topk, block_m, block_d, threads)
    if key not in _FP8_BLOCK_CANDIDATE_UPDATE_CACHE:
        _FP8_BLOCK_CANDIDATE_UPDATE_CACHE[key] = (
            _FP8_BLOCK_CANDIDATE_UPDATE_FACTORY(
                heads=heads,
                dim=dim,
                block_n=block_n,
                topk=topk,
                block_m=block_m,
                block_d=block_d,
                threads=threads,
            )
        )
    return _FP8_BLOCK_CANDIDATE_UPDATE_CACHE[key]


def _compute_block_logits_tilelang(
    *,
    q: torch.Tensor,
    k_tile: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    tile_local_starts: torch.Tensor,
    tile_lengths: torch.Tensor,
    block_m: int = 16,
    block_d: int = 32,
    threads: int = 128,
) -> torch.Tensor:
    _require_cuda_tensors(
        q,
        k_tile,
        k_scales,
        weights,
        tile_local_starts,
        tile_lengths,
    )
    ok, reason = is_tilelang_available()
    if not ok:
        raise RuntimeError(reason)
    if q.dtype is not torch.float16 or k_tile.dtype is not torch.float16:
        raise ValueError("TileLang block logits currently requires fp16 q/k")
    if q.ndim != 3 or k_tile.ndim != 2:
        raise ValueError("q must be [rows, heads, dim], k_tile must be [block_n, dim]")
    rows, heads, dim = q.shape
    block_n, k_dim = k_tile.shape
    if k_dim != dim:
        raise ValueError(f"k_tile dim {k_dim} must match q dim {dim}")
    if dim % block_d != 0:
        raise ValueError(f"q dim {dim} must be divisible by block_d={block_d}")
    # SM70's mma_m8n8k4 lowering is much less permissive than newer WGMMA
    # paths.  The FullRow policy used for N=16 matches the sparse-prefill
    # kernel's stable 64x16 tile; smaller M tiles fail TileLang layout
    # inference before codegen.
    kernel_block_m = max(64, ((block_m + 63) // 64) * 64)

    kernel_block_n = max(16, ((block_n + 15) // 16) * 16)
    if kernel_block_n != block_n:
        padded_k_tile = k_tile.new_zeros((kernel_block_n, dim))
        padded_k_tile[:block_n].copy_(k_tile)
        padded_scales = torch.zeros(
            (kernel_block_n,),
            dtype=torch.float32,
            device=k_tile.device,
        )
        padded_scales[:block_n].copy_(k_scales.reshape(-1).to(torch.float32))
        k_tile = padded_k_tile
        k_scales = padded_scales

    output = torch.empty(
        (rows, kernel_block_n),
        dtype=torch.float32,
        device=q.device,
    )
    kernel = _get_block_logits_kernel(
        heads=heads,
        dim=dim,
        block_n=kernel_block_n,
        block_m=kernel_block_m,
        block_d=block_d,
        threads=threads,
    )
    kernel(
        q.contiguous(),
        k_tile.contiguous(),
        k_scales.reshape(-1).to(torch.float32).contiguous(),
        weights.to(torch.float32).contiguous(),
        tile_local_starts.to(torch.int32).contiguous(),
        tile_lengths.to(torch.int32).contiguous(),
        output,
    )
    return output[:, :block_n]


def _update_best_candidates_from_block_tilelang(
    *,
    q: torch.Tensor,
    k_tile: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    best_scores: torch.Tensor,
    best_indices: torch.Tensor,
    row_starts: torch.Tensor,
    tile_local_starts: torch.Tensor,
    tile_lengths: torch.Tensor,
    block_start: int,
    topk_tokens: int,
    block_m: int = 16,
    block_d: int = 32,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda_tensors(
        q,
        k_tile,
        k_scales,
        weights,
        best_scores,
        best_indices,
        row_starts,
        tile_local_starts,
        tile_lengths,
    )
    ok, reason = is_tilelang_available()
    if not ok:
        raise RuntimeError(reason)
    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    use_fp16 = q.dtype is torch.float16 and k_tile.dtype is torch.float16
    use_fp8 = (
        fp8_e4m3fn is not None
        and q.dtype is fp8_e4m3fn
        and k_tile.dtype is fp8_e4m3fn
    )
    if not (use_fp16 or use_fp8):
        raise ValueError(
            "TileLang block candidate update requires fp16 or fp8_e4m3fn q/k"
        )
    if best_scores.dtype is not torch.float32 or best_indices.dtype is not torch.int32:
        raise ValueError("best_scores must be fp32 and best_indices must be int32")
    if best_scores.shape != best_indices.shape:
        raise ValueError("best_scores and best_indices must have identical shapes")
    rows, heads, dim = q.shape
    block_n, k_dim = k_tile.shape
    if best_scores.shape != (rows, topk_tokens):
        raise ValueError("best candidate tensors must be [rows, topk_tokens]")
    if k_dim != dim:
        raise ValueError(f"k_tile dim {k_dim} must match q dim {dim}")
    if dim % block_d != 0:
        raise ValueError(f"q dim {dim} must be divisible by block_d={block_d}")

    kernel_block_m = max(64, ((block_m + 63) // 64) * 64)
    kernel_block_n = max(16, ((block_n + 15) // 16) * 16)
    if kernel_block_n != block_n:
        padded_k_tile = k_tile.new_zeros((kernel_block_n, dim))
        padded_k_tile[:block_n].copy_(k_tile)
        padded_scales = torch.zeros(
            (kernel_block_n,),
            dtype=torch.float32,
            device=k_tile.device,
        )
        padded_scales[:block_n].copy_(k_scales.reshape(-1).to(torch.float32))
        k_tile = padded_k_tile
        k_scales = padded_scales
    next_scores = torch.empty_like(best_scores)
    next_indices = torch.empty_like(best_indices)
    tile_abs_starts = (
        tile_local_starts.to(torch.int32) + int(block_start)
    ).contiguous()
    if use_fp8:
        kernel = _get_fp8_block_candidate_update_kernel(
            heads=heads,
            dim=dim,
            block_n=kernel_block_n,
            topk=topk_tokens,
            block_m=kernel_block_m,
            block_d=block_d,
            threads=threads,
        )
        kernel(
            q.contiguous().view(torch.uint8),
            k_tile.contiguous().view(torch.uint8),
            k_scales.reshape(-1).to(torch.float32).contiguous(),
            weights.to(torch.float32).contiguous(),
            best_scores.contiguous(),
            best_indices.contiguous(),
            row_starts.to(torch.int32).contiguous(),
            tile_local_starts.to(torch.int32).contiguous(),
            tile_abs_starts,
            tile_lengths.to(torch.int32).contiguous(),
            next_scores,
            next_indices,
        )
    else:
        kernel = _get_block_candidate_update_kernel(
            heads=heads,
            dim=dim,
            block_n=kernel_block_n,
            topk=topk_tokens,
            block_m=kernel_block_m,
            block_d=block_d,
            threads=threads,
        )
        kernel(
            q.contiguous(),
            k_tile.contiguous(),
            k_scales.reshape(-1).to(torch.float32).contiguous(),
            weights.to(torch.float32).contiguous(),
            best_scores.contiguous(),
            best_indices.contiguous(),
            row_starts.to(torch.int32).contiguous(),
            tile_local_starts.to(torch.int32).contiguous(),
            tile_abs_starts,
            tile_lengths.to(torch.int32).contiguous(),
            next_scores,
            next_indices,
        )
    return next_scores, next_indices


def _compute_tile_logits(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    tile_start: int,
    tile_end: int,
) -> torch.Tensor:
    k_tile = k_cache_values[tile_start:tile_end]
    scale_tile = k_cache_scales.reshape(-1)[tile_start:tile_end]
    if _can_use_sm70_fp8_tile_logits(q, k_cache_values):
        tile_len = tile_end - tile_start
        tile_row_starts = torch.clamp(row_starts - tile_start, 0, tile_len)
        tile_row_ends = torch.clamp(row_ends - tile_start, 0, tile_len)
        logits_kernel = (
            sm70_fp8_mqa_logits if q.shape[1] <= 8 else sm70_fp8_mqa_logits_gemm
        )
        return logits_kernel(
            q,
            (k_tile, scale_tile),
            weights,
            tile_row_starts.to(torch.int32).contiguous(),
            tile_row_ends.to(torch.int32).contiguous(),
        )

    q_f32 = q.float()
    weights_t = weights.float().transpose(0, 1).unsqueeze(-1)
    k_tile_f32 = k_tile.float() * scale_tile.float().view(-1, 1)
    score = torch.einsum("mhd,nd->hmn", q_f32, k_tile_f32)
    tile_logits = (score.relu() * weights_t).sum(dim=0)
    positions = torch.arange(tile_start, tile_end, device=q.device)
    row_starts_i64 = row_starts.to(torch.int64)
    row_ends_i64 = row_ends.to(torch.int64)
    valid = (
        (positions.view(1, -1) >= row_starts_i64.view(-1, 1))
        & (positions.view(1, -1) < row_ends_i64.view(-1, 1))
    )
    return tile_logits.masked_fill(~valid, -torch.inf)


def _prefill_streaming_topk_chunked_torch(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    tile_k: int,
) -> None:
    row_starts_i64 = row_starts.to(torch.int64)
    row_ends_i64 = row_ends.to(torch.int64)
    rows = q.shape[0]
    kv_tokens = k_cache_values.shape[0]
    device = q.device

    best_scores = torch.empty((rows, 0), dtype=torch.float32, device=device)
    best_indices = torch.empty((rows, 0), dtype=torch.int32, device=device)
    for tile_start in range(0, kv_tokens, tile_k):
        tile_end = min(tile_start + tile_k, kv_tokens)
        tile_logits = _compute_tile_logits(
            q=q,
            k_cache_values=k_cache_values,
            k_cache_scales=k_cache_scales,
            weights=weights,
            row_starts=row_starts,
            row_ends=row_ends,
            tile_start=tile_start,
            tile_end=tile_end,
        )
        positions = torch.arange(tile_start, tile_end, device=device)
        tile_topk = min(topk_tokens, tile_end - tile_start)
        tile_scores, tile_offsets = tile_logits.topk(tile_topk, dim=1)
        tile_indices = (
            positions[tile_offsets].to(torch.int64) - row_starts_i64.view(-1, 1)
        ).to(torch.int32)

        merged_scores = torch.cat((best_scores, tile_scores), dim=1)
        merged_indices = torch.cat((best_indices, tile_indices), dim=1)
        keep = min(topk_tokens, merged_scores.shape[1])
        best_scores, keep_pos = merged_scores.topk(keep, dim=1)
        best_indices = merged_indices.gather(1, keep_pos)

    out_indices.fill_(-1)
    if best_indices.shape[1] == 0:
        return
    if best_indices.shape[1] < topk_tokens:
        pad = torch.full(
            (rows, topk_tokens - best_indices.shape[1]),
            -1,
            dtype=torch.int32,
            device=device,
        )
        best_indices = torch.cat((best_indices, pad), dim=1)
    valid_counts = torch.clamp(row_ends_i64 - row_starts_i64, min=0, max=topk_tokens)
    cols = torch.arange(topk_tokens, device=device).view(1, -1)
    valid_out = cols < valid_counts.view(-1, 1)
    selected = best_indices[:, :topk_tokens]
    out_indices.copy_(torch.where(valid_out, selected, torch.full_like(selected, -1)))


def _select_tile_offsets_tilelang(
    *,
    tile_logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    tile_start: int,
    topk_tokens: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, tile_len = tile_logits.shape
    tile_keep = topk_tokens
    tile_row_starts = torch.clamp(
        row_starts - tile_start,
        min=0,
        max=tile_len,
    ).to(torch.int32).contiguous()
    tile_row_ends = torch.clamp(
        row_ends - tile_start,
        min=0,
        max=tile_len,
    ).to(torch.int32).contiguous()
    tile_lengths = torch.clamp(
        tile_row_ends - tile_row_starts,
        min=0,
        max=tile_len,
    ).to(torch.int32).contiguous()
    tile_offsets = torch.empty(
        (rows, tile_keep),
        dtype=torch.int32,
        device=tile_logits.device,
    )
    tile_logits_for_topk = tile_logits
    if tile_len < tile_keep:
        tile_logits_for_topk = tile_logits.new_full(
            (rows, tile_keep),
            -torch.inf,
        )
        tile_logits_for_topk[:, :tile_len].copy_(tile_logits)
    prefill_topk_tilelang(
        tile_logits_for_topk.contiguous(),
        tile_offsets,
        tile_lengths,
        tile_row_starts,
        topk_tokens=tile_keep,
        threads=threads,
    )
    return tile_offsets, tile_row_starts


def _update_best_candidates_tilelang(
    *,
    best_scores: torch.Tensor,
    best_indices: torch.Tensor,
    tile_logits: torch.Tensor,
    tile_offsets: torch.Tensor,
    row_starts: torch.Tensor,
    tile_row_starts: torch.Tensor,
    tile_start: int,
    topk_tokens: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = best_scores.shape[0]
    tile_keep = tile_offsets.shape[1]
    tile_abs_starts = (tile_row_starts + tile_start).to(torch.int32).contiguous()
    next_scores = torch.empty_like(best_scores)
    next_indices = torch.empty_like(best_indices)
    update_kernel = _get_fused_candidate_update_kernel(
        topk_tokens,
        tile_keep,
        threads,
    )
    update_kernel(
        best_scores,
        best_indices,
        tile_logits.contiguous(),
        tile_offsets,
        row_starts.contiguous(),
        tile_row_starts,
        tile_abs_starts,
        next_scores,
        next_indices,
    )
    return next_scores, next_indices


def _update_best_candidates_from_scores_tilelang(
    *,
    best_scores: torch.Tensor,
    best_indices: torch.Tensor,
    tile_scores: torch.Tensor,
    row_starts: torch.Tensor,
    tile_local_starts: torch.Tensor,
    tile_abs_starts: torch.Tensor,
    tile_lengths: torch.Tensor | None = None,
    topk_tokens: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, tile_len = tile_scores.shape
    if tile_len == 0:
        return best_scores, best_indices

    tile_keep = min(topk_tokens, tile_len)
    tile_offsets = torch.empty(
        (rows, tile_keep),
        dtype=torch.int32,
        device=tile_scores.device,
    )
    if tile_lengths is None:
        tile_lengths = torch.clamp(
            tile_len - tile_local_starts,
            min=0,
            max=tile_len,
        )
    tile_lengths = tile_lengths.to(torch.int32).contiguous()
    tile_scores = tile_scores.contiguous()
    tile_local_starts = tile_local_starts.to(torch.int32).contiguous()
    tile_abs_starts = tile_abs_starts.to(torch.int32).contiguous()
    if tile_keep == tile_len and tile_keep <= topk_tokens:
        offsets_kernel = _get_dense_tile_offsets_kernel(tile_keep, threads)
        offsets_kernel(tile_local_starts, tile_lengths, tile_offsets)
    else:
        prefill_topk_tilelang(
            tile_scores,
            tile_offsets,
            tile_lengths,
            tile_local_starts,
            topk_tokens=tile_keep,
            threads=threads,
        )

    next_scores = torch.empty_like(best_scores)
    next_indices = torch.empty_like(best_indices)
    update_kernel = _get_fused_candidate_update_kernel(
        topk_tokens,
        tile_keep,
        threads,
    )
    update_kernel(
        best_scores,
        best_indices,
        tile_scores,
        tile_offsets,
        row_starts.contiguous(),
        tile_local_starts,
        tile_abs_starts,
        next_scores,
        next_indices,
    )
    return next_scores, next_indices


def _copy_final_indices_tilelang(
    *,
    best_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    threads: int,
) -> None:
    kernel = _get_final_indices_kernel(topk_tokens, threads)
    kernel(
        best_indices,
        row_starts.contiguous(),
        row_ends.contiguous(),
        out_indices,
    )


def prewarm_prefill_streaming_topk_tilelang(
    topk_tokens: int,
    *,
    tile_k: int = 1024,
    threads: int = 256,
) -> None:
    """JIT compile TileLang kernels used by the streaming top-k path.

    Keeping these compiles out of the request path avoids invoking TileLang's
    layout lowering from vLLM's compiled custom-op execution frame.
    """
    ok, reason = is_tilelang_available()
    if not ok:
        raise RuntimeError(reason)
    if topk_tokens > _MAX_BENCHED_TILELANG_CANDIDATE_TOPK:
        return
    if envs.VLLM_SPARSE_INDEXER_PREFILL_FUSED_TILE_TOPK:
        block_k = max(
            1,
            min(envs.VLLM_SPARSE_INDEXER_PREFILL_FUSED_TILE_BLOCK_K, tile_k),
        )
        tile_keep = min(topk_tokens, block_k)
        use_dense_offsets = block_k <= topk_tokens
    else:
        tile_keep = min(topk_tokens, tile_k)
        use_dense_offsets = False
    if use_dense_offsets:
        _get_dense_tile_offsets_kernel(tile_keep, threads)
    else:
        prewarm_prefill_topk_tilelang(tile_keep, threads)
    _get_fused_candidate_update_kernel(topk_tokens, tile_keep, threads)
    _get_final_indices_kernel(topk_tokens, threads)


def _prefill_streaming_topk_chunked_tilelang(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    tile_k: int,
    threads: int,
) -> None:
    rows = q.shape[0]
    kv_tokens = k_cache_values.shape[0]
    best_scores = torch.full(
        (rows, topk_tokens),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    best_indices = torch.full(
        (rows, topk_tokens),
        -1,
        dtype=torch.int32,
        device=q.device,
    )

    for tile_start in range(0, kv_tokens, tile_k):
        tile_end = min(tile_start + tile_k, kv_tokens)
        tile_logits = _compute_tile_logits(
            q=q,
            k_cache_values=k_cache_values,
            k_cache_scales=k_cache_scales,
            weights=weights,
            row_starts=row_starts,
            row_ends=row_ends,
            tile_start=tile_start,
            tile_end=tile_end,
        )
        tile_offsets, tile_row_starts = _select_tile_offsets_tilelang(
            tile_logits=tile_logits,
            row_starts=row_starts,
            row_ends=row_ends,
            tile_start=tile_start,
            topk_tokens=topk_tokens,
            threads=threads,
        )
        best_scores, best_indices = _update_best_candidates_tilelang(
            best_scores=best_scores,
            best_indices=best_indices,
            tile_logits=tile_logits,
            tile_offsets=tile_offsets,
            row_starts=row_starts,
            tile_row_starts=tile_row_starts,
            tile_start=tile_start,
            topk_tokens=topk_tokens,
            threads=threads,
        )

    _copy_final_indices_tilelang(
        best_indices=best_indices,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        threads=threads,
    )


def _prefill_streaming_topk_blocked_tilelang(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    tile_k: int,
    block_k: int,
    threads: int,
) -> None:
    rows = q.shape[0]
    kv_tokens = k_cache_values.shape[0]
    block_k = max(1, min(block_k, tile_k))
    best_scores = torch.full(
        (rows, topk_tokens),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    best_indices = torch.full(
        (rows, topk_tokens),
        -1,
        dtype=torch.int32,
        device=q.device,
    )

    for tile_start in range(0, kv_tokens, tile_k):
        tile_end = min(tile_start + tile_k, kv_tokens)
        for block_start in range(tile_start, tile_end, block_k):
            block_end = min(block_start + block_k, tile_end)
            tile_len = block_end - block_start
            tile_local_starts = torch.clamp(
                row_starts - block_start,
                min=0,
                max=tile_len,
            ).to(torch.int32).contiguous()
            tile_local_ends = torch.clamp(
                row_ends - block_start,
                min=0,
                max=tile_len,
            ).to(torch.int32).contiguous()
            tile_lengths = torch.clamp(
                tile_local_ends - tile_local_starts,
                min=0,
                max=tile_len,
            ).to(torch.int32).contiguous()
            if _can_use_tilelang_block_candidate_update(q, k_cache_values):
                best_scores, best_indices = (
                    _update_best_candidates_from_block_tilelang(
                        q=q,
                        k_tile=k_cache_values[block_start:block_end],
                        k_scales=k_cache_scales.reshape(-1)[block_start:block_end],
                        weights=weights,
                        best_scores=best_scores,
                        best_indices=best_indices,
                        row_starts=row_starts,
                        tile_local_starts=tile_local_starts,
                        tile_lengths=tile_lengths,
                        block_start=block_start,
                        topk_tokens=topk_tokens,
                        block_d=16,
                        threads=min(threads, 128),
                    )
                )
                continue
            if _can_use_sm70_fp8_fused_block_update(q, k_cache_values):
                best_scores, best_indices = sm70_fp8_mqa_block_candidate_update(
                    q=q,
                    kv=(k_cache_values, k_cache_scales.reshape(-1)),
                    weights=weights,
                    row_starts=row_starts,
                    row_ends=row_ends,
                    best_scores=best_scores,
                    best_indices=best_indices,
                    block_start=block_start,
                    block_end=block_end,
                    topk_tokens=topk_tokens,
                )
                continue
            tile_scores = _compute_tile_logits(
                q=q,
                k_cache_values=k_cache_values,
                k_cache_scales=k_cache_scales,
                weights=weights,
                row_starts=row_starts,
                row_ends=row_ends,
                tile_start=block_start,
                tile_end=block_end,
            )
            tile_abs_starts = (
                tile_local_starts + block_start
            ).to(torch.int32).contiguous()
            best_scores, best_indices = (
                _update_best_candidates_from_scores_tilelang(
                    best_scores=best_scores,
                    best_indices=best_indices,
                    tile_scores=tile_scores,
                    row_starts=row_starts,
                    tile_local_starts=tile_local_starts,
                    tile_abs_starts=tile_abs_starts,
                    tile_lengths=tile_lengths,
                    topk_tokens=topk_tokens,
                    threads=threads,
                )
            )

    _copy_final_indices_tilelang(
        best_indices=best_indices,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        threads=threads,
    )


def _prefill_streaming_topk_oracle(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    tile_k: int,
    threads: int,
) -> None:
    ok, _ = is_tilelang_available()
    if (
        ok
        and _is_sm70_tensor_device(q)
        and topk_tokens <= _MAX_BENCHED_TILELANG_CANDIDATE_TOPK
    ):
        if envs.VLLM_SPARSE_INDEXER_PREFILL_FUSED_TILE_TOPK:
            _prefill_streaming_topk_blocked_tilelang(
                q=q,
                k_cache_values=k_cache_values,
                k_cache_scales=k_cache_scales,
                weights=weights,
                row_starts=row_starts,
                row_ends=row_ends,
                out_indices=out_indices,
                topk_tokens=topk_tokens,
                tile_k=tile_k,
                block_k=envs.VLLM_SPARSE_INDEXER_PREFILL_FUSED_TILE_BLOCK_K,
                threads=threads,
            )
            return
        _prefill_streaming_topk_chunked_tilelang(
            q=q,
            k_cache_values=k_cache_values,
            k_cache_scales=k_cache_scales,
            weights=weights,
            row_starts=row_starts,
            row_ends=row_ends,
            out_indices=out_indices,
            topk_tokens=topk_tokens,
            tile_k=tile_k,
            threads=threads,
        )
        return
    _prefill_streaming_topk_chunked_torch(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=tile_k,
    )


def prefill_streaming_topk_tilelang(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    tile_k: int = 1024,
    threads: int = 256,
) -> None:
    if not _is_supported_q_dtype(q.dtype):
        raise ValueError("q must be fp16, bf16, or fp8")
    if q.ndim != 3:
        raise ValueError("q must be a 3D [rows, heads, dim] tensor")
    if k_cache_values.ndim != 2 or k_cache_values.shape[1] != q.shape[-1]:
        raise ValueError("k_cache_values must be 2D with the same dim as q")
    if k_cache_scales.reshape(-1).shape[0] != k_cache_values.shape[0]:
        raise ValueError("k_cache_scales must have one scale per K row")
    if weights.shape != q.shape[:2]:
        raise ValueError("weights must have shape [rows, heads]")
    if out_indices.dtype is not torch.int32 or out_indices.ndim != 2:
        raise ValueError("out_indices must be a 2D int32 tensor")
    if row_starts.dtype is not torch.int32 or row_ends.dtype is not torch.int32:
        raise ValueError("row_starts and row_ends must be int32")
    if row_starts.shape != row_ends.shape or row_starts.shape[0] != q.shape[0]:
        raise ValueError("one row range is required per query row")
    if topk_tokens != out_indices.shape[1]:
        raise ValueError("topk_tokens must match out_indices width")
    if tile_k <= 0 or tile_k % 128 != 0:
        raise ValueError("tile_k must be a positive multiple of 128")
    _require_cuda_tensors(
        q,
        k_cache_values,
        k_cache_scales,
        weights,
        row_starts,
        row_ends,
        out_indices,
    )
    return _prefill_streaming_topk_oracle(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=out_indices,
        topk_tokens=topk_tokens,
        tile_k=tile_k,
        threads=threads,
    )
