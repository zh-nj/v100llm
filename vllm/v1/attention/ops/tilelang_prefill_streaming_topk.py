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

from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_mqa_logits
from vllm.v1.attention.ops.tilelang_prefill_topk import (
    is_tilelang_available,
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
_FINAL_INDICES_FACTORY = None
_FINAL_INDICES_CACHE: dict[tuple[int, int], object] = {}


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
        and q.shape[1] <= 8
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
                            neg_inf,
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
        return sm70_fp8_mqa_logits(
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
    tile_keep = min(topk_tokens, tile_len)
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
    prefill_topk_tilelang(
        tile_logits.contiguous(),
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
    candidate_width = topk_tokens + tile_keep
    candidate_scores = torch.empty(
        (rows, candidate_width),
        dtype=torch.float32,
        device=best_scores.device,
    )
    candidate_indices = torch.empty(
        (rows, candidate_width),
        dtype=torch.int32,
        device=best_scores.device,
    )
    tile_abs_starts = (tile_row_starts + tile_start).to(torch.int32).contiguous()
    fill_kernel = _get_candidate_buffer_kernel(topk_tokens, tile_keep, threads)
    fill_kernel(
        best_scores,
        best_indices,
        tile_logits.contiguous(),
        tile_offsets,
        row_starts.contiguous(),
        tile_row_starts,
        tile_abs_starts,
        candidate_scores,
        candidate_indices,
    )

    merge_positions = torch.empty(
        (rows, topk_tokens),
        dtype=torch.int32,
        device=best_scores.device,
    )
    candidate_row_starts = torch.zeros(
        (rows,),
        dtype=torch.int32,
        device=best_scores.device,
    )
    candidate_lengths = torch.full(
        (rows,),
        candidate_width,
        dtype=torch.int32,
        device=best_scores.device,
    )
    prefill_topk_tilelang(
        candidate_scores,
        merge_positions,
        candidate_lengths,
        candidate_row_starts,
        topk_tokens=topk_tokens,
        threads=threads,
    )

    next_scores = torch.empty_like(best_scores)
    next_indices = torch.empty_like(best_indices)
    gather_kernel = _get_candidate_gather_kernel(topk_tokens, candidate_width, threads)
    gather_kernel(
        candidate_scores,
        candidate_indices,
        merge_positions,
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
    if ok and _is_sm70_tensor_device(q):
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
