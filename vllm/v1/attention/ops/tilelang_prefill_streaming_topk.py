# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming prefill top-k wrapper for the DSv4F sparse indexer.

The public call boundary is intentionally shaped like the final implementation:
query rows + gathered K cache + row ranges -> local top-k indices.  FP8 tile
logits use the SM70 CUDA/Triton logits kernel; non-FP8 shapes retain the torch
reference path.  Tile top-k and cross-tile candidate merge are still the next
kernelization target.
"""

import torch

from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_mqa_logits


_FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz")
    if hasattr(torch, name)
)


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors):
        raise ValueError("streaming topk inputs must be CUDA tensors")


def _is_supported_q_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16, *_FP8_DTYPES)


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
    del threads
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
