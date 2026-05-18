# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming prefill top-k wrapper for the DSv4F sparse indexer.

The public call boundary is intentionally shaped like the final implementation:
query rows + gathered K cache + row ranges -> local top-k indices.  The first
body uses torch tile chunks to establish the streaming candidate-merge
semantics before the hot loop is lowered into TileLang/CUDA.
"""

import torch


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors):
        raise ValueError("streaming topk inputs must be CUDA tensors")


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
    q_f32 = q.float()
    weights_t = weights.float().transpose(0, 1).unsqueeze(-1)
    scale_flat = k_cache_scales.reshape(-1).float()
    row_starts_i64 = row_starts.to(torch.int64)
    row_ends_i64 = row_ends.to(torch.int64)
    rows = q.shape[0]
    kv_tokens = k_cache_values.shape[0]
    device = q.device

    best_scores = torch.empty((rows, 0), dtype=torch.float32, device=device)
    best_indices = torch.empty((rows, 0), dtype=torch.int32, device=device)
    for tile_start in range(0, kv_tokens, tile_k):
        tile_end = min(tile_start + tile_k, kv_tokens)
        k_tile = (
            k_cache_values[tile_start:tile_end].float()
            * scale_flat[tile_start:tile_end].view(-1, 1)
        )
        score = torch.einsum("mhd,nd->hmn", q_f32, k_tile)
        tile_logits = (score.relu() * weights_t).sum(dim=0)
        positions = torch.arange(tile_start, tile_end, device=device)
        valid = (
            (positions.view(1, -1) >= row_starts_i64.view(-1, 1))
            & (positions.view(1, -1) < row_ends_i64.view(-1, 1))
        )
        tile_logits = tile_logits.masked_fill(~valid, -torch.inf)
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
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be fp16 or bf16")
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
