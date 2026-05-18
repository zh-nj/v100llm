# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors):
        raise ValueError("sparse prefill v2 inputs must be CUDA tensors")


def _flash_mla_sparse_prefill_v2_oracle(
    *,
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    swa_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    swa_block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    top_k: int,
    sm_scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "sparse prefill v2 oracle is not routed yet; Task 4 replaces this "
        "helper with direct-cache sparse prefill execution"
    )


def flash_mla_sparse_prefill_v2(
    *,
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    swa_k_cache: torch.Tensor,
    compressed_block_table: torch.Tensor,
    swa_block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    top_k: int,
    sm_scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.dtype is not torch.float16 or out.dtype is not torch.float16:
        raise ValueError("sparse prefill v2 starts with fp16 q/out only")
    if q.ndim != 3 or tuple(q.shape[1:]) != (64, 576):
        raise ValueError("q must have shape [tokens, 64, 576]")
    if out.ndim != 3 or tuple(out.shape[1:]) != (64, 512):
        raise ValueError("out must have shape [tokens, 64, 512]")
    if q.shape[0] != out.shape[0]:
        raise ValueError("q and out must have the same token count")
    if compress_ratio not in (4, 128):
        raise ValueError("compress_ratio must be 4 or 128")
    if topk_indices.dtype is not torch.int32:
        raise ValueError("topk_indices must be int32")

    _require_cuda_tensors(
        q,
        compressed_k_cache,
        swa_k_cache,
        compressed_block_table,
        swa_block_table,
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        attn_sink,
        out,
    )

    return _flash_mla_sparse_prefill_v2_oracle(
        q=q,
        compressed_k_cache=compressed_k_cache,
        swa_k_cache=swa_k_cache,
        compressed_block_table=compressed_block_table,
        swa_block_table=swa_block_table,
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=window_size,
        compress_ratio=compress_ratio,
        top_k=top_k,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        out=out,
    )
