# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops import tilelang_sparse_prefill
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    dequantize_and_gather_k_cache,
)


_DIRECT_CACHE_SOURCE_INVALID = 0
_DIRECT_CACHE_SOURCE_COMPRESSED = 1
_DIRECT_CACHE_SOURCE_SWA = 2
_TOKEN_FP8_DIM = 448
_TOKEN_BF16_DIM = 64
_TOKEN_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _TOKEN_FP8_DIM + _TOKEN_BF16_DIM * 2
_QUANT_BLOCK_SIZE = 64


@dataclass(frozen=True)
class _DirectCacheRowMap:
    source: torch.Tensor
    physical_block: torch.Tensor
    block_offset: torch.Tensor
    logical_position: torch.Tensor
    length: torch.Tensor


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not all(t.is_cuda for t in tensors):
        raise ValueError("sparse prefill v2 inputs must be CUDA tensors")


def _reference_direct_cache_row_map(
    *,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    compressed_block_table: torch.Tensor,
    swa_block_table: torch.Tensor,
    compressed_block_size: int,
    swa_block_size: int,
    window_size: int,
    compress_ratio: int,
    top_k: int,
) -> _DirectCacheRowMap:
    """Reference direct-cache row mapping used to lock v2 semantics.

    This mirrors `combine_topk_swa_indices()` plus the gather kernel's paged
    cache addressing, but keeps the result split into cache source,
    physical block, block offset, and source-local logical position. It is a
    slow correctness oracle for tests and for the future TileLang kernel.
    """
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must have shape [tokens, topk]")
    num_tokens = topk_indices.shape[0]
    row_capacity = top_k + window_size
    device = topk_indices.device

    source = torch.full(
        (num_tokens, row_capacity),
        _DIRECT_CACHE_SOURCE_INVALID,
        dtype=torch.int32,
        device=device,
    )
    physical_block = torch.full(
        (num_tokens, row_capacity), -1, dtype=torch.int32, device=device
    )
    block_offset = torch.full(
        (num_tokens, row_capacity), -1, dtype=torch.int32, device=device
    )
    logical_position = torch.full(
        (num_tokens, row_capacity), -1, dtype=torch.int32, device=device
    )
    length = torch.empty(num_tokens, dtype=torch.int32, device=device)

    query_base = int(query_start_loc[0].item())
    for req_idx in range(seq_lens.shape[0]):
        query_start = int(query_start_loc[req_idx].item()) - query_base
        query_end = int(query_start_loc[req_idx + 1].item()) - query_base
        query_len = query_end - query_start
        seq_len = int(seq_lens[req_idx].item())
        gather_len = int(gather_lens[req_idx].item())
        gather_start = seq_len - gather_len
        start_pos = seq_len - query_len
        compressed_seq_len = seq_len // compress_ratio

        for token_idx in range(query_start, query_end):
            token_pos = start_pos + token_idx - query_start
            topk_len = min((token_pos + 1) // compress_ratio, top_k)
            swa_len = min(token_pos + 1, window_size)
            length[token_idx] = topk_len + swa_len

            for topk_pos in range(topk_len):
                compressed_pos = int(topk_indices[token_idx, topk_pos].item())
                if compressed_pos < 0 or compressed_pos >= compressed_seq_len:
                    continue
                block_in_seq = compressed_pos // compressed_block_size
                pos_in_block = compressed_pos % compressed_block_size
                physical_block[token_idx, topk_pos] = compressed_block_table[
                    req_idx, block_in_seq
                ]
                block_offset[token_idx, topk_pos] = pos_in_block
                logical_position[token_idx, topk_pos] = compressed_pos
                source[token_idx, topk_pos] = _DIRECT_CACHE_SOURCE_COMPRESSED

            swa_start = token_pos - swa_len + 1
            if swa_start < gather_start:
                raise ValueError(
                    "gather_lens does not cover the requested SWA window"
                )
            for swa_pos in range(swa_len):
                logical_pos = swa_start + swa_pos
                block_in_seq = logical_pos // swa_block_size
                pos_in_block = logical_pos % swa_block_size
                row_pos = topk_len + swa_pos
                physical_block[token_idx, row_pos] = swa_block_table[
                    req_idx, block_in_seq
                ]
                block_offset[token_idx, row_pos] = pos_in_block
                logical_position[token_idx, row_pos] = logical_pos
                source[token_idx, row_pos] = _DIRECT_CACHE_SOURCE_SWA

    return _DirectCacheRowMap(
        source=source,
        physical_block=physical_block,
        block_offset=block_offset,
        logical_position=logical_position,
        length=length,
    )


def _reference_load_fp8_ds_mla_token(
    k_cache: torch.Tensor,
    *,
    physical_block: int,
    block_offset: int,
    block_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference load/dequant for one token in the fp8_ds_mla cache.

    The current paged-cache block layout matches `cache_utils.py`: all token
    data rows are stored first (`block_size * 576` bytes), followed by
    `block_size * 8` scale bytes. This is the layout the TileLang direct-cache
    kernel must read.
    """
    if k_cache.dtype is not torch.uint8:
        raise ValueError("fp8_ds_mla cache must be uint8")

    cache_2d = k_cache.reshape(k_cache.shape[0], -1)
    token_data_offset = block_offset * _TOKEN_DATA_SIZE
    token_scale_offset = block_size * _TOKEN_DATA_SIZE + (
        block_offset * _TOKEN_SCALE_DIM
    )
    fp8_bytes = cache_2d[
        physical_block, token_data_offset : token_data_offset + _TOKEN_FP8_DIM
    ].contiguous()
    bf16_bytes = cache_2d[
        physical_block,
        token_data_offset + _TOKEN_FP8_DIM : token_data_offset + _TOKEN_DATA_SIZE,
    ].contiguous()
    encoded_scales = cache_2d[
        physical_block, token_scale_offset : token_scale_offset + 7
    ].to(torch.float32)

    fp8_values = fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    scales = torch.exp2(encoded_scales - 127.0).repeat_interleave(
        _QUANT_BLOCK_SIZE
    )
    token = torch.empty(
        _TOKEN_FP8_DIM + _TOKEN_BF16_DIM,
        dtype=output_dtype,
        device=k_cache.device,
    )
    token[:_TOKEN_FP8_DIM] = (fp8_values * scales).to(output_dtype)
    token[_TOKEN_FP8_DIM:] = bf16_bytes.view(torch.bfloat16).to(output_dtype)
    return token


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
    num_reqs = int(seq_lens.shape[0])
    compressed_tokens = int(torch.max(seq_lens).item()) // compress_ratio
    m_tokens = compressed_tokens + int(window_size) + int(q.shape[0])
    kv = torch.empty(
        (num_reqs, m_tokens, q.shape[-1]),
        dtype=torch.bfloat16,
        device=q.device,
    )

    dequantize_and_gather_k_cache(
        kv,
        compressed_k_cache,
        seq_lens=seq_lens // compress_ratio,
        gather_lens=None,
        block_table=compressed_block_table,
        block_size=compressed_k_cache.shape[1],
        offset=0,
    )
    dequantize_and_gather_k_cache(
        kv,
        swa_k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=swa_block_table,
        block_size=swa_k_cache.shape[1],
        offset=compressed_tokens,
    )

    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        top_k,
        m_tokens,
        compressed_tokens,
    )
    flash_output, max_logits, lse = (
        tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
            q=q,
            kv=kv.view(-1, 1, q.shape[-1]),
            indices=combined_indices.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=512,
            attn_sink=attn_sink,
            topk_length=combined_lens,
            out=out,
            output_dtype=torch.bfloat16,
        )
    )
    if flash_output.data_ptr() != out.data_ptr() or flash_output.dtype != out.dtype:
        out.copy_(flash_output)
        flash_output = out
    return flash_output, max_logits, lse


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
