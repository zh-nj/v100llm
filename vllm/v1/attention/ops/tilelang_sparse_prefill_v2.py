# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


_DEBUG_LOAD_KERNEL_CACHE: dict[tuple[int, str, int], object] = {}
_DEBUG_LOAD_KERNEL_FACTORY = None


def _build_debug_load_kernel_factory():
    import tilelang
    from tilelang import language as T

    @tilelang.jit(
        out_idx=[-1],
        target="cuda -arch=sm_70",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def build_debug_load_kernel(
        block_size: int,
        output_dtype_str: str = "float16",
        threads: int = 128,
    ):
        num_tokens = T.dynamic("num_tokens")
        num_blocks = T.dynamic("num_blocks")
        block_bytes = T.dynamic("block_bytes")

        cache_shape = [num_blocks, block_bytes]
        row_shape = [num_tokens]
        out_shape = [num_tokens, _TOKEN_FP8_DIM + _TOKEN_BF16_DIM]
        out_dtype = (
            T.bfloat16 if output_dtype_str == "bfloat16" else T.float16
        )
        _shape_refs = (cache_shape, row_shape, out_shape)

        @T.prim_func
        def main(
            KCache: T.Tensor(cache_shape, T.uint8),
            PhysicalBlocks: T.Tensor(row_shape, T.int32),
            BlockOffsets: T.Tensor(row_shape, T.int32),
            Output: T.Tensor(out_shape, out_dtype),
        ):
            with T.Kernel(num_tokens, threads=threads) as bx:
                physical_block = PhysicalBlocks[bx]
                block_offset = BlockOffsets[bx]
                token_data_offset = block_offset * _TOKEN_DATA_SIZE
                token_scale_offset = (
                    block_size * _TOKEN_DATA_SIZE
                    + block_offset * _TOKEN_SCALE_DIM
                )

                for d_i in T.Parallel(_TOKEN_FP8_DIM + _TOKEN_BF16_DIM):
                    x_uint8 = KCache[
                        physical_block, token_data_offset + d_i
                    ]
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
                        sign_mask == 1, -subnorm_val, subnorm_val
                    )
                    x_float = T.if_then_else(
                        is_subnorm, subnorm_val, normal_val
                    )

                    scale_byte = KCache[
                        physical_block,
                        token_scale_offset + d_i // _QUANT_BLOCK_SIZE,
                    ]
                    scale = T.exp2(T.Cast(T.float32, scale_byte) - 127.0)
                    rope_i = T.if_then_else(
                        d_i >= _TOKEN_FP8_DIM, d_i - _TOKEN_FP8_DIM, 0
                    )
                    byte_offset = (
                        token_data_offset
                        + _TOKEN_FP8_DIM
                        + rope_i * 2
                    )
                    lo = T.Cast(
                        T.int32, KCache[physical_block, byte_offset]
                    )
                    hi = T.Cast(
                        T.int32, KCache[physical_block, byte_offset + 1]
                    )
                    bf16_u16 = lo | (hi << 8)
                    fp32_bits = bf16_u16 << 16
                    rope_val = T.reinterpret(fp32_bits, "float32")
                    Output[bx, d_i] = T.if_then_else(
                        d_i < _TOKEN_FP8_DIM,
                        T.Cast(out_dtype, x_float * scale),
                        T.Cast(out_dtype, rope_val),
                    )

        return main

    return build_debug_load_kernel


def _get_debug_load_kernel(
    *,
    block_size: int,
    output_dtype: torch.dtype,
    threads: int = 128,
):
    global _DEBUG_LOAD_KERNEL_FACTORY
    if _DEBUG_LOAD_KERNEL_FACTORY is None:
        _DEBUG_LOAD_KERNEL_FACTORY = _build_debug_load_kernel_factory()
    output_dtype_str = (
        "bfloat16" if output_dtype is torch.bfloat16 else "float16"
    )
    key = (block_size, output_dtype_str, threads)
    if key not in _DEBUG_LOAD_KERNEL_CACHE:
        _DEBUG_LOAD_KERNEL_CACHE[key] = _DEBUG_LOAD_KERNEL_FACTORY(
            block_size=block_size,
            output_dtype_str=output_dtype_str,
            threads=threads,
        )
    return _DEBUG_LOAD_KERNEL_CACHE[key]


def _tilelang_debug_load_fp8_ds_mla_tokens(
    k_cache: torch.Tensor,
    *,
    physical_blocks: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if k_cache.dtype is not torch.uint8:
        raise ValueError("fp8_ds_mla cache must be uint8")
    if not k_cache.is_cuda:
        raise ValueError("TileLang debug load requires CUDA cache")
    k_cache_2d = k_cache.reshape(k_cache.shape[0], -1).contiguous()
    kernel = _get_debug_load_kernel(
        block_size=block_size,
        output_dtype=output_dtype,
    )
    result = kernel(
        k_cache_2d,
        physical_blocks.to(torch.int32).contiguous(),
        block_offsets.to(torch.int32).contiguous(),
    )
    return result[0] if isinstance(result, tuple) else result


def _tilelang_gather_selected_fp8_ds_mla_cache(
    *,
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
    total_topk: int,
    dim: int,
    output_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_lens.numel() != 1:
        raise NotImplementedError(
            "selected direct-cache gather currently supports one prefill "
            "request per launch"
        )
    if dim != _TOKEN_FP8_DIM + _TOKEN_BF16_DIM:
        raise ValueError("selected direct-cache gather currently supports dim=512")
    if topk_indices.ndim != 2 or topk_indices.shape[1] != top_k:
        raise ValueError("topk_indices must have shape [tokens, top_k]")

    row_map = _reference_direct_cache_row_map(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        compressed_block_table=compressed_block_table,
        swa_block_table=swa_block_table,
        compressed_block_size=compressed_k_cache.shape[1],
        swa_block_size=swa_k_cache.shape[1],
        window_size=window_size,
        compress_ratio=compress_ratio,
        top_k=top_k,
    )
    num_tokens = topk_indices.shape[0]
    selected_kv = torch.zeros(
        (num_tokens, total_topk, dim),
        dtype=output_dtype,
        device=topk_indices.device,
    )
    local_indices = torch.zeros(
        (num_tokens, 1, total_topk),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    arange = torch.arange(
        num_tokens * total_topk,
        dtype=torch.int32,
        device=topk_indices.device,
    ).view(num_tokens, total_topk)
    row_capacity = row_map.source.shape[1]
    valid_span = min(row_capacity, total_topk)
    valid = row_map.source[:, :valid_span] != _DIRECT_CACHE_SOURCE_INVALID
    local_indices[:, 0, :valid_span] = torch.where(
        valid,
        arange[:, :valid_span],
        torch.zeros((), dtype=torch.int32, device=topk_indices.device),
    )

    compressed_mask = (
        row_map.source[:, :valid_span] == _DIRECT_CACHE_SOURCE_COMPRESSED
    )
    selected_flat = selected_kv.view(num_tokens * total_topk, dim)
    if bool(compressed_mask.any().item()):
        token_idx, selected_idx = compressed_mask.nonzero(as_tuple=True)
        loaded = _tilelang_debug_load_fp8_ds_mla_tokens(
            compressed_k_cache,
            physical_blocks=row_map.physical_block[:, :valid_span][
                compressed_mask
            ],
            block_offsets=row_map.block_offset[:, :valid_span][compressed_mask],
            block_size=compressed_k_cache.shape[1],
            output_dtype=output_dtype,
        )
        flat_idx = (token_idx * total_topk + selected_idx).to(torch.int64)
        selected_flat.index_copy_(0, flat_idx, loaded.contiguous())

    swa_mask = row_map.source[:, :valid_span] == _DIRECT_CACHE_SOURCE_SWA
    if bool(swa_mask.any().item()):
        token_idx, selected_idx = swa_mask.nonzero(as_tuple=True)
        loaded = _tilelang_debug_load_fp8_ds_mla_tokens(
            swa_k_cache,
            physical_blocks=row_map.physical_block[:, :valid_span][swa_mask],
            block_offsets=row_map.block_offset[:, :valid_span][swa_mask],
            block_size=swa_k_cache.shape[1],
            output_dtype=output_dtype,
        )
        flat_idx = (token_idx * total_topk + selected_idx).to(torch.int64)
        selected_flat.index_copy_(0, flat_idx, loaded.contiguous())

    topk_length = row_map.length.to(torch.int32).contiguous()
    return selected_kv, local_indices, topk_length


def _flash_mla_sparse_prefill_v2_direct_cache(
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
    block_I: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or q.shape[1] != 64 or q.shape[-1] != 512:
        raise ValueError("direct-cache v2 first kernel supports q [tokens, 64, 512]")
    if out.shape != q.shape:
        raise ValueError("out must match q shape for direct-cache v2 first kernel")
    if topk_indices.shape != (q.shape[0], top_k):
        raise ValueError("topk_indices must have shape [tokens, top_k]")

    total_topk = ((top_k + window_size + block_I - 1) // block_I) * block_I
    selected_kv, local_indices, topk_length = (
        _tilelang_gather_selected_fp8_ds_mla_cache(
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
            total_topk=total_topk,
            dim=q.shape[-1],
            output_dtype=torch.float16,
        )
    )
    flash_output, max_logits, lse = (
        tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
            q=q,
            kv=selected_kv.view(-1, 1, q.shape[-1]),
            indices=local_indices,
            sm_scale=sm_scale,
            d_v=512,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
            output_dtype=torch.float16,
            block_I=block_I,
        )
    )
    if flash_output.data_ptr() != out.data_ptr() or flash_output.dtype != out.dtype:
        out.copy_(flash_output)
        flash_output = out
    return flash_output, max_logits, lse


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
