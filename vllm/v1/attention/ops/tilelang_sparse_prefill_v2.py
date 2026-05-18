# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.triton_utils import tl, triton
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
_FUSED_KERNEL_CACHE: dict[tuple, object] = {}
_FUSED_KERNEL_FACTORY = None


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
    local_indices = torch.full(
        (num_tokens, 1, total_topk),
        fill_value=-1,
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
        torch.full((), -1, dtype=torch.int32, device=topk_indices.device),
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


@triton.jit
def _triton_gather_selected_fp8_ds_mla_kernel(
    SelectedKV,
    LocalIndices,
    TopkLength,
    CompressedCache,
    SwaCache,
    CompressedBlockTable,
    SwaBlockTable,
    TopkIndices,
    SeqLens,
    GatherLens,
    selected_stride0,
    selected_stride1,
    compressed_block_stride: tl.constexpr,
    swa_block_stride: tl.constexpr,
    topk_stride0: tl.constexpr,
    max_c_blocks: tl.constexpr,
    max_swa_blocks: tl.constexpr,
    num_tokens: tl.constexpr,
    total_query_tokens: tl.constexpr,
    query_token_offset: tl.constexpr,
    total_topk: tl.constexpr,
    top_k: tl.constexpr,
    window_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    compressed_block_size: tl.constexpr,
    swa_block_size: tl.constexpr,
    dim: tl.constexpr,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    token_data_size: tl.constexpr,
    token_scale_dim: tl.constexpr,
    quant_block_size: tl.constexpr,
    block_elems: tl.constexpr,
):
    token_idx = tl.program_id(0)
    selected_idx = tl.program_id(1)
    qblock_idx = tl.program_id(2)
    offsets = tl.arange(0, block_elems)

    seq_len_abs = tl.load(SeqLens)
    token_pos = seq_len_abs - total_query_tokens + query_token_offset + token_idx
    topk_len = tl.minimum((token_pos + 1) // compress_ratio, top_k)
    swa_len = tl.minimum(token_pos + 1, window_size)
    combined_len = topk_len + swa_len
    compressed_seq_len = seq_len_abs // compress_ratio

    is_compressed = selected_idx < topk_len
    topk_pos = tl.minimum(selected_idx, top_k - 1)
    compressed_idx_raw = tl.load(
        TopkIndices + token_idx * topk_stride0 + topk_pos,
        mask=selected_idx < top_k,
        other=0,
    )
    compressed_valid = (
        is_compressed
        & (compressed_idx_raw >= 0)
        & (compressed_idx_raw < compressed_seq_len)
    )
    compressed_idx = tl.where(compressed_valid, compressed_idx_raw, 0)
    c_block_in_seq = compressed_idx // compressed_block_size
    c_pos_in_block = compressed_idx % compressed_block_size
    c_phys = tl.load(
        CompressedBlockTable + c_block_in_seq,
        mask=c_block_in_seq < max_c_blocks,
        other=0,
    )

    swa_offset = selected_idx - topk_len
    swa_valid = (
        (selected_idx >= topk_len)
        & (swa_offset < swa_len)
        & (selected_idx < combined_len)
    )
    swa_logical = tl.where(
        swa_valid,
        token_pos - swa_len + 1 + swa_offset,
        0,
    )
    swa_block_in_seq = swa_logical // swa_block_size
    swa_pos_in_block = swa_logical % swa_block_size
    swa_phys = tl.load(
        SwaBlockTable + swa_block_in_seq,
        mask=swa_block_in_seq < max_swa_blocks,
        other=0,
    )
    row_valid = compressed_valid | swa_valid

    c_token_data_offset = c_pos_in_block * token_data_size
    c_token_scale_offset = (
        compressed_block_size * token_data_size
        + c_pos_in_block * token_scale_dim
    )
    swa_token_data_offset = swa_pos_in_block * token_data_size
    swa_token_scale_offset = (
        swa_block_size * token_data_size
        + swa_pos_in_block * token_scale_dim
    )
    c_block_base = c_phys.to(tl.int64) * compressed_block_stride
    swa_block_base = swa_phys.to(tl.int64) * swa_block_stride

    dim_offsets = qblock_idx * block_elems + offsets
    is_rope = qblock_idx == (fp8_dim // block_elems)
    nope_mask = (dim_offsets < fp8_dim) & (dim_offsets < dim)
    rope_mask = offsets < bf16_dim
    store_offsets = tl.where(is_rope, fp8_dim + offsets, dim_offsets)
    store_mask = row_valid & (store_offsets < dim)
    out_ptr = (
        SelectedKV
        + token_idx * selected_stride0
        + selected_idx * selected_stride1
        + store_offsets
    )

    x_uint8_c = tl.load(
        CompressedCache + c_block_base + c_token_data_offset + dim_offsets,
        mask=nope_mask,
        other=0,
    )
    x_uint8_swa = tl.load(
        SwaCache + swa_block_base + swa_token_data_offset + dim_offsets,
        mask=nope_mask,
        other=0,
    )
    x_uint8 = tl.where(is_compressed, x_uint8_c, x_uint8_swa)
    sign = ((x_uint8 >> 7) & 1).to(tl.int32)
    exp_bits = ((x_uint8 >> 3) & 0xF).to(tl.int32)
    mant_bits = (x_uint8 & 0x7).to(tl.int32)
    fp32_bits = (sign << 31) | ((exp_bits + 120) << 23) | (mant_bits << 20)
    is_zero = (exp_bits == 0) & (mant_bits == 0)
    fp32_bits = tl.where(is_zero, 0, fp32_bits)
    is_subnorm = (exp_bits == 0) & (mant_bits != 0)
    subnorm_val = mant_bits.to(tl.float32) * 1.953125e-3
    subnorm_val = tl.where(sign == 1, -subnorm_val, subnorm_val)
    x_float = tl.where(
        is_subnorm,
        subnorm_val,
        fp32_bits.to(tl.float32, bitcast=True),
    )
    c_scale = tl.load(
        CompressedCache + c_block_base + c_token_scale_offset + qblock_idx,
        mask=qblock_idx < (fp8_dim // quant_block_size),
        other=127,
    )
    swa_scale = tl.load(
        SwaCache + swa_block_base + swa_token_scale_offset + qblock_idx,
        mask=qblock_idx < (fp8_dim // quant_block_size),
        other=127,
    )
    scale_byte = tl.where(is_compressed, c_scale, swa_scale)
    dequant = x_float * tl.exp2(scale_byte.to(tl.float32) - 127.0)

    rope_u16_c = tl.load(
        (CompressedCache + c_block_base + c_token_data_offset + fp8_dim).to(
            tl.pointer_type(tl.uint16)
        )
        + offsets,
        mask=rope_mask,
        other=0,
    )
    rope_u16_swa = tl.load(
        (SwaCache + swa_block_base + swa_token_data_offset + fp8_dim).to(
            tl.pointer_type(tl.uint16)
        )
        + offsets,
        mask=rope_mask,
        other=0,
    )
    rope_u16 = tl.where(is_compressed, rope_u16_c, rope_u16_swa)
    rope_bits = rope_u16.to(tl.uint32) << 16
    rope_val = rope_bits.to(tl.float32, bitcast=True)
    value = tl.where(is_rope, rope_val, dequant)
    tl.store(out_ptr, tl.where(row_valid, value, 0.0), mask=store_mask)

    if qblock_idx == 0:
        local_idx = tl.where(
            row_valid,
            token_idx * total_topk + selected_idx,
            -1,
        )
        tl.store(LocalIndices + token_idx * total_topk + selected_idx, local_idx)
        if selected_idx == 0:
            tl.store(TopkLength + token_idx, combined_len)


def _triton_gather_selected_fp8_ds_mla_cache(
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
    total_query_tokens: int | None = None,
    query_token_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del query_start_loc
    if seq_lens.numel() != 1:
        raise NotImplementedError(
            "selected direct-cache gather currently supports one prefill "
            "request per launch"
        )
    if dim != _TOKEN_FP8_DIM + _TOKEN_BF16_DIM:
        raise ValueError("selected direct-cache gather currently supports dim=512")
    if output_dtype is not torch.float16:
        raise ValueError("Triton selected direct-cache gather emits fp16")
    if topk_indices.ndim != 2 or topk_indices.shape[1] != top_k:
        raise ValueError("topk_indices must have shape [tokens, top_k]")

    num_tokens = topk_indices.shape[0]
    if total_query_tokens is None:
        total_query_tokens = num_tokens
    if total_query_tokens < num_tokens:
        raise ValueError("total_query_tokens must cover the sliced token count")
    if query_token_offset < 0:
        raise ValueError("query_token_offset must be non-negative")
    if query_token_offset + num_tokens > total_query_tokens:
        raise ValueError(
            "query_token_offset + sliced tokens exceeds total_query_tokens"
        )
    selected_kv = torch.zeros(
        (num_tokens, total_topk, dim),
        dtype=output_dtype,
        device=topk_indices.device,
    )
    local_indices = torch.empty(
        (num_tokens, 1, total_topk),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    topk_length = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    grid = (num_tokens, total_topk, 8)
    _triton_gather_selected_fp8_ds_mla_kernel[grid](
        selected_kv,
        local_indices,
        topk_length,
        compressed_k_cache,
        swa_k_cache,
        compressed_block_table,
        swa_block_table,
        topk_indices,
        seq_lens,
        gather_lens,
        selected_kv.stride(0),
        selected_kv.stride(1),
        compressed_block_stride=compressed_k_cache.stride(0),
        swa_block_stride=swa_k_cache.stride(0),
        topk_stride0=topk_indices.stride(0),
        max_c_blocks=compressed_block_table.shape[-1],
        max_swa_blocks=swa_block_table.shape[-1],
        num_tokens=num_tokens,
        total_query_tokens=total_query_tokens,
        query_token_offset=query_token_offset,
        total_topk=total_topk,
        top_k=top_k,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compressed_block_size=compressed_k_cache.shape[1],
        swa_block_size=swa_k_cache.shape[1],
        dim=dim,
        fp8_dim=_TOKEN_FP8_DIM,
        bf16_dim=_TOKEN_BF16_DIM,
        token_data_size=_TOKEN_DATA_SIZE,
        token_scale_dim=_TOKEN_SCALE_DIM,
        quant_block_size=_QUANT_BLOCK_SIZE,
        block_elems=_QUANT_BLOCK_SIZE,
    )
    return selected_kv, local_indices, topk_length


def _selected_kv_chunk_tokens(
    *,
    num_tokens: int,
    total_topk: int,
    dim: int,
    dtype: torch.dtype,
    max_chunk_mb: int,
) -> int:
    if num_tokens <= 0:
        return 0
    if max_chunk_mb <= 0:
        return num_tokens
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_token = max(1, total_topk * dim * element_size)
    max_bytes = max_chunk_mb * 1024 * 1024
    return max(1, min(num_tokens, max_bytes // bytes_per_token))


def _build_fused_kernel_factory():
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
    def build_fused_sparse_prefill_kernel(
        heads: int,
        dim: int,
        top_k: int,
        window_size: int,
        compress_ratio: int,
        compressed_block_size: int,
        swa_block_size: int,
        sm_scale: float,
        block_I: int = 16,
        num_stages: int = 1,
        threads: int = 128,
        has_sink: bool = True,
        output_dtype_str: str = "float16",
    ):
        assert dim == 512
        assert heads == 64
        total_topk = ((top_k + window_size + block_I - 1) // block_I) * block_I
        assert total_topk % block_I == 0

        LOG2E = 1.4426950408889634
        sm_scale_log2 = sm_scale * LOG2E

        seq_len = T.dynamic("seq_len")
        num_c_blocks = T.dynamic("num_c_blocks")
        c_block_bytes = T.dynamic("c_block_bytes")
        num_swa_blocks = T.dynamic("num_swa_blocks")
        swa_block_bytes = T.dynamic("swa_block_bytes")
        max_c_blocks = T.dynamic("max_c_blocks")
        max_swa_blocks = T.dynamic("max_swa_blocks")

        q_shape = [seq_len, heads, dim]
        cache_c_shape = [num_c_blocks, c_block_bytes]
        cache_swa_shape = [num_swa_blocks, swa_block_bytes]
        c_table_shape = [max_c_blocks]
        swa_table_shape = [max_swa_blocks]
        topk_shape = [seq_len, top_k]
        scalar_shape = [1]
        sink_shape = [heads]
        output_shape = [seq_len, heads, dim]
        stats_shape = [seq_len, heads]
        dtype = T.float16
        out_dtype = T.bfloat16 if output_dtype_str == "bfloat16" else T.float16
        accum_dtype = T.float32
        _shape_refs = (
            q_shape, cache_c_shape, cache_swa_shape, c_table_shape,
            swa_table_shape, topk_shape, scalar_shape, sink_shape,
            output_shape, stats_shape,
        )

        H = heads
        D = dim
        BI = block_I
        NI = tilelang.cdiv(total_topk, block_I)

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            CompressedCache: T.Tensor(cache_c_shape, T.uint8),
            SwaCache: T.Tensor(cache_swa_shape, T.uint8),
            CompressedBlockTable: T.Tensor(c_table_shape, T.int32),
            SwaBlockTable: T.Tensor(swa_table_shape, T.int32),
            TopkIndices: T.Tensor(topk_shape, T.int32),
            SeqLens: T.Tensor(scalar_shape, T.int32),
            GatherLens: T.Tensor(scalar_shape, T.int32),
            Sink: T.Tensor(sink_shape, accum_dtype),
            Output: T.Tensor(output_shape, out_dtype),
            MaxLogits: T.Tensor(stats_shape, accum_dtype),
            Lse: T.Tensor(stats_shape, accum_dtype),
        ):
            with T.Kernel(seq_len, threads=threads) as bx:
                Q_shared = T.alloc_shared([H, D], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H, D], accum_dtype)
                acc_s = T.alloc_fragment([H, BI], accum_dtype)
                S_shared = T.alloc_shared([H, BI], dtype)
                sumexp = T.alloc_fragment([H], accum_dtype)
                sumexp_i = T.alloc_fragment([H], accum_dtype)
                alpha = T.alloc_fragment([H], accum_dtype)
                m_i = T.alloc_fragment([H], accum_dtype)
                m_i_prev = T.alloc_fragment([H], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2 ** 30))

                s_i = bx
                seq_len_abs = SeqLens[0]
                token_pos = seq_len_abs - seq_len + s_i
                topk_len = T.min((token_pos + 1) // compress_ratio, top_k)
                swa_len = T.min(token_pos + 1, window_size)
                combined_len = topk_len + swa_len
                compressed_seq_len = seq_len_abs // compress_ratio

                T.copy(Q[s_i, 0:H, 0:D], Q_shared)

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        pos = i_i * BI + bi_i
                        is_compressed = pos < topk_len
                        topk_pos = T.min(pos, top_k - 1)
                        compressed_idx_raw = TopkIndices[s_i, topk_pos]
                        compressed_valid = (
                            is_compressed
                            & (compressed_idx_raw >= 0)
                            & (compressed_idx_raw < compressed_seq_len)
                        )
                        compressed_idx = T.if_then_else(
                            compressed_valid, compressed_idx_raw, 0)
                        c_block_in_seq = compressed_idx // compressed_block_size
                        c_pos_in_block = compressed_idx % compressed_block_size
                        c_phys = CompressedBlockTable[c_block_in_seq]

                        swa_offset = pos - topk_len
                        swa_valid = (
                            (pos >= topk_len)
                            & (swa_offset < swa_len)
                            & (pos < combined_len)
                        )
                        swa_logical = T.if_then_else(
                            swa_valid,
                            token_pos - swa_len + 1 + swa_offset,
                            0,
                        )
                        swa_block_in_seq = swa_logical // swa_block_size
                        swa_pos_in_block = swa_logical % swa_block_size
                        swa_phys = SwaBlockTable[swa_block_in_seq]

                        mask[bi_i] = compressed_valid | swa_valid

                    for bi_i, d_i in T.Parallel(BI, D):
                        pos = i_i * BI + bi_i
                        is_compressed = pos < topk_len
                        topk_pos = T.min(pos, top_k - 1)
                        compressed_idx_raw = TopkIndices[s_i, topk_pos]
                        compressed_valid = (
                            is_compressed
                            & (compressed_idx_raw >= 0)
                            & (compressed_idx_raw < compressed_seq_len)
                        )
                        compressed_idx = T.if_then_else(
                            compressed_valid, compressed_idx_raw, 0)
                        c_block_in_seq = compressed_idx // compressed_block_size
                        c_pos_in_block = compressed_idx % compressed_block_size
                        c_phys = CompressedBlockTable[c_block_in_seq]

                        swa_offset = pos - topk_len
                        swa_valid = (
                            (pos >= topk_len)
                            & (swa_offset < swa_len)
                            & (pos < combined_len)
                        )
                        swa_logical = T.if_then_else(
                            swa_valid,
                            token_pos - swa_len + 1 + swa_offset,
                            0,
                        )
                        swa_block_in_seq = swa_logical // swa_block_size
                        swa_pos_in_block = swa_logical % swa_block_size
                        swa_phys = SwaBlockTable[swa_block_in_seq]

                        c_token_data_offset = c_pos_in_block * _TOKEN_DATA_SIZE
                        swa_token_data_offset = (
                            swa_pos_in_block * _TOKEN_DATA_SIZE)
                        token_scale_offset_c = (
                            compressed_block_size * _TOKEN_DATA_SIZE
                            + c_pos_in_block * _TOKEN_SCALE_DIM
                        )
                        token_scale_offset_swa = (
                            swa_block_size * _TOKEN_DATA_SIZE
                            + swa_pos_in_block * _TOKEN_SCALE_DIM
                        )

                        x_uint8_c = CompressedCache[
                            c_phys, c_token_data_offset + d_i
                        ]
                        x_uint8_swa = SwaCache[
                            swa_phys, swa_token_data_offset + d_i
                        ]
                        x_uint8 = T.if_then_else(
                            compressed_valid, x_uint8_c, x_uint8_swa)
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
                            sign_mask == 1, -subnorm_val, subnorm_val)
                        x_float = T.if_then_else(
                            is_subnorm, subnorm_val, normal_val)
                        scale_idx = d_i // _QUANT_BLOCK_SIZE
                        scale_c = CompressedCache[
                            c_phys, token_scale_offset_c + scale_idx
                        ]
                        scale_swa = SwaCache[
                            swa_phys, token_scale_offset_swa + scale_idx
                        ]
                        scale_byte = T.if_then_else(
                            compressed_valid, scale_c, scale_swa)
                        dequant = (
                            x_float
                            * T.exp2(T.Cast(T.float32, scale_byte) - 127.0)
                        )

                        rope_i = T.if_then_else(
                            d_i >= _TOKEN_FP8_DIM, d_i - _TOKEN_FP8_DIM, 0)
                        byte_offset = (
                            c_token_data_offset
                            + _TOKEN_FP8_DIM
                            + rope_i * 2
                        )
                        lo_c = T.Cast(T.int32, CompressedCache[
                            c_phys, byte_offset])
                        hi_c = T.Cast(T.int32, CompressedCache[
                            c_phys, byte_offset + 1])
                        swa_byte_offset = (
                            swa_token_data_offset
                            + _TOKEN_FP8_DIM
                            + rope_i * 2
                        )
                        lo_swa = T.Cast(T.int32, SwaCache[
                            swa_phys, swa_byte_offset])
                        hi_swa = T.Cast(T.int32, SwaCache[
                            swa_phys, swa_byte_offset + 1])
                        lo = T.if_then_else(compressed_valid, lo_c, lo_swa)
                        hi = T.if_then_else(compressed_valid, hi_c, hi_swa)
                        bf16_u16 = lo | (hi << 8)
                        rope_val = T.reinterpret(bf16_u16 << 16, "float32")
                        val = T.if_then_else(
                            d_i < _TOKEN_FP8_DIM, dequant, rope_val)
                        KV_shared[bi_i, d_i] = T.if_then_else(
                            mask[bi_i], T.Cast(dtype, val), T.Cast(dtype, 0.0))

                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        Q_shared, KV_shared, acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H):
                        alpha[h_i] = T.exp2(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale_log2)
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale_log2
                            - m_i[h_i] * sm_scale_log2)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H):
                        sumexp[h_i] = (
                            sumexp[h_i] * alpha[h_i] + sumexp_i[h_i])
                    for h_i, d_i in T.Parallel(H, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(
                        S_shared, KV_shared, acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for h_i in T.Parallel(H):
                    MaxLogits[s_i, h_i] = m_i[h_i] * sm_scale
                for h_i in T.Parallel(H):
                    Lse[s_i, h_i] = (
                        T.log2(sumexp[h_i]) / LOG2E
                        + m_i[h_i] * sm_scale
                    )
                if has_sink:
                    for h_i in T.Parallel(H):
                        extra = T.exp2(
                            (Sink[h_i] - m_i[h_i] * sm_scale) * LOG2E)
                        sumexp[h_i] = sumexp[h_i] + extra
                for h_i, d_i in T.Parallel(H, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] / sumexp[h_i]
                if output_dtype_str == "bfloat16":
                    for h_i, d_i in T.Parallel(H, D):
                        Output[s_i, h_i, d_i] = T.Cast(
                            out_dtype, acc_o[h_i, d_i])
                else:
                    T.copy(acc_o, Output[s_i, 0:H, 0:D])

        return main

    return build_fused_sparse_prefill_kernel


def _get_fused_kernel(
    *,
    heads: int,
    dim: int,
    top_k: int,
    window_size: int,
    compress_ratio: int,
    compressed_block_size: int,
    swa_block_size: int,
    sm_scale: float,
    block_I: int,
    num_stages: int,
    threads: int,
    has_sink: bool,
    output_dtype_str: str,
):
    global _FUSED_KERNEL_FACTORY
    if _FUSED_KERNEL_FACTORY is None:
        _FUSED_KERNEL_FACTORY = _build_fused_kernel_factory()
    key = (
        heads,
        dim,
        top_k,
        window_size,
        compress_ratio,
        compressed_block_size,
        swa_block_size,
        sm_scale,
        block_I,
        num_stages,
        threads,
        has_sink,
        output_dtype_str,
    )
    if key not in _FUSED_KERNEL_CACHE:
        _FUSED_KERNEL_CACHE[key] = _FUSED_KERNEL_FACTORY(
            heads=heads,
            dim=dim,
            top_k=top_k,
            window_size=window_size,
            compress_ratio=compress_ratio,
            compressed_block_size=compressed_block_size,
            swa_block_size=swa_block_size,
            sm_scale=sm_scale,
            block_I=block_I,
            num_stages=num_stages,
            threads=threads,
            has_sink=has_sink,
            output_dtype_str=output_dtype_str,
        )
    return _FUSED_KERNEL_CACHE[key]


def _flash_mla_sparse_prefill_v2_fused_tilelang(
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
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    block_I: int = 16,
    num_stages: int = 1,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Experimental monolithic fused direct-cache sparse prefill kernel.

    This is intentionally not routed from production.  It keeps the desired
    public shape for the real fused kernel, but the current TileLang lowering
    path is too slow/unstable when row mapping, fp8 dequant, QK, softmax, and PV
    all live in one SM70 kernel.  Keep it behind the explicit kernel test while
    the implementation is split into smaller stages or replaced by CUDA.
    """
    del query_start_loc, gather_lens
    if q.ndim != 3 or tuple(q.shape[1:]) != (64, 512):
        raise ValueError("fused v2 supports q [tokens, 64, 512]")
    if q.dtype is not torch.float16 or out.dtype is not torch.float16:
        raise ValueError("fused v2 starts with fp16 q/out only")
    if out.shape != q.shape:
        raise ValueError("out must match q shape for fused v2")
    if seq_lens.numel() != 1:
        raise NotImplementedError("fused v2 currently supports one request")
    if topk_indices.shape != (q.shape[0], top_k):
        raise ValueError("topk_indices must have shape [tokens, top_k]")

    compressed_cache_2d = compressed_k_cache.reshape(
        compressed_k_cache.shape[0], -1).contiguous()
    swa_cache_2d = swa_k_cache.reshape(swa_k_cache.shape[0], -1).contiguous()
    compressed_table = compressed_block_table.reshape(-1).to(
        torch.int32).contiguous()
    swa_table = swa_block_table.reshape(-1).to(torch.int32).contiguous()
    sink = (
        attn_sink.to(torch.float32).contiguous()
        if attn_sink is not None
        else torch.zeros(q.shape[1], dtype=torch.float32, device=q.device)
    )

    kernel = _get_fused_kernel(
        heads=q.shape[1],
        dim=q.shape[-1],
        top_k=top_k,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compressed_block_size=compressed_k_cache.shape[1],
        swa_block_size=swa_k_cache.shape[1],
        sm_scale=sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
        has_sink=attn_sink is not None,
        output_dtype_str="float16",
    )
    output, max_logits, lse = kernel(
        q.contiguous(),
        compressed_cache_2d,
        swa_cache_2d,
        compressed_table,
        swa_table,
        topk_indices.to(torch.int32).contiguous(),
        seq_lens.to(torch.int32).contiguous(),
        torch.empty(1, dtype=torch.int32, device=q.device),
        sink,
    )
    out.copy_(output)
    return out, max_logits, lse


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

    num_tokens = q.shape[0]
    total_topk = ((top_k + window_size + block_I - 1) // block_I) * block_I
    chunk_tokens = _selected_kv_chunk_tokens(
        num_tokens=num_tokens,
        total_topk=total_topk,
        dim=q.shape[-1],
        dtype=torch.float16,
        max_chunk_mb=envs.VLLM_SM70_SPARSE_PREFILL_V2_SELECTED_KV_CHUNK_MB,
    )
    max_logits_parts: list[torch.Tensor] = []
    lse_parts: list[torch.Tensor] = []

    for row_start in range(0, num_tokens, chunk_tokens):
        row_end = min(row_start + chunk_tokens, num_tokens)
        q_chunk = q[row_start:row_end]
        out_chunk = out[row_start:row_end]
        selected_kv, local_indices, topk_length = (
            _triton_gather_selected_fp8_ds_mla_cache(
                compressed_k_cache=compressed_k_cache,
                swa_k_cache=swa_k_cache,
                compressed_block_table=compressed_block_table,
                swa_block_table=swa_block_table,
                topk_indices=topk_indices[row_start:row_end],
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                window_size=window_size,
                compress_ratio=compress_ratio,
                top_k=top_k,
                total_topk=total_topk,
                dim=q.shape[-1],
                output_dtype=torch.float16,
                total_query_tokens=num_tokens,
                query_token_offset=row_start,
            )
        )
        flash_output, max_logits, lse = (
            tilelang_sparse_prefill.flash_mla_sparse_fwd_tilelang(
                q=q_chunk,
                kv=selected_kv.view(-1, 1, q.shape[-1]),
                indices=local_indices,
                sm_scale=sm_scale,
                d_v=512,
                attn_sink=attn_sink,
                topk_length=topk_length,
                out=out_chunk,
                output_dtype=torch.float16,
                block_I=block_I,
            )
        )
        if (
            flash_output.data_ptr() != out_chunk.data_ptr()
            or flash_output.dtype != out_chunk.dtype
        ):
            out_chunk.copy_(flash_output)
        max_logits_parts.append(max_logits)
        lse_parts.append(lse)

    return out, torch.cat(max_logits_parts, dim=0), torch.cat(lse_parts, dim=0)


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
    if q.ndim != 3 or q.shape[1] != 64 or q.shape[-1] not in (512, 576):
        raise ValueError("q must have shape [tokens, 64, 512 or 576]")
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

    if q.shape[-1] == 512:
        return _flash_mla_sparse_prefill_v2_direct_cache(
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
