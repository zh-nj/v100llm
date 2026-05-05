# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.utils import cublas_gemm_bf16_bf16_fp32
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


def _should_use_torch_fused_compressor_fallback(
    kv_cache: torch.Tensor,
    use_fp4_cache: bool,
) -> bool:
    if use_fp4_cache or not kv_cache.is_cuda:
        return False
    return torch.cuda.get_device_capability(kv_cache.device)[0] < 8


def _normalize_sm70_fp8_cache_exponents(exponents: torch.Tensor) -> torch.Tensor:
    return exponents


def _apply_gptj_rope_tail_1d(
    x: torch.Tensor,
    position: int,
    compress_ratio: int,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    nope_head_dim = x.shape[-1] - rope_head_dim
    compressed_pos = (position // compress_ratio) * compress_ratio
    cos_sin = cos_sin_cache[compressed_pos].float()
    half = rope_head_dim // 2

    out = x.clone()
    rope = out[nope_head_dim:]
    even = rope[::2]
    odd = rope[1::2]
    rotated = torch.empty_like(rope)
    rotated[::2] = even * cos_sin[:half] - odd * cos_sin[half:]
    rotated[1::2] = odd * cos_sin[:half] + even * cos_sin[half:]
    out[nope_head_dim:] = rotated
    return out


def _store_fp8_sparse_attention_cache_torch(
    kv_cache_2d: torch.Tensor,
    kv_block_idx: int,
    kv_pos_in_block: int,
    kv_cache_block_size: int,
    token_stride: int,
    scale_dim: int,
    normed: torch.Tensor,
    rotated: torch.Tensor,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
) -> None:
    nope_head_dim = normed.shape[-1] - rope_head_dim
    block_base = kv_block_idx
    token_data_offset = kv_pos_in_block * token_stride
    token_scale_offset = kv_cache_block_size * token_stride + (
        kv_pos_in_block * scale_dim
    )

    quant_input = normed.to(torch.bfloat16).float()[:nope_head_dim]
    blocks = quant_input.view(nope_head_dim // quant_block, quant_block)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    exponents = _normalize_sm70_fp8_cache_exponents(
        torch.ceil(torch.log2(absmax / fp8_max))
    )
    scales = torch.exp2(exponents)
    fp8_bytes = (
        (blocks / scales)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
        .view(-1)
    )
    rope_bytes = (
        rotated[nope_head_dim:]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .view(-1)
    )
    token_data = torch.cat((fp8_bytes, rope_bytes))
    kv_cache_2d[
        block_base,
        token_data_offset:token_data_offset + token_data.shape[0],
    ] = token_data

    encoded_scales = (exponents.flatten() + 127.0).clamp(0, 255).to(torch.uint8)
    scale_data = torch.zeros(scale_dim, dtype=torch.uint8, device=kv_cache_2d.device)
    scale_data[:encoded_scales.shape[0]] = encoded_scales
    kv_cache_2d[
        block_base,
        token_scale_offset:token_scale_offset + scale_dim,
    ] = scale_data


def _store_fp8_indexer_cache_torch(
    kv_cache_2d: torch.Tensor,
    kv_block_idx: int,
    kv_pos_in_block: int,
    kv_cache_block_size: int,
    token_stride: int,
    scale_dim: int,
    rotated: torch.Tensor,
    fp8_max: float,
) -> None:
    token_data_offset = kv_pos_in_block * token_stride
    token_scale_offset = kv_cache_block_size * token_stride + (
        kv_pos_in_block * scale_dim
    )
    quant_input = rotated.to(torch.bfloat16).float()
    absmax = quant_input.abs().amax().clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(absmax / fp8_max))
    scale = torch.exp2(exponent)
    fp8_bytes = (
        (quant_input / scale)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
    )
    kv_cache_2d[
        kv_block_idx,
        token_data_offset:token_data_offset + token_stride,
    ] = fp8_bytes
    kv_cache_2d[
        kv_block_idx,
        token_scale_offset:token_scale_offset + scale_dim,
    ] = scale.reshape(1).to(torch.float32).contiguous().view(torch.uint8)


def _torch_fused_compress_norm_rope_insert_fp8_fallback(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Torch correctness fallback for SM70, where Triton fp8e4nv is invalid."""
    kv_cache_2d = kv_cache.reshape(kv_cache.shape[0], -1)
    window = (1 + int(overlap)) * compress_ratio
    device = state_cache.device

    for token_idx in range(slot_mapping.shape[0]):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue

        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue

        kv_slot_idx = int(kv_slot_mapping[token_idx].item())
        if kv_slot_idx < 0:
            continue

        req_idx = int(token_to_req_indices[token_idx].item())
        start = position - window + 1
        kv_rows: list[torch.Tensor] = []
        score_rows: list[torch.Tensor] = []
        for local_idx in range(window):
            pos = start + local_idx
            if pos < 0:
                kv_rows.append(torch.zeros(head_size, device=device))
                score_rows.append(
                    torch.full((head_size, ), float("-inf"), device=device)
                )
                continue

            block_in_seq = pos // block_size
            pos_in_block = pos % block_size
            physical_block = int(block_table[req_idx, block_in_seq].item())
            head_offset = int(local_idx >= compress_ratio) * head_size
            row = state_cache[physical_block, pos_in_block]
            kv_rows.append(row[head_offset:head_offset + head_size].float())
            score_rows.append(
                row[
                    state_width + head_offset:
                    state_width + head_offset + head_size
                ].float()
            )

        score = torch.stack(score_rows).softmax(dim=0)
        kv = torch.stack(kv_rows)
        compressed_kv = (kv * score).sum(dim=0)
        variance = compressed_kv.pow(2).sum() / head_size
        normed = compressed_kv * torch.rsqrt(variance + rms_norm_eps)
        normed = normed * rms_norm_weight.float()
        rotated = _apply_gptj_rope_tail_1d(
            normed,
            position,
            compress_ratio,
            cos_sin_cache,
            rope_head_dim,
        )

        kv_block_idx = kv_slot_idx // kv_cache_block_size
        kv_pos_in_block = kv_slot_idx % kv_cache_block_size
        if head_size == 512:
            _store_fp8_sparse_attention_cache_torch(
                kv_cache_2d,
                kv_block_idx,
                kv_pos_in_block,
                kv_cache_block_size,
                token_stride,
                scale_dim,
                normed,
                rotated,
                rope_head_dim,
                fp8_max,
                quant_block,
            )
        elif head_size == 128:
            _store_fp8_indexer_cache_torch(
                kv_cache_2d,
                kv_block_idx,
                kv_pos_in_block,
                kv_cache_block_size,
                token_stride,
                scale_dim,
                rotated,
                fp8_max,
            )
        else:
            raise RuntimeError(
                f"Unsupported DeepSeek compressor fallback head size: {head_size}"
            )


# =============================================================================
# SM70 Triton kernel: Fused compress → RMSNorm → RoPE → FP8 quant → cache write
# Replaces _torch_fused_compress_norm_rope_insert_fp8_fallback with a Triton
# kernel that processes only tokens at compress-ratio boundaries (pre-filtered
# on the Python side via firing_indices).
# Grid: (num_firing_tokens,) — one program per boundary token.
# =============================================================================


@triton.jit
def _sm70_fused_compress_norm_rope_insert_fp8_kernel(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── firing indices (pre-filtered boundary tokens) ──
    firing_indices_ptr,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    IS_INDEXER: tl.constexpr,  # True for head=128 indexer path
):
    """SM70 fused compress → RMSNorm → RoPE → FP8 quant → cache write.

    Pre-filtered version: only launched for tokens at compress boundaries.
    Each program reads its actual token index from firing_indices.

    For IS_INDEXER=False (head=512 sparse attention path):
      - NoPE (448) encoded as 7 × 64-element FP8 blocks + UE8M0 scales
      - RoPE (64) stored as BF16
    For IS_INDEXER=True (head=128 indexer path):
      - Entire head encoded as single FP8 block + float32 scale
    """
    pid = tl.program_id(0)
    token_idx = tl.load(firing_indices_ptr + pid)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    WINDOW: tl.constexpr = (1 + OVERLAP) * COMPRESS_RATIO
    start = position - WINDOW + 1
    tokens = tl.arange(0, WINDOW)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ────────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32

    if IS_INDEXER:
        # ── Indexer path: single FP8 block + float32 scale ────────────
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

        # bf16 roundtrip via bit manipulation (SM70 compatible)
        result_u32 = result.to(tl.int32, bitcast=True)
        result_rounded = result_u32 + 0x7FFF + ((result_u32 >> 16) & 1)
        result_bf16_u32 = (result_rounded >> 16) << 16
        result_bf16 = result_bf16_u32.to(tl.float32, bitcast=True)

        absmax = tl.max(tl.abs(result_bf16), axis=0)
        absmax = tl.maximum(absmax, 1e-4)
        raw_scale = absmax * INV_FP8_MAX
        exponent = tl.ceil(tl.log2(raw_scale))
        inv_scale = tl.exp2(-exponent)

        x_scaled = result_bf16 * inv_scale
        x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)

        # Manual FP8 e4m3fn encode via fp16 bit manipulation
        x_f16_bits = (
            x_clamped.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        )
        fp16_sign = (x_f16_bits >> 15) & 1
        fp16_exp = (x_f16_bits >> 10) & 0x1F
        fp16_mant = x_f16_bits & 0x3FF
        exp_fp8 = fp16_exp - 8
        mant_fp8 = (fp16_mant >> 7) & 0x7
        round_bit = (fp16_mant >> 6) & 1
        sticky = fp16_mant & 0x3F
        do_round = round_bit & (sticky | (mant_fp8 & 1))
        mant_fp8 = mant_fp8 + do_round
        carry = mant_fp8 > 7
        mant_fp8 = tl.where(carry, 0, mant_fp8)
        exp_fp8 = tl.where(carry, exp_fp8 + 1, exp_fp8)
        is_max_exceeded = (exp_fp8 == 15) & (mant_fp8 > 6)
        mant_fp8 = tl.where(is_max_exceeded, 6, mant_fp8)
        is_overflow = exp_fp8 > 15
        exp_fp8 = tl.where(is_overflow, 15, exp_fp8)
        mant_fp8 = tl.where(is_overflow, 6, mant_fp8)
        is_underflow = (exp_fp8 <= 0) | (fp16_exp == 0)
        exp_fp8 = tl.where(is_underflow, 0, exp_fp8)
        mant_fp8 = tl.where(is_underflow, 0, mant_fp8)
        x_uint8 = ((fp16_sign << 7) | (exp_fp8 << 3) | mant_fp8).to(tl.uint8)

        tl.store(fp8_ptr + block, x_uint8, mask=mask)

        # Single float32 scale
        scale_val = tl.exp2(exponent)
        tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale_val)
    else:
        # ── Sparse attention path: 7 × FP8 blocks + BF16 RoPE ────────
        N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
        N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

        # bf16 roundtrip via bit manipulation (SM70 compatible)
        normed_u32 = normed.to(tl.int32, bitcast=True)
        normed_rounded = normed_u32 + 0x7FFF + ((normed_u32 >> 16) & 1)
        quant_bf16_u32 = (normed_rounded >> 16) << 16
        quant_input = quant_bf16_u32.to(tl.float32, bitcast=True)

        quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
        abs_2d = tl.abs(quant_2d)
        block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
        block_absmax = tl.maximum(block_absmax, 1e-4)

        raw_scales = block_absmax * INV_FP8_MAX
        exponents = tl.ceil(tl.log2(raw_scales))
        inv_scales = tl.exp2(-exponents)
        inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
        x_scaled = quant_2d * inv_scales_col
        x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)

        # Manual FP8 e4m3fn encode via fp16 bit manipulation
        x_f16_bits = (
            x_clamped.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        )
        fp16_sign = (x_f16_bits >> 15) & 1
        fp16_exp = (x_f16_bits >> 10) & 0x1F
        fp16_mant = x_f16_bits & 0x3FF
        exp_fp8 = fp16_exp - 8
        mant_fp8 = (fp16_mant >> 7) & 0x7
        round_bit = (fp16_mant >> 6) & 1
        sticky = fp16_mant & 0x3F
        do_round = round_bit & (sticky | (mant_fp8 & 1))
        mant_fp8 = mant_fp8 + do_round
        carry = mant_fp8 > 7
        mant_fp8 = tl.where(carry, 0, mant_fp8)
        exp_fp8 = tl.where(carry, exp_fp8 + 1, exp_fp8)
        is_max_exceeded = (exp_fp8 == 15) & (mant_fp8 > 6)
        mant_fp8 = tl.where(is_max_exceeded, 6, mant_fp8)
        is_overflow = exp_fp8 > 15
        exp_fp8 = tl.where(is_overflow, 15, exp_fp8)
        mant_fp8 = tl.where(is_overflow, 6, mant_fp8)
        is_underflow = (exp_fp8 <= 0) | (fp16_exp == 0)
        exp_fp8 = tl.where(is_underflow, 0, exp_fp8)
        mant_fp8 = tl.where(is_underflow, 0, mant_fp8)
        x_uint8 = ((fp16_sign << 7) | (exp_fp8 << 3) | mant_fp8).to(tl.uint8)
        x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

        nope_mask = block < NOPE_HEAD_DIM
        tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

        scale_idx = tl.arange(0, N_QUANT_BLOCKS)
        encoded = exponents + 127.0
        encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
        tl.store(
            scale_ptr + scale_idx,
            encoded.to(tl.uint8),
            mask=scale_idx < N_NOPE_BLOCKS,
        )
        tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

        # Store rotated rope portion as bf16 (uint16) into cache's bf16 area
        bf16_u16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.uint16))
        rope_local = block - NOPE_HEAD_DIM
        is_rope = (block >= NOPE_HEAD_DIM) & mask
        # fp32 → bf16 via bit manipulation
        result_u32 = result.to(tl.int32, bitcast=True)
        result_bf16_bits = (
            (result_u32 + 0x7FFF + ((result_u32 >> 16) & 1)) >> 16
        ).to(tl.uint16)
        tl.store(bf16_u16_ptr + rope_local, result_bf16_bits, mask=is_rope)


def _sm70_triton_fused_compress_norm_rope_insert_fp8(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """SM70 Triton replacement for _torch_fused_compress_norm_rope_insert_fp8_fallback.

    Pre-computes firing mask on the Python side, then launches the Triton kernel
    with grid=(num_firing_tokens,) to eliminate per-token boundary checks and
    Python-level loop overhead.
    """
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    # Pre-compute firing mask: only tokens at compress-ratio boundaries
    fire_mask = (positions[:num_tokens] + 1) % compress_ratio == 0
    # Also filter out padding tokens (slot_mapping < 0)
    fire_mask = fire_mask & (slot_mapping[:num_tokens] >= 0)
    firing_indices = fire_mask.nonzero(as_tuple=False).squeeze(1)

    if firing_indices.numel() == 0:
        return

    # Ensure int32 for kernel compatibility
    firing_indices = firing_indices.to(torch.int32)

    is_indexer = head_size == 128
    triton_block_size = triton.next_power_of_2(head_size)
    num_warps = 1 if is_indexer else 4

    _sm70_fused_compress_norm_rope_insert_fp8_kernel[(firing_indices.numel(),)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        # firing indices
        firing_indices,
        # constexprs
        HEAD_SIZE=head_size,
        TRITON_BLOCK_SIZE=triton_block_size,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=fp8_max,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        IS_INDEXER=is_indexer,
        num_warps=num_warps,
    )


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._fused_kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
            self._num_warps = 4
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._fused_kernel = (
                    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
                )
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._fused_kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
            self._num_warps = 1
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, hidden_size]
        x: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        num_tokens, _ = x.shape
        # bf16 weights/activations but fp32 output for numerical stability of
        # the downstream compressor math.
        kv_score = cublas_gemm_bf16_bf16_fp32(x, self.fused_wkv_wgate.weight)
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and _fused_kernel below
        # depend on preceding kernel outputs (kv/score from the cublas GEMM;
        # state_cache from this kernel) but neither emits/waits on PDL grid
        # dependency primitives, so launch_pdl=True caused a read-after-write
        # race and non-deterministic output.
        _save_partial_states_kernel[(num_actual,)](
            kv,
            kv.stride(0),
            score,
            score.stride(0),
            self.ape,
            self.ape.stride(0),
            positions,
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            slot_mapping,
            block_size,
            HEAD_SIZE=kv.shape[-1],
            TRITON_BLOCK_SIZE=triton.next_power_of_2(kv.shape[-1]),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            launch_pdl=False,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache

        self._fused_kernel[(num_actual,)](
            # state cache
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            # metadata
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            block_size,
            # RMSNorm
            self.norm.weight,
            self.rms_norm_eps,
            # RoPE
            cos_sin_cache,
            cos_sin_cache.stride(0),
            # KV cache
            kv_cache,
            k_cache_metadata.slot_mapping,
            kv_cache.shape[1],  # paged KV cache block size (tokens per block)
            # constexprs
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            ROPE_HEAD_DIM=self.rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=self._quant_block,
            TOKEN_STRIDE=self._token_stride,
            SCALE_DIM=self._scale_dim,
            KV_BLOCK_STRIDE=kv_cache.stride(0),
            num_warps=self._num_warps,
            launch_pdl=False,
        )


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    # Fused: score += ape[position % compress_ratio]
    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(
        base_ptr + STATE_WIDTH + block,
        score + ape,
        mask=mask,
    )
