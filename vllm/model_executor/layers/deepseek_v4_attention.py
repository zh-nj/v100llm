# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4
_QK_NOPE_DIM = 448
_QK_ROPE_DIM = 64
_QK_FP8_MAX = 448.0
_QK_QUANT_BLOCK = 64
_QK_TOKEN_DATA_BYTES = _QK_NOPE_DIM + _QK_ROPE_DIM * 2
_QK_SCALE_BYTES = 8
_SM70_FP16_ATTENTION_OUTPUT_MAX = float(torch.finfo(torch.float16).max)

# Triton import for SM70 fused kernels
from vllm.triton_utils import tl, triton


@triton.jit
def _sm70_fused_qnorm_rope_kv_insert_kernel(
    # Q: [num_tokens, head_dim] fp16 (in-place)
    q_ptr,
    # KV: [num_tokens, head_dim] fp16 (read-only)
    kv_ptr,
    # K cache: [num_blocks, block_bytes] uint8
    k_cache_ptr,
    # Slot mapping: [num_tokens] int64
    slot_mapping_ptr,
    # Positions: [num_tokens] int64
    positions_ptr,
    # Cos-sin cache: [max_pos, rope_dim] fp16
    cos_sin_cache_ptr,
    # Params
    eps: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,  # 512
    nope_dim: tl.constexpr,  # 448
    rope_dim: tl.constexpr,  # 64
    half_rope: tl.constexpr,  # 32
    quant_block: tl.constexpr,  # 64
    fp8_max: tl.constexpr,  # 448.0
    token_data_bytes: tl.constexpr,  # 576
    scale_bytes: tl.constexpr,  # 8
    max_pos: tl.constexpr,
    block_stride: tl.constexpr,  # total bytes per cache block
    NOPE_BLOCK: tl.constexpr,  # processing block for nope part
):
    """Fused Q-norm + RoPE + KV-RoPE + FP8-quant + cache-insert for SM70.

    One Triton program per token. Eliminates ~15 separate CUDA kernel
    launches from the torch fallback path.
    """
    pid = tl.program_id(0)

    # ---- Load position and cos/sin ----
    pos = tl.load(positions_ptr + pid)

    cos_offsets = tl.arange(0, 32)  # half_rope = 32
    cos_mask = cos_offsets < half_rope
    cos_vals = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + cos_offsets,
        mask=cos_mask, other=0.0,
    ).to(tl.float32)
    sin_vals = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + half_rope + cos_offsets,
        mask=cos_mask, other=0.0,
    ).to(tl.float32)

    # ---- Q: RMS norm ----
    sq_sum = tl.zeros([1], dtype=tl.float32)
    for start in range(0, head_dim, NOPE_BLOCK):
        offsets = start + tl.arange(0, NOPE_BLOCK)
        mask = offsets < head_dim
        qv = tl.load(q_ptr + pid * head_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        sq_sum += tl.sum(qv * qv, axis=0)
    rms_inv = tl.rsqrt(sq_sum / head_dim + eps)

    # ---- Q: apply norm + RoPE, write back ----
    # Nope part: just norm (no RoPE)
    for start in range(0, nope_dim, NOPE_BLOCK):
        offsets = start + tl.arange(0, NOPE_BLOCK)
        mask = offsets < nope_dim
        qv = tl.load(q_ptr + pid * head_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        qv = qv * rms_inv
        tl.store(q_ptr + pid * head_dim + offsets, qv.to(tl.float16), mask=mask)

    # Rope part: norm + GPT-J rotation
    rope_offsets_even = tl.arange(0, 32)  # even indices: 0,1,...,31
    rope_offsets_odd = tl.arange(0, 32)
    even_mask = rope_offsets_even < half_rope
    # Load Q rope part (interleaved even/odd)
    q_even = tl.load(
        q_ptr + pid * head_dim + nope_dim + rope_offsets_even * 2,
        mask=even_mask, other=0.0,
    ).to(tl.float32) * rms_inv
    q_odd = tl.load(
        q_ptr + pid * head_dim + nope_dim + rope_offsets_odd * 2 + 1,
        mask=even_mask, other=0.0,
    ).to(tl.float32) * rms_inv

    # GPT-J rotation: even' = even*cos - odd*sin, odd' = odd*cos + even*sin
    q_even_rot = q_even * cos_vals - q_odd * sin_vals
    q_odd_rot = q_odd * cos_vals + q_even * sin_vals
    tl.store(
        q_ptr + pid * head_dim + nope_dim + rope_offsets_even * 2,
        q_even_rot.to(tl.float16), mask=even_mask,
    )
    tl.store(
        q_ptr + pid * head_dim + nope_dim + rope_offsets_odd * 2 + 1,
        q_odd_rot.to(tl.float16), mask=even_mask,
    )

    # ---- KV: RoPE + FP8 quant + cache insert (all in Triton) ----
    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    pos_in_block = slot_idx % block_size
    cache_base = k_cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data_ptr = cache_base + pos_in_block * token_data_bytes
    token_scale_ptr = cache_base + block_size * token_data_bytes + pos_in_block * scale_bytes

    # KV RoPE on rope part (last 64 elements)
    kv_even = tl.load(
        kv_ptr + pid * head_dim + nope_dim + rope_offsets_even * 2,
        mask=even_mask, other=0.0,
    ).to(tl.float32)
    kv_odd = tl.load(
        kv_ptr + pid * head_dim + nope_dim + rope_offsets_odd * 2 + 1,
        mask=even_mask, other=0.0,
    ).to(tl.float32)
    kv_even_rot = kv_even * cos_vals - kv_odd * sin_vals
    kv_odd_rot = kv_odd * cos_vals + kv_even * sin_vals

    # Store rope part as bf16 in cache
    bf16_out_ptr = (token_data_ptr + nope_dim).to(tl.pointer_type(tl.bfloat16))
    tl.store(bf16_out_ptr + rope_offsets_even * 2, kv_even_rot.to(tl.bfloat16), mask=even_mask)
    tl.store(bf16_out_ptr + rope_offsets_odd * 2 + 1, kv_odd_rot.to(tl.bfloat16), mask=even_mask)

    # KV nope part: UE8M0 FP8 quant + store (all in Triton, no torch)
    # Manual fp8 e4m3fn encoding via fp16 bit manipulation.
    # After UE8M0 scaling, values are in [-448, 448] so no overflow.
    for qb in range(nope_dim // quant_block):
        qb_offsets = qb * quant_block + tl.arange(0, 64)
        qb_mask = qb_offsets < nope_dim
        nope_vals = tl.load(
            kv_ptr + pid * head_dim + qb_offsets,
            mask=qb_mask, other=0.0,
        ).to(tl.float32)

        # UE8M0: scale = 2^ceil(log2(absmax / fp8_max))
        abs_vals = tl.abs(nope_vals)
        block_max = tl.max(abs_vals, axis=0)
        block_max = tl.maximum(block_max, 1e-4)
        raw_scale = block_max / fp8_max
        exponent = tl.ceil(tl.log2(raw_scale))
        scale = tl.exp2(exponent)

        # Scale and clamp to fp8 range
        x_scaled = nope_vals / scale
        x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)

        # fp16→fp8 e4m3fn via bit manipulation
        # Cast to fp16, extract bits
        x_fp16 = x_clamped.to(tl.float16)
        fp16_bits = x_fp16.to(tl.uint16, bitcast=True)
        fp16_sign = (fp16_bits >> 15) & 1  # 1 bit
        fp16_exp = (fp16_bits >> 10) & 0x1F  # 5 bits, bias=15
        fp16_mant = fp16_bits & 0x3FF  # 10 bits

        # fp8 exponent = fp16_exp - 8 (bias 15→7)
        exp_fp8 = (fp16_exp.to(tl.int32) - 8)

        # Round mantissa from 10 to 3 bits (round-to-nearest-even)
        mant_fp8 = ((fp16_mant >> 7) & 0x7).to(tl.int32)
        round_bit = ((fp16_mant >> 6) & 1).to(tl.int32)
        sticky = (fp16_mant & 0x3F).to(tl.int32)
        do_round = (round_bit != 0) & ((sticky != 0) | ((mant_fp8 & 1) != 0))
        mant_fp8 = tl.where(do_round, mant_fp8 + 1, mant_fp8)
        # Handle mantissa carry
        carry = (mant_fp8 > 7)
        mant_fp8 = tl.where(carry, 0, mant_fp8)
        exp_fp8 = tl.where(carry, exp_fp8 + 1, exp_fp8)

        # Clamp exp=15 mant to 6 (e4m3fn: mant=7 at exp=15 is NaN)
        mant_fp8 = tl.where((exp_fp8 == 15) & (mant_fp8 > 6), 6, mant_fp8)
        # Overflow: exp > 15 → max value (0x7E)
        overflow = exp_fp8 > 15
        exp_fp8 = tl.where(overflow, 15, exp_fp8)
        mant_fp8 = tl.where(overflow, 6, mant_fp8)
        # Underflow: exp <= 0 → zero (denorms negligible for UE8M0 scaled values)
        underflow = exp_fp8 <= 0
        exp_fp8 = tl.where(underflow, 0, exp_fp8)
        mant_fp8 = tl.where(underflow, 0, mant_fp8)
        # Zero input
        is_zero = fp16_exp == 0
        exp_fp8 = tl.where(is_zero, 0, exp_fp8)
        mant_fp8 = tl.where(is_zero, 0, mant_fp8)

        # Assemble fp8 byte: sign(1) | exp(4) | mant(3)
        fp8_byte = (fp16_sign.to(tl.uint8) << 7) | (exp_fp8.to(tl.uint8) << 3) | mant_fp8.to(tl.uint8)
        tl.store(token_data_ptr + qb_offsets, fp8_byte, mask=qb_mask)

        # Store UE8M0 encoded scale
        encoded = (exponent + 127.0)
        encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
        tl.store(token_scale_ptr + qb, encoded.to(tl.uint8))

    # Padding scale byte at index 7
    tl.store(token_scale_ptr + 7, tl.zeros((), dtype=tl.uint8))


def _sm70_fused_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """SM70 fully fused Triton kernel for Q-norm + RoPE + KV FP8 quant + cache insert.

    Single kernel launch replaces ~15 CUDA ops from the torch fallback.
    FP8 e4m3fn encoding done via fp16 bit manipulation (no tl.float8e4nv needed).
    """
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return

    head_dim = q.shape[-1]
    block_stride = k_cache.shape[1]

    grid = (num_tokens,)
    _sm70_fused_qnorm_rope_kv_insert_kernel[grid](
        q, kv, k_cache, slot_mapping, positions, cos_sin_cache,
        eps=eps,
        block_size=block_size,
        head_dim=head_dim,
        nope_dim=_QK_NOPE_DIM,
        rope_dim=_QK_ROPE_DIM,
        half_rope=_QK_ROPE_DIM // 2,
        quant_block=_QK_QUANT_BLOCK,
        fp8_max=_QK_FP8_MAX,
        token_data_bytes=_QK_TOKEN_DATA_BYTES,
        scale_bytes=_QK_SCALE_BYTES,
        max_pos=cos_sin_cache.shape[0],
        block_stride=block_stride,
        NOPE_BLOCK=128,
    )


def _trace_nonfinite_tensor(label: str, tensor: torch.Tensor) -> None:
    if os.getenv("VLLM_DEEPSEEK_V4_NAN_TRACE", "0") != "1":
        return
    if not torch.is_floating_point(tensor):
        return
    if torch.isfinite(tensor).all():
        return
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    if finite_values.numel() == 0:
        min_value = max_value = float("nan")
    else:
        stats = finite_values.float()
        min_value = float(stats.min().item())
        max_value = float(stats.max().item())
    logger.error(
        "DeepSeek V4 attention nonfinite tensor at %s: shape=%s dtype=%s "
        "nan=%d inf=%d finite_min=%s finite_max=%s",
        label,
        tuple(tensor.shape),
        tensor.dtype,
        int(torch.isnan(tensor).sum().item()),
        int(torch.isinf(tensor).sum().item()),
        min_value,
        max_value,
    )


def _trace_tensor_summary(label: str, tensor: torch.Tensor) -> None:
    if os.getenv("VLLM_DEEPSEEK_V4_NAN_TRACE", "0") != "1":
        return
    if tensor.numel() == 0:
        logger.error(
            "DeepSeek V4 trace tensor at %s: shape=%s dtype=%s empty",
            label,
            tuple(tensor.shape),
            tensor.dtype,
        )
        return
    flat = tensor
    if not torch.is_floating_point(flat):
        flat = flat.to(torch.float32)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if finite_values.numel() == 0:
        min_value = max_value = float("nan")
    else:
        stats = finite_values.float()
        min_value = float(stats.min().item())
        max_value = float(stats.max().item())
    logger.error(
        "DeepSeek V4 trace tensor at %s: shape=%s dtype=%s "
        "nan=%d inf=%d finite_min=%s finite_max=%s",
        label,
        tuple(tensor.shape),
        tensor.dtype,
        int(torch.isnan(flat).sum().item()),
        int(torch.isinf(flat).sum().item()),
        min_value,
        max_value,
    )


def _trace_layer16_summary(prefix: str, label: str, tensor: torch.Tensor) -> None:
    if prefix.endswith("layers.16.attn"):
        _trace_tensor_summary(f"{prefix}.{label}", tensor)


def _should_use_qnorm_rope_kv_insert_fallback(q: torch.Tensor) -> bool:
    if not q.is_cuda:
        return False
    capability = torch.cuda.get_device_capability(q.device)
    return capability[0] < 8


def _normalize_flashmla_sm70_prefill_kv_(kv: torch.Tensor) -> torch.Tensor:
    return kv


def _normalize_sm70_fp8_cache_exponents(exponents: torch.Tensor) -> torch.Tensor:
    return exponents


def _should_clamp_sm70_fp16_attention_output(out: torch.Tensor) -> bool:
    if not out.is_cuda or out.dtype != torch.float16:
        return False
    capability = torch.cuda.get_device_capability(out.device)
    return capability[0] < 8


def _clamp_sm70_fp16_attention_output_(out: torch.Tensor) -> torch.Tensor:
    return out.clamp_(
        min=-_SM70_FP16_ATTENTION_OUTPUT_MAX,
        max=_SM70_FP16_ATTENTION_OUTPUT_MAX,
    )


def _should_use_sm70_decode_prefill_fallback(
    q: torch.Tensor,
    swa_only: bool,
) -> bool:
    if not q.is_cuda:
        return False
    capability = torch.cuda.get_device_capability(q.device)
    return capability[0] < 8


def _get_decode_prefill_fallback_workspace(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if is_workspace_manager_initialized():
        return current_workspace_manager().get_simultaneous((shape, dtype))[0]
    return torch.empty(shape, dtype=dtype, device=device)


def _decode_prefill_fallback_slots(
    global_indices: torch.Tensor,
) -> torch.Tensor:
    if global_indices.ndim == 3:
        assert global_indices.shape[1] == 1
        return global_indices[:, 0, :]
    if global_indices.ndim == 2:
        return global_indices
    raise ValueError(
        "Decode fallback indices must have shape [tokens, topk] or "
        f"[tokens, 1, topk], got {tuple(global_indices.shape)}"
    )


def _build_decode_prefill_fallback_indices(
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    *,
    row_stride: int | None = None,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = _decode_prefill_fallback_slots(global_indices)
    local_lens = global_lens.reshape(-1)
    topk = slots.shape[-1]
    if row_stride is None:
        row_stride = topk

    offsets = torch.arange(topk, device=slots.device, dtype=torch.int32)
    bases = (
        torch.arange(slots.shape[0], device=slots.device, dtype=torch.int32)
        .unsqueeze(1)
        .mul_(row_stride)
        .add_(offset)
    )
    valid = (offsets.unsqueeze(0) < local_lens.unsqueeze(1)) & (slots >= 0)
    local_indices = torch.where(
        valid,
        bases + offsets.unsqueeze(0),
        torch.full_like(slots, -1),
    )
    return local_indices.unsqueeze(1), local_lens


def _gather_decode_prefill_fallback_kv_(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    slots = _decode_prefill_fallback_slots(global_indices)
    lens = global_lens.reshape(-1)
    topk = slots.shape[-1]
    if out.shape[0] != slots.shape[0] or out.shape[1] != topk:
        raise ValueError(
            "Decode fallback KV workspace shape must match indices, got "
            f"out={tuple(out.shape)} indices={tuple(global_indices.shape)}"
        )

    out.zero_()
    if slots.numel() == 0:
        return out

    offsets = torch.arange(topk, device=slots.device, dtype=torch.int32)
    valid = (offsets.unsqueeze(0) < lens.unsqueeze(1)) & (slots >= 0)
    if not valid.any():
        return out

    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_indices = torch.div(
        safe_slots, block_size, rounding_mode="floor"
    ).to(torch.long)
    pos_in_block = (safe_slots % block_size).to(torch.long)

    k_cache_2d = k_cache.reshape(k_cache.shape[0], -1)
    token_offsets = (
        pos_in_block.unsqueeze(-1) * _QK_TOKEN_DATA_BYTES
        + torch.arange(_QK_TOKEN_DATA_BYTES, device=k_cache.device)
    )
    token_bytes = k_cache_2d[block_indices.unsqueeze(-1), token_offsets]

    fp8_values = token_bytes[..., :_QK_NOPE_DIM].contiguous().view(
        torch.float8_e4m3fn
    )
    scale_offsets = (
        block_size * _QK_TOKEN_DATA_BYTES
        + pos_in_block.unsqueeze(-1) * _QK_SCALE_BYTES
        + torch.arange(_QK_NOPE_DIM // _QK_QUANT_BLOCK, device=k_cache.device)
    )
    scales = torch.exp2(
        k_cache_2d[block_indices.unsqueeze(-1), scale_offsets].to(torch.float32)
        - 127.0
    ).repeat_interleave(_QK_QUANT_BLOCK, dim=-1)
    out[..., :_QK_NOPE_DIM] = (fp8_values.float() * scales).to(out.dtype)

    rope_values = token_bytes[..., _QK_NOPE_DIM:].contiguous().view(torch.bfloat16)
    out[..., _QK_NOPE_DIM:] = rope_values.to(out.dtype)
    out[~valid] = 0
    return out


def _apply_gptj_rope_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    assert nope_dim >= 0

    out = x.clone().float()
    rope = out[..., nope_dim:]
    even = rope[..., ::2]
    odd = rope[..., 1::2]

    cos_sin = cos_sin_cache[positions].float()
    view_shape = (positions.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    cos = cos_sin[..., :half].view(view_shape)
    sin = cos_sin[..., half:].view(view_shape)

    rotated = torch.empty_like(rope)
    rotated[..., ::2] = even * cos - odd * sin
    rotated[..., 1::2] = odd * cos + even * sin
    out[..., nope_dim:] = rotated
    return out.to(x.dtype)


def _torch_qnorm_rope_kv_insert_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """Torch correctness fallback for SM70, where the fused CUDA op is sm80+."""
    q_float = q.float()
    variance = q_float.pow(2).mean(dim=-1, keepdim=True)
    q_norm = (q_float * torch.rsqrt(variance + eps)).to(q.dtype)
    q.copy_(_apply_gptj_rope_tail(q_norm, positions, cos_sin_cache))

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    kv_rope = _apply_gptj_rope_tail(
        kv[:num_tokens],
        positions[:num_tokens],
        cos_sin_cache,
    )
    valid_mask = slot_mapping >= 0
    if not valid_mask.any():
        return

    kv_valid = kv_rope[valid_mask]
    slots = slot_mapping[valid_mask]
    block_indices = slots // block_size
    pos_in_block = slots % block_size

    nope = kv_valid[:, :_QK_NOPE_DIM].float()
    blocks = nope.view(-1, _QK_NOPE_DIM // _QK_QUANT_BLOCK, _QK_QUANT_BLOCK)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    exponents = _normalize_sm70_fp8_cache_exponents(
        torch.ceil(torch.log2(absmax / _QK_FP8_MAX))
    )
    scales = torch.exp2(exponents)
    fp8_data = (blocks / scales).clamp(-_QK_FP8_MAX, _QK_FP8_MAX)
    fp8_bytes = (
        fp8_data.to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
        .view(-1, _QK_NOPE_DIM)
    )
    rope_bytes = (
        kv_valid[:, _QK_NOPE_DIM:]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .view(-1, _QK_ROPE_DIM * 2)
    )
    token_data = torch.cat((fp8_bytes, rope_bytes), dim=-1)

    num_valid = slots.shape[0]
    data_offsets = (
        pos_in_block[:, None] * _QK_TOKEN_DATA_BYTES
        + torch.arange(_QK_TOKEN_DATA_BYTES, device=k_cache.device)
    )
    k_cache[block_indices[:, None], data_offsets] = token_data

    encoded_scales = (exponents.squeeze(-1) + 127.0).clamp(0, 255).to(torch.uint8)
    scale_data = torch.zeros(
        num_valid,
        _QK_SCALE_BYTES,
        dtype=torch.uint8,
        device=k_cache.device,
    )
    scale_data[:, : _QK_NOPE_DIM // _QK_QUANT_BLOCK] = encoded_scales
    scale_offsets = (
        block_size * _QK_TOKEN_DATA_BYTES
        + pos_in_block[:, None] * _QK_SCALE_BYTES
        + torch.arange(_QK_SCALE_BYTES, device=k_cache.device)
    )
    k_cache[block_indices[:, None], scale_offsets] = scale_data


def _flashmla_bf16_io(
    q: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_q = q if q.dtype is torch.bfloat16 else q.to(torch.bfloat16)
    flash_output = (
        output
        if output.dtype is torch.bfloat16
        else torch.empty_like(output, dtype=torch.bfloat16)
    )
    return flash_q, flash_output


def _copy_flashmla_output(
    flash_output: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if flash_output.data_ptr() != output.data_ptr() or flash_output.dtype != output.dtype:
        output.copy_(flash_output.to(output.dtype))


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream: torch.cuda.Stream | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128
        # SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        assert cap is not None, "DeepseekV4 attention requires a CUDA device"
        self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
        self._tma_aligned_scales = cap.major >= 10

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        self.aux_stream = mla_modules.aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.input", hidden_states)
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.qr", qr)
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.kv", kv)

        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention (inside custom op for torch.compile boundary)
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            qr,
            kv,
            positions,
            o_padded,
            self.layer_name,
        )
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.o_padded", o_padded)
        o = o_padded[:, : self.n_local_heads, :]
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.o", o)
        _trace_layer16_summary(self.prefix, "wrapper.o", o)

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.o_scale", o_scale)
        _trace_layer16_summary(self.prefix, "wrapper.o_scale", o_scale)

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=hidden_states.dtype,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.z", z)
        _trace_layer16_summary(self.prefix, "wrapper.z", z)

        out = self.wo_b(z.flatten(1))
        if isinstance(out, tuple):
            out = out[0]
        if _should_clamp_sm70_fp16_attention_output(out):
            _clamp_sm70_fp16_attention_output_(out)
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.wo_b", out)
        _trace_layer16_summary(self.prefix, "wrapper.wo_b", out)
        return out

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        _trace_nonfinite_tensor(f"{self.prefix}.impl.qr_norm", qr)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.kv_norm", kv)
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.q_proj", q)

        # Overlap kv_insert with whichever of indexer/compressor is present.
        # Indexer implies compressor; when both exist, compressor rides on the
        # aux stream alongside kv_insert so the heavy indexer owns default.
        if self.indexer is not None:
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def kv_insert_and_compress() -> None:
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                compressor(hidden_states, positions, self.rotary_emb)

            maybe_execute_in_parallel(
                lambda: indexer(hidden_states, qr, positions, self.indexer_rotary_emb),
                kv_insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        elif self.compressor is not None:
            # Compressor on default, kv_insert on aux.
            compressor = self.compressor
            maybe_execute_in_parallel(
                lambda: compressor(hidden_states, positions, self.rotary_emb),
                lambda: self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                ),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.q_after_insert", q)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            # Reserve _forward_prefill's bf16-gather workspace; the dummy
            # run returns before mla_attn runs, so without this the shared
            # workspace locks below the real prefill size.
            sub = self.mla_attn
            swa_only = sub.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (sub.max_model_len + sub.compress_ratio - 1) // sub.compress_ratio
            )
            M = N + sub.window_size + sub.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            out.zero_()
            return

        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.mla_attn(q, kv, positions, output=out)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.mla_out", out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        if _should_use_qnorm_rope_kv_insert_fallback(q):
            _sm70_fused_qnorm_rope_kv_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
            return

        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, qr, kv, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if _should_use_torch_fp8_einsum_fallback(a):
        _sm70_fp8_einsum_bmm(a, a_scale, b, b_scale, out, equation)
        return
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def _should_use_torch_fp8_einsum_fallback(a: torch.Tensor) -> bool:
    if not has_deep_gemm():
        return True
    if not a.is_cuda:
        return False
    return torch.cuda.get_device_capability(a.device)[0] < 8


def _deepseek_v4_fp8_einsum_torch_fallback(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(
            "DeepSeek V4 torch fp8 einsum fallback only supports "
            f"'bhr,hdr->bhd', got {equation!r}."
        )

    groups = a.shape[1]
    hidden = a.shape[2]
    rank = b.shape[1] if b.dim() == 3 else b.shape[0] // groups
    b_3d = b.reshape(groups, rank, hidden)

    a_blocks = a_scale.shape[-1]
    weight_scale_shape = (groups, rank // 128, hidden // 128)
    b_scale_3d = b_scale.reshape(weight_scale_shape)
    a_deq = a.float() * a_scale.repeat_interleave(hidden // a_blocks, dim=-1)
    b_deq = b_3d.float() * b_scale_3d.repeat_interleave(
        128, dim=1
    ).repeat_interleave(128, dim=2)
    result = torch.einsum(equation, a_deq, b_deq)
    out.copy_(result.to(out.dtype))


def _sm70_fp8_einsum_bmm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    """SM70 fast path: pre-dequant weight to fp16 (cached) + bmm.

    The wo_a weight (b) is dequantized to fp16 once and cached on the tensor.
    At runtime, only the activation (a) needs dequant, then a single cuBLAS
    bmm computes the grouped einsum.
    """
    if equation != "bhr,hdr->bhd":
        return _deepseek_v4_fp8_einsum_torch_fallback(
            a, a_scale, b, b_scale, out, equation
        )

    groups = a.shape[1]
    hidden = a.shape[2]
    rank = b.shape[1] if b.dim() == 3 else b.shape[0] // groups

    # Lazily pre-dequant b (weight) to fp16 and cache as [groups, hidden, rank]
    # for bmm: [groups, 1, hidden] @ [groups, hidden, rank] → [groups, 1, rank]
    b_t_fp16 = getattr(b, "_sm70_predequant_t", None)
    if b_t_fp16 is None:
        b_3d = b.reshape(groups, rank, hidden)
        weight_scale_shape = (groups, rank // 128, hidden // 128)
        b_scale_3d = b_scale.reshape(weight_scale_shape)
        b_deq = b_3d.float() * b_scale_3d.repeat_interleave(
            128, dim=1
        ).repeat_interleave(128, dim=2)
        b_t_fp16 = b_deq.half().transpose(1, 2).contiguous()  # [groups, hidden, rank]
        b._sm70_predequant_t = b_t_fp16  # type: ignore[attr-defined]

    # Dequant a (activation) to fp16
    a_blocks = a_scale.shape[-1]
    a_deq = (a.float() * a_scale.repeat_interleave(
        hidden // a_blocks, dim=-1
    )).half()

    # bmm: [groups, batch, hidden] @ [groups, hidden, rank] → [groups, batch, rank]
    batch = a_deq.shape[0]
    a_3d = a_deq.transpose(0, 1)  # [groups, batch, hidden]
    result = torch.bmm(a_3d, b_t_fp16)  # [groups, batch, rank]
    out.copy_(result.transpose(0, 1).to(out.dtype))


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        # DeepseekV4 only supports fp8 kv-cache format for now
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 only supports fp8 kv-cache format for now, "
            f"got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for DeepseekV4 for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        q, flash_output = _flashmla_bf16_io(q, output)
        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        if _should_use_sm70_decode_prefill_fallback(q, swa_only):
            if swa_only:
                fallback_indices, fallback_lens = (
                    _build_decode_prefill_fallback_indices(swa_indices, swa_lens)
                )
                fallback_topk_length: torch.Tensor | None = fallback_lens
                fallback_kv = _get_decode_prefill_fallback_workspace(
                    (num_decode_tokens, fallback_indices.shape[-1], q.shape[-1]),
                    torch.bfloat16,
                    q.device,
                )
                _gather_decode_prefill_fallback_kv_(
                    fallback_kv,
                    self.swa_cache_layer.kv_cache,
                    swa_indices,
                    fallback_lens,
                    swa_metadata.block_size,
                )
            else:
                assert kv_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                compressed_topk = topk_indices.shape[-1]
                swa_topk = swa_indices.shape[-1]
                total_topk = compressed_topk + swa_topk
                fallback_topk_length = None
                fallback_kv = _get_decode_prefill_fallback_workspace(
                    (num_decode_tokens, total_topk, q.shape[-1]),
                    torch.bfloat16,
                    q.device,
                )
                compressed_kv = fallback_kv[:, :compressed_topk]
                _gather_decode_prefill_fallback_kv_(
                    compressed_kv,
                    kv_cache,
                    topk_indices,
                    topk_lens,
                    attn_metadata.block_size // self.compress_ratio,
                )
                swa_kv = fallback_kv[:, compressed_topk:]
                _gather_decode_prefill_fallback_kv_(
                    swa_kv,
                    self.swa_cache_layer.kv_cache,
                    swa_indices,
                    swa_lens,
                    swa_metadata.block_size,
                )
                compressed_indices, _ = _build_decode_prefill_fallback_indices(
                    topk_indices,
                    topk_lens,
                    row_stride=total_topk,
                )
                swa_fallback_indices, _ = _build_decode_prefill_fallback_indices(
                    swa_indices,
                    swa_lens,
                    row_stride=total_topk,
                    offset=compressed_topk,
                )
                fallback_indices = torch.cat(
                    (compressed_indices, swa_fallback_indices), dim=-1
                )
            _normalize_flashmla_sm70_prefill_kv_(fallback_kv)
            flash_output, _, _ = flash_mla_sparse_fwd(
                q=q.squeeze(1),
                kv=fallback_kv.view(-1, 1, q.shape[-1]),
                indices=fallback_indices,
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=fallback_topk_length,
                out=flash_output,
            )
            _copy_flashmla_output(flash_output, output)
            return

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=flash_output.unsqueeze(1),
        )
        _copy_flashmla_output(out.squeeze(1), output)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE
        trace_prefill = (
            os.getenv("VLLM_DEEPSEEK_V4_NAN_TRACE", "0") == "1"
            and self.prefix.endswith("layers.1.attn")
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )
            kv_chunk = kv[:chunk_size]
            _normalize_flashmla_sm70_prefill_kv_(kv_chunk)
            if trace_prefill:
                _trace_tensor_summary(f"{self.prefix}.prefill.kv", kv_chunk)
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.seq_lens",
                    seq_lens[chunk_start:chunk_end],
                )
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.gather_lens",
                    gather_lens[chunk_start:chunk_end],
                )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            if trace_prefill:
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.combined_indices", combined_indices
                )
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.combined_lens", combined_lens
                )

            output_slice = output[query_start:query_end]
            q_chunk, output_chunk = _flashmla_bf16_io(
                q[query_start:query_end],
                output_slice,
            )
            if trace_prefill:
                _trace_tensor_summary(f"{self.prefix}.prefill.q_chunk", q_chunk)
            flash_output, max_logits, lse = flash_mla_sparse_fwd(
                q=q_chunk,
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output_chunk,
            )
            if trace_prefill:
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.flash_output", flash_output
                )
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.max_logits", max_logits
                )
                _trace_tensor_summary(f"{self.prefix}.prefill.lse", lse)
            _copy_flashmla_output(flash_output, output_slice)


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lighening Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(hidden_states, positions, rotary_emb)
        weights, _ = self.weights_proj(hidden_states)
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
