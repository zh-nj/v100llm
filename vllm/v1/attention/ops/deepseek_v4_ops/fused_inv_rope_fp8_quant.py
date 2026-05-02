# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused inverse RoPE + block-scaled FP8 quantization kernel for DeepseekV4 attention.

Output scale format is pre-transformed (MN-major TMA-aligned; FP32 on SM90,
INT32-packed UE8M0 on SM100) so fp8_einsum skips transform_sf_into_required_layout.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    # int64: stride multiply overflows int32 past num_tokens=32768 (IMA).
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD

    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + block-scaled FP8 quantization.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] with cos||sin.
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        tma_aligned_scales: Output INT32 packed UE8M0 for SM100 (True)
                            or FP32 for SM90 (False).

    Returns:
        o_fp8: [T, G, D] float8_e4m3fn, strides (D, T*D, 1).
        o_scale: Pre-transformed scale tensor for fp8_einsum.
    """
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=fp8_dtype,
        device=o.device,
    )

    tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
    if tma_aligned_scales:
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_buf = torch.empty(
            n_groups * packed_sf_k * tma_aligned_T,
            dtype=torch.int32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, packed_sf_k),
            (packed_sf_k * tma_aligned_T, 1, tma_aligned_T),
        )
    else:
        scale_buf = torch.empty(
            n_groups * num_scale_blocks * tma_aligned_T,
            dtype=torch.float32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, num_scale_blocks),
            (num_scale_blocks * tma_aligned_T, 1, tma_aligned_T),
        )

    if _should_use_torch_fallback(o):
        assert not tma_aligned_scales, "SM70 fallback does not support SM100 scales"
        _sm70_fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            fp8_buf,
            scale_buf,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=quant_group_size,
            fp8_max=fp8_max,
        )
        return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)

    common_args = dict(
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        num_stages=1,
        launch_pdl=False,
    )

    grid = (tma_aligned_T, n_groups * heads_per_group)
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        **common_args,
        num_warps=1,
    )

    return fp8_buf.transpose(0, 1), scale_buf.transpose(0, 1)


def _should_use_torch_fallback(o: torch.Tensor) -> bool:
    if not o.is_cuda:
        return False
    capability = torch.cuda.get_device_capability(o.device)
    return capability[0] < 8


@triton.jit
def _sm70_inv_rope_fp8_quant_kernel(
    # o: [num_tokens, num_heads, head_dim] fp16
    o_ptr,
    # positions: [num_tokens] int64
    positions_ptr,
    # cos_sin_cache: [max_pos, rope_dim] fp16
    cos_sin_cache_ptr,
    # fp8_buf: [n_groups, num_tokens, d] uint8 (output)
    fp8_ptr,
    # scale_buf: [n_groups, num_tokens, num_scale_blocks] fp32 (output)
    scale_ptr,
    # Dims
    num_tokens,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope: tl.constexpr,
    d: tl.constexpr,  # heads_per_group * head_dim
    quant_group_size: tl.constexpr,
    num_scale_blocks: tl.constexpr,
    fp8_max: tl.constexpr,
    cache_stride: tl.constexpr,
    o_stride_token: tl.constexpr,
    o_stride_head: tl.constexpr,
    fp8_stride_group: tl.constexpr,
    fp8_stride_token: tl.constexpr,
    scale_stride_group: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused inverse RoPE + FP8 quant for SM70.

    One program per (token, group). Applies inverse GPT-J RoPE to the rope
    portion of each head, then quantizes the full output to FP8 e4m3fn
    using fp16 bit manipulation (no tl.float8e4nv needed).
    """
    pid_token = tl.program_id(0)
    pid_group = tl.program_id(1)

    if pid_token >= num_tokens:
        return

    # Load position and cos/sin
    pos = tl.load(positions_ptr + pid_token)
    cos_offsets = tl.arange(0, 32)
    cos_mask = cos_offsets < half_rope
    cos_vals = tl.load(
        cos_sin_cache_ptr + pos * cache_stride + cos_offsets,
        mask=cos_mask, other=0.0,
    ).to(tl.float32)
    sin_vals = tl.load(
        cos_sin_cache_ptr + pos * cache_stride + half_rope + cos_offsets,
        mask=cos_mask, other=0.0,
    ).to(tl.float32)

    # Process each head in this group: apply inverse RoPE and collect values
    # Then quantize the full d-dimensional vector in blocks
    # We write the inverse-RoPE'd values into a contiguous output vector
    # of size d = heads_per_group * head_dim, then quantize in blocks.

    # For each quant block: load values, compute UE8M0 scale, quantize, store
    for qb in range(num_scale_blocks):
        qb_start = qb * quant_group_size
        qb_offsets = tl.arange(0, BLOCK_D)
        qb_mask = qb_offsets < quant_group_size

        # Map flat offset within d to (head_idx, dim_within_head)
        flat_offsets = qb_start + qb_offsets
        head_idx = flat_offsets // head_dim
        dim_in_head = flat_offsets % head_dim

        # Load o values for this quant block
        # o[pid_token, pid_group * heads_per_group + head_idx, dim_in_head]
        o_offsets = (
            pid_token * o_stride_token
            + (pid_group * heads_per_group + head_idx) * o_stride_head
            + dim_in_head
        )
        valid = qb_mask & (head_idx < heads_per_group)
        vals = tl.load(o_ptr + o_offsets, mask=valid, other=0.0).to(tl.float32)

        # Apply inverse RoPE to rope portion
        # Rope region: dim_in_head in [nope_dim, nope_dim + rope_dim)
        is_rope = (dim_in_head >= nope_dim) & (dim_in_head < nope_dim + rope_dim)
        rope_offset = dim_in_head - nope_dim
        is_even = (rope_offset % 2) == 0
        pair_offset = rope_offset // 2

        # Load the paired value (even↔odd)
        even_dim = nope_dim + pair_offset * 2
        odd_dim = nope_dim + pair_offset * 2 + 1
        pair_o_even = (
            pid_token * o_stride_token
            + (pid_group * heads_per_group + head_idx) * o_stride_head
            + even_dim
        )
        pair_o_odd = (
            pid_token * o_stride_token
            + (pid_group * heads_per_group + head_idx) * o_stride_head
            + odd_dim
        )
        even_val = tl.load(o_ptr + pair_o_even, mask=valid & is_rope, other=0.0).to(tl.float32)
        odd_val = tl.load(o_ptr + pair_o_odd, mask=valid & is_rope, other=0.0).to(tl.float32)

        # Inverse GPT-J RoPE:
        #   even' = even * cos + odd * sin
        #   odd' = odd * cos - even * sin
        cos_v = tl.load(
            cos_sin_cache_ptr + pos * cache_stride + pair_offset,
            mask=valid & is_rope & (pair_offset < half_rope), other=0.0,
        ).to(tl.float32)
        sin_v = tl.load(
            cos_sin_cache_ptr + pos * cache_stride + half_rope + pair_offset,
            mask=valid & is_rope & (pair_offset < half_rope), other=0.0,
        ).to(tl.float32)

        inv_even = even_val * cos_v + odd_val * sin_v
        inv_odd = odd_val * cos_v - even_val * sin_v
        rope_val = tl.where(is_even, inv_even, inv_odd)
        vals = tl.where(is_rope & valid, rope_val, vals)

        # UE8M0 FP8 quantization (same as qnorm kernel)
        abs_vals = tl.abs(vals)
        block_max = tl.max(tl.where(valid, abs_vals, 0.0), axis=0)
        block_max = tl.maximum(block_max, 1e-10)
        raw_scale = block_max / fp8_max
        exponent = tl.ceil(tl.log2(raw_scale))
        scale = tl.exp2(exponent)
        x_scaled = vals / scale
        x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)

        # fp16→fp8 e4m3fn bit manipulation
        x_fp16 = x_clamped.to(tl.float16)
        fp16_bits = x_fp16.to(tl.uint16, bitcast=True)
        fp16_sign = (fp16_bits >> 15) & 1
        fp16_exp = (fp16_bits >> 10) & 0x1F
        fp16_mant = fp16_bits & 0x3FF

        exp_fp8 = (fp16_exp.to(tl.int32) - 8)
        mant_fp8 = ((fp16_mant >> 7) & 0x7).to(tl.int32)
        round_bit = ((fp16_mant >> 6) & 1).to(tl.int32)
        sticky = (fp16_mant & 0x3F).to(tl.int32)
        do_round = (round_bit != 0) & ((sticky != 0) | ((mant_fp8 & 1) != 0))
        mant_fp8 = tl.where(do_round, mant_fp8 + 1, mant_fp8)
        carry = mant_fp8 > 7
        mant_fp8 = tl.where(carry, 0, mant_fp8)
        exp_fp8 = tl.where(carry, exp_fp8 + 1, exp_fp8)
        mant_fp8 = tl.where((exp_fp8 == 15) & (mant_fp8 > 6), 6, mant_fp8)
        overflow = exp_fp8 > 15
        exp_fp8 = tl.where(overflow, 15, exp_fp8)
        mant_fp8 = tl.where(overflow, 6, mant_fp8)
        underflow = exp_fp8 <= 0
        exp_fp8 = tl.where(underflow, 0, exp_fp8)
        mant_fp8 = tl.where(underflow, 0, mant_fp8)
        is_zero = fp16_exp == 0
        exp_fp8 = tl.where(is_zero, 0, exp_fp8)
        mant_fp8 = tl.where(is_zero, 0, mant_fp8)

        fp8_byte = (fp16_sign.to(tl.uint8) << 7) | (exp_fp8.to(tl.uint8) << 3) | mant_fp8.to(tl.uint8)

        # Store fp8 bytes
        fp8_offsets = pid_group * fp8_stride_group + pid_token * fp8_stride_token + flat_offsets
        tl.store(fp8_ptr + fp8_offsets, fp8_byte, mask=valid)

        # Store scale (one per quant block)
        scale_val = scale  # scalar, broadcast
        # scale_buf[pid_group, pid_token, qb]
        scale_offset = pid_group * scale_stride_group + pid_token + qb * scale_stride_group  # need correct stride
        # Actually: scale_buf is [n_groups, num_tokens, num_scale_blocks]
        # with strides (scale_stride_group, 1, scale_stride_k)
        # So offset = pid_group * scale_stride_group + pid_token * 1 + qb * scale_stride_k
        # But scale_stride_k might be tma_aligned_T, let me use the passed strides
        # For SM70 (no TMA), scale_buf is contiguous [n_groups, num_tokens, num_scale_blocks]
        # offset = pid_group * (num_tokens * num_scale_blocks) + pid_token * num_scale_blocks + qb
        tl.store(
            scale_ptr + pid_group * scale_stride_group + pid_token + qb * (scale_stride_group // num_scale_blocks),
            scale,
        )


def _sm70_fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fp8_buf: torch.Tensor,
    scale_buf: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    fp8_max: float,
) -> None:
    """SM70 fused Triton kernel for inverse RoPE + FP8 quantization."""
    num_tokens, num_heads, head_dim = o.shape
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size

    BLOCK_D = quant_group_size  # process one quant block at a time

    grid = (num_tokens, n_groups)
    _sm70_inv_rope_fp8_quant_kernel[grid](
        o, positions, cos_sin_cache, fp8_buf, scale_buf,
        num_tokens=num_tokens,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        half_rope=rope_dim // 2,
        d=d,
        quant_group_size=quant_group_size,
        num_scale_blocks=num_scale_blocks,
        fp8_max=fp8_max,
        cache_stride=cos_sin_cache.stride(0),
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        BLOCK_D=BLOCK_D,
    )


def _torch_inv_rope_fp8_quant_fallback(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fp8_buf: torch.Tensor,
    scale_buf: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    fp8_max: float,
) -> None:
    """Torch correctness fallback for SM70, where tl.float8e4nv is unsupported."""
    num_tokens, _num_heads, head_dim = o.shape
    half_rope = rope_dim // 2
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size

    x = o.reshape(num_tokens, n_groups, heads_per_group, head_dim).float()
    cos_sin = cos_sin_cache[positions]
    cos = cos_sin[:, :half_rope].view(num_tokens, 1, 1, half_rope)
    sin = cos_sin[:, half_rope:].view(num_tokens, 1, 1, half_rope)

    rope = x[..., nope_dim : nope_dim + rope_dim].clone()
    x_vals = rope[..., ::2]
    y_vals = rope[..., 1::2]
    rotated = torch.empty_like(rope)
    rotated[..., ::2] = x_vals * cos + y_vals * sin
    rotated[..., 1::2] = y_vals * cos - x_vals * sin
    x[..., nope_dim : nope_dim + rope_dim] = rotated

    x_blocks = (
        x.reshape(num_tokens, n_groups, d)
        .permute(1, 0, 2)
        .contiguous()
        .view(n_groups, num_tokens, num_scale_blocks, quant_group_size)
    )
    absmax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / fp8_max))))
    x_scaled = (x_blocks / scales).clamp(-fp8_max, fp8_max)

    fp8_buf.copy_(x_scaled.to(fp8_buf.dtype).view(n_groups, num_tokens, d))
    scale_buf.copy_(scales.squeeze(-1))
