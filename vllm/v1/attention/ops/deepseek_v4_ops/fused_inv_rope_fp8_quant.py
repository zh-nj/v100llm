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
    USE_SM70_FP8_ENCODE: tl.constexpr,
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
    x_quant_f32 = tl.clamp(x / scales_exp, -fp8_max, fp8_max)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )

    if USE_SM70_FP8_ENCODE:
        # SM70: manual FP8 e4m3fn encode via fp16 bit manipulation
        # Convert to fp16 first (closest representable), then extract bits
        # fp16: 1 sign + 5 exp (bias 15) + 10 mantissa
        # fp8 e4m3fn: 1 sign + 4 exp (bias 7) + 3 mantissa
        x_f16_bits = x_quant_f32.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        fp16_sign = (x_f16_bits >> 15) & 1
        fp16_exp = (x_f16_bits >> 10) & 0x1F
        fp16_mant = x_f16_bits & 0x3FF

        # Rebias exponent: fp8_exp = fp16_exp - 15 + 7 = fp16_exp - 8
        exp_fp8 = fp16_exp - 8
        # Truncate mantissa: keep top 3 bits, round-to-nearest-even
        mant_fp8 = (fp16_mant >> 7) & 0x7
        round_bit = (fp16_mant >> 6) & 1
        sticky = fp16_mant & 0x3F
        do_round = round_bit & (sticky | (mant_fp8 & 1))
        mant_fp8 = mant_fp8 + do_round
        # Handle mantissa overflow
        carry = mant_fp8 > 7
        mant_fp8 = tl.where(carry, 0, mant_fp8)
        exp_fp8 = tl.where(carry, exp_fp8 + 1, exp_fp8)
        # Clamp to fp8 e4m3fn max (exp=15, mant=6 → 448.0)
        is_max_exceeded = (exp_fp8 == 15) & (mant_fp8 > 6)
        mant_fp8 = tl.where(is_max_exceeded, 6, mant_fp8)
        is_overflow = exp_fp8 > 15
        exp_fp8 = tl.where(is_overflow, 15, exp_fp8)
        mant_fp8 = tl.where(is_overflow, 6, mant_fp8)
        # Underflow: subnormals → zero
        is_underflow = (exp_fp8 <= 0) | (fp16_exp == 0)
        exp_fp8 = tl.where(is_underflow, 0, exp_fp8)
        mant_fp8 = tl.where(is_underflow, 0, mant_fp8)

        x_uint8 = ((fp16_sign << 7) | (exp_fp8 << 3) | mant_fp8).to(tl.uint8)
        tl.store(fp8_base + offsets, x_uint8)

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
        use_sm70_fp8 = True
    else:
        use_sm70_fp8 = False

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
        USE_SM70_FP8_ENCODE=use_sm70_fp8,
        num_stages=1,
        launch_pdl=False,
    )

    grid = (tma_aligned_T, n_groups * heads_per_group)
    # On SM70, pass fp8_buf as uint8 view to avoid Triton fp8e4nv pointer type error
    fp8_kernel_buf = fp8_buf.view(torch.uint8) if use_sm70_fp8 else fp8_buf
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_kernel_buf,
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


# -----------------------------------------------------------------------------
# Custom op wrapping to bypass torch._inductor.fx_passes.post_grad
# `decompose_triton_kernel_wrapper_functional` pattern-matcher pass, which
# mishandles our triton kernel's as_strided + transpose output layout on
# torch 2.9 (eager trace keeps a redundant `stride * 1` mul; inductor trace
# constant-folds it; the resulting node-count mismatch fires an assertion
# inside `replace_by_example`, crashing AOT compile).
# By registering `fused_inv_rope_fp8_quant` as a torch custom op with a
# matching `fake_impl`, Inductor treats it as an opaque call_function node
# and never tries to decompose the triton kernel wrapper.
# -----------------------------------------------------------------------------

from vllm.utils.torch_utils import direct_register_custom_op as _register_op  # noqa: E402

_FUSED_INV_ROPE_EAGER = fused_inv_rope_fp8_quant


def _fused_inv_rope_fp8_quant_op(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FUSED_INV_ROPE_EAGER(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        tma_aligned_scales=tma_aligned_scales,
    )


def _fused_inv_rope_fp8_quant_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
    tma_aligned_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    tma_aligned_T = get_tma_aligned_size(num_tokens, 4)

    # Layout must match the eager impl's .transpose(0, 1) outputs, not just
    # their shapes. Inductor specializes downstream Triton stride arguments
    # from fake tensors; returning contiguous scales here silently bakes
    # scale_stride_k=1 and corrupts SM70 dequant under torch.compile.
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=fp8_dtype,
        device=o.device,
    )
    fp8_out = fp8_buf.transpose(0, 1)
    if tma_aligned_scales:
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_out = torch.empty(
            n_groups * packed_sf_k * tma_aligned_T,
            dtype=torch.int32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, packed_sf_k),
            (packed_sf_k * tma_aligned_T, 1, tma_aligned_T),
        ).transpose(0, 1)
    else:
        scale_out = torch.empty(
            n_groups * num_scale_blocks * tma_aligned_T,
            dtype=torch.float32,
            device=o.device,
        ).as_strided(
            (n_groups, num_tokens, num_scale_blocks),
            (num_scale_blocks * tma_aligned_T, 1, tma_aligned_T),
        ).transpose(0, 1)
    return fp8_out, scale_out


try:
    _register_op(
        op_name="fused_inv_rope_fp8_quant",
        op_func=_fused_inv_rope_fp8_quant_op,
        mutates_args=[],
        fake_impl=_fused_inv_rope_fp8_quant_fake,
    )
    _FUSED_INV_ROPE_OP = torch.ops.vllm.fused_inv_rope_fp8_quant


    def fused_inv_rope_fp8_quant(  # type: ignore[no-redef]
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
        return _FUSED_INV_ROPE_OP(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            quant_group_size,
            tma_aligned_scales,
        )
except (RuntimeError, AttributeError):
    # Custom op registration may fail in environments without a full torch
    # library setup (e.g., CPU-only tests that stub out the op registry).
    # In that case, keep the eager implementation live.
    pass
