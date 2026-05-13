# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import cache
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.deepseek_v4_copy_source_trace import (
    copy_source_trace,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_deep_gemm, has_tilelang
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

# tilelang is only available on CUDA platforms.  Keep a torch fallback available
# for correctness smoke on systems without tilelang or DeepGEMM.
# NOTE: `mhc_post_tilelang` below uses `T.pdl_sync()` / `T.pdl_trigger()` which
# require SM90+ Programmatic Dependent Launch. On SM70 (Volta), these calls
# fail TileLang's MarkCudaSyncCalls check. Treat SM70 as "tilelang not
# available for mhc" so the dedicated SM70 fast path is used instead; this
# does not affect other tilelang paths (e.g. sparse_prefill) that live in
# their own modules.
_TILELANG_AVAILABLE = False
if TYPE_CHECKING:
    import tilelang
    import tilelang.language as T
elif current_platform.is_cuda_alike() and has_tilelang() \
        and not current_platform.is_device_capability_family(70):
    import tilelang
    import tilelang.language as T

    _TILELANG_AVAILABLE = True
else:
    class _MissingTilelang:
        class PassConfigKey:
            TL_DISABLE_WARP_SPECIALIZED = object()
            TL_DISABLE_TMA_LOWER = object()
            TL_PTXAS_REGISTER_USAGE_LEVEL = object()

        JITKernel = object

        @staticmethod
        def jit(*args, **kwargs):
            del args, kwargs

            def decorator(fn):
                return fn

            return decorator

    tilelang = _MissingTilelang()  # type: ignore[assignment]
    T = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# SM70 optimised paths: replace pure-torch fallback with cuBLAS fp16 GEMM +
# a fused Triton post-GEMM kernel for norm / sigmoid / sinkhorn / weighted-sum.
# ---------------------------------------------------------------------------
def _is_sm70_fast_path_available() -> bool:
    """Return True when SM70 Triton path can replace the torch fallback."""
    if not envs.VLLM_SM70_MHC_FAST:
        return False
    if not current_platform.is_cuda_alike():
        return False
    cap = current_platform.get_device_capability()
    if cap is None:
        return False
    # Use fast path for SM70 (no tilelang/deep_gemm) up to SM80 exclusive
    return cap.major == 7


@triton.jit
def _mhc_pre_post_gemm_kernel(
    # GEMM result: [num_tokens, n_padded] (float16) – raw TurboMind output
    gemm_out_ptr,
    # Residual: [num_tokens, hc_mult * hidden_size] (fp16)
    residual_vec_ptr,
    # Residual 3D: [num_tokens, hc_mult, hidden_size] (fp16)
    residual_ptr,
    # Constants (small, loaded once)
    hc_scale_ptr,  # [3] float32
    hc_base_ptr,  # [hc_mult3] float32
    # Outputs
    post_mix_ptr,  # [num_tokens, hc_mult] float32
    comb_mix_ptr,  # [num_tokens, hc_mult * hc_mult] float32
    layer_input_ptr,  # [num_tokens, hidden_size] fp16
    # Scalar params
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    # Dims
    hidden_size: tl.constexpr,
    hc_hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    hc_mult3: tl.constexpr,
    n_padded: tl.constexpr,
    BLOCK_H: tl.constexpr,
    RMS_BLOCK: tl.constexpr,
):
    """Fully fused post-GEMM kernel for mhc_pre on SM70.

    One program per token. Fuses: GEMM output fp16→fp32 slice, RMS norm
    computation, sigmoid, sinkhorn normalisation, and weighted-sum
    layer_input — all in a single kernel launch.
    """
    pid = tl.program_id(0)

    # --- 1. Load GEMM output (fp16) and cast to fp32 ---
    mix_offsets = tl.arange(0, 32)  # hc_mult3 <= 24, pad to power-of-2
    mix_mask = mix_offsets < hc_mult3
    mixes_raw = tl.load(
        gemm_out_ptr + pid * n_padded + mix_offsets, mask=mix_mask, other=0.0
    ).to(tl.float32)

    # --- 2. Compute RMS norm inline (avoid separate kernel) ---
    # Accumulate squared sum over residual_vec in blocks
    sq_sum = tl.zeros([1], dtype=tl.float32)
    for rms_start in range(0, hc_hidden_size, RMS_BLOCK):
        rms_offsets = rms_start + tl.arange(0, RMS_BLOCK)
        rms_mask = rms_offsets < hc_hidden_size
        rv = tl.load(
            residual_vec_ptr + pid * hc_hidden_size + rms_offsets,
            mask=rms_mask, other=0.0,
        ).to(tl.float32)
        sq_sum += tl.sum(rv * rv, axis=0)
    rms_inv = tl.rsqrt(sq_sum / hc_hidden_size + rms_eps)

    # --- 3. Apply RMS norm to mixes ---
    mixes = mixes_raw * rms_inv

    # --- 4. Load constants ---
    hc_scale_0 = tl.load(hc_scale_ptr)
    hc_scale_1 = tl.load(hc_scale_ptr + 1)
    hc_scale_2 = tl.load(hc_scale_ptr + 2)

    # --- 5. pre_mix: first hc_mult elements ---
    pre_offsets = tl.arange(0, 4)
    pre_base = tl.load(hc_base_ptr + pre_offsets, mask=pre_offsets < hc_mult, other=0.0)
    pre_raw = tl.load(
        gemm_out_ptr + pid * n_padded + pre_offsets,
        mask=pre_offsets < hc_mult, other=0.0,
    ).to(tl.float32) * rms_inv
    pre_mix = tl.sigmoid(pre_raw * hc_scale_0 + pre_base) + hc_pre_eps

    # --- 6. post_mix: next hc_mult elements ---
    post_offsets = hc_mult + tl.arange(0, 4)
    post_base = tl.load(
        hc_base_ptr + post_offsets, mask=post_offsets < 2 * hc_mult, other=0.0
    )
    post_raw = tl.load(
        gemm_out_ptr + pid * n_padded + post_offsets,
        mask=post_offsets < 2 * hc_mult, other=0.0,
    ).to(tl.float32) * rms_inv
    post_mix = tl.sigmoid(post_raw * hc_scale_1 + post_base) * hc_post_mult_value

    tl.store(
        post_mix_ptr + pid * hc_mult + tl.arange(0, 4),
        post_mix, mask=tl.arange(0, 4) < hc_mult,
    )

    # --- 7. comb_mix: sinkhorn on [hc_mult, hc_mult] ---
    comb_offset_base = 2 * hc_mult
    comb_offsets = tl.arange(0, 16)
    comb_raw = tl.load(
        gemm_out_ptr + pid * n_padded + comb_offset_base + comb_offsets,
        mask=comb_offsets < hc_mult * hc_mult, other=0.0,
    ).to(tl.float32) * rms_inv
    comb_base = tl.load(
        hc_base_ptr + comb_offset_base + comb_offsets,
        mask=comb_offsets < hc_mult * hc_mult, other=0.0,
    )
    comb = comb_raw * hc_scale_2 + comb_base

    # Softmax per row
    row_idx = comb_offsets // hc_mult
    NEG_INF: tl.constexpr = -1e30
    row0_mask = row_idx == 0
    row1_mask = row_idx == 1
    row2_mask = row_idx == 2
    row3_mask = row_idx == 3
    max0 = tl.max(tl.where(row0_mask, comb, NEG_INF), axis=0)
    max1 = tl.max(tl.where(row1_mask, comb, NEG_INF), axis=0)
    max2 = tl.max(tl.where(row2_mask, comb, NEG_INF), axis=0)
    max3 = tl.max(tl.where(row3_mask, comb, NEG_INF), axis=0)
    row_max = tl.where(row0_mask, max0, tl.where(row1_mask, max1,
              tl.where(row2_mask, max2, max3)))
    comb = tl.exp(comb - row_max)
    sum0 = tl.sum(tl.where(row0_mask, comb, 0.0), axis=0)
    sum1 = tl.sum(tl.where(row1_mask, comb, 0.0), axis=0)
    sum2 = tl.sum(tl.where(row2_mask, comb, 0.0), axis=0)
    sum3 = tl.sum(tl.where(row3_mask, comb, 0.0), axis=0)
    row_sum = tl.where(row0_mask, sum0, tl.where(row1_mask, sum1,
              tl.where(row2_mask, sum2, sum3)))
    comb = comb / row_sum + hc_sinkhorn_eps

    # Column normalisation
    col_idx = comb_offsets % hc_mult
    col0_mask = col_idx == 0
    col1_mask = col_idx == 1
    col2_mask = col_idx == 2
    col3_mask = col_idx == 3
    cs0 = tl.sum(tl.where(col0_mask, comb, 0.0), axis=0)
    cs1 = tl.sum(tl.where(col1_mask, comb, 0.0), axis=0)
    cs2 = tl.sum(tl.where(col2_mask, comb, 0.0), axis=0)
    cs3 = tl.sum(tl.where(col3_mask, comb, 0.0), axis=0)
    col_sum = tl.where(col0_mask, cs0, tl.where(col1_mask, cs1,
              tl.where(col2_mask, cs2, cs3)))
    comb = comb / (col_sum + hc_sinkhorn_eps)

    for _ in range(sinkhorn_repeat - 1):
        rs0 = tl.sum(tl.where(row0_mask, comb, 0.0), axis=0)
        rs1 = tl.sum(tl.where(row1_mask, comb, 0.0), axis=0)
        rs2 = tl.sum(tl.where(row2_mask, comb, 0.0), axis=0)
        rs3 = tl.sum(tl.where(row3_mask, comb, 0.0), axis=0)
        row_s = tl.where(row0_mask, rs0, tl.where(row1_mask, rs1,
                 tl.where(row2_mask, rs2, rs3)))
        comb = comb / (row_s + hc_sinkhorn_eps)
        cs0_ = tl.sum(tl.where(col0_mask, comb, 0.0), axis=0)
        cs1_ = tl.sum(tl.where(col1_mask, comb, 0.0), axis=0)
        cs2_ = tl.sum(tl.where(col2_mask, comb, 0.0), axis=0)
        cs3_ = tl.sum(tl.where(col3_mask, comb, 0.0), axis=0)
        col_s = tl.where(col0_mask, cs0_, tl.where(col1_mask, cs1_,
                 tl.where(col2_mask, cs2_, cs3_)))
        comb = comb / (col_s + hc_sinkhorn_eps)

    tl.store(
        comb_mix_ptr + pid * hc_mult * hc_mult + comb_offsets,
        comb, mask=comb_offsets < hc_mult * hc_mult,
    )

    # --- 8. layer_input: einsum("nh,nhd->nd", pre_mix, residual) ---
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for hc_idx in range(hc_mult):
            r_vals = tl.load(
                residual_ptr + pid * hc_mult * hidden_size
                + hc_idx * hidden_size + h_offsets,
                mask=h_mask, other=0.0,
            ).to(tl.float32)
            pm_mask = tl.arange(0, 4) == hc_idx
            pm_val = tl.sum(tl.where(pm_mask, pre_mix, 0.0), axis=0)
            acc += pm_val * r_vals
        tl.store(
            layer_input_ptr + pid * hidden_size + h_offsets,
            acc.to(tl.float16), mask=h_mask,
        )


@triton.jit
def _mhc_post_fused_kernel(
    # comb_res_mix: [num_tokens, hc_mult, hc_mult] float32
    comb_ptr,
    # residual: [num_tokens, hc_mult, hidden_size] fp16
    residual_ptr,
    # post_layer_mix: [num_tokens, hc_mult, 1] float32
    post_ptr,
    # x (layer output): [num_tokens, hidden_size] fp16
    x_ptr,
    # output: [num_tokens, hc_mult, hidden_size] fp16
    out_ptr,
    # Dims
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused hc_post kernel: out[n,o,d] = sum_i(comb[n,i,o]*res[n,i,d])
                                         + post[n,o]*x[n,d]
    One program per token.
    """
    pid = tl.program_id(0)

    # Load comb matrix [hc_mult, hc_mult] = 16 floats
    comb_offsets = tl.arange(0, 16)
    comb = tl.load(
        comb_ptr + pid * hc_mult * hc_mult + comb_offsets,
        mask=comb_offsets < hc_mult * hc_mult, other=0.0,
    )

    # Load post_mix [hc_mult] floats
    post_offsets = tl.arange(0, 4)
    post = tl.load(
        post_ptr + pid * hc_mult + post_offsets,
        mask=post_offsets < hc_mult, other=0.0,
    )

    # Stream through hidden_size in blocks
    for h_start in range(0, hidden_size, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < hidden_size

        # Load x[n, d]
        x_vals = tl.load(
            x_ptr + pid * hidden_size + h_offsets,
            mask=h_mask, other=0.0,
        ).to(tl.float32)

        # Load residual[n, :, d] for all hc_mult rows
        # and compute output for each output row
        res = tl.zeros([4, BLOCK_H], dtype=tl.float32)
        for hc_idx in tl.static_range(4):
            r = tl.load(
                residual_ptr + pid * hc_mult * hidden_size
                + hc_idx * hidden_size + h_offsets,
                mask=h_mask, other=0.0,
            ).to(tl.float32)
            # Store in res[hc_idx, :]
            # Triton doesn't support 2D indexing easily; unroll manually
            if hc_idx == 0:
                res0 = r
            elif hc_idx == 1:
                res1 = r
            elif hc_idx == 2:
                res2 = r
            else:
                res3 = r

        # For each output row o: out[o,d] = sum_i(comb[i,o]*res[i,d]) + post[o]*x[d]
        for o_idx in tl.static_range(4):
            # Extract comb column o: comb[i, o] for i=0..3
            # comb is stored row-major: comb[i,o] = comb_flat[i*hc_mult + o]
            c0 = tl.sum(tl.where(comb_offsets == 0 * hc_mult + o_idx, comb, 0.0), axis=0)
            c1 = tl.sum(tl.where(comb_offsets == 1 * hc_mult + o_idx, comb, 0.0), axis=0)
            c2 = tl.sum(tl.where(comb_offsets == 2 * hc_mult + o_idx, comb, 0.0), axis=0)
            c3 = tl.sum(tl.where(comb_offsets == 3 * hc_mult + o_idx, comb, 0.0), axis=0)

            # Extract post[o]
            p_val = tl.sum(tl.where(post_offsets == o_idx, post, 0.0), axis=0)

            acc = c0 * res0 + c1 * res1 + c2 * res2 + c3 * res3 + p_val * x_vals
            tl.store(
                out_ptr + pid * hc_mult * hidden_size
                + o_idx * hidden_size + h_offsets,
                acc.to(tl.float16),
                mask=h_mask,
            )


def _mhc_pre_sm70_fast(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM70 fast path: TurboMind MMA_884 GEMM + fused Triton post-GEMM kernel.

    Uses sm70_f16_prepare + sm70_f16_gemm_out (Volta mma.sync.aligned.m8n8k4)
    for the GEMM when available, falls back to cuBLAS fp16 GEMM otherwise.
    A single Triton kernel handles norm/sigmoid/sinkhorn/mix.
    The prepared weight is cached on the fn tensor (lazy, once per parameter).
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    # Force contiguous residual up front. Under the FULL_DECODE_ONLY compile
    # graph partitioner, the residual arriving at this boundary may have
    # non-natural strides (from a view/reshape that inductor left behind);
    # the downstream .reshape(num_tokens, hc_hidden_size) silently returns a
    # non-contiguous view which the Triton load-then-cast-to-half path has
    # been observed to fault on at T=2048. Contiguous is cheap when already
    # so (torch.Tensor.is_contiguous short-circuit) and guarantees the
    # Triton kernel sees canonical strides.
    # See .kiro/specs/deepseek-v4-flash-compile-path-regression/.
    with copy_source_trace("mhc_pre.residual_contiguous"):
        if not residual.is_contiguous():
            residual = residual.contiguous()

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    residual_vec = residual_flat.reshape(num_tokens, hc_hidden_size)

    # Step 1: GEMM – [N, K] x [hc_mult3, K]^T → [N, hc_mult3]
    # Try TurboMind MMA_884 first (requires N%32==0 padding), fall back to cuBLAS.
    with copy_source_trace("mhc_pre.residual_half"):
        res_fp16 = (
            residual_vec
            if residual_vec.dtype == torch.float16
            else residual_vec.half()
        )
    with copy_source_trace("mhc_pre.residual_half_contiguous"):
        if not res_fp16.is_contiguous():
            res_fp16 = res_fp16.contiguous()

    _use_tm = getattr(fn, "_sm70_use_turbomind", None)
    if _use_tm is None:
        # Decide once: use TurboMind if sm70_f16_prepare is available and
        # K dimension satisfies alignment (K%16==0).
        _use_tm = False  # DISABLED: TurboMind MMA_884 has precision issues
        fn._sm70_use_turbomind = _use_tm  # type: ignore[attr-defined]

    if _use_tm:
        from vllm import _custom_ops as ops

        _tm = getattr(fn, "_sm70_tm_prepared", None)
        if _tm is None:
            with copy_source_trace("mhc_pre.fn_half"):
                fn_fp16 = fn.half()
            n_padded = ((hc_mult3 + 31) // 32) * 32
            if n_padded != hc_mult3:
                fn_padded = torch.zeros(
                    n_padded, hc_hidden_size,
                    dtype=torch.float16, device=fn.device,
                )
                fn_padded[:hc_mult3] = fn_fp16
            else:
                with copy_source_trace("mhc_pre.fn_half_contiguous"):
                    fn_padded = fn_fp16.contiguous()
            prepared = ops.sm70_f16_prepare(fn_padded)
            _tm = {
                "weight": prepared[0],
                "k_ld": int(prepared[1][0].item()),
                "n_padded": n_padded,
            }
            fn._sm70_tm_prepared = _tm  # type: ignore[attr-defined]

        n_padded = _tm["n_padded"]
        gemm_out_raw = torch.empty(
            num_tokens, n_padded, dtype=torch.float16, device=residual.device,
        )
        ops.sm70_f16_gemm_out(
            gemm_out_raw, res_fp16, _tm["weight"], _tm["k_ld"], False
        )
        gemm_out = gemm_out_raw
        n_padded = _tm["n_padded"]
    else:
        # cuBLAS fp16 fallback (still faster than float32 torch matmul)
        fn_fp16 = getattr(fn, "_sm70_fp16_cache", None)
        if fn_fp16 is None:
            with copy_source_trace("mhc_pre.fn_half_cache"):
                fn_fp16 = fn.half()
            fn._sm70_fp16_cache = fn_fp16  # type: ignore[attr-defined]
        gemm_out = res_fp16 @ fn_fp16.t()
        n_padded = hc_mult3

    # Allocate outputs
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.float16, device=residual.device
    )

    # Fully fused Triton kernel: reads GEMM fp16 output + residual_vec,
    # computes RMS norm inline, then sigmoid/sinkhorn/weighted-sum.
    # Eliminates 3 separate CUDA kernels (slice, float cast, squared-sum).
    BLOCK_H = min(1024, hidden_size)
    RMS_BLOCK = min(4096, hc_hidden_size)
    grid = (num_tokens,)
    with copy_source_trace("mhc_pre.residual_flat_half"):
        res_fp16_flat = (
            residual_flat
            if residual_flat.dtype == torch.float16
            else residual_flat.half()
        )
    _mhc_pre_post_gemm_kernel[grid](
        gemm_out,
        res_fp16,
        res_fp16_flat,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        hidden_size=hidden_size,
        hc_hidden_size=hc_hidden_size,
        hc_mult=hc_mult,
        hc_mult3=hc_mult3,
        n_padded=n_padded,
        BLOCK_H=BLOCK_H,
        RMS_BLOCK=RMS_BLOCK,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def _mhc_post_sm70_fast(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """SM70 fast path: fused Triton hc_post kernel."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # Ensure contiguous layout for Triton
    comb_flat = comb_res_mix.reshape(num_tokens, hc_mult * hc_mult)
    with copy_source_trace("mhc_post.comb_contiguous"):
        if not comb_flat.is_contiguous():
            comb_flat = comb_flat.contiguous()
    res_flat = residual_flat
    with copy_source_trace("mhc_post.residual_contiguous"):
        if not res_flat.is_contiguous():
            res_flat = res_flat.contiguous()
    post_flat = post_layer_mix.reshape(num_tokens, hc_mult)
    with copy_source_trace("mhc_post.post_contiguous"):
        if not post_flat.is_contiguous():
            post_flat = post_flat.contiguous()
    x_flat = x.reshape(num_tokens, hidden_size)
    with copy_source_trace("mhc_post.x_contiguous"):
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

    out = torch.empty(
        num_tokens, hc_mult, hidden_size,
        dtype=torch.float16, device=residual.device,
    )
    BLOCK_H = min(1024, hidden_size)
    grid = (num_tokens,)
    _mhc_post_fused_kernel[grid](
        comb_flat,
        res_flat,
        post_flat,
        x_flat,
        out,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        BLOCK_H=BLOCK_H,
    )
    return out.view(*outer_shape, hc_mult, hidden_size)


@cache
def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        T.pdl_sync()
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])
        T.pdl_trigger()


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    if not _TILELANG_AVAILABLE or not has_deep_gemm():
        if _is_sm70_fast_path_available():
            # Inductor graph-partitioner may export a non-fp16 residual at
            # this custom-op boundary (see compile-path-regression spec).
            # Shield the SM70 fast path's fp16 contract explicitly; strict
            # no-op when dtype already matches.
            from vllm.model_executor.layers.sm70_compile_boundary_shield import (
                ensure_boundary_dtype,
            )
            with copy_source_trace("mhc_pre.boundary_residual"):
                residual = ensure_boundary_dtype(residual, torch.float16)
            return _mhc_pre_sm70_fast(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
        return _mhc_pre_torch_fallback(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    # these number are from deepgemm kernel impl
    block_k = 64
    block_m = 64
    n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))

    post_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )

    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn_flat,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def _mhc_pre_torch_fallback(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    with copy_source_trace("mhc_pre_fallback.residual_float"):
        residual_flat = residual.reshape(-1, hc_mult, hidden_size).float()
    residual_vec = residual_flat.reshape(-1, hc_hidden_size)
    mixes = residual_vec @ fn.t()
    rms = torch.rsqrt(
        residual_vec.square().sum(dim=-1, keepdim=True) / hc_hidden_size + rms_eps
    )
    mixes = mixes * rms

    pre_mix = torch.sigmoid(
        mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    ) + hc_pre_eps
    post_mix = torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = post_mix * hc_post_mult_value

    comb_mix = mixes[:, 2 * hc_mult :].reshape(-1, hc_mult, hc_mult)
    comb_mix = comb_mix * hc_scale[2] + hc_base[2 * hc_mult :].reshape(
        hc_mult, hc_mult
    )
    comb_mix = torch.softmax(comb_mix, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    with copy_source_trace("mhc_pre_fallback.layer_input_to_dtype"):
        layer_input = torch.einsum("nh,nhd->nd", pre_mix, residual_flat).to(
            residual.dtype
        )

    post_mix = post_mix.reshape(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.reshape(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.reshape(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def _mhc_pre_fake_layer_input_dtype(residual: torch.Tensor) -> torch.dtype:
    if not _TILELANG_AVAILABLE or not has_deep_gemm():
        if _is_sm70_fast_path_available():
            return torch.float16
        return residual.dtype
    return torch.bfloat16


def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=_mhc_pre_fake_layer_input_dtype(residual),
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a,
    b,
    c,
    d,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
        T.pdl_trigger()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if not _TILELANG_AVAILABLE:
        if _is_sm70_fast_path_available():
            from vllm.model_executor.layers.sm70_compile_boundary_shield import (
                ensure_boundary_dtype,
            )
            with copy_source_trace("mhc_post.boundary_x"):
                x = ensure_boundary_dtype(x, torch.float16)
            with copy_source_trace("mhc_post.boundary_residual"):
                residual = ensure_boundary_dtype(residual, torch.float16)
            return _mhc_post_sm70_fast(
                x, residual, post_layer_mix, comb_res_mix
            )
        return _mhc_post_torch_fallback(
            x, residual, post_layer_mix, comb_res_mix
        )

    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def _mhc_post_torch_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    with copy_source_trace("mhc_post_fallback.residual_float"):
        residual_flat = residual.reshape(-1, hc_mult, hidden_size).float()
    with copy_source_trace("mhc_post_fallback.x_float"):
        x_flat = x.reshape(-1, hidden_size).float()
    with copy_source_trace("mhc_post_fallback.post_float"):
        post_flat = post_layer_mix.reshape(-1, hc_mult, 1).float()
    with copy_source_trace("mhc_post_fallback.comb_float"):
        comb_flat = comb_res_mix.reshape(-1, hc_mult, hc_mult).float()

    out = torch.einsum("nio,nih->noh", comb_flat, residual_flat)
    out = out + post_flat * x_flat.unsqueeze(-2)
    with copy_source_trace("mhc_post_fallback.output_to_dtype"):
        return out.reshape(*outer_shape, hc_mult, hidden_size).to(residual.dtype)


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="mhc_pre",
    op_func=mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="mhc_post",
    op_func=mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)
