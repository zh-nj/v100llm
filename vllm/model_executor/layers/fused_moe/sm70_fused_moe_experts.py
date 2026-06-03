# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (V100) fused MoE expert orchestration for dsv4f MXFP4 experts.

This module is part of the ``deepgemm-megamoe-sm70-port`` spec (Requirement 2:
"SM70 上的融合 MoE 前向算子（单卡, MXFP4）"). It hosts the Python orchestration that
drives the hand-written fused CUDA kernel ``ops.sm70_fused_moe_out`` (the MXFP4
signature added by tasks 4.1/4.4) which fuses ``linear1 -> SwiGLU -> linear2``
for one ``(expert, m_block)`` tile while keeping the ``gate/up`` (``[M_b, 2*I]``)
and intermediate ``h`` (``[M_b, I]``) activations on chip (never materialised to
HBM — the core saving over the three-kernel baseline, R2.1 / R2.6).

dsv4f's V100 MoE expert weights are **MXFP4** (E2M1 FP4 nibbles packed two per
byte plus a per-32 E8M0 block scale). The kernel dequantizes MXFP4 -> fp16 in
software before the first-generation ``mma.sync`` (HMMA) GEMM — there are no
FP4/FP8 tensor cores on SM70 (R2.2). The software dequant is the shared
:func:`mxfp4_dequant_to_fp16` (``sm70_moe_reference.py``), so the per-operator
L0/L1 / masked realizations and the fused kernel decode the *same* numbers.

It provides two things:

* :class:`SM70MXFP4QuantParams` — the per-expert MXFP4 weight pack the kernel op
  consumes directly: the packed ``w13_weight`` / ``w13_weight_scale`` (gate/up)
  and ``w2_weight`` / ``w2_weight_scale`` (down) tensors plus the ``group_size``
  (32) and ``num_experts`` / ``hidden_K`` / ``inter_I`` dims. Build it from raw
  tensors (:meth:`SM70MXFP4QuantParams.from_mxfp4_weights`) or from a prepared
  ``Mxfp4SM70MoEMethod`` layer (:meth:`SM70MXFP4QuantParams.from_layer`, which
  reads the raw MXFP4 params task 5.4 stashes before
  ``process_weights_after_loading`` deletes them — see the method for the stash
  contract).

* :class:`SM70FusedMoEExperts` — the orchestration layer. ``forward`` builds the
  grouped/contiguous token layout with :class:`SM70MoELayoutAdapter`
  (task 2), then runs the expert FFN over that layout using one of four
  progressively-more-fused, numerically-equivalent strategies selected by
  :class:`SM70FusionLevel` (``SM70FusedConfig.fusion_level``):

  * **L0** — un-fused baseline: linear1, ``SiluAndMul`` and linear2 as three
    separate steps (per-operator, dense fp16; equivalence baseline / fallback).
  * **L1** — linear1 + SwiGLU epilogue fused, reusing buffers to drop
    the Python-level ``[M, 2*I]`` intermediate materialization.
  * **L2** (default) — eliminate the ``[M, 2*I]`` / ``[M, I]`` HBM round trip by
    chaining linear1 -> linear2 inside the fused CUDA kernel.
  * **L3** — single mega-kernel, fully on-chip token-block flow.

  L2/L3 both dispatch to the MXFP4 ``ops.sm70_fused_moe_out`` (the hand-written
  CUDA kernel that decodes MXFP4 -> fp16 on chip and realises the on-chip
  ``linear1 -> SwiGLU -> linear2`` hand-off so the intermediate activation is
  never round-tripped through HBM, R2.1/R2.6). L0/L1 are the per-operator
  realizations (dense fp16 from :func:`mxfp4_dequant_to_fp16`) used as the
  equivalence baseline and a CPU-friendly fallback. ``forward`` finally
  ``combine``\\ s (unpermute + weighted reduce, fp32 router weights) the
  per-expert outputs back to per-token order.

All compute is ``float16`` (V100 has no bfloat16); MXFP4 (software dequant to
fp16) is the only quantized path (no FP8/FP4 tensor cores, R2.2).

Legacy AWQ int4 shim
--------------------
:class:`SM70QuantParams` (the AWQ int4 weight pack + its ``FusedStridedPtr``
weight-prep contract) is retained below for backward-compatible imports by the
out-of-scope ``AWQSM70MoEMethod`` (AWQ int4 is **not** a dsv4f target — see the
spec "超出范围"). The fused CUDA mega-kernel (L2/L3) now consumes MXFP4 only;
the AWQ pack still drives the per-operator dense-fp16 L0/L1 / masked paths via
its own ``to_dense_fp16``. New dsv4f code SHALL use
:class:`SM70MXFP4QuantParams`.

Legacy AWQ weight-prep contract (``SM70QuantParams`` only)
----------------------------------------------------------
The following describes the *legacy AWQ int4* ``FusedStridedPtr`` contract used
by :class:`SM70QuantParams` (retained for ``AWQSM70MoEMethod`` import
compatibility; not a dsv4f target). The MXFP4 path
(:class:`SM70MXFP4QuantParams`) does **not** use ``StridedPtr`` arrays — it
hands the packed weight/scale tensors straight to ``ops.sm70_fused_moe_out``.

The AWQ kernel resolves per-expert pointers from four ``StridedPtr`` arrays. For
expert ``e`` (hidden ``K``, intermediate ``I``, group size ``gs``):

* ``w13_ptrs_w[e]`` -> uint32 packed int4 ``[2*I, K/8]``; **gate** rows occupy
  ``[0, I)`` then **up** rows ``[I, 2*I)``; ``stride = K/8``. Each uint32 packs 8
  contraction-``K`` nibbles in *natural* K order (nibble ``o`` at bits
  ``[4o, 4o+4)``) — the kernel's ``dequant_s4_to_f16x2_sm70`` emits the AWQ
  interleaved lane order ``{0,4,1,5,2,6,3,7}`` and ``stage_dequant_w_tile``
  de-interleaves back to natural K, so the packing here must be plain natural-K.
* ``w13_ptrs_s[e]`` -> fp16 ``[2*(2*I), K/gs]``: the ``[2*I, K/gs]`` per-group
  **scales** block (gate then up) immediately followed by an identically shaped
  **zero-point** block (gate then up); ``stride = K/gs``. Zero points are the
  integer AWQ zero *codes* (0..15) stored as fp16, matching the affine de-quant
  ``w = (q_code - zero_code) * scale``.
* ``w2_ptrs_w[e]`` -> uint32 packed int4 ``[K, I/8]`` (down output-feature rows,
  packed contraction ``I``); ``stride = I/8``.
* ``w2_ptrs_s[e]`` -> fp16 ``[2*K, I/gs]``: ``[K, I/gs]`` scales then ``[K, I/gs]``
  zeros; ``stride = I/gs``.

The original AWQ checkpoint stores weights transposed relative to this contract
(input features as rows, packed *output* features as columns, 8 output features
per uint32). :meth:`SM70QuantParams.from_awq_weights` therefore unpacks AWQ to
dense codes, transposes so output features become rows, and re-packs along the
contraction dim in natural order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusedConfig,
    SM70FusionLevel,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_layout import (
    SM70MoELayoutAdapter,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_reference import (
    mxfp4_dequant_to_fp16,
)

__all__ = [
    "SM70MXFP4QuantParams",
    "SM70QuantParams",
    "SM70FusedMoEExperts",
]

# MXFP4 micro-scaling block size: one E8M0 block scale per 32 packed E2M1
# elements along the contraction (K / I) axis. dsv4f experts use 32.
_MXFP4_GROUP_SIZE = 32

# AWQ 4-bit reverse-pack order. A packed uint32 stores 8 logical values whose
# nibble positions are permuted by this order; undoing it recovers the natural
# (sequential) value order. Mirrors ``reverse_awq_order`` used throughout the
# repo (e.g. ``awq_triton.py`` / ``tests/.../test_awq_triton.py``) and the lane
# order emitted by the kernel's ``dequant_s4_to_f16x2_sm70``.
_AWQ_REVERSE_ORDER: tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)
_PACK_FACTOR = 8  # int4 values per int32

# Default token-count threshold that distinguishes a *prefill* context (large M)
# from a *decode* context (small M) when no CUDA-graph capture signal is present.
# A batch with at most this many tokens is treated as decode (masked layout);
# anything larger is prefill (contiguous layout). Decode steps emit one token
# per running sequence, so even a large concurrent batch stays well under a
# typical prefill chunk; 16 mirrors the default ``m_block`` / kernel ``M_TILE``
# granularity and keeps the boundary aligned with the grouped-GEMM block. This
# is only a heuristic for the *non-capturing* case — an active CUDA-graph capture
# always forces ``masked`` regardless of M (R2.7).
_DEFAULT_PREFILL_MIN_TOKENS: int = 16


def _reverse_awq_order_cols(codes: torch.Tensor) -> torch.Tensor:
    """Apply the AWQ reverse-pack column permutation to a ``[R, C]`` tensor.

    ``C`` must be a multiple of 8. Returns a tensor of the same shape with the
    last dim de-interleaved into natural order (inverse of the AWQ packing
    permutation), matching ``reverse_awq_order`` in ``test_awq_triton.py``.
    """
    cols = codes.shape[-1]
    assert cols % _PACK_FACTOR == 0, (
        f"reverse_awq_order expects a multiple of 8 columns, got {cols}"
    )
    order = torch.arange(cols, dtype=torch.int64, device=codes.device)
    order = order.view(-1, _PACK_FACTOR)[:, list(_AWQ_REVERSE_ORDER)].reshape(-1)
    return codes.index_select(-1, order)


def _awq_unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack an AWQ int32 weight/zero tensor to dense int4 *codes*.

    Args:
        packed: ``[R, C // 8]`` int32 AWQ-packed tensor (weights or zero points).

    Returns:
        ``[R, C]`` int32 tensor of integer codes in ``[0, 15]`` in *natural*
        column order (the AWQ pack interleave is undone). This matches the
        ``iweights``/``zeros`` intermediate of ``awq_dequantize_torch``.
    """
    if packed.dim() != 2:
        raise ValueError(
            f"_awq_unpack_codes expects a 2-D tensor, got shape {tuple(packed.shape)}"
        )
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    # [R, C//8, 8] nibble extraction, then flatten to [R, C].
    codes = torch.bitwise_right_shift(packed[:, :, None], shifts[None, None, :])
    codes = codes.to(torch.int32).reshape(packed.shape[0], -1)
    codes = _reverse_awq_order_cols(codes)
    return torch.bitwise_and(codes, 0xF)


def _pack_natural_k(codes: torch.Tensor) -> torch.Tensor:
    """Pack dense int4 codes along the last (contraction) dim in natural order.

    Args:
        codes: ``[N, K]`` int tensor of codes in ``[0, 15]``; ``K % 8 == 0``.

    Returns:
        ``[N, K // 8]`` int32 tensor where each int32 packs 8 consecutive ``K``
        codes with code ``o`` at bit offset ``4 * o`` (natural order). The
        int64 accumulation is narrowed to int32, which preserves the exact
        two's-complement bit pattern the kernel reinterprets as ``uint32``.
    """
    n, k = codes.shape
    assert k % _PACK_FACTOR == 0, (
        f"_pack_natural_k expects K to be a multiple of 8, got {k}"
    )
    grouped = codes.view(n, k // _PACK_FACTOR, _PACK_FACTOR).to(torch.int64)
    shifts = torch.arange(0, 32, 4, device=codes.device, dtype=torch.int64)
    packed = (grouped << shifts[None, None, :]).sum(dim=-1)
    # Narrowing int64 -> int32 wraps modulo 2**32 (two's complement), giving the
    # exact uint32 bit pattern the kernel's dequant reads.
    return packed.to(torch.int32)


def _unpack_natural_k(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_natural_k`: recover dense codes from natural pack.

    Args:
        packed: ``[N, K // 8]`` int32 tensor where each int32 packs 8 consecutive
            ``K`` codes with code ``o`` at bit offset ``4 * o`` (natural order,
            i.e. *no* AWQ reverse-order interleave — this undoes exactly what
            :func:`_pack_natural_k` produced).

    Returns:
        ``[N, K]`` int32 tensor of integer codes in ``[0, 15]`` in natural ``K``
        order.
    """
    if packed.dim() != 2:
        raise ValueError(
            f"_unpack_natural_k expects a 2-D tensor, got shape {tuple(packed.shape)}"
        )
    n, k_packed = packed.shape
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    codes = torch.bitwise_right_shift(packed[:, :, None], shifts[None, None, :])
    codes = torch.bitwise_and(codes, 0xF).to(torch.int32)
    return codes.reshape(n, k_packed * _PACK_FACTOR)


def _silu_and_mul(
    gate_up: torch.Tensor, swiglu_limit: float = 0.0
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` for a ``[..., 2 * I]`` tensor.

    Mirrors the SwiGLU (``kGatedSilu``) semantics used by the reference path
    (:func:`sm70_moe_reference`) and the fused kernel epilogue: ``silu`` applied
    to the first half (gate) multiplied element-wise by the second half (up).

    When ``swiglu_limit > 0`` the DeepSeek-V4 style SwiGLU clamp is applied
    BEFORE silu/mul (matching ``swiglu_limit_func`` / the kernel epilogue):
    ``gate = min(gate, limit)`` and ``up = clamp(up, -limit, +limit)``. With
    ``swiglu_limit <= 0`` (the default) it reduces to plain ``silu(gate)*up``
    and reuses the compiled ``torch.ops._C.silu_and_mul`` op when available on
    CUDA (the exact op the production path uses), falling back to the
    torch-native equivalent otherwise. The computation runs in the input dtype
    (fp16 for this path).
    """
    assert gate_up.shape[-1] % 2 == 0, (
        f"silu_and_mul expects an even last dim, got {gate_up.shape[-1]}"
    )
    d = gate_up.shape[-1] // 2

    if swiglu_limit and swiglu_limit > 0:
        # Clamp gate/up to the limit (DeepSeek-V4 SwiGLU limit), then silu*mul.
        gate = torch.clamp(gate_up[..., :d], max=swiglu_limit)
        up = torch.clamp(gate_up[..., d:], min=-swiglu_limit, max=swiglu_limit)
        return torch.nn.functional.silu(gate) * up

    c_op = getattr(getattr(torch.ops, "_C", None), "silu_and_mul", None)
    if c_op is not None and gate_up.is_cuda:
        gate_up = gate_up.contiguous()
        out = torch.empty(
            (*gate_up.shape[:-1], d), dtype=gate_up.dtype, device=gate_up.device
        )
        c_op(out, gate_up)
        return out

    gate = gate_up[..., :d]
    up = gate_up[..., d:]
    return torch.nn.functional.silu(gate) * up


@dataclass
class SM70QuantParams:
    """Per-expert AWQ int4 weight pack for the SM70 fused MoE kernel.

    Holds both the natural-layout stacked weight/scale buffers (kept alive for
    pointer lifetime + CPU shape verification) and the four ``FusedStridedPtr``
    arrays the kernel consumes. Built by :meth:`from_awq_weights`.

    Fields (see design §Data Models / the module docstring weight-prep contract):

    * ``w13_qweight``: ``int32 [E, 2*I, K//8]`` — natural-K packed gate/up int4
      weights (gate rows then up rows).
    * ``w13_scales``: ``fp16 [E, 2*(2*I), K//gs]`` — gate/up per-group scales
      block followed by the zero-point (code) block.
    * ``w2_qweight``: ``int32 [E, K, I//8]`` — natural-I packed down int4 weights.
    * ``w2_scales``: ``fp16 [E, 2*K, I//gs]`` — down scales block then zeros block.
    * ``w13_ptrs_w`` / ``w13_ptrs_s`` / ``w2_ptrs_w`` / ``w2_ptrs_s``: ``uint8``
      ``[E * 16]`` ``FusedStridedPtr`` arrays (one 16-byte record per expert)
      produced by ``ops.awq_moe_build_strided_ptrs``.
    * ``num_experts`` / ``hidden_K`` / ``inter_I`` / ``group_size``: dims passed
      straight to ``ops.sm70_fused_moe_out``.
    * ``hidden_logical_size``: logical (unpadded) hidden size used to slice the
      kernel output before ``combine`` (``<= hidden_K`` when the AWQ method
      padded the hidden dim for alignment).
    * ``interleave_gated_silu``: documents the SwiGLU (``kGatedSilu``) semantics;
      the natural pack here lays gate then up explicitly so the flag is purely
      informational for this path.
    """

    w13_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_qweight: torch.Tensor
    w2_scales: torch.Tensor

    w13_ptrs_w: torch.Tensor
    w13_ptrs_s: torch.Tensor
    w2_ptrs_w: torch.Tensor
    w2_ptrs_s: torch.Tensor

    num_experts: int
    hidden_K: int
    inter_I: int
    group_size: int
    hidden_logical_size: int
    interleave_gated_silu: bool = True

    @staticmethod
    def repack_w13(
        w13_qweight: torch.Tensor,
        w13_scales: torch.Tensor,
        w13_qzeros: torch.Tensor,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-pack one expert's AWQ gate/up weights into the kernel's layout.

        Args (AWQ checkpoint orientation, matching ``AWQSM70MoEMethod``):
            w13_qweight: ``int32 [K, 2*I // 8]`` (input features as rows, packed
                output features as columns).
            w13_scales:  ``fp16  [K // gs, 2*I]`` per-group scales.
            w13_qzeros:  ``int32 [K // gs, 2*I // 8]`` packed zero points.
            group_size: AWQ group size (32 / 64 / 128).

        Returns:
            ``(qweight, scales)`` where
            ``qweight`` is ``int32 [2*I, K // 8]`` natural-K packed (gate rows
            then up rows) and ``scales`` is ``fp16 [2*(2*I), K // gs]`` (scales
            block then zero-code block).
        """
        k = w13_qweight.shape[0]
        two_i = w13_qweight.shape[1] * _PACK_FACTOR
        # Dense codes in natural [K, 2*I] order, then transpose so output
        # features (gate then up) become rows: [2*I, K].
        w_codes = _awq_unpack_codes(w13_qweight)  # [K, 2*I]
        w_codes_t = w_codes.transpose(0, 1).contiguous()  # [2*I, K]
        qweight = _pack_natural_k(w_codes_t)  # [2*I, K//8]

        # Scales: [K//gs, 2*I] -> [2*I, K//gs].
        scales_t = w13_scales.transpose(0, 1).contiguous().to(torch.float16)
        # Zero points: unpack codes [K//gs, 2*I] -> transpose -> fp16 [2*I, K//gs].
        z_codes = _awq_unpack_codes(w13_qzeros).to(torch.float16)  # [K//gs, 2*I]
        zeros_t = z_codes.transpose(0, 1).contiguous()  # [2*I, K//gs]
        scales = torch.cat([scales_t, zeros_t], dim=0)  # [2*(2*I), K//gs]

        assert qweight.shape == (two_i, k // _PACK_FACTOR), qweight.shape
        assert scales.shape == (2 * two_i, k // group_size), scales.shape
        return qweight, scales

    @staticmethod
    def repack_w2(
        w2_qweight: torch.Tensor,
        w2_scales: torch.Tensor,
        w2_qzeros: torch.Tensor,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-pack one expert's AWQ down weights into the kernel's layout.

        Args (AWQ checkpoint orientation):
            w2_qweight: ``int32 [I, K // 8]`` (input features ``I`` as rows,
                packed output features ``K`` as columns).
            w2_scales:  ``fp16  [I // gs, K]`` per-group scales.
            w2_qzeros:  ``int32 [I // gs, K // 8]`` packed zero points.
            group_size: AWQ group size.

        Returns:
            ``(qweight, scales)`` where ``qweight`` is ``int32 [K, I // 8]``
            natural-I packed and ``scales`` is ``fp16 [2*K, I // gs]`` (scales
            block then zero-code block).
        """
        i = w2_qweight.shape[0]
        k = w2_qweight.shape[1] * _PACK_FACTOR
        w_codes = _awq_unpack_codes(w2_qweight)  # [I, K]
        w_codes_t = w_codes.transpose(0, 1).contiguous()  # [K, I]
        qweight = _pack_natural_k(w_codes_t)  # [K, I//8]

        scales_t = w2_scales.transpose(0, 1).contiguous().to(torch.float16)  # [K, I//gs]
        z_codes = _awq_unpack_codes(w2_qzeros).to(torch.float16)  # [I//gs, K]
        zeros_t = z_codes.transpose(0, 1).contiguous()  # [K, I//gs]
        scales = torch.cat([scales_t, zeros_t], dim=0)  # [2*K, I//gs]

        assert qweight.shape == (k, i // _PACK_FACTOR), qweight.shape
        assert scales.shape == (2 * k, i // group_size), scales.shape
        return qweight, scales

    @classmethod
    def from_awq_weights(
        cls,
        *,
        w13_qweight: torch.Tensor,
        w13_scales: torch.Tensor,
        w13_qzeros: torch.Tensor,
        w2_qweight: torch.Tensor,
        w2_scales: torch.Tensor,
        w2_qzeros: torch.Tensor,
        group_size: int,
        hidden_logical_size: int | None = None,
        interleave_gated_silu: bool = True,
        build_strided_ptrs: bool = True,
    ) -> "SM70QuantParams":
        """Build :class:`SM70QuantParams` from stacked per-expert AWQ weights.

        The input tensors follow the AWQ-checkpoint orientation created by
        ``AWQSM70MoEMethod.create_weights`` (before its TurboMind conversion):

        * ``w13_qweight``: ``int32 [E, K, 2*I // 8]``
        * ``w13_scales``:  ``fp16  [E, K // gs, 2*I]``
        * ``w13_qzeros``:  ``int32 [E, K // gs, 2*I // 8]``
        * ``w2_qweight``:  ``int32 [E, I, K // 8]``
        * ``w2_scales``:   ``fp16  [E, I // gs, K]``
        * ``w2_qzeros``:   ``int32 [E, I // gs, K // 8]``

        Args:
            group_size: AWQ group size (32 / 64 / 128); must divide ``K`` and
                ``I``.
            hidden_logical_size: logical (unpadded) hidden size for slicing the
                kernel output before ``combine``. Defaults to ``K``.
            interleave_gated_silu: recorded on the result (informational here).
            build_strided_ptrs: when ``True`` (default) build the
                ``FusedStridedPtr`` arrays via ``ops.awq_moe_build_strided_ptrs``
                (requires CUDA + the compiled op). Set ``False`` to only re-pack
                the natural-layout weight buffers (CPU-friendly; the pointer
                arrays are left as empty placeholders) — used for shape/contract
                verification where no GPU is available.

        Returns:
            A fully populated :class:`SM70QuantParams`.
        """
        if group_size not in (32, 64, 128):
            raise ValueError(
                f"SM70QuantParams supports group_size 32/64/128, got {group_size}"
            )
        for name, t in (
            ("w13_qweight", w13_qweight),
            ("w13_scales", w13_scales),
            ("w13_qzeros", w13_qzeros),
            ("w2_qweight", w2_qweight),
            ("w2_scales", w2_scales),
            ("w2_qzeros", w2_qzeros),
        ):
            if t.dim() != 3:
                raise ValueError(
                    f"{name} must be a 3-D stacked [E, ...] tensor, got "
                    f"shape {tuple(t.shape)}"
                )

        num_experts = w13_qweight.shape[0]
        hidden_k = w13_qweight.shape[1]
        two_i = w13_qweight.shape[2] * _PACK_FACTOR
        inter_i = two_i // 2

        # Cross-check the shapes implied by w2 against w13.
        if w2_qweight.shape[1] != inter_i:
            raise ValueError(
                f"inter_I mismatch: w13 implies I={inter_i} but w2_qweight rows="
                f"{w2_qweight.shape[1]}"
            )
        if w2_qweight.shape[2] * _PACK_FACTOR != hidden_k:
            raise ValueError(
                f"hidden_K mismatch: w13 implies K={hidden_k} but w2_qweight "
                f"packs K={w2_qweight.shape[2] * _PACK_FACTOR}"
            )
        if hidden_k % group_size != 0:
            raise ValueError(
                f"hidden_K={hidden_k} not divisible by group_size={group_size}"
            )
        if inter_i % group_size != 0:
            raise ValueError(
                f"inter_I={inter_i} not divisible by group_size={group_size}"
            )

        w13_qw_list, w13_sc_list = [], []
        w2_qw_list, w2_sc_list = [], []
        for e in range(num_experts):
            qw13, sc13 = cls.repack_w13(
                w13_qweight[e], w13_scales[e], w13_qzeros[e], group_size
            )
            qw2, sc2 = cls.repack_w2(
                w2_qweight[e], w2_scales[e], w2_qzeros[e], group_size
            )
            w13_qw_list.append(qw13)
            w13_sc_list.append(sc13)
            w2_qw_list.append(qw2)
            w2_sc_list.append(sc2)

        w13_qw = torch.stack(w13_qw_list).contiguous()  # [E, 2*I, K//8]
        w13_sc = torch.stack(w13_sc_list).contiguous()  # [E, 2*(2*I), K//gs]
        w2_qw = torch.stack(w2_qw_list).contiguous()  # [E, K, I//8]
        w2_sc = torch.stack(w2_sc_list).contiguous()  # [E, 2*K, I//gs]

        if build_strided_ptrs:
            # Reuse the verified StridedPtr builder. The weight ptr stride is the
            # packed contraction columns (qw_ld); the scale ptr stride is the
            # per-group columns (sc_ld). These match the kernel's expectations:
            #   w13: weight stride = K/8,  scale stride = K/gs
            #   w2 : weight stride = I/8,  scale stride = I/gs
            w13_ptrs = ops.awq_moe_build_strided_ptrs(
                w13_qw, w13_sc, hidden_k // _PACK_FACTOR, hidden_k // group_size,
                num_experts,
            )
            w2_ptrs = ops.awq_moe_build_strided_ptrs(
                w2_qw, w2_sc, inter_i // _PACK_FACTOR, inter_i // group_size,
                num_experts,
            )
            w13_ptrs_w, w13_ptrs_s = w13_ptrs[0], w13_ptrs[1]
            w2_ptrs_w, w2_ptrs_s = w2_ptrs[0], w2_ptrs[1]
        else:
            empty = torch.empty(0, dtype=torch.uint8, device=w13_qw.device)
            w13_ptrs_w = w13_ptrs_s = w2_ptrs_w = w2_ptrs_s = empty

        return cls(
            w13_qweight=w13_qw,
            w13_scales=w13_sc,
            w2_qweight=w2_qw,
            w2_scales=w2_sc,
            w13_ptrs_w=w13_ptrs_w,
            w13_ptrs_s=w13_ptrs_s,
            w2_ptrs_w=w2_ptrs_w,
            w2_ptrs_s=w2_ptrs_s,
            num_experts=num_experts,
            hidden_K=hidden_k,
            inter_I=inter_i,
            group_size=group_size,
            hidden_logical_size=(
                hidden_k if hidden_logical_size is None else hidden_logical_size
            ),
            interleave_gated_silu=interleave_gated_silu,
        )

    def _dequant_block(
        self, qweight: torch.Tensor, scales_zeros: torch.Tensor, contraction: int
    ) -> torch.Tensor:
        """De-quantize one natural-packed int4 block to dense fp16 weights.

        Implements the affine AWQ de-quant ``w = (q_code - zero_code) * scale``
        the kernel performs, but in dense torch so the L0/L1 per-operator paths
        can reuse the *same* numerical weights the fused kernel consumes.

        Args:
            qweight: ``int32 [R, contraction // 8]`` natural-packed codes
                (output features ``R`` as rows, contraction dim packed columns).
            scales_zeros: ``fp16 [2 * R, contraction // gs]`` — the per-group
                scale block ``[R, contraction // gs]`` followed by the
                zero-point (code) block of the same shape.
            contraction: the unpacked contraction dimension length.

        Returns:
            ``fp16 [R, contraction]`` dense weights.
        """
        gs = self.group_size
        rows = qweight.shape[0]
        codes = _unpack_natural_k(qweight).to(torch.float16)  # [R, contraction]
        scales = scales_zeros[:rows]  # [R, contraction // gs]
        zeros = scales_zeros[rows:]  # [R, contraction // gs]
        # Expand each per-group scale/zero across its `gs` contraction columns.
        scales_full = scales.repeat_interleave(gs, dim=1)  # [R, contraction]
        zeros_full = zeros.repeat_interleave(gs, dim=1)  # [R, contraction]
        assert scales_full.shape == (rows, contraction), scales_full.shape
        return ((codes - zeros_full) * scales_full).to(torch.float16)

    def to_dense_fp16(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct dense fp16 expert weights (inverse of the int4 pack).

        Returns ``(w1, w2)`` in the *reference* orientation used by
        :func:`sm70_moe_reference` and ``fused_experts``:

        * ``w1``: ``fp16 [E, 2*I, K]`` gate/up (linear1) weights (gate rows then
          up rows along the output axis).
        * ``w2``: ``fp16 [E, K, I]`` down (linear2) weights.

        This de-quantizes the natural-packed int4 weights stored on this object
        (``w = (q_code - zero_code) * scale``) so the L0/L1 per-operator paths
        operate on exactly the weights the fused kernel decodes on-chip — making
        their outputs numerically equivalent to the fused path (and to the AWQ
        per-operator reference) by construction (R2.5 / Property 1).
        """
        w1_list, w2_list = [], []
        for e in range(self.num_experts):
            w1_list.append(
                self._dequant_block(
                    self.w13_qweight[e], self.w13_scales[e], self.hidden_K
                )
            )  # [2*I, K]
            w2_list.append(
                self._dequant_block(
                    self.w2_qweight[e], self.w2_scales[e], self.inter_I
                )
            )  # [K, I]
        w1 = torch.stack(w1_list).contiguous()  # [E, 2*I, K]
        w2 = torch.stack(w2_list).contiguous()  # [E, K, I]
        return w1, w2


@dataclass
class SM70MXFP4QuantParams:
    """Per-expert dsv4f **MXFP4** weight pack for the SM70 fused MoE path.

    Holds the *packed* MXFP4 expert weights exactly as dsv4f stores them — E2M1
    FP4 nibbles packed two-per-byte plus a per-32 E8M0 block scale — so the
    hand-written fused CUDA kernel (``ops.sm70_fused_moe_out``, L2/L3) can
    consume them straight from HBM and dequantize MXFP4 -> fp16 *on chip* (no
    FP4/FP8 tensor cores on V100, R2.2). Unlike the legacy AWQ
    :class:`SM70QuantParams` this pack carries **no** ``StridedPtr`` arrays: the
    MXFP4 op takes the packed weight/scale tensors directly.

    The per-operator L0/L1 levels and the masked (decode) path realise the same
    expert FFN in dense fp16; their weights are produced by :meth:`to_dense_fp16`
    via the **single shared** :func:`mxfp4_dequant_to_fp16` decode — the *same*
    numerical source the fused kernel and the per-operator reference
    (:func:`sm70_moe_reference`) use, so every fusion level is numerically
    equivalent by construction (R2.5 / Property 1, no divergent dequant).

    Fields (see design §Data Models ``SM70MXFP4QuantParams``):

    * ``w13_weight``: ``uint8 [E, 2*I, K // 2]`` — gate/up (linear1) MXFP4 packed
      weights (two E2M1 nibbles per byte along the in-feature ``K`` axis; gate
      rows ``[0, I)`` then up rows ``[I, 2*I)``).
    * ``w13_weight_scale``: ``uint8 [E, 2*I, K // group_size]`` — per-32 E8M0
      block scales for ``w13_weight``.
    * ``w2_weight``: ``uint8 [E, K, I // 2]`` — down (linear2) MXFP4 packed
      weights.
    * ``w2_weight_scale``: ``uint8 [E, K, I // group_size]`` — per-32 E8M0 block
      scales for ``w2_weight``.
    * ``num_experts`` / ``hidden_K`` / ``inter_I`` / ``group_size``: dims passed
      straight to ``ops.sm70_fused_moe_out``. ``hidden_K`` is the (possibly
      padded) hidden size the weights decode to; ``group_size`` is the MXFP4
      block (32 for dsv4f).
    * ``hidden_logical_size``: logical (unpadded) hidden size used to slice the
      expert output before ``combine`` (``<= hidden_K`` when the MXFP4 method
      padded the hidden dim up to a ``group_size`` multiple).
    """

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor

    num_experts: int
    hidden_K: int
    inter_I: int
    group_size: int
    hidden_logical_size: int

    @classmethod
    def from_mxfp4_weights(
        cls,
        *,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        group_size: int = _MXFP4_GROUP_SIZE,
        hidden_logical_size: int | None = None,
    ) -> "SM70MXFP4QuantParams":
        """Build the pack from stacked per-expert MXFP4 weight/scale tensors.

        The tensors follow the dsv4f / ``Mxfp4SM70MoEMethod.create_weights``
        orientation:

        * ``w13_weight``       ``uint8 [E, 2*I, K // 2]``
        * ``w13_weight_scale`` ``uint8 [E, 2*I, K // group_size]``
        * ``w2_weight``        ``uint8 [E, K, I // 2]``
        * ``w2_weight_scale``  ``uint8 [E, K, I // group_size]``

        Dimensions are derived from the shapes (``E`` and ``2*I`` from
        ``w13_weight``; ``K = 2 * w13_weight.shape[-1]``; ``I = inter`` and the
        decoded ``hidden_K`` are cross-checked against ``w2_weight``) and every
        shape/dtype invariant the shared decode + kernel rely on is validated up
        front so a malformed pack fails loudly here rather than deep in the op.

        Args:
            group_size: MXFP4 block size (32 for dsv4f); must divide ``K`` and
                ``I``.
            hidden_logical_size: logical (unpadded) hidden size for slicing the
                expert output before ``combine``. Defaults to the decoded
                ``hidden_K``.

        Returns:
            A validated :class:`SM70MXFP4QuantParams`.
        """
        if group_size <= 0:
            raise ValueError(
                f"SM70MXFP4QuantParams requires a positive group_size, got "
                f"{group_size}"
            )
        for name, t in (
            ("w13_weight", w13_weight),
            ("w13_weight_scale", w13_weight_scale),
            ("w2_weight", w2_weight),
            ("w2_weight_scale", w2_weight_scale),
        ):
            if t.dim() != 3:
                raise ValueError(
                    f"{name} must be a 3-D stacked [E, ...] tensor, got "
                    f"shape {tuple(t.shape)}"
                )
            if t.dtype != torch.uint8:
                raise ValueError(
                    f"{name} must be uint8 (packed MXFP4 / E8M0), got {t.dtype}"
                )

        num_experts = w13_weight.shape[0]
        two_i = w13_weight.shape[1]
        if two_i % 2 != 0:
            raise ValueError(
                f"w13_weight axis 1 (2*I) must be even, got {two_i}"
            )
        inter_i = two_i // 2
        hidden_k = w13_weight.shape[2] * 2  # two nibbles per packed byte.

        if hidden_k % group_size != 0:
            raise ValueError(
                f"hidden_K={hidden_k} not divisible by group_size={group_size}"
            )
        if inter_i % group_size != 0:
            raise ValueError(
                f"inter_I={inter_i} not divisible by group_size={group_size}"
            )

        # Cross-check w13 scale + w2 weight/scale shapes against the derived dims.
        expected = {
            "w13_weight_scale": (num_experts, two_i, hidden_k // group_size),
            "w2_weight": (num_experts, hidden_k, inter_i // 2),
            "w2_weight_scale": (num_experts, hidden_k, inter_i // group_size),
        }
        for name, want in expected.items():
            got = tuple(locals()[name].shape)
            if got != want:
                raise ValueError(
                    f"{name} shape {got} does not match the shape implied by "
                    f"w13_weight (E={num_experts}, I={inter_i}, K={hidden_k}, "
                    f"group_size={group_size}): expected {want}"
                )

        return cls(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            num_experts=num_experts,
            hidden_K=hidden_k,
            inter_I=inter_i,
            group_size=group_size,
            hidden_logical_size=(
                hidden_k if hidden_logical_size is None else hidden_logical_size
            ),
        )

    @classmethod
    def from_layer(
        cls,
        layer,
        *,
        group_size: int | None = None,
        hidden_logical_size: int | None = None,
    ) -> "SM70MXFP4QuantParams":
        """Build the pack from a prepared ``Mxfp4SM70MoEMethod`` layer.

        Reads the *raw* MXFP4 expert weights that task 5.4 stashes on the layer
        before ``process_weights_after_loading`` deletes the original
        ``w13_weight`` / ``w2_weight`` parameters (the TurboMind conversion frees
        them). The stash contract (attribute names tried in order):

        1. ``layer.sm70_mxfp4_w13_weight`` / ``..._w13_weight_scale`` /
           ``..._w2_weight`` / ``..._w2_weight_scale`` — the explicit stash
           task 5.4 writes (preferred; survives the TurboMind prep that deletes
           the originals).
        2. ``layer.w13_weight`` / ``layer.w13_weight_scale`` /
           ``layer.w2_weight`` / ``layer.w2_weight_scale`` — the original
           parameters, used when called *before* ``process_weights_after_loading``
           (e.g. in tests).

        Args:
            layer: the MoE layer carrying the (stashed) MXFP4 weights.
            group_size: MXFP4 block size; defaults to ``layer.group_size`` when
                present, else the dsv4f block size (32).
            hidden_logical_size: logical hidden size; defaults to
                ``layer.sm70_hidden_logical_size`` when present, else the decoded
                ``hidden_K``.

        Returns:
            A validated :class:`SM70MXFP4QuantParams`.

        Raises:
            AttributeError: if neither the stash nor the raw params are present
                (a clear signal that task 5.4's stash must run first).
        """

        def _pick(stash_name: str, raw_name: str) -> torch.Tensor:
            t = getattr(layer, stash_name, None)
            if t is None:
                t = getattr(layer, raw_name, None)
            if t is None:
                raise AttributeError(
                    f"SM70MXFP4QuantParams.from_layer: neither {stash_name!r} "
                    f"(task 5.4 stash) nor {raw_name!r} (raw param) is present "
                    f"on the layer; the raw MXFP4 weights must be stashed before "
                    f"process_weights_after_loading deletes them."
                )
            # nn.Parameter -> underlying tensor.
            return t.data if isinstance(t, torch.nn.Parameter) else t

        w13_weight = _pick("sm70_mxfp4_w13_weight", "w13_weight")
        w13_weight_scale = _pick("sm70_mxfp4_w13_weight_scale", "w13_weight_scale")
        w2_weight = _pick("sm70_mxfp4_w2_weight", "w2_weight")
        w2_weight_scale = _pick("sm70_mxfp4_w2_weight_scale", "w2_weight_scale")

        gs = (
            group_size
            if group_size is not None
            else int(getattr(layer, "group_size", _MXFP4_GROUP_SIZE))
        )
        hls = (
            hidden_logical_size
            if hidden_logical_size is not None
            else getattr(layer, "sm70_hidden_logical_size", None)
        )
        return cls.from_mxfp4_weights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            group_size=gs,
            hidden_logical_size=hls,
        )

    def to_dense_fp16(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the MXFP4 pack to dense fp16 expert weights (shared decode).

        Returns ``(w1, w2)`` in the *reference* orientation used by
        :func:`sm70_moe_reference` and the L0/L1 / masked per-operator paths:

        * ``w1``: ``fp16 [E, 2*I, K]`` gate/up (linear1) weights (gate rows then
          up rows along the output axis).
        * ``w2``: ``fp16 [E, K, I]`` down (linear2) weights.

        Both are produced by the **single shared** :func:`mxfp4_dequant_to_fp16`
        decode (E2M1 unpack + per-32 E8M0 block scale) — byte-for-byte the same
        numbers the fused CUDA kernel materialises on chip and the per-operator
        reference consumes — so the L0/L1 / masked realizations are numerically
        equivalent to L2/L3 and to :func:`sm70_moe_reference` by construction
        (R2.5 / Property 1). There is no divergent dequant.
        """
        w1 = mxfp4_dequant_to_fp16(
            self.w13_weight, self.w13_weight_scale, self.group_size
        )  # [E, 2*I, K]
        w2 = mxfp4_dequant_to_fp16(
            self.w2_weight, self.w2_weight_scale, self.group_size
        )  # [E, K, I]
        return w1, w2


# A fused-MoE expert weight pack the orchestrator accepts: the dsv4f MXFP4 pack
# (:class:`SM70MXFP4QuantParams`, the Phase-1 target) or the legacy AWQ int4 pack
# (:class:`SM70QuantParams`, retained for the out-of-scope ``AWQSM70MoEMethod``).
# Both expose the dispatch surface ``SM70FusedMoEExperts`` relies on
# (``to_dense_fp16`` + ``num_experts`` / ``hidden_K`` / ``inter_I`` /
# ``group_size`` / ``hidden_logical_size``); only the L2/L3 fused-kernel call
# differs (MXFP4 packed tensors vs AWQ ``StridedPtr`` arrays).
SM70QuantPack = "SM70MXFP4QuantParams | SM70QuantParams"


class SM70FusedMoEExperts:
    """Orchestrate the SM70 fused MoE forward with selectable fusion level (R2).

    The progressive-fusion strategy from the design (§"渐进式融合策略") is
    realised here as four monotonically-more-fused, numerically-equivalent
    execution strategies selected by :class:`SM70FusionLevel`
    (``SM70FusedConfig.fusion_level``). Every level shares the *same* routing
    path — :meth:`SM70MoELayoutAdapter.to_contiguous` (group + ``m_block`` pad,
    R3.2) and :meth:`SM70MoELayoutAdapter.combine` (unpermute + weighted reduce,
    R3.4) — so the only thing the level changes is *how the expert FFN
    (``linear1 -> SwiGLU -> linear2``) is computed* over the grouped layout.
    Because all four compute the identical fp16 math, their outputs are
    numerically equivalent to the per-operator reference
    (:func:`sm70_moe_reference`) by construction (R2.5, validated by task 5.5).

    Levels (least -> most fused):

    * **L0 — baseline (three independent kernels).** linear1, ``SiluAndMul`` and
      linear2 run as three separate steps, each materialising its intermediate
      (``gate/up`` ``[M, 2*I]`` then ``h`` ``[M, I]``) as a named Python tensor.
      This is the un-fused baseline used for equivalence comparison / fallback.
      Realised by a per-expert torch computation on the dense fp16 weights
      reconstructed from the AWQ int4 pack (:meth:`SM70QuantParams.to_dense_fp16`),
      mirroring the existing per-operator ``awq_moe_gemm_sm70`` path. Runs on CPU
      or GPU (no compiled fused kernel required).
    * **L1 — linear1 + SwiGLU epilogue fused.** The ``silu(gate)*up``
      epilogue is fused onto linear1's output so the ``[M, 2*I]`` ``gate/up``
      tensor is a transient consumed immediately (never held as a named Python
      intermediate alongside ``h``) — i.e. buffer reuse drops the Python-level
      intermediate materialization. Still a per-operator (two-matmul) realization
      on the dense fp16 weights; CPU/GPU friendly.
    * **L2 — eliminate the ``[M, 2*I]`` / ``[M, I]`` HBM round trip.** linear1 ->
      linear2 are chained inside the single hand-written CUDA kernel
      (``ops.sm70_fused_moe_out``); the ``gate/up`` and ``h`` activations live on
      chip (SMEM/registers) and are never round-tripped through HBM (R2.1/R2.6).
    * **L3 — single mega-kernel, fully on-chip token-block flow.** Same kernel
      entry point as L2: ``ops.sm70_fused_moe_out`` *is* the mega-kernel that
      realises the on-chip ``linear1 -> SwiGLU -> linear2`` hand-off per
      ``(expert, m_block)`` tile. L2 and L3 therefore dispatch to the same op;
      the level is recorded for observability and future kernel variants that
      may differentiate the two (e.g. multi-tile vs whole-segment on-chip flow).

    ``layout`` selection (prefill ``contiguous`` vs decode ``masked``) is driven
    by :meth:`select_layout` (R2.6/R2.7): when :meth:`forward` is called without
    an explicit ``layout`` it auto-selects ``contiguous`` for a *prefill* context
    (large M / not capturing a CUDA graph) and ``masked`` for a *decode* context
    (small M / CUDA graph capture). The selection is a **pure decision function**
    that only picks *which* layout to build — it never changes the numerical
    result, because both layouts feed the identical fp16 MoE math (Property 5,
    validated by task 5.7).

    Execution coverage (be honest about select vs run): every fusion level
    supports the ``contiguous`` (grouped) layout, and the fused CUDA mega-kernel
    (L2/L3) consumes the contiguous layout only. The ``masked`` (decode / CUDA
    graph) layout is also fully executable here via :meth:`_forward_masked`,
    which builds the shape-stable ``[E, max_tokens, K]`` batch with
    :meth:`SM70MoELayoutAdapter.to_masked` and runs the per-operator dense-fp16
    FFN over it (the fused kernel reads the contiguous layout, so the masked
    path always uses the per-operator realization). Because both layouts apply
    the identical fp16 math, the masked output equals the contiguous output
    within the fp16 tolerance (Property 1 / 5); the layout choice changes only
    the token bookkeeping, never the numbers.
    """

    #: Levels realised by the per-operator (dense-fp16) path — importable and
    #: runnable without the compiled CUDA kernel (used as equivalence baseline
    #: and CPU-side fallback).
    _PER_OPERATOR_LEVELS = (SM70FusionLevel.L0, SM70FusionLevel.L1)
    #: Levels realised by the single fused CUDA mega-kernel.
    _FUSED_KERNEL_LEVELS = (SM70FusionLevel.L2, SM70FusionLevel.L3)

    def __init__(
        self,
        adapter: SM70MoELayoutAdapter | None = None,
        *,
        config: SM70FusedConfig | None = None,
        fusion_level: SM70FusionLevel | None = None,
        decode_threshold: int = _DEFAULT_PREFILL_MIN_TOKENS,
    ) -> None:
        """Create the orchestrator.

        Args:
            adapter: layout adapter to reuse; a default one is created if omitted.
            config: optional :class:`SM70FusedConfig`; when given (and no explicit
                ``fusion_level``) its ``fusion_level`` selects the default level.
            fusion_level: explicit default level; takes precedence over
                ``config``. When neither is given, defaults to the same value as
                :class:`SM70FusedConfig` (``L1``).
            decode_threshold: maximum token count that :meth:`select_layout`
                classifies as a *decode* (``masked``) context when no CUDA-graph
                capture is in progress; batches larger than this are *prefill*
                (``contiguous``). Defaults to :data:`_DEFAULT_PREFILL_MIN_TOKENS`
                (16). Must be a non-negative int.
        """
        self.adapter = adapter if adapter is not None else SM70MoELayoutAdapter()
        if fusion_level is not None:
            self.fusion_level = fusion_level
        elif config is not None:
            self.fusion_level = config.fusion_level
        else:
            self.fusion_level = SM70FusedConfig().fusion_level
        if not isinstance(decode_threshold, int) or decode_threshold < 0:
            raise ValueError(
                f"decode_threshold must be a non-negative int, got "
                f"{decode_threshold!r}"
            )
        self.decode_threshold = decode_threshold

    @staticmethod
    def _detect_graph_capturing() -> bool:
        """Best-effort detection of an active CUDA-graph capture / compile trace.

        Reuses the exact signals the rest of the vLLM MoE stack uses to spot a
        graph-stable context (see ``fused_batched_moe.py``):

        * ``torch.compiler.is_compiling()`` — inside a ``torch.compile`` trace
          (CUDA graphs are captured under compile), and
        * ``torch.cuda.is_current_stream_capturing()`` — an active CUDA graph
          capture on the current stream.

        This is a read-only query of runtime state (no side effects). It is only
        consulted by :meth:`select_layout` when the caller does not pass an
        explicit ``is_graph_capturing`` flag; the capture probe is guarded behind
        ``torch.cuda.is_available()`` so the module stays importable/usable on a
        CPU-only host. Any unexpected error is swallowed and treated as "not
        capturing" so layout selection can never break the forward.
        """
        try:
            if torch.compiler.is_compiling():
                return True
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return True
        except Exception:
            return False
        return False

    def select_layout(
        self,
        num_tokens: int,
        *,
        is_graph_capturing: bool | None = None,
        decode_threshold: int | None = None,
    ) -> Literal["contiguous", "masked"]:
        """Pick the token layout for an execution context (R2.6/R2.7).

        Implements the prefill/decode layout decision (design Property 5). The
        rule is a **pure decision** of the inputs (side-effect free) — it only
        chooses *which* layout :meth:`forward` builds and never changes the
        numerical result, because both layouts feed the identical fp16 MoE math:

        * **decode → ``"masked"``**: an active CUDA-graph capture (shape must stay
          constant across replays, R2.7) **or** a small batch
          (``num_tokens <= threshold``, i.e. the one-token-per-sequence decode
          step) selects the masked layout.
        * **prefill → ``"contiguous"``**: otherwise (large M, not capturing) the
          grouped/contiguous layout is selected (R2.6).

        Given an explicit ``is_graph_capturing`` this function is fully pure and
        deterministic (the form task 5.7 property-tests). When
        ``is_graph_capturing`` is ``None`` the capture state is resolved once via
        :meth:`_detect_graph_capturing` (a read-only runtime probe).

        Args:
            num_tokens: number of tokens in the batch (the GEMM ``M`` dim).
            is_graph_capturing: explicit CUDA-graph-capture signal; when ``None``
                (default) it is auto-detected from the runtime.
            decode_threshold: max token count classified as decode when not
                capturing; defaults to ``self.decode_threshold``.

        Returns:
            ``"masked"`` for a decode context, ``"contiguous"`` for prefill.
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        threshold = (
            self.decode_threshold if decode_threshold is None else decode_threshold
        )
        if threshold < 0:
            raise ValueError(
                f"decode_threshold must be non-negative, got {threshold}"
            )
        capturing = (
            self._detect_graph_capturing()
            if is_graph_capturing is None
            else bool(is_graph_capturing)
        )
        # CUDA graph capture always forces the shape-stable masked layout,
        # regardless of M (R2.7). Otherwise a small batch is a decode step.
        if capturing or num_tokens <= threshold:
            return "masked"
        return "contiguous"

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        quant: "SM70MXFP4QuantParams | SM70QuantParams",
        layout: Literal["contiguous", "masked"] | None = None,
        m_block: int = 32,
        i_block: int = 64,
        fusion_level: SM70FusionLevel | None = None,
        is_graph_capturing: bool | None = None,
        swiglu_limit: float = 0.0,
    ) -> torch.Tensor:
        """Run the fused MoE forward and return ``[num_tokens, hidden]`` fp16.

        Args:
            x: ``fp16 [num_tokens, K]`` hidden states.
            topk_weights: ``fp16/fp32 [num_tokens, topk]`` router weights.
            topk_ids: ``[num_tokens, topk]`` routed expert ids.
            quant: per-expert weight pack — the dsv4f MXFP4 pack
                (:class:`SM70MXFP4QuantParams`, the Phase-1 target) or the legacy
                AWQ int4 pack (:class:`SM70QuantParams`, out of scope). The
                fusion-level dispatch is identical for both; only the L2/L3 op
                call differs (MXFP4 packed tensors vs AWQ ``StridedPtr`` arrays).
            layout: token layout to use. When ``None`` (default) it is chosen by
                :meth:`select_layout` from the execution context (prefill →
                ``"contiguous"``, decode / CUDA-graph capture → ``"masked"``);
                pass ``"contiguous"`` / ``"masked"`` to force one (R2.6/R2.7).
            m_block: token-block granularity each expert segment is aligned to;
                must be a multiple of the kernel ``M_TILE`` (16).
            i_block: intermediate tiling hint forwarded to the kernel.
            fusion_level: progressive fusion level to use for this call; defaults
                to the instance's ``fusion_level`` (from
                :class:`SM70FusedConfig`). See the class docstring for L0–L3.
            is_graph_capturing: explicit CUDA-graph-capture signal forwarded to
                :meth:`select_layout` when ``layout`` is auto-selected; ``None``
                (default) auto-detects it from the runtime.

        Returns:
            ``fp16 [num_tokens, hidden_logical_size]`` combined output.

        Notes:
            Layout selection is a pure decision (Property 5): ``"contiguous"``
            (prefill, fused-kernel-capable for L2/L3) and ``"masked"`` (decode /
            CUDA-graph, shape-stable, R2.7) feed the identical fp16 MoE math, so
            the resolved layout never changes the numerical result. The masked
            path uses the per-operator (dense fp16) realization regardless of
            ``fusion_level`` because the fused CUDA mega-kernel
            (``ops.sm70_fused_moe_out``) consumes the contiguous layout only.
        """
        level = fusion_level if fusion_level is not None else self.fusion_level
        if not isinstance(level, SM70FusionLevel):
            raise ValueError(
                f"fusion_level must be an SM70FusionLevel, got {level!r}"
            )

        if x.dim() != 2:
            raise ValueError(
                f"forward expects 2-D hidden states, got shape {tuple(x.shape)}"
            )

        num_tokens = x.shape[0]
        hidden_logical = quant.hidden_logical_size

        # Resolve the layout: an explicit value forces it; otherwise the
        # prefill/decode context picks it (R2.6/R2.7). Selection is pure and
        # never changes the numerical result (Property 5).
        if layout is None:
            layout = self.select_layout(
                num_tokens, is_graph_capturing=is_graph_capturing
            )
        elif layout not in ("contiguous", "masked"):
            raise ValueError(f"unsupported layout {layout!r}")

        # Degenerate: no tokens. The empty output is layout-independent (no
        # expert math runs), so short-circuit before the masked guard — this
        # keeps a zero-token decode/graph-capture batch working. moe_permute /
        # moe_unpermute also need a non-empty grid.
        if num_tokens == 0:
            return torch.empty(
                (0, hidden_logical), dtype=x.dtype, device=x.device
            )

        if layout == "masked":
            # Decode / CUDA-graph context (R2.7): build the shape-stable masked
            # batched layout (``[E, max_tokens, K]``) via ``to_masked`` and run
            # the expert FFN over it. The output is numerically equivalent to
            # the contiguous path (Property 5) because both feed the identical
            # fp16 MoE math (same dense weights, same SwiGLU, same per-slot
            # ``topk_weights`` combine); only the token bookkeeping differs. The
            # fused CUDA mega-kernel consumes the contiguous layout only, so the
            # masked path uses the per-operator (dense fp16) realization.
            return self._forward_masked(
                x, topk_weights, topk_ids, quant=quant,
                swiglu_limit=swiglu_limit,
            )

        # 1. Grouped/contiguous layout (reuses moe_permute, m_block aligned).
        #    Shared by every fusion level so routing is identical across L0-L3.
        layout_out = self.adapter.to_contiguous(
            x, topk_ids, quant.num_experts, m_block
        )
        expert_offsets64 = layout_out.expert_first_token_offset

        # 2. Expert FFN over the contiguous layout, dispatched by fusion level.
        if level in self._PER_OPERATOR_LEVELS:
            sorted_out = self._experts_per_operator(
                layout_out.permuted_input, expert_offsets64, quant, level,
                swiglu_limit=swiglu_limit,
            )
        elif level in self._FUSED_KERNEL_LEVELS:
            sorted_out = self._experts_fused_kernel(
                layout_out.permuted_input,
                expert_offsets64,
                quant,
                m_block,
                i_block,
                swiglu_limit=swiglu_limit,
            )
        else:  # pragma: no cover - guarded by the isinstance check above.
            raise ValueError(f"unhandled fusion level {level!r}")

        # 3. Unpermute + weighted reduce back to per-token order. Slice to the
        # logical hidden size (the expert output width is the padded hidden_K).
        #
        # The contiguous ``combine`` reuses the ``moe_unpermute`` CUDA op, whose
        # ``topk_weights`` argument must be ``float32`` — feeding it fp16 weights
        # makes the kernel read the wrong stride and yields near-zero / incorrect
        # output (observed while wiring task 5.3). The router already emits fp32
        # weights in production, but cast defensively here so the fused path is
        # numerically correct for any caller-supplied dtype (Property 1 / R2.5).
        sorted_out_logical = sorted_out[:, :hidden_logical]
        topk_weights_f32 = (
            topk_weights
            if topk_weights.dtype == torch.float32
            else topk_weights.to(torch.float32)
        )
        return self.adapter.combine(
            sorted_out_logical,
            topk_weights_f32,
            layout_out.inv_permuted_idx,
            expert_offsets64,
        )

    def _forward_masked(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        quant: "SM70MXFP4QuantParams | SM70QuantParams",
        swiglu_limit: float = 0.0,
    ) -> torch.Tensor:
        """Decode / CUDA-graph masked-layout MoE forward (R2.7, Property 5).

        Builds the shape-stable masked batched layout via
        :meth:`SM70MoELayoutAdapter.to_masked` and runs the per-expert FFN
        (``linear1 -> SwiGLU -> linear2``) over it in dense fp16, then combines
        the per-expert outputs back to per-token order with the router weights.

        Numerical equivalence to the contiguous path (Property 5 / R2.7): both
        paths apply the *identical* fp16 math — the same dense weights
        (:meth:`SM70QuantParams.to_dense_fp16`, exactly what the fused kernel
        decodes on chip), the same ``silu(gate)*up`` SwiGLU, and the same
        per-token combine weight ``sum_j topk_weights[t, j]`` over the slots
        ``j`` of token ``t`` that route to expert ``e``. Grouping the
        per-operator reference by expert,
        ``out[t] = sum_e (sum_{j: topk_ids[t,j]==e} topk_weights[t,j]) * FFN_e(x[t])``,
        which is what this method computes — so its result matches the reference
        and the contiguous path within the fp16 tolerance. Only the token
        bookkeeping (a fixed ``[E, max_tokens, K]`` batch vs a grouped permute)
        differs; the layout choice does not change the numbers.

        The per-expert capacity is sized to ``num_tokens`` so the worst case
        (every token routed to one expert) fits and no routed token is dropped,
        preserving exact equivalence. The shape ``[E, num_tokens, K]`` is
        invariant to the *routing distribution* (only depends on the batch
        size, which is fixed during a CUDA-graph capture), keeping it
        replay-stable (R2.7). The fused CUDA mega-kernel consumes the contiguous
        layout only, so the masked path uses the per-operator dense-fp16
        realization regardless of fusion level.

        Returns the per-token output ``fp16 [num_tokens, hidden_logical_size]``.
        """
        num_tokens = x.shape[0]
        hidden_logical = quant.hidden_logical_size
        num_experts = quant.num_experts

        # Capacity = num_tokens admits the worst case (all tokens -> one expert)
        # with no dropped token, preserving equivalence with the contiguous
        # path; the shape stays invariant to the routing distribution (R2.7).
        masked = self.adapter.to_masked(x, topk_ids, num_experts, num_tokens)
        batched_input = masked.batched_input  # [E, num_tokens, K] fp16
        expert_num_tokens = masked.expert_num_tokens  # [E] int32

        # Dense fp16 weights (the same weights the fused kernel decodes on chip):
        # w1 [E, 2*I, K] gate/up, w2 [E, K, I] down.
        w1, w2 = quant.to_dense_fp16()

        out = torch.zeros(
            (num_tokens, hidden_logical), dtype=x.dtype, device=x.device
        )

        # One CPU sync for all per-expert valid-row counts (mirrors the
        # contiguous per-operator loop; avoids a per-expert device->host sync).
        host_counts = expert_num_tokens.detach().to("cpu").tolist()
        for e in range(num_experts):
            rows = int(host_counts[e])
            if rows == 0:
                continue
            inp = batched_input[e, :rows]  # [rows, K] fp16
            gate_up = inp @ w1[e].transpose(0, 1)  # linear1 -> [rows, 2*I]
            h = _silu_and_mul(gate_up, swiglu_limit)  # SwiGLU -> [rows, I]
            expert_out = h @ w2[e].transpose(0, 1)  # linear2 -> [rows, K]
            expert_out = expert_out[:, :hidden_logical]

            # Recover the source tokens for these rows: to_masked scattered the
            # tokens routed to e (via any topk slot) in token order, so rebuild
            # the same selection — row r corresponds to token_idx[r].
            token_mask = torch.any(topk_ids == e, dim=1).flatten()
            token_idx = torch.nonzero(token_mask, as_tuple=False).flatten()[:rows]

            # Per-token combine weight for expert e = sum of topk_weights over
            # the slots of that token routing to e (matches the per-slot
            # reference exactly, and handles duplicate routing).
            w_e = torch.where(
                topk_ids == e,
                topk_weights,
                torch.zeros_like(topk_weights),
            ).sum(dim=1)  # [num_tokens]
            w_sel = w_e[token_idx].to(expert_out.dtype).unsqueeze(1)  # [rows, 1]

            # token_idx is unique (each token selected at most once), so this
            # in-place scatter-add has no aliasing.
            out[token_idx] += (expert_out * w_sel).to(out.dtype)

        return out

    def _experts_fused_kernel(
        self,
        permuted_input: torch.Tensor,
        expert_offsets64: torch.Tensor,
        quant: "SM70MXFP4QuantParams | SM70QuantParams",
        m_block: int,
        i_block: int,
        swiglu_limit: float = 0.0,
    ) -> torch.Tensor:
        """L2/L3: single fused mega-kernel ``linear1 -> SwiGLU -> linear2``.

        Both L2 and L3 dispatch here: ``ops.sm70_fused_moe_out`` is the
        hand-written CUDA kernel that chains linear1 -> linear2 with the
        ``gate/up`` / ``h`` activations kept on chip (SMEM/registers), so the
        ``[M, 2*I]`` / ``[M, I]`` HBM round trip is eliminated (R2.1/R2.6) and
        the on-chip token-block hand-off is realised in one launch. The Python
        orchestration never allocates an intermediate-activation buffer.

        The op call differs by weight pack:

        * :class:`SM70MXFP4QuantParams` (dsv4f, the Phase-1 target) — hands the
          packed MXFP4 weight/scale tensors (``w13_weight`` / ``w13_weight_scale``
          / ``w2_weight`` / ``w2_weight_scale``) straight to the kernel, which
          dequantizes MXFP4 -> fp16 on chip via the same decode as
          :func:`mxfp4_dequant_to_fp16` (task 4.4 signature). When the compiled
          op is unavailable (no rebuilt extension on this host) the call raises
          and the caller falls back — full numerical validation of L2/L3 is
          deferred to task 5.5 on a real V100 build.
        * :class:`SM70QuantParams` (legacy AWQ int4, out of scope) — passes the
          four ``FusedStridedPtr`` arrays the AWQ kernel resolves per expert.

        Returns ``fp16 [M_padded, hidden_K]`` per-expert grouped outputs.
        """
        expert_offsets32 = expert_offsets64.to(torch.int32)
        # The kernel writes one output row per permuted_input row (padding rows
        # are skipped by its m_valid guard); combine only gathers valid rows via
        # inv_permuted_idx, so a zero-init keeps padding rows finite (R5.4).
        sorted_out = torch.zeros_like(permuted_input)  # [M_padded, K] fp16
        if isinstance(quant, SM70MXFP4QuantParams):
            # dsv4f MXFP4: the kernel consumes the packed weight/scale tensors
            # directly and decodes MXFP4 -> fp16 on chip (no StridedPtr arrays).
            ops.sm70_fused_moe_out(
                sorted_out,
                permuted_input,
                expert_offsets32,
                quant.w13_weight,
                quant.w13_weight_scale,
                quant.w2_weight,
                quant.w2_weight_scale,
                quant.num_experts,
                quant.hidden_K,
                quant.inter_I,
                quant.group_size,
                m_block,
                i_block,
                swiglu_limit,
            )
        else:
            # Legacy AWQ int4 pack: four FusedStridedPtr arrays (out of scope;
            # retained for AWQSM70MoEMethod import/usage compatibility).
            ops.sm70_fused_moe_out(
                sorted_out,
                permuted_input,
                expert_offsets32,
                quant.w13_ptrs_w,
                quant.w13_ptrs_s,
                quant.w2_ptrs_w,
                quant.w2_ptrs_s,
                quant.num_experts,
                quant.hidden_K,
                quant.inter_I,
                quant.group_size,
                m_block,
                i_block,
                swiglu_limit,
            )
        return sorted_out

    def _experts_per_operator(
        self,
        permuted_input: torch.Tensor,
        expert_offsets64: torch.Tensor,
        quant: "SM70MXFP4QuantParams | SM70QuantParams",
        level: SM70FusionLevel,
        swiglu_limit: float = 0.0,
    ) -> torch.Tensor:
        """L0/L1: per-operator FFN over the grouped layout (dense fp16).

        Reconstructs the dense fp16 expert weights from the weight pack
        (:meth:`SM70MXFP4QuantParams.to_dense_fp16` via the shared
        :func:`mxfp4_dequant_to_fp16` decode for dsv4f MXFP4, or
        :meth:`SM70QuantParams.to_dense_fp16` for the legacy AWQ pack) — the
        *same* weights the fused kernel decodes on chip — and runs
        ``linear1 -> SwiGLU -> linear2`` per
        expert segment, mirroring the repo's existing per-operator
        ``awq_moe_gemm_sm70`` path. The output is byte-for-byte produced from the
        identical fp16 math as the fused path, so L0/L1 are numerically
        equivalent to L2/L3 and to :func:`sm70_moe_reference` (R2.5).

        The L0 vs L1 difference is the intermediate-materialization strategy:

        * **L0** materialises ``gate/up`` (``[n, 2*I]``) and ``h`` (``[n, I]``)
          as two distinct named tensors — the three-independent-kernels baseline.
        * **L1** fuses the ``silu(gate)*up`` epilogue onto linear1's output, so
          ``gate/up`` is a transient consumed immediately (buffer reuse drops the
          Python-level intermediate materialization).

        Padding rows of ``permuted_input`` are zero-filled by ``moe_permute`` so
        computing over the full ``m_block``-aligned segment stays finite; combine
        ignores those rows via ``inv_permuted_idx`` (R5.4).

        Returns ``fp16 [M_padded, hidden_K]`` per-expert grouped outputs.
        """
        # Dense fp16 weights: w1 [E, 2*I, K], w2 [E, K, I] (== padded hidden_K).
        w1, w2 = quant.to_dense_fp16()
        m_padded = permuted_input.shape[0]
        out = torch.zeros(
            (m_padded, quant.hidden_K),
            dtype=permuted_input.dtype,
            device=permuted_input.device,
        )

        # One CPU sync up front for all expert segment boundaries (mirrors
        # _apply_sorted_loop) — avoids a per-expert device->host sync.
        host_offsets = expert_offsets64.detach().to("cpu").tolist()
        for e in range(quant.num_experts):
            start, end = int(host_offsets[e]), int(host_offsets[e + 1])
            if end <= start:
                continue
            seg = permuted_input[start:end]  # [n, K] fp16
            if level is SM70FusionLevel.L0:
                # Baseline: three independent steps, both intermediates named.
                gate_up = seg @ w1[e].transpose(0, 1)  # linear1 -> [n, 2*I]
                h = _silu_and_mul(gate_up, swiglu_limit)  # SiluAndMul -> [n, I]
                out[start:end] = h @ w2[e].transpose(0, 1)  # linear2 -> [n, K]
            else:  # SM70FusionLevel.L1
                # linear1 + SwiGLU epilogue fused: gate_up is a transient that is
                # consumed immediately, dropping the Python-level [n, 2*I]
                # intermediate materialization (buffer reuse).
                h = _silu_and_mul(seg @ w1[e].transpose(0, 1), swiglu_limit)  # [n, I]
                out[start:end] = h @ w2[e].transpose(0, 1)  # [n, K]
        return out
