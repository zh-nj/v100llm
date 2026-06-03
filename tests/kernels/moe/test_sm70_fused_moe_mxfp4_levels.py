# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAMPLE/unit tests for the SM70 fused MoE progressive fusion levels (L0->L3).

Feature: deepgemm-megamoe-sm70-port

These concrete example/unit tests (NOT property tests) pin down task 5.2 — the
progressive-fusion dispatch L0->L3 of :class:`SM70FusedMoEExperts` over the
dsv4f **MXFP4** weight pack (:class:`SM70MXFP4QuantParams`):

* **Shared MXFP4 decode (no divergent dequant).**
  :meth:`SM70MXFP4QuantParams.to_dense_fp16` (used by the L0/L1 per-operator
  levels and the masked path) must produce *byte-for-byte* the same dense fp16
  weights as the single shared :func:`mxfp4_dequant_to_fp16` decode that the
  per-operator reference (:func:`sm70_moe_reference`) and the fused CUDA kernel
  rely on — this is what makes every fusion level numerically equivalent by
  construction (R2.2 / R2.5).

* **L0 / L1 / masked numerical equivalence (R2.1 / R2.5 / R2.6).** Every
  runnable level produces output equal to the per-operator MXFP4 fp16 reference
  within the design's fp16 tolerance (relative L2 + anti-noise vs the fp32
  golden). L0 (three independent steps), L1 (linear1+SwiGLU epilogue fused,
  dropping the Python ``[M, 2*I]`` intermediate) and the masked decode layout
  all share the identical fp16 math, so they agree.

* **L2 / L3 reachability (R2.6).** Both higher levels dispatch to the MXFP4
  ``ops.sm70_fused_moe_out`` mega-kernel (intermediate activations on chip, no
  HBM round trip). When the compiled CUDA op is unavailable (no rebuilt
  extension on this host) the call is *reachable in code* and raises cleanly so
  the caller can fall back; full numerical validation of L2/L3 is deferred to
  task 5.5 on a real V100 build. When the op *is* present we additionally assert
  L2/L3 agree with the reference.

The executed-math tests run on CUDA (the contiguous ``combine`` reuses the
``moe_unpermute`` CUDA op, which also needs fp32 ``topk_weights``); the
decode-parity check is CPU-friendly. Following the repo's V100 target,
``float16`` is the only compute dtype.

Validates: Requirements 2.1, 2.6
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from vllm.platforms import current_platform

with contextlib.suppress(ImportError):
    import vllm._C  # noqa: F401
with contextlib.suppress(ImportError):
    import vllm._moe_C  # noqa: F401

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute_unpermute_supported,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusionLevel,
)
from vllm.model_executor.layers.fused_moe.sm70_moe_reference import (
    check_fp16_close,
    mxfp4_dequant_to_fp16,
    sm70_moe_reference,
    sm70_moe_reference_golden,
)

_MXFP4_GROUP_SIZE = 32

# The fused CUDA mega-kernel (L2/L3) is only present after the tasks 4.1-4.3
# rebuild on a V100; probe once so we additionally exercise it when available.
_HAS_FUSED_OP = hasattr(getattr(torch.ops, "_C", None), "sm70_fused_moe_out")


def _make_mxfp4_pack(
    *,
    E: int,
    I: int,
    K: int,
    gs: int,
    device: torch.device,
    seed: int = 0,
) -> SM70MXFP4QuantParams:
    """Build a random dsv4f MXFP4 weight pack with O(1) decoded weights.

    The E8M0 block-scale exponents are drawn near the bias (127) so the decoded
    fp16 weights stay O(1) and the per-operator reference does not overflow
    (overflow behaviour is Property 4's concern, not this dispatch test's).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randint(
        0, 256, (E, 2 * I, K // 2), generator=gen, device=device, dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (E, K, I // 2), generator=gen, device=device, dtype=torch.uint8
    )
    # raw in [122, 128) -> 2**(raw-127) in [0.03125, 2), modest magnitudes.
    w13s = torch.randint(
        122, 128, (E, 2 * I, K // gs), generator=gen, device=device, dtype=torch.uint8
    )
    w2s = torch.randint(
        122, 128, (E, K, I // gs), generator=gen, device=device, dtype=torch.uint8
    )
    return SM70MXFP4QuantParams.from_mxfp4_weights(
        w13_weight=w13,
        w13_weight_scale=w13s,
        w2_weight=w2,
        w2_weight_scale=w2s,
        group_size=gs,
    )


# --- CPU-friendly: shared-decode parity ------------------------------------


def test_to_dense_fp16_matches_shared_decode_exactly() -> None:
    """``to_dense_fp16`` reproduces the shared MXFP4 decode byte-for-byte.

    The L0/L1 / masked dense-fp16 path must decode MXFP4 with the *same*
    :func:`mxfp4_dequant_to_fp16` the reference and the fused kernel use, so
    there is no divergent dequant (R2.2 / R2.5). We assert exact equality.
    """
    device = torch.device("cpu")
    pack = _make_mxfp4_pack(E=3, I=64, K=64, gs=_MXFP4_GROUP_SIZE, device=device)

    w1, w2 = pack.to_dense_fp16()
    ref_w1 = mxfp4_dequant_to_fp16(
        pack.w13_weight, pack.w13_weight_scale, pack.group_size
    )
    ref_w2 = mxfp4_dequant_to_fp16(
        pack.w2_weight, pack.w2_weight_scale, pack.group_size
    )

    assert w1.dtype == w2.dtype == torch.float16
    assert tuple(w1.shape) == (3, 128, 64)  # [E, 2*I, K]
    assert tuple(w2.shape) == (3, 64, 64)  # [E, K, I]
    assert torch.equal(w1, ref_w1)
    assert torch.equal(w2, ref_w2)


def test_from_mxfp4_weights_rejects_bad_shapes() -> None:
    """The pack validates dtype/shape invariants up front (clear failure)."""
    device = torch.device("cpu")
    E, I, K, gs = 2, 64, 64, _MXFP4_GROUP_SIZE
    w13 = torch.randint(0, 256, (E, 2 * I, K // 2), device=device, dtype=torch.uint8)
    w13s = torch.randint(
        0, 255, (E, 2 * I, K // gs), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (E, K, I // 2), device=device, dtype=torch.uint8)
    w2s = torch.randint(0, 255, (E, K, I // gs), device=device, dtype=torch.uint8)

    # Wrong dtype for a packed weight.
    with pytest.raises(ValueError, match="uint8"):
        SM70MXFP4QuantParams.from_mxfp4_weights(
            w13_weight=w13.to(torch.int8),
            w13_weight_scale=w13s,
            w2_weight=w2,
            w2_weight_scale=w2s,
            group_size=gs,
        )
    # Mismatched w2 scale last dim.
    with pytest.raises(ValueError, match="does not match"):
        SM70MXFP4QuantParams.from_mxfp4_weights(
            w13_weight=w13,
            w13_weight_scale=w13s,
            w2_weight=w2,
            w2_weight_scale=w2s[..., :-1],
            group_size=gs,
        )


# --- GPU: executed L0/L1/masked equivalence + L2/L3 reachability -----------

requires_runnable = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform.is_rocm()
    or not moe_permute_unpermute_supported(),
    reason="contiguous/masked paths reuse the moe_permute/moe_unpermute CUDA ops",
)


@requires_runnable
def test_mxfp4_levels_equivalent_to_reference() -> None:
    """L0 / L1 / masked over the MXFP4 pack match the per-operator reference.

    For a supported MoE shape, every runnable fusion level produces output
    equal to :func:`sm70_moe_reference` (the per-operator fp16 golden over the
    *same* MXFP4-decoded weights) within the design's fp16 tolerance.
    """
    device = torch.device("cuda")
    E, I, K, gs, topk, M = 4, 64, 64, _MXFP4_GROUP_SIZE, 2, 48
    pack = _make_mxfp4_pack(E=E, I=I, K=K, gs=gs, device=device, seed=7)

    torch.manual_seed(7)
    x = (torch.randn((M, K), device=device) * 0.3).to(torch.float16)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
    # moe_unpermute (contiguous combine) requires fp32 router weights.
    topk_weights = torch.rand((M, topk), dtype=torch.float32, device=device)

    ref = sm70_moe_reference(
        x,
        topk_weights,
        topk_ids,
        pack.w13_weight,
        pack.w13_weight_scale,
        pack.w2_weight,
        pack.w2_weight_scale,
        E,
        group_size=gs,
    )
    assert bool(torch.isfinite(ref).all())
    golden = sm70_moe_reference_golden(
        x,
        topk_weights,
        topk_ids,
        pack.w13_weight,
        pack.w13_weight_scale,
        pack.w2_weight,
        pack.w2_weight_scale,
        E,
        group_size=gs,
    )

    experts = SM70FusedMoEExperts()
    runnable = [
        ("contiguous", SM70FusionLevel.L0),
        ("contiguous", SM70FusionLevel.L1),
        ("masked", SM70FusionLevel.L1),
    ]
    if _HAS_FUSED_OP and torch.cuda.get_device_capability(0) == (7, 0):
        runnable += [
            ("contiguous", SM70FusionLevel.L2),
            ("contiguous", SM70FusionLevel.L3),
        ]

    for layout, level in runnable:
        out = experts.forward(
            x,
            topk_weights,
            topk_ids,
            quant=pack,
            layout=layout,
            m_block=32,
            i_block=64,
            fusion_level=level,
        )
        assert out.shape == (M, pack.hidden_logical_size)
        assert out.dtype == torch.float16
        check = check_fp16_close(out, ref, K=K, I=I, topk=topk, golden=golden)
        assert check.passed, (
            f"MXFP4 fused level disagrees with the reference: {check.reason} | "
            f"layout={layout}, level={level.value}"
        )


@requires_runnable
def test_l2_l3_reachable_and_dispatch_to_fused_op() -> None:
    """L2/L3 dispatch to the MXFP4 ``ops.sm70_fused_moe_out`` mega-kernel.

    When the compiled CUDA op is unavailable on this host, the L2/L3 call is
    still *reachable* in code and raises cleanly (so the caller can fall back) —
    full numerical validation is deferred to task 5.5 on a real V100 build. When
    the op is present, L2/L3 are exercised by
    :func:`test_mxfp4_levels_equivalent_to_reference`.
    """
    device = torch.device("cuda")
    E, I, K, gs, topk, M = 2, 64, 64, _MXFP4_GROUP_SIZE, 2, 32
    pack = _make_mxfp4_pack(E=E, I=I, K=K, gs=gs, device=device, seed=3)

    torch.manual_seed(3)
    x = (torch.randn((M, K), device=device) * 0.3).to(torch.float16)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
    topk_weights = torch.rand((M, topk), dtype=torch.float32, device=device)

    experts = SM70FusedMoEExperts()
    for level in (SM70FusionLevel.L2, SM70FusionLevel.L3):
        if _HAS_FUSED_OP and torch.cuda.get_device_capability(0) == (7, 0):
            out = experts.forward(
                x, topk_weights, topk_ids, quant=pack,
                layout="contiguous", fusion_level=level,
            )
            assert out.shape == (M, pack.hidden_logical_size)
            assert out.dtype == torch.float16
        else:
            # Reachable in code; the missing compiled op surfaces as an error
            # rather than a silent wrong result (deferred to task 5.5).
            with pytest.raises(Exception):
                experts.forward(
                    x, topk_weights, topk_ids, quant=pack,
                    layout="contiguous", fusion_level=level,
                )
