# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAMPLE/unit tests for the SM70 fused MoE path (R2.1, R2.2, R2.6, R2.7).

Feature: deepgemm-megamoe-sm70-port

These are concrete example/unit tests (NOT property tests) that pin down four
behaviors of the dsv4f **MXFP4** fused expert path from the design's
Requirements Mapping (R2 -> "P1, P5 + EXAMPLE (HBM/FP4 检查)"):

* **R2.1 / R2.6 — no ``[M, 2*I]`` HBM block materialization (prefill).** The
  fused-kernel (L2/L3) contiguous path keeps the ``gate/up`` (``[M, 2*I]``) and
  intermediate ``h`` (``[M, I]``) activations on chip; the Python orchestration
  must never allocate a full ``[*, 2*I]`` HBM *activation* buffer. We assert
  this by intercepting every tensor allocation made during the fused forward
  and checking that none is a 2-D ``[*, 2*I]`` matrix (a gate/up activation) —
  1-D bookkeeping tensors that happen to be ``2*I`` long, e.g. an
  ``m_indices`` of length ``m_padded``, are *not* a materialized gate/up block,
  so the check only flags genuine 2-D activations.

* **R2.2 — MXFP4 path uses *software* dequant, no FP4/FP8 *tensor-core* op
  (static symbol check).** V100 (SM70) has no FP8/FP4 tensor cores, no TMA /
  WGMMA / ``cp.async`` / clusters / tcgen05. The fused expert module's *code*
  must therefore reference none of those hardware primitives. Crucially, the
  re-scoped spec (R2.2) **requires** the MXFP4 -> fp16 *software* dequant, so the
  check **must not** ban the ``mxfp4`` / ``fp4`` substrings that appear in the
  software-dequant helper names (``mxfp4_dequant_to_fp16`` /
  ``SM70MXFP4QuantParams`` / ``from_mxfp4_weights``). It bans only the forbidden
  *hardware tensor-core* primitives (FP8 dtypes/ops, ``__nv_fp8``,
  ``scaled_mm``, bare/``nv`` FP4, WGMMA, TMA, ``cp.async``, clusters, tcgen05),
  while a positive assertion confirms the MXFP4 software-dequant *is* wired in.

* **R2.7 — masked path is CUDA-graph friendly (shape constant).** The masked
  (decode) layout's buffer and output shapes depend only on ``(E, num_tokens,
  K)`` and are invariant to the routing distribution (how many valid tokens land
  on each expert), which is exactly what lets a CUDA graph replay it.

The tests that exercise executed math run on CUDA (the contiguous ``combine``
reuses the ``moe_permute`` / ``moe_unpermute`` CUDA ops, which also need fp32
``topk_weights``); the static-symbol check and the MXFP4 pack-shape check are
CPU-only. Following the repo's V100 target, ``float16`` is the only compute
dtype.

Validates: Requirements 2.1, 2.2, 2.6, 2.7
"""

from __future__ import annotations

import inspect
import io
import re
import tokenize

import pytest
import torch

from vllm.platforms import current_platform

from vllm.model_executor.layers.fused_moe import sm70_fused_moe_experts as experts_mod
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute_unpermute_supported,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import SM70FusionLevel

# MXFP4 micro-scaling block size: one E8M0 block scale per 32 packed E2M1
# elements along the contraction axis. dsv4f experts use 32.
_MXFP4_GROUP_SIZE = 32

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the executed fused-MoE paths build/run on a CUDA device",
)

# The contiguous layout reuses the moe_permute/moe_unpermute CUDA ops (the
# contiguous combine also needs fp32 router weights); gate the same way the
# passing MXFP4-levels test does.
requires_runnable = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform.is_rocm()
    or not moe_permute_unpermute_supported(),
    reason="contiguous path reuses the moe_permute/moe_unpermute CUDA ops",
)


# --- shared MXFP4 helpers ---------------------------------------------------
#
# dsv4f MXFP4 facts (design §Data Models ``SM70MXFP4QuantParams``):
#   w13_weight       uint8 [E, 2*I, K // 2]    (two E2M1 nibbles per byte)
#   w13_weight_scale uint8 [E, 2*I, K // 32]   (per-32 E8M0 block scale)
#   w2_weight        uint8 [E, K,   I // 2]
#   w2_weight_scale  uint8 [E, K,   I // 32]

# E8M0 block-scale exponents kept in a modest band so the de-quantized fp16
# weights stay small/finite for the executed (masked/decode) paths; 127 == 2**0.
_E8M0_LO = 125
_E8M0_HI = 129


def _rand_packed_u8(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Random uint8 tensor whose two nibbles are arbitrary E2M1 codes (0..15)."""
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)


def _rand_e8m0(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Random uint8 E8M0 block-scale exponents in a modest finite band."""
    return torch.randint(
        _E8M0_LO, _E8M0_HI, shape, dtype=torch.uint8, device=device
    )


def _build_mxfp4_quant(
    E: int, K: int, I: int, gs: int, device: torch.device
) -> SM70MXFP4QuantParams:
    """Build a supported dsv4f :class:`SM70MXFP4QuantParams` with random weights.

    Packed E2M1 nibbles are fully random; the E8M0 block scales sit in a narrow
    band so the de-quantized fp16 weights stay finite for the executed paths.
    """
    two_i = 2 * I
    return SM70MXFP4QuantParams.from_mxfp4_weights(
        w13_weight=_rand_packed_u8((E, two_i, K // 2), device),
        w13_weight_scale=_rand_e8m0((E, two_i, K // gs), device),
        w2_weight=_rand_packed_u8((E, K, I // 2), device),
        w2_weight_scale=_rand_e8m0((E, K, I // gs), device),
        group_size=gs,
    )


# --- R2.1 / R2.6: no [M, 2*I] HBM block materialization ---------------------


class _AllocTracker:
    """Patches the torch allocators to record the shape of every allocation.

    Wraps ``torch.empty`` / ``torch.zeros`` / ``torch.empty_like`` /
    ``torch.zeros_like`` / ``torch.ones`` / ``torch.full`` for the duration of a
    ``with`` block and collects the shape of each tensor produced, so a test can
    assert what the orchestration layer did (and did not) allocate.
    """

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self._saved: dict[str, object] = {}

    def __enter__(self) -> "_AllocTracker":
        funcs = ("empty", "zeros", "empty_like", "zeros_like", "ones", "full")
        for name in funcs:
            self._saved[name] = getattr(torch, name)

        def _wrap(orig):
            def inner(*args, **kwargs):
                out = orig(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    self.shapes.append(tuple(out.shape))
                return out

            return inner

        for name in funcs:
            setattr(torch, name, _wrap(self._saved[name]))
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._saved.items():
            setattr(torch, name, fn)


@requires_runnable
@pytest.mark.parametrize("level", [SM70FusionLevel.L2, SM70FusionLevel.L3])
def test_fused_kernel_path_no_gate_up_hbm_block(monkeypatch, level) -> None:
    """R2.1/R2.6: the fused (L2/L3) path allocates no ``[*, 2*I]`` HBM activation.

    The hand-written CUDA mega-kernel keeps ``gate/up`` (``[M, 2*I]``) on chip,
    so the Python orchestration only allocates the ``[M_padded, K]`` per-expert
    output and the routing/combine buffers — never the gate/up activation. The
    L2/L3 MXFP4 CUDA op is not built on this host, so we stub it with a
    shape-correct no-op (matching its ``out``-first signature) to drive the
    *runnable* fused dispatch, and intercept every torch allocation during the
    forward to assert none is a 2-D ``[*, 2*I]`` matrix.

    Shapes are chosen so ``2*I`` (192) collides with neither ``K`` (64) nor any
    plausible ``m_padded``, and the offending check is restricted to 2-D
    activations so a 1-D bookkeeping tensor (e.g. ``m_indices`` of length
    ``m_padded``) is never mistaken for a materialized gate/up block. The pack
    is the dsv4f MXFP4 pack: at L2/L3 the kernel consumes the packed weight /
    scale tensors directly (no dense fp16 dequant in Python), so no ``2*I``-wide
    buffer is ever allocated host-side.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    # 2*I == 192 is distinct from K (64) and from any reachable m_padded for
    # this routing (<= 160 with m_block=16), so a [*, 2*I] hit can only be a
    # genuine gate/up activation.
    E, K, I, gs, topk = 4, 64, 96, _MXFP4_GROUP_SIZE, 2
    two_i = 2 * I
    num_tokens = 32

    quant = _build_mxfp4_quant(E, K, I, gs, device)
    x = (torch.randn((num_tokens, K), device=device) * 0.5).to(torch.float16)
    topk_ids = torch.randint(0, E, (num_tokens, topk), dtype=torch.int32, device=device)
    # moe_unpermute (contiguous combine) requires fp32 router weights.
    topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)

    # Stub the uncompiled MXFP4 fused kernel: it writes one output row per input
    # row (the on-chip gate/up never surfaces to Python), matching the real op's
    # ``out``-first signature ``(out, permuted_input, expert_offsets, w13, ...)``.
    def _fake_fused_out(out, permuted_input, *args, **kwargs):
        out.zero_()
        return None

    monkeypatch.setattr(experts_mod.ops, "sm70_fused_moe_out", _fake_fused_out)

    fused = SM70FusedMoEExperts(fusion_level=level)
    with _AllocTracker() as tracker:
        result = fused.forward(
            x, topk_weights, topk_ids, quant=quant, layout="contiguous",
            m_block=16,
        )

    assert result.shape == (num_tokens, K)
    # The defining check: no 2-D allocation has the gate/up width (2*I) as its
    # last dim, i.e. the [M, 2*I] gate/up activation is never materialized in
    # HBM by the Python orchestration (R2.1/R2.6). 1-D bookkeeping tensors that
    # happen to be 2*I long are not gate/up activations and are ignored.
    offending = [s for s in tracker.shapes if len(s) >= 2 and s[-1] == two_i]
    assert not offending, (
        f"fused {level} path materialized a [*, 2*I={two_i}] HBM activation: "
        f"{offending}; gate/up must stay on chip (R2.1/R2.6)"
    )
    # Sanity: the expected on-chip-output buffer ([M_padded, K]) was allocated.
    assert any(s and s[-1] == K for s in tracker.shapes)


# --- R2.2: MXFP4 software dequant, no FP4/FP8 tensor-core op (static check) --

# Forbidden *hardware tensor-core* primitives. V100 (SM70) has no FP8/FP4 tensor
# cores and none of the SM80+/SM90+/SM100 async/tensor-core primitives, so the
# fused MoE *code* must reference none of these. The MXFP4 -> fp16 *software*
# dequant that the re-scoped spec requires (R2.2) is **allowed** — its helper
# names ``mxfp4`` / ``mxfp4_dequant_to_fp16`` are explicitly NOT in this list,
# and the bare-``fp4`` pattern uses a negative look-behind so it never matches
# the ``mxfp4`` prefix. Matched case-insensitively against the module's *code*
# tokens only (comments / string & f-string literals excluded).
_FORBIDDEN_TENSORCORE_SYMBOLS = (
    # FP8 hardware dtypes / tensor-core ops (no FP8 cores on SM70).
    r"float8",
    r"fp8",
    r"e4m3",
    r"e5m2",
    r"__nv_fp8",
    r"scaled_mm",  # FP8 tensor-core scaled GEMM
    # FP4 hardware dtypes / tensor-core ops (no FP4 cores on SM70). The negative
    # look-behind allows the *software* MXFP4 dequant names (``mxfp4`` ...) while
    # still banning bare ``fp4`` and ``nvfp4``.
    r"nvfp4",
    r"(?<!mx)fp4",
    r"float4",
    # SM80+/SM90+/SM100 async + tensor-core primitives absent on SM70 (the V100
    # path uses synchronous double-buffered SMEM + first-gen ``mma.sync``/HMMA).
    r"wgmma",  # SM90 warpgroup MMA
    r"tcgen",  # SM100 tcgen05 / TMEM
    r"cp\.async",  # SM80 async copy
    r"cp_async",
    r"\btma\b",  # SM90 Tensor Memory Accelerator
    r"\bcluster\b",  # SM90 thread-block clusters / DSMEM
)

# Names proving the REQUIRED MXFP4 -> fp16 software dequant is wired in (R2.2).
_REQUIRED_MXFP4_SOFTWARE_DEQUANT = ("mxfp4_dequant_to_fp16",)


def _module_code_text(module) -> str:
    """Return the module source as a space-joined stream of *code* tokens.

    Comments and string literals — including f-string literal segments, which in
    Python 3.12+ tokenize as ``FSTRING_*`` rather than ``STRING`` — are dropped,
    so an honest "no FP8/FP4 tensor cores" disclaimer in a docstring or an
    error-message f-string does not trip the check; only a forbidden symbol used
    in actual executable code would. (Embedded f-string *expressions* such as
    ``{name}`` remain as NAME/OP tokens and are kept, as they are real code.)
    """
    skip_types = {tokenize.COMMENT, tokenize.STRING}
    for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        _tt = getattr(tokenize, _name, None)
        if _tt is not None:
            skip_types.add(_tt)

    src = inspect.getsource(module)
    code_tokens: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in skip_types:
            continue
        code_tokens.append(tok.string)
    return " ".join(code_tokens)


def test_fused_experts_source_has_no_fp4_fp8_tensorcore_symbols() -> None:
    """R2.2: the fused expert module references no FP4/FP8 *tensor-core* op.

    The V100 MXFP4 path dequantizes MXFP4 -> fp16 in *software* (the re-scoped
    spec, R2.2, requires this) and computes with first-gen ``mma.sync`` (HMMA)
    only. The module's *code* must therefore reference none of the forbidden
    hardware primitives — FP8 dtypes/ops, ``__nv_fp8``, ``scaled_mm``,
    bare/``nv`` FP4, WGMMA, TMA, ``cp.async``, clusters or tcgen05 — while the
    MXFP4 *software*-dequant helper names (``mxfp4`` / ``fp4`` substrings inside
    ``mxfp4_dequant_to_fp16`` etc.) are explicitly allowed.
    """
    code_text = _module_code_text(experts_mod)

    hits: list[str] = []
    for pat in _FORBIDDEN_TENSORCORE_SYMBOLS:
        if re.search(pat, code_text, flags=re.IGNORECASE):
            hits.append(pat)
    assert not hits, (
        f"SM70 fused expert module references forbidden FP4/FP8 tensor-core "
        f"symbol(s) in code: {hits}; the V100 path must use MXFP4 -> fp16 "
        f"software dequant + fp16 mma.sync only (R2.2)"
    )


def test_fused_experts_uses_mxfp4_software_dequant() -> None:
    """R2.2: the module *does* wire in the required MXFP4 -> fp16 software dequant.

    The complement of the ban above: the re-scoped spec REQUIRES the MXFP4
    software dequant, so the ``mxfp4_dequant_to_fp16`` helper must be referenced
    in the module's code (the shared numeric source the per-operator paths and
    the fused kernel both decode through). This guards against an over-zealous
    ban that strips the legitimate software-dequant path.
    """
    code_text = _module_code_text(experts_mod)
    missing = [
        name for name in _REQUIRED_MXFP4_SOFTWARE_DEQUANT if name not in code_text
    ]
    assert not missing, (
        f"SM70 fused expert module is missing the required MXFP4 software "
        f"dequant reference(s): {missing}; the V100 MXFP4 path must dequantize "
        f"MXFP4 -> fp16 in software (R2.2)"
    )


def test_fused_quant_params_is_mxfp4() -> None:
    """R2.2: the quant payload is dsv4f MXFP4 (packed uint8 E2M1 + E8M0 scale)."""
    device = torch.device("cpu")
    E, K, I, gs = 2, 64, 64, _MXFP4_GROUP_SIZE
    quant = _build_mxfp4_quant(E, K, I, gs, device)
    # MXFP4 packs two E2M1 nibbles per byte along the contraction axis, so the
    # decoded K / I are twice the packed last-dim width.
    assert quant.hidden_K == quant.w13_weight.shape[-1] * 2
    assert quant.inter_I == quant.w2_weight.shape[-1] * 2
    assert quant.group_size == _MXFP4_GROUP_SIZE
    # Packed weights and block scales are uint8 (not any float8/float4 dtype).
    assert quant.w13_weight.dtype == torch.uint8
    assert quant.w2_weight.dtype == torch.uint8
    assert quant.w13_weight_scale.dtype == torch.uint8
    assert quant.w2_weight_scale.dtype == torch.uint8
    # Block scale covers ``group_size`` contraction elements per stored exponent.
    assert quant.w13_weight_scale.shape[-1] == quant.hidden_K // gs
    assert quant.w2_weight_scale.shape[-1] == quant.inter_I // gs


# --- R2.7: masked path is CUDA-graph friendly (shape constant) --------------

# Distinct routing distributions with the *same* num_tokens. The masked layout
# must produce identical-shaped buffers/outputs for all of them.
def _routings(num_tokens: int, E: int, topk: int, device: torch.device):
    g = torch.Generator(device="cpu").manual_seed(1234)
    uniform = torch.randint(
        0, E, (num_tokens, topk), generator=g, dtype=torch.int32
    ).to(device)
    all_to_one = torch.zeros((num_tokens, topk), dtype=torch.int32, device=device)
    # Half the experts empty: route only to even expert ids.
    half = (
        torch.randint(0, max(1, E // 2), (num_tokens, topk), generator=g, dtype=torch.int32)
        * 2
    ).clamp_max(E - 1).to(device)
    return {"uniform": uniform, "all_to_one": all_to_one, "half_empty": half}


@requires_cuda
def test_masked_layout_shapes_constant_across_routing() -> None:
    """R2.7: masked buffers/outputs are shape-stable across routing distributions.

    The ``to_masked`` batch is ``[E, num_tokens, K]`` and ``expert_num_tokens``
    is ``[E]`` regardless of how tokens distribute over experts; the masked
    forward output is ``[num_tokens, K]``. Shape constancy w.r.t. the routing
    distribution is precisely what makes the layout safe to capture / replay in
    a CUDA graph. Uses the dsv4f MXFP4 pack (the masked path decodes it to dense
    fp16 via the shared software dequant; the fused CUDA op is not needed here).
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    E, K, I, gs, topk = 6, 64, 32, _MXFP4_GROUP_SIZE, 2
    num_tokens = 24

    quant = _build_mxfp4_quant(E, K, I, gs, device)
    fused = SM70FusedMoEExperts()
    adapter = fused.adapter

    out_shapes = set()
    batch_shapes = set()
    count_shapes = set()
    for _name, topk_ids in _routings(num_tokens, E, topk, device).items():
        x = (torch.randn((num_tokens, K), device=device) * 0.5).to(torch.float16)
        topk_weights = torch.rand(
            (num_tokens, topk), dtype=torch.float32, device=device
        )

        masked = adapter.to_masked(x, topk_ids, E, num_tokens)
        batch_shapes.add(tuple(masked.batched_input.shape))
        count_shapes.add(tuple(masked.expert_num_tokens.shape))

        out = fused.forward(
            x, topk_weights, topk_ids, quant=quant, layout="masked"
        )
        out_shapes.add(tuple(out.shape))

    assert batch_shapes == {(E, num_tokens, K)}, batch_shapes
    assert count_shapes == {(E,)}, count_shapes
    assert out_shapes == {(num_tokens, K)}, out_shapes


@requires_cuda
def test_decode_context_selects_masked_and_runs() -> None:
    """R2.7: a CUDA-graph-capture context auto-selects masked and executes.

    With ``layout=None`` and ``is_graph_capturing=True`` the orchestrator must
    pick the shape-stable masked layout (regardless of M) and produce a correctly
    shaped, finite output over the dsv4f MXFP4 pack.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    E, K, I, gs, topk = 4, 64, 32, _MXFP4_GROUP_SIZE, 2
    num_tokens = 128  # large M, but capture forces masked

    quant = _build_mxfp4_quant(E, K, I, gs, device)
    fused = SM70FusedMoEExperts(decode_threshold=16)
    assert fused.select_layout(num_tokens, is_graph_capturing=True) == "masked"

    x = (torch.randn((num_tokens, K), device=device) * 0.5).to(torch.float16)
    topk_ids = torch.randint(0, E, (num_tokens, topk), dtype=torch.int32, device=device)
    topk_weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)

    out = fused.forward(
        x, topk_weights, topk_ids, quant=quant, layout=None, is_graph_capturing=True
    )
    assert out.shape == (num_tokens, K)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
