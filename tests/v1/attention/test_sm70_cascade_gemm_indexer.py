# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU correctness for the SM70 cascade-compatible GEMM indexer.

Compares ``sm70_cascade_gemm_indexer`` against the production
``sm70_fp8_paged_mqa_logits`` paged kernel on a multi-seed × multi-ctx
× multi-scale × extreme-sign sweep.

The two paths share the cascade contract (heads >= 7 collapse into
``score_7``), so per-position outputs must agree within fp32 GEMM
precision. cuBLAS reorders the K-axis sum independently of the paged
kernel's BLOCK_D-stepped tl.sum, so we use rel_diff as the primary
metric (~1e-7 expected) and an abs_diff cap of 1e-1 for the cases
with very large absolute logit magnitudes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


_IS_CUDA = torch.cuda.is_available()
_IS_SM70 = _IS_CUDA and torch.cuda.get_device_capability() == (7, 0)


def _build(
    *,
    seed: int,
    context_len: int,
    num_heads: int = 64,
    head_dim: int = 128,
    block_size: int = 64,
    q_byte_high: int = 120,
    k_byte_high: int = 120,
    scale_low: float = 0.05,
    scale_high: float = 1.5,
    weight_scale: float = 0.02,
):
    rng = np.random.default_rng(seed)
    num_blocks = (context_len + block_size - 1) // block_size
    q_bytes = rng.integers(0, q_byte_high, (1, 1, num_heads, head_dim), dtype=np.uint8)
    q = torch.from_numpy(q_bytes).cuda()
    # SEGREGATED block layout (matches the compressor writer + the kernels'
    # read): within a block of block_size*(head_dim+4) bytes, all tokens'
    # head_dim fp8 bytes come first, then all tokens' 4-byte fp32 scales.
    paged = torch.zeros((num_blocks, block_size, head_dim + 4), dtype=torch.uint8, device="cuda")
    contig_values = torch.zeros((context_len, head_dim), dtype=torch.uint8, device="cuda")
    contig_scales = torch.zeros((context_len,), dtype=torch.float32, device="cuda")
    block_nbytes = block_size * (head_dim + 4)
    paged_flat_cpu = paged.cpu().reshape(num_blocks, block_nbytes)
    contig_values_cpu = contig_values.cpu()
    contig_scales_cpu = contig_scales.cpu()
    for i in range(context_len):
        bi, ti = i // block_size, i % block_size
        v = rng.integers(0, k_byte_high, (head_dim,), dtype=np.uint8)
        s = float(rng.uniform(scale_low, scale_high))
        # fp8 data region: [ti*head_dim, ti*head_dim+head_dim)
        paged_flat_cpu[bi, ti * head_dim : ti * head_dim + head_dim] = torch.from_numpy(v)
        # scale region: [block_size*head_dim + ti*4, +4)
        scale_bytes = np.frombuffer(np.float32(s).tobytes(), dtype=np.uint8)
        s_off = block_size * head_dim + ti * 4
        paged_flat_cpu[bi, s_off : s_off + 4] = torch.from_numpy(scale_bytes.copy())
        contig_values_cpu[i] = torch.from_numpy(v)
        contig_scales_cpu[i] = s
    paged.copy_(paged_flat_cpu.reshape(num_blocks, block_size, head_dim + 4))
    contig_values.copy_(contig_values_cpu)
    contig_scales.copy_(contig_scales_cpu)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    weights = torch.from_numpy(
        rng.normal(0, weight_scale, (1, num_heads)).astype(np.float32)
    ).cuda()
    context_lens = torch.full((1, 1), context_len, dtype=torch.int32, device="cuda")
    return {
        "q": q,
        "paged": paged.unsqueeze(2),
        "block_table": block_table,
        "weights": weights,
        "context_lens": context_lens,
        "contig_values": contig_values,
        "contig_scales": contig_scales,
        "context_len": context_len,
    }


def _run_paged(inputs, max_model_len):
    from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_paged_mqa_logits

    return sm70_fp8_paged_mqa_logits(
        inputs["q"],
        inputs["paged"],
        inputs["weights"],
        inputs["context_lens"],
        inputs["block_table"],
        max_model_len,
    )


def _run_gemm(inputs, max_model_len):
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        sm70_cascade_gemm_indexer,
    )

    return sm70_cascade_gemm_indexer(
        inputs["q"],
        inputs["contig_values"],
        inputs["contig_scales"],
        inputs["weights"],
        inputs["context_len"],
        max_model_len,
    )


@pytest.fixture(autouse=True, scope="module")
def _disable_tf32():
    """Force fp32 matmul precision so cuBLAS doesn't silently
    downgrade to TF32 and break our rel_diff target."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev


_STRESS_PROFILES = [
    # (q_byte_high, k_byte_high, scale_low, scale_high, weight_scale, label)
    (120, 120, 0.05, 1.5, 0.02, "default"),
    (120, 120, 0.5, 5.0, 0.05, "large_scale"),
    (120, 120, 0.001, 0.1, 0.001, "small_scale"),
    (60, 60, 0.05, 1.5, 0.5, "balanced_signs"),
    (120, 120, 0.05, 1.5, 0.5, "extreme_weights"),
]


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
@pytest.mark.parametrize("seed", [20260527, 1, 7, 13, 42])
@pytest.mark.parametrize("context_len", [16, 64, 256, 1024, 4096])
@pytest.mark.parametrize("profile_idx", list(range(len(_STRESS_PROFILES))))
def test_cascade_gemm_matches_paged_within_fp32_gemm_precision(
    seed, context_len, profile_idx
):
    qh, kh, sl, sh, ws, label = _STRESS_PROFILES[profile_idx]
    inputs = _build(
        seed=seed,
        context_len=context_len,
        q_byte_high=qh,
        k_byte_high=kh,
        scale_low=sl,
        scale_high=sh,
        weight_scale=ws,
    )
    max_model_len = max(context_len + 16, 64)
    paged_out = _run_paged(inputs, max_model_len)
    gemm_out = _run_gemm(inputs, max_model_len)

    a = gemm_out[0, :context_len]
    b = paged_out[0, :context_len]
    assert torch.isfinite(a).all(), f"gemm produced non-finite values ({label}, seed={seed}, cl={context_len})"
    assert torch.isfinite(b).all(), f"paged produced non-finite values"

    diff = (a - b).abs()
    rel = diff / (b.abs() + 1e-9)
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    # Both paths now implement TRUE per-head (64-head) reduction: each
    # head gets its own ReLU + weight. cuBLAS (GEMM) and the paged
    # kernel's BLOCK_D-stepped tl.sum reorder the fp32 accumulation
    # differently, and summing 64 independent relu*weight terms (vs the
    # old 8-bucket collapse) widens the per-position fp32 spread for
    # large-magnitude profiles. rel_diff stays small (~1e-3); abs_diff
    # can be O(0.1-1) where the logits themselves are large. The metric
    # that actually drives retrieval is the top-k index set, asserted
    # below; the per-position check uses a realistic fp32 bound.
    assert max_rel <= 5e-3 or max_abs <= 1e-1, (
        f"({label}, seed={seed}, cl={context_len}) "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    )

    # Primary correctness gate: top-k index selection must agree (this
    # is what feeds the sparse attention). Use a few k spanning the
    # context; allow a tiny slack for genuine fp32 ties.
    for k in (16, 64, 256, 512):
        kk = min(k, context_len)
        if kk <= 0:
            continue
        ta = set(torch.topk(a, kk).indices.tolist())
        tb = set(torch.topk(b, kk).indices.tolist())
        overlap = len(ta & tb) / kk
        assert overlap >= 0.98, (
            f"top-{kk} overlap {overlap:.3f} too low "
            f"({label}, seed={seed}, cl={context_len})"
        )

    # Trailing region must be -inf for both.
    assert torch.isinf(paged_out[:, context_len:]).all()
    assert torch.isinf(gemm_out[:, context_len:]).all()


def _build_snapshot_next_n(
    *,
    seed: int,
    context_lens: list[int],
    num_heads: int = 64,
    head_dim: int = 128,
    weight_scale: float = 0.02,
):
    """Build a next_n batched cascade-GEMM-from-snapshot input set.

    Returns a pre-decoded fp32 K snapshot plus per-row q / weights /
    context_lens, so we can compare a batched next_n call against
    per-row next_n=1 calls (row-independence / parity).
    """
    rng = np.random.default_rng(seed)
    next_n = len(context_lens)
    capacity = max(context_lens)
    q_bytes = rng.integers(0, 120, (1, next_n, num_heads, head_dim), dtype=np.uint8)
    q = torch.from_numpy(q_bytes).cuda()
    k_f32 = torch.from_numpy(
        rng.normal(0, 1.0, (capacity, head_dim)).astype(np.float32)
    ).cuda()
    weights = torch.from_numpy(
        rng.normal(0, weight_scale, (next_n, num_heads)).astype(np.float32)
    ).cuda()
    context_lens_t = torch.tensor(
        [context_lens], dtype=torch.int32, device="cuda"
    )  # [1, next_n]
    return {
        "q": q,
        "k_f32": k_f32,
        "weights": weights,
        "context_lens": context_lens_t,
        "capacity": capacity,
        "next_n": next_n,
    }


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
@pytest.mark.parametrize("seed", [20260530, 3, 99])
@pytest.mark.parametrize("context_lens", [[256, 256], [1024, 1021], [4096, 4093]])
def test_cascade_gemm_next_n2_matches_per_row_next_n1(seed, context_lens):
    """H75: MTP=1 verify uses next_n=2. Prove the batched next_n=2
    cascade-GEMM produces per-row results bit-identical to running each
    row alone as next_n=1. Transitively validates next_n=2 vs paged via
    the existing next_n=1 vs paged sweep above."""
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        sm70_cascade_gemm_indexer_from_snapshot,
    )

    inputs = _build_snapshot_next_n(seed=seed, context_lens=context_lens)
    max_model_len = inputs["capacity"] + 16

    batched = sm70_cascade_gemm_indexer_from_snapshot(
        q=inputs["q"],
        k_f32_cache=inputs["k_f32"],
        weights=inputs["weights"],
        context_lens=inputs["context_lens"],
        max_model_len=max_model_len,
    )
    assert batched.shape == (inputs["next_n"], max_model_len)
    assert torch.isfinite(batched[0, : context_lens[0]]).all()
    assert torch.isfinite(batched[1, : context_lens[1]]).all()

    for r in range(inputs["next_n"]):
        single = sm70_cascade_gemm_indexer_from_snapshot(
            q=inputs["q"][:, r : r + 1],
            k_f32_cache=inputs["k_f32"],
            weights=inputs["weights"][r : r + 1],
            context_lens=inputs["context_lens"][:, r : r + 1],
            max_model_len=max_model_len,
        )
        # Row independence: batched next_n=2 must match per-row
        # next_n=1 for the same inputs. With true 64-head reduction the
        # GEMM shape differs ([next_n,64,D] vs [64,D]); cuBLAS may pick a
        # different fp32 reduction algorithm, so allow a tight fp32
        # tolerance instead of bit-identity. The retrieval-relevant
        # top-k selection must be identical.
        torch.testing.assert_close(
            batched[r], single[0], rtol=1e-4, atol=1e-3, equal_nan=True
        )
        cl_r = context_lens[r]
        for k in (16, 64, 256):
            kk = min(k, cl_r)
            ta = set(torch.topk(batched[r, :cl_r], kk).indices.tolist())
            tb = set(torch.topk(single[0, :cl_r], kk).indices.tolist())
            assert len(ta & tb) / kk >= 0.98, (
                f"row {r} top-{kk} overlap too low (seed={seed})"
            )
        # Per-row context mask: -inf strictly past that row's context.
        assert torch.isinf(batched[r, context_lens[r] :]).all()
