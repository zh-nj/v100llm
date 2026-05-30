# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 indexer head-reduction parity vs the canonical 64-head reference.

This closes the blind spot that hid the long-context haystack bug: the
existing cascade-vs-paged suite only proved the two SM70 paths AGREE
with each other, not that either matches the model's defined indexer
score. Both shared an 8-bucket head collapse (heads >= 7 folded into one
ReLU + weight w_7, weights 8..63 discarded), which is wrong for the real
indexer (NUM_HEADS=64) because relu(sum) != sum(relu).

The model's indexer score (config index_n_heads=64) is per-head:

    logit(k) = sum_{h=0..63} relu(q_h . k) * w_h

This test compares both the paged decode kernel and the cascade-GEMM
decode path against a torch reference implementing exactly that, on the
indexer's real dims (H=64, D=128), and asserts:
  - score Pearson correlation ~ 1.0
  - top-k index-set overlap ~ 1.0 (what actually drives retrieval)
  - dropping ANY per-head weight changes the output (weight completeness)

See .kiro/specs/deepseek-v4-indexer-head-reduction-correctness/.
"""

from __future__ import annotations

import pytest
import torch


_IS_CUDA = torch.cuda.is_available()
_IS_SM70 = _IS_CUDA and torch.cuda.get_device_capability() == (7, 0)

FP8 = torch.float8_e4m3fn
H = 64
D = 128
BLOCK_SIZE = 64


def _build_paged_kv(n_ctx: int, seed: int):
    """Paged KV cache [num_blocks, block_size, 1, D+4] uint8 + the exact
    dequantized fp32 K the kernel sees + contiguous K bytes/scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    npages = (n_ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = npages * BLOCK_SIZE
    bytes_per_tok = D + 4
    kv_u8 = torch.zeros(
        (npages, BLOCK_SIZE, 1, bytes_per_tok), dtype=torch.uint8, device="cuda"
    )
    k_raw = torch.randn((padded, D), generator=g, device="cuda") * 0.5
    absmax = k_raw.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    scale = absmax / 448.0
    k_fp8 = (k_raw / scale).to(FP8)
    k_f32_true = k_fp8.to(torch.float32) * scale
    flat = kv_u8.view(npages * BLOCK_SIZE, bytes_per_tok)
    flat[:, :D] = k_fp8.view(torch.uint8)
    flat[:, D : D + 4] = (
        scale.squeeze(1).to(torch.float32).view(torch.uint8).view(-1, 4)
    )
    block_table = torch.arange(
        npages, dtype=torch.int32, device="cuda"
    ).view(1, npages)
    contig_values = k_fp8.view(torch.uint8)[:n_ctx].contiguous()
    contig_scales = scale.squeeze(1)[:n_ctx].contiguous()
    return {
        "kv_u8": kv_u8,
        "k_f32_true": k_f32_true[:n_ctx],
        "block_table": block_table,
        "contig_values": contig_values,
        "contig_scales": contig_scales,
    }


def _canonical_64head(q_fp8, k_f32_true, weights, n_ctx, mml):
    """Per-head ReLU + weighted sum over ALL heads (model semantics)."""
    out = torch.full([1, mml], float("-inf"), dtype=torch.float32, device="cuda")
    q_i = q_fp8[0, 0].to(torch.float32)        # [H, D]
    w_i = weights[0].float()                   # [H]
    score = torch.einsum("nd,hd->nh", k_f32_true, q_i)  # [n_ctx, H]
    score = torch.relu(score) * w_i[None, :]
    out[0, :n_ctx] = score.sum(dim=1)
    return out


def _topk_overlap(a, b, k):
    kk = min(k, a.numel())
    ta = set(torch.topk(a, kk).indices.tolist())
    tb = set(torch.topk(b, kk).indices.tolist())
    return len(ta & tb) / max(kk, 1)


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
@pytest.mark.parametrize("n_ctx,topk", [(512, 256), (2048, 256), (8192, 256), (16384, 512)])
def test_paged_kernel_matches_64head_reference(n_ctx, topk):
    from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_paged_mqa_logits

    data = _build_paged_kv(n_ctx, seed=11 + n_ctx)
    g = torch.Generator(device="cuda").manual_seed(7 + n_ctx)
    q_fp8 = torch.randn((1, 1, H, D), generator=g, device="cuda").to(FP8)
    weights = torch.randn((1, H), generator=g, device="cuda").abs()
    cl = torch.tensor([[n_ctx]], dtype=torch.int32, device="cuda")
    mml = ((n_ctx + 255) // 256) * 256

    got = sm70_fp8_paged_mqa_logits(
        q_fp8, data["kv_u8"], weights, cl, data["block_table"], mml
    )[0, :n_ctx]
    ref = _canonical_64head(q_fp8, data["k_f32_true"], weights, n_ctx, mml)[0, :n_ctx]

    corr = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
    assert corr >= 0.999, f"paged vs 64-head corr {corr:.4f} (ctx={n_ctx})"
    for k in (64, topk):
        ov = _topk_overlap(got, ref, k)
        assert ov >= 0.99, f"paged top-{k} overlap {ov:.3f} (ctx={n_ctx})"


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
@pytest.mark.parametrize("n_ctx,topk", [(512, 256), (2048, 256), (8192, 256), (16384, 512)])
def test_cascade_gemm_matches_64head_reference(n_ctx, topk):
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        sm70_cascade_gemm_indexer,
    )

    data = _build_paged_kv(n_ctx, seed=11 + n_ctx)
    g = torch.Generator(device="cuda").manual_seed(7 + n_ctx)
    q_fp8 = torch.randn((1, 1, H, D), generator=g, device="cuda").to(FP8)
    weights = torch.randn((1, H), generator=g, device="cuda").abs()
    mml = ((n_ctx + 255) // 256) * 256

    got = sm70_cascade_gemm_indexer(
        q_fp8, data["contig_values"], data["contig_scales"], weights, n_ctx, mml
    )[0, :n_ctx]
    ref = _canonical_64head(q_fp8, data["k_f32_true"], weights, n_ctx, mml)[0, :n_ctx]

    corr = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
    assert corr >= 0.999, f"cascade vs 64-head corr {corr:.4f} (ctx={n_ctx})"
    for k in (64, topk):
        ov = _topk_overlap(got, ref, k)
        assert ov >= 0.99, f"cascade top-{k} overlap {ov:.3f} (ctx={n_ctx})"


@pytest.mark.skipif(not _IS_SM70, reason="SM70 GPU required")
def test_every_head_weight_is_used():
    """Weight completeness: zeroing any single per-head weight must change
    the output. The old 8-bucket collapse silently dropped weights 8..63,
    so this guards against any head-folding regression."""
    from vllm.model_executor.layers.sm70_mqa_logits import sm70_fp8_paged_mqa_logits

    n_ctx = 1024
    data = _build_paged_kv(n_ctx, seed=123)
    g = torch.Generator(device="cuda").manual_seed(321)
    q_fp8 = torch.randn((1, 1, H, D), generator=g, device="cuda").to(FP8)
    weights = torch.randn((1, H), generator=g, device="cuda").abs() + 0.1
    cl = torch.tensor([[n_ctx]], dtype=torch.int32, device="cuda")
    mml = ((n_ctx + 255) // 256) * 256

    base = sm70_fp8_paged_mqa_logits(
        q_fp8, data["kv_u8"], weights, cl, data["block_table"], mml
    )[0, :n_ctx].clone()

    # Check a spread of heads, including ones the 8-bucket bug dropped.
    for h in (0, 7, 8, 31, 63):
        w2 = weights.clone()
        w2[0, h] = 0.0
        out = sm70_fp8_paged_mqa_logits(
            q_fp8, data["kv_u8"], w2, cl, data["block_table"], mml
        )[0, :n_ctx]
        max_change = (out - base).abs().max().item()
        assert max_change > 1e-3, (
            f"zeroing weight[{h}] did not change output "
            f"(max_change={max_change:.3e}) -- head {h} is being dropped"
        )
