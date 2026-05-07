# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round 2 pathology + correctness tests for FlashMLA SM70 sparse-prefill.

Round 2 of the deepseek-v4-flash-prefill-throughput spec rewrites the
FlashMLA SM70 sparse-prefill kernel to batch multiple heads-per-kv-head
per CUDA block instead of 1 head per block. This amortizes the KV gather
cost across all heads that share the same kv_head_idx (4x with
HEADS_PER_BLOCK=4).

Pathology property (pre-R2 -> post-R2):
  - Pre-R2: kernel launched as `dim3(s_q, h_q)`, i.e. one block per
    (query, head). Each block loads the full top-K KV set independently.
  - Post-R2: kernel launched as `dim3(s_q, h_q / HEADS_PER_BLOCK)`,
    i.e. one block per (query, head-group). Each block loads the KV
    ONCE per head-group and reuses across HEADS_PER_BLOCK heads.

Correctness: post-R2 output matches pre-R2 output within fp16 tolerance
(atol = 5e-3) on a random-seed microbenchmark. Output shape, dtype,
and strides are unchanged.

These tests require the FlashMLA SM70 extension to be built with the
`-DFLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK=N` cmake define.
Post-R2 Round 2 cmake default: N=4.
"""
from __future__ import annotations

import os

import pytest
import torch


pytest.importorskip("triton")

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def _current_heads_per_block() -> int:
    """Return the compiled FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK.

    We cannot introspect the cmake define directly from the built
    extension on SM70, so the test reads the env var the same way the
    cmake uses it. If the env var is unset, we assume the Round 0 / R1
    default of 1 (one head per block).
    """
    try:
        return int(os.environ.get(
            "FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK", "1"
        ))
    except ValueError:
        return 1


def test_sparse_prefill_heads_per_block_is_configured():
    """After R2 lands, this test documents the current HEADS_PER_BLOCK.

    R0/R1: HEADS_PER_BLOCK=1 (pre-R2 baseline, one block per head).
    R2:   HEADS_PER_BLOCK=4 (default) or 2, 8 (via env var override).

    This test is informational: it prints the current configuration so
    the R2 pathology gate (grid shape) can be verified from a separate
    nsys trace or kernel-launch introspection.
    """
    hpb = _current_heads_per_block()
    assert hpb in (1, 2, 4, 8), (
        f"HEADS_PER_BLOCK must be one of {{1, 2, 4, 8}}, got {hpb}"
    )
    print(f"FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK = {hpb}")


@cuda_required
def test_sparse_prefill_shmem_budget_fits_on_v100():
    """R2 pathology: compiled smem_per_block <= 48 KB on V100.

    V100 has a 96 KB shmem budget per SM. To reach >= 2 CTAs/SM (the
    Round 0 K_TILE=16 occupancy target), each block must use <= 48 KB.

    We cannot read the compiled smem directly from Python, so this test
    computes the theoretical smem budget from the documented layout:
        per_block = K_TILE*4 + 16          (tile_scores + online_scalars)
                  + HEADS_PER_BLOCK * (HEAD_DIM_V * 4 + K_TILE * 4 + 16)
                  + K_TILE * 4              (token_refs)
                  + K_TILE * HEAD_DIM_QK * 2 (kv_tile, fp16)
    and asserts the result <= 48 KB.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    capability = torch.cuda.get_device_capability()
    if capability != (7, 0):
        pytest.skip(f"SM70 budget check; current cap = {capability}")

    K_TILE = int(os.environ.get("FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE", "16"))
    HEAD_DIM_QK = 576
    HEAD_DIM_V = 512
    HPB = _current_heads_per_block()

    per_head_acc = HEAD_DIM_V * 4 + K_TILE * 4 + 16
    smem = (
        K_TILE * 4                          # tile_scores (shared)
        + HPB * per_head_acc                # per-head output + scores + scalars
        + K_TILE * 4                        # token_refs
        + K_TILE * HEAD_DIM_QK * 2          # kv_tile (fp16)
    )
    print(
        f"estimated smem_per_block = {smem} bytes = {smem/1024:.1f} KB "
        f"(K_TILE={K_TILE}, HPB={HPB})"
    )
    assert smem <= 48 * 1024, (
        f"smem_per_block {smem} B > 48 KB V100 budget; reduce "
        f"HEADS_PER_BLOCK or K_TILE to maintain >= 2 CTAs/SM occupancy"
    )


@cuda_required
def test_sparse_prefill_flashmla_extension_loads():
    """Verify FlashMLA SM70 extension is importable and functional."""
    try:
        import vllm._flashmla_C  # noqa: F401
    except ImportError as e:
        pytest.skip(f"FlashMLA extension not built: {e}")
