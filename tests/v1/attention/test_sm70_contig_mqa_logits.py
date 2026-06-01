# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU correctness for the SM70 contiguous-K MQA logits decode kernel.

Compares ``sm70_fp8_contiguous_mqa_logits`` against the existing
``sm70_fp8_paged_mqa_logits`` on a synthetic single-decode-row input.
The two paths must produce element-wise identical fp32 logits because
the FP8 e4m3fn decode is deterministic and the head/D-chunk reduction
order is mirrored.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


_IS_CUDA = torch.cuda.is_available()
_IS_SM70 = _IS_CUDA and torch.cuda.get_device_capability() == (7, 0)

cuda_required = pytest.mark.skipif(
    not _IS_CUDA, reason="CUDA device required"
)


def _build_synthetic_inputs(
    *,
    seed: int,
    context_len: int,
    block_size: int = 64,
    num_heads: int = 8,
    head_dim: int = 128,
):
    """Build paged + contiguous views of the same FP8 K bytes."""
    rng = np.random.default_rng(seed)
    num_blocks = (context_len + block_size - 1) // block_size

    # FP8 bytes (avoid 0x7F/0xFF NaN encodings).
    q_bytes = rng.integers(low=0, high=120, size=(1, 1, num_heads, head_dim), dtype=np.uint8)
    q = torch.from_numpy(q_bytes).cuda()

    # Paged layout: [num_blocks, block_size, head_dim+4] uint8, SEGREGATED
    # within a block (all tokens' head_dim fp8 first, then all 4-byte
    # scales) -- matches the compressor writer + the kernel reads.
    paged = torch.zeros((num_blocks, block_size, head_dim + 4), dtype=torch.uint8, device="cuda")
    contig_values = torch.zeros((context_len, head_dim), dtype=torch.uint8, device="cuda")
    contig_scales = torch.zeros((context_len,), dtype=torch.float32, device="cuda")

    block_nbytes = block_size * (head_dim + 4)
    paged_flat_cpu = paged.cpu().reshape(num_blocks, block_nbytes)
    contig_values_cpu = contig_values.cpu()
    contig_scales_cpu = contig_scales.cpu()
    for i in range(context_len):
        bi, ti = i // block_size, i % block_size
        v = rng.integers(low=0, high=120, size=(head_dim,), dtype=np.uint8)
        s = float(rng.uniform(0.05, 1.5))
        paged_flat_cpu[bi, ti * head_dim : ti * head_dim + head_dim] = torch.from_numpy(v)
        scale_bytes = np.frombuffer(np.float32(s).tobytes(), dtype=np.uint8)
        s_off = block_size * head_dim + ti * 4
        paged_flat_cpu[bi, s_off : s_off + 4] = torch.from_numpy(scale_bytes.copy())
        contig_values_cpu[i] = torch.from_numpy(v)
        contig_scales_cpu[i] = float(s)

    paged.copy_(paged_flat_cpu.reshape(num_blocks, block_size, head_dim + 4))
    contig_values.copy_(contig_values_cpu)
    contig_scales.copy_(contig_scales_cpu)

    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    weights = torch.from_numpy(
        rng.normal(0, 0.02, size=(1, num_heads)).astype(np.float32)
    ).cuda()
    context_lens = torch.full((1, 1), context_len, dtype=torch.int32, device="cuda")

    # Wrap the paged buffer to look like vLLM's `kv_cache` layout
    # [num_blocks, block_size, 1, head_dim+4].
    paged_4d = paged.unsqueeze(2)

    return {
        "q": q,
        "paged": paged_4d,
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


def _run_contig(inputs, max_model_len):
    from vllm.model_executor.layers.sm70_contig_mqa_logits import (
        sm70_fp8_contiguous_mqa_logits,
    )

    return sm70_fp8_contiguous_mqa_logits(
        inputs["q"],
        inputs["contig_values"],
        inputs["contig_scales"],
        inputs["weights"],
        inputs["context_len"],
        max_model_len,
    )


@cuda_required
@pytest.mark.parametrize("seed,context_len", [
    (20260527, 8),
    (20260527, 64),
    (20260527, 257),
    (1, 8),
    (1, 64),
    (1, 257),
    (2, 1024),
    (3, 4096),
])
def test_contig_matches_paged_bit_exact(seed, context_len):
    """Contiguous and paged kernels must produce element-wise equal
    fp32 logits up to ``context_len`` keys."""
    if not _IS_SM70:
        pytest.skip("SM70-only kernel")
    inputs = _build_synthetic_inputs(seed=seed, context_len=context_len)
    max_model_len = max(context_len + 16, 64)
    paged_logits = _run_paged(inputs, max_model_len)
    contig_logits = _run_contig(inputs, max_model_len)

    # Both must use the same -inf padding past context_len.
    paged_view = paged_logits[:, :context_len]
    contig_view = contig_logits[:, :context_len]
    torch.testing.assert_close(
        contig_view, paged_view, rtol=0, atol=0,
        msg=f"seed={seed} cl={context_len}: contig vs paged not bit-exact",
    )

    # Trailing region must be -inf for both.
    assert torch.isinf(paged_logits[:, context_len:]).all()
    assert torch.isinf(contig_logits[:, context_len:]).all()
