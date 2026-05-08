# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for sm70_fp16_einsum_bhr_hdr_bhd (R5b)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture
def device():
    return torch.device("cuda")


def _run_and_compare(a, b, atol=5e-3, rtol=1e-2):
    from vllm.model_executor.layers.sm70_fp16_einsum_bmm import (
        sm70_fp16_einsum_bhr_hdr_bhd,
    )

    got = sm70_fp16_einsum_bhr_hdr_bhd(a, b)
    # Reference in fp32 — fp16 einsum would use the same accumulator so
    # we must compare against a fp32 reference to give the kernel a fair
    # numerical envelope.
    ref = torch.einsum("bhr,hdr->bhd", a.float(), b.float()).to(torch.float16)
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    return got


@pytest.mark.parametrize("T", [1, 7, 32, 100])
@pytest.mark.parametrize("G", [1, 2])
@pytest.mark.parametrize("D,R", [(64, 64), (1024, 4096)])
def test_sm70_fp16_einsum_shapes(T, G, D, R, device):
    """Correctness across shapes relevant to DeepSeek V4 prefill."""
    torch.manual_seed(0)
    a = torch.randn(T, G, R, dtype=torch.float16, device=device) * 0.1
    b = torch.randn(G, D, R, dtype=torch.float16, device=device) * 0.1
    _run_and_compare(a, b)


def test_sm70_fp16_einsum_zero_tokens(device):
    """Empty T must return shape-correct empty tensor without launching."""
    from vllm.model_executor.layers.sm70_fp16_einsum_bmm import (
        sm70_fp16_einsum_bhr_hdr_bhd,
    )
    a = torch.empty(0, 1, 4096, dtype=torch.float16, device=device)
    b = torch.empty(1, 1024, 4096, dtype=torch.float16, device=device)
    out = sm70_fp16_einsum_bhr_hdr_bhd(a, b)
    assert out.shape == (0, 1, 1024)
    assert out.dtype == torch.float16


def test_sm70_fp16_einsum_small_D(device):
    """D < BLOCK_D (64) triggers the clamp branch."""
    torch.manual_seed(1)
    a = torch.randn(16, 1, 128, dtype=torch.float16, device=device) * 0.1
    b = torch.randn(1, 32, 128, dtype=torch.float16, device=device) * 0.1
    _run_and_compare(a, b)


def test_sm70_fp16_einsum_non_contiguous_a(device):
    """Strided a input should work via stride arguments."""
    torch.manual_seed(2)
    a_big = torch.randn(16, 2, 4096, dtype=torch.float16, device=device) * 0.1
    a = a_big[:, :1, :]  # slice G -> non-contig stride
    b = torch.randn(1, 1024, 4096, dtype=torch.float16, device=device) * 0.1
    _run_and_compare(a, b)


def test_sm70_fp16_einsum_large_prefill(device):
    """Production-sized tile — prefill 2k tokens on TP=8."""
    torch.manual_seed(3)
    T, G, D, R = 1803, 1, 1024, 4096
    a = torch.randn(T, G, R, dtype=torch.float16, device=device) * 0.05
    b = torch.randn(G, D, R, dtype=torch.float16, device=device) * 0.05
    _run_and_compare(a, b, atol=1e-2, rtol=2e-2)
