# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops


def _require_sm70():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 runtime decode test requires a V100")


def _to_block_fp8(weight_fp16: torch.Tensor):
    block_n = 128
    block_k = 128
    out_dim, in_dim = weight_fp16.shape
    scales = []
    q_rows = []
    for row_start in range(0, out_dim, block_n):
        row_end = min(row_start + block_n, out_dim)
        row_q = []
        row_scales = []
        for col_start in range(0, in_dim, block_k):
            col_end = min(col_start + block_k, in_dim)
            block = weight_fp16[row_start:row_end, col_start:col_end].float()
            scale = block.abs().amax().clamp(min=1e-6) / 448.0
            row_scales.append(scale)
            row_q.append((block / scale).to(torch.float8_e4m3fn).to(torch.float16))
        q_rows.append(torch.cat(row_q, dim=1))
        scales.append(torch.stack(row_scales))
    q = torch.cat(q_rows, dim=0).to(torch.float8_e4m3fn)
    s = torch.stack(scales).to(torch.float32).cuda()
    return q.cuda(), s


@pytest.mark.cuda
@pytest.mark.parametrize("out_dim", [128, 320])
def test_sm70_fp8_runtime_gemm_matches_reference(out_dim: int):
    _require_sm70()
    torch.manual_seed(0)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(out_dim, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)

    out = ops.sm70_fp8_runtime_gemm(
        x,
        w_fp8,
        w_scale,
        128,
        128,
        128,
    )
    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)
