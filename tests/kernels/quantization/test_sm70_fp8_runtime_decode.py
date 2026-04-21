# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode import (
    alloc_sm70_fp8_workspace,
)


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
def test_sm70_fp8_prepare_and_runtime_gemm_block_layout_matches_reference():
    _require_sm70()
    torch.manual_seed(0)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(320, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)

    prepared_w, prepared_s, prepared_meta, workspace_meta = ops.sm70_fp8_prepare(
        w_fp8,
        w_scale,
        2,
        -1,
        128,
        128,
        128,
    )
    workspace = alloc_sm70_fp8_workspace(
        workspace_meta,
        device=x.device,
        m_capacity=x.shape[0],
    )
    out = torch.empty((x.shape[0], w_ref.shape[0]), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_runtime_gemm_out(
        out,
        x,
        prepared_w,
        prepared_s,
        prepared_meta,
        workspace.decoded_panel,
        workspace.packed_panel,
        workspace.meta_buffer,
    )

    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)


def test_sm70_fp8_runtime_gemm_fake_uses_prepared_meta_output_dim():
    import vllm._custom_ops as ops_module

    input = torch.empty((4, 256), device="meta", dtype=torch.float16)
    prepared_weight = torch.empty(
        (384, 256), device="meta", dtype=torch.float8_e4m3fn
    )
    prepared_scale = torch.empty((3, 2), device="meta", dtype=torch.float32)
    prepared_meta = torch.tensor(
        [320, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64
    )
    decoded = torch.empty((128, 256), device="meta", dtype=torch.float16)
    packed = torch.empty((384, 256), device="meta", dtype=torch.float16)
    meta_buffer = torch.empty((4,), device="meta", dtype=torch.int64)

    out = ops_module._sm70_fp8_runtime_gemm_fake(
        input,
        prepared_weight,
        prepared_scale,
        prepared_meta,
        decoded,
        packed,
        meta_buffer,
    )

    assert tuple(out.shape) == (4, 320)
