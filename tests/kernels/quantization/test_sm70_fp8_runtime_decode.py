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


def _to_block_fp8_3d(weight_fp16: torch.Tensor):
    weights = []
    scales = []
    for expert in range(weight_fp16.shape[0]):
        q, s = _to_block_fp8(weight_fp16[expert])
        weights.append(q)
        scales.append(s)
    return torch.stack(weights), torch.stack(scales)


def _sm70_decode_reference_block(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    rows = []
    for row_start in range(0, weight_fp8.shape[0], block_n):
        row_chunks = []
        for col_start in range(0, weight_fp8.shape[1], block_k):
            block = weight_fp8[
                row_start : row_start + block_n, col_start : col_start + block_k
            ].float()
            scale = weight_scale[row_start // block_n, col_start // block_k].float()
            row_chunks.append((block * scale).to(torch.float16))
        rows.append(torch.cat(row_chunks, dim=1))
    return torch.cat(rows, dim=0)


def _sm70_decode_reference_block_3d(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    decoded = []
    for expert in range(weight_fp8.shape[0]):
        decoded.append(
            _sm70_decode_reference_block(
                weight_fp8[expert],
                weight_scale[expert],
                128,
                128,
            )
        )
    return torch.stack(decoded)


def _to_tensor_fp8(weight_fp16: torch.Tensor):
    scale = weight_fp16.float().abs().amax().clamp(min=1e-6) / 448.0
    q = (weight_fp16.float() / scale).to(torch.float8_e4m3fn)
    return q.cuda(), scale.reshape(1).to(torch.float32).cuda()


def _to_channel_fp8(weight_fp16: torch.Tensor, axis: int):
    reduce_dim = 1 if axis == 0 else 0
    scale = weight_fp16.float().abs().amax(dim=reduce_dim).clamp(min=1e-6)
    scale = scale / 448.0
    if axis == 0:
        q = weight_fp16.float() / scale[:, None]
    else:
        q = weight_fp16.float() / scale[None, :]
    return q.to(torch.float8_e4m3fn).cuda(), scale.to(torch.float32).cuda()


def _dequantize_reference(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    layout_kind: int,
    scale_axis: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    if layout_kind == 0:
        return (weight_fp8.float() * float(weight_scale.item())).to(torch.float16)
    if layout_kind == 1 and scale_axis == 0:
        return (weight_fp8.float() * weight_scale[:, None].float()).to(torch.float16)
    if layout_kind == 1 and scale_axis == 1:
        return (weight_fp8.float() * weight_scale[None, :].float()).to(torch.float16)
    return _sm70_decode_reference_block(weight_fp8, weight_scale, block_n, block_k)


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


@pytest.mark.cuda
def test_sm70_fp8_direct_gemm_block_layout_matches_reference():
    _require_sm70()
    torch.manual_seed(3)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(384, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)

    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
    )
    out = torch.empty((x.shape[0], w_ref.shape[0]), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_direct_gemm_out(out, x, prepared_w, prepared_s, prepared_meta)

    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)


@pytest.mark.cuda
def test_sm70_fp8_runtime_gemm_out_is_cuda_graph_capturable():
    _require_sm70()
    torch.manual_seed(0)
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(256, 256, device="cuda", dtype=torch.float16) / 4
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

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
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
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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

    graph.replay()
    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)


@pytest.mark.cuda
def test_sm70_fp8_direct_gemm_out_is_cuda_graph_capturable():
    _require_sm70()
    torch.manual_seed(4)
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(256, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
    )
    out = torch.empty((x.shape[0], w_ref.shape[0]), dtype=torch.float16, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        ops.sm70_fp8_direct_gemm_out(out, x, prepared_w, prepared_s, prepared_meta)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.sm70_fp8_direct_gemm_out(out, x, prepared_w, prepared_s, prepared_meta)

    graph.replay()
    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)


@pytest.mark.cuda
def test_sm70_fp8_moe_direct_gemm_matches_reference():
    _require_sm70()
    torch.manual_seed(10)
    num_experts = 2
    k = 256
    n = 256
    x = torch.randn(3, k, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(num_experts, n, k, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8_3d(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_moe_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
        False,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_w,
        prepared_s,
        int(prepared_meta[3].item()),
        int(prepared_meta[4].item()),
        num_experts,
    )
    expert_offsets = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    out = torch.empty((3, n), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_moe_gemm_out(
        out,
        x,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        128,
        False,
    )

    decoded = _sm70_decode_reference_block_3d(w_fp8, w_scale)
    ref = torch.empty_like(out)
    ref[:2] = x[:2] @ decoded[0].t()
    ref[2:] = x[2:] @ decoded[1].t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)


@pytest.mark.cuda
def test_sm70_fp8_moe_direct_gemm_gated_silu_matches_reference():
    _require_sm70()
    torch.manual_seed(11)
    num_experts = 2
    k = 256
    intermediate = 128
    n = intermediate * 2
    x = torch.randn(3, k, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(num_experts, n, k, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8_3d(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_moe_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
        True,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_w,
        prepared_s,
        int(prepared_meta[3].item()),
        int(prepared_meta[4].item()),
        num_experts,
    )
    expert_offsets = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    out = torch.empty((3, intermediate), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_moe_gemm_out(
        out,
        x,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        128,
        True,
    )

    decoded = _sm70_decode_reference_block_3d(w_fp8, w_scale)
    ref = torch.empty_like(out)
    full0 = x[:2] @ decoded[0].t()
    full1 = x[2:] @ decoded[1].t()
    gate0, up0 = full0.chunk(2, dim=1)
    gate1, up1 = full1.chunk(2, dim=1)
    ref[:2] = torch.nn.functional.silu(gate0) * up0
    ref[2:] = torch.nn.functional.silu(gate1) * up1
    torch.testing.assert_close(out, ref, atol=7e-1, rtol=1e-1)


@pytest.mark.cuda
def test_sm70_fp8_moe_direct_gemm_out_is_cuda_graph_capturable():
    _require_sm70()
    torch.manual_seed(12)
    num_experts = 2
    k = 256
    n = 256
    x = torch.randn(1, k, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(num_experts, n, k, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8_3d(w_ref)
    prepared_w, prepared_s, prepared_meta = ops.sm70_fp8_moe_direct_prepare(
        w_fp8,
        w_scale,
        128,
        128,
        False,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_w,
        prepared_s,
        int(prepared_meta[3].item()),
        int(prepared_meta[4].item()),
        num_experts,
    )
    expert_offsets = torch.tensor([0, 1, 1], dtype=torch.int32, device="cuda")
    out = torch.empty((1, n), dtype=torch.float16, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        ops.sm70_fp8_moe_gemm_out(
            out, x, expert_offsets, ptrs_w, ptrs_s, num_experts, k, n, 128, False
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.sm70_fp8_moe_gemm_out(
            out, x, expert_offsets, ptrs_w, ptrs_s, num_experts, k, n, 128, False
        )
    graph.replay()

    decoded = _sm70_decode_reference_block_3d(w_fp8, w_scale)
    ref = x @ decoded[0].t()
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


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("layout_kind", "scale_axis"),
    [(0, -1), (1, 0), (1, 1), (2, -1)],
)
def test_sm70_fp8_runtime_gemm_supports_multiple_scale_layouts(
    layout_kind: int,
    scale_axis: int,
):
    _require_sm70()
    torch.manual_seed(1)
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(256, 256, device="cuda", dtype=torch.float16) / 4
    if layout_kind == 0:
        w_fp8, w_scale = _to_tensor_fp8(w_ref)
        block_n = block_k = 0
    elif layout_kind == 1:
        w_fp8, w_scale = _to_channel_fp8(w_ref, axis=scale_axis)
        block_n = block_k = 0
    else:
        w_fp8, w_scale = _to_block_fp8(w_ref)
        block_n = block_k = 128

    prepared_w, prepared_s, prepared_meta, workspace_meta = ops.sm70_fp8_prepare(
        w_fp8,
        w_scale,
        layout_kind,
        scale_axis,
        block_n,
        block_k,
        128,
    )
    workspace = alloc_sm70_fp8_workspace(
        workspace_meta,
        device=x.device,
        m_capacity=x.shape[0],
    )
    out = ops.sm70_fp8_runtime_gemm(
        x,
        prepared_w,
        prepared_s,
        prepared_meta,
        workspace.decoded_panel,
        workspace.packed_panel,
        workspace.meta_buffer,
    )
    ref = x @ _dequantize_reference(
        w_fp8,
        w_scale,
        layout_kind,
        scale_axis,
        block_n,
        block_k,
    ).t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)


@pytest.mark.cuda
def test_sm70_fp8_runtime_gemm_raises_capacity_error_for_small_workspace():
    _require_sm70()
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(256, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)
    prepared_w, prepared_s, prepared_meta, _ = ops.sm70_fp8_prepare(
        w_fp8,
        w_scale,
        2,
        -1,
        128,
        128,
        128,
    )
    out = torch.empty((2, 256), device="cuda", dtype=torch.float16)
    decoded = torch.empty((64, 256), device="cuda", dtype=torch.float16)
    packed = torch.empty((64, 256), device="cuda", dtype=torch.float16)
    meta_buffer = torch.empty((4,), device="cuda", dtype=torch.int64)

    with pytest.raises(RuntimeError, match="workspace.*too small"):
        ops.sm70_fp8_runtime_gemm_out(
            out,
            x,
            prepared_w,
            prepared_s,
            prepared_meta,
            decoded,
            packed,
            meta_buffer,
        )
