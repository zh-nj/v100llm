# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.utils import replace_parameter

from .sm70_fp8_runtime_decode import (
    SM70_FP8_LAYOUT_BLOCK,
    ensure_sm70_fp8_workspace,
    get_or_create_sm70_fp8_workspace,
    infer_sm70_fp8_layout,
    select_sm70_fp8_panel_n,
)


def _adopt_resident_parameter(
    layer: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if name not in layer._parameters or getattr(layer, name) is None:
        return tensor

    replace_parameter(layer, name, tensor)
    return getattr(layer, name)


def prepare_sm70_fp8_runtime_decode_layer(
    layer: torch.nn.Module,
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_block_size: tuple[int, int] | None,
    direct_block_gemm_enabled: bool,
    adopt_prepared_scale: bool = True,
) -> None:
    layout_kind, scale_axis, block_n, block_k = infer_sm70_fp8_layout(
        weight_scale,
        weight_block_size,
    )
    panel_block_n = block_n if block_n > 0 else 128
    direct_gemm = (
        direct_block_gemm_enabled
        and layout_kind == SM70_FP8_LAYOUT_BLOCK
        and block_n == 128
        and block_k == 128
    )

    if direct_gemm:
        prepared_weight, prepared_scale, prepared_meta = ops.sm70_fp8_direct_prepare(
            weight,
            weight_scale,
            block_n,
            block_k,
        )
        layer._sm70_fp8_direct_prepared = True
    else:
        panel_n = select_sm70_fp8_panel_n(
            logical_n=int(weight.shape[0]),
            logical_k=int(weight.shape[1]),
            block_n=panel_block_n,
        )
        prepared_weight, prepared_scale, prepared_meta, workspace_meta = (
            ops.sm70_fp8_prepare(
                weight,
                weight_scale,
                layout_kind,
                scale_axis,
                block_n,
                block_k,
                panel_n,
            )
        )
        layer._sm70_fp8_direct_prepared = False
        layer._sm70_fp8_panel_n = panel_n
        layer._sm70_fp8_workspace_meta = workspace_meta
        layer._sm70_fp8_workspace_cols = int(workspace_meta[1].item())
        ensure_sm70_fp8_workspace(layer, weight.device)

    prepared_weight = _adopt_resident_parameter(layer, "weight", prepared_weight)
    if adopt_prepared_scale:
        if "weight_scale_inv" in layer._parameters:
            prepared_scale = _adopt_resident_parameter(
                layer,
                "weight_scale_inv",
                prepared_scale,
            )
        elif "weight_scale" in layer._parameters:
            prepared_scale = _adopt_resident_parameter(
                layer,
                "weight_scale",
                prepared_scale,
            )
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_block_shape = weight_block_size
    layer._sm70_fp8_output_size = int(getattr(layer, "output_size_per_partition"))
    layer._sm70_fp8_logical_widths = tuple(getattr(layer, "logical_widths"))
    layer._sm70_fp8_prepared_weight = prepared_weight
    layer._sm70_fp8_prepared_scale = prepared_scale
    layer._sm70_fp8_prepared_meta = prepared_meta


def apply_sm70_fp8_runtime_decode_layer(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not getattr(layer, "_sm70_fp8_runtime_prepared", False):
        raise RuntimeError("SM70 FP8 runtime decode weights were not prepared.")

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    padded_out_dim = int(layer.weight.size(0))
    logical_out_dim = int(getattr(layer, "_sm70_fp8_output_size", padded_out_dim))
    out_padded = torch.empty(
        (x_2d.size(0), padded_out_dim),
        dtype=x_2d.dtype,
        device=x_2d.device,
    )
    if getattr(layer, "_sm70_fp8_direct_prepared", False):
        ops.sm70_fp8_direct_gemm_out(
            out_padded,
            x_2d,
            layer._sm70_fp8_prepared_weight,
            layer._sm70_fp8_prepared_scale,
            layer._sm70_fp8_prepared_meta,
        )
    else:
        workspace = get_or_create_sm70_fp8_workspace(layer, x_2d)
        ops.sm70_fp8_runtime_gemm_out(
            out_padded,
            x_2d,
            layer._sm70_fp8_prepared_weight,
            layer._sm70_fp8_prepared_scale,
            layer._sm70_fp8_prepared_meta,
            workspace.decoded_panel,
            workspace.packed_panel,
            workspace.meta_buffer,
        )
    out = out_padded[:, :logical_out_dim]
    if bias is not None:
        out.add_(bias)
    return out.reshape(x.shape[:-1] + (out.shape[-1],))
