# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


SM70_FP8_LAYOUT_TENSOR = 0
SM70_FP8_LAYOUT_CHANNEL = 1
SM70_FP8_LAYOUT_BLOCK = 2
SM70_FP8_SMALL_PANEL_N = 128
SM70_FP8_MEDIUM_PANEL_N = 512
SM70_FP8_MAX_PANEL_N = 1024
# decoded_panel + packed_panel are both fp16, so runtime workspace bytes are
# approximately 4 * panel_n * logical_k.
SM70_FP8_MAX_RUNTIME_WORKSPACE_BYTES = 64 * 1024 * 1024


@dataclass
class Sm70Fp8RuntimeWorkspace:
    decoded_panel: torch.Tensor
    packed_panel: torch.Tensor
    meta_buffer: torch.Tensor
    decoded_rows: int
    decoded_cols: int
    packed_rows: int
    packed_cols: int
    meta_ints: int
    capacity_m: int


_SM70_FP8_SHARED_WORKSPACE_CACHE: dict[
    tuple[str, int, int], Sm70Fp8RuntimeWorkspace
] = {}


def infer_sm70_fp8_layout(
    weight_scale: torch.Tensor,
    weight_block_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    if weight_block_size is not None:
        return (SM70_FP8_LAYOUT_BLOCK, -1, weight_block_size[0], weight_block_size[1])
    if weight_scale.ndim == 0 or tuple(weight_scale.shape) == (1,):
        return (SM70_FP8_LAYOUT_TENSOR, -1, 0, 0)
    if weight_scale.ndim == 1:
        return (SM70_FP8_LAYOUT_CHANNEL, 0, 0, 0)
    return (SM70_FP8_LAYOUT_CHANNEL, 1, 0, 0)


def _align_down(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


def select_sm70_fp8_panel_n(
    *,
    logical_n: int,
    logical_k: int,
    block_n: int,
) -> int:
    if logical_n <= 0:
        raise ValueError("logical_n must be positive")
    if logical_k <= 0:
        raise ValueError("logical_k must be positive")
    if block_n <= 0:
        raise ValueError("block_n must be positive")

    if logical_n >= SM70_FP8_MAX_PANEL_N:
        target_panel_n = SM70_FP8_MAX_PANEL_N
    elif logical_n >= SM70_FP8_MEDIUM_PANEL_N:
        target_panel_n = SM70_FP8_MEDIUM_PANEL_N
    else:
        target_panel_n = SM70_FP8_SMALL_PANEL_N

    budget_rows = SM70_FP8_MAX_RUNTIME_WORKSPACE_BYTES // (4 * logical_k)
    budget_panel_n = max(block_n, _align_down(budget_rows, block_n))
    return max(block_n, min(target_panel_n, budget_panel_n))


def alloc_sm70_fp8_workspace(
    workspace_meta: torch.Tensor,
    *,
    device: torch.device,
    m_capacity: int,
) -> Sm70Fp8RuntimeWorkspace:
    decoded_rows, decoded_cols, packed_rows, packed_cols, meta_ints = (
        int(v) for v in workspace_meta.tolist()
    )
    return Sm70Fp8RuntimeWorkspace(
        decoded_panel=torch.empty(
            (decoded_rows, decoded_cols), dtype=torch.float16, device=device
        ),
        packed_panel=torch.empty(
            (packed_rows, packed_cols), dtype=torch.float16, device=device
        ),
        meta_buffer=torch.empty((meta_ints,), dtype=torch.int64, device=device),
        decoded_rows=decoded_rows,
        decoded_cols=decoded_cols,
        packed_rows=packed_rows,
        packed_cols=packed_cols,
        meta_ints=meta_ints,
        capacity_m=m_capacity,
    )


def _workspace_has_capacity(
    workspace: Sm70Fp8RuntimeWorkspace,
    workspace_meta: torch.Tensor,
) -> bool:
    decoded_rows, decoded_cols, packed_rows, packed_cols, meta_ints = (
        int(v) for v in workspace_meta.tolist()
    )
    return (
        workspace.decoded_rows >= decoded_rows
        and workspace.decoded_cols >= decoded_cols
        and workspace.packed_rows >= packed_rows
        and workspace.packed_cols >= packed_cols
        and workspace.meta_ints >= meta_ints
    )


def _workspace_cache_key(
    layer: torch.nn.Module,
    device: torch.device,
) -> tuple[str, int, int]:
    device_idx = -1 if device.index is None else int(device.index)
    workspace_cols = getattr(layer, "_sm70_fp8_workspace_cols", None)
    if workspace_cols is None:
        workspace_cols = int(layer._sm70_fp8_workspace_meta[1].item())
    return (device.type, device_idx, int(workspace_cols))


def ensure_sm70_fp8_workspace(
    layer: torch.nn.Module,
    device: torch.device,
) -> Sm70Fp8RuntimeWorkspace:
    cache_key = _workspace_cache_key(layer, device)
    workspace = _SM70_FP8_SHARED_WORKSPACE_CACHE.get(cache_key)
    if workspace is None or not _workspace_has_capacity(
        workspace,
        layer._sm70_fp8_workspace_meta,
    ):
        workspace = alloc_sm70_fp8_workspace(
            layer._sm70_fp8_workspace_meta,
            device=device,
            m_capacity=1,
        )
        _SM70_FP8_SHARED_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def get_or_create_sm70_fp8_workspace(
    layer: torch.nn.Module,
    x_2d: torch.Tensor,
) -> Sm70Fp8RuntimeWorkspace:
    cache_key = _workspace_cache_key(layer, x_2d.device)
    workspace = _SM70_FP8_SHARED_WORKSPACE_CACHE.get(cache_key)
    if workspace is None:
        workspace = alloc_sm70_fp8_workspace(
            layer._sm70_fp8_workspace_meta,
            device=x_2d.device,
            m_capacity=int(x_2d.shape[0]),
        )
        _SM70_FP8_SHARED_WORKSPACE_CACHE[cache_key] = workspace
    return workspace
