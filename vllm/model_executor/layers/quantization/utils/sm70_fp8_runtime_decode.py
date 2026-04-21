# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


SM70_FP8_LAYOUT_BLOCK = 2


@dataclass
class Sm70Fp8RuntimeWorkspace:
    decoded_panel: torch.Tensor
    packed_panel: torch.Tensor
    meta_buffer: torch.Tensor
    capacity_m: int


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
        capacity_m=m_capacity,
    )


def get_or_create_sm70_fp8_workspace(
    layer: torch.nn.Module,
    x_2d: torch.Tensor,
) -> Sm70Fp8RuntimeWorkspace:
    cache = getattr(layer, "_sm70_fp8_workspace_cache", None)
    if cache is None:
        cache = {}
        layer._sm70_fp8_workspace_cache = cache

    device_idx = -1 if x_2d.device.index is None else int(x_2d.device.index)
    workspace = cache.get(device_idx)
    if workspace is None or workspace.capacity_m < int(x_2d.shape[0]):
        workspace = alloc_sm70_fp8_workspace(
            layer._sm70_fp8_workspace_meta,
            device=x_2d.device,
            m_capacity=int(x_2d.shape[0]),
        )
        cache[device_idx] = workspace
    return workspace
