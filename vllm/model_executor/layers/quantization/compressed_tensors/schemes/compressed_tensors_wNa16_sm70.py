# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)

from .compressed_tensors_wNa16 import CompressedTensorsWNA16

__all__ = ["CompressedTensorsSM70WNA16"]


class CompressedTensorsSM70WNA16(CompressedTensorsWNA16):
    """SM70 dense int4 path backed by the TurboMind AWQ kernels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sm70_compatible = False

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        self._sm70_compatible = self._can_use_sm70_dense(
            input_size_per_partition=input_size_per_partition,
            output_size_per_partition=output_size_per_partition,
        )

        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel
        )
        scales_and_zp_size = (input_size + group_size - 1) // group_size
        if partition_scales:
            scales_and_zp_size = (
                input_size_per_partition + group_size - 1
            ) // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )
        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }
        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            ),
        }
        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
            if not self.symmetric:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(
                    input_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

    def _infer_sm70_compatibility(self, layer: torch.nn.Module) -> bool:
        if not hasattr(layer, "weight_packed"):
            return False
        return self._can_use_sm70_dense(
            input_size_per_partition=layer.weight_packed.shape[1] * self.pack_factor,
            output_size_per_partition=layer.weight_packed.shape[0],
        )

    def _can_use_sm70_dense(
        self, *, input_size_per_partition: int, output_size_per_partition: int
    ) -> bool:
        return (
            self.symmetric
            and not self.has_g_idx
            and self.pack_factor == 8
            and self.group_size in (32, 64, 128)
            and input_size_per_partition % self.group_size == 0
            and input_size_per_partition % 8 == 0
            and output_size_per_partition % self.pack_factor == 0
        )

    def _ct_to_awq_qweight(self, ct_weight_packed: torch.Tensor) -> torch.Tensor:
        """Convert compressed-tensors [N, K/8] packing to AWQ [K, N/8]."""
        ct_weight_packed = ct_weight_packed.transpose(0, 1).contiguous()
        k_div_pack, out_features = ct_weight_packed.shape
        in_features = k_div_pack * self.pack_factor
        unpacked = torch.zeros(
            in_features,
            out_features,
            dtype=torch.uint8,
            device=ct_weight_packed.device,
        )

        tmp = ct_weight_packed.clone()
        for i in range(self.pack_factor):
            unpacked[i::self.pack_factor, :] = (tmp & 0xF).to(torch.uint8)
            tmp = tmp >> 4

        awq_pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
        grouped = unpacked.view(in_features, -1, self.pack_factor)
        result = grouped[:, :, awq_pack_order[-1]].to(torch.int32)
        for i in range(self.pack_factor - 2, -1, -1):
            result = (result << 4) | grouped[:, :, awq_pack_order[i]].to(torch.int32)

        return result

    def _make_symmetric_awq_qzeros(
        self, *, num_groups: int, output_size_per_partition: int, device: torch.device
    ) -> torch.Tensor:
        zp = (
            torch.tensor([0x88888888], dtype=torch.uint32, device=device)
            .view(torch.int32)
            .item()
        )
        return torch.full(
            (num_groups, output_size_per_partition // self.pack_factor),
            zp,
            dtype=torch.int32,
            device=device,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self._sm70_compatible:
            self._sm70_compatible = self._infer_sm70_compatibility(layer)

        if not self._sm70_compatible or not hasattr(ops, "awq_sm70_prepare"):
            return

        qweight = self._ct_to_awq_qweight(layer.weight_packed.data)
        scales = layer.weight_scale.data.transpose(0, 1).contiguous().to(torch.float16)
        qzeros = self._make_symmetric_awq_qzeros(
            num_groups=scales.shape[0],
            output_size_per_partition=layer.weight_packed.shape[0],
            device=qweight.device,
        )
        tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
            qweight,
            scales,
            qzeros,
            self.group_size,
        )

        layer._awq_sm70_weight = tm_weight
        layer._awq_sm70_scales = tm_scales
        layer._awq_sm70_k_ld = int(meta[0])
        layer._awq_sm70_q_ld = int(meta[1])
        layer._awq_sm70_prepared = True

        layer.weight_packed = Parameter(
            torch.empty(0, dtype=torch.int32, device=tm_weight.device),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.empty(0, dtype=torch.float16, device=tm_weight.device),
            requires_grad=False,
        )
        if hasattr(layer, "weight_zero_point"):
            layer.weight_zero_point = Parameter(
                torch.empty(0, dtype=torch.int32, device=tm_weight.device),
                requires_grad=False,
            )

    def _unpack_ct_weight(self, ct_weight_packed: torch.Tensor) -> torch.Tensor:
        out_features, in_features_div_pack = ct_weight_packed.shape
        unpacked = torch.zeros(
            out_features,
            in_features_div_pack * self.pack_factor,
            dtype=torch.int32,
            device=ct_weight_packed.device,
        )
        tmp = ct_weight_packed.clone()
        for i in range(self.pack_factor):
            unpacked[:, i::self.pack_factor] = (tmp & 0xF).to(torch.int32)
            tmp = tmp >> 4
        return unpacked

    def _apply_ct_fallback(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        assert self.symmetric, "SM70 dense fallback only supports symmetric weights."
        group_size = self.group_size if self.group_size != -1 else x.shape[-1]
        reshaped_x = x.reshape(-1, x.shape[-1])
        qweight = self._unpack_ct_weight(layer.weight_packed.data)
        scales = layer.weight_scale.data.to(reshaped_x.dtype)
        scales = scales.repeat_interleave(group_size, dim=1)[:, : qweight.shape[1]]
        dequant_weight = (qweight.to(reshaped_x.dtype) - 8) * scales
        out = reshaped_x @ dequant_weight.t()
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (dequant_weight.shape[0],))

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if not getattr(layer, "_awq_sm70_prepared", False):
            return self._apply_ct_fallback(layer, x, bias)

        reshaped_x = x.reshape(-1, x.shape[-1])
        out = ops.awq_gemm_sm70(
            reshaped_x,
            layer._awq_sm70_weight,
            layer._awq_sm70_scales,
            self.group_size,
            layer._awq_sm70_k_ld,
            layer._awq_sm70_q_ld,
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))
