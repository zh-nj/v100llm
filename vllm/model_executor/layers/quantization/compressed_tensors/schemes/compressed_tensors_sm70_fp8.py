# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import QuantizationStrategy

import vllm.envs as envs
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
    CompressedTensorsW8A16Fp8,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
)
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode_linear import (  # noqa: E501
    apply_sm70_fp8_runtime_decode_layer,
    prepare_sm70_fp8_runtime_decode_layer,
)
from vllm.model_executor.utils import replace_parameter

__all__ = ["CompressedTensorsSM70Fp8"]


class CompressedTensorsSM70Fp8(CompressedTensorsW8A16Fp8):
    """SM70 dense FP8 path backed by the runtime-decode GEMM foundation."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def _direct_gemm_enabled() -> bool:
        return envs.VLLM_SM70_FP8_DIRECT_GEMM

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if self.strategy == QuantizationStrategy.TENSOR:
            weight, weight_scale, _ = process_fp8_weight_tensor_strategy(
                layer.weight,
                layer.weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight, weight_scale, _ = process_fp8_weight_channel_strategy(
                layer.weight,
                layer.weight_scale,
                getattr(layer, "input_scale", None),
            )
            weight_scale = weight_scale.reshape(-1)
        elif self.strategy == QuantizationStrategy.BLOCK:
            weight, weight_scale = process_fp8_weight_block_strategy(
                layer.weight,
                layer.weight_scale,
            )
        else:
            raise ValueError(
                f"Unsupported SM70 compressed-tensors FP8 strategy: {self.strategy}"
            )

        replace_parameter(layer, "weight", weight.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        prepare_sm70_fp8_runtime_decode_layer(
            layer,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_block_size=tuple(self.weight_block_size)
            if self.weight_block_size is not None
            else None,
            direct_block_gemm_enabled=self._direct_gemm_enabled(),
        )
        layer.input_scale = None
        layer._already_called_process_weights_after_loading = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_sm70_fp8_runtime_decode_layer(layer, x, bias)
