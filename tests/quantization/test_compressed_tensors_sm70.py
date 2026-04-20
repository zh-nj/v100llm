# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from copy import deepcopy

import torch

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsSM70WNA16,
)


PACK_QUANTIZED_CONFIG = {
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": 32,
                "symmetric": True,
                "dynamic": False,
            },
        }
    },
}


class _Capability:
    def __init__(self, value: int):
        self._value = value

    def to_int(self) -> int:
        return self._value


def _make_quant_config():
    ct_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors"
    )
    return ct_module.CompressedTensorsConfig.from_config(
        deepcopy(PACK_QUANTIZED_CONFIG)
    )


def _make_linear(
    *,
    input_size: int = 256,
    output_size: int = 64,
    prefix: str = "model.layers.0.mlp.gate_up_proj",
) -> ReplicatedLinear:
    return ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_quant_config(),
        prefix=prefix,
        disable_tp=True,
    )


def _patch_single_rank_params(monkeypatch) -> None:
    parameter_module = importlib.import_module("vllm.model_executor.parameter")
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def test_ct_dense_sm70_selects_sm70_scheme(monkeypatch) -> None:
    from vllm.platforms import current_platform

    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: _Capability(70),
    )

    layer = _make_linear(input_size=256, output_size=64)

    assert isinstance(layer.scheme, CompressedTensorsSM70WNA16)
    assert layer.scheme._sm70_compatible is True


def test_ct_dense_sm70_incompatible_layer_uses_internal_fallback(monkeypatch) -> None:
    from vllm.platforms import current_platform

    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: _Capability(70),
    )

    layer = _make_linear(input_size=256, output_size=62)

    assert isinstance(layer.scheme, CompressedTensorsSM70WNA16)
    assert layer.scheme._sm70_compatible is False


def test_ct_dense_sm70_post_load_prepares_awq_buffers(monkeypatch) -> None:
    scheme_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "schemes.compressed_tensors_wNa16_sm70"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    called: dict[str, tuple] = {}

    def fake_prepare(qweight, scales, qzeros, group_size):
        called["args"] = (
            tuple(qweight.shape),
            tuple(scales.shape),
            tuple(qzeros.shape),
            group_size,
        )
        return (
            torch.ones(32, 8, dtype=torch.int32, device=qweight.device),
            torch.ones(8, 64, dtype=torch.float16, device=scales.device),
            torch.tensor([256, 32], dtype=torch.int32, device=qweight.device),
        )

    monkeypatch.setattr(ops_module, "awq_sm70_prepare", fake_prepare, raising=False)

    scheme = scheme_module.CompressedTensorsSM70WNA16(
        strategy="group",
        num_bits=4,
        group_size=32,
        symmetric=True,
        layer_name="model.layers.0.mlp.down_proj",
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.zeros(64, 32, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(64, 8, dtype=torch.float16), requires_grad=False),
    )
    layer.register_parameter(
        "weight_shape",
        torch.nn.Parameter(
            torch.tensor([64, 256], dtype=torch.int64),
            requires_grad=False,
        ),
    )

    scheme.process_weights_after_loading(layer)

    assert called["args"] == ((256, 8), (8, 64), (8, 8), 32)
    assert layer._awq_sm70_prepared is True
    assert layer._awq_sm70_k_ld == 256
    assert layer._awq_sm70_q_ld == 32
    assert layer.weight_packed.numel() == 0
    assert layer.weight_scale.numel() == 0


def test_ct_dense_sm70_apply_uses_awq_gemm_sm70(monkeypatch) -> None:
    scheme_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "schemes.compressed_tensors_wNa16_sm70"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    def fake_awq_gemm_sm70(x, qweight, scales, group_size, k_ld, q_ld):
        assert tuple(x.shape) == (2, 256)
        assert tuple(qweight.shape) == (32, 8)
        assert tuple(scales.shape) == (8, 64)
        assert group_size == 32
        assert k_ld == 256
        assert q_ld == 32
        return torch.full((2, 64), 3.0, dtype=torch.float16, device=x.device)

    monkeypatch.setattr(ops_module, "awq_gemm_sm70", fake_awq_gemm_sm70, raising=False)

    scheme = scheme_module.CompressedTensorsSM70WNA16(
        strategy="group",
        num_bits=4,
        group_size=32,
        symmetric=True,
        layer_name="model.layers.0.self_attn.o_proj",
    )

    layer = torch.nn.Module()
    layer._awq_sm70_prepared = True
    layer._awq_sm70_weight = torch.zeros(32, 8, dtype=torch.int32)
    layer._awq_sm70_scales = torch.ones(8, 64, dtype=torch.float16)
    layer._awq_sm70_k_ld = 256
    layer._awq_sm70_q_ld = 32

    x = torch.ones(2, 256, dtype=torch.float16)
    bias = torch.ones(64, dtype=torch.float16)
    out = scheme.apply_weights(layer, x, bias)

    assert tuple(out.shape) == (2, 64)
    assert torch.all(out == 4)
