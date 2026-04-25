# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from copy import deepcopy

import torch
from compressed_tensors.quantization import QuantizationStrategy

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

FLOAT_QUANTIZED_CHANNEL_CONFIG = {
    "format": "float-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "channel",
                "symmetric": True,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
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


def _make_fp8_quant_config():
    ct_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors"
    )
    return ct_module.CompressedTensorsConfig.from_config(
        deepcopy(FLOAT_QUANTIZED_CHANNEL_CONFIG)
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


def _make_fp8_linear(
    *,
    input_size: int = 256,
    output_size: int = 64,
    prefix: str = "model.layers.0.mlp.down_proj",
) -> ReplicatedLinear:
    return ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_fp8_quant_config(),
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


def test_ct_dense_fp8_sm70_selects_runtime_decode_scheme(monkeypatch) -> None:
    from vllm.platforms import current_platform

    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: _Capability(70),
    )

    layer = _make_fp8_linear(input_size=256, output_size=64)

    assert layer.scheme.__class__.__name__ == "CompressedTensorsSM70Fp8"
    assert layer.scheme.strategy == QuantizationStrategy.CHANNEL


def test_ct_dense_fp8_sm70_process_and_apply_use_runtime_decode(monkeypatch) -> None:
    from vllm.platforms import current_platform

    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: _Capability(70),
    )

    layer = _make_fp8_linear(input_size=256, output_size=384)
    layer.weight = torch.nn.Parameter(
        torch.linspace(-1.0, 1.0, steps=layer.weight.numel(), dtype=torch.float32)
        .reshape_as(layer.weight)
        .to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(layer.weight_scale.shape, dtype=torch.float32),
        requires_grad=False,
    )

    ops_module = importlib.import_module("vllm._custom_ops")
    workspace_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode"
    )
    called: dict[str, tuple] = {}

    def fake_prepare(
        weight,
        weight_scale,
        layout_kind,
        scale_axis,
        block_n,
        block_k,
        panel_n,
    ):
        called["prepare"] = (
            weight.dtype,
            tuple(weight.shape),
            tuple(weight_scale.shape),
            layout_kind,
            scale_axis,
            block_n,
            block_k,
            panel_n,
        )
        return (
            weight,
            weight_scale,
            torch.tensor([384, 256, 384, 256, panel_n, layout_kind], dtype=torch.int64),
            torch.tensor([panel_n, 256, panel_n, 256, 8], dtype=torch.int64),
        )

    class _Workspace:
        decoded_panel = torch.empty((128, 256), dtype=torch.float16)
        packed_panel = torch.empty((128, 256), dtype=torch.float16)
        meta_buffer = torch.empty((8,), dtype=torch.int64)

    def fake_runtime_gemm_out(
        out,
        x,
        prepared_weight,
        prepared_scale,
        prepared_meta,
        decoded_panel,
        packed_panel,
        meta_buffer,
    ):
        called["apply"] = (
            tuple(x.shape),
            tuple(prepared_weight.shape),
            tuple(prepared_scale.shape),
            tuple(prepared_meta.tolist()),
            tuple(decoded_panel.shape),
            tuple(packed_panel.shape),
            tuple(meta_buffer.shape),
        )
        out.fill_(5)

    monkeypatch.setattr(ops_module, "sm70_fp8_prepare", fake_prepare, raising=False)
    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )
    monkeypatch.setattr(
        workspace_module,
        "ensure_sm70_fp8_workspace",
        lambda layer, device: _Workspace(),
        raising=False,
    )
    monkeypatch.setattr(
        workspace_module,
        "get_or_create_sm70_fp8_workspace",
        lambda layer, x_2d: _Workspace(),
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)
    out = layer.quant_method.apply(
        layer,
        torch.ones(2, 256, dtype=torch.float16),
        torch.ones(384, dtype=torch.float16),
    )

    assert called["prepare"] == (
        torch.float8_e4m3fn,
        (384, 256),
        (384,),
        1,
        0,
        0,
        0,
        128,
    )
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.numel() == 384 * 256
    assert layer.weight_scale.dtype == torch.float32
    assert tuple(layer.weight_scale.shape) == (384,)
    assert layer._sm70_fp8_runtime_prepared is True
    assert layer._sm70_fp8_direct_prepared is False
    assert layer.input_scale is None
    assert called["apply"] == (
        (2, 256),
        (384, 256),
        (384,),
        (384, 256, 384, 256, 128, 1),
        (128, 256),
        (128, 256),
        (8,),
    )
    assert tuple(out.shape) == (2, 384)
    assert torch.all(out == 6)


def test_sm70_fp8_direct_prepare_adopts_prepared_tensors_as_resident_parameters(
    monkeypatch,
) -> None:
    helper_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.utils."
        "sm70_fp8_runtime_decode_linear"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    for scale_name in ("weight_scale_inv", "weight_scale"):
        layer = torch.nn.Module()
        layer.output_size_per_partition = 256
        layer.logical_widths = [256]
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((256, 128), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            scale_name,
            torch.nn.Parameter(
                torch.ones((2, 1), dtype=torch.float32),
                requires_grad=False,
            ),
        )

        old_weight = layer.weight
        old_scale = getattr(layer, scale_name)
        prepared_weight = torch.empty_like(layer.weight)
        prepared_scale = torch.empty((1, 256), dtype=torch.float16)
        prepared_meta = torch.tensor([256, 128, 128, 128, 128], dtype=torch.int64)

        def fake_direct_prepare(weight, weight_scale, block_n, block_k):
            assert weight is old_weight
            assert weight_scale is old_scale
            assert block_n == 128
            assert block_k == 128
            return prepared_weight, prepared_scale, prepared_meta

        monkeypatch.setattr(
            ops_module,
            "sm70_fp8_direct_prepare",
            fake_direct_prepare,
            raising=False,
        )

        helper_module.prepare_sm70_fp8_runtime_decode_layer(
            layer,
            weight=layer.weight,
            weight_scale=getattr(layer, scale_name),
            weight_block_size=(128, 128),
            direct_block_gemm_enabled=True,
        )

        assert layer.weight is not old_weight
        assert layer.weight.data_ptr() == prepared_weight.data_ptr()
        assert layer._sm70_fp8_prepared_weight is layer.weight
        assert getattr(layer, scale_name) is not old_scale
        assert getattr(layer, scale_name).data_ptr() == prepared_scale.data_ptr()
        assert layer._sm70_fp8_prepared_scale is getattr(layer, scale_name)
        assert layer._sm70_fp8_direct_prepared is True
