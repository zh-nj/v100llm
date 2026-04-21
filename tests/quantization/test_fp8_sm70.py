# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config


def _patch_single_rank_params(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter_module = importlib.import_module("vllm.model_executor.parameter")
    fused_moe_layer_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.layer"
    )
    distributed_module = importlib.import_module("vllm.distributed")
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
    monkeypatch.setattr(
        fused_moe_layer_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        distributed_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _make_fp8_config() -> Fp8Config:
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )


def _make_linear(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_size: int = 256,
    output_size: int = 64,
    prefix: str = "model.layers.0.mlp.down_proj",
) -> ReplicatedLinear:
    _patch_single_rank_params(monkeypatch)
    return ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_fp8_config(),
        prefix=prefix,
        disable_tp=True,
    )


def _populate_fp8_block_weights(layer: ReplicatedLinear) -> None:
    fp8_dtype = torch.float8_e4m3fn
    layer.weight = torch.nn.Parameter(
        torch.linspace(-1.0, 1.0, steps=layer.weight.numel(), dtype=torch.float32)
        .reshape_as(layer.weight)
        .to(fp8_dtype),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(
            layer.weight_scale_inv.shape,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )


def test_fp8_sm70_linear_uses_runtime_decode_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = _make_linear(monkeypatch)

    assert layer.quant_method.__class__.__name__ == (
        "Fp8SM70RuntimeDecodeLinearMethod"
    )


def test_fp8_sm70_merged_linear_uses_runtime_decode_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = MergedColumnParallelLinear(
        input_size=256,
        output_sizes=[128, 128, 64],
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_fp8_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )

    assert layer.quant_method.__class__.__name__ == (
        "Fp8SM70RuntimeDecodeLinearMethod"
    )


def test_fp8_config_keeps_modules_to_not_convert_alias() -> None:
    config = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": [
                "model.language_model.layers.0.linear_attn.in_proj_a",
            ],
            "weight_block_size": [128, 128],
        }
    )

    assert config.ignored_layers == config.modules_to_not_convert
    assert config.modules_to_not_convert == [
        "model.language_model.layers.0.linear_attn.in_proj_a",
    ]


def test_fp8_sm70_linear_repacks_to_awq_and_prepares(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")

    monkeypatch.setattr(
        fp8_module,
        "_get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = _make_linear(monkeypatch)
    _populate_fp8_block_weights(layer)

    called: dict[str, tuple[tuple[int, ...], ...] | int] = {}

    def fake_prepare(qweight, scales, qzeros, group_size, **kwargs):
        called["shapes"] = (
            tuple(qweight.shape),
            tuple(scales.shape),
            tuple(qzeros.shape),
        )
        called["group_size"] = group_size
        return (
            torch.ones(256, 8, dtype=torch.int32, device=qweight.device),
            torch.ones(2, 64, dtype=torch.int32, device=qweight.device),
            torch.tensor([256, 32], dtype=torch.int64, device=qweight.device),
        )

    monkeypatch.setattr(ops_module, "awq_sm70_prepare", fake_prepare, raising=False)

    layer.quant_method.process_weights_after_loading(layer)

    assert called["shapes"] == ((256, 8), (2, 64), (2, 8))
    assert called["group_size"] == 128
    assert layer._awq_sm70_prepared is True
    assert layer._awq_sm70_k_ld == 256
    assert layer._awq_sm70_q_ld == 32


def test_fp8_sm70_linear_apply_uses_awq_gemm(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70LinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer._awq_sm70_prepared = True
    layer._awq_sm70_weight = torch.zeros(256, 8, dtype=torch.int32)
    layer._awq_sm70_scales = torch.ones(2, 64, dtype=torch.int32)
    layer._awq_sm70_k_ld = 256
    layer._awq_sm70_q_ld = 32

    def fake_awq_gemm_sm70(x, qweight, scales, group_size, k_ld, q_ld):
        assert tuple(x.shape) == (2, 256)
        assert tuple(qweight.shape) == (256, 8)
        assert tuple(scales.shape) == (2, 64)
        assert group_size == 128
        assert k_ld == 256
        assert q_ld == 32
        return torch.full((2, 64), 5, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(ops_module, "awq_gemm_sm70", fake_awq_gemm_sm70, raising=False)

    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert tuple(out.shape) == (2, 64)
    assert torch.all(out == 5)


def test_fp8_sm70_serialized_moe_raises_clear_error(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    config = _make_fp8_config()
    layer = FusedMoE(
        num_experts=1,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        prefix="model.layers.0.moe",
    )

    with pytest.raises(ValueError, match="sm70.*MoE.*not supported"):
        config.get_quant_method(layer, "model.layers.0.moe")
