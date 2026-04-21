# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.parameter import BlockQuantScaleParameter


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


def test_fp8_sm70_process_keeps_fp8_resident_and_records_runtime_meta(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = _make_linear(monkeypatch, output_size=384)
    _populate_fp8_block_weights(layer)
    ops_module = importlib.import_module("vllm._custom_ops")

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        lambda weight, scale, *_args: [
            weight,
            scale,
            torch.tensor(
                [384, 256, 384, 256, 128, 2, -1, 128, 128],
                dtype=torch.int64,
            ),
            torch.tensor([128, 256, 384, 256, 4], dtype=torch.int64),
        ],
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.numel() > 0
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer._sm70_fp8_runtime_prepared is True
    assert layer._sm70_fp8_panel_n == 128
    assert layer._sm70_fp8_block_shape == (128, 128)
    assert layer._sm70_fp8_output_size == 384
    assert layer._sm70_fp8_logical_widths == (384,)
    assert layer._sm70_fp8_prepared_weight is layer.weight
    assert layer._sm70_fp8_prepared_scale is layer.weight_scale_inv
    assert tuple(layer._sm70_fp8_prepared_meta.tolist()) == (
        384,
        256,
        384,
        256,
        128,
        2,
        -1,
        128,
        128,
    )
    assert tuple(layer._sm70_fp8_workspace_meta.tolist()) == (
        128,
        256,
        384,
        256,
        4,
    )
    assert not hasattr(layer, "_awq_sm70_prepared")


def test_fp8_sm70_merged_linear_process_records_logical_widths(
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
    ops_module = importlib.import_module("vllm._custom_ops")

    layer = MergedColumnParallelLinear(
        input_size=256,
        output_sizes=[128, 128, 64],
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_fp8_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )
    _populate_fp8_block_weights(layer)

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        lambda weight, scale, *_args: [
            weight,
            scale,
            torch.tensor(
                [384, 256, 384, 256, 128, 2, -1, 128, 128],
                dtype=torch.int64,
            ),
            torch.tensor([128, 256, 384, 256, 4], dtype=torch.int64),
        ],
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer._sm70_fp8_output_size == 320
    assert layer._sm70_fp8_logical_widths == (128, 128, 64)


def test_fp8_sm70_process_calls_prepare_op_and_stores_prepared_tensors(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    layer = _make_linear(monkeypatch, output_size=384)
    _populate_fp8_block_weights(layer)
    called = {}

    def fake_prepare(
        weight,
        weight_scale,
        layout_kind,
        scale_axis,
        block_n,
        block_k,
        panel_n,
    ):
        called["args"] = (
            tuple(weight.shape),
            tuple(weight_scale.shape),
            layout_kind,
            scale_axis,
            block_n,
            block_k,
            panel_n,
        )
        prepared_meta = torch.tensor(
            [384, 256, 384, 256, 128, layout_kind, scale_axis, block_n, block_k],
            dtype=torch.int64,
        )
        workspace_meta = torch.tensor([128, 256, 384, 256, 4], dtype=torch.int64)
        return [weight, weight_scale, prepared_meta, workspace_meta]

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        fake_prepare,
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert called["args"] == ((384, 256), (3, 2), 2, -1, 128, 128, 128)
    assert layer._sm70_fp8_prepared_weight is layer.weight
    assert layer._sm70_fp8_prepared_scale is layer.weight_scale_inv
    assert tuple(layer._sm70_fp8_prepared_meta.tolist()) == (
        384,
        256,
        384,
        256,
        128,
        2,
        -1,
        128,
        128,
    )
    assert tuple(layer._sm70_fp8_workspace_meta.tolist()) == (
        128,
        256,
        384,
        256,
        4,
    )


def test_fp8_sm70_apply_calls_runtime_decode_custom_op(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = layer.weight_scale_inv
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [384, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64
    )
    layer._sm70_fp8_workspace_meta = torch.tensor(
        [128, 256, 384, 256, 4], dtype=torch.int64
    )
    layer._sm70_fp8_output_size = 384
    layer._sm70_fp8_logical_widths = (384,)

    called = {}

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
        called["shape"] = (
            tuple(out.shape),
            tuple(x.shape),
            tuple(prepared_weight.shape),
        )
        called["scale_shape"] = tuple(prepared_scale.shape)
        called["prepared_meta"] = tuple(prepared_meta.tolist())
        called["workspace_shapes"] = (
            tuple(decoded_panel.shape),
            tuple(packed_panel.shape),
            tuple(meta_buffer.shape),
        )
        out.copy_(torch.full_like(out, 7))

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert called["shape"] == ((2, 384), (2, 256), (384, 256))
    assert called["scale_shape"] == (3, 2)
    assert called["prepared_meta"] == (384, 256, 384, 256, 128, 2, -1, 128, 128)
    assert called["workspace_shapes"] == ((128, 256), (384, 256), (4,))
    assert torch.all(out == 7)


def test_fp8_sm70_apply_reuses_workspace_and_slices_logical_width(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.fp8"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = layer.weight_scale_inv
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [384, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64
    )
    layer._sm70_fp8_workspace_meta = torch.tensor(
        [128, 256, 384, 256, 4], dtype=torch.int64
    )
    layer._sm70_fp8_output_size = 320
    layer._sm70_fp8_logical_widths = (128, 128, 64)

    seen_workspace_ptrs = []

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
        seen_workspace_ptrs.append(
            (
                decoded_panel.data_ptr(),
                packed_panel.data_ptr(),
                meta_buffer.data_ptr(),
            )
        )
        out.copy_(
            torch.arange(
                out.numel(),
                dtype=out.dtype,
                device=out.device,
            ).reshape_as(out)
        )

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    out1 = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)
    out2 = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert tuple(out1.shape) == (2, 320)
    assert tuple(out2.shape) == (2, 320)
    assert seen_workspace_ptrs[0] == seen_workspace_ptrs[1]
    assert torch.equal(
        out1,
        torch.arange(2 * 384, dtype=torch.float16).reshape(2, 384)[:, :320],
    )


def test_fp8_sm70_apply_slices_runtime_decode_output_to_logical_width(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.fp8"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = layer.weight_scale_inv
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [384, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64
    )
    layer._sm70_fp8_workspace_meta = torch.tensor(
        [128, 256, 384, 256, 4], dtype=torch.int64
    )
    layer._sm70_fp8_output_size = 320
    layer._sm70_fp8_logical_widths = (128, 128, 64)

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
        out.copy_(
            torch.arange(
                out.numel(),
                dtype=out.dtype,
                device=out.device,
            ).reshape_as(out)
        )

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert tuple(out.shape) == (2, 320)
    assert torch.equal(
        out,
        torch.arange(2 * 384, dtype=torch.float16).reshape(2, 384)[:, :320],
    )


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


def test_qwen35_linear_attn_tuple_shard_adjusts_block_scale_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    qwen35_module = importlib.import_module("vllm.model_executor.models.qwen3_5")
    utils_module = importlib.import_module("vllm.model_executor.models.utils")

    monkeypatch.setattr(
        utils_module,
        "is_pp_missing_parameter",
        lambda name, model: False,
    )

    class DummyOwner:
        def __init__(self) -> None:
            self.output_sizes = [2048, 2048, 2048, 2048]
            self.weight_block_size = (128, 128)
            self.loaded: list[tuple[int, tuple[int, ...]]] = []

        def weight_loader(
            self,
            param: BlockQuantScaleParameter,
            loaded_weight: torch.Tensor,
            shard_id: int,
        ) -> None:
            self.loaded.append((shard_id, tuple(loaded_weight.shape)))

    class DummyModel:
        def __init__(self, owner: DummyOwner, param: BlockQuantScaleParameter):
            self._owner = owner
            self._param = param
            self._extra_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def named_parameters(self):
            return iter([
                (
                    "model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv",
                    self._param,
                ),
                (
                    "model.layers.0.linear_attn.in_proj_ba.weight",
                    self._extra_param,
                ),
            ])

        def get_expert_mapping(self):
            return []

    owner = DummyOwner()
    param = BlockQuantScaleParameter(
        data=torch.empty((64, 8), dtype=torch.float32),
        input_dim=1,
        output_dim=0,
        weight_loader=owner.weight_loader,
    )
    model = DummyModel(owner, param)

    qwen35_module.Qwen3_5Model.load_weights(
        model,
        [
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv",
                torch.ones((48, 8), dtype=torch.float32),
            )
        ],
    )

    assert owner.loaded == [
        (0, (16, 8)),
        (1, (16, 8)),
        (2, (16, 8)),
    ]
