# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode import (
    select_sm70_fp8_panel_n,
)
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


def _reset_sm70_fp8_workspace_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode"
    )
    monkeypatch.setattr(
        workspace_module,
        "_SM70_FP8_SHARED_WORKSPACE_CACHE",
        {},
        raising=False,
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
        "sm70_fp8_direct_prepare",
        lambda weight, scale, block_n, block_k: [
            weight,
            torch.empty((2, 384), dtype=torch.float16),
            torch.tensor([384, 256, block_k, 8192, 384], dtype=torch.int64),
        ],
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.numel() > 0
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer._sm70_fp8_runtime_prepared is True
    assert layer._sm70_fp8_direct_prepared is True
    assert layer._sm70_fp8_block_shape == (128, 128)
    assert layer._sm70_fp8_output_size == 384
    assert layer._sm70_fp8_logical_widths == (384,)
    assert layer._sm70_fp8_prepared_weight is layer.weight
    assert layer._sm70_fp8_prepared_scale.shape == (2, 384)
    assert tuple(layer._sm70_fp8_prepared_meta.tolist()) == (384, 256, 128, 8192, 384)
    assert not hasattr(layer, "_sm70_fp8_workspace_meta")
    assert not hasattr(layer, "_awq_sm70_prepared")


def test_fp8_sm70_bmm_weight_keeps_raw_layout_for_einsum(
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
    raw_weight = layer.weight.detach().clone()
    raw_scale = layer.weight_scale_inv.detach().clone()
    layer.is_bmm = True
    layer.bmm_batch_size = 3

    ops_module = importlib.import_module("vllm._custom_ops")

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        lambda weight, scale, *args: [
            weight,
            scale,
            torch.tensor([384, 256, 384, 256, 128, 2, -1, 128, 128]),
            torch.tensor([128, 256, 384, 256, 4]),
        ],
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_direct_prepare",
        lambda weight, scale, block_n, block_k: [
            torch.empty_like(weight),
            torch.empty((2, 384), dtype=torch.float16),
            torch.tensor([384, 256, block_k, 8192, 384], dtype=torch.int64),
        ],
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer._sm70_fp8_direct_prepared is False
    assert torch.equal(layer.weight, raw_weight)
    assert torch.equal(layer.weight_scale_inv, raw_scale)
    assert layer._sm70_fp8_prepared_weight is layer.weight
    assert layer._sm70_fp8_prepared_scale is layer.weight_scale_inv


def test_select_sm70_fp8_panel_n_uses_wider_panels_for_decode_hot_shapes() -> None:
    assert select_sm70_fp8_panel_n(logical_n=384, logical_k=256, block_n=128) == 128
    assert select_sm70_fp8_panel_n(logical_n=1024, logical_k=3584, block_n=128) == 1024
    assert select_sm70_fp8_panel_n(logical_n=8224, logical_k=1024, block_n=128) == 1024
    assert select_sm70_fp8_panel_n(logical_n=5120, logical_k=17408, block_n=128) == 896


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
        "sm70_fp8_direct_prepare",
        lambda weight, scale, block_n, block_k: [
            weight,
            torch.empty((2, 384), dtype=torch.float16),
            torch.tensor([384, 256, block_k, 8192, 384], dtype=torch.int64),
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
    monkeypatch.setenv("VLLM_SM70_FP8_DIRECT_GEMM", "0")
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
    assert layer._sm70_fp8_direct_prepared is False
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


def test_fp8_sm70_process_uses_adaptive_panel_for_wide_layer(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    monkeypatch.setenv("VLLM_SM70_FP8_DIRECT_GEMM", "0")
    ops_module = importlib.import_module("vllm._custom_ops")
    layer = _make_linear(monkeypatch, input_size=1024, output_size=3072)
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
        called["panel_n"] = panel_n
        prepared_meta = torch.tensor(
            [3072, 1024, 3072, 1024, panel_n, layout_kind, scale_axis, block_n, block_k],
            dtype=torch.int64,
        )
        workspace_meta = torch.tensor(
            [panel_n, 1024, panel_n, 1024, 4], dtype=torch.int64
        )
        return [weight, weight_scale, prepared_meta, workspace_meta]

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        fake_prepare,
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert called["panel_n"] == 1024
    assert layer._sm70_fp8_panel_n == 1024


def test_fp8_sm70_process_calls_direct_prepare_by_default(
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
    prepared_scale = torch.empty((2, 384), dtype=torch.float16)

    def fake_direct_prepare(weight, weight_scale, block_n, block_k):
        called["args"] = (
            tuple(weight.shape),
            tuple(weight_scale.shape),
            block_n,
            block_k,
        )
        return [
            weight,
            prepared_scale,
            torch.tensor([384, 256, block_k, 8192, 384], dtype=torch.int64),
        ]

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_direct_prepare",
        fake_direct_prepare,
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert called["args"] == ((384, 256), (3, 2), 128, 128)
    assert layer._sm70_fp8_direct_prepared is True
    assert layer._sm70_fp8_prepared_weight is layer.weight
    assert layer._sm70_fp8_prepared_scale is prepared_scale
    assert tuple(layer._sm70_fp8_prepared_meta.tolist()) == (384, 256, 128, 8192, 384)
    assert not hasattr(layer, "_sm70_fp8_workspace_meta")


def test_fp8_sm70_apply_calls_runtime_decode_custom_op(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_sm70_fp8_workspace_cache(monkeypatch)
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_direct_prepared = False
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


def test_fp8_sm70_apply_calls_direct_gemm_custom_op(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_sm70_fp8_workspace_cache(monkeypatch)
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_direct_prepared = True
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = torch.empty((2, 384), dtype=torch.float16)
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [384, 256, 128, 8192, 384], dtype=torch.int64
    )
    layer._sm70_fp8_output_size = 320
    layer._sm70_fp8_logical_widths = (128, 128, 64)

    called = {}

    def fake_direct_gemm_out(out, x, prepared_weight, prepared_scale, prepared_meta):
        called["shape"] = (
            tuple(out.shape),
            tuple(x.shape),
            tuple(prepared_weight.shape),
            tuple(prepared_scale.shape),
            tuple(prepared_meta.tolist()),
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
        "sm70_fp8_direct_gemm_out",
        fake_direct_gemm_out,
        raising=False,
    )

    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert called["shape"] == (
        (2, 384),
        (2, 256),
        (384, 256),
        (2, 384),
        (384, 256, 128, 8192, 384),
    )
    assert tuple(out.shape) == (2, 320)
    assert torch.equal(
        out,
        torch.arange(2 * 384, dtype=torch.float16).reshape(2, 384)[:, :320],
    )


def test_fp8_sm70_apply_reuses_workspace_and_slices_logical_width(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_sm70_fp8_workspace_cache(monkeypatch)
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


def test_fp8_sm70_workspace_cache_is_shared_across_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_sm70_fp8_workspace_cache(monkeypatch)
    workspace_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode"
    )

    small_layer = torch.nn.Module()
    small_layer._sm70_fp8_workspace_meta = torch.tensor(
        [128, 256, 128, 256, 4], dtype=torch.int64
    )
    big_layer = torch.nn.Module()
    big_layer._sm70_fp8_workspace_meta = torch.tensor(
        [512, 256, 512, 256, 4], dtype=torch.int64
    )
    x = torch.empty((1, 256), dtype=torch.float16)

    small_workspace = workspace_module.ensure_sm70_fp8_workspace(
        small_layer, x.device
    )
    big_workspace = workspace_module.ensure_sm70_fp8_workspace(big_layer, x.device)
    small_workspace_after_growth = workspace_module.get_or_create_sm70_fp8_workspace(
        small_layer,
        x,
    )

    assert tuple(small_workspace.decoded_panel.shape) == (128, 256)
    assert tuple(big_workspace.decoded_panel.shape) == (512, 256)
    assert (
        small_workspace_after_growth.decoded_panel.data_ptr()
        == big_workspace.decoded_panel.data_ptr()
    )
    assert (
        small_workspace_after_growth.packed_panel.data_ptr()
        == big_workspace.packed_panel.data_ptr()
    )


def test_fp8_sm70_apply_does_not_reallocate_workspace_for_larger_batch(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_sm70_fp8_workspace_cache(monkeypatch)
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
    layer._sm70_fp8_output_size = 384
    layer._sm70_fp8_logical_widths = (384,)

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
        out.zero_()

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)
    method.apply(layer, torch.ones(4, 256, dtype=torch.float16), None)

    assert seen_workspace_ptrs[0] == seen_workspace_ptrs[1]


def test_fp8_sm70_apply_slices_runtime_decode_output_to_logical_width(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_sm70_fp8_workspace_cache(monkeypatch)
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


def test_fp8_sm70_serialized_moe_uses_direct_moe_method(
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
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="model.layers.0.mlp.experts",
    )

    assert layer.quant_method.__class__.__name__ == "Fp8SM70DirectMoEMethod"
    assert tuple(layer.w13_weight.shape) == (2, 256, 256)
    assert tuple(layer.w2_weight.shape) == (2, 256, 128)
    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_weight.dtype == torch.float8_e4m3fn
    assert tuple(layer.w13_weight_scale_inv.shape) == (2, 2, 2)
    assert tuple(layer.w2_weight_scale_inv.shape) == (2, 2, 1)


def test_fp8_sm70_moe_process_prepares_fp8_resident_weights(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    config = _make_fp8_config()
    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=config,
        prefix="model.layers.0.moe",
    )
    layer.w13_weight.data.zero_()
    layer.w2_weight.data.zero_()
    layer.w13_weight_scale_inv.data.fill_(1)
    layer.w2_weight_scale_inv.data.fill_(1)

    prepare_calls = []

    def fake_prepare(weight, scale, block_n, block_k, interleave_gated_silu):
        prepare_calls.append(
            (
                tuple(weight.shape),
                tuple(scale.shape),
                block_n,
                block_k,
                interleave_gated_silu,
            )
        )
        prepared_weight = torch.empty_like(weight)
        prepared_scale = torch.empty(
            (weight.size(0), weight.size(2) // block_k, weight.size(1)),
            dtype=torch.float16,
            device=weight.device,
        )
        prepared_meta = torch.tensor(
            [weight.size(1), weight.size(2), block_k, 4096, weight.size(1)],
            dtype=torch.int64,
            device=weight.device,
        )
        return [prepared_weight, prepared_scale, prepared_meta]

    def fake_ptrs(weight, scale, k_ld, q_ld, num_experts):
        return [
            torch.empty((num_experts * 16,), dtype=torch.uint8, device=weight.device),
            torch.empty((num_experts * 16,), dtype=torch.uint8, device=weight.device),
        ]

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_moe_direct_prepare",
        fake_prepare,
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "awq_moe_build_strided_ptrs",
        fake_ptrs,
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert prepare_calls == [
        ((2, 256, 256), (2, 2, 2), 128, 128, True),
        ((2, 256, 128), (2, 2, 1), 128, 128, False),
    ]
    assert layer._sm70_fp8_moe_direct_prepared is True
    assert layer.sm70_batched_ready is True
    assert layer.w13_tm_weight.dtype == torch.float8_e4m3fn
    assert layer.w2_tm_weight.dtype == torch.float8_e4m3fn
    assert layer.w13_tm_scales.dtype == torch.float16
    assert layer.w2_tm_scales.dtype == torch.float16
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert not hasattr(layer, "w2_weight_scale_inv")


def test_fp8_sm70_moe_apply_calls_fp8_moe_gemm(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    ops_module = importlib.import_module("vllm._custom_ops")
    method_cls = getattr(fp8_module, "Fp8SM70DirectMoEMethod")
    dummy_moe_layer = torch.nn.Module()
    dummy_moe_layer.moe_config = type("MoeCfg", (), {"experts_per_token": 1})()
    method = method_cls(_make_fp8_config(), dummy_moe_layer)

    layer = torch.nn.Module()
    layer.sm70_batched_ready = True
    layer.sm70_num_experts = 2
    layer.sm70_hidden_logical_size = 256
    layer.sm70_w13_k_dim = 256
    layer.sm70_w13_n_dim = 256
    layer.sm70_w2_k_dim = 128
    layer.sm70_w2_n_dim = 256
    layer.sm70_intermediate_size = 128
    layer._buf_max_tokens = 32
    layer._buf_max_slots = 32
    layer._buf_top_k = 1
    layer._buf_output = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_permuted_input = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_sorted_output = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_gate_up = torch.empty(32, 256, dtype=torch.float16)
    layer._buf_intermediate = torch.empty(32, 128, dtype=torch.float16)
    layer._buf_expert_offsets = torch.empty(3, dtype=torch.int32)
    layer._buf_expert_offsets64 = torch.empty(3, dtype=torch.int64)
    layer._buf_inv_permuted_idx = torch.empty(32, 1, dtype=torch.int32)
    layer._buf_topk_ids_i32 = torch.empty(32, 1, dtype=torch.int32)
    layer._buf_token_expert_indices = torch.arange(32, dtype=torch.int32).view(32, 1)
    layer._buf_permuted_idx = torch.empty(32, dtype=torch.int32)
    layer._buf_m_indices = torch.empty(32, dtype=torch.int32)
    layer.w13_strided_ptrs_w = torch.empty(32, dtype=torch.uint8)
    layer.w13_strided_ptrs_s = torch.empty(32, dtype=torch.uint8)
    layer.w2_strided_ptrs_w = torch.empty(32, dtype=torch.uint8)
    layer.w2_strided_ptrs_s = torch.empty(32, dtype=torch.uint8)

    def fake_permute(
        x,
        topk_ids,
        token_expert_indices,
        scales,
        num_experts,
        padded_num_experts,
        top_k,
        maybe_unused,
        permuted_input,
        expert_offsets64,
        inv_permuted_idx,
        permuted_idx,
        m_indices,
    ):
        del topk_ids, token_expert_indices, scales, num_experts
        del padded_num_experts, top_k, maybe_unused, permuted_idx, m_indices
        permuted_input[: x.size(0)].copy_(x)
        expert_offsets64.copy_(torch.tensor([0, x.size(0), x.size(0)]))
        inv_permuted_idx.zero_()

    def fake_unpermute(sorted_output, topk_weights, inv_idx, offsets, top_k, output):
        del topk_weights, inv_idx, offsets, top_k
        output.copy_(sorted_output[: output.size(0)])

    gemm_calls = []

    def fake_fp8_moe_gemm_out(
        out,
        sorted_input,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu=False,
    ):
        del sorted_input, expert_offsets, ptrs_w, ptrs_s, num_experts
        gemm_calls.append((tuple(out.shape), k, n, group_size, gated_silu))
        out.fill_(3 if gated_silu else 5)

    monkeypatch.setattr(torch.ops._moe_C, "moe_permute", fake_permute)
    monkeypatch.setattr(torch.ops._moe_C, "moe_unpermute", fake_unpermute)
    monkeypatch.setattr(
        fp8_module,
        "_moe_permute_accepts_scale_and_m_indices",
        lambda: True,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_moe_gemm_out",
        fake_fp8_moe_gemm_out,
        raising=False,
    )

    x = torch.ones(2, 256, dtype=torch.float16)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    out = method.apply(layer, x, topk_weights, topk_ids, None)

    assert gemm_calls == [
        ((2, 128), 256, 256, 128, True),
        ((2, 256), 128, 256, 128, False),
    ]
    assert tuple(out.shape) == (2, 256)
    assert torch.all(out == 5)


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
