# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config


def _patch_single_rank_params(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter_module = importlib.import_module("vllm.model_executor.parameter")
    fused_moe_layer_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.layer"
    )
    fused_moe_config_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.config"
    )
    distributed_module = importlib.import_module("vllm.distributed")
    parallel_state_module = importlib.import_module("vllm.distributed.parallel_state")
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
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
    monkeypatch.setattr(
        parallel_state_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        parallel_state_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        fused_moe_config_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )


def _patch_sm70_mxfp4_available(monkeypatch: pytest.MonkeyPatch) -> None:
    mxfp4_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.mxfp4"
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    cpp_ops = SimpleNamespace(
        sm70_mxfp4_moe_direct_prepare=lambda *args, **kwargs: None,
        sm70_mxfp4_moe_gemm_out=lambda *args, **kwargs: None,
        sm70_moe_add_bias_out=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mxfp4_module,
        "_get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    monkeypatch.setattr(torch.ops, "_C", cpp_ops, raising=False)
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_direct_prepare",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_gemm_out",
        lambda *args, **kwargs: None,
        raising=False,
    )


def test_mxfp4_min_capability_allows_sm70() -> None:
    assert Mxfp4Config.get_min_capability() == 70


def test_mxfp4_supported_act_dtypes_include_sm70_fp16() -> None:
    assert torch.float16 in Mxfp4Config.get_supported_act_dtypes()


def test_mxfp4_moe_weight_scales_are_block_scales_for_loader(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    mxfp4_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.mxfp4"
    )
    monkeypatch.setattr(
        mxfp4_module,
        "_is_sm70_mxfp4_moe_available",
        lambda: False,
        raising=False,
    )

    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=Mxfp4Config(),
        prefix="model.layers.0.mlp.experts",
        activation="swigluoai",
        has_bias=False,
    )

    assert layer.quant_method.__class__.__name__ == "Mxfp4MoEMethod"
    assert layer.w13_weight_scale.quant_method == "block"
    assert layer.w2_weight_scale.quant_method == "block"


def test_mxfp4_sm70_availability_requires_registered_cpp_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mxfp4_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.mxfp4"
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    monkeypatch.setattr(
        mxfp4_module,
        "_get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_direct_prepare",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_gemm_out",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(torch.ops, "_C", SimpleNamespace(), raising=False)

    assert mxfp4_module._is_sm70_mxfp4_moe_available() is False

    cpp_ops_without_bias = SimpleNamespace(
        sm70_mxfp4_moe_direct_prepare=lambda *args, **kwargs: None,
        sm70_mxfp4_moe_gemm_out=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(torch.ops, "_C", cpp_ops_without_bias, raising=False)

    assert mxfp4_module._is_sm70_mxfp4_moe_available() is False


def test_mxfp4_sm70_oracle_selects_turbomind_when_cpp_ops_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4"
    )
    cpp_ops = SimpleNamespace(
        sm70_mxfp4_moe_direct_prepare=lambda *args, **kwargs: None,
        sm70_mxfp4_moe_gemm_out=lambda *args, **kwargs: None,
        sm70_moe_add_bias_out=lambda *args, **kwargs: None,
    )
    capability = SimpleNamespace(to_int=lambda: 70)
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        get_device_capability=lambda: capability,
    )
    monkeypatch.setattr(torch.ops, "_C", cpp_ops, raising=False)
    monkeypatch.setattr(oracle_module, "current_platform", platform)

    backend, kernel_cls = oracle_module.select_mxfp4_moe_backend(object())

    assert backend == oracle_module.Mxfp4MoeBackend.SM70_TURBOMIND
    assert kernel_cls is None


def test_mxfp4_sm70_uses_turbomind_direct_moe_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    _patch_sm70_mxfp4_available(monkeypatch)

    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.bfloat16,
        quant_config=Mxfp4Config(),
        prefix="model.layers.0.mlp.experts",
        activation="swigluoai",
        has_bias=True,
    )

    assert layer.quant_method.__class__.__name__ == "Mxfp4SM70MoEMethod"
    assert tuple(layer.w13_weight.shape) == (2, 256, 128)
    assert tuple(layer.w13_weight_scale.shape) == (2, 256, 8)
    assert tuple(layer.w2_weight.shape) == (2, 256, 64)
    assert tuple(layer.w2_weight_scale.shape) == (2, 256, 4)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w2_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.quant_method == "block"
    assert layer.w2_weight_scale.quant_method == "block"


def test_deepseek_v4_fp8_config_uses_sm70_turbomind_direct_moe_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    _patch_sm70_mxfp4_available(monkeypatch)
    deepseek_v4_module = importlib.import_module(
        "vllm.model_executor.models.deepseek_v4"
    )

    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=deepseek_v4_module.DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        ),
        prefix="model.layers.0.mlp.experts",
        activation="swigluoai",
        has_bias=True,
    )

    assert layer.quant_method.__class__.__name__ == "Mxfp4SM70MoEMethod"


def test_mxfp4_sm70_rounds_intermediate_partition_for_tp4_gpt_oss(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    _patch_sm70_mxfp4_available(monkeypatch)

    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=2880,
        intermediate_size=2880,
        tp_size=4,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.float16,
        quant_config=Mxfp4Config(),
        prefix="model.layers.0.mlp.experts",
        activation="swigluoai",
        has_bias=True,
    )

    assert layer.moe_config.intermediate_size_per_partition == 736
    assert tuple(layer.w13_weight.shape) == (2, 1472, 1440)
    assert tuple(layer.w13_weight_scale.shape) == (2, 1472, 90)
    assert tuple(layer.w2_weight.shape) == (2, 2880, 368)
    assert tuple(layer.w2_weight_scale.shape) == (2, 2880, 23)


def test_mxfp4_sm70_process_prepares_turbomind_weights(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_single_rank_params(monkeypatch)
    _patch_sm70_mxfp4_available(monkeypatch)
    ops_module = importlib.import_module("vllm._custom_ops")

    layer = FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=256,
        intermediate_size=128,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        params_dtype=torch.bfloat16,
        quant_config=Mxfp4Config(),
        prefix="model.layers.0.moe",
        activation="swigluoai",
        has_bias=True,
    )
    layer.w13_weight.data.zero_()
    layer.w2_weight.data.zero_()
    layer.w13_weight_scale.data.fill_(127)
    layer.w2_weight_scale.data.fill_(127)

    prepare_calls = []

    def fake_prepare(weight, scale, interleave_gated_silu):
        prepare_calls.append(
            (tuple(weight.shape), tuple(scale.shape), interleave_gated_silu)
        )
        logical_n = weight.size(1)
        logical_k = weight.size(2) * 2
        prepared_weight = torch.empty_like(weight)
        prepared_scale = torch.empty(
            (weight.size(0), logical_k // 32, logical_n),
            dtype=torch.uint8,
            device=weight.device,
        )
        prepared_meta = torch.tensor(
            [logical_n, logical_k, 32, 4096, logical_n],
            dtype=torch.int64,
            device=weight.device,
        )
        return [prepared_weight, prepared_scale, prepared_meta]

    def fake_ptrs(weight, scale, k_ld, q_ld, num_experts):
        del k_ld, q_ld
        return [
            torch.empty((num_experts * 16,), dtype=torch.uint8, device=weight.device),
            torch.empty((num_experts * 16,), dtype=torch.uint8, device=scale.device),
        ]

    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_direct_prepare",
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
        ((2, 256, 128), (2, 256, 8), False),
        ((2, 256, 64), (2, 256, 4), False),
    ]
    assert layer._sm70_mxfp4_moe_direct_prepared is True
    assert layer.sm70_batched_ready is True
    assert layer.w13_tm_weight.dtype == torch.uint8
    assert layer.w2_tm_weight.dtype == torch.uint8
    assert layer.w13_tm_scales.dtype == torch.uint8
    assert layer.w2_tm_scales.dtype == torch.uint8
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w13_weight_scale")
    assert not hasattr(layer, "w2_weight_scale")


def test_mxfp4_sm70_apply_uses_unfused_swigluoai_activation(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del default_vllm_config
    sm70_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.sm70_mxfp4_moe"
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    method_cls = getattr(sm70_module, "Mxfp4SM70MoEMethod")
    dummy_moe_config = type(
        "MoeCfg",
        (),
        {"experts_per_token": 1, "activation": "swigluoai", "has_bias": False},
    )()
    method = method_cls(dummy_moe_config)

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
    activation_calls = []

    def fake_mxfp4_moe_gemm_out(
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
        out.fill_(3 if k == 256 else 5)

    def fake_activation(activation, output, input):
        activation_calls.append((activation.value, tuple(output.shape), tuple(input.shape)))
        output.fill_(7)

    monkeypatch.setattr(torch.ops._moe_C, "moe_permute", fake_permute)
    monkeypatch.setattr(torch.ops._moe_C, "moe_unpermute", fake_unpermute)
    monkeypatch.setattr(
        sm70_module,
        "_moe_permute_accepts_scale_and_m_indices",
        lambda: True,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_gemm_out",
        fake_mxfp4_moe_gemm_out,
        raising=False,
    )
    monkeypatch.setattr(sm70_module, "apply_moe_activation", fake_activation)

    x = torch.ones(2, 256, dtype=torch.float16)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    out = method.apply(layer, x, topk_weights, topk_ids, None)

    assert gemm_calls == [
        ((2, 256), 256, 256, 32, False),
        ((2, 256), 128, 256, 32, False),
    ]
    assert activation_calls == [("swigluoai", (2, 128), (2, 256))]
    assert tuple(out.shape) == (2, 256)
    assert torch.all(out == 5)


def test_mxfp4_sm70_apply_honors_deepseek_v4_swiglu_limit(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del default_vllm_config
    sm70_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.sm70_mxfp4_moe"
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    method_cls = getattr(sm70_module, "Mxfp4SM70MoEMethod")
    dummy_moe_config = type(
        "MoeCfg",
        (),
        {"experts_per_token": 1, "activation": "silu", "has_bias": False},
    )()
    method = method_cls(dummy_moe_config)

    layer = torch.nn.Module()
    layer.sm70_batched_ready = True
    layer.sm70_num_experts = 2
    layer.sm70_hidden_logical_size = 256
    layer.sm70_w13_k_dim = 256
    layer.sm70_w13_n_dim = 256
    layer.sm70_w2_k_dim = 128
    layer.sm70_w2_n_dim = 256
    layer.sm70_intermediate_size = 128
    layer.swiglu_limit = 10.0
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

    events = []

    def fake_mxfp4_moe_gemm_out(
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
        events.append(("gemm", tuple(out.shape), k, n, group_size, gated_silu))
        out.fill_(3 if k == 256 else 5)

    def fake_swiglu_limit(output, input, swiglu_limit):
        events.append(("swiglu_limit", tuple(output.shape), tuple(input.shape),
                       swiglu_limit))
        output.fill_(7)

    monkeypatch.setattr(torch.ops._moe_C, "moe_permute", fake_permute)
    monkeypatch.setattr(torch.ops._moe_C, "moe_unpermute", fake_unpermute)
    monkeypatch.setattr(
        sm70_module,
        "_moe_permute_accepts_scale_and_m_indices",
        lambda: True,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_gemm_out",
        fake_mxfp4_moe_gemm_out,
        raising=False,
    )
    monkeypatch.setattr(
        sm70_module,
        "swiglu_limit_func",
        fake_swiglu_limit,
        raising=False,
    )

    x = torch.ones(2, 256, dtype=torch.float16)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    out = method.apply(layer, x, topk_weights, topk_ids, None)

    assert events == [
        ("gemm", (2, 256), 256, 256, 32, False),
        ("swiglu_limit", (2, 128), (2, 256), 10.0),
        ("gemm", (2, 256), 128, 256, 32, False),
    ]
    assert tuple(out.shape) == (2, 256)
    assert torch.all(out == 5)


def test_mxfp4_sm70_apply_adds_expert_biases(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del default_vllm_config
    sm70_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.sm70_mxfp4_moe"
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    method_cls = getattr(sm70_module, "Mxfp4SM70MoEMethod")
    dummy_moe_config = type(
        "MoeCfg",
        (),
        {"experts_per_token": 1, "activation": "swigluoai", "has_bias": True},
    )()
    method = method_cls(dummy_moe_config)

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
    layer.w13_bias = torch.nn.Parameter(
        torch.ones(2, 256, dtype=torch.float16), requires_grad=False
    )
    layer.w2_bias = torch.nn.Parameter(
        torch.ones(2, 256, dtype=torch.float16), requires_grad=False
    )

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
        expert_offsets64.copy_(torch.tensor([0, 1, x.size(0)]))
        inv_permuted_idx.zero_()

    def fake_unpermute(sorted_output, topk_weights, inv_idx, offsets, top_k, output):
        del topk_weights, inv_idx, offsets, top_k
        output.copy_(sorted_output[: output.size(0)])

    events = []

    def fake_mxfp4_moe_gemm_out(
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
        events.append(("gemm", tuple(out.shape), k, n, group_size, gated_silu))
        out.fill_(3 if k == 256 else 5)

    def fake_add_bias(out, expert_offsets, bias, num_experts):
        events.append(("bias", tuple(out.shape), tuple(bias.shape), num_experts))
        assert expert_offsets.dtype == torch.int32
        out.add_(1)

    def fake_activation(activation, output, input):
        events.append(("activation", activation.value, tuple(output.shape), tuple(input.shape)))
        output.fill_(7)

    monkeypatch.setattr(torch.ops._moe_C, "moe_permute", fake_permute)
    monkeypatch.setattr(torch.ops._moe_C, "moe_unpermute", fake_unpermute)
    monkeypatch.setattr(
        sm70_module,
        "_moe_permute_accepts_scale_and_m_indices",
        lambda: True,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_mxfp4_moe_gemm_out",
        fake_mxfp4_moe_gemm_out,
        raising=False,
    )
    monkeypatch.setattr(
        ops_module,
        "sm70_moe_add_bias_out",
        fake_add_bias,
        raising=False,
    )
    monkeypatch.setattr(sm70_module, "apply_moe_activation", fake_activation)

    x = torch.ones(2, 256, dtype=torch.float16)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    out = method.apply(layer, x, topk_weights, topk_ids, None)

    assert events == [
        ("gemm", (2, 256), 256, 256, 32, False),
        ("bias", (2, 256), (2, 256), 2),
        ("activation", "swigluoai", (2, 128), (2, 256)),
        ("gemm", (2, 256), 128, 256, 32, False),
        ("bias", (2, 256), (2, 256), 2),
    ]
    assert tuple(out.shape) == (2, 256)
    assert torch.all(out == 6)


def test_mxfp4_sm70_cuda_gemm_matches_unit_scale_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SM70 MXFP4 GEMM smoke test.")
    if torch.cuda.get_device_capability(0) != (7, 0):
        pytest.skip("SM70 MXFP4 GEMM smoke test requires an SM70 CUDA device.")
    c_ops = getattr(torch.ops, "_C", None)
    if (
        c_ops is None
        or not hasattr(c_ops, "sm70_mxfp4_moe_direct_prepare")
        or not hasattr(c_ops, "sm70_mxfp4_moe_gemm_out")
    ):
        pytest.skip("SM70 MXFP4 custom ops are not registered.")

    from vllm import _custom_ops as ops

    num_experts, num_tokens, k_dim, n_dim = 1, 16, 32, 64
    weight = torch.full(
        (num_experts, n_dim, k_dim // 2),
        0x22,
        dtype=torch.uint8,
        device="cuda",
    )
    scale = torch.full(
        (num_experts, n_dim, k_dim // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    prepared_weight, prepared_scale, meta = ops.sm70_mxfp4_moe_direct_prepare(
        weight,
        scale,
        False,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_weight,
        prepared_scale,
        int(meta[3].item()),
        int(meta[4].item()),
        num_experts,
    )
    x = torch.ones(num_tokens, k_dim, dtype=torch.float16, device="cuda")
    out = torch.empty(num_tokens, n_dim, dtype=torch.float16, device="cuda")
    expert_offsets = torch.tensor(
        [0, num_tokens],
        dtype=torch.int32,
        device="cuda",
    )

    ops.sm70_mxfp4_moe_gemm_out(
        out,
        x,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k_dim,
        n_dim,
        32,
        False,
    )

    torch.testing.assert_close(out, torch.full_like(out, float(k_dim)))


@pytest.mark.parametrize(
    "packed_value,scale_byte,expected",
    [
        (0x44, 128, 128.0),
        (0x55, 126, 48.0),
        (0x99, 128, -32.0),
    ],
)
def test_mxfp4_sm70_cuda_gemm_applies_non_unit_scales_and_signed_nibbles(
    packed_value: int,
    scale_byte: int,
    expected: float,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SM70 MXFP4 GEMM smoke test.")
    if torch.cuda.get_device_capability(0) != (7, 0):
        pytest.skip("SM70 MXFP4 GEMM smoke test requires an SM70 CUDA device.")
    c_ops = getattr(torch.ops, "_C", None)
    if (
        c_ops is None
        or not hasattr(c_ops, "sm70_mxfp4_moe_direct_prepare")
        or not hasattr(c_ops, "sm70_mxfp4_moe_gemm_out")
    ):
        pytest.skip("SM70 MXFP4 custom ops are not registered.")

    from vllm import _custom_ops as ops

    num_experts, num_tokens, k_dim, n_dim = 1, 4, 32, 64
    weight = torch.full(
        (num_experts, n_dim, k_dim // 2),
        packed_value,
        dtype=torch.uint8,
        device="cuda",
    )
    scale = torch.full(
        (num_experts, n_dim, k_dim // 32),
        scale_byte,
        dtype=torch.uint8,
        device="cuda",
    )
    prepared_weight, prepared_scale, meta = ops.sm70_mxfp4_moe_direct_prepare(
        weight,
        scale,
        False,
    )
    ptrs_w, ptrs_s = ops.awq_moe_build_strided_ptrs(
        prepared_weight,
        prepared_scale,
        int(meta[3].item()),
        int(meta[4].item()),
        num_experts,
    )
    x = torch.ones(num_tokens, k_dim, dtype=torch.float16, device="cuda")
    out = torch.empty(num_tokens, n_dim, dtype=torch.float16, device="cuda")
    expert_offsets = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")

    ops.sm70_mxfp4_moe_gemm_out(
        out,
        x,
        expert_offsets,
        ptrs_w,
        ptrs_s,
        num_experts,
        k_dim,
        n_dim,
        32,
        False,
    )

    torch.testing.assert_close(out, torch.full_like(out, expected))


def test_sm70_moe_add_bias_out_cuda_adds_bias_by_expert_offsets() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SM70 MoE bias add smoke test.")
    c_ops = getattr(torch.ops, "_C", None)
    if c_ops is None or not hasattr(c_ops, "sm70_moe_add_bias_out"):
        pytest.skip("SM70 MoE bias add custom op is not registered.")

    from vllm import _custom_ops as ops

    out = torch.zeros(5, 4, dtype=torch.float16, device="cuda")
    expert_offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    bias = torch.tensor(
        [[1, 2, 3, 4], [10, 20, 30, 40]],
        dtype=torch.bfloat16,
        device="cuda",
    )

    ops.sm70_moe_add_bias_out(out, expert_offsets, bias, 2)

    expected = torch.tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
        ],
        dtype=torch.float16,
        device="cuda",
    )
    torch.testing.assert_close(out, expected)
