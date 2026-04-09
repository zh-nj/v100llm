# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization import awq_sm70_moe
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_sm70_moe import AWQSM70MoEMethod
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method


def _make_moe_config(
    *,
    hidden_size: int = 256,
    intermediate_size_per_partition: int = 256,
    tp_size: int = 8,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        num_local_experts=2,
        activation="silu",
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=tp_size,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            use_ep=False,
            all2all_backend="naive",
            is_sequence_parallel=False,
            enable_eplb=False,
        ),
        in_dtype=torch.float16,
    )


def _make_moe_placeholder(**kwargs) -> FusedMoE:
    layer = object.__new__(FusedMoE)
    layer.moe_config = _make_moe_config(**kwargs)
    return layer


def _make_awq_config() -> AWQConfig:
    return AWQConfig(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        modules_to_not_convert=[
            "self_attn",
            "block_sparse_moe.gate",
            "model.layers.0.",
        ],
    )


def _make_loader_test_layer(method: AWQSM70MoEMethod) -> FusedMoE:
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_config = _make_moe_config(
        hidden_size=256,
        intermediate_size_per_partition=192,
        tp_size=8,
    )
    layer.moe_parallel_config = layer.moe_config.moe_parallel_config
    layer.quant_config = None
    layer.quant_method = method
    layer._expert_map = None
    layer._map_global_expert_id_to_local_expert_id = lambda expert_id: expert_id
    return layer


def _make_process_test_layer(method: AWQSM70MoEMethod) -> FusedMoE:
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_config = _make_moe_config(
        hidden_size=256,
        intermediate_size_per_partition=256,
        tp_size=8,
    )
    layer.moe_parallel_config = layer.moe_config.moe_parallel_config
    layer.quant_config = None
    layer.quant_method = method
    layer._expert_map = None
    layer._map_global_expert_id_to_local_expert_id = lambda expert_id: expert_id
    return layer


def test_awq_fused_moe_skips_layer_zero_on_sm70(
    monkeypatch,
    default_vllm_config,
) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_placeholder()

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.0.mlp.experts",
    )

    assert isinstance(quant_method, UnquantizedFusedMoEMethod)


def test_awq_fused_moe_uses_sm70_method_for_tp8_shard(monkeypatch) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_placeholder(intermediate_size_per_partition=192)

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.1.mlp.experts",
    )

    assert isinstance(quant_method, AWQSM70MoEMethod)
    assert quant_method.group_size == 64
    assert quant_method.checkpoint_group_size == 128


def test_awq_sm70_moe_repeats_checkpoint_group_scales() -> None:
    method = AWQSM70MoEMethod(
        weight_bits=4,
        group_size=64,
        checkpoint_group_size=128,
        zero_point=True,
        moe=_make_moe_config(intermediate_size_per_partition=192),
    )
    layer = _make_loader_test_layer(method)
    layer.quant_method = method

    method.create_weights(
        layer=layer,
        num_experts=1,
        hidden_size=256,
        intermediate_size_per_partition=192,
        params_dtype=torch.float16,
        weight_loader=FusedMoE.weight_loader.__get__(layer, FusedMoE),
    )

    loaded_weight = (
        torch.arange(12, dtype=torch.float16).view(12, 1).expand(12, 256).clone()
    )
    success = layer.w2_scales.weight_loader(
        layer.w2_scales,
        loaded_weight,
        "w2_scales",
        "w2",
        0,
        return_success=True,
    )

    assert success is True
    expected_groups = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float16)
    assert torch.equal(layer.w2_scales[0, :, 0], expected_groups)


def test_awq_fused_moe_falls_back_when_tp8_shard_cannot_be_adapted(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_placeholder(intermediate_size_per_partition=200)

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.1.mlp.experts",
    )

    assert isinstance(quant_method, MoeWNA16Method)


def test_awq_sm70_moe_prepares_experts_without_torch_stack(
    monkeypatch,
) -> None:
    method = AWQSM70MoEMethod(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        moe=_make_moe_config(),
    )
    layer = _make_process_test_layer(method)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=256,
        params_dtype=torch.float16,
    )

    for expert_idx in range(layer.w13_qweight.shape[0]):
        layer.w13_qweight.data[expert_idx].fill_(expert_idx + 1)
        layer.w13_scales.data[expert_idx].fill_(expert_idx + 11)
        layer.w13_qzeros.data[expert_idx].fill_(expert_idx + 21)
        layer.w2_qweight.data[expert_idx].fill_(expert_idx + 31)
        layer.w2_scales.data[expert_idx].fill_(expert_idx + 41)
        layer.w2_qzeros.data[expert_idx].fill_(expert_idx + 51)

    def fake_prepare(qweight, scales, qzeros, group_size, interleave_gated_silu=False):
        marker = int(qweight.flatten()[0].item())
        width = 4 if interleave_gated_silu else 2
        weight = torch.full(
            (2, width),
            marker,
            dtype=torch.int32,
            device=qweight.device,
        )
        scale = torch.full(
            (1, width * 8),
            marker + 100,
            dtype=scales.dtype,
            device=scales.device,
        )
        meta = torch.tensor(
            [width, width + 1],
            dtype=torch.int32,
            device=qweight.device,
        )
        return weight, scale, meta

    def fake_build_strided_ptrs(weight, scales, k_ld, q_ld, num_experts):
        ptr_rows = torch.zeros(
            (num_experts, 8),
            dtype=torch.uint8,
            device=weight.device,
        )
        return ptr_rows, ptr_rows.clone()

    monkeypatch.setattr(awq_sm70_moe.ops, "awq_sm70_prepare", fake_prepare)
    monkeypatch.setattr(
        awq_sm70_moe.ops,
        "awq_moe_build_strided_ptrs",
        fake_build_strided_ptrs,
    )

    def fail_stack(*args, **kwargs):
        raise AssertionError(
            "process_weights_after_loading should not call torch.stack"
        )

    monkeypatch.setattr(torch, "stack", fail_stack)
    method.process_weights_after_loading(layer)

    assert torch.equal(
        layer.w13_tm_weight[0],
        torch.full((2, 4), 1, dtype=torch.int32),
    )
    assert torch.equal(
        layer.w13_tm_weight[1],
        torch.full((2, 4), 2, dtype=torch.int32),
    )
    assert torch.equal(
        layer.w13_tm_scales[0],
        torch.full((1, 32), 101, dtype=torch.float16),
    )
    assert torch.equal(
        layer.w2_tm_weight[0],
        torch.full((2, 2), 31, dtype=torch.int32),
    )
    assert layer.w13_meta_list == [(4, 5), (4, 5)]
    assert layer.w2_meta_list == [(2, 3), (2, 3)]


def test_awq_sm70_moe_releases_w13_sources_before_w2_prepare(
    monkeypatch,
) -> None:
    method = AWQSM70MoEMethod(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        moe=_make_moe_config(),
    )
    layer = _make_process_test_layer(method)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=256,
        params_dtype=torch.float16,
    )

    def fake_prepare(qweight, scales, qzeros, group_size, interleave_gated_silu=False):
        if not interleave_gated_silu:
            assert not hasattr(layer, "w13_qweight")
            assert not hasattr(layer, "w13_scales")
            assert not hasattr(layer, "w13_qzeros")
        width = 4 if interleave_gated_silu else 2
        weight = torch.zeros(
            (2, width),
            dtype=torch.int32,
            device=qweight.device,
        )
        scale = torch.zeros(
            (1, width * 8),
            dtype=scales.dtype,
            device=scales.device,
        )
        meta = torch.tensor(
            [width, width + 1],
            dtype=torch.int32,
            device=qweight.device,
        )
        return weight, scale, meta

    def fake_build_strided_ptrs(weight, scales, k_ld, q_ld, num_experts):
        ptr_rows = torch.zeros(
            (num_experts, 8),
            dtype=torch.uint8,
            device=weight.device,
        )
        return ptr_rows, ptr_rows.clone()

    monkeypatch.setattr(awq_sm70_moe.ops, "awq_sm70_prepare", fake_prepare)
    monkeypatch.setattr(
        awq_sm70_moe.ops,
        "awq_moe_build_strided_ptrs",
        fake_build_strided_ptrs,
    )

    method.process_weights_after_loading(layer)
