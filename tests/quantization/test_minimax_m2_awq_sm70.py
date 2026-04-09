# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization import awq_sm70_moe
from vllm.model_executor.layers.quantization.awq_sm70_moe import AWQSM70MoEMethod
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method


pytestmark = pytest.mark.usefixtures("default_vllm_config")


def _make_moe_layer(
    *,
    hidden_size: int = 256,
    intermediate_size: int = 256,
) -> FusedMoE:
    return FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=torch.float16,
        prefix="model.layers.0.mlp.experts",
    )


def _make_awq_config() -> AWQConfig:
    return AWQConfig(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        modules_to_not_convert=["self_attn", "block_sparse_moe.gate", "model.layers.0."],
    )


def test_awq_fused_moe_skips_layer_zero_on_sm70(monkeypatch, dist_init) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_layer()

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.0.mlp.experts",
    )

    assert isinstance(quant_method, UnquantizedFusedMoEMethod)


def test_awq_fused_moe_uses_sm70_method_for_non_skipped_layer(
    monkeypatch, dist_init
) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_layer()

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.1.mlp.experts",
    )

    assert isinstance(quant_method, AWQSM70MoEMethod)


def test_awq_fused_moe_falls_back_when_sm70_shape_is_incompatible(
    monkeypatch, dist_init
) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_layer(intermediate_size=192)

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.1.mlp.experts",
    )

    assert isinstance(quant_method, MoeWNA16Method)


def test_awq_sm70_moe_prepares_experts_without_torch_stack(dist_init, monkeypatch):
    layer = _make_moe_layer()
    method = AWQSM70MoEMethod(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        moe=layer.moe_config,
    )
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
        weight = torch.full((2, width), marker, dtype=torch.int32, device=qweight.device)
        scale = torch.full((1, width * 8), marker + 100, dtype=scales.dtype, device=scales.device)
        meta = torch.tensor([width, width + 1], dtype=torch.int32, device=qweight.device)
        return weight, scale, meta

    def fake_build_strided_ptrs(weight, scales, k_ld, q_ld, num_experts):
        ptr_rows = torch.zeros((num_experts, 8), dtype=torch.uint8, device=weight.device)
        return ptr_rows, ptr_rows.clone()

    monkeypatch.setattr(awq_sm70_moe.ops, "awq_sm70_prepare", fake_prepare)
    monkeypatch.setattr(
        awq_sm70_moe.ops, "awq_moe_build_strided_ptrs", fake_build_strided_ptrs
    )

    def fail_stack(*args, **kwargs):
        raise AssertionError("process_weights_after_loading should not call torch.stack")

    monkeypatch.setattr(torch, "stack", fail_stack)
    method.process_weights_after_loading(layer)

    assert torch.equal(layer.w13_tm_weight[0], torch.full((2, 4), 1, dtype=torch.int32))
    assert torch.equal(layer.w13_tm_weight[1], torch.full((2, 4), 2, dtype=torch.int32))
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
    dist_init, monkeypatch
):
    layer = _make_moe_layer()
    method = AWQSM70MoEMethod(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        moe=layer.moe_config,
    )
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
        weight = torch.zeros((2, width), dtype=torch.int32, device=qweight.device)
        scale = torch.zeros((1, width * 8), dtype=scales.dtype, device=scales.device)
        meta = torch.tensor([width, width + 1], dtype=torch.int32, device=qweight.device)
        return weight, scale, meta

    def fake_build_strided_ptrs(weight, scales, k_ld, q_ld, num_experts):
        ptr_rows = torch.zeros((num_experts, 8), dtype=torch.uint8, device=weight.device)
        return ptr_rows, ptr_rows.clone()

    monkeypatch.setattr(awq_sm70_moe.ops, "awq_sm70_prepare", fake_prepare)
    monkeypatch.setattr(
        awq_sm70_moe.ops, "awq_moe_build_strided_ptrs", fake_build_strided_ptrs
    )

    method.process_weights_after_loading(layer)
