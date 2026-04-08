# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
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
