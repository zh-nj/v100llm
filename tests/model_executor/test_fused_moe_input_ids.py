# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
    DefaultMoERunner,
)

pytestmark = pytest.mark.cpu_test


def test_fused_moe_forward_cuda_accepts_input_ids() -> None:
    class RecordingRunner:
        def __init__(self) -> None:
            self.input_ids: torch.Tensor | None = None

        def forward(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.input_ids = input_ids
            return hidden_states + router_logits

    layer = object.__new__(FusedMoE)
    layer.runner = RecordingRunner()

    hidden_states = torch.ones((2, 4), dtype=torch.float32)
    router_logits = torch.full((2, 4), 2.0, dtype=torch.float32)
    input_ids = torch.tensor([101, 102], dtype=torch.long)

    output = FusedMoE.forward_cuda(
        layer,
        hidden_states,
        router_logits,
        input_ids=input_ids,
    )

    assert layer.runner.input_ids is input_ids
    assert torch.equal(output, hidden_states + router_logits)


def test_default_moe_runner_passes_input_ids_to_router() -> None:
    class RecordingRouter:
        def __init__(self) -> None:
            self.input_ids: torch.Tensor | None = None

        def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            *,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del router_logits
            self.input_ids = input_ids
            topk_weights = torch.ones((hidden_states.size(0), 1))
            topk_ids = torch.zeros((hidden_states.size(0), 1), dtype=torch.int64)
            return topk_weights, topk_ids

    class RecordingQuantMethod:
        is_monolithic = False
        mk_owns_shared_expert = False

        def apply(
            self,
            layer: object,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            del layer, topk_weights, topk_ids, shared_experts_input
            return x

    runner = object.__new__(DefaultMoERunner)
    runner.router = RecordingRouter()
    runner.quant_method = RecordingQuantMethod()
    runner.shared_experts = None
    runner.use_shared_experts_stream = False

    hidden_states = torch.randn((3, 4), dtype=torch.float32)
    router_logits = torch.randn((3, 8), dtype=torch.float32)
    input_ids = torch.tensor([11, 12, 13], dtype=torch.long)

    _, output = DefaultMoERunner._apply_quant_method(
        runner,
        layer=object(),
        hidden_states=hidden_states,
        router_logits=router_logits,
        shared_input=None,
        input_ids=input_ids,
    )

    assert runner.router.input_ids is input_ids
    assert output is hidden_states
