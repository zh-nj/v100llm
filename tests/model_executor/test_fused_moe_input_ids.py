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


@pytest.mark.parametrize(
    ("use_early_shared_stream", "expected_events"),
    [
        (False, ["select", "quant_apply", "legacy_shared"]),
        (True, ["shared_start", "select", "quant_apply", "shared_wait"]),
    ],
)
def test_default_moe_runner_streamed_shared_experts_order(
    use_early_shared_stream: bool,
    expected_events: list[str],
) -> None:
    """Early shared experts launch is opt-in; default keeps legacy ordering."""
    events: list[str] = []

    class RecordingRouter:
        def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            *,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del router_logits, input_ids
            events.append("select")
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
            events.append("quant_apply")
            return x

    runner = object.__new__(DefaultMoERunner)
    runner.router = RecordingRouter()
    runner.quant_method = RecordingQuantMethod()
    runner.shared_experts = object()
    runner.use_shared_experts_stream = True
    runner.use_early_shared_experts_stream = use_early_shared_stream

    hidden_states = torch.randn((3, 4), dtype=torch.float32)
    router_logits = torch.randn((3, 8), dtype=torch.float32)
    shared_input = torch.randn((3, 4), dtype=torch.float32)
    shared_output = torch.randn((3, 4), dtype=torch.float32)

    def start_shared_experts(hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states is shared_input
        events.append("shared_start")
        return shared_output

    def wait_shared_experts_stream() -> None:
        events.append("shared_wait")

    def legacy_apply_shared_experts(
        hidden_states: torch.Tensor,
        allow_streaming: bool = False,
    ) -> torch.Tensor:
        del hidden_states, allow_streaming
        events.append("legacy_shared")
        return shared_output

    runner._start_shared_experts_stream = start_shared_experts
    runner._wait_shared_experts_stream = wait_shared_experts_stream
    runner._apply_shared_experts = legacy_apply_shared_experts

    shared, output = DefaultMoERunner._apply_quant_method(
        runner,
        layer=object(),
        hidden_states=hidden_states,
        router_logits=router_logits,
        shared_input=shared_input,
        run_shared_experts_before=False,
    )

    assert shared is shared_output
    assert output is hidden_states
    assert events == expected_events
