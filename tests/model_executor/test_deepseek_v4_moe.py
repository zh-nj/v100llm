# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.deepseek_v4 import DeepseekV4MoE

pytestmark = pytest.mark.cpu_test


def test_deepseek_v4_fused_moe_combines_shared_expert_tuple() -> None:
    class TupleExperts:
        is_internal_router = True

        def __init__(self) -> None:
            self.input_ids: torch.Tensor | None = None

        def __call__(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del router_logits
            self.input_ids = input_ids
            shared_output = hidden_states + 1
            routed_output = hidden_states + 2
            return shared_output, routed_output

    moe = object.__new__(DeepseekV4MoE)
    moe.experts = TupleExperts()
    moe.shared_experts = object()

    hidden_states = torch.ones((2, 3), dtype=torch.float32)
    input_ids = torch.tensor([17, 19], dtype=torch.int32)

    output = DeepseekV4MoE._forward_fused_moe(
        moe,
        hidden_states,
        input_ids=input_ids,
    )

    assert moe.experts.input_ids is input_ids
    assert torch.equal(output, hidden_states * 2 + 3)
