# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.activation import (
    activation_without_mul as activation_without_mul_impl,
)
from vllm.model_executor.layers.fused_moe.activation import (
    apply_moe_activation as apply_moe_activation_impl,
)
from vllm.model_executor.layers.fused_moe.utils import activation_without_mul
from vllm.model_executor.layers.fused_moe.utils import apply_moe_activation


def test_utils_exports_activation_compat_helpers() -> None:
    assert activation_without_mul is activation_without_mul_impl
    assert apply_moe_activation is apply_moe_activation_impl
