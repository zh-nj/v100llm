# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers import utils

pytestmark = pytest.mark.cpu_test


def test_router_gemm_falls_back_when_custom_op_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(utils.ops, "router_gemm_bf16_fp32", raising=False)

    x = torch.randn((3, 5), dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn((7, 5), dtype=torch.float32).to(torch.bfloat16)

    output = utils.cublas_gemm_bf16_bf16_fp32(x, weight)
    expected = x.float() @ weight.float().t()

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected)


def test_router_gemm_falls_back_for_non_bf16_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("bf16-only router GEMM custom op should not run")

    monkeypatch.setattr(utils.ops, "router_gemm_bf16_fp32", fail_if_called, raising=False)

    x = torch.randn((3, 5), dtype=torch.float16)
    weight = torch.randn((7, 5), dtype=torch.float16)

    output = utils.cublas_gemm_bf16_bf16_fp32(x, weight)
    expected = x.float() @ weight.float().t()

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected)
