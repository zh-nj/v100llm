# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest

if not torch.cuda.is_available():
    pytest.skip("CUDA is required to import DeepSeek V4 CUDA quant helpers.",
                allow_module_level=True)

from vllm.model_executor.models import deepseek_v4


def test_sm70_hyperconnection_output_clamp_keeps_fp16_values_finite():
    out = torch.tensor(
        [-float("inf"), -65504.0, 0.0, 65504.0, float("inf"), float("nan")],
        dtype=torch.float16,
    )

    clamped = deepseek_v4._clamp_sm70_fp16_hc_output_(out)

    assert clamped is out
    expected = torch.tensor(
        [-65504.0, -65504.0, 0.0, 65504.0, 65504.0],
        dtype=torch.float16,
    )
    torch.testing.assert_close(out[:-1], expected)
    assert torch.isnan(out[-1])
