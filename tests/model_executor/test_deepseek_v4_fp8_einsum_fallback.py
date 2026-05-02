# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
QUANT_GROUP_SIZE = 128


def _ue8m0_scales(absmax: torch.Tensor) -> torch.Tensor:
    scale_raw = absmax.clamp(min=1e-10) * (1.0 / FP8_MAX)
    return torch.exp2(torch.ceil(torch.log2(scale_raw)))


def _quant_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = x.view(*x.shape[:-1], x.shape[-1] // QUANT_GROUP_SIZE, QUANT_GROUP_SIZE)
    scales = _ue8m0_scales(blocks.abs().amax(dim=-1, keepdim=True))
    x_fp8 = (blocks / scales).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8.reshape_as(x), scales.squeeze(-1)


def _quant_weight(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, d, r = x.shape
    blocks = x.view(
        h,
        d // QUANT_GROUP_SIZE,
        QUANT_GROUP_SIZE,
        r // QUANT_GROUP_SIZE,
        QUANT_GROUP_SIZE,
    )
    blocks = blocks.permute(0, 1, 3, 2, 4)
    scales = _ue8m0_scales(blocks.abs().amax(dim=(-1, -2), keepdim=True))
    x_fp8 = (blocks / scales).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    x_fp8 = x_fp8.permute(0, 1, 3, 2, 4).reshape_as(x)
    return x_fp8, scales.squeeze(-1).squeeze(-1)


def _dequant_act(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return x.float() * scales.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)


def _dequant_weight(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale_expanded = scales.repeat_interleave(QUANT_GROUP_SIZE, dim=1)
    scale_expanded = scale_expanded.repeat_interleave(QUANT_GROUP_SIZE, dim=2)
    return x.float() * scale_expanded


@torch.inference_mode()
def test_deepseek_v4_fp8_einsum_torch_fallback_without_deepgemm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSeek V4 fp8 einsum fallback")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 fallback coverage")

    import vllm.model_executor.layers.deepseek_v4_attention  # noqa: F401

    bsz = 3
    groups = 2
    rank = 128
    hidden = 256
    device = "cuda"
    torch.manual_seed(0)

    a_ref = torch.randn(bsz, groups, hidden, device=device, dtype=torch.float16)
    b_ref = torch.randn(groups, rank, hidden, device=device, dtype=torch.float16)
    a, a_scale = _quant_act(a_ref.float())
    b, b_scale = _quant_weight(b_ref.float())
    out = torch.empty(bsz, groups, rank, device=device, dtype=torch.float16)

    torch.ops.vllm.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b.reshape(groups * rank, hidden),
        b_scale.reshape(groups * (rank // QUANT_GROUP_SIZE), hidden // QUANT_GROUP_SIZE),
        out,
        "bhr,hdr->bhd",
        [1, 128, 128],
    )

    expected = torch.einsum(
        "bhr,hdr->bhd",
        _dequant_act(a, a_scale),
        _dequant_weight(b, b_scale),
    ).to(out.dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_deepseek_v4_o_projection_uses_hidden_state_dtype():
    import inspect

    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    source = inspect.getsource(DeepseekV4MultiHeadLatentAttentionWrapper.forward)

    assert "dtype=hidden_states.dtype" in source
    assert "dtype=torch.bfloat16" not in source
