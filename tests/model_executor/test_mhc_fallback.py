# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch

from vllm.platforms.interface import DeviceCapability


def _reference_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size).float()
    residual_vec = residual_flat.reshape(-1, hc_mult * hidden_size)
    mixes = residual_vec @ fn.t()
    rms = torch.rsqrt(
        residual_vec.square().sum(dim=-1, keepdim=True)
        / (hc_mult * hidden_size)
        + rms_eps
    )
    mixes = mixes * rms

    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    pre = pre + hc_pre_eps
    post = torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    post = post * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(-1, hc_mult, hc_mult)
    comb_logits = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :].reshape(
        hc_mult, hc_mult
    )
    comb = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.einsum("nh,nhd->nd", pre, residual_flat).to(residual.dtype)
    return (
        post.reshape(*outer_shape, hc_mult, 1),
        comb.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def test_mhc_torch_fallback_imports_without_tilelang_and_accepts_fp16() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MHC SM70 fallback smoke test.")
    if importlib.util.find_spec("tilelang") is not None:
        pytest.skip("This test covers the no-tilelang fallback path.")

    from vllm.model_executor.layers import mhc

    torch.manual_seed(0)
    residual = torch.randn(3, 4, 8, device="cuda", dtype=torch.float16)
    fn = torch.randn(24, 32, device="cuda", dtype=torch.float32)
    hc_scale = torch.tensor([0.7, 0.5, 0.3], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(24, device="cuda", dtype=torch.float32)

    actual_post, actual_comb, actual_layer_input = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-5,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=2,
    )
    expected_post, expected_comb, expected_layer_input = _reference_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-5,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=2,
    )

    torch.testing.assert_close(actual_post, expected_post)
    torch.testing.assert_close(actual_comb, expected_comb)
    torch.testing.assert_close(actual_layer_input, expected_layer_input)

    actual_out = mhc.mhc_post(
        actual_layer_input,
        residual,
        actual_post,
        actual_comb,
    )
    expected_out = (
        torch.einsum("nio,nih->noh", actual_comb, residual.float())
        + actual_post.squeeze(-1).unsqueeze(-1) * actual_layer_input.float().unsqueeze(-2)
    ).to(residual.dtype)

    torch.testing.assert_close(actual_out, expected_out)


def test_mhc_sm70_fast_path_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor.layers import mhc

    monkeypatch.setattr(mhc.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        mhc.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(7, 0),
    )
    monkeypatch.setenv("VLLM_SM70_MHC_FAST", "0")

    assert not mhc._is_sm70_fast_path_available()


def test_mhc_sm70_fast_path_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers import mhc

    monkeypatch.setattr(mhc.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        mhc.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(7, 0),
    )
    monkeypatch.delenv("VLLM_SM70_MHC_FAST", raising=False)

    assert not mhc._is_sm70_fast_path_available()

    monkeypatch.setenv("VLLM_SM70_MHC_FAST", "1")

    assert mhc._is_sm70_fast_path_available()
