# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import pytest


def _make_hc_head_inputs(tokens: int = 5, hc_mult: int = 4, hidden: int = 8):
    torch.manual_seed(0)
    hidden_states = torch.randn(tokens, hc_mult, hidden, dtype=torch.float16)
    hc_fn = torch.randn(hc_mult, hc_mult * hidden, dtype=torch.float32)
    hc_scale = torch.randn(1, dtype=torch.float32)
    hc_base = torch.randn(hc_mult, dtype=torch.float32)
    return hidden_states, hc_fn, hc_scale, hc_base


def test_hc_head_rows_per_chunk_caps_fp32_temp(monkeypatch):
    from vllm.model_executor.models import deepseek_v4

    monkeypatch.setenv("VLLM_SM70_HC_HEAD_CHUNK_MB", "128")

    assert deepseek_v4._hc_head_rows_per_chunk(16_384) == 2048


def test_hc_head_chunked_impl_matches_full_impl():
    from vllm.model_executor.models import deepseek_v4

    hidden_states, hc_fn, hc_scale, hc_base = _make_hc_head_inputs()

    expected = deepseek_v4._hc_head_eager_impl(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=1e-6,
        hc_eps=1e-4,
    )
    actual = deepseek_v4._hc_head_chunked_impl(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=1e-6,
        hc_eps=1e-4,
        chunk_rows=2,
        chunk_kernel=deepseek_v4._hc_head_eager_impl,
    )

    assert actual.dtype == hidden_states.dtype
    assert actual.shape == (hidden_states.shape[0], hidden_states.shape[-1])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_hc_head_chunked_impl_slices_rows():
    from vllm.model_executor.models import deepseek_v4

    hidden_states, hc_fn, hc_scale, hc_base = _make_hc_head_inputs()
    calls: list[int] = []

    def chunk_kernel(*args, **kwargs):
        calls.append(args[0].shape[0])
        return deepseek_v4._hc_head_eager_impl(*args, **kwargs)

    deepseek_v4._hc_head_chunked_impl(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=1e-6,
        hc_eps=1e-4,
        chunk_rows=2,
        chunk_kernel=chunk_kernel,
    )

    assert calls == [2, 2, 1]


def test_hc_head_cuda_sm70_custom_op_matches_full_impl(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 7:
        pytest.skip("SM70 custom-op route is only enabled on Volta")

    from vllm.model_executor.models import deepseek_v4

    monkeypatch.setenv("VLLM_SM70_HC_HEAD_CHUNK_MB", "1")
    hidden_states, hc_fn, hc_scale, hc_base = _make_hc_head_inputs(
        tokens=17, hc_mult=4, hidden=4096
    )
    hidden_states = hidden_states.cuda()
    hc_fn = hc_fn.cuda()
    hc_scale = hc_scale.cuda()
    hc_base = hc_base.cuda()

    expected = deepseek_v4._hc_head_compiled_impl(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=1e-6,
        hc_eps=1e-4,
    )
    actual = deepseek_v4.hc_head(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=1e-6,
        hc_eps=1e-4,
    )

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=1e-3)
