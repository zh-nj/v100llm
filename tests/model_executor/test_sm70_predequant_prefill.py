# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SM70 fp8 weight pre-dequant *prefill* path.

The lazy `_sm70_ensure_predequant_weight` helper populates
``b._sm70_predequant_f16`` on first use, but H43/H44/H59/H60 nsys decode
captures showed the underlying dequant kernel still firing every step
(~0.8 ms/step on 8x V100). Root cause: lazy populate happens during
cudagraph capture, so the kernel launch is recorded into the replay
graph and reissued every step.

The fix is `_sm70_prefill_predequant_weight`, which eagerly populates
the cache **before** any cudagraph capture, by calling the underlying
Triton dequant directly (bypassing the shielded custom op).

These tests are CPU-only — they exercise the Python-level cache /
attribute / opt-out logic. Numerical equivalence with the lazy path is
covered by `tests/model_executor/test_sm70_fp8_a_dequant.py` and the
existing exploration suite which run on GPU.
"""

from __future__ import annotations

import os
import types

import pytest
import torch


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for predequant prefill tests",
)


@cuda_required
def test_prefill_populates_attribute_and_skips_when_already_set(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_PREDEQUANT_PREFILL_DISABLE", "0")
    import vllm.envs as envs
    envs.disable_envs_cache()
    assert envs.VLLM_SM70_PREDEQUANT_PREFILL_DISABLE is False

    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    helper = getattr(dsa, "_sm70_prefill_predequant_weight", None)
    assert helper is not None, "_sm70_prefill_predequant_weight must exist"

    # Build a synthetic fp8 wo_a-shaped weight + scale.
    groups, rank, hidden = 2, 128, 128
    fp8_uint8 = torch.zeros(
        (groups, rank, hidden), dtype=torch.uint8, device="cuda"
    )
    scale = torch.ones(
        (groups, rank // 128, hidden // 128), dtype=torch.float32, device="cuda"
    )

    # Simulate: cache empty -> prefill populates it.
    if hasattr(fp8_uint8, "_sm70_predequant_f16"):
        delattr(fp8_uint8, "_sm70_predequant_f16")

    out = helper(fp8_uint8, scale, groups, rank, hidden)
    assert out is not None, "first call must populate the cache"
    assert hasattr(fp8_uint8, "_sm70_predequant_f16")
    cached = fp8_uint8._sm70_predequant_f16
    assert cached.dtype == torch.float16
    assert cached.shape == (groups, rank, hidden)

    # Second call must be a no-op (returns None to signal cache hit).
    out2 = helper(fp8_uint8, scale, groups, rank, hidden)
    assert out2 is None, "second call must short-circuit on cache hit"
    # Underlying tensor identity preserved.
    assert fp8_uint8._sm70_predequant_f16 is cached


@cuda_required
def test_prefill_disabled_via_env(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_PREDEQUANT_PREFILL_DISABLE", "1")
    import vllm.envs as envs
    envs.disable_envs_cache()
    assert envs.VLLM_SM70_PREDEQUANT_PREFILL_DISABLE is True

    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    helper = dsa._sm70_prefill_predequant_weight

    groups, rank, hidden = 2, 128, 128
    fp8_uint8 = torch.zeros(
        (groups, rank, hidden), dtype=torch.uint8, device="cuda"
    )
    scale = torch.ones(
        (groups, rank // 128, hidden // 128), dtype=torch.float32, device="cuda"
    )
    if hasattr(fp8_uint8, "_sm70_predequant_f16"):
        delattr(fp8_uint8, "_sm70_predequant_f16")

    out = helper(fp8_uint8, scale, groups, rank, hidden)
    assert out is None, "env opt-out must skip prefill"
    assert not hasattr(fp8_uint8, "_sm70_predequant_f16"), (
        "opt-out must not populate the attribute"
    )


@cuda_required
def test_prefill_matches_lazy_cache_numerically(monkeypatch):
    """Eager prefill must produce the same fp16 tensor as the lazy
    `_sm70_ensure_predequant_weight` path."""
    monkeypatch.setenv("VLLM_SM70_PREDEQUANT_PREFILL_DISABLE", "0")
    import vllm.envs as envs
    envs.disable_envs_cache()

    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )
    prefill = dsa._sm70_prefill_predequant_weight
    ensure = dsa._sm70_ensure_predequant_weight

    groups, rank, hidden = 2, 128, 128
    torch.manual_seed(20260526)
    # Restrict byte range to avoid fp8e4m3 NaN encodings (0x7F/0xFF).
    raw = torch.randint(
        low=0,
        high=120,
        size=(groups, rank, hidden),
        dtype=torch.uint8,
        device="cuda",
    )
    scale = torch.rand(
        (groups, rank // 128, hidden // 128),
        dtype=torch.float32,
        device="cuda",
    ).add_(0.1)

    # Eager prefill on a copy.
    weight_a = raw.clone()
    if hasattr(weight_a, "_sm70_predequant_f16"):
        delattr(weight_a, "_sm70_predequant_f16")
    prefill(weight_a, scale, groups, rank, hidden)
    assert hasattr(weight_a, "_sm70_predequant_f16")

    # Lazy populate on a separate copy.
    weight_b = raw.clone()
    if hasattr(weight_b, "_sm70_predequant_f16"):
        delattr(weight_b, "_sm70_predequant_f16")
    cached_b = ensure(weight_b, scale, groups, rank, hidden)
    assert cached_b is weight_b._sm70_predequant_f16

    # Same shape / dtype / values.
    cached_a = weight_a._sm70_predequant_f16
    assert cached_a.dtype == cached_b.dtype == torch.float16
    assert cached_a.shape == cached_b.shape
    torch.testing.assert_close(cached_a, cached_b, rtol=0, atol=0)


@cuda_required
def test_lazy_path_is_skipped_when_attribute_already_set(monkeypatch):
    """After prefill, calling ``_sm70_ensure_predequant_weight`` must
    return the cached tensor without invoking the shielded custom op.

    This is the contract that makes the cudagraph fix work: once the
    attribute is set before capture, the lazy wrapper takes the
    fast no-op branch on every replay.
    """
    monkeypatch.setenv("VLLM_SM70_PREDEQUANT_PREFILL_DISABLE", "0")
    import vllm.envs as envs
    envs.disable_envs_cache()

    dsa = pytest.importorskip(
        "vllm.model_executor.layers.deepseek_v4_attention"
    )

    groups, rank, hidden = 2, 128, 128
    torch.manual_seed(20260527)
    # Restrict byte range to avoid fp8e4m3 NaN encodings.
    raw = torch.randint(
        low=0,
        high=120,
        size=(groups, rank, hidden),
        dtype=torch.uint8,
        device="cuda",
    )
    scale = torch.rand(
        (groups, rank // 128, hidden // 128),
        dtype=torch.float32,
        device="cuda",
    ).add_(0.1)

    # Prefill the cache.
    dsa._sm70_prefill_predequant_weight(raw, scale, groups, rank, hidden)
    cached_before = raw._sm70_predequant_f16

    # Replace the shielded op with a tripwire so any call would explode.
    sentinel_calls: list = []

    def _trip(b, b_scale, g, r, h):
        sentinel_calls.append((b.shape, g, r, h))
        return torch.empty(
            (g, r, h), dtype=torch.float16, device=b.device
        )

    monkeypatch.setattr(dsa, "_SM70_FP8_WEIGHT_PREDEQUANT", _trip)

    out = dsa._sm70_ensure_predequant_weight(raw, scale, groups, rank, hidden)
    assert out is cached_before, (
        "ensure helper must reuse the prefilled cache, not call the op"
    )
    assert sentinel_calls == [], (
        "shielded op must not be invoked when the cache is already set"
    )


def test_env_var_registered_and_off_by_default(monkeypatch):
    """The opt-out env must be registered with default 0."""
    monkeypatch.delenv("VLLM_SM70_PREDEQUANT_PREFILL_DISABLE", raising=False)

    import vllm.envs as envs
    envs.disable_envs_cache()
    assert hasattr(envs, "VLLM_SM70_PREDEQUANT_PREFILL_DISABLE")
    assert envs.VLLM_SM70_PREDEQUANT_PREFILL_DISABLE is False


def test_env_var_respects_opt_out(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_PREDEQUANT_PREFILL_DISABLE", "1")
    import vllm.envs as envs

    envs.disable_envs_cache()
    assert envs.VLLM_SM70_PREDEQUANT_PREFILL_DISABLE is True
