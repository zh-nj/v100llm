# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the byte-budgeted sparse-prefill gather workspace sizing
(#5: bound prefill activation peak).

The DeepSeek-V4 sparse prefill gathers compressed+SWA KV into a
(reqs, M, head_dim) bf16 workspace where M ~= seq_len/compress_ratio +
window + max_num_batched_tokens. At long context M is huge, so the number of
requests sharing one workspace is reduced to keep the live buffer under
VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB. These tests pin that arithmetic.
"""
import pytest

import vllm.envs as envs
from vllm.model_executor.layers import deepseek_v4_attention as m

pytestmark = pytest.mark.cpu_test

HEAD_DIM = 576


def _ws_bytes(reqs, M, head_dim=HEAD_DIM):
    return reqs * M * head_dim * 2  # bf16


def test_small_M_keeps_full_chunk(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB", 512)
    # 4 reqs * 2000 * 576 * 2 = ~18 MB << 512 MB budget
    assert m._sparse_prefill_reqs_per_subchunk(M=2000, head_dim=HEAD_DIM) == (
        m.PREFILL_CHUNK_SIZE
    )


def test_large_M_shrinks_to_one_request(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB", 512)
    # 1M ctx C4A: N=250k -> M~254k. 4 reqs would be ~1.1 GB > 512 MB.
    reqs = m._sparse_prefill_reqs_per_subchunk(M=254000, head_dim=HEAD_DIM)
    assert reqs == 1
    assert _ws_bytes(reqs, 254000) <= 512 * 1024 * 1024


def test_budget_zero_disables_bound(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB", 0)
    # 0 = legacy: always PREFILL_CHUNK_SIZE regardless of M.
    assert m._sparse_prefill_reqs_per_subchunk(M=10_000_000, head_dim=HEAD_DIM) == (
        m.PREFILL_CHUNK_SIZE
    )


def test_intermediate_budget_picks_largest_fitting(monkeypatch):
    # Budget that fits exactly 2 requests at M, not 3.
    M = 60000
    two_req = _ws_bytes(2, M)
    monkeypatch.setattr(
        envs,
        "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB",
        # budget just below 3 reqs, at/above 2 reqs
        (two_req + _ws_bytes(1, M) // 2) // (1024 * 1024),
    )
    reqs = m._sparse_prefill_reqs_per_subchunk(M=M, head_dim=HEAD_DIM)
    assert reqs == min(2, m.PREFILL_CHUNK_SIZE)
    assert _ws_bytes(reqs, M) <= envs.VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB * (
        1024 * 1024
    )


def test_never_returns_zero(monkeypatch):
    # Even an absurdly tiny budget must still admit one request.
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB", 1)
    assert m._sparse_prefill_reqs_per_subchunk(M=10_000_000, head_dim=HEAD_DIM) == 1


def test_never_exceeds_prefill_chunk_size(monkeypatch):
    # Huge budget must not exceed the static PREFILL_CHUNK_SIZE.
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SPARSE_PREFILL_TEMP_MB", 1_000_000)
    assert m._sparse_prefill_reqs_per_subchunk(M=64, head_dim=HEAD_DIM) == (
        m.PREFILL_CHUNK_SIZE
    )
