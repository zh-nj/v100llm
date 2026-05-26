# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SWARingSnapshotData
from vllm.v1.core.sched.scheduler import (
    _deepseek_v4_prefill_budget_cap,
    _swa_snapshot_hashes_for_token_range,
)


def _hashes(n: int) -> list[BlockHash]:
    return [BlockHash(bytes([i]) * 32) for i in range(n)]


def test_swa_snapshot_hashes_select_newly_completed_blocks():
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=_hashes(4),
        start_tokens=65,
        end_tokens=128,
        hash_block_size=64,
        swa_block_size=64,
    )

    assert data == SWARingSnapshotData(
        start_block_idx=1,
        block_hashes=[bytes([1]) * 32],
    )


def test_swa_snapshot_hashes_select_cached_prefix_from_zero():
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=_hashes(5),
        start_tokens=0,
        end_tokens=192,
        hash_block_size=64,
        swa_block_size=64,
    )

    assert data == SWARingSnapshotData(
        start_block_idx=0,
        block_hashes=[bytes([0]) * 32, bytes([1]) * 32, bytes([2]) * 32],
    )


def test_swa_snapshot_hashes_can_limit_cached_prefix_to_tail_blocks():
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=_hashes(8),
        start_tokens=0,
        end_tokens=512,
        hash_block_size=64,
        swa_block_size=64,
        max_blocks=2,
    )

    assert data == SWARingSnapshotData(
        start_block_idx=6,
        block_hashes=[bytes([6]) * 32, bytes([7]) * 32],
    )


def test_swa_snapshot_hashes_tail_limit_keeps_short_prefix_intact():
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=_hashes(2),
        start_tokens=0,
        end_tokens=128,
        hash_block_size=64,
        swa_block_size=64,
        max_blocks=4,
    )

    assert data == SWARingSnapshotData(
        start_block_idx=0,
        block_hashes=[bytes([0]) * 32, bytes([1]) * 32],
    )


def test_swa_snapshot_hashes_combine_finer_hash_granularity():
    base = _hashes(8)
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=base,
        start_tokens=0,
        end_tokens=128,
        hash_block_size=16,
        swa_block_size=64,
    )

    assert data == SWARingSnapshotData(
        start_block_idx=0,
        block_hashes=[
            b"".join(base[:4]),
            b"".join(base[4:8]),
        ],
    )


def test_swa_snapshot_hashes_returns_none_when_no_full_block_completed():
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=_hashes(4),
        start_tokens=64,
        end_tokens=127,
        hash_block_size=64,
        swa_block_size=64,
    )

    assert data is None


def test_swa_snapshot_hashes_returns_none_for_coarser_hash_granularity():
    data = _swa_snapshot_hashes_for_token_range(
        block_hashes=_hashes(2),
        start_tokens=0,
        end_tokens=128,
        hash_block_size=128,
        swa_block_size=64,
    )

    assert data is None


def test_deepseek_v4_prefill_budget_cap_disabled(monkeypatch):
    monkeypatch.setattr(
        "vllm.envs.VLLM_DEEPSEEK_V4_PREFILL_BUDGET_CAP_AFTER_TOKENS",
        0,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.envs.VLLM_DEEPSEEK_V4_PREFILL_BUDGET_CAP_TOKENS",
        0,
        raising=False,
    )

    assert _deepseek_v4_prefill_budget_cap(32768, 4096) == 4096


def test_deepseek_v4_prefill_budget_cap_after_threshold(monkeypatch):
    monkeypatch.setattr(
        "vllm.envs.VLLM_DEEPSEEK_V4_PREFILL_BUDGET_CAP_AFTER_TOKENS",
        32768,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.envs.VLLM_DEEPSEEK_V4_PREFILL_BUDGET_CAP_TOKENS",
        2048,
        raising=False,
    )

    assert _deepseek_v4_prefill_budget_cap(32768, 4096) == 2048
    assert _deepseek_v4_prefill_budget_cap(49152, 4096) == 2048


def test_deepseek_v4_prefill_budget_cap_crossing_threshold(monkeypatch):
    monkeypatch.setattr(
        "vllm.envs.VLLM_DEEPSEEK_V4_PREFILL_BUDGET_CAP_AFTER_TOKENS",
        32768,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.envs.VLLM_DEEPSEEK_V4_PREFILL_BUDGET_CAP_TOKENS",
        2048,
        raising=False,
    )

    assert _deepseek_v4_prefill_budget_cap(28672, 4096) == 4096
    assert _deepseek_v4_prefill_budget_cap(30720, 4096) == 2048
