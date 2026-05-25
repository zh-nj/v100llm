# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SWARingSnapshotData
from vllm.v1.core.sched.scheduler import _swa_snapshot_hashes_for_token_range


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
