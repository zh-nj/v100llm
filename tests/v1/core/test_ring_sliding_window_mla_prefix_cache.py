# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for RingSlidingWindowMLAManager.find_longest_cache_hit
prefix-cache opt-in (P6 task 41b).

Run with:
  pytest -xvs tests/v1/core/test_ring_sliding_window_mla_prefix_cache.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm import envs
from vllm.v1.attention.backends.mla.swa_ring_snapshot_pool import (
    SWARingSnapshotIndex,
    SWARingSnapshotPool,
    SWARingSnapshotPoolRegistry,
)
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    RingSlidingWindowMLAManager,
)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec


@pytest.fixture(autouse=True)
def reset_registry():
    """Each test starts with a fresh registry and index."""
    SWARingSnapshotPoolRegistry._instance = None
    SWARingSnapshotIndex._instance = None
    yield
    SWARingSnapshotPoolRegistry._instance = None
    SWARingSnapshotIndex._instance = None


def _make_pool(layer_prefix="layer.0", max_blocks=64) -> SWARingSnapshotPool:
    return SWARingSnapshotPool(
        layer_prefix=layer_prefix,
        block_size_tokens=64,
        per_token_bytes=584,
        device=torch.device("cpu"),
        dtype=torch.uint8,
        max_bytes=max_blocks * 64 * 584,
    )


def _make_block(fill: int = 0) -> torch.Tensor:
    return torch.full(
        (64, 584), fill_value=fill, dtype=torch.uint8, device="cpu"
    )


def _mark_available(*hashes: BlockHash, expected_ranks: int = 1) -> None:
    """Helper: ack each hash from rank 0 then promote with expected=1."""
    index = SWARingSnapshotIndex.get()
    for h in hashes:
        index.add_ack(bytes(h), rank=0)
        index.mark_available_if_ack_complete(bytes(h),
                                             expected_ranks=expected_ranks)


def _make_spec(sliding_window: int = 256) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=sliding_window,
        cache_dtype_str="fp8_ds_mla",
        compress_ratio=1,
        model_version="deepseek_v4",
        alignment=576,
    )


def _make_block_pool() -> MagicMock:
    """Mock BlockPool with a sentinel null_block."""
    bp = MagicMock()
    null = KVCacheBlock(block_id=-1)
    bp.null_block = null
    return bp


def _hashes(n: int) -> list[BlockHash]:
    return [BlockHash(f"hash_{i:04d}".encode()) for i in range(n)]


def test_disabled_returns_empty(monkeypatch):
    """Default behavior: env var off → always returns []."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", False)
    spec = _make_spec(sliding_window=256)
    bp = _make_block_pool()
    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=_hashes(10),
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert result == ([],)


def test_enabled_no_pools_returns_empty(monkeypatch):
    """Env on but no hashes registered → still empty."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)
    bp = _make_block_pool()
    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=_hashes(10),
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert result == ([],)


def test_enabled_full_match_returns_window_blocks(monkeypatch):
    """Index holds all blocks → match found."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # 4 blocks (each 64 tokens)
    bp = _make_block_pool()

    hashes = _hashes(10)  # 10 blocks
    _mark_available(*hashes)

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,  # 10 * 64 tokens
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert len(result) == 1
    blocks = result[0]
    assert len(blocks) == 10


def test_enabled_partial_match_at_end_returns_empty(monkeypatch):
    """Index only holds last 2 blocks; right-to-left scan resets on
    the miss at index 7. After loop num_contiguous_blocks = 0 →
    no usable prefix, returns empty."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # needs 4 contiguous
    bp = _make_block_pool()

    hashes = _hashes(10)
    # Register only last 2 hashes.
    _mark_available(*hashes[8:10])

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert len(result[0]) == 0


def test_enabled_partial_prefix_short_of_window(monkeypatch):
    """Index only holds first 2 blocks; partial leading prefix
    returns 2 blocks (less than sliding_window contiguous)."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # needs 4 contiguous
    bp = _make_block_pool()

    hashes = _hashes(10)
    _mark_available(*hashes[0:2])

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert len(result[0]) == 2


def test_enabled_no_match_returns_empty(monkeypatch):
    """Index empty → no match."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)
    bp = _make_block_pool()

    hashes = _hashes(10)
    # No hashes registered.

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert result == ([],)


def test_enabled_holes_break_contiguous_run(monkeypatch):
    """Hash at index 7 missing splits the contiguous window."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # 4 contiguous
    bp = _make_block_pool()

    hashes = _hashes(10)
    # Register all except index 7.
    _mark_available(*[h for i, h in enumerate(hashes) if i != 7])

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert len(result[0]) == 7


def test_enabled_with_multiple_groups(monkeypatch):
    """Multiple cache_group_ids share the same hits."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)
    bp = _make_block_pool()

    hashes = _hashes(10)
    _mark_available(*hashes)

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1, 2, 3],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert len(result) == 3
    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 10
