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
    """Each test starts with a fresh registry."""
    SWARingSnapshotPoolRegistry._instance = None
    yield
    SWARingSnapshotPoolRegistry._instance = None


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
    """Env on but no layer registered → still empty."""
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
    """Pool holds last sliding_window blocks → match found."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # 4 blocks (each 64 tokens)
    bp = _make_block_pool()
    pool = _make_pool()
    SWARingSnapshotPoolRegistry.get().register_layer(pool)

    hashes = _hashes(10)  # 10 blocks
    # Register hashes 0..9 (full prefix in pool).
    for h in hashes:
        pool.register(h, _make_block(fill=1))

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,  # 10 * 64 tokens
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    # Should find the last 4 contiguous blocks (sliding_window/64 = 4).
    # Note: SlidingWindowManager scans right-to-left; when the first
    # 4-contiguous match is found at indexes 6..9 (rightmost), it
    # truncates trailing blocks past i+sliding_window_contiguous_blocks
    # and returns. The leading slots (0..5) are null blocks per design
    # of SlidingWindowManager.find_longest_cache_hit.
    assert len(result) == 1
    blocks = result[0]
    # Right-to-left scan with sliding_window_contiguous_blocks = ceil(255/64) = 4
    # finds match at index 6 (when num_contiguous_blocks reaches 4 starting
    # from i=9).  That leaves blocks[0..5] as null + blocks[6..9] as cached
    # (10 entries total before trim, then truncated to 10 entries kept).
    # Per SlidingWindowManager semantics: `del computed[i + 4 :]` so
    # blocks[10:] is removed; final length = 10.
    assert len(blocks) == 10


def test_enabled_partial_match_at_end_returns_empty(monkeypatch):
    """Pool only holds last 2 blocks; right-to-left scan resets on
    the miss at index 7. After loop num_contiguous_blocks = 0 →
    no usable prefix, returns empty."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # needs 4 contiguous
    bp = _make_block_pool()
    pool = _make_pool()
    SWARingSnapshotPoolRegistry.get().register_layer(pool)

    hashes = _hashes(10)
    # Register only last 2 hashes.
    for h in hashes[8:10]:
        pool.register(h, _make_block(fill=1))

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    # Right-to-left: hits at 9, 8 (num_contiguous=2), miss at 7
    # resets num to 0, no further hits. After loop: match_found=False,
    # num=0, computed[0:] deleted → empty.
    assert len(result[0]) == 0


def test_enabled_partial_prefix_short_of_window(monkeypatch):
    """Pool only holds first 2 blocks; partial leading prefix
    returns 2 blocks (less than sliding_window contiguous)."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)  # needs 4 contiguous
    bp = _make_block_pool()
    pool = _make_pool()
    SWARingSnapshotPoolRegistry.get().register_layer(pool)

    hashes = _hashes(10)
    # Register only first 2 hashes (a leading prefix).
    for h in hashes[0:2]:
        pool.register(h, _make_block(fill=1))

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    # Right-to-left: 9..2 all miss (num stays 0), 1 hit (num=1), 0 hit (num=2).
    # After loop: match_found=False, num=2, computed[2:] deleted → 2 blocks.
    assert len(result[0]) == 2


def test_enabled_no_match_returns_empty(monkeypatch):
    """Pool empty → no match."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)
    bp = _make_block_pool()
    pool = _make_pool()
    SWARingSnapshotPoolRegistry.get().register_layer(pool)

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
    pool = _make_pool()
    SWARingSnapshotPoolRegistry.get().register_layer(pool)

    hashes = _hashes(10)
    # Register all except index 7.
    for i, h in enumerate(hashes):
        if i != 7:
            pool.register(h, _make_block(fill=1))

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    # Right-to-left: 9, 8 hit → num=2. 7 miss → num=0. 6, 5, 4, 3 hits → num=4
    # match_found at i=3, truncates to i+4=7 entries.
    assert len(result[0]) == 7


def test_enabled_with_multiple_groups(monkeypatch):
    """Multiple cache_group_ids share the same hits."""
    monkeypatch.setattr(envs, "VLLM_DEEPSEEK_V4_SWA_PREFIX_CACHE", True)
    spec = _make_spec(sliding_window=256)
    bp = _make_block_pool()
    pool = _make_pool()
    SWARingSnapshotPoolRegistry.get().register_layer(pool)

    hashes = _hashes(10)
    for h in hashes:
        pool.register(h, _make_block(fill=1))

    result = RingSlidingWindowMLAManager.find_longest_cache_hit(
        block_hashes=hashes,
        max_length=640,
        kv_cache_group_ids=[1, 2, 3],
        block_pool=bp,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    # All groups should report identical hit patterns.
    assert len(result) == 3
    assert len(result[0]) == 10
    assert len(result[1]) == 10
    assert len(result[2]) == 10
