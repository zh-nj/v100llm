# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SWARingSnapshotPool (P6 task 41a).

Run with:
  pytest -xvs tests/v1/attention/test_swa_ring_snapshot_pool.py

These tests do not require a GPU; the pool stores tensors on CPU
(or any device), and the device match check uses the test tensor's
device.
"""
from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.mla.swa_ring_snapshot_pool import (
    SWARingSnapshotPool,
    SWARingSnapshotPoolRegistry,
)
from vllm.v1.core.kv_cache_utils import BlockHash


def _make_block(block_size: int = 64, per_token: int = 584,
                fill: int = 0) -> torch.Tensor:
    return torch.full(
        (block_size, per_token),
        fill_value=fill,
        dtype=torch.uint8,
        device="cpu",
    )


def _make_pool(max_bytes: int = 1 << 20) -> SWARingSnapshotPool:
    """Default 64-token / 584-byte SWA block; 1 MiB cap = ~28 blocks."""
    return SWARingSnapshotPool(
        layer_prefix="test_layer.0",
        block_size_tokens=64,
        per_token_bytes=584,
        device=torch.device("cpu"),
        dtype=torch.uint8,
        max_bytes=max_bytes,
    )


def test_register_and_lookup_roundtrip():
    pool = _make_pool()
    h = BlockHash(b"hash_001")
    block = _make_block(fill=0xAB)
    pool.register(h, block)

    out = pool.lookup(h)
    assert out is not None
    assert out.shape == (64, 584)
    assert torch.all(out == 0xAB)


def test_lookup_miss_returns_none():
    pool = _make_pool()
    assert pool.lookup(BlockHash(b"missing")) is None


def test_register_overwrites_existing_hash():
    pool = _make_pool()
    h = BlockHash(b"hash_dup")
    pool.register(h, _make_block(fill=0x11))
    pool.register(h, _make_block(fill=0x22))

    out = pool.lookup(h)
    assert out is not None
    assert torch.all(out == 0x22)
    assert pool.stats["current_entries"] == 1


def test_returned_tensor_is_independent_from_pool():
    """Caller can mutate the returned tensor without corrupting the
    cached snapshot."""
    pool = _make_pool()
    h = BlockHash(b"hash_iso")
    pool.register(h, _make_block(fill=0xCC))

    out1 = pool.lookup(h)
    assert out1 is not None
    out1.fill_(0x00)  # caller mutates returned tensor

    out2 = pool.lookup(h)
    assert out2 is not None
    assert torch.all(out2 == 0xCC), "pool snapshot must not change"


def test_register_decoupled_from_source_buffer():
    """Caller can mutate the source ring block buffer after register
    without corrupting the snapshot (e.g. when ring slots get reused
    for a new request)."""
    pool = _make_pool()
    h = BlockHash(b"hash_src")
    src = _make_block(fill=0x55)
    pool.register(h, src)
    src.fill_(0xFF)  # source was overwritten by next request

    out = pool.lookup(h)
    assert out is not None
    assert torch.all(out == 0x55)


def test_lru_eviction_on_overflow():
    # 64 * 584 = 37376 bytes per block; cap at 3 blocks worth.
    pool = _make_pool(max_bytes=3 * 64 * 584)
    pool.register(BlockHash(b"a"), _make_block(fill=1))
    pool.register(BlockHash(b"b"), _make_block(fill=2))
    pool.register(BlockHash(b"c"), _make_block(fill=3))
    # Now full. Inserting d should evict a (LRU).
    pool.register(BlockHash(b"d"), _make_block(fill=4))

    assert pool.lookup(BlockHash(b"a")) is None
    assert pool.lookup(BlockHash(b"b")) is not None
    assert pool.lookup(BlockHash(b"c")) is not None
    assert pool.lookup(BlockHash(b"d")) is not None
    assert pool.stats["evict_count"] == 1


def test_lookup_refreshes_lru_position():
    pool = _make_pool(max_bytes=3 * 64 * 584)
    pool.register(BlockHash(b"a"), _make_block(fill=1))
    pool.register(BlockHash(b"b"), _make_block(fill=2))
    pool.register(BlockHash(b"c"), _make_block(fill=3))

    # Touch a to refresh; b should now be LRU.
    pool.lookup(BlockHash(b"a"))

    pool.register(BlockHash(b"d"), _make_block(fill=4))
    assert pool.lookup(BlockHash(b"a")) is not None, "a refreshed, should survive"
    assert pool.lookup(BlockHash(b"b")) is None, "b should be evicted"


def test_register_wrong_shape_raises():
    pool = _make_pool()
    bad = torch.zeros((63, 584), dtype=torch.uint8)
    with pytest.raises(ValueError, match="bad ring_block shape"):
        pool.register(BlockHash(b"x"), bad)


def test_register_wrong_dtype_raises():
    pool = _make_pool()
    bad = torch.zeros((64, 584), dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        pool.register(BlockHash(b"x"), bad)


def test_register_wrong_device_raises():
    pool = _make_pool()
    if not torch.cuda.is_available():
        pytest.skip("Need CUDA for device mismatch test")
    bad = torch.zeros((64, 584), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="on .* but pool on"):
        pool.register(BlockHash(b"x"), bad)


def test_clear_resets_state():
    pool = _make_pool()
    pool.register(BlockHash(b"a"), _make_block(fill=1))
    pool.register(BlockHash(b"b"), _make_block(fill=2))
    assert pool.stats["current_entries"] == 2

    pool.clear()
    assert pool.stats["current_entries"] == 0
    assert pool.stats["current_bytes"] == 0
    assert pool.lookup(BlockHash(b"a")) is None


def test_stats_track_lookups_and_hits():
    pool = _make_pool()
    pool.register(BlockHash(b"a"), _make_block(fill=1))
    pool.lookup(BlockHash(b"a"))     # hit
    pool.lookup(BlockHash(b"a"))     # hit
    pool.lookup(BlockHash(b"miss"))  # miss

    assert pool.stats["lookup_count"] == 3
    assert pool.stats["lookup_hits"] == 2
    assert pool.stats["register_count"] == 1


def test_pool_too_small_raises_on_register():
    """Even a single block exceeds the cap -> RuntimeError on register."""
    pool = _make_pool(max_bytes=1024)  # 1 KiB < 36 KiB block
    with pytest.raises(RuntimeError, match="cannot fit"):
        pool.register(BlockHash(b"x"), _make_block())


# ---------------- Registry tests -----------------


def test_registry_register_and_lookup():
    SWARingSnapshotPoolRegistry._instance = None  # fresh
    reg = SWARingSnapshotPoolRegistry.get()
    pool = _make_pool()
    reg.register_layer(pool)

    assert reg.lookup_layer("test_layer.0") is pool
    assert reg.lookup_layer("nonexistent") is None
    assert "test_layer.0" in reg.all_layers()


def test_registry_reset_clears_pools():
    SWARingSnapshotPoolRegistry._instance = None
    reg = SWARingSnapshotPoolRegistry.get()
    pool = _make_pool()
    pool.register(BlockHash(b"x"), _make_block(fill=1))
    reg.register_layer(pool)

    reg.reset()
    assert reg.lookup_layer("test_layer.0") is None
    assert pool.stats["current_entries"] == 0


def test_registry_singleton():
    SWARingSnapshotPoolRegistry._instance = None
    a = SWARingSnapshotPoolRegistry.get()
    b = SWARingSnapshotPoolRegistry.get()
    assert a is b



# ---------------- Index tests -----------------


def _fresh_index():
    from vllm.v1.attention.backends.mla.swa_ring_snapshot_pool import (
        SWARingSnapshotIndex,
    )
    SWARingSnapshotIndex._instance = None
    return SWARingSnapshotIndex.get()


def test_index_singleton():
    from vllm.v1.attention.backends.mla.swa_ring_snapshot_pool import (
        SWARingSnapshotIndex,
    )
    SWARingSnapshotIndex._instance = None
    a = SWARingSnapshotIndex.get()
    b = SWARingSnapshotIndex.get()
    assert a is b


def test_index_promotes_when_all_ranks_ack():
    idx = _fresh_index()
    h = b"hash_promo"
    # rank 0 ack only — pending.
    idx.add_ack(h, rank=0)
    assert not idx.is_available(h)
    promoted = idx.mark_available_if_ack_complete(h, expected_ranks=2)
    assert promoted is False

    # rank 1 ack — now eligible.
    idx.add_ack(h, rank=1)
    promoted = idx.mark_available_if_ack_complete(h, expected_ranks=2)
    assert promoted is True
    assert idx.is_available(h)


def test_index_idempotent_promotion():
    idx = _fresh_index()
    h = b"hash_idem"
    idx.add_ack(h, rank=0)
    idx.mark_available_if_ack_complete(h, expected_ranks=1)
    # Promoting again returns False; already available.
    again = idx.mark_available_if_ack_complete(h, expected_ranks=1)
    assert again is False


def test_index_redundant_ack_after_promotion():
    """Ack arriving after promotion should not corrupt state."""
    idx = _fresh_index()
    h = b"hash_redundant"
    idx.add_ack(h, rank=0)
    idx.mark_available_if_ack_complete(h, expected_ranks=1)
    # Late ack from same/different rank: just refreshes LRU position.
    idx.add_ack(h, rank=1)
    assert idx.is_available(h)


def test_index_lookup_miss():
    idx = _fresh_index()
    assert not idx.is_available(b"missing")


def test_index_eviction_caps_size():
    from vllm.v1.attention.backends.mla.swa_ring_snapshot_pool import (
        SWARingSnapshotIndex,
    )
    SWARingSnapshotIndex._instance = None
    idx = SWARingSnapshotIndex(max_entries=3)
    for i in range(5):
        h = f"h{i}".encode()
        idx.add_ack(h, rank=0)
        idx.mark_available_if_ack_complete(h, expected_ranks=1)
    assert idx.stats["available_entries"] == 3
    # Oldest two evicted.
    assert not idx.is_available(b"h0")
    assert not idx.is_available(b"h1")
    assert idx.is_available(b"h2")
    assert idx.is_available(b"h3")
    assert idx.is_available(b"h4")


def test_index_lookup_refreshes_lru():
    from vllm.v1.attention.backends.mla.swa_ring_snapshot_pool import (
        SWARingSnapshotIndex,
    )
    SWARingSnapshotIndex._instance = None
    idx = SWARingSnapshotIndex(max_entries=3)
    for i in range(3):
        h = f"h{i}".encode()
        idx.add_ack(h, rank=0)
        idx.mark_available_if_ack_complete(h, expected_ranks=1)
    # touch h0 to refresh.
    assert idx.is_available(b"h0")
    # add h3 → should evict h1 (now LRU).
    idx.add_ack(b"h3", rank=0)
    idx.mark_available_if_ack_complete(b"h3", expected_ranks=1)
    assert idx.is_available(b"h0")
    assert not idx.is_available(b"h1")
    assert idx.is_available(b"h2")
    assert idx.is_available(b"h3")


def test_index_stats_track_lookups_and_hits():
    idx = _fresh_index()
    idx.add_ack(b"a", rank=0)
    idx.mark_available_if_ack_complete(b"a", expected_ranks=1)
    idx.is_available(b"a")
    idx.is_available(b"a")
    idx.is_available(b"miss")
    s = idx.stats
    assert s["acks"] == 1
    assert s["promotions"] == 1
    assert s["lookups"] == 3
    assert s["hits"] == 2


def test_index_reset_clears_state():
    idx = _fresh_index()
    idx.add_ack(b"a", rank=0)
    idx.mark_available_if_ack_complete(b"a", expected_ranks=1)
    idx.add_ack(b"b", rank=0)  # pending only
    assert idx.stats["available_entries"] == 1
    assert idx.stats["pending_entries"] == 1

    idx.reset()
    assert idx.stats["available_entries"] == 0
    assert idx.stats["pending_entries"] == 0
    assert not idx.is_available(b"a")
