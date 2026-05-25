# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 SWA ring snapshot pool.

The SWA ring is a per-request, fixed-size physical ring that stores
the last `sliding_window` tokens of FP8-quantized RoPE'd KV. Today
the ring is *not* prefix-cached: each new request starts with an
empty ring, so the first sliding-window tokens of any prefix
prefill must be recomputed even when the full-MLA cache already
holds the prefix.

This module adds a hash-keyed snapshot pool that lets the SWA ring
reuse another request's just-computed ring blocks when prefix
matches. On `register`, a 64-token SWA ring block is copied into a
GPU-resident snapshot slot keyed by the same `BlockHash` the
full-MLA cache uses. On `lookup`, callers retrieve the snapshot
tensor, copy it back into a fresh request's ring, and skip the
prefix prefill for that block.

Design choices:
- One snapshot pool **per layer**. Each DeepseekV4 SWA layer owns
  its own ring tensor of shape `(num_blocks, block_size, 584)`.
- LRU eviction with a configurable byte budget (`max_bytes`),
  default 2 GiB across all layers.
- Returned snapshots are *views* into the pool's storage; copying
  must happen before the slot can be evicted, or we copy on read
  and refcount manually. We choose **copy on register and copy on
  lookup** to avoid lifetime complexity.
- Pool is read-mostly; concurrent register/lookup are protected
  with a thin lock since worker calls are serialized per layer in
  practice.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


@dataclass
class _SnapshotEntry:
    block_hash: BlockHash
    # Stored in pinned-host or device-resident memory; for V100/SM70
    # we keep the snapshot on-device because the SWA ring lives on
    # device and we want the copy in/out path to be device-to-device
    # (one async memcpy, no PCIe round-trip).
    tensor: torch.Tensor


class SWARingSnapshotPool:
    """LRU-cached pool of SWA ring blocks keyed by full-MLA BlockHash.

    Each layer should own one of these. Snapshots are stored on the
    same device as the source SWA ring tensor.

    Memory model:
    - Each snapshot block is `block_size_tokens × per_token_bytes`
      (default 64 × 584 = 36,864 bytes for V4 fp8_ds_mla).
    - When a register would exceed `max_bytes`, oldest entries are
      evicted until the budget fits.

    Thread-safety:
    - All public methods take a Python lock. SM70 V4 worker calls
      to register/lookup are serialized within a single rank per
      step (one builder per kv cache group), so contention is low.
    """

    def __init__(
        self,
        layer_prefix: str,
        block_size_tokens: int,
        per_token_bytes: int,
        device: torch.device,
        dtype: torch.dtype,
        max_bytes: int,
    ) -> None:
        self.layer_prefix = layer_prefix
        self.block_size_tokens = int(block_size_tokens)
        self.per_token_bytes = int(per_token_bytes)
        self.device = device
        self.dtype = dtype
        self.max_bytes = int(max_bytes)

        self._bytes_per_block = self.block_size_tokens * self.per_token_bytes
        self._cur_bytes = 0
        self._lru: OrderedDict[BlockHash, _SnapshotEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self._stats_register_count = 0
        self._stats_lookup_count = 0
        self._stats_lookup_hits = 0
        self._stats_evict_count = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "register_count": self._stats_register_count,
            "lookup_count": self._stats_lookup_count,
            "lookup_hits": self._stats_lookup_hits,
            "evict_count": self._stats_evict_count,
            "current_bytes": self._cur_bytes,
            "current_entries": len(self._lru),
        }

    def lookup(self, block_hash: BlockHash) -> torch.Tensor | None:
        """Return a fresh tensor copy of the snapshot for `block_hash`,
        or None if not cached. Updates LRU order on hit.

        Caller is responsible for any subsequent device copy into the
        SWA ring.
        """
        with self._lock:
            self._stats_lookup_count += 1
            entry = self._lru.get(block_hash)
            if entry is None:
                return None
            # Move to end (most recently used).
            self._lru.move_to_end(block_hash)
            self._stats_lookup_hits += 1
            # Return a clone to decouple lifetime; caller may do
            # async copy on its own stream.
            return entry.tensor.clone()

    def register(self, block_hash: BlockHash, ring_block: torch.Tensor) -> None:
        """Snapshot a freshly-completed SWA ring block.

        Args:
            block_hash: same hash key as the corresponding full-MLA
                cached block. Must be `BlockHash` type so that
                lookup can match by hash chain.
            ring_block: source tensor on device,
                shape (block_size_tokens, per_token_bytes), dtype uint8.

        The implementation copies the tensor into pool-owned storage
        to decouple lifetime from the source ring buffer (which is
        a request-local view that gets reused).
        """
        if ring_block.shape != (self.block_size_tokens, self.per_token_bytes):
            raise ValueError(
                f"SWARingSnapshotPool[{self.layer_prefix}]: bad ring_block shape "
                f"{tuple(ring_block.shape)}, expected "
                f"({self.block_size_tokens}, {self.per_token_bytes})"
            )
        if ring_block.device != self.device:
            raise ValueError(
                f"SWARingSnapshotPool[{self.layer_prefix}]: ring_block on "
                f"{ring_block.device} but pool on {self.device}"
            )
        if ring_block.dtype != self.dtype:
            raise ValueError(
                f"SWARingSnapshotPool[{self.layer_prefix}]: ring_block dtype "
                f"{ring_block.dtype} but pool dtype {self.dtype}"
            )

        with self._lock:
            self._stats_register_count += 1
            # Existing entry: refresh by overwriting and moving to end.
            existing = self._lru.get(block_hash)
            if existing is not None:
                existing.tensor.copy_(ring_block)
                self._lru.move_to_end(block_hash)
                return

            # New entry: evict if needed.
            self._evict_until_fits(self._bytes_per_block)

            stored = ring_block.detach().clone().contiguous()
            self._lru[block_hash] = _SnapshotEntry(
                block_hash=block_hash,
                tensor=stored,
            )
            self._cur_bytes += self._bytes_per_block

    def _evict_until_fits(self, additional_bytes: int) -> None:
        """LRU evict until `current_bytes + additional_bytes <= max_bytes`."""
        while (
            self._cur_bytes + additional_bytes > self.max_bytes
            and self._lru
        ):
            old_hash, old_entry = self._lru.popitem(last=False)
            self._cur_bytes -= self._bytes_per_block
            self._stats_evict_count += 1
            # Drop reference to free GPU memory.
            del old_entry

        if self._cur_bytes + additional_bytes > self.max_bytes:
            raise RuntimeError(
                f"SWARingSnapshotPool[{self.layer_prefix}]: cannot fit "
                f"additional {additional_bytes} B into max_bytes={self.max_bytes}"
            )

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            self._cur_bytes = 0


class SWARingSnapshotPoolRegistry:
    """Per-process registry mapping layer prefix -> pool.

    The DeepseekV4SWACache layer registers itself here at construction
    time so the metadata builder and worker hooks can find the right
    pool without going through the layer instance directly.
    """

    _instance: "SWARingSnapshotPoolRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._pools: dict[str, SWARingSnapshotPool] = {}

    @classmethod
    def get(cls) -> "SWARingSnapshotPoolRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_layer(self, pool: SWARingSnapshotPool) -> None:
        self._pools[pool.layer_prefix] = pool

    def lookup_layer(self, layer_prefix: str) -> SWARingSnapshotPool | None:
        return self._pools.get(layer_prefix)

    def all_layers(self) -> list[str]:
        return list(self._pools.keys())

    def reset(self) -> None:
        for pool in self._pools.values():
            pool.clear()
        self._pools.clear()


class SWARingSnapshotIndex:
    """Engine-core-side set of available SWA snapshot BlockHashes.

    The actual GPU tensors live in worker-process `SWARingSnapshotPool`
    instances (one per layer per rank). The engine-core scheduler has no
    direct access to those tensors; it only needs to know *which*
    BlockHashes have been fully snapshotted (acknowledged by all TP
    ranks) so that prefix-cache lookups in `RingSlidingWindowMLAManager.find_longest_cache_hit`
    can answer "is hash H available on the worker side?".

    Eviction policy:
    - LRU bounded by `max_entries`; oldest hash is dropped when the cap
      is exceeded.
    - When the engine core evicts a hash, it does NOT signal workers to
      drop their tensors. Workers' own pool LRU eviction is independent.
      Worst case: a worker still holds a tensor that the engine never
      consults, which costs a small amount of GPU memory until the
      worker pool itself evicts it. Acceptable for prototype.

    Per-rank ack tracking:
    - `add_ack(hash, rank)` records that one rank has snapshotted the
      block.
    - `mark_available_if_ack_complete(hash, expected_ranks)` promotes
      the hash to "available" once all expected ranks have ack'd.
    - Scheduler queries via `is_available(hash)`.

    Thread-safe under a single Python lock; scheduler is single-threaded
    today so contention is not a concern.
    """

    _instance: "SWARingSnapshotIndex | None" = None
    _lock = threading.Lock()

    def __init__(self, max_entries: int = 100_000) -> None:
        self.max_entries = int(max_entries)
        self._available: OrderedDict[bytes, None] = OrderedDict()
        self._pending_acks: dict[bytes, set[int]] = {}
        self._inner_lock = threading.Lock()

        self._stats_acks = 0
        self._stats_promotions = 0
        self._stats_lookups = 0
        self._stats_hits = 0

    @classmethod
    def get(cls) -> "SWARingSnapshotIndex":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def stats(self) -> dict[str, int]:
        return {
            "acks": self._stats_acks,
            "promotions": self._stats_promotions,
            "lookups": self._stats_lookups,
            "hits": self._stats_hits,
            "available_entries": len(self._available),
            "pending_entries": len(self._pending_acks),
        }

    def add_ack(self, block_hash: bytes, rank: int) -> None:
        """Record that `rank` has snapshotted `block_hash`."""
        with self._inner_lock:
            self._stats_acks += 1
            if block_hash in self._available:
                # Already promoted; this is a redundant ack from a
                # later step. Refresh LRU position.
                self._available.move_to_end(block_hash)
                return
            self._pending_acks.setdefault(block_hash, set()).add(rank)

    def mark_available_if_ack_complete(
        self,
        block_hash: bytes,
        expected_ranks: int,
    ) -> bool:
        """Promote a hash to "available" if all expected ranks have ack'd.

        Returns True if newly promoted, False otherwise.
        """
        with self._inner_lock:
            if block_hash in self._available:
                return False
            ranks = self._pending_acks.get(block_hash)
            if ranks is None:
                return False
            if len(ranks) < expected_ranks:
                return False
            # All ranks ack'd. Promote.
            self._pending_acks.pop(block_hash)
            self._available[block_hash] = None
            self._stats_promotions += 1
            self._evict_to_cap()
            return True

    def _evict_to_cap(self) -> None:
        while len(self._available) > self.max_entries:
            self._available.popitem(last=False)

    def is_available(self, block_hash: bytes) -> bool:
        with self._inner_lock:
            self._stats_lookups += 1
            if block_hash in self._available:
                self._stats_hits += 1
                self._available.move_to_end(block_hash)
                return True
            return False

    def reset(self) -> None:
        with self._inner_lock:
            self._available.clear()
            self._pending_acks.clear()
