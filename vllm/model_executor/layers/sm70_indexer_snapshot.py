# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-(layer, request) contiguous K snapshot pool for the SM70
cascade-GEMM decode indexer path.

The snapshot stores ``K_f32 = decode_fp8(k_paged) * k_scales``
contiguously, sized by ``compressed_seq_lens`` so the GEMM can read
N consecutive rows without paged dereferences. Rows are filled
incrementally: at decode step T, only the new compressed tail (one
new row every ``compress_ratio`` steps) needs to be dequanted.

Layout choices:
  - One pool per ``k_cache_prefix`` (one per indexer-bearing layer).
  - LRU within a layer; capacity bounded by an env-tunable byte budget.
  - Snapshot key includes the kv_cache pointer + block_table prefix,
    mirroring the prefill ``_GATHERED_K_PREFIX_CACHE`` pattern.

Lifecycle:
  - Decode step calls ``ensure_decode_snapshot(...)`` with
    (layer prefix, kv_cache, block_table[req], block_size,
     compressed_seq_len, head_dim).
  - Helper gathers the missing rows ``[valid_count, compressed_seq_len)``
    using a Triton kernel, advances valid_count.
  - Returns the ``(values_fp32, valid_count)`` pair the GEMM needs.

Snapshot is invalidated when:
  - block_table prefix mismatch -> the request has been reset (new
    paged blocks); recompute from scratch.
  - capacity exceeded -> drop snapshot, fall back to paged kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


_DEFAULT_BYTES_PER_LAYER = 0  # 0 == auto-size from max_model_len; >0 == hard cap


def _env_bytes(name: str, default: int) -> int:
    """Resolve a byte-quota env var. Prefers the registered ``vllm.envs``
    attribute when it exists; falls back to ``os.environ`` for tests
    that monkey-patch the env directly."""
    try:
        import vllm.envs as envs
        if hasattr(envs, name):
            return int(getattr(envs, name))
    except Exception:
        pass
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_capacity_rows(head_dim: int, requested_rows: int) -> int:
    """Pick a snapshot row capacity given the requested context.

    Behaviour:

    - ``requested_rows <= 0``                    -> 0 (caller should fall back).
    - ``VLLM_SM70_INDEXER_CONTIGUOUS_KV_BYTES`` unset or 0 -> auto-size to exactly
      ``requested_rows`` so the snapshot always covers ``max_model_len``.
    - explicit byte cap > 0 ->
      ``capacity = bytes // (head_dim*4)``;
      if ``requested_rows > capacity`` we return 0 (preserves the prior
      reject-rather-than-truncate guarantee that callers rely on).
    """
    if requested_rows <= 0:
        return 0
    bytes_per_layer = _env_bytes(
        "VLLM_SM70_INDEXER_CONTIGUOUS_KV_BYTES", _DEFAULT_BYTES_PER_LAYER
    )
    if bytes_per_layer <= 0:
        return requested_rows
    capacity_rows = bytes_per_layer // (head_dim * 4)
    if requested_rows > capacity_rows:
        return 0
    return requested_rows


@dataclass
class _SnapshotEntry:
    values_fp32: torch.Tensor    # [capacity, head_dim] fp32
    valid_count: int             # rows currently populated
    block_ids: tuple[int, ...]   # snapshot of block_table[req, :n_blocks]
    row_block_ids: torch.Tensor | None = None  # int32 [capacity], graph-safe path


# Per-layer pool, keyed by ``k_cache_prefix``.
_DECODE_SNAPSHOT_POOLS: dict[str, dict[tuple, _SnapshotEntry]] = {}


def _reset_decode_snapshot_pools_for_tests() -> None:
    _DECODE_SNAPSHOT_POOLS.clear()


@triton.jit
def _decode_indexer_k_to_fp32_kernel(
    kv_cache_ptr,                   # uint8 [num_blocks, block_size, 1, D+4]
    kv_cache_block_stride,          # bytes per paged block
    kv_cache_token_stride,          # bytes per token within a block
    block_table_ptr,                # int32 [max_blocks]
    out_ptr,                        # fp32 [N, D]
    out_stride_n,
    block_size,
    start_idx,                      # gather range [start_idx, end_idx)
    end_idx,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gather + decode FP8 K rows ``[start_idx, end_idx)`` from a paged
    kv_cache into the contiguous snapshot ``out``.

    One program per BLOCK_N rows. Each program walks block_table,
    reads HEAD_DIM bytes of FP8 + 4 bytes of fp32 scale per row, and
    writes the decoded fp32 row into the snapshot.
    """
    pid = tl.program_id(0)
    n_off = start_idx + pid * BLOCK_N
    n_idx = n_off + tl.arange(0, BLOCK_N)
    n_mask = n_idx < end_idx

    cache_block_idx = n_idx // block_size
    pos_in_block = n_idx % block_size
    physical_block = tl.load(block_table_ptr + cache_block_idx, mask=n_mask, other=0)

    d_range = tl.arange(0, HEAD_DIM)
    base_addr = (
        kv_cache_ptr
        + physical_block.to(tl.int64) * kv_cache_block_stride
        + pos_in_block * kv_cache_token_stride
    )
    addr = base_addr[:, None] + d_range[None, :]
    k_uint = tl.load(addr, mask=n_mask[:, None], other=0)

    # FP8 e4m3fn decode (mirror of `_decode_fp8_e4m3fn`).
    val32 = k_uint.to(tl.int32)
    sign_bit = (val32 & 0x80) << 24
    low7 = val32 & 0x7F
    fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
    fp32_bits = tl.where(low7 == 0, 0, fp32_bits)
    k_f32 = fp32_bits.to(tl.float32, bitcast=True)

    # Load the per-row fp32 scale stored at byte offset HEAD_DIM in the
    # paged token slot.
    scale_addr = base_addr + HEAD_DIM
    scale_typed_addr = scale_addr.to(tl.pointer_type(tl.float32))
    k_scale = tl.load(scale_typed_addr, mask=n_mask, other=1.0)
    k_scaled = k_f32 * k_scale[:, None]

    out_addr = out_ptr + (n_idx - start_idx)[:, None] * out_stride_n + d_range[None, :]
    tl.store(out_addr, k_scaled, mask=n_mask[:, None])


def _block_ids_tuple(block_table_row: torch.Tensor, n_blocks: int) -> tuple[int, ...]:
    if n_blocks <= 0:
        return ()
    return tuple(int(v) for v in block_table_row[:n_blocks].detach().cpu().tolist())


@triton.jit
def _decode_indexer_k_to_fp32_cudagraph_kernel(
    kv_cache_ptr,                   # uint8 [num_blocks, block_size, hd+4]
    kv_cache_block_stride,
    kv_cache_token_stride,
    block_table_ptr,                # int32 [max_blocks]
    seq_lens_ptr,                   # flattened int32, first decode row
    out_ptr,                        # fp32 [capacity, D]
    out_stride_n,
    row_block_ids_ptr,              # int32 [capacity]
    block_size,
    capacity_rows,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Graph-safe gather/dequant from paged indexer KV into a fixed snapshot.

    The launch shape is fixed to the snapshot capacity.  Each row copies only
    when it is inside the runtime context and its physical block id differs
    from the last populated value.  Rows outside context are marked invalid so
    request resets / context shrinkage force a fresh copy before reuse.
    """
    pid = tl.program_id(0)
    n_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    in_capacity = n_idx < capacity_rows
    context_len = tl.load(seq_lens_ptr)
    context_len = tl.minimum(context_len, capacity_rows)
    in_context = in_capacity & (n_idx < context_len)

    cache_block_idx = n_idx // block_size
    pos_in_block = n_idx % block_size
    physical_block = tl.load(
        block_table_ptr + cache_block_idx, mask=in_context, other=-1
    )
    old_block = tl.load(row_block_ids_ptr + n_idx, mask=in_capacity, other=-1)
    needs_copy = in_context & (old_block != physical_block)

    d_range = tl.arange(0, HEAD_DIM)
    # SEGREGATED block layout (matches the compressor writer and the
    # reference fp8_paged_mqa_logits_torch): within a block of
    # block_size*(HEAD_DIM+4) bytes, ALL tokens' HEAD_DIM fp8 bytes come
    # first, THEN all tokens' 4-byte fp32 scales. (The old interleaved
    # read -- token = [HEAD_DIM fp8 | 4 scale] -- mismatched the writer
    # and dequantized garbage up to ~3e38, corrupting the indexer score.)
    block_base = physical_block.to(tl.int64) * kv_cache_block_stride
    fp8_base = kv_cache_ptr + block_base + pos_in_block.to(tl.int64) * HEAD_DIM
    addr = fp8_base[:, None] + d_range[None, :]
    k_uint = tl.load(addr, mask=needs_copy[:, None], other=0)

    val32 = k_uint.to(tl.int32)
    sign_bit = (val32 & 0x80) << 24
    low7 = val32 & 0x7F
    fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
    fp32_bits = tl.where(low7 == 0, 0, fp32_bits)
    k_f32 = fp32_bits.to(tl.float32, bitcast=True)

    scale_addr = (
        kv_cache_ptr
        + block_base
        + block_size * HEAD_DIM
        + pos_in_block.to(tl.int64) * 4
    )
    scale_typed_addr = scale_addr.to(tl.pointer_type(tl.float32))
    k_scale = tl.load(scale_typed_addr, mask=needs_copy, other=1.0)
    k_scaled = k_f32 * k_scale[:, None]

    out_addr = out_ptr + n_idx[:, None] * out_stride_n + d_range[None, :]
    tl.store(out_addr, k_scaled, mask=needs_copy[:, None])
    tl.store(row_block_ids_ptr + n_idx, physical_block, mask=needs_copy)
    tl.store(row_block_ids_ptr + n_idx, -1, mask=in_capacity & (n_idx >= context_len))


def _snapshot_capacity_rows(head_dim: int, requested_rows: int) -> int:
    return _resolve_capacity_rows(head_dim, requested_rows)


def _graph_snapshot_key(
    *,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    head_dim: int,
    capacity_rows: int,
) -> tuple[object, ...]:
    device = kv_cache.device
    return (
        "cudagraph",
        k_cache_prefix,
        device.type,
        device.index,
        int(kv_cache.data_ptr()),
        int(head_dim),
        int(capacity_rows),
    )


def reserve_decode_snapshot_cudagraph(
    *,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    head_dim: int,
    max_model_len: int,
) -> torch.Tensor | None:
    """Preallocate the fixed-size snapshot used by the capture-safe path."""
    capacity_rows = _snapshot_capacity_rows(head_dim, max_model_len)
    if capacity_rows <= 0:
        return None

    layer_pool = _DECODE_SNAPSHOT_POOLS.setdefault(k_cache_prefix, {})
    key = _graph_snapshot_key(
        k_cache_prefix=k_cache_prefix,
        kv_cache=kv_cache,
        head_dim=head_dim,
        capacity_rows=capacity_rows,
    )
    entry = layer_pool.get(key)
    if entry is not None:
        return entry.values_fp32

    values = torch.zeros(
        (capacity_rows, head_dim), dtype=torch.float32, device=kv_cache.device
    )
    row_block_ids = torch.full(
        (capacity_rows,), -1, dtype=torch.int32, device=kv_cache.device
    )
    entry = _SnapshotEntry(
        values_fp32=values,
        valid_count=0,
        block_ids=(),
        row_block_ids=row_block_ids,
    )
    layer_pool[key] = entry
    return entry.values_fp32


def ensure_decode_snapshot_cudagraph(
    *,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    head_dim: int,
    max_model_len: int,
) -> torch.Tensor | None:
    """Return a fixed-size fp32 K snapshot and update it without host sync."""
    snapshot = reserve_decode_snapshot_cudagraph(
        k_cache_prefix=k_cache_prefix,
        kv_cache=kv_cache,
        head_dim=head_dim,
        max_model_len=max_model_len,
    )
    if snapshot is None:
        return None

    layer_pool = _DECODE_SNAPSHOT_POOLS[k_cache_prefix]
    key = _graph_snapshot_key(
        k_cache_prefix=k_cache_prefix,
        kv_cache=kv_cache,
        head_dim=head_dim,
        capacity_rows=max_model_len,
    )
    entry = layer_pool[key]
    assert entry.row_block_ids is not None

    head_dim_with_scale = head_dim + 4
    kv_cache_u8 = kv_cache.view(torch.uint8)
    if kv_cache_u8.dim() == 4:
        kv_cache_u8_flat = kv_cache_u8.reshape(
            kv_cache_u8.shape[0], block_size, head_dim_with_scale
        )
    elif kv_cache_u8.dim() == 3:
        kv_cache_u8_flat = kv_cache_u8
    else:
        return None

    seq_lens_flat = seq_lens.reshape(-1)
    BLOCK_N = 32
    n_blocks_program = (max_model_len + BLOCK_N - 1) // BLOCK_N
    _decode_indexer_k_to_fp32_cudagraph_kernel[(n_blocks_program,)](
        kv_cache_u8_flat,
        kv_cache_u8_flat.stride(0),
        kv_cache_u8_flat.stride(1),
        block_table_row,
        seq_lens_flat,
        entry.values_fp32,
        entry.values_fp32.stride(0),
        entry.row_block_ids,
        block_size,
        max_model_len,
        HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N,
    )
    return entry.values_fp32


def ensure_decode_snapshot(
    *,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,    # int32 [max_blocks_per_req]
    block_size: int,
    compressed_seq_len: int,
    head_dim: int,
    max_model_len_bytes: int = 0,     # purely for capacity sizing
) -> torch.Tensor | None:
    """Return a contiguous fp32 K snapshot covering ``[0, compressed_seq_len)``
    rows, or ``None`` if the snapshot can't be served (capacity, mismatch,
    etc.) and the caller should fall back to the paged kernel.

    The returned tensor is always size ``[compressed_seq_len, head_dim]``.
    """
    if compressed_seq_len <= 0:
        return None

    capacity_rows = _resolve_capacity_rows(head_dim, compressed_seq_len)
    if capacity_rows <= 0:
        return None

    # Use kv_cache.data_ptr + dtype as the layer-instance identity. The
    # block_table prefix is the per-request key.
    layer_pool = _DECODE_SNAPSHOT_POOLS.setdefault(k_cache_prefix, {})
    n_blocks_used = (compressed_seq_len + block_size - 1) // block_size
    block_ids = _block_ids_tuple(block_table_row, n_blocks_used)
    if not block_ids:
        return None

    layer_key = (kv_cache.data_ptr(), block_ids[0])
    entry = layer_pool.get(layer_key)

    if entry is None:
        values = torch.empty(
            (capacity_rows, head_dim), dtype=torch.float32, device=kv_cache.device
        )
        entry = _SnapshotEntry(values_fp32=values, valid_count=0, block_ids=())
        layer_pool[layer_key] = entry

    # Validate block_table prefix matches what the snapshot was built for.
    if entry.block_ids and entry.block_ids[: len(entry.block_ids)] != block_ids[: len(entry.block_ids)]:
        # Mismatch => request has been reset; rebuild from scratch.
        entry.valid_count = 0
        entry.block_ids = ()

    # Ensure rows up to compressed_seq_len are populated.
    if entry.valid_count < compressed_seq_len:
        start_idx = entry.valid_count
        end_idx = compressed_seq_len
        BLOCK_N = 32
        n_blocks_program = (end_idx - start_idx + BLOCK_N - 1) // BLOCK_N
        # The kv_cache is shaped [num_blocks, block_size, 1, head_dim+4];
        # we treat it as a flat byte tensor for stride math.
        # Each token slot is head_dim+4 bytes. Each block is
        # block_size * (head_dim+4) bytes.
        head_dim_with_scale = head_dim + 4
        kv_cache_block_stride = block_size * head_dim_with_scale
        kv_cache_token_stride = head_dim_with_scale
        kv_cache_u8 = kv_cache.view(torch.uint8)
        if kv_cache_u8.dim() == 4:
            # Already [num_blocks, block_size, 1, hd+4] - flatten last 2.
            kv_cache_u8_flat = kv_cache_u8.reshape(kv_cache_u8.shape[0], block_size, head_dim_with_scale)
        elif kv_cache_u8.dim() == 3:
            kv_cache_u8_flat = kv_cache_u8
        else:
            return None

        out_slice = entry.values_fp32[start_idx:end_idx]
        _decode_indexer_k_to_fp32_kernel[(n_blocks_program,)](
            kv_cache_u8_flat,
            kv_cache_u8_flat.stride(0),
            kv_cache_u8_flat.stride(1),
            block_table_row,
            out_slice,
            out_slice.stride(0),
            block_size,
            0,                # start_idx within out_slice
            end_idx - start_idx,
            HEAD_DIM=head_dim,
            BLOCK_N=BLOCK_N,
        )
        entry.valid_count = compressed_seq_len
        entry.block_ids = block_ids

    return entry.values_fp32[:compressed_seq_len]


def evict_decode_snapshot(k_cache_prefix: str, kv_cache: torch.Tensor, block_table_row: torch.Tensor, n_blocks: int) -> None:
    """Drop the snapshot for a specific request when it exits."""
    pool = _DECODE_SNAPSHOT_POOLS.get(k_cache_prefix)
    if pool is None:
        return
    block_ids = _block_ids_tuple(block_table_row, n_blocks)
    if not block_ids:
        return
    pool.pop((kv_cache.data_ptr(), block_ids[0]), None)
