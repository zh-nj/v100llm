# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.sm70_mqa_logits import (
    sm70_fp8_mqa_logits,
    sm70_fp8_mqa_logits_gemm,
    sm70_fp8_paged_mqa_logits,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
TILELANG_TOPK_THREADS = int(
    os.environ.get("VLLM_SPARSE_INDEXER_PREFILL_TILELANG_TOPK_THREADS", "256")
)
TILELANG_TOPK_MIN_ROW_LEN = int(
    os.environ.get("VLLM_SPARSE_INDEXER_PREFILL_TILELANG_TOPK_MIN_ROW_LEN", "16384")
)
DEFAULT_PREFILL_LOGITS_CHUNK_MB = 64
_LARGE_CONTEXT_TOPK_TOKENS = 2048
_LARGE_CONTEXT_TOPK_MIN_ROW_LEN = 8192
_PERSISTENT_TOPK_PREFILL_TOKENS = (512, 1024, 2048)

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


@dataclass
class _GatheredKPrefixCacheEntry:
    values: torch.Tensor
    scales: torch.Tensor
    valid_tokens: int
    block_ids: tuple[int, ...]
    tail_cu_seq_lens: torch.Tensor


_GATHERED_K_PREFIX_CACHE: dict[tuple[object, ...], _GatheredKPrefixCacheEntry] = {}


def _reset_gathered_k_prefix_cache_for_tests() -> None:
    _GATHERED_K_PREFIX_CACHE.clear()


def _gathered_k_prefix_cache_key(
    *,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    fallback_k_quant: torch.Tensor,
    fallback_k_scale: torch.Tensor,
    use_fp4_cache: bool,
) -> tuple[object, ...]:
    device = fallback_k_quant.device
    return (
        k_cache_prefix,
        device.type,
        device.index,
        int(kv_cache.data_ptr()),
        fallback_k_quant.dtype,
        fallback_k_scale.dtype,
        int(fallback_k_quant.shape[1]),
        int(fallback_k_scale.shape[1]),
        bool(use_fp4_cache),
    )


def _gathered_k_cache_block_ids(
    block_table: torch.Tensor,
    num_blocks: int,
) -> tuple[int, ...]:
    if num_blocks <= 0:
        return ()
    return tuple(int(v) for v in block_table[0, :num_blocks].detach().cpu().tolist())


def _gathered_k_cache_capacity_tokens(
    *,
    value_width: int,
    scale_width: int,
    max_bytes: int,
) -> int:
    per_token_bytes = value_width + scale_width
    if per_token_bytes <= 0 or max_bytes <= 0:
        return 0
    return max_bytes // per_token_bytes


def _allocate_gathered_k_prefix_entry(
    *,
    total_seq_lens: int,
    fallback_k_quant: torch.Tensor,
    fallback_k_scale: torch.Tensor,
    max_tokens: int,
    old_entry: _GatheredKPrefixCacheEntry | None = None,
) -> _GatheredKPrefixCacheEntry | None:
    if total_seq_lens <= 0 or total_seq_lens > max_tokens:
        return None

    current_capacity = (
        old_entry.values.shape[0] if old_entry is not None else 0
    )
    if old_entry is not None and current_capacity >= total_seq_lens:
        return old_entry

    new_capacity = max(total_seq_lens, max(current_capacity * 2, 1))
    new_capacity = min(max_tokens, new_capacity)
    if new_capacity < total_seq_lens:
        return None

    values = torch.empty(
        (new_capacity, fallback_k_quant.shape[1]),
        dtype=fallback_k_quant.dtype,
        device=fallback_k_quant.device,
    )
    scales = torch.empty(
        (new_capacity, fallback_k_scale.shape[1]),
        dtype=fallback_k_scale.dtype,
        device=fallback_k_scale.device,
    )
    if old_entry is not None and old_entry.valid_tokens > 0:
        copy_tokens = min(old_entry.valid_tokens, new_capacity)
        with _profile_indexer_or_null(
            "indexer.prefill.k_prefix_grow_copy", values
        ):
            values[:copy_tokens].copy_(old_entry.values[:copy_tokens])
            scales[:copy_tokens].copy_(old_entry.scales[:copy_tokens])
        valid_tokens = copy_tokens
        block_ids = old_entry.block_ids
    else:
        valid_tokens = 0
        block_ids = ()
    tail_cu_seq_lens = torch.empty(
        (2,), dtype=torch.int32, device=fallback_k_quant.device
    )
    return _GatheredKPrefixCacheEntry(
        values=values,
        scales=scales,
        valid_tokens=valid_tokens,
        block_ids=block_ids,
        tail_cu_seq_lens=tail_cu_seq_lens,
    )


def _try_gather_indexer_k_with_prefix_cache(
    *,
    kv_cache: torch.Tensor,
    fallback_k_quant: torch.Tensor,
    fallback_k_scale: torch.Tensor,
    chunk,
    k_cache_prefix: str,
    use_fp4_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not envs.VLLM_SPARSE_INDEXER_PREFILL_GATHERED_K_PREFIX_CACHE:
        return None
    if use_fp4_cache or chunk.num_reqs != 1 or chunk.skip_kv_gather:
        return None
    if fallback_k_quant.dim() != 2 or fallback_k_scale.dim() != 2:
        return None
    if chunk.total_seq_lens <= 0:
        return None

    cache_block_size = int(kv_cache.shape[1])
    total_seq_lens = int(chunk.total_seq_lens)
    total_blocks = cdiv(total_seq_lens, cache_block_size)
    max_tokens = _gathered_k_cache_capacity_tokens(
        value_width=int(fallback_k_quant.shape[1]) * fallback_k_quant.element_size(),
        scale_width=int(fallback_k_scale.shape[1]) * fallback_k_scale.element_size(),
        max_bytes=envs.VLLM_SPARSE_INDEXER_PREFILL_GATHERED_K_PREFIX_CACHE_BYTES,
    )
    if total_seq_lens > max_tokens:
        return None

    key = _gathered_k_prefix_cache_key(
        k_cache_prefix=k_cache_prefix,
        kv_cache=kv_cache,
        fallback_k_quant=fallback_k_quant,
        fallback_k_scale=fallback_k_scale,
        use_fp4_cache=use_fp4_cache,
    )
    entry = _GATHERED_K_PREFIX_CACHE.get(key)
    entry = _allocate_gathered_k_prefix_entry(
        total_seq_lens=total_seq_lens,
        fallback_k_quant=fallback_k_quant,
        fallback_k_scale=fallback_k_scale,
        max_tokens=max_tokens,
        old_entry=entry,
    )
    if entry is None:
        _GATHERED_K_PREFIX_CACHE.pop(key, None)
        return None
    _GATHERED_K_PREFIX_CACHE[key] = entry

    if entry.valid_tokens >= total_seq_lens:
        expected_blocks = _gathered_k_cache_block_ids(
            chunk.block_table, total_blocks
        )
        if entry.block_ids[:total_blocks] == expected_blocks:
            return (
                entry.values[:total_seq_lens],
                entry.scales[:total_seq_lens],
            )
        entry.valid_tokens = 0
        entry.block_ids = ()

    can_append_tail = entry.valid_tokens > 0 and (
        entry.valid_tokens % cache_block_size == 0
    )
    if can_append_tail:
        prefix_blocks = entry.valid_tokens // cache_block_size
        expected_prefix = _gathered_k_cache_block_ids(
            chunk.block_table, prefix_blocks
        )
        if entry.block_ids[:prefix_blocks] == expected_prefix:
            tail_tokens = total_seq_lens - entry.valid_tokens
            if tail_tokens > 0:
                entry.tail_cu_seq_lens[0] = 0
                entry.tail_cu_seq_lens[1] = tail_tokens
                tail_block_table = chunk.block_table[:, prefix_blocks:]
                with _profile_indexer_or_null(
                    "indexer.prefill.k_gather_tail",
                    entry.values[entry.valid_tokens:total_seq_lens],
                ):
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache,
                        entry.values[entry.valid_tokens:total_seq_lens],
                        entry.scales[entry.valid_tokens:total_seq_lens],
                        tail_block_table,
                        entry.tail_cu_seq_lens,
                    )
            entry.valid_tokens = total_seq_lens
            entry.block_ids = _gathered_k_cache_block_ids(
                chunk.block_table, total_blocks
            )
            return (
                entry.values[:total_seq_lens],
                entry.scales[:total_seq_lens],
            )

    entry.valid_tokens = 0
    entry.block_ids = ()
    with _profile_indexer_or_null(
        "indexer.prefill.k_gather_full_snapshot", entry.values
    ):
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            entry.values[:total_seq_lens],
            entry.scales[:total_seq_lens],
            chunk.block_table,
            chunk.cu_seq_lens,
        )
    entry.valid_tokens = total_seq_lens
    entry.block_ids = _gathered_k_cache_block_ids(chunk.block_table, total_blocks)
    return (
        entry.values[:total_seq_lens],
        entry.scales[:total_seq_lens],
    )


def _profile_indexer_or_null(label: str, ref: torch.Tensor):
    """Use DeepSeek V4 phase profiler from this module without import cycles."""
    try:
        if not ref.is_cuda:
            return nullcontext()
        from vllm.model_executor.layers import deepseek_v4_attention

        return deepseek_v4_attention._profile_or_null(label, ref)
    except Exception:
        return nullcontext()


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def _can_use_sm70_torch_indexer_fallback(use_fp4_cache: bool) -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(70)
        and not use_fp4_cache
    )


def _sm70_mqa_logits_impl() -> str:
    """Return the SM70 prefill logits implementation.

    New deployments default to the headwise GEMM path.  The legacy
    VLLM_SM70_MQA_LOGITS_TRITON switch is still honored when the new selector
    is absent so old A/B scripts keep their meaning.
    """
    impl = os.environ.get("VLLM_SM70_MQA_LOGITS_IMPL")
    if impl is None:
        legacy_triton = os.environ.get("VLLM_SM70_MQA_LOGITS_TRITON")
        if legacy_triton is None:
            impl = "gemm"
        else:
            impl = "triton" if legacy_triton == "1" else "torch"
    impl = impl.strip().lower()
    if impl not in {"gemm", "triton", "torch"}:
        raise ValueError(
            "VLLM_SM70_MQA_LOGITS_IMPL must be one of: gemm, triton, torch"
        )
    return impl


def _prefill_logits_chunk_mb() -> int:
    value = int(
        os.environ.get(
            "VLLM_SPARSE_INDEXER_PREFILL_LOGITS_CHUNK_MB",
            str(DEFAULT_PREFILL_LOGITS_CHUNK_MB),
        )
    )
    return max(value, 1)


def _iter_prefill_logits_row_chunks(num_rows: int, num_kv_tokens: int):
    """Yield row slices that cap the temporary fp32 [rows, kv] logits tensor."""
    if num_rows <= 0:
        return
    if num_kv_tokens <= 0:
        yield 0, num_rows
        return
    max_logits_bytes = _prefill_logits_chunk_mb() * 1024 * 1024
    bytes_per_row = num_kv_tokens * torch.float32.itemsize
    rows_per_chunk = max(1, max_logits_bytes // bytes_per_row)
    if rows_per_chunk >= num_rows:
        yield 0, num_rows
        return
    for row_start in range(0, num_rows, rows_per_chunk):
        yield row_start, min(row_start + rows_per_chunk, num_rows)


def _fp8_mqa_logits_torch_fallback(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    if _can_use_sm70_torch_indexer_fallback(use_fp4_cache=False):
        impl = _sm70_mqa_logits_impl()
        if impl == "gemm":
            return sm70_fp8_mqa_logits_gemm(
                q, kv, weights, cu_seqlen_ks, cu_seqlen_ke
            )
        if impl == "triton":
            return sm70_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

    k_fp8, k_scale = kv
    seq_len_kv = k_fp8.shape[0]
    q_f32 = q.float()
    k_f32 = k_fp8.float() * k_scale.reshape(-1).float().view(-1, 1)

    positions = torch.arange(0, seq_len_kv, device=q.device)
    mask = (positions[None, :] >= cu_seqlen_ks[:, None]) & (
        positions[None, :] < cu_seqlen_ke[:, None]
    )

    score = torch.einsum("mhd,nd->hmn", q_f32, k_f32)
    logits = (score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def _context_lens_as_2d(context_lens: torch.Tensor, next_n: int) -> torch.Tensor:
    if context_lens.ndim == 2:
        return context_lens
    next_n_arange = torch.arange(next_n, device=context_lens.device, dtype=torch.int32)
    return (context_lens.unsqueeze(-1) - next_n + 1 + next_n_arange).contiguous()


def _fp8_paged_mqa_logits_torch_fallback(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    if _can_use_sm70_torch_indexer_fallback(use_fp4_cache=False):
        return sm70_fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.shape
    raw_k = kv_cache[..., :dim].contiguous()
    k = raw_k.view(fp8_dtype).float() if raw_k.dtype == torch.uint8 else raw_k.float()
    k_scale = kv_cache[..., dim:].contiguous().view(torch.float32)
    k = k * k_scale

    context_lens_2d = _context_lens_as_2d(context_lens, next_n)
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    q_f32 = q.float()
    block_size = k.shape[1]

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens_2d[batch_idx, next_idx].item())
            if context_len <= 0:
                continue

            row_weights = weights[row].float().unsqueeze(-1)
            q_row = q_f32[batch_idx, next_idx]
            for block_rk in range(cdiv(context_len, block_size)):
                block_idx = int(block_tables[batch_idx, block_rk].item())
                block_start = block_rk * block_size
                block_end = min(block_start + block_size, max_model_len)
                if block_start >= max_model_len:
                    break

                k_block = k[block_idx, : block_end - block_start, 0, :]
                scores = q_row @ k_block.transpose(0, 1)
                values = (scores.relu() * row_weights).sum(dim=0)
                offsets = torch.arange(block_start, block_end, device=q.device)
                valid = offsets < context_len
                logits[row, block_start:block_end] = torch.where(
                    valid, values, float("-inf")
                )
    return logits


def _fp8_fp4_mqa_logits_with_fallback(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    clean_logits: bool,
    use_fp4_cache: bool,
) -> torch.Tensor:
    if has_deep_gemm():
        return fp8_fp4_mqa_logits(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits
        )
    q_values, q_scale = q
    if q_scale is not None or use_fp4_cache:
        raise RuntimeError(
            "Sparse Attention Indexer FP4 fallback requires DeepGEMM."
        )
    return _fp8_mqa_logits_torch_fallback(
        q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke
    )


def _fp8_fp4_paged_mqa_logits_with_fallback(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    *,
    max_model_len: int,
    clean_logits: bool,
    use_fp4_cache: bool,
) -> torch.Tensor:
    if has_deep_gemm():
        return fp8_fp4_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=clean_logits,
        )
    q_values, q_scale = q
    if q_scale is not None or use_fp4_cache:
        raise RuntimeError(
            "Sparse Attention Indexer FP4 fallback requires DeepGEMM."
        )
    return _fp8_paged_mqa_logits_torch_fallback(
        q_values, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _should_use_large_context_topk_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_tokens: int,
    *,
    max_row_len: int | None = None,
) -> bool:
    """Return whether prefill top-k can use the filtered large-context kernel."""
    if os.environ.get("VLLM_SPARSE_INDEXER_PREFILL_FILTERED_TOPK", "0") != "1":
        return False
    if not current_platform.is_cuda() or current_platform.is_xpu():
        return False
    if topk_tokens != _LARGE_CONTEXT_TOPK_TOKENS:
        return False
    if logits.dtype != torch.float32 or logits.ndim != 2 or logits.stride(1) != 1:
        return False
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        return False
    if row_starts.shape[0] != logits.shape[0] or row_ends.shape[0] != logits.shape[0]:
        return False

    if max_row_len is None:
        # Tests and rare direct callers may not have CPU-side metadata. The hot
        # inference path passes max_row_len to avoid introducing a GPU sync here.
        max_row_len = int((row_ends - row_starts).max().item())
    return max_row_len >= _LARGE_CONTEXT_TOPK_MIN_ROW_LEN


def _should_use_persistent_topk_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_tokens: int,
    *,
    max_row_len: int | None = None,
    all_row_starts_zero: bool = False,
) -> bool:
    if os.environ.get("VLLM_SPARSE_INDEXER_PREFILL_FILTERED_TOPK", "0") != "1":
        return False
    if not current_platform.is_cuda() or current_platform.is_xpu():
        return False
    if topk_tokens not in _PERSISTENT_TOPK_PREFILL_TOKENS:
        return False
    if not all_row_starts_zero:
        return False
    if logits.dtype != torch.float32 or logits.ndim != 2 or logits.stride(1) != 1:
        return False
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        return False
    if row_starts.shape[0] != logits.shape[0] or row_ends.shape[0] != logits.shape[0]:
        return False
    if max_row_len is None:
        max_row_len = int((row_ends - row_starts).max().item())
    return max_row_len > topk_tokens


def _should_use_tilelang_topk_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_tokens: int,
    *,
    max_row_len: int | None = None,
    all_row_starts_zero: bool = False,
) -> bool:
    if os.environ.get("VLLM_SPARSE_INDEXER_PREFILL_TILELANG_TOPK", "0") != "1":
        return False
    if not current_platform.is_cuda() or current_platform.is_xpu():
        return False
    if topk_tokens != 512:
        return False
    if logits.dtype != torch.float32 or logits.ndim != 2 or logits.stride(1) != 1:
        return False
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        return False
    if row_starts.shape[0] != logits.shape[0] or row_ends.shape[0] != logits.shape[0]:
        return False
    if not all_row_starts_zero:
        return False
    if max_row_len is None:
        max_row_len = int((row_ends - row_starts).max().item())
    if max_row_len < TILELANG_TOPK_MIN_ROW_LEN:
        return False
    try:
        from vllm.v1.attention.ops.tilelang_prefill_topk import (
            is_tilelang_available,
        )

        ok, _reason = is_tilelang_available()
        return ok
    except Exception:
        return False


def _should_use_streaming_topk_prefill(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    use_fp4_cache: bool,
    max_row_len: int | None,
) -> bool:
    del q, kv_cache
    if not envs.VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK:
        return False
    if use_fp4_cache:
        return False
    if not current_platform.is_cuda():
        return False
    if not current_platform.is_device_capability_family(70):
        return False
    if topk_tokens not in (512, 1024, 2048):
        return False
    if max_row_len is None or max_row_len < 8192:
        return False
    return True


def _try_prefill_streaming_topk_indices(
    *,
    q: torch.Tensor,
    k_cache_values: torch.Tensor,
    k_cache_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk_tokens: int,
    use_fp4_cache: bool,
    max_row_len: int | None,
) -> bool:
    if not _should_use_streaming_topk_prefill(
        q=q,
        kv_cache=k_cache_values,
        topk_tokens=topk_tokens,
        use_fp4_cache=use_fp4_cache,
        max_row_len=max_row_len,
    ):
        return False

    from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
        prefill_streaming_topk_tilelang,
    )

    streaming_out_indices = out_indices
    copy_streaming_output = False
    if not out_indices.is_contiguous():
        streaming_out_indices = torch.empty(
            out_indices.shape,
            dtype=out_indices.dtype,
            device=out_indices.device,
        )
        copy_streaming_output = True

    prefill_streaming_topk_tilelang(
        q=q,
        k_cache_values=k_cache_values,
        k_cache_scales=k_cache_scales,
        weights=weights,
        row_starts=row_starts,
        row_ends=row_ends,
        out_indices=streaming_out_indices,
        topk_tokens=topk_tokens,
    )
    if copy_streaming_output:
        out_indices.copy_(streaming_out_indices)
    if envs.VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK_DEBUG_COMPARE:
        reference_logits = _fp8_fp4_mqa_logits_with_fallback(
            (q, None),
            (k_cache_values, k_cache_scales),
            weights,
            row_starts,
            row_ends,
            clean_logits=False,
            use_fp4_cache=False,
        )
        reference_indices = torch.empty_like(out_indices)
        _prefill_topk_indices(
            reference_logits,
            row_starts,
            row_ends,
            reference_indices,
            topk_tokens,
            max_row_len=max_row_len,
            all_row_starts_zero=bool(torch.count_nonzero(row_starts).item() == 0),
        )
        if not _streaming_topk_indices_match_scores(
            reference_logits,
            row_starts,
            row_ends,
            out_indices,
            reference_indices,
        ):
            logger.warning(
                "Streaming prefill top-k debug compare failed; "
                "using logits-input top-k output for this chunk."
            )
            out_indices.copy_(reference_indices)
    return True


def _streaming_topk_indices_match_scores(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    streaming_indices: torch.Tensor,
    reference_indices: torch.Tensor,
) -> bool:
    def selected_scores(indices: torch.Tensor) -> torch.Tensor:
        max_col = logits.shape[1] - 1
        local = indices.to(torch.int64)
        abs_cols = row_starts.to(torch.int64).view(-1, 1) + local.clamp_min(0)
        abs_cols = abs_cols.clamp(0, max_col)
        scores = logits.gather(1, abs_cols)
        valid = (local >= 0) & (
            local < (row_ends - row_starts).to(torch.int64).view(-1, 1)
        )
        return torch.where(valid, scores, torch.full_like(scores, -torch.inf))

    streaming_scores = selected_scores(streaming_indices)
    reference_scores = selected_scores(reference_indices)
    streaming_sorted = streaming_scores.sort(dim=1).values
    reference_sorted = reference_scores.sort(dim=1).values
    if torch.allclose(streaming_sorted, reference_sorted, atol=2e-2, rtol=2e-2):
        return True
    finite = torch.isfinite(streaming_sorted) & torch.isfinite(reference_sorted)
    if finite.any():
        max_gap = (streaming_sorted[finite] - reference_sorted[finite]).abs().max()
        logger.warning(
            "Streaming prefill top-k debug compare max score gap: %.6f",
            float(max_gap.item()),
        )
    return False


def _prefill_topk_indices(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    *,
    max_row_len: int | None = None,
    all_row_starts_zero: bool = False,
    topk_workspace: torch.Tensor | None = None,
    causal_row_offset: int | None = None,
) -> None:
    lengths = row_ends - row_starts
    if _should_use_tilelang_topk_prefill(
        logits,
        row_starts,
        row_ends,
        topk_tokens,
        max_row_len=max_row_len,
        all_row_starts_zero=all_row_starts_zero,
    ):
        from vllm.v1.attention.ops.tilelang_prefill_topk import (
            prefill_topk_tilelang,
        )

        prefill_topk_tilelang(
            logits,
            topk_indices,
            lengths,
            row_starts,
            topk_tokens=topk_tokens,
            threads=TILELANG_TOPK_THREADS,
            causal_row_offset=causal_row_offset,
        )
        return

    if _should_use_persistent_topk_prefill(
        logits,
        row_starts,
        row_ends,
        topk_tokens,
        max_row_len=max_row_len,
        all_row_starts_zero=all_row_starts_zero,
    ):
        if topk_workspace is None:
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
        torch.ops._C.persistent_topk(
            logits,
            lengths,
            topk_indices,
            topk_workspace,
            topk_tokens,
            max_row_len if max_row_len is not None else int(lengths.max().item()),
        )
        return

    if _should_use_large_context_topk_prefill(
        logits,
        row_starts,
        row_ends,
        topk_tokens,
        max_row_len=max_row_len,
    ):
        torch.ops._C.large_context_topk(logits, topk_indices, lengths, row_starts)
        return

    num_rows = logits.shape[0]
    if current_platform.is_xpu():
        xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
            logits,
            row_starts,
            row_ends,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
    else:
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )


# ---------------------------------------------------------------------------
# Cascade-GEMM decode dispatch (path A from
# .kiro/specs/deepseek-v4-decode-indexer-on-compressed-kv/).
# ---------------------------------------------------------------------------


_CASCADE_GEMM_DEFAULT_THRESHOLD = 2048
_CASCADE_DEBUG_LOGGED: bool = False


def _cascade_gemm_threshold() -> int:
    return max(1, int(envs.VLLM_SM70_INDEXER_CASCADE_GEMM_THRESHOLD))


def _cascade_gemm_enabled() -> bool:
    return bool(envs.VLLM_SM70_INDEXER_CASCADE_GEMM)


def _maybe_cascade_gemm_decode_logits(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    max_model_len: int,
    use_fp4_cache: bool,
    k_cache_prefix: str,
    cache_block_size: int,
) -> torch.Tensor | None:
    """Return logits via the SM70 cascade-GEMM path, or None to fall
    back. Guards: env-on, SM70 only, FP8 cache only, batch=1, next_n=1,
    and static context capacity >= threshold. Snapshot lookup may also
    return None (capacity exceeded) -> caller falls back."""
    global _CASCADE_DEBUG_LOGGED
    debug = bool(envs.VLLM_SM70_INDEXER_CASCADE_GEMM_DEBUG)
    if not _cascade_gemm_enabled():
        if debug and not _CASCADE_DEBUG_LOGGED:
            logger.info("cascade-gemm: skipped (env disabled)")
            _CASCADE_DEBUG_LOGGED = True
        return None
    if not _can_use_sm70_torch_indexer_fallback(use_fp4_cache=use_fp4_cache):
        if debug and not _CASCADE_DEBUG_LOGGED:
            logger.info("cascade-gemm: skipped (not SM70 / use_fp4_cache=%s)", use_fp4_cache)
            _CASCADE_DEBUG_LOGGED = True
        return None
    if q_scale is not None:
        if debug and not _CASCADE_DEBUG_LOGGED:
            logger.info("cascade-gemm: skipped (q_scale present, fp4 path)")
            _CASCADE_DEBUG_LOGGED = True
        return None
    if q_quant.dim() != 4 or q_quant.shape[0] != 1 or q_quant.shape[1] != 1:
        if debug and not _CASCADE_DEBUG_LOGGED:
            logger.info("cascade-gemm: skipped (q_quant.shape=%s)", tuple(q_quant.shape))
            _CASCADE_DEBUG_LOGGED = True
        return None
    threshold = _cascade_gemm_threshold()
    if max_model_len < threshold:
        if debug and not _CASCADE_DEBUG_LOGGED:
            logger.info("cascade-gemm: skipped (max_model_len=%d < threshold=%d)", max_model_len, threshold)
            _CASCADE_DEBUG_LOGGED = True
        return None

    head_dim = int(q_quant.shape[-1])
    from vllm.model_executor.layers.sm70_indexer_snapshot import (
        ensure_decode_snapshot_cudagraph,
    )
    from vllm.model_executor.layers.sm70_cascade_gemm_indexer import (
        sm70_cascade_gemm_indexer_from_snapshot,
    )

    block_table_row = block_table[0]
    snapshot = ensure_decode_snapshot_cudagraph(
        k_cache_prefix=k_cache_prefix,
        kv_cache=kv_cache,
        block_table_row=block_table_row,
        seq_lens=seq_lens,
        block_size=cache_block_size,
        head_dim=head_dim,
        max_model_len=max_model_len,
    )
    if snapshot is None:
        if debug and not _CASCADE_DEBUG_LOGGED:
            logger.info(
                "cascade-gemm: skipped (snapshot reservation returned None; "
                "max_model_len=%d head_dim=%d)",
                max_model_len, head_dim,
            )
            _CASCADE_DEBUG_LOGGED = True
        return None

    if debug and not _CASCADE_DEBUG_LOGGED:
        logger.info(
            "cascade-gemm: ENGAGED (k_cache_prefix=%s max_model_len=%d "
            "snapshot.shape=%s threshold=%d)",
            k_cache_prefix, max_model_len, tuple(snapshot.shape), threshold,
        )
        _CASCADE_DEBUG_LOGGED = True

    # weights here is [num_padded_tokens, H] = [1, 64] for batch=1 next_n=1.
    return sm70_cascade_gemm_indexer_from_snapshot(
        q=q_quant,
        k_f32_cache=snapshot,
        weights=weights,
        context_lens=seq_lens,
        max_model_len=max_model_len,
    )


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        if (
            envs.VLLM_SM70_INDEXER_CASCADE_GEMM
            and _can_use_sm70_torch_indexer_fallback(use_fp4_cache=use_fp4_cache)
            and not use_fp4_cache
        ):
            from vllm.model_executor.layers.sm70_indexer_snapshot import (
                reserve_decode_snapshot_cudagraph,
            )

            reserve_decode_snapshot_cudagraph(
                k_cache_prefix=k_cache_prefix,
                kv_cache=kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache),
                head_dim=head_dim,
                max_model_len=max_model_len,
            )

        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        topk_workspace = None
        if os.environ.get("VLLM_SPARSE_INDEXER_PREFILL_FILTERED_TOPK", "0") == "1":
            k_quant_full, k_scale_full, topk_workspace = (
                workspace_manager.get_simultaneous(
                    values_spec,
                    scales_spec,
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
            )
        else:
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )
        last_prefill_gather: tuple[int, torch.Tensor, torch.Tensor] | None = None
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                cached_gather = _try_gather_indexer_k_with_prefix_cache(
                    kv_cache=kv_cache,
                    fallback_k_quant=k_quant,
                    fallback_k_scale=k_scale,
                    chunk=chunk,
                    k_cache_prefix=k_cache_prefix,
                    use_fp4_cache=use_fp4_cache,
                )
                if cached_gather is None:
                    with _profile_indexer_or_null("indexer.prefill.k_gather", k_quant):
                        ops.cp_gather_indexer_k_quant_cache(
                            kv_cache,
                            k_quant,
                            k_scale,
                            chunk.block_table,
                            chunk.cu_seq_lens,
                        )
                else:
                    k_quant, k_scale = cached_gather
                last_prefill_gather = (chunk.total_seq_lens, k_quant, k_scale)
            elif (
                last_prefill_gather is not None
                and last_prefill_gather[0] == chunk.total_seq_lens
            ):
                k_quant = last_prefill_gather[1]
                k_scale = last_prefill_gather[2]

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            num_q_rows = q_slice_cast.shape[0]
            num_kv_tokens = k_quant_cast.shape[0]
            with _profile_indexer_or_null(
                "indexer.prefill.streaming_topk", q_slice_cast
            ):
                streaming_handled = _try_prefill_streaming_topk_indices(
                    q=q_slice_cast,
                    k_cache_values=k_quant_cast,
                    k_cache_scales=k_scale_cast,
                    weights=weights[chunk.token_start : chunk.token_end],
                    row_starts=chunk.cu_seqlen_ks,
                    row_ends=chunk.cu_seqlen_ke,
                    out_indices=topk_indices,
                    topk_tokens=topk_tokens,
                    use_fp4_cache=use_fp4_cache,
                    max_row_len=attn_metadata_narrowed.max_seq_len,
                )
            if streaming_handled:
                continue

            for row_start, row_end in _iter_prefill_logits_row_chunks(
                num_q_rows, num_kv_tokens
            ):
                with _profile_indexer_or_null(
                    "indexer.prefill.mqa_logits",
                    q_slice_cast[row_start:row_end],
                ):
                    logits = _fp8_fp4_mqa_logits_with_fallback(
                        (
                            q_slice_cast[row_start:row_end],
                            (
                                q_scale_slice[row_start:row_end]
                                if q_scale_slice is not None
                                else None
                            ),
                        ),
                        (k_quant_cast, k_scale_cast),
                        weights[
                            chunk.token_start + row_start : chunk.token_start + row_end
                        ],
                        chunk.cu_seqlen_ks[row_start:row_end],
                        chunk.cu_seqlen_ke[row_start:row_end],
                        clean_logits=False,
                        use_fp4_cache=use_fp4_cache,
                    )

                with _profile_indexer_or_null("indexer.prefill.topk", logits):
                    _prefill_topk_indices(
                        logits,
                        chunk.cu_seqlen_ks[row_start:row_end],
                        chunk.cu_seqlen_ke[row_start:row_end],
                        topk_indices[row_start:row_end],
                        topk_tokens,
                        max_row_len=attn_metadata_narrowed.max_seq_len,
                        all_row_starts_zero=chunk.num_reqs == 1,
                        topk_workspace=topk_workspace,
                        causal_row_offset=(
                            chunk.token_start + row_start
                            if chunk.num_reqs == 1
                            else None
                        ),
                    )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        cache_block_size = int(kv_cache.shape[1])
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        with _profile_indexer_or_null(
            "indexer.decode.mqa_logits", padded_q_quant_cast
        ):
            logits = _maybe_cascade_gemm_decode_logits(
                padded_q_quant_cast,
                padded_q_scale,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len=max_model_len,
                use_fp4_cache=use_fp4_cache,
                k_cache_prefix=k_cache_prefix,
                cache_block_size=cache_block_size,
            )
            if logits is None:
                logits = _fp8_fp4_paged_mqa_logits_with_fallback(
                    (padded_q_quant_cast, padded_q_scale),
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    max_model_len=max_model_len,
                    clean_logits=False,
                    use_fp4_cache=use_fp4_cache,
                )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        with _profile_indexer_or_null("indexer.decode.topk", logits):
            if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    topk_tokens,
                    attn_metadata_narrowed.max_seq_len,
                )
            else:
                if current_platform.is_xpu():
                    xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                        logits,
                        next_n,
                        seq_lens,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
                else:
                    torch.ops._C.top_k_per_row_decode(
                        logits,
                        next_n,
                        seq_lens,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if (
            current_platform.is_cuda()
            and not has_deep_gemm()
            and not _can_use_sm70_torch_indexer_fallback(use_fp4_cache)
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )
        if (
            current_platform.is_cuda()
            and not has_deep_gemm()
            and _can_use_sm70_torch_indexer_fallback(use_fp4_cache)
        ):
            logger.warning_once(
                "DeepGEMM is not installed; using SM70 Triton fallback for "
                "Sparse Attention Indexer logits (prefill + decode). This "
                "path handles FP8 E4M3 via manual bit-decode and is "
                "correctness-safe on Volta."
            )
        if (
            envs.VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(70)
            and not use_fp4_cache
        ):
            from vllm.v1.attention.ops.tilelang_prefill_streaming_topk import (
                prewarm_prefill_streaming_topk_tilelang,
            )

            prewarm_prefill_streaming_topk_tilelang(
                topk_tokens,
                threads=TILELANG_TOPK_THREADS,
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.skip_k_cache_insert, (
            "AMD platform doesn't support skip cache insert yet"
        )
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
