"""P5 production wrapper for the indexed sparse MLA prefill kernel.

This module exposes ``flash_mla_sparse_fwd_indexed_fp8`` as a vLLM
attention op so the model executor can dispatch to it under
``VLLM_DEEPSEEK_V4_PREFILL_INDEXED=1``.

The actual kernel lives in
``.kiro/specs/deepseek-v4-long-context-decay-mitigation/tools/p5_indexed_fp8_kernel.py``
and is imported lazily so a missing TileLang install or a non-SM70
GPU does not block import-time.

Status
------
**EXPERIMENTAL** — gated OFF by default. P5-C bit-exactness is
proven on synthetic caches (with and without per-block padding); a
real-model microtest is still required before flipping the default.
See ``.kiro/specs/deepseek-v4-long-context-decay-mitigation/measurements/p5d_cache_layout_audit.md``
for the resolution status.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


# Lazy import of the kernel module from the spec directory. We do not
# move the kernel into vllm/ proper yet so it stays revertable as a
# single git rm of the spec subtree until P5-I sign-off.
_KERNEL_MODULE_PATH = (
    Path("/mnt/data/apps/1Cat-vLLM/.kiro/specs/"
         "deepseek-v4-long-context-decay-mitigation/tools/"
         "p5_indexed_fp8_kernel.py")
)
_kernel_mod = None


def _load_kernel_module():
    global _kernel_mod
    if _kernel_mod is not None:
        return _kernel_mod
    if not _KERNEL_MODULE_PATH.exists():
        raise ImportError(
            f"P5 indexed kernel module not found at {_KERNEL_MODULE_PATH}. "
            "Set VLLM_DEEPSEEK_V4_PREFILL_INDEXED=0 to disable."
        )
    spec = importlib.util.spec_from_file_location(
        "_p5_indexed_fp8_kernel", _KERNEL_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p5_indexed_fp8_kernel"] = mod
    spec.loader.exec_module(mod)
    _kernel_mod = mod
    return mod


def flash_mla_sparse_fwd_indexed_fp8(
    *,
    q: torch.Tensor,
    comp_kv_bytes: torch.Tensor,
    swa_kv_bytes: torch.Tensor,
    block_table: torch.Tensor,
    swa_block_table: torch.Tensor,
    compressed_local_indices: torch.Tensor,
    compressed_lens: torch.Tensor,
    swa_lens: torch.Tensor,
    abs_pos: torch.Tensor,
    query_to_req: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    comp_block_size: int = 64,
    swa_block_size: int = 64,
    comp_block_stride: Optional[int] = None,
    swa_block_stride: Optional[int] = None,
    block_I: int = 16,
    num_stages: int = 1,
    heads_per_block: int = 64,
    threads: int = 128,
    pv_gemm_policy: str = "full_row",
    output_dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production wrapper for the P5-C indexed FP8 prefill kernel.

    See the underlying ``flash_mla_sparse_fwd_indexed_fp8`` in the
    spec module for parameter docs. Returns
    ``(output, max_logits, lse)`` matching the existing TileLang
    sparse prefill kernel's contract.
    """
    mod = _load_kernel_module()
    return mod.flash_mla_sparse_fwd_indexed_fp8(
        q=q,
        comp_kv_bytes=comp_kv_bytes,
        swa_kv_bytes=swa_kv_bytes,
        block_table=block_table,
        swa_block_table=swa_block_table,
        compressed_local_indices=compressed_local_indices,
        compressed_lens=compressed_lens,
        swa_lens=swa_lens,
        abs_pos=abs_pos,
        query_to_req=query_to_req,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        comp_block_size=comp_block_size,
        swa_block_size=swa_block_size,
        comp_block_stride=comp_block_stride,
        swa_block_stride=swa_block_stride,
        block_I=block_I,
        num_stages=num_stages,
        heads_per_block=heads_per_block,
        threads=threads,
        pv_gemm_policy=pv_gemm_policy,
        output_dtype=output_dtype,
        rows_per_chunk=rows_per_chunk,
    )


def is_indexed_prefill_enabled() -> bool:
    """Returns True if the env var is set and the kernel module is
    available. Used by ``_forward_prefill`` for runtime dispatch.
    """
    if os.environ.get("VLLM_DEEPSEEK_V4_PREFILL_INDEXED", "0") != "1":
        return False
    return _KERNEL_MODULE_PATH.exists()
