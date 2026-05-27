#!/usr/bin/env python3
"""Profile or summarize DeepSeek V4 Nsight Systems SQLite captures.

This tool handles both prefill and decode captures.

Examples:

  # Backward-compatible summary of an existing prefill SQLite export.
  python3 tools/summarize_deepseek_v4_nsys_sqlite.py measurements/prefill.sqlite \
      --mode prefill --prompt-tokens 4096

  # Decode summary from an existing SQLite export.
  python3 tools/summarize_deepseek_v4_nsys_sqlite.py summarize measurements/decode.sqlite \
      --mode decode --decode-steps auto

  # Capture a measured CUDA-profiler range, export SQLite, then summarize.
  python3 tools/summarize_deepseek_v4_nsys_sqlite.py profile \
      --out measurements/my_decode_capture \
      --mode decode \
      --capture-range cudaProfilerApi \
      --capture-range-end stop \
      -- \
      python3 my_probe_or_server_wrapper.py

The summary intentionally reports one selected device/rank by default. Tensor
parallel ranks mostly run in lockstep, so all-rank sums are useful for resource
accounting but are the wrong latency denominator for one request. Use the
critical device, or pass ``--device`` to reproduce a specific rank.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LAYERS = 43
MODES = ("auto", "prefill", "decode")
ANALYSIS_LEVELS = ("basic", "full")
SPARSE_DECODE_KERNEL = "sparse_fp8::detail::flash_fwd_splitkv_mla_sm70_sparse_kernel"
SPARSE_COMBINE_KERNEL = "flash_fwd_mla_combine"
REJECTION_SAMPLE_KERNEL = "rejection_greedy_sample_kernel"
EAGLE_PREPARE_INPUTS_KERNEL = "eagle_prepare_inputs_padded_kernel"
EAGLE_PREPARE_NEXT_KERNEL = "eagle_prepare_next_token_padded_kernel"
COMPUTE_SLOT_MAPPING_KERNEL = "_compute_slot_mapping_kernel"
QNORM_ROPE_KV_INSERT_KERNEL_FRAG = "qnorm_rope_kv_insert"
DEFAULT_MODEL = "/mnt/data6/models/DeepSeek-V4-Flash"
DEFAULT_CONDA_PYTHON = "/home/z/anaconda3/envs/gptq/bin/python"
DEFAULT_CONDA_BIN = "/home/z/anaconda3/envs/gptq/bin"
FASTLLM_CHUNKED_PREFILL_SIZE = 16384


@dataclass(frozen=True)
class KernelRow:
    name: str
    short_name: str
    count: int
    total_ns: int
    avg_ns: float
    registers: int
    grid_x: int
    block_x: int
    dyn_smem: int
    local_mem_total: int


@dataclass(frozen=True)
class DecodeStepInfo:
    steps: float | None
    source: str


def _kernel_name_expr() -> str:
    return "COALESCE(dem.value, short.value)"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _kernel_col_expr(columns: set[str], column: str, default: str = "0") -> str:
    if column in columns:
        return f"COALESCE(k.{column}, {default})"
    return default


def _is_tilelang_sparse_prefill_main(row: KernelRow) -> bool:
    return (
        (row.short_name == "main_kernel" or row.name == "main_kernel")
        and row.registers == 255
        and row.block_x == 128
        and row.dyn_smem >= 80_000
    )


def _is_streaming_topk_candidate(row: KernelRow) -> bool:
    return (
        (row.short_name == "main_kernel" or row.name == "main_kernel")
        and row.registers in (30, 32)
        and row.block_x == 256
        and row.dyn_smem in (1136, 1152)
    )


def _is_streaming_topk_final(row: KernelRow) -> bool:
    return (
        (row.short_name == "main_kernel" or row.name == "main_kernel")
        and row.registers == 16
        and row.block_x == 256
        and row.dyn_smem == 0
    )


def _classify_nccl(name: str) -> str | None:
    if "ncclDevKernel_AllReduce" in name:
        return "NCCL AllReduce (TP)"
    if "ncclDevKernel_AllGather" in name:
        return "NCCL AllGather"
    if "ncclDev" in name:
        return "NCCL other"
    return None


def _classify_turbomind_prefill(name: str) -> str | None:
    if "turbomind::gemm::gemm_kernel" in name or name.startswith("gemm_kernel"):
        if "fp4_e2m1" in name or "gemmSN_TN" in name:
            return "turbomind GEMM (FP4 MoE)"
        return "turbomind GEMM (FP8 dense)"
    return None


def _classify_turbomind_decode(name: str) -> str | None:
    if "turbomind::gemm::gemm_kernel" not in name and not name.startswith("gemm_kernel"):
        return None
    # MMA_Map<8,...> is the decode-shape micro-shape (single-row Q across
    # MLA proj wq/wkv/wo and MTP/aux paths). It is *not* exclusively wo_a/wo_b
    # — the previous label conflated several proj GEMMs. Keep it as a single
    # bucket but name it honestly so downstream readers don't assume only
    # the output projection is here.
    if "MMA_Map<(int)8," in name or "MMA_Map<\\(int\\)8," in name:
        return "decode proj GEMM (turbomind FP8 dense)"
    if "fp4_e2m1" in name:
        return "MoE GEMM (FP4)"
    if "__nv_fp8_e4m3" in name:
        return "MoE/GEMM (FP8)"
    return "turbomind GEMM"


def classify_prefill(row: KernelRow) -> str:
    name = row.name
    short = row.short_name
    lower = name.lower()

    nccl = _classify_nccl(name)
    if nccl is not None:
        return nccl

    # TileLang emits several kernels with the same short name. Resource
    # signatures are the stable discriminator in exported nsys SQLite traces.
    if short == "main_kernel" or name == "main_kernel":
        if _is_tilelang_sparse_prefill_main(row):
            return "TileLang sparse prefill attention main"
        if _is_streaming_topk_candidate(row):
            return "Streaming TopK TileLang radix/candidate"
        if _is_streaming_topk_final(row):
            return "Streaming TopK TileLang final/fill"
        return (
            "TileLang main_kernel other "
            f"(regs={row.registers}, block={row.block_x}, smem={row.dyn_smem})"
        )

    if "volta_sgemm" in name:
        return "indexer logits GEMM (cuBLAS volta_sgemm)"
    if "volta_fp16_s884gemm" in name:
        return "volta s884 GEMM (FP16)"
    if "cublaslt::splitkreduce" in lower or "splitkreduce" in lower:
        return "cuBLAS splitK reduce"
    if "gemv2t" in lower:
        return "cuBLAS GEMV"
    if "kernel2" in lower:
        return "cuBLAS Kernel2"

    tm = _classify_turbomind_prefill(name)
    if tm is not None:
        return tm

    if SPARSE_DECODE_KERNEL in name:
        return "sparse decode (FlashMLA SM70)"
    if "flash_fwd_mla_combine" in name:
        return "sparse decode combine"

    if "paged_mqa_logits" in name or "_sm70_fp8_paged_mqa_logits_kernel" in name:
        return "indexer paged logits"
    if "fused_indexer" in name:
        return "indexer q/rope/quant"
    if "cp_gather_indexer_k_quant_cache" in name:
        return "indexer K gather"
    if "DeviceRadixSort" in name:
        return "topk/sort"
    if "topk" in lower:
        return "topk"

    if "qnorm_rope_kv_insert" in name:
        return "qnorm_rope_kv_insert"
    if "kv_compress" in name:
        return "kv_compress"
    if "fused_inv_rope_fp8" in name or "_fused_inv_rope_fp8_quant_per_head" in name:
        return "rope_inv_quant"
    if "fp8_a_dequant_to_fp16" in name or "fp8_weight_predequant" in name:
        return "fp8 dequant"

    if "topkGatingSoftplusSqrt" in name:
        return "MoE gating"
    if "finalizeMoeRouting" in name:
        return "MoE finalize"
    if "expandInputRowsKernel" in name:
        return "MoE expand"
    if "computeExpertFirstTokenOffset" in name:
        return "MoE offset"
    if (
        "fused_swiglu" in name
        or "fused_clamp_mul_silu" in name
        or "triton_poi_fused_clamp" in name
    ):
        return "swiglu/silu"

    if "_mhc_" in name:
        return "MHC pre/post"
    if "hc_head" in lower:
        return "hc_head"

    if "direct_copy_kernel" in name:
        return "memcpy/copy (direct)"
    if "unrolled_elementwise" in name:
        return "memcpy/copy (unrolled)"
    if "vectorized_elementwise" in name:
        return "elementwise (vec)"
    if short == "elementwise_kernel" or name == "elementwise_kernel":
        return "elementwise (generic)"
    if short == "elementwise_kernel_with_index" or "elementwise_kernel_with_index" in name:
        return "elementwise (idx)"
    if "triton_red_fused" in name:
        return "triton reduce"
    if "triton_poi_fused" in name:
        return "triton point"
    if "reduce_kernel" in name or "reduce_1Block" in name or "DeviceReduce" in name:
        return "reduce"

    if "rms" in lower or "rmsnorm" in lower:
        return "rmsnorm"
    if "softmax" in lower:
        return "softmax"
    if "_save_partial_states" in name:
        return "save_partial_states"
    if "embedding" in lower:
        return "embedding"
    if "index_elementwise" in name:
        return "index"
    if "dot_kernel" in name:
        return "dot"
    if "masked_select" in name:
        return "mask"
    return f"OTHER: {name[:80]}"


def classify_decode(row: KernelRow) -> str:
    name = row.name
    lower = name.lower()

    nccl = _classify_nccl(name)
    if nccl is not None:
        return nccl

    if SPARSE_DECODE_KERNEL in name:
        return "flashmla_decode.sparse"
    if "flash_fwd_mla_combine" in name:
        return "flashmla_decode.combine"

    if "_sm70_qnorm_rope_kv_insert" in name or "qnorm_rope_kv_insert" in name:
        return "qnorm_rope_kv_insert"
    if "_fused_kv_compress_norm_rope_insert_indexer" in name:
        return "indexer.kv_compress_insert"
    if "_fused_kv_compress_norm_rope_insert_sparse_attn" in name:
        return "compressor.kv_insert"
    if "_sm70_fp8_paged_mqa_logits_kernel" in name or "paged_mqa_logits" in name:
        return "indexer.mqa_logits"
    if "_fused_indexer_q_rope_quant_kernel" in name:
        return "indexer.q_rope_quant"
    if "_save_partial_states_kernel" in name or "_save_partial_states" in name:
        return "indexer.save_partial"
    if "_compute_global_topk_indices_and_lens_kernel" in name:
        return "indexer.global_topk_indices"
    if "vllm::topKPerRowDecode" in name:
        return "indexer.topk_per_row_decode"
    if "_build_c128a_topk_metadata_kernel" in name:
        return "indexer.c128a_topk_metadata"
    if "DeviceRadixSort" in name:
        return "indexer.topk_sort"

    if "_mhc_pre_post_gemm_kernel" in name:
        return "mhc_pre"
    if "_mhc_post_fused_kernel" in name:
        return "mhc_post"
    if "_mhc_" in name:
        return "mhc_other"

    if "topkGatingSoftplusSqrt" in name:
        return "moe.routing"
    if "finalizeMoeRoutingKernel" in name or "finalizeMoeRouting" in name:
        return "moe.finalize"
    if "expandInputRowsKernel" in name:
        return "moe.expand"
    if "computeExpertFirstTokenOffset" in name:
        return "moe.offset"
    if "_fused_swiglu_limit_kernel" in name or "fused_swiglu" in name:
        return "moe.swiglu"

    tm = _classify_turbomind_decode(name)
    if tm is not None:
        return tm

    if "_fused_inv_rope_fp8_quant_per_head" in name or "fused_inv_rope_fp8" in name:
        return "o_inv_rope_fp8_quant"
    if "_fp8_a_dequant_to_fp16_kernel" in name:
        return "fp8 dequant"
    if "_fp8_weight_predequant_to_fp16_kernel" in name or "predequant_to_fp16" in name:
        return "fp8_weight_predequant"
    if "volta_sgemm" in name:
        return "cuBLAS_sgemm_fallback"
    if "gemv2T_kernel_val" in name or "gemv2t" in lower:
        return "cublas.gemv"
    if "dot_kernel" in name:
        return "cublas.dot"
    if "reduce_1Block_kernel" in name:
        return "cublas.reduce_1Block"
    if "cublasLt::splitKreduce" in name or "splitkreduce" in lower:
        return "cublas.splitK_reduce"
    if "volta_fp16_s884gemm" in name:
        return "volta s884 GEMM (FP16)"

    if "float8_copy_kernel" in name:
        return "copy.fp8"
    if "float16_copy_kernel" in name:
        return "copy.fp16_vec"
    if "direct_copy_kernel_cuda" in name and "lambda(float)" in name:
        return "copy.float"
    if "direct_copy_kernel" in name:
        return "copy.direct"
    if "unrolled_elementwise_kernel" in name:
        return "copy/unrolled_elementwise"
    if "vectorized_elementwise_kernel" in name:
        return "vec.elementwise"
    if "triton_poi_fused_clamp" in name:
        return "triton.clamp"
    if "triton_poi_fused" in name:
        return "triton.pointwise"
    if "triton_red_fused" in name:
        return "triton.reduce"
    if "DeviceReduce" in name or "reduce_kernel" in name:
        return "cub.reduce"
    if "launch_clamp_scalar" in name:
        return "torch.clamp"
    if "cutlass_70_wmma" in name:
        return "cutlass.wmma"
    if "elementwise_kernel" in name:
        return "torch.elementwise"

    if row.short_name == "main_kernel" or name == "main_kernel":
        if _is_tilelang_sparse_prefill_main(row):
            return "TileLang prefill main in decode capture"
        return (
            "TileLang main_kernel other "
            f"(regs={row.registers}, block={row.block_x}, smem={row.dyn_smem})"
        )

    if "get_mla_metadata_kernel" in name:
        return "flashmla_decode.metadata"
    if "_compute_slot_mapping_kernel" in name:
        return "slot_mapping"
    if "_compressed_slot_mapping_kernel" in name:
        return "compressed_slot_mapping"
    if "_compute_swa_indices_and_lens_kernel" in name:
        return "swa_indices"
    if "rms" in lower or "rmsnorm" in lower:
        return "rmsnorm"
    return f"OTHER: {name[:80]}"


def classify(row: KernelRow, mode: str) -> str:
    if mode == "decode":
        return classify_decode(row)
    if mode == "prefill":
        return classify_prefill(row)
    raise ValueError(f"classify requires resolved mode, got {mode!r}")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"SQLite file not found: {path}")
    return sqlite3.connect(str(path))


def capture_wall_ns(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    start, end = cur.fetchone()
    if start is None or end is None:
        raise SystemExit("No CUPTI kernels found in SQLite DB")
    return int(end - start)


def device_summaries(conn: sqlite3.Connection) -> list[tuple[int, int, int, int, int]]:
    cur = conn.execute(
        """
        SELECT deviceId, COUNT(*), SUM(end-start), MIN(start), MAX(end)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY deviceId
        ORDER BY deviceId
        """
    )
    return [(int(d), int(c), int(t), int(s), int(e)) for d, c, t, s, e in cur]


def select_device(
    summaries: list[tuple[int, int, int, int, int]], requested: str
) -> int:
    if requested == "critical":
        return max(summaries, key=lambda item: item[2])[0]
    try:
        device = int(requested)
    except ValueError as exc:
        raise SystemExit("--device must be 'critical' or an integer") from exc
    known = {item[0] for item in summaries}
    if device not in known:
        raise SystemExit(f"device {device} not found; available={sorted(known)}")
    return device


def load_kernel_rows(conn: sqlite3.Connection, device: int) -> list[KernelRow]:
    name_expr = _kernel_name_expr()
    columns = _table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    registers = _kernel_col_expr(columns, "registersPerThread")
    grid_x = _kernel_col_expr(columns, "gridX")
    block_x = _kernel_col_expr(columns, "blockX")
    dyn_smem = _kernel_col_expr(columns, "dynamicSharedMemory")
    local_mem = _kernel_col_expr(columns, "localMemoryTotal")
    cur = conn.execute(
        f"""
        SELECT
          {name_expr} AS name,
          short.value AS short_name,
          COUNT(*) AS calls,
          SUM(k.end-k.start) AS total_ns,
          AVG(k.end-k.start) AS avg_ns,
          {registers} AS registers,
          {grid_x} AS grid_x,
          {block_x} AS block_x,
          {dyn_smem} AS dyn_smem,
          {local_mem} AS local_mem_total
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds short ON short.id = k.shortName
        LEFT JOIN StringIds dem ON dem.id = k.demangledName
        WHERE k.deviceId = ?
        GROUP BY
          name,
          short_name,
          registers,
          grid_x,
          block_x,
          dyn_smem,
          local_mem_total
        ORDER BY total_ns DESC
        """,
        (device,),
    )
    rows = []
    for row in cur:
        rows.append(
            KernelRow(
                name=str(row[0]),
                short_name=str(row[1]),
                count=int(row[2]),
                total_ns=int(row[3]),
                avg_ns=float(row[4]),
                registers=int(row[5]),
                grid_x=int(row[6]),
                block_x=int(row[7]),
                dyn_smem=int(row[8]),
                local_mem_total=int(row[9]),
            )
        )
    return rows


def resolve_mode(
    requested: str,
    rows: list[KernelRow],
    prompt_tokens: int | None,
    decode_steps_arg: str | None,
) -> tuple[str, str]:
    if requested != "auto":
        return requested, "explicit"
    if prompt_tokens is not None and decode_steps_arg is None:
        return "prefill", "--prompt-tokens was supplied"
    if decode_steps_arg is not None and prompt_tokens is None:
        return "decode", "--decode-steps was supplied"

    prefill_ns = 0
    decode_ns = 0
    for row in rows:
        if (
            _is_tilelang_sparse_prefill_main(row)
            or _is_streaming_topk_candidate(row)
            or _is_streaming_topk_final(row)
            or "volta_sgemm" in row.name
        ):
            prefill_ns += row.total_ns
        if SPARSE_DECODE_KERNEL in row.name or "flash_fwd_mla_combine" in row.name:
            decode_ns += row.total_ns

    if decode_ns > 0 and (prefill_ns == 0 or decode_ns >= prefill_ns):
        return "decode", "FlashMLA decode kernels dominate selected device"
    return "prefill", "prefill TileLang/indexer kernels dominate or decode absent"


def _parse_decode_steps_arg(value: str | None) -> float | None:
    if value is None or value == "auto":
        return None
    try:
        steps = float(value)
    except ValueError as exc:
        raise SystemExit("--decode-steps must be 'auto' or a positive number") from exc
    if steps <= 0:
        raise SystemExit("--decode-steps must be positive")
    return steps


def resolve_decode_steps(
    rows: list[KernelRow],
    layers: int,
    decode_steps_arg: str | None,
) -> DecodeStepInfo:
    explicit = _parse_decode_steps_arg(decode_steps_arg)
    if explicit is not None:
        return DecodeStepInfo(explicit, "explicit --decode-steps")

    sparse_count = sum(row.count for row in rows if SPARSE_DECODE_KERNEL in row.name)
    if sparse_count and layers:
        return DecodeStepInfo(
            sparse_count / layers,
            f"FlashMLA sparse decode kernels: {sparse_count} / {layers} layers",
        )
    return DecodeStepInfo(None, "not inferred; pass --decode-steps N if needed")


def shorten(name: str, limit: int = 110) -> str:
    name = name.replace("|", "\\|")
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Pipeline-stage grouping
#
# Each category produced by classify_prefill() / classify_decode() is mapped to
# a coarse "pipeline stage" so the summary report can show the time spent in
# each logical phase of the DeepSeek V4 forward pass:
#
#   Attention           — sparse attention itself (TileLang main + FlashMLA)
#   Indexer             — top-k selection, Q/K logits, sort
#   GEMM (proj)         — Q/K/V projection, wo_a/wo_b, dense projections
#   MoE/FFN             — gating, expand, expert GEMM, finalize, swiglu
#   Norm / RoPE / KV    — RMSNorm, RoPE rotation, KV insert/compress, dequant
#   Communication       — NCCL AllReduce / AllGather
#   Memory ops          — memcpy / elementwise / cast / reduce / triton small
#   MHC / hc_head       — DeepSeek V4 multi-head compress + hc_head epilogue
#   Misc / Other        — anything not yet recognised
# ---------------------------------------------------------------------------

_STAGE_ATTENTION = "Attention"
_STAGE_INDEXER = "Indexer"
_STAGE_GEMM_PROJ = "GEMM (proj)"
_STAGE_MOE_FFN = "MoE/FFN"
_STAGE_NORM_KV = "Norm/RoPE/KV"
_STAGE_COMM = "Communication"
_STAGE_MEM = "Memory ops"
_STAGE_MHC = "MHC / hc_head"
_STAGE_OTHER = "Misc / Other"

_STAGE_ORDER = (
    _STAGE_ATTENTION,
    _STAGE_INDEXER,
    _STAGE_GEMM_PROJ,
    _STAGE_MOE_FFN,
    _STAGE_NORM_KV,
    _STAGE_COMM,
    _STAGE_MEM,
    _STAGE_MHC,
    _STAGE_OTHER,
)

_CATEGORY_STAGE: dict[str, str] = {
    # Attention
    "TileLang sparse prefill attention main": _STAGE_ATTENTION,
    "sparse decode (FlashMLA SM70)": _STAGE_ATTENTION,
    "sparse decode combine": _STAGE_ATTENTION,
    "flashmla_decode.sparse": _STAGE_ATTENTION,
    "flashmla_decode.combine": _STAGE_ATTENTION,
    "flashmla_decode.metadata": _STAGE_ATTENTION,
    # Indexer
    "Streaming TopK TileLang radix/candidate": _STAGE_INDEXER,
    "Streaming TopK TileLang final/fill": _STAGE_INDEXER,
    "indexer logits GEMM (cuBLAS volta_sgemm)": _STAGE_INDEXER,
    "indexer paged logits": _STAGE_INDEXER,
    "indexer q/rope/quant": _STAGE_INDEXER,
    "indexer K gather": _STAGE_INDEXER,
    "indexer.kv_compress_insert": _STAGE_INDEXER,
    "indexer.mqa_logits": _STAGE_INDEXER,
    "indexer.q_rope_quant": _STAGE_INDEXER,
    "indexer.save_partial": _STAGE_INDEXER,
    "indexer.global_topk_indices": _STAGE_INDEXER,
    "indexer.topk_per_row_decode": _STAGE_INDEXER,
    "indexer.c128a_topk_metadata": _STAGE_INDEXER,
    "indexer.topk_sort": _STAGE_INDEXER,
    "topk/sort": _STAGE_INDEXER,
    "topk": _STAGE_INDEXER,
    # Projection / dense GEMM
    "turbomind GEMM (FP8 dense)": _STAGE_GEMM_PROJ,
    "turbomind GEMM": _STAGE_GEMM_PROJ,
    "decode proj GEMM (turbomind FP8 dense)": _STAGE_GEMM_PROJ,
    "decode GEMM wo_a/wo_b": _STAGE_GEMM_PROJ,  # legacy label, kept for reading old SQLite
    "volta s884 GEMM (FP16)": _STAGE_GEMM_PROJ,
    "cuBLAS Kernel2": _STAGE_GEMM_PROJ,
    "cuBLAS GEMV": _STAGE_GEMM_PROJ,
    "cublas.gemv": _STAGE_GEMM_PROJ,
    "cuBLAS splitK reduce": _STAGE_GEMM_PROJ,
    "cublas.splitK_reduce": _STAGE_GEMM_PROJ,
    "cublas.dot": _STAGE_GEMM_PROJ,
    "cublas.reduce_1Block": _STAGE_GEMM_PROJ,
    "cuBLAS_sgemm_fallback": _STAGE_GEMM_PROJ,
    "cutlass.wmma": _STAGE_GEMM_PROJ,
    "dot": _STAGE_GEMM_PROJ,
    # MoE/FFN
    "turbomind GEMM (FP4 MoE)": _STAGE_MOE_FFN,
    "MoE GEMM (FP4)": _STAGE_MOE_FFN,
    "MoE/GEMM (FP8)": _STAGE_MOE_FFN,
    "MoE gating": _STAGE_MOE_FFN,
    "MoE finalize": _STAGE_MOE_FFN,
    "MoE expand": _STAGE_MOE_FFN,
    "MoE offset": _STAGE_MOE_FFN,
    "moe.routing": _STAGE_MOE_FFN,
    "moe.finalize": _STAGE_MOE_FFN,
    "moe.expand": _STAGE_MOE_FFN,
    "moe.offset": _STAGE_MOE_FFN,
    "moe.swiglu": _STAGE_MOE_FFN,
    "swiglu/silu": _STAGE_MOE_FFN,
    # Norm / RoPE / KV
    "qnorm_rope_kv_insert": _STAGE_NORM_KV,
    "kv_compress": _STAGE_NORM_KV,
    "rope_inv_quant": _STAGE_NORM_KV,
    "o_inv_rope_fp8_quant": _STAGE_NORM_KV,
    "fp8 dequant": _STAGE_NORM_KV,
    "fp8_weight_predequant": _STAGE_NORM_KV,
    "compressor.kv_insert": _STAGE_NORM_KV,
    "rmsnorm": _STAGE_NORM_KV,
    "softmax": _STAGE_NORM_KV,
    "save_partial_states": _STAGE_NORM_KV,
    "embedding": _STAGE_NORM_KV,
    "slot_mapping": _STAGE_NORM_KV,
    "compressed_slot_mapping": _STAGE_NORM_KV,
    "swa_indices": _STAGE_NORM_KV,
    # Communication
    "NCCL AllReduce (TP)": _STAGE_COMM,
    "NCCL AllGather": _STAGE_COMM,
    "NCCL other": _STAGE_COMM,
    # Memory ops
    "memcpy/copy (direct)": _STAGE_MEM,
    "memcpy/copy (unrolled)": _STAGE_MEM,
    "elementwise (vec)": _STAGE_MEM,
    "elementwise (generic)": _STAGE_MEM,
    "elementwise (idx)": _STAGE_MEM,
    "triton reduce": _STAGE_MEM,
    "triton point": _STAGE_MEM,
    "triton.pointwise": _STAGE_MEM,
    "triton.reduce": _STAGE_MEM,
    "triton.clamp": _STAGE_MEM,
    "torch.elementwise": _STAGE_MEM,
    "torch.clamp": _STAGE_MEM,
    "copy.fp8": _STAGE_MEM,
    "copy.fp16_vec": _STAGE_MEM,
    "copy.float": _STAGE_MEM,
    "copy.direct": _STAGE_MEM,
    "copy/unrolled_elementwise": _STAGE_MEM,
    "vec.elementwise": _STAGE_MEM,
    "reduce": _STAGE_MEM,
    "cub.reduce": _STAGE_MEM,
    "index": _STAGE_MEM,
    "mask": _STAGE_MEM,
    # MHC / hc_head
    "MHC pre/post": _STAGE_MHC,
    "hc_head": _STAGE_MHC,
    "mhc_pre": _STAGE_MHC,
    "mhc_post": _STAGE_MHC,
    "mhc_other": _STAGE_MHC,
}


def _stage_for(category: str) -> str:
    """Return the pipeline stage name for ``category``.

    Categories starting with ``OTHER:`` and unmapped TileLang variants fall
    through to ``Misc / Other`` so the summary still accounts for them.
    """
    if category in _CATEGORY_STAGE:
        return _CATEGORY_STAGE[category]
    if category.startswith("OTHER:"):
        return _STAGE_OTHER
    if category.startswith("TileLang main_kernel other"):
        # Unidentified TileLang JIT variant — report under Misc instead of
        # silently inflating Attention or Indexer numbers.
        return _STAGE_OTHER
    if category.startswith("TileLang prefill main in decode capture"):
        # Sometimes a captured decode trace includes a prefill kernel from a
        # bonus token; account for it under attention to keep continuity.
        return _STAGE_ATTENTION
    return _STAGE_OTHER


def _fmt_float(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def print_decode_step_rollup(
    rows: list[KernelRow],
    wall_ns: int,
    decode_steps: float | None,
    mode: str,
) -> None:
    """One-table per-step rollup for decode runs.

    Surface ms/step, %step, and call counts by pipeline stage so the
    reader can read the bottleneck order without scanning every section
    that follows. Auto-skips for prefill or when decode step count is
    unknown.
    """
    if mode != "decode" or not decode_steps:
        return

    stage_calls: dict[str, int] = defaultdict(int)
    stage_time: dict[str, int] = defaultdict(int)
    for row in rows:
        cat = classify(row, mode)
        stage = _stage_for(cat)
        stage_calls[stage] += row.count
        stage_time[stage] += row.total_ns

    if not stage_time:
        return

    total_busy = sum(stage_time.values())
    step_wall_ms = wall_ns / decode_steps / 1e6
    step_busy_ms = total_busy / decode_steps / 1e6

    print("## Decode Step Roll-up (per scheduler step)")
    print()
    print(
        f"- Steps: {_fmt_float(decode_steps, 2)}; "
        f"wall/step: {step_wall_ms:.3f} ms; "
        f"GPU-busy/step: {step_busy_ms:.3f} ms"
    )
    print(
        "- Use this table as the single source of truth for which stage to "
        "attack first; subsequent sections drill into individual kernels."
    )
    print()
    print("| stage | ms/step | %step (busy) | calls/step |")
    print("|---|---:|---:|---:|")
    ordered = sorted(_STAGE_ORDER, key=lambda s: -stage_time.get(s, 0))
    for stage in ordered:
        if stage not in stage_time:
            continue
        ms_step = stage_time[stage] / decode_steps / 1e6
        pct = 100.0 * stage_time[stage] / total_busy if total_busy else 0.0
        calls_step = stage_calls[stage] / decode_steps if decode_steps else 0.0
        print(
            f"| {stage} | {ms_step:.3f} | {pct:.1f}% | {calls_step:.1f} |"
        )
    idle_ns = max(0, wall_ns - total_busy)
    if idle_ns:
        idle_ms_step = idle_ns / decode_steps / 1e6
        print(
            f"| _idle/bubble (host-side)_ | {idle_ms_step:.3f} | "
            f"{100.0*idle_ns/wall_ns:.1f}% (of wall) | — |"
        )
    print()


def print_device_table(
    summaries: list[tuple[int, int, int, int, int]], wall_ns: int, selected: int
) -> None:
    print("## Device Summary")
    print()
    print("| device | kernels | busy ms | %wall | active span ms | selected |")
    print("|---:|---:|---:|---:|---:|:---:|")
    for device, count, busy_ns, start, end in summaries:
        marker = "yes" if device == selected else ""
        print(
            f"| {device} | {count} | {busy_ns/1e6:.1f} | "
            f"{100.0*busy_ns/wall_ns:.1f}% | {(end-start)/1e6:.1f} | {marker} |"
        )
    print()


def print_category_table(
    rows: list[KernelRow],
    wall_ns: int,
    layers: int,
    prompt_tokens: int | None,
    decode_steps: float | None,
    mode: str,
) -> dict[str, list[KernelRow]]:
    cat_rows: dict[str, list[KernelRow]] = defaultdict(list)
    cat_calls: dict[str, int] = defaultdict(int)
    cat_time: dict[str, int] = defaultdict(int)
    cat_lmem: dict[str, int] = defaultdict(int)
    for row in rows:
        cat = classify(row, mode)
        cat_rows[cat].append(row)
        cat_calls[cat] += row.count
        cat_time[cat] += row.total_ns
        if row.local_mem_total > 0:
            cat_lmem[cat] = max(cat_lmem[cat], row.local_mem_total)

    headers = [
        "category",
        "calls",
        "time ms",
        "%wall",
        "calls/layer",
        "ms/layer",
    ]
    if decode_steps is not None:
        headers.extend(["calls/step", "ms/step"])
    headers.append("max localMemTotal MB")
    if prompt_tokens:
        headers.append("us/prompt tok")

    print("## Category Summary")
    print()
    print("| " + " | ".join(headers) + " |")
    print("|---|" + "|".join("---:" for _ in headers[1:]) + "|")
    for cat, total_ns in sorted(cat_time.items(), key=lambda item: -item[1]):
        calls = cat_calls[cat]
        values: list[str] = [
            cat,
            str(calls),
            f"{total_ns/1e6:.1f}",
            f"{100.0*total_ns/wall_ns:.1f}%",
            f"{calls/layers:.2f}" if layers else "",
            f"{total_ns/1e6/layers:.2f}" if layers else "",
        ]
        if decode_steps is not None:
            values.extend(
                [
                    f"{calls/decode_steps:.2f}",
                    f"{total_ns/1e6/decode_steps:.3f}",
                ]
            )
        values.append(f"{cat_lmem[cat] / 1024.0 / 1024.0:.1f}")
        if prompt_tokens:
            values.append(f"{total_ns / max(prompt_tokens, 1) / 1e3:.1f}")
        print("| " + " | ".join(values) + " |")
    print()
    return cat_rows


def print_top_kernels(cat_rows: dict[str, list[KernelRow]], top: int) -> None:
    print("## Top Kernels Per Category")
    print()
    ordered = sorted(
        cat_rows.items(),
        key=lambda item: -sum(row.total_ns for row in item[1]),
    )
    for cat, rows in ordered[:top]:
        total_ms = sum(row.total_ns for row in rows) / 1e6
        print(f"### {cat} ({total_ms:.1f} ms)")
        print()
        print(
            "| total ms | calls | avg us | regs | gridX | blockX | "
            "dyn smem | lmem MB | kernel |"
        )
        print(
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        )
        for row in sorted(rows, key=lambda item: -item.total_ns)[:5]:
            lmem_mb = row.local_mem_total / 1024.0 / 1024.0
            print(
                f"| {row.total_ns/1e6:.1f} | {row.count} | "
                f"{row.avg_ns/1e3:.1f} | {row.registers} | {row.grid_x} | "
                f"{row.block_x} | {row.dyn_smem} | {lmem_mb:.1f} | "
                f"`{shorten(row.name)}` |"
            )
        print()


def print_stage_summary(
    cat_rows: dict[str, list[KernelRow]],
    wall_ns: int,
    layers: int,
    decode_steps: float | None,
) -> None:
    """Group categories into pipeline stages and print stage-level totals.

    For each stage the table reports total time, count, %wall, and a
    per-layer / per-step normalisation. After the stage roll-up a
    sub-listing shows the categories that contribute >= 1% of the stage,
    so the breakdown is detailed enough to spot regressions without
    burying the reader in long-tail entries.
    """
    stage_calls: dict[str, int] = defaultdict(int)
    stage_time: dict[str, int] = defaultdict(int)
    stage_categories: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for cat, rows in cat_rows.items():
        stage = _stage_for(cat)
        cat_calls_n = sum(row.count for row in rows)
        cat_time_n = sum(row.total_ns for row in rows)
        stage_calls[stage] += cat_calls_n
        stage_time[stage] += cat_time_n
        stage_categories[stage].append((cat, cat_calls_n, cat_time_n))

    print("## Pipeline-Stage Breakdown")
    print()
    headers = ["stage", "calls", "time ms", "%wall", "calls/layer", "ms/layer"]
    if decode_steps is not None:
        headers.extend(["calls/step", "ms/step"])
    print("| " + " | ".join(headers) + " |")
    print("|---|" + "|".join("---:" for _ in headers[1:]) + "|")

    ordered_stages = sorted(
        _STAGE_ORDER,
        key=lambda stage: -stage_time.get(stage, 0),
    )
    total_stage_ns = sum(stage_time.values())
    for stage in ordered_stages:
        if stage not in stage_time:
            continue
        ns = stage_time[stage]
        calls = stage_calls[stage]
        values = [
            stage,
            str(calls),
            f"{ns/1e6:.1f}",
            f"{100.0*ns/wall_ns:.1f}%",
            f"{calls/layers:.2f}" if layers else "",
            f"{ns/1e6/layers:.2f}" if layers else "",
        ]
        if decode_steps is not None:
            values.extend(
                [
                    f"{calls/decode_steps:.2f}",
                    f"{ns/1e6/decode_steps:.3f}",
                ]
            )
        print("| " + " | ".join(values) + " |")
    print()

    print("### Stage Sub-Breakdown (categories ≥ 1% of stage)")
    print()
    for stage in ordered_stages:
        if stage not in stage_time:
            continue
        stage_ns = stage_time[stage]
        if stage_ns == 0:
            continue
        contribs = sorted(stage_categories[stage], key=lambda item: -item[2])
        print(f"#### {stage} ({stage_ns/1e6:.1f} ms, {100.0*stage_ns/total_stage_ns:.1f}% of GPU work)")
        print()
        print("| category | calls | time ms | %stage | %wall |")
        print("|---|---:|---:|---:|---:|")
        residual_ns = 0
        residual_calls = 0
        for cat, calls, ns in contribs:
            pct_stage = 100.0 * ns / stage_ns if stage_ns else 0.0
            if pct_stage < 1.0 and ns / 1e6 < 0.5:
                residual_ns += ns
                residual_calls += calls
                continue
            print(
                f"| {cat} | {calls} | {ns/1e6:.1f} | "
                f"{pct_stage:.1f}% | {100.0*ns/wall_ns:.1f}% |"
            )
        if residual_ns:
            print(
                f"| _residual (<1%)_ | {residual_calls} | "
                f"{residual_ns/1e6:.1f} | "
                f"{100.0*residual_ns/stage_ns:.1f}% | "
                f"{100.0*residual_ns/wall_ns:.1f}% |"
            )
        print()


def _kernel_idle_gaps(
    conn: sqlite3.Connection, device: int, gap_threshold_us: float
) -> tuple[int, int, int, list[tuple[float, float]], dict[str, tuple[int, int]]]:
    """Return (n_gaps, total_idle_ns, n_big_gaps, big_gap_samples, hist).

    A gap is the time between the end of one kernel and the start of the
    next on the same device. Big gaps (>= ``gap_threshold_us``) are
    candidates for host-side stalls or scheduling bubbles.

    ``hist`` maps a coarse bucket label ("<10us", "10-50us", "50-200us",
    ">=200us") to ``(count, total_ns)`` so the report can distinguish
    launch-overhead noise from real synchronization waits.
    """
    cur = conn.execute(
        """
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId = ? ORDER BY start
        """,
        (device,),
    )
    prev_end: int | None = None
    n_gaps = 0
    total_idle = 0
    n_big = 0
    threshold_ns = int(gap_threshold_us * 1000)
    samples: list[tuple[float, float]] = []
    bucket_keys = ("<10us", "10-50us", "50-200us", ">=200us")
    bucket_counts = {k: 0 for k in bucket_keys}
    bucket_ns = {k: 0 for k in bucket_keys}
    for start, end in cur:
        if prev_end is not None and start > prev_end:
            gap = start - prev_end
            n_gaps += 1
            total_idle += gap
            gap_us = gap / 1000.0
            if gap_us < 10:
                bucket = "<10us"
            elif gap_us < 50:
                bucket = "10-50us"
            elif gap_us < 200:
                bucket = "50-200us"
            else:
                bucket = ">=200us"
            bucket_counts[bucket] += 1
            bucket_ns[bucket] += gap
            if gap >= threshold_ns:
                n_big += 1
                if len(samples) < 8:
                    samples.append((prev_end / 1e6, gap / 1e3))
        if prev_end is None or end > prev_end:
            prev_end = end
    hist = {k: (bucket_counts[k], bucket_ns[k]) for k in bucket_keys}
    return n_gaps, total_idle, n_big, samples, hist


def print_idle_gap_summary(
    conn: sqlite3.Connection,
    device: int,
    wall_ns: int,
    *,
    gap_threshold_us: float = 200.0,
) -> None:
    """Show how the device's idle time is distributed across kernel gaps.

    Useful for spotting host-side bubbles (scheduling, kernel launch,
    metadata builds, NCCL coordination) that the per-category table
    cannot expose because they happen outside any kernel.
    """
    n_gaps, idle_ns, n_big, samples, hist = _kernel_idle_gaps(
        conn, device, gap_threshold_us
    )
    print("## Idle-Gap Analysis (selected device)")
    print()
    if n_gaps == 0:
        print("No inter-kernel gaps recorded for the selected device.")
        print()
        return
    print(f"- Total inter-kernel idle: {idle_ns/1e6:.1f} ms ({100.0*idle_ns/wall_ns:.1f}% wall)")
    print(f"- Number of gaps: {n_gaps}")
    print(
        f"- Gaps >= {gap_threshold_us:.0f} us: {n_big} "
        f"({100.0*n_big/n_gaps:.1f}% of gaps)"
    )
    avg_gap_us = idle_ns / 1e3 / n_gaps if n_gaps else 0.0
    print(f"- Average gap: {avg_gap_us:.1f} us")
    print()
    print("### Gap-Duration Histogram")
    print()
    print(
        "Buckets separate launch-overhead noise (`<10us` per gap) from real "
        "synchronization waits (`>=200us`). A high `>=200us` total relative "
        "to wall time usually means the host CPU is the bottleneck (Python "
        "scheduling, metadata build, NCCL coordination)."
    )
    print()
    print("| bucket | gaps | %gaps | total ms | %wall |")
    print("|---|---:|---:|---:|---:|")
    for bucket in ("<10us", "10-50us", "50-200us", ">=200us"):
        cnt, tot_ns = hist[bucket]
        if cnt == 0 and tot_ns == 0:
            continue
        print(
            f"| {bucket} | {cnt} | "
            f"{100.0*cnt/n_gaps:.1f}% | {tot_ns/1e6:.1f} | "
            f"{100.0*tot_ns/wall_ns:.1f}% |"
        )
    print()
    if samples:
        print("First few large idle bubbles (timeline-relative):")
        print()
        print("| t (ms) | gap (us) |")
        print("|---:|---:|")
        for t_ms, gap_us in samples:
            print(f"| {t_ms:.1f} | {gap_us:.1f} |")
        print()


# ---------------------------------------------------------------------------
# Internal sub-stage estimators for the two hot kernels.
#
# Both `flashmla_decode.sparse` (FlashMLA SM70 splitkv kernel) and the TileLang
# sparse-prefill `main_kernel` are monolithic CUDA kernels: nsys reports a
# single timing per launch and provides no internal stage information. The
# per-stage shares below come from H17 / H37 / H56 dedicated meter runs (clock64
# instrumentation + microbench inside the kernel), and are kept as constants so
# nsys SQLite summaries can give an apples-to-apples decomposition without
# requiring a separate clock64 capture.
#
# These shares should be revisited if either kernel is restructured (e.g.
# changes to BI, heads_per_block, num_stages, or PV mainloop layout), but they
# are stable across topk and seq-len under the current SM70 emitter.
# ---------------------------------------------------------------------------

# FlashMLA SM70 sparse decode kernel internal split (H17 measurement).
# Stages within one kernel call:
#   s0  KV tile HBM->smem + fp8 dequant
#   s1  QK MMA884 + sync (dominant due to V100 mma_m8n8k4 STEP serialisation)
#   s2  online softmax + sync
#   s3  output accumulator scale + sync
#   s4  PV MMA + sync
#   s5  epilogue (LSE + write per block)
_DECODE_SPARSE_STAGE_SHARES: tuple[tuple[str, float], ...] = (
    ("s0 KV tile load + fp8 dequant", 0.184),
    ("s1 QK MMA884", 0.669),
    ("s2 online softmax", 0.011),
    ("s3 acc rescale", 0.002),
    ("s4 PV MMA", 0.133),
    ("s5 epilogue + LSE write", 0.001),
)

# TileLang sparse prefill main kernel internal split. Estimated from the
# kernel structure (Q load + per-tile {KV load, QK GEMM, online softmax,
# alpha rescale, PV GEMM} + epilogue) under default config (BI=16,
# heads_per_block=64, num_stages=1, fp32 PV acc). PV dominates because
# acc_o[H=64,D=512] fp32 spills register file on V100; QK is amortised
# across topk tiles.
_PREFILL_TILELANG_STAGE_SHARES: tuple[tuple[str, float], ...] = (
    ("Q load (one-shot)", 0.04),
    ("KV tile load + dequant (per tile)", 0.18),
    ("QK GEMM (mma_m8n8k4)", 0.32),
    ("online softmax + alpha rescale", 0.06),
    ("PV GEMM (acc_o fp32, register-spill heavy)", 0.34),
    ("epilogue (sumexp divide + sink + cast)", 0.06),
)


def _print_internal_stage_table(
    title: str,
    shares: tuple[tuple[str, float], ...],
    total_ns: int,
    calls: int,
    layers: int,
    decode_steps: float | None,
) -> None:
    print(f"#### {title} ({total_ns/1e6:.1f} ms, {calls} calls)")
    print()
    headers = ["sub-stage", "%kernel", "time ms", "ms/call (us)"]
    if layers:
        headers.append("ms/layer")
    if decode_steps is not None:
        headers.append("ms/step")
    print("| " + " | ".join(headers) + " |")
    print("|---|" + "|".join("---:" for _ in headers[1:]) + "|")
    avg_ns = total_ns / calls if calls else 0
    for label, share in shares:
        ns = total_ns * share
        values = [
            label,
            f"{100.0 * share:.1f}%",
            f"{ns/1e6:.2f}",
            f"{avg_ns * share / 1e3:.2f}",
        ]
        if layers:
            values.append(f"{ns/1e6/layers:.3f}")
        if decode_steps is not None:
            values.append(f"{ns/1e6/decode_steps:.4f}")
        print("| " + " | ".join(values) + " |")
    print()


def _aggregate_kernel_rows(rows: list[KernelRow]) -> tuple[int, int]:
    return sum(row.count for row in rows), sum(row.total_ns for row in rows)


def print_hot_kernel_internal_breakdown(
    cat_rows: dict[str, list[KernelRow]],
    layers: int,
    decode_steps: float | None,
    mode: str,
) -> None:
    """Decompose the two hot monolithic kernels into internal sub-stages.

    Source priors:
      - flashmla_decode.sparse stage shares: H17 clock64 meter results.
      - TileLang sparse prefill main shares: estimated from kernel
        structure + H56 register-spill measurements.

    The sub-stage table is *estimated*; raw nsys CUPTI cannot inject
    clock64 markers. It is included so optimization decisions can compare
    a candidate kernel-level change (e.g. multi-stage KV pipeline)
    against the share of total time the targeted sub-stage actually
    consumes.
    """
    sections: list[tuple[str, tuple[tuple[str, float], ...], list[KernelRow]]] = []

    if mode == "decode":
        sparse_rows = cat_rows.get("flashmla_decode.sparse", [])
        if sparse_rows:
            sections.append(
                (
                    "FlashMLA SM70 splitkv (sparse_decode) internal stages",
                    _DECODE_SPARSE_STAGE_SHARES,
                    sparse_rows,
                )
            )
    else:
        # Prefill mode may surface the FlashMLA sparse decode kernel when
        # the capture brackets a non-uniform query window; show both if
        # both are present so the reader does not miss either.
        sparse_decode_rows = cat_rows.get("sparse decode (FlashMLA SM70)", [])
        if sparse_decode_rows:
            sections.append(
                (
                    "FlashMLA SM70 splitkv (sparse_decode) internal stages",
                    _DECODE_SPARSE_STAGE_SHARES,
                    sparse_decode_rows,
                )
            )

    if mode == "prefill":
        prefill_rows = cat_rows.get(
            "TileLang sparse prefill attention main", []
        )
        if prefill_rows:
            sections.append(
                (
                    "TileLang SM70 sparse prefill main internal stages",
                    _PREFILL_TILELANG_STAGE_SHARES,
                    prefill_rows,
                )
            )

    if not sections:
        return

    print("## Hot Kernel Internal Sub-Stage Estimates")
    print()
    print(
        "Sub-stage shares are *priors* from clock64 / microbench runs of the "
        "monolithic kernels. nsys CUPTI cannot subdivide a kernel by itself; "
        "the percentages here are constant per kernel and were collected at:"
    )
    print()
    print(
        "- FlashMLA SM70 sparse_decode: H17 in-kernel clock64 meter "
        "(s0/s1/s2/s4 stage IDs)"
    )
    print(
        "- TileLang sparse prefill main: kernel-source structural estimate, "
        "anchored to H56 register-spill measurements"
    )
    print()
    print(
        "Use these to weigh whether an optimisation that targets a single "
        "stage is worth the engineering cost — e.g. shaving QK MMA884 only "
        "helps the ~67% of decode-sparse_decode time spent there."
    )
    print()
    for title, shares, rows in sections:
        calls, total_ns = _aggregate_kernel_rows(rows)
        if not calls or not total_ns:
            continue
        _print_internal_stage_table(
            title,
            shares,
            total_ns,
            calls,
            layers,
            decode_steps,
        )


# ---------------------------------------------------------------------------
# MTP timeline split (7 phases per scheduler step), sparse_decode shape split,
# and per-kernel duration distribution. These are auto-skipped when their
# trigger kernels are absent in the trace, so a no-MTP capture still produces
# a clean report.
# ---------------------------------------------------------------------------


def _fetch_kernels_in_time_order(
    conn: sqlite3.Connection, device: int
) -> list[tuple[int, int, str, int, int, int]]:
    """Return [(start, end, name, gridX, gridY, gridZ), ...] on device."""
    name_expr = _kernel_name_expr()
    columns = _table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    grid_x = _kernel_col_expr(columns, "gridX")
    grid_y = _kernel_col_expr(columns, "gridY")
    grid_z = _kernel_col_expr(columns, "gridZ")
    cur = conn.execute(
        f"""
        SELECT k.start, k.end, {name_expr} AS name,
               {grid_x}, {grid_y}, {grid_z}
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds short ON short.id = k.shortName
        LEFT JOIN StringIds dem ON dem.id = k.demangledName
        WHERE k.deviceId = ?
        ORDER BY k.start
        """,
        (device,),
    )
    return [
        (int(r[0]), int(r[1]), str(r[2]), int(r[3]), int(r[4]), int(r[5]))
        for r in cur
    ]


_MTP_PHASE_ORDER = (
    "accept_prepare",
    "mtp_h_e_pre_attn",
    "mtp_attn_o",
    "mtp_ffn_moe",
    "mtp_logits_sample",
    "metadata_target_prelude",
    "target_verify_from_sparse0",
)


def _split_mtp_phases(
    rows: list[tuple[int, int, str, int, int, int]],
) -> tuple[
    dict[str, list[list[tuple[int, int, str]]]],
    int,
    int,
]:
    """Split scheduler steps into 7 MTP phases.

    Rules (DSv4F MTP=1):
      1. ``rejection_greedy_sample_kernel`` ends a step.
      2. First sparse_decode after rejection = MTP draft.
      3. Second sparse_decode = target verify layer 0.
      4. ``_compute_slot_mapping_kernel`` between them marks the start
         of metadata prelude.
      5. AllReduce calls between MTP draft sparse and target sparse mark
         attention-out and FFN/MoE boundaries (1st AR ends mtp_attn_o,
         2nd AR ends mtp_ffn_moe).
      6. First GEMM and first qnorm before draft sparse split
         accept_prepare / mtp_h_e_pre_attn / mtp_attn_o.
    """
    rejects = [
        i
        for i, r in enumerate(rows)
        if r[2] == REJECTION_SAMPLE_KERNEL
        or r[2].endswith("rejection_greedy_sample_kernel")
    ]
    phases: dict[str, list[list[tuple[int, int, str]]]] = {
        name: [] for name in _MTP_PHASE_ORDER
    }
    skipped = 0
    if len(rejects) < 3:
        return phases, 0, len(rejects)

    for a, b in zip(rejects, rejects[1:]):
        start_t = rows[a][0]
        end_t = rows[b][0]
        window = [r for r in rows if start_t <= r[0] < end_t]
        sparse_idx = [
            i for i, r in enumerate(window) if SPARSE_DECODE_KERNEL in r[2]
        ]
        if len(sparse_idx) < 2:
            skipped += 1
            continue
        draft_sparse = sparse_idx[0]
        target_sparse0 = sparse_idx[1]

        def first_idx(pred, lo: int, hi: int) -> int | None:
            for i in range(lo, hi):
                if pred(window[i]):
                    return i
            return None

        first_gemm = first_idx(
            lambda r: (
                "turbomind::gemm::gemm_kernel" in r[2]
                or r[2] == "gemm_kernel"
                or "gemmSN_TN_kernel" in r[2]
                or "Kernel2" in r[2]
            ),
            0,
            draft_sparse,
        )
        first_qnorm = first_idx(
            lambda r: QNORM_ROPE_KV_INSERT_KERNEL_FRAG in r[2],
            0,
            draft_sparse,
        )
        compute_slot = first_idx(
            lambda r: COMPUTE_SLOT_MAPPING_KERNEL in r[2],
            draft_sparse + 1,
            target_sparse0,
        )
        ar_after_sparse = [
            i
            for i in range(draft_sparse + 1, compute_slot or target_sparse0)
            if "ncclDevKernel_AllReduce" in window[i][2]
        ]
        if (
            first_gemm is None
            or first_qnorm is None
            or compute_slot is None
            or len(ar_after_sparse) < 2
        ):
            skipped += 1
            continue
        ar1_end = window[ar_after_sparse[0]][1]
        ar2_end = window[ar_after_sparse[1]][1]

        boundaries: dict[str, tuple[int, int]] = {
            "accept_prepare": (start_t, window[first_gemm][0]),
            "mtp_h_e_pre_attn": (
                window[first_gemm][0],
                window[first_qnorm][0],
            ),
            "mtp_attn_o": (window[first_qnorm][0], ar1_end),
            "mtp_ffn_moe": (ar1_end, ar2_end),
            "mtp_logits_sample": (ar2_end, window[compute_slot][0]),
            "metadata_target_prelude": (
                window[compute_slot][0],
                window[target_sparse0][0],
            ),
            "target_verify_from_sparse0": (window[target_sparse0][0], end_t),
        }
        for name, (lo, hi) in boundaries.items():
            phases[name].append(
                [(r[0], r[1], r[2]) for r in window if lo <= r[0] < hi]
            )

    return phases, skipped, len(rejects)


def print_mtp_timeline_split(
    conn: sqlite3.Connection,
    device: int,
    mode: str,
    ordered_rows: list[tuple[int, int, str, int, int, int]] | None = None,
) -> None:
    """Decompose decode trace into MTP draft + target verify phases.

    Auto-skips when the trace has no MTP markers, when too few rejection
    intervals are present, or when the requested mode is prefill.
    """
    if mode != "decode":
        return
    rows = ordered_rows if ordered_rows is not None else _fetch_kernels_in_time_order(conn, device)
    phases, skipped, n_rejects = _split_mtp_phases(rows)
    usable = min((len(v) for v in phases.values()), default=0)
    if usable == 0:
        return  # No-MTP trace; suppress section.

    print("## MTP Timeline Split (per scheduler step)")
    print()
    print(
        f"- Rejection markers: {n_rejects}; usable intervals: {usable}; "
        f"skipped: {skipped}"
    )
    print(
        "- Split rule: first sparse_decode after rejection is MTP draft, "
        "second sparse_decode opens target verify."
    )
    print()
    print(
        "| Phase | wall ms/step | kernel ms/step | calls/step |"
    )
    print("|---|---:|---:|---:|")
    rollups_draft_total_wall = 0
    rollups_draft_total_kernel = 0
    rollups_draft_total_calls = 0
    for name in _MTP_PHASE_ORDER:
        bucket = phases[name]
        if not bucket:
            continue
        total_wall = 0
        total_busy = 0
        total_calls = 0
        for ks in bucket:
            if not ks:
                continue
            total_wall += max(k[1] for k in ks) - min(k[0] for k in ks)
            total_busy += sum(k[1] - k[0] for k in ks)
            total_calls += len(ks)
        if name in (
            "accept_prepare",
            "mtp_h_e_pre_attn",
            "mtp_attn_o",
            "mtp_ffn_moe",
            "mtp_logits_sample",
        ):
            rollups_draft_total_wall += total_wall
            rollups_draft_total_kernel += total_busy
            rollups_draft_total_calls += total_calls
        print(
            f"| `{name}` | {total_wall / usable / 1e6:.3f} | "
            f"{total_busy / usable / 1e6:.3f} | "
            f"{total_calls / usable:.1f} |"
        )
    print()
    print("### MTP Rollup")
    print()
    print("| Rollup | wall ms/step | kernel ms/step | calls/step |")
    print("|---|---:|---:|---:|")
    print(
        f"| MTP draft end-to-end | "
        f"{rollups_draft_total_wall / usable / 1e6:.3f} | "
        f"{rollups_draft_total_kernel / usable / 1e6:.3f} | "
        f"{rollups_draft_total_calls / usable:.1f} |"
    )

    metadata_bucket = phases["metadata_target_prelude"]
    target_bucket = phases["target_verify_from_sparse0"]
    for label, bucket in (
        ("Metadata + target layer0 prelude", metadata_bucket),
        ("Target verify from sparse0", target_bucket),
    ):
        wall = 0
        busy = 0
        calls = 0
        for ks in bucket:
            if not ks:
                continue
            wall += max(k[1] for k in ks) - min(k[0] for k in ks)
            busy += sum(k[1] - k[0] for k in ks)
            calls += len(ks)
        print(
            f"| {label} | {wall / usable / 1e6:.3f} | "
            f"{busy / usable / 1e6:.3f} | {calls / usable:.1f} |"
        )
    print()

    # Sparse position sanity table.
    draft_sparse_us: list[float] = []
    target_sparse_us: list[float] = []
    for ks in metadata_bucket:
        # nothing in metadata; sparse stays in the previous/next bucket
        pass
    for name in (
        "accept_prepare",
        "mtp_h_e_pre_attn",
        "mtp_attn_o",
        "mtp_ffn_moe",
        "mtp_logits_sample",
    ):
        for ks in phases[name]:
            for k in ks:
                if SPARSE_DECODE_KERNEL in k[2]:
                    draft_sparse_us.append((k[1] - k[0]) / 1000.0)
    for ks in target_bucket:
        for k in ks:
            if SPARSE_DECODE_KERNEL in k[2]:
                target_sparse_us.append((k[1] - k[0]) / 1000.0)
    if draft_sparse_us or target_sparse_us:
        print("### MTP Sparse Sanity")
        print()
        print("| group | calls/step | mean us | min us | max us |")
        print("|---|---:|---:|---:|---:|")
        if draft_sparse_us:
            print(
                f"| MTP draft sparse | {len(draft_sparse_us) / usable:.2f} | "
                f"{sum(draft_sparse_us) / len(draft_sparse_us):.1f} | "
                f"{min(draft_sparse_us):.1f} | "
                f"{max(draft_sparse_us):.1f} |"
            )
        if target_sparse_us:
            print(
                f"| Target verify sparse | "
                f"{len(target_sparse_us) / usable:.2f} | "
                f"{sum(target_sparse_us) / len(target_sparse_us):.1f} | "
                f"{min(target_sparse_us):.1f} | "
                f"{max(target_sparse_us):.1f} |"
            )
        print()


def print_sparse_decode_shape_split(
    conn: sqlite3.Connection,
    device: int,
    decode_steps: float | None,
    ordered_rows: list[tuple[int, int, str, int, int, int]] | None = None,
) -> None:
    """Group FlashMLA sparse_decode calls by combine gridX (b=1 vs b=2).

    The splitkv kernel itself launches identically for both shapes (it
    iterates batch internally), but the combine kernel exposes the real
    batch via gridX. b=1 = single-row decode; b=2 = MTP verify two-row.

    Auto-skips when the FlashMLA combine kernel does not appear.
    """
    rows = ordered_rows if ordered_rows is not None else _fetch_kernels_in_time_order(conn, device)
    pairs: list[tuple[float, int]] = []  # (sparse_us, combine_gridX)
    last_sparse: tuple[int, int, str, int, int, int] | None = None
    for r in rows:
        if SPARSE_DECODE_KERNEL in r[2]:
            last_sparse = r
            continue
        if last_sparse is not None and SPARSE_COMBINE_KERNEL in r[2]:
            us = (last_sparse[1] - last_sparse[0]) / 1000.0
            pairs.append((us, r[3]))
            last_sparse = None
    if not pairs:
        return
    by_grid: dict[int, list[float]] = {}
    for us, gx in pairs:
        by_grid.setdefault(gx, []).append(us)
    print("## FlashMLA sparse_decode Shape Split (combine gridX)")
    print()
    print(
        "Calls are paired with the next combine kernel; gridX exposes the "
        "real batch (1 = standard decode, 2 = MTP verify two-row, "
        ">2 = batched decode)."
    )
    print()
    headers = ["combine gridX", "interpretation", "calls", "mean us",
               "p50 us", "p99 us", "total ms"]
    if decode_steps is not None:
        headers += ["calls/step", "ms/step"]
    print("| " + " | ".join(headers) + " |")
    print("|---|" + "|".join("---:" if i > 0 else "---"
                              for i in range(len(headers) - 1)) + "|")
    interp_map = {
        1: "single-row decode",
        2: "MTP verify (b=2)",
    }
    for gx in sorted(by_grid):
        durs = sorted(by_grid[gx])
        mean = sum(durs) / len(durs)
        total_ms = sum(durs) / 1000.0
        p50 = durs[len(durs) // 2]
        p99 = durs[max(0, int(len(durs) * 0.99) - 1)]
        interp = interp_map.get(gx, f"batched (b={gx})")
        values = [
            str(gx),
            interp,
            str(len(durs)),
            f"{mean:.1f}",
            f"{p50:.1f}",
            f"{p99:.1f}",
            f"{total_ms:.1f}",
        ]
        if decode_steps is not None:
            values += [
                f"{len(durs) / decode_steps:.2f}",
                f"{total_ms / decode_steps:.3f}",
            ]
        print("| " + " | ".join(values) + " |")
    print()


def _percentile(durs_sorted_us: list[float], q: float) -> float:
    if not durs_sorted_us:
        return 0.0
    idx = max(0, min(len(durs_sorted_us) - 1, int(round(q * (len(durs_sorted_us) - 1)))))
    return durs_sorted_us[idx]


def _stddev_us(durs_us: list[float]) -> float:
    if len(durs_us) < 2:
        return 0.0
    mean = sum(durs_us) / len(durs_us)
    var = sum((x - mean) ** 2 for x in durs_us) / (len(durs_us) - 1)
    return var ** 0.5


def print_hot_kernel_duration_distribution(
    conn: sqlite3.Connection,
    device: int,
    rows: list[KernelRow],
    wall_ns: int,
    mode: str,
    *,
    min_pct_wall: float = 1.0,
    max_categories: int = 8,
) -> None:
    """Distribution of per-call duration for hot categories.

    Helps diagnose chunked prefill skew (TileLang sparse prefill main
    spread over chunk sizes), MTP-verify shape mixing (sparse_decode
    bimodal), and network jitter (NCCL AllReduce p99 long tail).
    """
    cat_total: dict[str, int] = {}
    for row in rows:
        cat = classify(row, mode)
        cat_total[cat] = cat_total.get(cat, 0) + row.total_ns
    hot_cats = [
        cat
        for cat, ns in cat_total.items()
        if 100.0 * ns / wall_ns >= min_pct_wall
    ]
    hot_cats.sort(key=lambda c: -cat_total[c])
    hot_cats = hot_cats[:max_categories]
    if not hot_cats:
        return

    # Build a single name->category lookup so we can attribute per-launch
    # durations cheaply via the wide kernel rows.
    name_to_cat: dict[str, str] = {}
    for row in rows:
        name_to_cat[row.name] = classify(row, mode)

    name_expr = _kernel_name_expr()
    cur = conn.execute(
        f"""
        SELECT {name_expr} AS name, k.end - k.start AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds short ON short.id = k.shortName
        LEFT JOIN StringIds dem ON dem.id = k.demangledName
        WHERE k.deviceId = ?
        """,
        (device,),
    )
    durs_by_cat: dict[str, list[float]] = {c: [] for c in hot_cats}
    for name, dur in cur:
        cat = name_to_cat.get(str(name))
        if cat in durs_by_cat:
            durs_by_cat[cat].append(int(dur) / 1000.0)

    if not any(durs_by_cat.values()):
        return

    print(f"## Per-Kernel Duration Distribution (categories ≥ {min_pct_wall:g}% of wall)")
    print()
    print(
        "Reveals per-launch skew that the mean/total table cannot show: "
        "chunked prefill chunks of different sizes, MTP-verify b=1 vs b=2, "
        "and NCCL p99 jitter."
    )
    print()
    print(
        "| category | calls | mean us | stddev us | min | p50 | p90 | "
        "p99 | max |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    jitter_flags: list[str] = []
    for cat in hot_cats:
        durs = durs_by_cat.get(cat) or []
        if not durs:
            continue
        durs_sorted = sorted(durs)
        mean = sum(durs) / len(durs)
        std = _stddev_us(durs)
        p50 = _percentile(durs_sorted, 0.50)
        p99 = _percentile(durs_sorted, 0.99)
        print(
            f"| {cat} | {len(durs)} | {mean:.1f} | {std:.1f} | "
            f"{durs_sorted[0]:.1f} | "
            f"{p50:.1f} | "
            f"{_percentile(durs_sorted, 0.90):.1f} | "
            f"{p99:.1f} | "
            f"{durs_sorted[-1]:.1f} |"
        )
        # Flag categories where p99 / p50 >= 1.5 — these are the per-call
        # outliers that drag the critical-rank wall time and are usually
        # the right place to look for kernel-level skew or NCCL stragglers.
        if p50 > 0 and p99 / p50 >= 1.5:
            jitter_flags.append(f"`{cat}` p99/p50={p99/p50:.2f}")
    print()
    if jitter_flags:
        print(
            "**Per-call jitter flags (p99 / p50 ≥ 1.5×):** "
            + "; ".join(jitter_flags)
            + "."
        )
        print(
            "  These are the categories where the long tail dominates the "
            "per-step wall time more than the mean implies (e.g. NCCL "
            "stragglers, MTP-shape mixing, chunked-prefill skew)."
        )
        print()



def summarize_sqlite(
    sqlite_path: Path,
    *,
    requested_mode: str,
    device_request: str,
    layers: int,
    prompt_tokens: int | None,
    decode_steps_arg: str | None,
    top: int,
    analysis_level: str,
    duration_min_pct_wall: float,
    duration_max_categories: int,
) -> None:
    with connect(sqlite_path) as conn:
        wall_ns = capture_wall_ns(conn)
        summaries = device_summaries(conn)
        if not summaries:
            raise SystemExit("No CUPTI kernels found in SQLite DB")
        device = select_device(summaries, device_request)
        selected_busy = next(item[2] for item in summaries if item[0] == device)
        rows = load_kernel_rows(conn, device)

        mode, mode_source = resolve_mode(
            requested_mode, rows, prompt_tokens, decode_steps_arg
        )
        decode_info = (
            resolve_decode_steps(rows, layers, decode_steps_arg)
            if mode == "decode"
            else DecodeStepInfo(None, "not decode mode")
        )

        title_mode = "Decode" if mode == "decode" else "Prefill"
        print(f"# {title_mode} Nsight Kernel Summary")
        print()
        print(f"- SQLite: `{sqlite_path}`")
        print(f"- Mode: `{mode}` ({mode_source})")
        print(f"- Capture wall: {wall_ns/1e6:.1f} ms")
        print(f"- Selected device: {device}")
        print(
            "- Denominator: selected critical-rank wall time; TP ranks run "
            "concurrently, so NCCL/device times are not summed across ranks."
        )
        print(
            f"- Selected GPU busy: {selected_busy/1e6:.1f} ms "
            f"({100.0*selected_busy/wall_ns:.1f}% wall)"
        )
        print(
            f"- Idle/bubble on selected device: "
            f"{(wall_ns-selected_busy)/1e6:.1f} ms"
        )
        if prompt_tokens:
            print(
                f"- Prompt tokens: {prompt_tokens}; "
                f"wall per prompt token: {wall_ns/prompt_tokens/1e3:.1f} us"
            )
        if mode == "decode":
            print(f"- Decode step estimate: {decode_info.source}")
            if decode_info.steps:
                print(
                    f"- Decode steps: {_fmt_float(decode_info.steps, 2)}; "
                    f"wall per step: "
                    f"{wall_ns/decode_info.steps/1e6:.3f} ms; "
                    f"busy per step: "
                    f"{selected_busy/decode_info.steps/1e6:.3f} ms"
                )
        print()
        print_device_table(summaries, wall_ns, device)
        print_decode_step_rollup(rows, wall_ns, decode_info.steps, mode)
        cat_rows = print_category_table(
            rows,
            wall_ns,
            layers,
            prompt_tokens,
            decode_info.steps,
            mode,
        )
        print_stage_summary(
            cat_rows,
            wall_ns,
            layers,
            decode_info.steps,
        )
        if analysis_level == "full":
            print_idle_gap_summary(conn, device, wall_ns)
            ordered_rows = (
                _fetch_kernels_in_time_order(conn, device)
                if mode == "decode"
                else None
            )
            print_mtp_timeline_split(conn, device, mode, ordered_rows)
            if mode == "decode":
                print_sparse_decode_shape_split(
                    conn, device, decode_info.steps, ordered_rows
                )
            print_hot_kernel_duration_distribution(
                conn,
                device,
                rows,
                wall_ns,
                mode,
                min_pct_wall=duration_min_pct_wall,
                max_categories=duration_max_categories,
            )
        print_hot_kernel_internal_breakdown(
            cat_rows,
            layers,
            decode_info.steps,
            mode,
        )
        print_top_kernels(cat_rows, top)


def _strip_known_report_suffix(out: Path) -> str:
    text = str(out)
    for suffix in (".nsys-rep", ".sqlite"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _report_paths(out: Path) -> tuple[str, Path, Path]:
    base = _strip_known_report_suffix(out)
    return base, Path(base + ".nsys-rep"), Path(base + ".sqlite")


def _which_nsys(nsys: str) -> str:
    if os.path.sep in nsys:
        path = Path(nsys)
        if path.exists():
            return str(path)
        raise SystemExit(f"nsys not found: {nsys}")
    found = shutil.which(nsys)
    if found:
        return found
    default = Path("/usr/local/cuda-12.8/bin/nsys")
    if default.exists():
        return str(default)
    raise SystemExit("nsys not found in PATH; pass --nsys /path/to/nsys")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def _run_capture(
    cmd: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    proc.check_returncode()
    return proc


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _emit_summary(
    sqlite_path: Path,
    *,
    requested_mode: str,
    device_request: str,
    layers: int,
    prompt_tokens: int | None,
    decode_steps_arg: str | None,
    top: int,
    analysis_level: str,
    duration_min_pct_wall: float,
    duration_max_categories: int,
    summary_out: Path | None,
) -> None:
    if summary_out is None:
        summarize_sqlite(
            sqlite_path,
            requested_mode=requested_mode,
            device_request=device_request,
            layers=layers,
            prompt_tokens=prompt_tokens,
            decode_steps_arg=decode_steps_arg,
            top=top,
            analysis_level=analysis_level,
            duration_min_pct_wall=duration_min_pct_wall,
            duration_max_categories=duration_max_categories,
        )
        return

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as out:
        with contextlib.redirect_stdout(_Tee(sys.stdout, out)):
            summarize_sqlite(
                sqlite_path,
                requested_mode=requested_mode,
                device_request=device_request,
                layers=layers,
                prompt_tokens=prompt_tokens,
                decode_steps_arg=decode_steps_arg,
                top=top,
                analysis_level=analysis_level,
                duration_min_pct_wall=duration_min_pct_wall,
                duration_max_categories=duration_max_categories,
            )
    print(f"Summary written: {summary_out}", file=sys.stderr)


def _extract_prefill_prompt_tokens(stdout: str) -> int | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("prompt_tokens"), int):
            return int(payload["prompt_tokens"])
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            for item in payload["results"]:
                if isinstance(item, dict) and isinstance(item.get("prompt_tokens"), int):
                    return int(item["prompt_tokens"])
    return None


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    parts = Path(__file__).resolve().parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        if idx + 2 < len(parts):
            return Path(*parts[:idx])
    return _tool_root()


def _default_measurements_dir() -> Path:
    repo_root = _repo_root()
    candidate = (
        repo_root
        / ".kiro/specs/deepseek-v4-flash-flashmla-sparse-internals/measurements"
    )
    if candidate.exists():
        return candidate
    return _tool_root() / "measurements"


def _default_python() -> str:
    return DEFAULT_CONDA_PYTHON if Path(DEFAULT_CONDA_PYTHON).exists() else sys.executable


def _prepend_path(env: dict[str, str], value: str) -> None:
    current = env.get("PATH", "")
    if current:
        env["PATH"] = value + os.pathsep + current
    else:
        env["PATH"] = value


def _service_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if Path(DEFAULT_CONDA_BIN).exists():
        _prepend_path(env, DEFAULT_CONDA_BIN)
    if extra:
        env.update(extra)
    return env


def _set_env_if_not_none(
    env: dict[str, str], key: str, value: int | str | None
) -> None:
    if value is not None:
        env[key] = str(value)


def _interesting_vllm_env(env: dict[str, str]) -> dict[str, str]:
    prefixes = (
        "VLLM_DEEPSEEK_V4_",
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_",
        "VLLM_SM70_DEEPSEEK_V4_",
        "VLLM_SPARSE_INDEXER_",
        "VLLM_MOE_",
    )
    return {
        key: env[key]
        for key in sorted(env)
        if key.startswith(prefixes)
    }


def _write_collect_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)


def _append_collect_config_summary(summary_path: Path, config: dict) -> None:
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n## Formal Collect Config\n\n")
        fh.write(f"- Collect mode: `{config.get('mode')}`\n")
        fh.write(f"- Model: `{config.get('model')}`\n")
        fh.write(
            "- Server CUDA_VISIBLE_DEVICES: "
            f"`{config.get('server_cuda_visible_devices')}`\n"
        )
        if config.get("mode") == "prefill":
            fh.write(
                "- fastllm chunked_prefill_size reference: "
                f"`{config.get('fastllm_chunked_prefill_size')}`\n"
            )
            fh.write(
                "- vLLM --max-num-batched-tokens: "
                f"`{config.get('max_num_batched_tokens')}`\n"
            )
            fh.write(
                "- fastllm-aligned chunk applied: "
                f"`{config.get('fastllm_aligned_prefill_chunk')}`\n"
            )
            fh.write(f"- Profile length request: `{config.get('profile_length')}`\n")
        else:
            fh.write(f"- Decode indexer topk: `{config.get('indexer_topk')}`\n")
            fh.write(
                "- Capture duration seconds: "
                f"`{config.get('capture_duration_s')}`\n"
            )
        env_overrides = config.get("env_overrides") or {}
        if env_overrides:
            fh.write("\n| env | value |\n")
            fh.write("|---|---|\n")
            for key, value in sorted(env_overrides.items()):
                fh.write(f"| `{key}` | `{value}` |\n")
        fh.write(f"\n- Collect config JSON: `{config.get('config_path')}`\n")


def _http_get_status(url: str, timeout_s: float = 2.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return int(resp.status)
    except Exception:
        return 0


def _http_post(url: str, timeout_s: float = 5.0) -> int:
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return 0


def _tail_file(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(data[-lines:])


def _failure_hint_from_log(tail: str) -> str:
    lower = tail.lower()
    hints: list[str] = []
    if "address already in use" in lower or "errno 98" in lower:
        hints.append("port appears to be in use; choose another --port or stop the old server")
    if "cuda out of memory" in lower or "outofmemoryerror" in lower:
        hints.append("CUDA OOM; lower --gpu-memory-utilization/--max-model-len or free GPUs")
    if "no available memory for the cache blocks" in lower or "kv cache" in lower and "larger than" in lower:
        hints.append("KV cache capacity is too small for the requested max model length / sequence count")
    if "deepgemm" in lower and ("sm70" in lower or "capability" in lower):
        hints.append("non-SM70/DeepGEMM path may have been selected; verify the CUDA mask is pure V100")
    if "compute capability" in lower and ("7.0" in lower or "sm70" in lower):
        hints.append("GPU capability mismatch; formal SM70 capture expects only compute capability 7.0 GPUs")
    if "traceback" in lower:
        hints.append("Python traceback found in server log; inspect the tail below for the fatal frame")
    if not hints:
        return ""
    return "\n".join(f"- {hint}" for hint in hints)


def _process_table() -> list[tuple[int, int, str]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,command="],
        text=True,
        capture_output=True,
        check=True,
    )
    rows: list[tuple[int, int, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def _process_ids() -> set[int]:
    return {pid for pid, _, _ in _process_table()}


def _parse_visible_device_indices(spec: str) -> list[int]:
    indices: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            indices.append(int(item))
        except ValueError:
            # UUID-style CUDA masks cannot be checked with the simple
            # nvidia-smi index query below, so leave them to CUDA.
            return []
    return indices


def _gpu_compute_caps() -> dict[int, str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return {}
    caps: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        try:
            caps[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return caps


def _validate_sm70_devices(spec: str, *, skip: bool) -> None:
    if skip:
        return
    indices = _parse_visible_device_indices(spec)
    if not indices:
        return
    caps = _gpu_compute_caps()
    if not caps:
        return
    bad = [(idx, caps.get(idx, "missing")) for idx in indices if caps.get(idx) != "7.0"]
    if bad:
        details = ", ".join(f"{idx}:{cap}" for idx, cap in bad)
        raise SystemExit(
            "DeepSeek V4 SM70 formal capture requires all visible GPUs to be "
            f"compute capability 7.0; bad entries in CUDA_VISIBLE_DEVICES: {details}. "
            "Pass a pure V100 mask or --skip-sm70-device-check."
        )


def _looks_like_vllm_capture_process(cmd: str, port: int, base: str) -> bool:
    port_arg = f"--port {port}"
    return (
        port_arg in cmd
        or base in cmd
        or "VLLM::EngineCore" in cmd
        or "VLLM::Worker_TP" in cmd
        or "vllm.entrypoints.openai.api_server" in cmd
        or " vllm serve " in f" {cmd} "
    )


def _descendants(roots: set[int], rows: list[tuple[int, int, str]]) -> set[int]:
    by_parent: dict[int, list[int]] = defaultdict(list)
    for pid, ppid, _ in rows:
        by_parent[ppid].append(pid)

    found: set[int] = set()
    stack = list(roots)
    while stack:
        parent = stack.pop()
        for child in by_parent.get(parent, []):
            if child in found:
                continue
            found.add(child)
            stack.append(child)
    return found


def _kill_pids(pids: set[int], sig: signal.Signals) -> None:
    current = os.getpid()
    for pid in sorted(pids):
        if pid == current:
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _cleanup_new_capture_processes(
    before_pids: set[int],
    *,
    port: int,
    base: str,
    grace_s: float = 15.0,
) -> None:
    rows = _process_table()
    roots = {
        pid
        for pid, _, cmd in rows
        if pid not in before_pids and _looks_like_vllm_capture_process(cmd, port, base)
    }
    targets = roots | _descendants(roots, rows)
    if not targets:
        return

    _kill_pids(targets, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        live = _process_ids()
        if not (targets & live):
            return
        time.sleep(0.5)
    _kill_pids(targets & _process_ids(), signal.SIGKILL)


def _terminate_group(proc: subprocess.Popen[str] | None, grace_s: float = 10.0) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _run_curl_json(
    url: str,
    payload: dict[str, object],
    *,
    stream: bool = False,
    stdout_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str] | subprocess.CompletedProcess[str]:
    # Write the JSON body to a temp file so very large prompts (>~128KB)
    # do not blow up the curl argv. Linux's `getconf ARG_MAX` is typically
    # 2 MiB, but the practical exec ceiling is much lower once env is
    # included; long-context decay traces routinely cross it.
    import tempfile
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    tmp = tempfile.NamedTemporaryFile(
        prefix="dsv4_curl_", suffix=".json", delete=False
    )
    try:
        tmp.write(body)
        tmp.close()
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        raise
    cmd = [
        "curl",
        "-s",
        "-N" if stream else "-s",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        f"@{tmp.name}",
    ]
    if stream:
        out = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.DEVNULL
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                start_new_session=True,
            )
            if stdout_path:
                out.close()
            # Stash the temp-file path on the process so the caller
            # (or our process-cleanup helpers) can delete it after
            # curl is fully done. Deleting earlier races with curl's
            # `--data-binary @file` open() on the body.
            proc._dsv4_body_tempfile = tmp.name  # type: ignore[attr-defined]
            return proc
        except Exception:
            if stdout_path and not out.closed:
                out.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
    try:
        return subprocess.run(cmd, check=True, env=env, text=True, capture_output=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _wait_for_server(
    port: int,
    log_path: Path,
    timeout_s: int = 900,
    server: subprocess.Popen[str] | None = None,
) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _http_get_status(url, timeout_s=2.0) == 200:
            return
        if server is not None and server.poll() is not None:
            tail = _tail_file(log_path)
            hint = _failure_hint_from_log(tail)
            hint_block = f"\nLikely cause(s):\n{hint}\n" if hint else ""
            raise SystemExit(
                f"server exited before becoming ready on port {port} "
                f"(returncode={server.returncode});{hint_block}"
                f"last log lines:\n{tail}"
            )
        time.sleep(5)
    tail = _tail_file(log_path)
    hint = _failure_hint_from_log(tail)
    hint_block = f"\nLikely cause(s):\n{hint}\n" if hint else ""
    raise SystemExit(
        f"server never became ready on port {port};{hint_block}last log lines:\n{tail}"
    )


def _wait_for_path(path: Path, timeout_s: float, description: str) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(1.0)
    raise SystemExit(f"{description} missing after waiting {timeout_s:.0f}s: {path}")


def _make_out_base(
    out: Path | None,
    out_dir: Path,
    prefix: str,
    suffix: str = "",
) -> Path:
    if out is not None:
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{prefix}{suffix}_{stamp}"


def run_profile(args: argparse.Namespace) -> None:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("profile requires a command after '--'")

    nsys = _which_nsys(args.nsys)
    base, rep_path, sqlite_path = _report_paths(args.out)
    Path(base).parent.mkdir(parents=True, exist_ok=True)

    profile_cmd = [
        nsys,
        "profile",
        f"--trace={args.trace}",
        f"--sample={args.sample}",
        f"--cuda-memory-usage={args.cuda_memory_usage}",
        "--force-overwrite=true",
        "-o",
        base,
    ]
    if args.capture_range:
        profile_cmd.append(f"--capture-range={args.capture_range}")
    if args.capture_range_end:
        profile_cmd.append(f"--capture-range-end={args.capture_range_end}")
    if args.cuda_graph_trace:
        profile_cmd.append(f"--cuda-graph-trace={args.cuda_graph_trace}")
    for item in args.nsys_arg or []:
        profile_cmd.append(item)
    profile_cmd.extend(command)

    _run(profile_cmd)
    if not rep_path.exists():
        raise SystemExit(f"nsys profile completed but report is missing: {rep_path}")

    export_cmd = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(rep_path),
    ]
    _run(export_cmd)
    if not sqlite_path.exists():
        raise SystemExit(f"nsys export completed but SQLite is missing: {sqlite_path}")

    if args.no_summary:
        print(f"SQLite written: {sqlite_path}")
        return

    print()
    _emit_summary(
        sqlite_path,
        requested_mode=args.mode,
        device_request=args.device,
        layers=args.layers,
        prompt_tokens=args.prompt_tokens,
        decode_steps_arg=args.decode_steps,
        top=args.top,
        analysis_level=args.analysis_level,
        duration_min_pct_wall=args.duration_min_pct_wall,
        duration_max_categories=args.duration_max_categories,
        summary_out=args.summary_out,
    )


def _export_and_summarize(
    nsys: str,
    rep_path: Path,
    sqlite_path: Path,
    *,
    requested_mode: str,
    device_request: str,
    layers: int,
    prompt_tokens: int | None,
    decode_steps_arg: str | None,
    top: int,
    analysis_level: str,
    duration_min_pct_wall: float,
    duration_max_categories: int,
    summary_out: Path | None,
    no_summary: bool,
) -> None:
    export_cmd = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(rep_path),
    ]
    _run(export_cmd)
    if not sqlite_path.exists():
        raise SystemExit(f"nsys export completed but SQLite is missing: {sqlite_path}")
    if no_summary:
        print(f"SQLite written: {sqlite_path}")
        return
    print()
    _emit_summary(
        sqlite_path,
        requested_mode=requested_mode,
        device_request=device_request,
        layers=layers,
        prompt_tokens=prompt_tokens,
        decode_steps_arg=decode_steps_arg,
        top=top,
        analysis_level=analysis_level,
        duration_min_pct_wall=duration_min_pct_wall,
        duration_max_categories=duration_max_categories,
        summary_out=summary_out,
    )


def run_collect_prefill(args: argparse.Namespace) -> None:
    nsys = _which_nsys(args.nsys)
    _validate_sm70_devices(args.server_cuda_visible_devices, skip=args.skip_sm70_device_check)
    out_base = _make_out_base(
        args.out,
        args.out_dir,
        "h58_prefill_formal",
        f"_{args.profile_length}",
    )
    base, rep_path, sqlite_path = _report_paths(out_base)
    Path(base).parent.mkdir(parents=True, exist_ok=True)

    log_path = args.server_log or Path(base + "_server.log")
    harness = _tool_root() / "tests/benchmarks/deepseek_v4_sparse_prefill_v2_lengths.py"
    if not harness.exists():
        raise SystemExit(f"prefill lengths harness not found: {harness}")

    max_num_batched_tokens = args.max_num_batched_tokens
    if args.fastllm_aligned_prefill_chunk:
        max_num_batched_tokens = args.fastllm_chunked_prefill_size
    if max_num_batched_tokens <= 0:
        raise SystemExit("--max-num-batched-tokens must be positive")
    if args.fastllm_chunked_prefill_size <= 0:
        raise SystemExit("--fastllm-chunked-prefill-size must be positive")
    env = _service_env(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": args.server_cuda_visible_devices,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK": "1",
            "VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK_DEBUG_COMPARE": "0",
            "VLLM_DEEPSEEK_V4_PROFILE_NVTX": "1",
            "TORCHINDUCTOR_CACHE_DIR": args.torchinductor_cache_dir,
            "TRITON_CACHE_DIR": args.triton_cache_dir,
        }
    )
    _set_env_if_not_none(
        env,
        "VLLM_DEEPSEEK_V4_PREFILL_CHUNK_SIZE",
        args.prefill_request_chunk_size,
    )
    _set_env_if_not_none(
        env,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_PREWARM_MAX_CONTEXT",
        args.tilelang_prefill_prewarm_max_context,
    )
    _set_env_if_not_none(
        env,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_BI",
        args.tilelang_prefill_bi,
    )
    _set_env_if_not_none(
        env,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_STAGES",
        args.tilelang_prefill_stages,
    )
    _set_env_if_not_none(
        env,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_HEADS_PER_BLOCK",
        args.tilelang_prefill_heads_per_block,
    )
    _set_env_if_not_none(
        env,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_THREADS",
        args.tilelang_prefill_threads,
    )
    _set_env_if_not_none(
        env,
        "VLLM_SM70_TILELANG_SPARSE_PREFILL_PV_POLICY",
        args.tilelang_prefill_pv_policy,
    )
    if args.tilelang_prefill_assume_valid_indices:
        env["VLLM_SM70_TILELANG_SPARSE_PREFILL_ASSUME_VALID_INDICES"] = "1"
    if args.disable_tilelang_prefill_fast_io:
        env["VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO"] = "0"
    env.pop("VLLM_DEEPSEEK_V4_INDEXER_TOPK", None)
    env["PYTHONPATH"] = str(_tool_root()) + os.pathsep + env.get("PYTHONPATH", "")
    before_pids = _process_ids()

    compilation_config = json.dumps(
        {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8]},
        separators=(",", ":"),
    )
    service_cmd = [
        "vllm",
        "serve",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--trust-remote-code",
        "--tokenizer-mode",
        "deepseek_v4",
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_v4",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--kv-cache-dtype",
        "fp8_ds_mla",
        "--port",
        str(args.port),
        "--compilation-config",
        compilation_config,
        "--profiler-config.profiler=cuda",
    ]
    if args.no_enable_prefix_caching:
        service_cmd.append("--no-enable-prefix-caching")

    collect_config_path = Path(base + "_collect_config.json")
    collect_config = {
        "mode": "prefill",
        "model": args.model,
        "server_cuda_visible_devices": args.server_cuda_visible_devices,
        "client_cuda_visible_devices": args.client_cuda_visible_devices,
        "port": args.port,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "fastllm_chunked_prefill_size": args.fastllm_chunked_prefill_size,
        "fastllm_aligned_prefill_chunk": args.fastllm_aligned_prefill_chunk,
        "profile_length": args.profile_length,
        "warmup_length": args.warmup_length,
        "max_tokens": args.max_tokens,
        "compilation_config": json.loads(compilation_config),
        "no_enable_prefix_caching": args.no_enable_prefix_caching,
        "env_overrides": _interesting_vllm_env(env),
        "service_cmd": service_cmd,
        "config_path": str(collect_config_path),
    }
    _write_collect_config(collect_config_path, collect_config)

    profile_cmd = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cuda-memory-usage=false",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        base,
        *service_cmd,
    ]

    print(f"OUT_REP={base}")
    print(f"SERVER_LOG={log_path}")
    print(f"COLLECT_CONFIG={collect_config_path}")
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"PROFILE_LENGTH={args.profile_length}")
    print(f"FASTLLM_CHUNKED_PREFILL_SIZE_REF={args.fastllm_chunked_prefill_size}")
    print(f"MAX_NUM_BATCHED_TOKENS={max_num_batched_tokens}")
    server: subprocess.Popen[str] | None = None
    measured_prompt_tokens: int | None = None
    try:
        with open(log_path, "w") as log:
            print("+ " + shlex.join(profile_cmd), file=sys.stderr)
            server = subprocess.Popen(
                profile_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd="/tmp",
                text=True,
                start_new_session=True,
            )
        print(f"nsys+server pid={server.pid}; waiting for ready...")
        _wait_for_server(
            args.port,
            log_path,
            timeout_s=args.ready_timeout_s,
            server=server,
        )
        print("server ready")

        client_env = env.copy()
        client_env["CUDA_VISIBLE_DEVICES"] = args.client_cuda_visible_devices
        base_url = f"http://127.0.0.1:{args.port}"
        warmup_cmd = [
            args.python,
            str(harness),
            base_url,
            args.model,
            str(args.warmup_length),
        ]
        _run(warmup_cmd, env=client_env)

        status = _http_post(f"{base_url}/start_profile")
        print(f"start_profile http={status}")
        if status != 200:
            raise SystemExit(f"start_profile failed with HTTP {status}")

        profile_cmd_client = [
            args.python,
            str(harness),
            base_url,
            args.model,
            str(args.profile_length),
            "--stream",
            "--max-tokens",
            str(args.max_tokens),
        ]
        profile_result = _run_capture(profile_cmd_client, env=client_env)
        measured_prompt_tokens = _extract_prefill_prompt_tokens(profile_result.stdout)

        status = _http_post(
            f"{base_url}/stop_profile",
            timeout_s=args.profile_stop_timeout_s,
        )
        print(f"stop_profile http={status}")
        if status != 200:
            raise SystemExit(f"stop_profile failed with HTTP {status}")
        time.sleep(args.post_stop_sleep_s)
    finally:
        _terminate_group(server, grace_s=15)
        _cleanup_new_capture_processes(
            before_pids,
            port=args.port,
            base=base,
            grace_s=15.0,
        )

    _wait_for_path(rep_path, args.report_ready_timeout_s, "nsys report")
    summary_path = args.summary_out or Path(base + "_summary.md")
    _export_and_summarize(
        nsys,
        rep_path,
        sqlite_path,
        requested_mode=args.mode,
        device_request=args.device,
        layers=args.layers,
        prompt_tokens=args.prompt_tokens or measured_prompt_tokens or args.profile_length,
        decode_steps_arg=args.decode_steps,
        top=args.top,
        analysis_level=args.analysis_level,
        duration_min_pct_wall=args.duration_min_pct_wall,
        duration_max_categories=args.duration_max_categories,
        summary_out=summary_path,
        no_summary=args.no_summary,
    )
    if not args.no_summary:
        _append_collect_config_summary(summary_path, collect_config)


def _resolve_prompt(args: argparse.Namespace) -> str:
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8")
    return args.prompt


def run_collect_decode(args: argparse.Namespace) -> None:
    nsys = _which_nsys(args.nsys)
    _validate_sm70_devices(args.server_cuda_visible_devices, skip=args.skip_sm70_device_check)
    out_base = _make_out_base(args.out, args.out_dir, "h58_decode_formal")
    base, rep_path, sqlite_path = _report_paths(out_base)
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    log_path = args.server_log or Path(base + "_server.log")
    probe_out = args.probe_out or Path(base + "_probe.txt")

    env = _service_env(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": args.server_cuda_visible_devices,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_DEEPSEEK_V4_INDEXER_TOPK": str(args.indexer_topk),
            "VLLM_MOE_EARLY_SHARED_EXPERTS_STREAM": "1",
            "TORCHINDUCTOR_CACHE_DIR": args.torchinductor_cache_dir,
            "TRITON_CACHE_DIR": args.triton_cache_dir,
        }
    )
    if args.copy_source_trace:
        env["VLLM_DEEPSEEK_V4_COPY_SOURCE_TRACE"] = "1"
    env["PYTHONPATH"] = str(_tool_root()) + os.pathsep + env.get("PYTHONPATH", "")
    before_pids = _process_ids()

    compilation_config = json.dumps(
        {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1]},
        separators=(",", ":"),
    )
    service_cmd = [
        "vllm",
        "serve",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--dtype",
        args.dtype,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--kv-cache-dtype",
        "fp8_ds_mla",
        "--port",
        str(args.port),
        "--compilation-config",
        compilation_config,
        "--profiler-config.profiler=cuda",
    ]
    if args.no_enable_prefix_caching:
        service_cmd.append("--no-enable-prefix-caching")

    collect_config_path = Path(base + "_collect_config.json")
    collect_config = {
        "mode": "decode",
        "model": args.model,
        "server_cuda_visible_devices": args.server_cuda_visible_devices,
        "client_cuda_visible_devices": args.server_cuda_visible_devices,
        "port": args.port,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "indexer_topk": args.indexer_topk,
        "warmup_count": args.warmup_count,
        "warmup_max_tokens": args.warmup_max_tokens,
        "max_tokens": args.max_tokens,
        "capture_delay_s": args.capture_delay_s,
        "capture_duration_s": args.capture_duration_s,
        "cuda_graph_trace": args.cuda_graph_trace,
        "no_enable_prefix_caching": args.no_enable_prefix_caching,
        "copy_source_trace": args.copy_source_trace,
        "env_overrides": _interesting_vllm_env(env),
        "service_cmd": service_cmd,
        "config_path": str(collect_config_path),
    }
    _write_collect_config(collect_config_path, collect_config)

    profile_cmd = [
        nsys,
        "profile",
        "--trace=cuda,osrt",
        f"--cuda-graph-trace={args.cuda_graph_trace}",
        "--sample=none",
        "--cuda-memory-usage=false",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "-o",
        base,
        *service_cmd,
    ]

    print(f"OUT_REP={base}")
    print(f"SERVER_LOG={log_path}")
    print(f"PROBE_OUT={probe_out}")
    print(f"COLLECT_CONFIG={collect_config_path}")
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    server: subprocess.Popen[str] | None = None
    probe: subprocess.Popen[str] | None = None
    try:
        with open(log_path, "w") as log:
            print("+ " + shlex.join(profile_cmd), file=sys.stderr)
            server = subprocess.Popen(
                profile_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd="/tmp",
                text=True,
                start_new_session=True,
            )
        print(f"nsys+server pid={server.pid}; waiting for ready...")
        _wait_for_server(
            args.port,
            log_path,
            timeout_s=args.ready_timeout_s,
            server=server,
        )
        print("server ready")

        url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
        for idx in range(args.warmup_count):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": f"warm {idx + 1}"}],
                "max_tokens": args.warmup_max_tokens,
                "temperature": 0,
                "stream": True,
            }
            _run_curl_json(url, payload, stream=False, env=env)
        print("warmup complete")

        long_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": _resolve_prompt(args)}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
        }
        probe = _run_curl_json(
            url,
            long_payload,
            stream=True,
            stdout_path=probe_out,
            env=env,
        )
        time.sleep(args.capture_delay_s)
        status = _http_post(f"http://127.0.0.1:{args.port}/start_profile")
        print(f"start_profile http={status}")
        if status != 200:
            raise SystemExit(f"start_profile failed with HTTP {status}")
        time.sleep(args.capture_duration_s)
        status = _http_post(
            f"http://127.0.0.1:{args.port}/stop_profile",
            timeout_s=args.profile_stop_timeout_s,
        )
        print(f"stop_profile http={status}")
        if status != 200:
            raise SystemExit(f"stop_profile failed with HTTP {status}")
        _terminate_group(probe, grace_s=2)
        probe = None
        time.sleep(args.post_stop_sleep_s)
    finally:
        _terminate_group(probe, grace_s=2)
        _terminate_group(server, grace_s=15)
        _cleanup_new_capture_processes(
            before_pids,
            port=args.port,
            base=base,
            grace_s=15.0,
        )

    _wait_for_path(rep_path, args.report_ready_timeout_s, "nsys report")
    summary_path = args.summary_out or Path(base + "_summary.md")
    _export_and_summarize(
        nsys,
        rep_path,
        sqlite_path,
        requested_mode=args.mode,
        device_request=args.device,
        layers=args.layers,
        prompt_tokens=args.prompt_tokens,
        decode_steps_arg=args.decode_steps or "auto",
        top=args.top,
        analysis_level=args.analysis_level,
        duration_min_pct_wall=args.duration_min_pct_wall,
        duration_max_categories=args.duration_max_categories,
        summary_out=summary_path,
        no_summary=args.no_summary,
    )
    if not args.no_summary:
        _append_collect_config_summary(summary_path, collect_config)


def add_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="auto",
        help="Summary mode. 'auto' chooses from selected-device kernels.",
    )
    parser.add_argument(
        "--device",
        default="critical",
        help="Device id to summarize, or 'critical' for max busy time (default)",
    )
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="Optional prompt token count for prefill per-token timing columns",
    )
    parser.add_argument(
        "--decode-steps",
        default=None,
        help=(
            "Decode steps for per-step timing, or 'auto' to infer from "
            "FlashMLA sparse decode kernel count"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of categories to show in the top-kernel drilldown",
    )
    parser.add_argument(
        "--analysis-level",
        choices=ANALYSIS_LEVELS,
        default="full",
        help=(
            "Summary depth. 'basic' prints device/category/stage/top-kernel "
            "tables; 'full' also scans launch order for idle gaps, MTP splits, "
            "shape splits, and per-launch duration distributions."
        ),
    )
    parser.add_argument(
        "--duration-min-pct-wall",
        type=float,
        default=1.0,
        help="Minimum %%wall for categories included in the full duration distribution.",
    )
    parser.add_argument(
        "--duration-max-categories",
        type=int,
        default=8,
        help="Maximum categories included in the full duration distribution.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help=(
            "Optional Markdown summary output. collect-* defaults to "
            "<out>_summary.md when summary is enabled."
        ),
    )


def build_summarize_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Summarize DeepSeek V4 prefill/decode kernels from an "
            "nsys-exported SQLite database."
        ),
    )
    parser.add_argument("sqlite", type=Path, help="Path to nsys-exported SQLite DB")
    add_summary_args(parser)
    return parser


def build_profile_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run nsys profile, export SQLite, then summarize it.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output basename, .nsys-rep path, or .sqlite path",
    )
    parser.add_argument(
        "--nsys",
        default=os.environ.get("NSYS", "nsys"),
        help="nsys executable path/name (default: $NSYS or nsys)",
    )
    parser.add_argument(
        "--trace",
        default="cuda,nvtx",
        help="nsys --trace value (default: cuda,nvtx)",
    )
    parser.add_argument(
        "--sample",
        default="none",
        help="nsys --sample value (default: none)",
    )
    parser.add_argument(
        "--cuda-memory-usage",
        default="false",
        help="nsys --cuda-memory-usage value (default: false)",
    )
    parser.add_argument(
        "--capture-range",
        default=None,
        help="Optional nsys capture range, e.g. cudaProfilerApi",
    )
    parser.add_argument(
        "--capture-range-end",
        default=None,
        help="Optional nsys capture range end, e.g. stop",
    )
    parser.add_argument(
        "--cuda-graph-trace",
        default=None,
        help="Optional nsys CUDA graph trace mode, e.g. graph or node",
    )
    parser.add_argument(
        "--nsys-arg",
        action="append",
        default=[],
        help="Extra raw argument appended to 'nsys profile' before the command",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Only profile/export; do not summarize the resulting SQLite",
    )
    add_summary_args(parser)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to profile. Put it after '--'.",
    )
    return parser


def add_collect_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_port: int,
    default_cuda_visible_devices: str,
    default_max_model_len: int,
) -> None:
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output basename, .nsys-rep path, or .sqlite path. Defaults to measurements/h58_* timestamp.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_default_measurements_dir(),
        help="Directory for auto-generated output basenames.",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="Server log path. Defaults to <out>_server.log.",
    )
    parser.add_argument(
        "--nsys",
        default=os.environ.get("NSYS", "nsys"),
        help="nsys executable path/name (default: $NSYS or nsys)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=default_max_model_len)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--server-cuda-visible-devices",
        default=default_cuda_visible_devices,
        help="CUDA_VISIBLE_DEVICES for the profiled vLLM server.",
    )
    parser.add_argument(
        "--skip-sm70-device-check",
        action="store_true",
        help="Skip the startup check that every visible GPU is compute capability 7.0.",
    )
    parser.add_argument(
        "--ready-timeout-s",
        type=int,
        default=900,
        help="Seconds to wait for /v1/models before failing.",
    )
    parser.add_argument("--post-stop-sleep-s", type=float, default=5.0)
    parser.add_argument(
        "--profile-stop-timeout-s",
        type=float,
        default=180.0,
        help="Seconds to wait for /stop_profile before failing.",
    )
    parser.add_argument(
        "--report-ready-timeout-s",
        type=float,
        default=240.0,
        help="Seconds to wait for nsys to write the .nsys-rep after capture.",
    )
    parser.add_argument(
        "--torchinductor-cache-dir",
        default="/home/z/.cache/torchinductor",
    )
    parser.add_argument("--triton-cache-dir", default="/home/z/.cache/triton")
    parser.add_argument(
        "--enable-prefix-caching",
        dest="no_enable_prefix_caching",
        action="store_false",
        help="Do not pass --no-enable-prefix-caching to vLLM.",
    )
    parser.set_defaults(no_enable_prefix_caching=True)
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Only capture/export; do not summarize the resulting SQLite.",
    )
    add_summary_args(parser)


def build_collect_prefill_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the formal DeepSeek V4 prefill nsys capture: start vLLM under "
            "nsys, warm up, bracket one prefill request with /start_profile and "
            "/stop_profile, export SQLite, then summarize it."
        ),
    )
    add_collect_common_args(
        parser,
        default_port=8299,
        default_cuda_visible_devices="3,5,4,6,7,8,9,10",
        default_max_model_len=524288,
    )
    parser.set_defaults(mode="prefill")
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--client-cuda-visible-devices", default="0")
    parser.add_argument("--profile-length", type=int, default=10000)
    parser.add_argument("--warmup-length", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--fastllm-chunked-prefill-size",
        type=int,
        default=FASTLLM_CHUNKED_PREFILL_SIZE,
        help=(
            "fastllm reference chunked_prefill_size recorded in collect "
            "metadata. Use --fastllm-aligned-prefill-chunk to apply it to "
            "vLLM --max-num-batched-tokens."
        ),
    )
    parser.add_argument(
        "--fastllm-aligned-prefill-chunk",
        action="store_true",
        help=(
            "Set vLLM --max-num-batched-tokens to "
            "--fastllm-chunked-prefill-size for the profiled run. This is an "
            "experiment knob; the safe formal default remains 4096."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help=(
            "vLLM --max-num-batched-tokens for the profiled server. "
            "Default: 4096, matching the previously successful formal "
            "prefill captures."
        ),
    )
    parser.add_argument(
        "--prefill-request-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional VLLM_DEEPSEEK_V4_PREFILL_CHUNK_SIZE override. "
            "This is request-count chunking inside _forward_prefill, not "
            "token chunking."
        ),
    )
    parser.add_argument("--tilelang-prefill-bi", type=int, default=None)
    parser.add_argument("--tilelang-prefill-stages", type=int, default=None)
    parser.add_argument("--tilelang-prefill-heads-per-block", type=int, default=None)
    parser.add_argument("--tilelang-prefill-threads", type=int, default=None)
    parser.add_argument("--tilelang-prefill-pv-policy", default=None)
    parser.add_argument(
        "--tilelang-prefill-prewarm-max-context",
        type=int,
        default=None,
        help=(
            "Optional VLLM_SM70_TILELANG_SPARSE_PREFILL_PREWARM_MAX_CONTEXT "
            "override for long-context JIT prewarm."
        ),
    )
    parser.add_argument(
        "--tilelang-prefill-assume-valid-indices",
        action="store_true",
        help="Set VLLM_SM70_TILELANG_SPARSE_PREFILL_ASSUME_VALID_INDICES=1.",
    )
    parser.add_argument(
        "--disable-tilelang-prefill-fast-io",
        action="store_true",
        help="Set VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO=0 for fallback A/B.",
    )
    return parser


def build_collect_decode_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the formal DeepSeek V4 decode nsys capture: start vLLM under "
            "nsys, warm up decode, capture a timed decode window with "
            "/start_profile and /stop_profile, export SQLite, then summarize it."
        ),
    )
    add_collect_common_args(
        parser,
        default_port=8199,
        default_cuda_visible_devices="3,5,4,6,7,8,9,10",
        default_max_model_len=4096,
    )
    parser.set_defaults(mode="decode", decode_steps="auto")
    parser.add_argument("--indexer-topk", type=int, default=256)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--warmup-max-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--capture-delay-s", type=float, default=5.0)
    parser.add_argument("--capture-duration-s", type=float, default=12.0)
    parser.add_argument("--cuda-graph-trace", default="node")
    parser.add_argument("--copy-source-trace", action="store_true")
    parser.add_argument(
        "--probe-out",
        type=Path,
        default=None,
        help="Streaming probe output path. Defaults to <out>_probe.txt.",
    )
    parser.add_argument(
        "--prompt",
        default="请详细描述大语言模型推理系统。",
        help="Prompt used for the long decode request.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help=(
            "Optional path; when set the file's UTF-8 contents replace "
            "--prompt. Useful for long-context decay measurements."
        ),
    )
    return parser


def run_summarize(args: argparse.Namespace) -> None:
    _emit_summary(
        args.sqlite,
        requested_mode=args.mode,
        device_request=args.device,
        layers=args.layers,
        prompt_tokens=args.prompt_tokens,
        decode_steps_arg=args.decode_steps,
        top=args.top,
        analysis_level=args.analysis_level,
        duration_min_pct_wall=args.duration_min_pct_wall,
        duration_max_categories=args.duration_max_categories,
        summary_out=args.summary_out,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    prog = Path(sys.argv[0]).name

    if argv and argv[0] == "profile":
        parser = build_profile_parser(f"{prog} profile")
        run_profile(parser.parse_args(argv[1:]))
        return 0

    if argv and argv[0] == "collect-prefill":
        parser = build_collect_prefill_parser(f"{prog} collect-prefill")
        run_collect_prefill(parser.parse_args(argv[1:]))
        return 0

    if argv and argv[0] == "collect-decode":
        parser = build_collect_decode_parser(f"{prog} collect-decode")
        run_collect_decode(parser.parse_args(argv[1:]))
        return 0

    if argv and argv[0] == "summarize":
        parser = build_summarize_parser(f"{prog} summarize")
        run_summarize(parser.parse_args(argv[1:]))
        return 0

    # Backward-compatible shorthand:
    #   summarize_prefill_nsys_sqlite.py <sqlite> [summary options]
    parser = build_summarize_parser(prog)
    run_summarize(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
