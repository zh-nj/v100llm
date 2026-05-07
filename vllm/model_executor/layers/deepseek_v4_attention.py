# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs

from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.fp8_a_dequant_triton import (
    sm70_fp8_a_dequant_to_fp16,
)
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4
_QK_NOPE_DIM = 448
_QK_ROPE_DIM = 64
_QK_FP8_MAX = 448.0
_QK_QUANT_BLOCK = 64
_QK_TOKEN_DATA_BYTES = _QK_NOPE_DIM + _QK_ROPE_DIM * 2
_QK_SCALE_BYTES = 8
_SM70_FP16_ATTENTION_OUTPUT_MAX = float(torch.finfo(torch.float16).max)

_PREFILL_CUDAGRAPH_ENABLED = os.getenv("VLLM_PREFILL_CUDAGRAPH", "0") == "1"
_PREFILL_CUDAGRAPH_DEBUG = os.getenv("VLLM_PREFILL_CUDAGRAPH_DEBUG", "0") == "1"
_PREFILL_CUDAGRAPH_PERF_GATE = (
    os.getenv("VLLM_PREFILL_CUDAGRAPH_PERF_GATE", "0") == "1"
)
_DEEPSEEK_V4_PROFILE_ENABLED = os.getenv("VLLM_DEEPSEEK_V4_PROFILE", "0") == "1"
_DEEPSEEK_V4_PROFILE_NVTX = os.getenv("VLLM_DEEPSEEK_V4_PROFILE_NVTX", "0") == "1"
_DEEPSEEK_V4_PROFILE_LOG_EVERY = int(
    os.getenv("VLLM_DEEPSEEK_V4_PROFILE_LOG_EVERY", "200")
)
# Event-record mode: "queue" (default) batches CUDA event syncs at step
# boundary; "eager" syncs each end-event immediately (legacy behavior).
_DEEPSEEK_V4_PROFILE_MODE = os.getenv(
    "VLLM_DEEPSEEK_V4_PROFILE_MODE", "queue"
).strip().lower()
if _DEEPSEEK_V4_PROFILE_MODE not in ("queue", "eager"):
    _DEEPSEEK_V4_PROFILE_MODE = "queue"
# Cap recorded steps; 0 / unset => unlimited.
_DEEPSEEK_V4_PROFILE_STEP_LIMIT = int(
    os.getenv("VLLM_DEEPSEEK_V4_PROFILE_STEP_LIMIT", "0")
)
# Phase filter: "decode", "prefill", or "both" (default).
_DEEPSEEK_V4_PROFILE_PHASE_FILTER = os.getenv(
    "VLLM_DEEPSEEK_V4_PROFILE_PHASE_FILTER", "both"
).strip().lower()
if _DEEPSEEK_V4_PROFILE_PHASE_FILTER not in ("decode", "prefill", "both"):
    _DEEPSEEK_V4_PROFILE_PHASE_FILTER = "both"

# O3 fix: Tunable num_warps for the SM70 qnorm+RoPE+KV-insert Triton kernel.
# Under CUDA-graph capture_size=1 decode the launch grid is degenerate (1,);
# bumping num_warps boosts per-program parallelism to compensate. Default 4.
try:
    _SM70_DEEPSEEK_V4_KV_INSERT_NUM_WARPS = max(
        1,
        int(os.getenv("VLLM_SM70_DEEPSEEK_V4_KV_INSERT_NUM_WARPS", "4")),
    )
except ValueError:
    _SM70_DEEPSEEK_V4_KV_INSERT_NUM_WARPS = 4


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class _DeepseekV4PhasePrometheus:
    def __init__(self) -> None:
        import prometheus_client

        self._prometheus_client = prometheus_client
        self.duration_us = self._counter(
            "vllm:deepseek_v4_phase_duration_us",
            "DeepSeek V4 attention phase CUDA event duration in microseconds.",
            ("phase",),
        )
        self.records = self._counter(
            "vllm:deepseek_v4_phase_records",
            "DeepSeek V4 attention phase record count.",
            ("phase",),
        )
        self.failed_events = self._counter(
            "vllm:deepseek_v4_phase_failed_events",
            "DeepSeek V4 attention phase CUDA event failures.",
            ("phase",),
        )

    def _counter(self, name: str, documentation: str, labelnames: tuple[str, ...]):
        try:
            return self._prometheus_client.Counter(
                name=name,
                documentation=documentation,
                labelnames=labelnames,
            )
        except ValueError:
            names_to_collectors = self._prometheus_client.REGISTRY._names_to_collectors
            collector = names_to_collectors.get(name)
            if collector is None:
                collector = names_to_collectors.get(f"{name}_total")
            if collector is None:
                raise
            return collector

    def record(self, label: str, elapsed_us: float) -> None:
        self.duration_us.labels(label).inc(float(elapsed_us))
        self.records.labels(label).inc()

    def record_failure(self, label: str) -> None:
        try:
            self.failed_events.labels(label).inc()
        except Exception:
            pass


_DEEPSEEK_V4_PROFILE_PROMETHEUS: _DeepseekV4PhasePrometheus | None = None
_DEEPSEEK_V4_PROFILE_PROMETHEUS_LOCK = threading.Lock()
_DEEPSEEK_V4_PROFILE_PROMETHEUS_DISABLED = False


def _get_deepseek_v4_phase_prometheus():
    global _DEEPSEEK_V4_PROFILE_PROMETHEUS_DISABLED
    global _DEEPSEEK_V4_PROFILE_PROMETHEUS
    sink = _DEEPSEEK_V4_PROFILE_PROMETHEUS
    if sink is not None:
        return sink
    if _DEEPSEEK_V4_PROFILE_PROMETHEUS_DISABLED:
        return None
    with _DEEPSEEK_V4_PROFILE_PROMETHEUS_LOCK:
        sink = _DEEPSEEK_V4_PROFILE_PROMETHEUS
        if sink is not None:
            return sink
        try:
            _DEEPSEEK_V4_PROFILE_PROMETHEUS = _DeepseekV4PhasePrometheus()
        except Exception:
            _DEEPSEEK_V4_PROFILE_PROMETHEUS_DISABLED = True
            logger.exception(
                "DeepSeek V4 phase Prometheus metrics disabled after init failure"
            )
            return None
        return _DEEPSEEK_V4_PROFILE_PROMETHEUS


class _DeepseekV4PhaseRawTrace:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Best-effort env_metadata sidecar (Task 1.4).
        try:
            self._write_env_metadata_sidecar()
        except Exception:
            logger.warning(
                "DeepSeek V4 profile: failed to write env_metadata sidecar",
                exc_info=True,
            )

    def _write_env_metadata_sidecar(self) -> None:
        import datetime
        import subprocess

        meta_path = self.path + ".meta.json"
        meta: dict = {}

        def _safe(fn, key, default=None):
            try:
                meta[key] = fn()
            except Exception:
                logger.warning(
                    "DeepSeek V4 profile env_metadata: %s unavailable", key
                )
                meta[key] = default

        def _git(args):
            return subprocess.run(
                ["git", *args],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()

        _safe(lambda: _git(["rev-parse", "HEAD"]), "git_commit")
        _safe(lambda: _git(["rev-parse", "--abbrev-ref", "HEAD"]), "git_branch")
        _safe(lambda: torch.__version__, "torch_version")
        _safe(lambda: torch.version.cuda, "cuda_version")
        meta["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
        meta["model_path"] = os.environ.get("VLLM_MODEL_PATH") or os.environ.get(
            "MODEL_PATH"
        )
        meta["timestamp_start_utc"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

        gpus: list[dict] = []
        try:
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    try:
                        props = torch.cuda.get_device_properties(idx)
                        gpus.append({
                            "index": idx,
                            "name": props.name,
                            "bus_id": getattr(props, "pci_bus_id", None),
                        })
                    except Exception:
                        gpus.append({"index": idx})
        except Exception:
            logger.warning(
                "DeepSeek V4 profile env_metadata: gpu enumeration failed"
            )
        meta["gpus"] = gpus

        try:
            meta["vllm_env_vars"] = {
                k: v for k, v in os.environ.items() if k.startswith("VLLM_")
            }
        except Exception:
            meta["vllm_env_vars"] = {}

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

    def record(
        self,
        label: str,
        elapsed_us: float,
        *,
        step_id: int | None = None,
        step_kind: str | None = None,
        step_token_count: int | None = None,
        layer_idx: int | None = None,
        compress_ratio: int | None = None,
        extra: dict | None = None,
    ) -> None:
        cuda_device = None
        if torch.cuda.is_available():
            try:
                cuda_device = int(torch.cuda.current_device())
            except Exception:
                cuda_device = None
        row = {
            "pid": os.getpid(),
            "cuda_device": cuda_device,
            "phase": label,
            "elapsed_us": float(elapsed_us),
            "step_id": step_id,
            "step_kind": step_kind,
            "step_token_count": step_token_count,
            "layer_idx": layer_idx,
            "compress_ratio": compress_ratio,
        }
        if extra:
            for k, v in extra.items():
                row.setdefault(k, v)
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as trace:
                trace.write(line)


_DEEPSEEK_V4_PROFILE_RAW_TRACE: _DeepseekV4PhaseRawTrace | None = None
_DEEPSEEK_V4_PROFILE_RAW_TRACE_LOCK = threading.Lock()
_DEEPSEEK_V4_PROFILE_RAW_TRACE_DISABLED = False


def _get_deepseek_v4_phase_raw_trace():
    global _DEEPSEEK_V4_PROFILE_RAW_TRACE
    global _DEEPSEEK_V4_PROFILE_RAW_TRACE_DISABLED
    path = os.getenv("VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH")
    if not path or _DEEPSEEK_V4_PROFILE_RAW_TRACE_DISABLED:
        return None
    sink = _DEEPSEEK_V4_PROFILE_RAW_TRACE
    if sink is not None and sink.path == path:
        return sink
    with _DEEPSEEK_V4_PROFILE_RAW_TRACE_LOCK:
        sink = _DEEPSEEK_V4_PROFILE_RAW_TRACE
        if sink is not None and sink.path == path:
            return sink
        try:
            _DEEPSEEK_V4_PROFILE_RAW_TRACE = _DeepseekV4PhaseRawTrace(path)
        except Exception:
            _DEEPSEEK_V4_PROFILE_RAW_TRACE_DISABLED = True
            logger.exception(
                "DeepSeek V4 phase raw trace disabled after init failure"
            )
            return None
        return _DEEPSEEK_V4_PROFILE_RAW_TRACE


class _DeepseekV4PhaseProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._records = 0
        self._failed_events = 0

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._records = 0
            self._failed_events = 0

    def record_failure(self, label: str) -> None:
        with self._lock:
            self._failed_events += 1
        sink = _get_deepseek_v4_phase_prometheus()
        if sink is not None:
            sink.record_failure(label)

    @property
    def failed_events(self) -> int:
        with self._lock:
            return self._failed_events

    def record(
        self,
        label: str,
        elapsed_us: float,
        *,
        step_id: int | None = None,
        step_kind: str | None = None,
        step_token_count: int | None = None,
        layer_idx: int | None = None,
        compress_ratio: int | None = None,
        extra: dict | None = None,
    ) -> None:
        with self._lock:
            stats = self._stats[label]
            stats[0] += 1.0
            stats[1] += float(elapsed_us)
            self._records += 1
            should_log = (
                _DEEPSEEK_V4_PROFILE_LOG_EVERY > 0
                and self._records % _DEEPSEEK_V4_PROFILE_LOG_EVERY == 0
            )
            snapshot = self._snapshot_locked(reset=False) if should_log else None
        sink = _get_deepseek_v4_phase_prometheus()
        if sink is not None:
            sink.record(label, elapsed_us)
        raw_trace = _get_deepseek_v4_phase_raw_trace()
        if raw_trace is not None:
            raw_trace.record(
                label,
                elapsed_us,
                step_id=step_id,
                step_kind=step_kind,
                step_token_count=step_token_count,
                layer_idx=layer_idx,
                compress_ratio=compress_ratio,
                extra=extra,
            )
        if snapshot is not None:
            top = sorted(
                snapshot.items(),
                key=lambda item: item[1]["total_us"],
                reverse=True,
            )[:16]
            logger.info(
                "DeepSeek V4 phase profile top=%s",
                {
                    name: {
                        "count": data["count"],
                        "total_ms": round(data["total_us"] / 1000.0, 3),
                        "avg_us": round(data["avg_us"], 1),
                    }
                    for name, data in top
                },
            )

    def _snapshot_locked(self, *, reset: bool) -> dict[str, dict[str, float]]:
        snapshot = {
            label: {
                "count": int(values[0]),
                "total_us": values[1],
                "avg_us": values[1] / values[0] if values[0] else 0.0,
            }
            for label, values in self._stats.items()
        }
        if reset:
            self._stats.clear()
            self._records = 0
        return snapshot

    def snapshot(self, *, reset: bool = False) -> dict[str, dict[str, float]]:
        with self._lock:
            return self._snapshot_locked(reset=reset)


_DEEPSEEK_V4_PROFILE = _DeepseekV4PhaseProfiler()


# ---------------------------------------------------------------------------
# Thread-local step/layer context (Task 1.1) + queued CUDA events (Task 1.2).
# ---------------------------------------------------------------------------


class _DeepseekV4PhaseContext(threading.local):
    """Thread-local profiling context.

    Fields default to None so existing call sites that omit step/layer
    information remain backward compatible.
    """

    def __init__(self) -> None:  # noqa: D401 - threading.local init
        super().__init__()
        self.step_id: int | None = None
        self.step_kind: str | None = None
        self.step_token_count: int | None = None
        self.layer_idx: int | None = None
        self.compress_ratio: int | None = None
        # Per-thread CUDA-event ring buffer for queue mode. Each entry is
        # (start, end, label, ctx_snapshot).
        self.event_queue: list[tuple] = []
        # Monotonic step counter used when no caller provides explicit ids.
        self._auto_step_counter = 0


_DEEPSEEK_V4_PHASE_CONTEXT = _DeepseekV4PhaseContext()


def _phase_context_snapshot() -> dict:
    ctx = _DEEPSEEK_V4_PHASE_CONTEXT
    return {
        "step_id": ctx.step_id,
        "step_kind": ctx.step_kind,
        "step_token_count": ctx.step_token_count,
        "layer_idx": ctx.layer_idx,
        "compress_ratio": ctx.compress_ratio,
    }


def _phase_filter_allows(step_kind: str | None) -> bool:
    if _DEEPSEEK_V4_PROFILE_PHASE_FILTER == "both" or step_kind is None:
        return True
    return step_kind == _DEEPSEEK_V4_PROFILE_PHASE_FILTER


def _phase_step_limit_exceeded(step_id: int | None) -> bool:
    if _DEEPSEEK_V4_PROFILE_STEP_LIMIT <= 0 or step_id is None:
        return False
    return step_id >= _DEEPSEEK_V4_PROFILE_STEP_LIMIT


def _flush_event_queue() -> None:
    queue = _DEEPSEEK_V4_PHASE_CONTEXT.event_queue
    if not queue:
        return
    # One sync per step boundary (queue mode) is sufficient: synchronize
    # the last enqueued end event, then read elapsed_time on each pair.
    try:
        queue[-1][1].synchronize()
    except Exception:
        logger.warning(
            "DeepSeek V4 profile: queued event sync failed; dropping batch",
            exc_info=True,
        )
        queue.clear()
        return
    for entry in queue:
        if len(entry) == 5:
            start, end, label, ctx, extra = entry
        else:
            start, end, label, ctx = entry
            extra = None
        try:
            elapsed_us = start.elapsed_time(end) * 1000.0
        except Exception:
            logger.warning(
                "DeepSeek V4 profile: elapsed_time failed for phase=%s "
                "layer_idx=%s step_id=%s",
                label,
                ctx.get("layer_idx"),
                ctx.get("step_id"),
            )
            _DEEPSEEK_V4_PROFILE.record_failure(label)
            continue
        _DEEPSEEK_V4_PROFILE.record(
            label,
            elapsed_us,
            step_id=ctx.get("step_id"),
            step_kind=ctx.get("step_kind"),
            step_token_count=ctx.get("step_token_count"),
            layer_idx=ctx.get("layer_idx"),
            compress_ratio=ctx.get("compress_ratio"),
            extra=extra,
        )
    queue.clear()


def _is_cuda_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


@contextmanager
def _profile_phase(
    label: str, ref: torch.Tensor, *, extra: dict | None = None
) -> Iterator[None]:
    if not ref.is_cuda or _is_cuda_stream_capturing():
        yield
        return
    use_events = _DEEPSEEK_V4_PROFILE_ENABLED
    use_nvtx = _DEEPSEEK_V4_PROFILE_NVTX
    if not use_events and not use_nvtx:
        yield
        return
    ctx_snapshot = _phase_context_snapshot()
    if use_events and (
        not _phase_filter_allows(ctx_snapshot["step_kind"])
        or _phase_step_limit_exceeded(ctx_snapshot["step_id"])
    ):
        use_events = False
    if not use_events and not use_nvtx:
        yield
        return
    if use_nvtx:
        torch.cuda.nvtx.range_push(label)
    if not use_events:
        try:
            yield
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        if _DEEPSEEK_V4_PROFILE_MODE == "queue":
            _DEEPSEEK_V4_PHASE_CONTEXT.event_queue.append(
                (start, end, label, ctx_snapshot, extra)
            )
        else:
            try:
                end.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000.0
            except Exception:
                logger.warning(
                    "DeepSeek V4 profile: elapsed_time failed for phase=%s "
                    "layer_idx=%s step_id=%s",
                    label,
                    ctx_snapshot.get("layer_idx"),
                    ctx_snapshot.get("step_id"),
                )
                _DEEPSEEK_V4_PROFILE.record_failure(label)
            else:
                _DEEPSEEK_V4_PROFILE.record(
                    label,
                    elapsed_us,
                    step_id=ctx_snapshot["step_id"],
                    step_kind=ctx_snapshot["step_kind"],
                    step_token_count=ctx_snapshot["step_token_count"],
                    layer_idx=ctx_snapshot["layer_idx"],
                    compress_ratio=ctx_snapshot["compress_ratio"],
                    extra=extra,
                )
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


@contextmanager
def _profile_step(
    step_idx: int | None = None,
    phase_kind: str | None = None,
    token_count: int | None = None,
) -> Iterator[None]:
    """Push thread-local step context + NVTX range; flush queued events on exit.

    Safe to call when profiling is disabled — becomes a near no-op.
    """
    profiling_active = (
        _DEEPSEEK_V4_PROFILE_ENABLED or _DEEPSEEK_V4_PROFILE_NVTX
    )
    ctx = _DEEPSEEK_V4_PHASE_CONTEXT
    prev = (ctx.step_id, ctx.step_kind, ctx.step_token_count)
    if step_idx is None:
        step_idx = ctx._auto_step_counter
        ctx._auto_step_counter += 1
    ctx.step_id = step_idx
    ctx.step_kind = phase_kind
    ctx.step_token_count = token_count
    pushed_nvtx = False
    if profiling_active and _DEEPSEEK_V4_PROFILE_NVTX and torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(
                f"step[idx={step_idx},type={phase_kind},tokens={token_count}]"
            )
            pushed_nvtx = True
        except Exception:
            pushed_nvtx = False
    try:
        yield
    finally:
        if (
            _DEEPSEEK_V4_PROFILE_ENABLED
            and _DEEPSEEK_V4_PROFILE_MODE == "queue"
            and torch.cuda.is_available()
            and not _is_cuda_stream_capturing()
        ):
            try:
                _flush_event_queue()
            except Exception:
                logger.warning(
                    "DeepSeek V4 profile: queue flush failed",
                    exc_info=True,
                )
                ctx.event_queue.clear()
        if pushed_nvtx:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
        ctx.step_id, ctx.step_kind, ctx.step_token_count = prev


@contextmanager
def _profile_layer(
    layer_idx: int | None = None,
    compress_ratio: int | None = None,
) -> Iterator[None]:
    """Push thread-local layer context + NVTX range. Safe no-op when disabled."""
    ctx = _DEEPSEEK_V4_PHASE_CONTEXT
    prev = (ctx.layer_idx, ctx.compress_ratio)
    ctx.layer_idx = layer_idx
    ctx.compress_ratio = compress_ratio
    pushed_nvtx = False
    if (
        _DEEPSEEK_V4_PROFILE_NVTX
        and torch.cuda.is_available()
    ):
        try:
            torch.cuda.nvtx.range_push(
                f"layer[idx={layer_idx},compress_ratio={compress_ratio}]"
            )
            pushed_nvtx = True
        except Exception:
            pushed_nvtx = False
    try:
        yield
    finally:
        if pushed_nvtx:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
        ctx.layer_idx, ctx.compress_ratio = prev


def _profile_or_null(label: str, ref: torch.Tensor, *, extra: dict | None = None):
    if not _DEEPSEEK_V4_PROFILE_ENABLED and not _DEEPSEEK_V4_PROFILE_NVTX:
        return nullcontext()
    return _profile_phase(label, ref, extra=extra)


def _default_prefill_cudagraph_capture_sizes(
    *,
    max_num_batched_tokens: int,
    max_model_len: int,
    max_M: int,
) -> list[tuple[int, int]]:
    env_sizes = os.getenv("VLLM_PREFILL_CUDAGRAPH_CAPTURE_TOKENS")
    max_tokens = max(1, min(max_num_batched_tokens, max_model_len))
    if env_sizes:
        tokens = sorted(
            {
                int(part.strip())
                for part in env_sizes.split(",")
                if part.strip()
            }
        )
        return [(min(token, max_tokens), max_M) for token in tokens if token > 0]

    tokens: list[int] = []
    size = 1
    while size < max_tokens:
        tokens.append(size)
        size *= 2
    tokens.append(max_tokens)
    return [(token, max_M) for token in sorted(set(tokens))]


def _trace_nonfinite_tensor(label: str, tensor: torch.Tensor) -> None:
    if os.getenv("VLLM_DEEPSEEK_V4_NAN_TRACE", "0") != "1":
        return
    if not torch.is_floating_point(tensor):
        return
    if torch.isfinite(tensor).all():
        return
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    if finite_values.numel() == 0:
        min_value = max_value = float("nan")
    else:
        stats = finite_values.float()
        min_value = float(stats.min().item())
        max_value = float(stats.max().item())
    logger.error(
        "DeepSeek V4 attention nonfinite tensor at %s: shape=%s dtype=%s "
        "nan=%d inf=%d finite_min=%s finite_max=%s",
        label,
        tuple(tensor.shape),
        tensor.dtype,
        int(torch.isnan(tensor).sum().item()),
        int(torch.isinf(tensor).sum().item()),
        min_value,
        max_value,
    )


def _trace_tensor_summary(label: str, tensor: torch.Tensor) -> None:
    if os.getenv("VLLM_DEEPSEEK_V4_NAN_TRACE", "0") != "1":
        return
    if tensor.numel() == 0:
        logger.error(
            "DeepSeek V4 trace tensor at %s: shape=%s dtype=%s empty",
            label,
            tuple(tensor.shape),
            tensor.dtype,
        )
        return
    flat = tensor
    if not torch.is_floating_point(flat):
        flat = flat.to(torch.float32)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if finite_values.numel() == 0:
        min_value = max_value = float("nan")
    else:
        stats = finite_values.float()
        min_value = float(stats.min().item())
        max_value = float(stats.max().item())
    logger.error(
        "DeepSeek V4 trace tensor at %s: shape=%s dtype=%s "
        "nan=%d inf=%d finite_min=%s finite_max=%s",
        label,
        tuple(tensor.shape),
        tensor.dtype,
        int(torch.isnan(flat).sum().item()),
        int(torch.isinf(flat).sum().item()),
        min_value,
        max_value,
    )


def _trace_layer16_summary(prefix: str, label: str, tensor: torch.Tensor) -> None:
    if prefix.endswith("layers.16.attn"):
        _trace_tensor_summary(f"{prefix}.{label}", tensor)


def _should_use_qnorm_rope_kv_insert_fallback(q: torch.Tensor) -> bool:
    if not q.is_cuda:
        return False
    capability = torch.cuda.get_device_capability(q.device)
    return capability[0] < 8


def _normalize_flashmla_sm70_prefill_kv_(kv: torch.Tensor) -> torch.Tensor:
    return kv


def _normalize_sm70_fp8_cache_exponents(exponents: torch.Tensor) -> torch.Tensor:
    return exponents


def _should_clamp_sm70_fp16_attention_output(out: torch.Tensor) -> bool:
    if not out.is_cuda or out.dtype != torch.float16:
        return False
    capability = torch.cuda.get_device_capability(out.device)
    return capability[0] < 8


def _clamp_sm70_fp16_attention_output_(out: torch.Tensor) -> torch.Tensor:
    return out.clamp_(
        min=-_SM70_FP16_ATTENTION_OUTPUT_MAX,
        max=_SM70_FP16_ATTENTION_OUTPUT_MAX,
    )


def _should_use_sm70_decode_prefill_fallback(
    q: torch.Tensor,
    swa_only: bool,
) -> bool:
    del swa_only
    if not q.is_cuda:
        return False
    capability = torch.cuda.get_device_capability(q.device)
    major, minor = capability
    if major >= 8:
        return False
    if major == 7 and minor == 0 and _env_flag(
        "VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE", default=True
    ):
        return False
    return True


def _get_decode_prefill_fallback_workspace(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if is_workspace_manager_initialized():
        return current_workspace_manager().get_simultaneous((shape, dtype))[0]
    return torch.empty(shape, dtype=dtype, device=device)


def _decode_prefill_fallback_slots(
    global_indices: torch.Tensor,
) -> torch.Tensor:
    if global_indices.ndim == 3:
        assert global_indices.shape[1] == 1
        return global_indices[:, 0, :]
    if global_indices.ndim == 2:
        return global_indices
    raise ValueError(
        "Decode fallback indices must have shape [tokens, topk] or "
        f"[tokens, 1, topk], got {tuple(global_indices.shape)}"
    )


def _build_decode_prefill_fallback_indices(
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    *,
    row_stride: int | None = None,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = _decode_prefill_fallback_slots(global_indices)
    local_lens = global_lens.reshape(-1)
    topk = slots.shape[-1]
    if row_stride is None:
        row_stride = topk

    offsets = torch.arange(topk, device=slots.device, dtype=torch.int32)
    bases = (
        torch.arange(slots.shape[0], device=slots.device, dtype=torch.int32)
        .unsqueeze(1)
        .mul_(row_stride)
        .add_(offset)
    )
    valid = (offsets.unsqueeze(0) < local_lens.unsqueeze(1)) & (slots >= 0)
    local_indices = torch.where(
        valid,
        bases + offsets.unsqueeze(0),
        torch.full_like(slots, -1),
    )
    return local_indices.unsqueeze(1), local_lens


@triton.jit
def _gather_decode_kv_triton_kernel(
    # Output: [num_tokens, topk, 512] viewed as uint16
    out_ptr,
    out_stride0,
    out_stride1,
    # KV cache: [num_blocks, block_bytes] uint8
    k_cache_ptr,
    block_stride,
    # Slots: [num_tokens, topk] int32
    slots_ptr,
    slots_stride0,
    # Lens: [num_tokens] int32
    lens_ptr,
    num_tokens,
    topk,
    cache_block_size: tl.constexpr,
    # Optional local-index output (fused index-building).
    # When non-zero, each program writes pid_token * idx_row_stride +
    # pid_topk + idx_offset for valid slots, or -1 for invalid.
    local_indices_ptr,
    idx_stride0,
    idx_row_stride,
    idx_offset,
    EMIT_INDICES: tl.constexpr = False,
    FP8_DIM: tl.constexpr = 448,
    ROPE_DIM: tl.constexpr = 64,
    SCALE_DIM: tl.constexpr = 8,
    QUANT_BLOCK: tl.constexpr = 64,
    TOKEN_DATA_BYTES: tl.constexpr = 576,
    N_QUANT_BLOCKS: tl.constexpr = 7,
    OUTPUT_DIM: tl.constexpr = 512,
):
    """Triton kernel to gather and dequantize KV from paged FP8 cache.

    Replaces the pure-torch _gather_decode_prefill_fallback_kv_ for SM70.
    FP8 decode logic matches _dequantize_and_gather_k_kernel exactly.

    When EMIT_INDICES is True, also writes local dense indices into
    local_indices_ptr, fusing the work of _build_decode_prefill_fallback_indices.
    """
    pid_token = tl.program_id(0).to(tl.int64)
    pid_topk = tl.program_id(1).to(tl.int64)

    if pid_token >= num_tokens:
        return

    out_row = out_ptr + pid_token * out_stride0 + pid_topk * out_stride1

    tok_len = tl.load(lens_ptr + pid_token)
    if pid_topk >= tok_len:
        zero_offsets = tl.arange(0, OUTPUT_DIM)
        tl.store(out_row + zero_offsets, tl.zeros((OUTPUT_DIM,), dtype=tl.uint16))
        if EMIT_INDICES:
            tl.store(local_indices_ptr + pid_token * idx_stride0 + pid_topk, -1)
        return

    slot_idx = tl.load(
        slots_ptr + pid_token * slots_stride0 + pid_topk
    ).to(tl.int64)

    if slot_idx < 0:
        zero_offsets = tl.arange(0, OUTPUT_DIM)
        tl.store(out_row + zero_offsets, tl.zeros((OUTPUT_DIM,), dtype=tl.uint16))
        if EMIT_INDICES:
            tl.store(local_indices_ptr + pid_token * idx_stride0 + pid_topk, -1)
        return

    if EMIT_INDICES:
        local_idx = pid_token * idx_row_stride + pid_topk + idx_offset
        tl.store(
            local_indices_ptr + pid_token * idx_stride0 + pid_topk,
            local_idx.to(tl.int32),
        )

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    cache_block_base = k_cache_ptr + block_idx * block_stride
    token_data_base = cache_block_base + pos_in_block * TOKEN_DATA_BYTES
    token_scale_base = (
        cache_block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_DIM
    )

    # Batch-load all 7 UE8M0 scale bytes before the quant block loop
    # to reduce scattered reads (one vector load vs 7 scalar loads).
    # Compute all scales upfront as a vector. Use power-of-2 arange(0,8)
    # with mask for the 7 valid scale bytes.
    scale_offsets = tl.arange(0, 8)
    scale_mask = scale_offsets < N_QUANT_BLOCKS
    scale_bytes = tl.load(
        token_scale_base + scale_offsets, mask=scale_mask, other=127
    )
    all_exponents = scale_bytes.to(tl.float32) - 127.0
    all_scales = tl.exp2(all_exponents)

    # FP8 dequant with optimized bit manipulation (matches
    # _decode_fp8_e4m3fn from sm70_mqa_logits.py: reduces 5
    # intermediates to 2 by shifting low7 bits directly).
    for qb_idx in tl.static_range(N_QUANT_BLOCKS):
        qb_start = qb_idx * QUANT_BLOCK
        fp8_offsets = tl.arange(0, QUANT_BLOCK)

        x_uint8 = tl.load(token_data_base + qb_start + fp8_offsets)

        # Optimized FP8 e4m3fn → float32 with subnormal handling.
        # Normal path: sign_bit | ((low7 + 120<<3) << 20) — same as
        # _decode_fp8_e4m3fn from sm70_mqa_logits.py.
        # Subnormal path (exp_bits==0, mant!=0): mant * 2^-9.
        val32 = x_uint8.to(tl.int32)
        sign_bit = (val32 & 0x80) << 24
        low7 = val32 & 0x7F
        fp32_bits = sign_bit | ((low7 + (120 << 3)) << 20)
        fp32_bits = tl.where(low7 == 0, 0, fp32_bits)
        normal_val = fp32_bits.to(tl.float32, bitcast=True)
        # Subnormals: exp_bits==0 means low7 in [1,7]; mantissa = low7 * 2^-9
        is_subnorm = low7 < 8  # low7 in [1..7] when low7 != 0
        is_subnorm = is_subnorm & (low7 != 0)
        mant_bits = low7  # For subnormals, low7 == mant_bits (exp=0)
        subnorm_val = mant_bits.to(tl.float32) * 1.953125e-3
        sign_mask = (val32 >> 7) & 1
        subnorm_val = tl.where(sign_mask == 1, -subnorm_val, subnorm_val)
        x_float = tl.where(is_subnorm, subnorm_val, normal_val)

        # UE8M0 scale from pre-loaded batch via mask extraction
        qb_mask = tl.arange(0, 8) == qb_idx
        scale = tl.sum(tl.where(qb_mask, all_scales, 0.0))

        x_dequant = x_float * scale

        # float32 → bf16 as uint16 (same rounding as gather kernel)
        x_u32 = x_dequant.to(tl.int32, bitcast=True)
        bf16_bits = ((x_u32 + 0x7FFF + ((x_u32 >> 16) & 1)) >> 16).to(
            tl.uint16
        )
        tl.store(out_row + qb_start + fp8_offsets, bf16_bits)

    # BF16 RoPE portion: vectorized copy as uint16 (single 64-element load)
    rope_u16_ptr = (token_data_base + FP8_DIM).to(tl.pointer_type(tl.uint16))
    rope_offsets = tl.arange(0, ROPE_DIM)
    rope_u16 = tl.load(rope_u16_ptr + rope_offsets)
    tl.store(out_row + FP8_DIM + rope_offsets, rope_u16)


def _gather_decode_prefill_fallback_kv_(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    slots = _decode_prefill_fallback_slots(global_indices)
    lens = global_lens.reshape(-1)
    topk = slots.shape[-1]
    if out.shape[0] != slots.shape[0] or out.shape[1] != topk:
        raise ValueError(
            "Decode fallback KV workspace shape must match indices, got "
            f"out={tuple(out.shape)} indices={tuple(global_indices.shape)}"
        )

    if slots.numel() == 0:
        out.zero_()
        return out

    num_tokens = slots.shape[0]
    block_stride = k_cache.stride(0)

    out_u16 = out.view(torch.uint16)
    _gather_decode_kv_triton_kernel[(num_tokens, topk)](
        out_u16,
        out_u16.stride(0),
        out_u16.stride(1),
        k_cache,
        block_stride,
        slots,
        slots.stride(0),
        lens,
        num_tokens,
        topk,
        block_size,
        0,  # local_indices_ptr (unused)
        0,  # idx_stride0 (unused)
        0,  # idx_row_stride (unused)
        0,  # idx_offset (unused)
        EMIT_INDICES=False,
    )
    return out


def _gather_decode_prefill_fallback_kv_with_indices_(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    block_size: int,
    *,
    row_stride: int | None = None,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused gather + index-build: calls the enhanced Triton kernel once to
    produce both the BF16 KV workspace *and* local dense indices, replacing
    the separate ``_gather_decode_prefill_fallback_kv_`` +
    ``_build_decode_prefill_fallback_indices`` sequence.

    Returns ``(out, local_indices, local_lens)`` where ``local_indices`` has
    shape ``[num_tokens, 1, topk]`` (matching the format expected by
    ``flash_mla_sparse_fwd``) and ``local_lens`` is ``[num_tokens]``.
    """
    slots = _decode_prefill_fallback_slots(global_indices)
    lens = global_lens.reshape(-1)
    topk = slots.shape[-1]
    if row_stride is None:
        row_stride = topk

    if out.shape[0] != slots.shape[0] or out.shape[1] != topk:
        raise ValueError(
            "Decode fallback KV workspace shape must match indices, got "
            f"out={tuple(out.shape)} indices={tuple(global_indices.shape)}"
        )

    num_tokens = slots.shape[0]
    local_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=slots.device,
    )

    if slots.numel() == 0:
        out.zero_()
        local_indices.fill_(-1)
        return out, local_indices.unsqueeze(1), lens

    block_stride = k_cache.stride(0)
    out_u16 = out.view(torch.uint16)

    _gather_decode_kv_triton_kernel[(num_tokens, topk)](
        out_u16,
        out_u16.stride(0),
        out_u16.stride(1),
        k_cache,
        block_stride,
        slots,
        slots.stride(0),
        lens,
        num_tokens,
        topk,
        block_size,
        local_indices,
        local_indices.stride(0),
        row_stride,
        offset,
        EMIT_INDICES=True,
    )
    return out, local_indices.unsqueeze(1), lens


# ---------------------------------------------------------------------------
# SM70 Triton kernel: fused Q-norm + GPT-J RoPE + FP8 quant + KV cache write
# Replaces _torch_qnorm_rope_kv_insert_fallback with a single kernel launch.
# ---------------------------------------------------------------------------

@triton.jit
def _encode_fp8_e4m3fn(value):
    """Encode float32 value to FP8 e4m3fn as uint8.

    Reverse of _decode_fp8_e4m3fn from sm70_mqa_logits.py.
    FP8 e4m3fn: sign(1) | exp(4) | mantissa(3), bias=7, max=448.0
    """
    # Extract sign
    val_i32 = value.to(tl.int32, bitcast=True)
    sign = (val_i32 >> 31) & 1  # 0 or 1

    abs_val = tl.abs(value)

    # Clamp to FP8 e4m3fn max
    abs_val = tl.minimum(abs_val, 448.0)

    # Handle zero / subnormal boundary
    # FP8 e4m3fn subnormals: exp=0, mant in [1..7] → values 1*2^-9 .. 7*2^-9
    # Smallest normal: exp=1, mant=0 → 2^(1-7) = 2^-6 = 0.015625
    # We use a threshold to decide normal vs subnormal
    is_zero = abs_val == 0.0

    # For normal values: extract FP32 exponent and mantissa
    abs_i32 = abs_val.to(tl.int32, bitcast=True)
    fp32_exp = (abs_i32 >> 23) & 0xFF  # biased FP32 exponent
    fp32_mant = abs_i32 & 0x7FFFFF  # 23-bit FP32 mantissa

    # FP8 biased exponent = fp32_exp - 127 + 7 = fp32_exp - 120
    fp8_exp = fp32_exp - 120

    # Round mantissa: FP8 has 3 mantissa bits, FP32 has 23
    # Shift right by 20, with round-to-nearest-even
    fp8_mant = (fp32_mant + (1 << 19)) >> 20
    # Handle mantissa overflow (rounding up can overflow 3 bits)
    carry = fp8_mant >> 3
    fp8_exp = fp8_exp + carry
    fp8_mant = fp8_mant & 0x7

    # Clamp exponent to valid range [1..15] for normal, handle overflow
    fp8_exp = tl.minimum(fp8_exp, 15)

    # Subnormal path: fp8_exp <= 0
    # For subnormal: mantissa encodes value / 2^-9
    # subnorm_mant = round(abs_val / 2^-9) = round(abs_val * 512)
    subnorm_mant = (abs_val * 512.0 + 0.5).to(tl.int32)
    subnorm_mant = tl.minimum(subnorm_mant, 7)

    is_subnorm = fp8_exp <= 0

    # Assemble FP8 byte
    normal_byte = (fp8_exp << 3) | fp8_mant
    subnorm_byte = subnorm_mant
    fp8_byte = tl.where(is_subnorm, subnorm_byte, normal_byte)
    fp8_byte = tl.where(is_zero, 0, fp8_byte)

    # Apply sign
    fp8_byte = fp8_byte | (sign << 7)
    return fp8_byte.to(tl.uint8)


@triton.jit
def _sm70_qnorm_rope_kv_insert_triton_kernel(
    # Q: [num_tokens, n_heads, head_dim] fp16, modified in-place
    q_ptr,
    q_stride0,
    q_stride1,
    # KV: [num_tokens, kv_dim] fp16 (kv_dim = 512)
    kv_ptr,
    kv_stride0,
    # K cache: [num_blocks, block_bytes] uint8
    k_cache_ptr,
    k_cache_block_stride,
    # slot_mapping: [num_tokens] int32
    slot_mapping_ptr,
    # positions: [num_tokens] int64
    positions_ptr,
    # cos_sin_cache: [max_pos, rope_dim] fp32 (rope_dim = 64, first 32=cos, last 32=sin)
    cos_sin_cache_ptr,
    cos_sin_cache_stride0,
    # Scalars
    eps: tl.constexpr,
    num_tokens,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    # Cache layout constants
    NOPE_DIM: tl.constexpr = 448,
    ROPE_DIM: tl.constexpr = 64,
    ROPE_HALF: tl.constexpr = 32,
    QUANT_BLOCK: tl.constexpr = 64,
    N_QUANT_BLOCKS: tl.constexpr = 7,
    TOKEN_DATA_BYTES: tl.constexpr = 576,
    SCALE_BYTES: tl.constexpr = 8,
    FP8_MAX: tl.constexpr = 448.0,
):
    """One program per token.

    Q-side (in-place): per-head RMSNorm + GPT-J interleaved RoPE
    KV-side: GPT-J RoPE + FP8 block quant + paged cache scatter write
    """
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    # ---- Load position and cos/sin for this token ----
    pos = tl.load(positions_ptr + pid)
    cs_base = cos_sin_cache_ptr + pos * cos_sin_cache_stride0
    cos_offsets = tl.arange(0, 32)
    sin_offsets = 32 + tl.arange(0, 32)
    cos_vals = tl.load(cs_base + cos_offsets).to(tl.float32)
    sin_vals = tl.load(cs_base + sin_offsets).to(tl.float32)

    # ---- Q-side: per-head RMSNorm + GPT-J RoPE (in-place) ----
    q_token_base = q_ptr + pid * q_stride0

    for h in tl.static_range(n_heads):
        q_head_base = q_token_base + h * q_stride1

        # Compute variance (sum of squares) over the full head_dim.
        # Process NoPE in 7 blocks of 64, then RoPE as 1 block of 64.
        sq_sum = tl.zeros((), dtype=tl.float32)
        for qb in tl.static_range(N_QUANT_BLOCKS):
            offs = qb * QUANT_BLOCK + tl.arange(0, 64)
            vals = tl.load(q_head_base + offs).to(tl.float32)
            sq_sum += tl.sum(vals * vals, axis=0)

        # RoPE portion of Q head (last 64 dims)
        rope_offs = NOPE_DIM + tl.arange(0, 64)
        q_rope_fp16 = tl.load(q_head_base + rope_offs)
        q_rope = q_rope_fp16.to(tl.float32)
        sq_sum += tl.sum(q_rope * q_rope, axis=0)

        variance = sq_sum / head_dim
        inv_rms = 1.0 / tl.sqrt(variance + eps)

        # Write back NoPE portion (normalized, no rotation)
        for qb in tl.static_range(N_QUANT_BLOCKS):
            offs = qb * QUANT_BLOCK + tl.arange(0, 64)
            vals = tl.load(q_head_base + offs).to(tl.float32)
            normed = vals * inv_rms
            tl.store(q_head_base + offs, normed.to(tl.float16))

        # GPT-J interleaved RoPE on the rope portion
        # even indices [0,2,4,...,62], odd indices [1,3,...,63]
        even_offs = NOPE_DIM + tl.arange(0, 32) * 2
        odd_offs = NOPE_DIM + tl.arange(0, 32) * 2 + 1
        q_even = tl.load(q_head_base + even_offs).to(tl.float32) * inv_rms
        q_odd = tl.load(q_head_base + odd_offs).to(tl.float32) * inv_rms

        new_even = q_even * cos_vals - q_odd * sin_vals
        new_odd = q_odd * cos_vals + q_even * sin_vals

        tl.store(q_head_base + even_offs, new_even.to(tl.float16))
        tl.store(q_head_base + odd_offs, new_odd.to(tl.float16))

    # ---- KV-side: GPT-J RoPE + FP8 quant + cache scatter write ----
    slot = tl.load(slot_mapping_ptr + pid)
    if slot < 0:
        return

    kv_base = kv_ptr + pid * kv_stride0

    # Apply GPT-J RoPE to KV rope portion (last 64 dims)
    kv_even_offs = NOPE_DIM + tl.arange(0, 32) * 2
    kv_odd_offs = NOPE_DIM + tl.arange(0, 32) * 2 + 1
    kv_rope_even = tl.load(kv_base + kv_even_offs).to(tl.float32)
    kv_rope_odd = tl.load(kv_base + kv_odd_offs).to(tl.float32)

    kv_rope_new_even = kv_rope_even * cos_vals - kv_rope_odd * sin_vals
    kv_rope_new_odd = kv_rope_odd * cos_vals + kv_rope_even * sin_vals

    # ---- Compute cache write addresses ----
    block_idx = (slot // cache_block_size).to(tl.int64)
    pos_in_block = slot % cache_block_size

    cache_block_base = k_cache_ptr + block_idx * k_cache_block_stride
    token_data_base = cache_block_base + pos_in_block * TOKEN_DATA_BYTES
    token_scale_base = (
        cache_block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_BYTES
    )

    # ---- FP8 block quantization of NoPE portion (7 blocks of 64) ----
    for qb_idx in tl.static_range(N_QUANT_BLOCKS):
        qb_start = qb_idx * QUANT_BLOCK
        offsets = tl.arange(0, 64)
        nope_vals = tl.load(kv_base + qb_start + offsets).to(tl.float32)

        # Per-block absmax
        absmax = tl.max(tl.abs(nope_vals), axis=0)
        absmax = tl.maximum(absmax, 1e-4)

        # UE8M0 scale: ceil(log2(absmax / 448.0))
        exponent = tl.math.ceil(tl.math.log2(absmax / FP8_MAX))
        scale = tl.exp2(exponent)

        # Quantize to FP8 range
        scaled_vals = nope_vals / scale
        scaled_vals = tl.maximum(tl.minimum(scaled_vals, FP8_MAX), -FP8_MAX)

        # Encode to FP8 e4m3fn bytes
        fp8_bytes = _encode_fp8_e4m3fn(scaled_vals)
        tl.store(token_data_base + qb_start + offsets, fp8_bytes)

        # Encode UE8M0 scale byte
        encoded_scale = (exponent + 127.0)
        encoded_scale = tl.maximum(tl.minimum(encoded_scale, 254.0), 0.0)
        tl.store(token_scale_base + qb_idx, encoded_scale.to(tl.uint8))

    # ---- Write RoPE portion as BF16 bytes ----
    # Convert rotated even/odd values to BF16 uint16, then write as uint8 pairs
    rope_u8_base = token_data_base + NOPE_DIM

    # Convert even values to BF16
    even_u32 = kv_rope_new_even.to(tl.int32, bitcast=True)
    even_bf16 = ((even_u32 + 0x7FFF + ((even_u32 >> 16) & 1)) >> 16).to(tl.uint16)
    # Convert odd values to BF16
    odd_u32 = kv_rope_new_odd.to(tl.int32, bitcast=True)
    odd_bf16 = ((odd_u32 + 0x7FFF + ((odd_u32 >> 16) & 1)) >> 16).to(tl.uint16)

    # Write interleaved as uint8: [even0_lo, even0_hi, odd0_lo, odd0_hi, ...]
    # Each bf16 value is 2 bytes, interleaved pattern: even[0], odd[0], even[1], odd[1]...
    # Total: 32 even + 32 odd = 64 bf16 values = 128 bytes
    rope_u16_ptr = rope_u8_base.to(tl.pointer_type(tl.uint16))
    even_store_offs = tl.arange(0, 32) * 2  # positions 0,2,4,...,62
    odd_store_offs = tl.arange(0, 32) * 2 + 1  # positions 1,3,...,63
    tl.store(rope_u16_ptr + even_store_offs, even_bf16)
    tl.store(rope_u16_ptr + odd_store_offs, odd_bf16)


def _sm70_triton_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """SM70 Triton replacement for _torch_qnorm_rope_kv_insert_fallback.

    Fuses Q-norm + GPT-J RoPE + FP8 quantization + paged cache write
    into a single Triton kernel launch. Grid: (num_tokens,).
    """
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    n_heads = q.shape[1]
    head_dim = q.shape[2]
    block_stride = k_cache.stride(0)

    # Q is fp16, kv is fp16, cos_sin_cache is fp32
    # Ensure positions are int64
    positions_i64 = positions.to(torch.int64) if positions.dtype != torch.int64 else positions

    grid = (num_tokens,)
    _sm70_qnorm_rope_kv_insert_triton_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        kv,
        kv.stride(0),
        k_cache,
        block_stride,
        slot_mapping,
        positions_i64,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        eps,
        num_tokens,
        n_heads,
        head_dim,
        block_size,
        num_warps=_SM70_DEEPSEEK_V4_KV_INSERT_NUM_WARPS,
    )


def _sm70_triton_qnorm_rope_kv_insert_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    # Mutation-only (no return); fake_impl is a no-op so dynamo/inductor
    # can reason about the op without tracing the triton kernel body.
    # This is what lets us opt out of the broken
    # `decompose_triton_kernel_wrapper_functional` pattern-matcher pass
    # in torch._inductor (2.9) that mishandles our kernels.
    return None


try:
    direct_register_custom_op(
        op_name="sm70_qnorm_rope_kv_insert",
        op_func=_sm70_triton_qnorm_rope_kv_insert,
        mutates_args=["q", "k_cache"],
        fake_impl=_sm70_triton_qnorm_rope_kv_insert_fake,
    )
    _sm70_triton_qnorm_rope_kv_insert_op = (
        torch.ops.vllm.sm70_qnorm_rope_kv_insert
    )
except (RuntimeError, AttributeError):
    _sm70_triton_qnorm_rope_kv_insert_op = None


def _apply_gptj_rope_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    assert nope_dim >= 0

    out = x.clone().float()
    rope = out[..., nope_dim:]
    even = rope[..., ::2]
    odd = rope[..., 1::2]

    cos_sin = cos_sin_cache[positions].float()
    view_shape = (positions.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    cos = cos_sin[..., :half].view(view_shape)
    sin = cos_sin[..., half:].view(view_shape)

    rotated = torch.empty_like(rope)
    rotated[..., ::2] = even * cos - odd * sin
    rotated[..., 1::2] = odd * cos + even * sin
    out[..., nope_dim:] = rotated
    return out.to(x.dtype)


def _torch_qnorm_rope_kv_insert_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """Torch correctness fallback for SM70, where the fused CUDA op is sm80+."""
    q_float = q.float()
    variance = q_float.pow(2).mean(dim=-1, keepdim=True)
    q_norm = (q_float * torch.rsqrt(variance + eps)).to(q.dtype)
    q.copy_(_apply_gptj_rope_tail(q_norm, positions, cos_sin_cache))

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    kv_rope = _apply_gptj_rope_tail(
        kv[:num_tokens],
        positions[:num_tokens],
        cos_sin_cache,
    )
    valid_mask = slot_mapping >= 0
    if not valid_mask.any():
        return

    kv_valid = kv_rope[valid_mask]
    slots = slot_mapping[valid_mask]
    block_indices = slots // block_size
    pos_in_block = slots % block_size

    nope = kv_valid[:, :_QK_NOPE_DIM].float()
    blocks = nope.view(-1, _QK_NOPE_DIM // _QK_QUANT_BLOCK, _QK_QUANT_BLOCK)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    exponents = _normalize_sm70_fp8_cache_exponents(
        torch.ceil(torch.log2(absmax / _QK_FP8_MAX))
    )
    scales = torch.exp2(exponents)
    fp8_data = (blocks / scales).clamp(-_QK_FP8_MAX, _QK_FP8_MAX)
    fp8_bytes = (
        fp8_data.to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
        .view(-1, _QK_NOPE_DIM)
    )
    rope_bytes = (
        kv_valid[:, _QK_NOPE_DIM:]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .view(-1, _QK_ROPE_DIM * 2)
    )
    token_data = torch.cat((fp8_bytes, rope_bytes), dim=-1)

    num_valid = slots.shape[0]
    data_offsets = (
        pos_in_block[:, None] * _QK_TOKEN_DATA_BYTES
        + torch.arange(_QK_TOKEN_DATA_BYTES, device=k_cache.device)
    )
    k_cache[block_indices[:, None], data_offsets] = token_data

    encoded_scales = (exponents.squeeze(-1) + 127.0).clamp(0, 255).to(torch.uint8)
    scale_data = torch.zeros(
        num_valid,
        _QK_SCALE_BYTES,
        dtype=torch.uint8,
        device=k_cache.device,
    )
    scale_data[:, : _QK_NOPE_DIM // _QK_QUANT_BLOCK] = encoded_scales
    scale_offsets = (
        block_size * _QK_TOKEN_DATA_BYTES
        + pos_in_block[:, None] * _QK_SCALE_BYTES
        + torch.arange(_QK_SCALE_BYTES, device=k_cache.device)
    )
    k_cache[block_indices[:, None], scale_offsets] = scale_data


@dataclass
class PrefillCaptureSize:
    max_tokens: int
    max_M: int
    graph: torch.cuda.CUDAGraph | None = None
    enabled: bool = True
    eager_latency_us: float = 0.0
    graph_latency_us: float = 0.0


@dataclass
class PrefillGraphConfig:
    enabled: bool = False
    capture_sizes: list[tuple[int, int]] = field(default_factory=list)
    performance_gate_tolerance: float = 0.05
    debug_mode: bool = False


class PrefillGraphDispatcher:
    def __init__(
        self,
        capture_sizes: list[tuple[int, int]],
        padded_heads: int,
        head_dim: int,
        scale: float,
        attn_sink: torch.Tensor,
        device: torch.device,
    ):
        self.padded_heads = padded_heads
        self.head_dim = head_dim
        self.scale = scale
        self.device = device
        self.enabled = True
        self.debug_validated = False

        self.sizes: list[PrefillCaptureSize] = sorted(
            [PrefillCaptureSize(max_tokens=t, max_M=m) for t, m in capture_sizes],
            key=lambda s: s.max_tokens,
        )

        # Dispatch counters for metrics
        self.graph_dispatch_count: int = 0
        self.eager_dispatch_count: int = 0

        # Workspace buffers (allocated below; None if OOM)
        self.q_padded: torch.Tensor | None = None
        self.indices_padded: torch.Tensor | None = None
        self.topk_length_padded: torch.Tensor | None = None
        self.kv_padded: torch.Tensor | None = None
        self.attn_sink_workspace: torch.Tensor | None = None
        self.output_padded: torch.Tensor | None = None
        self.flash_output_bf16: torch.Tensor | None = None

        if not self.sizes:
            self.enabled = False
            return

        max_tokens = max(s.max_tokens for s in self.sizes)
        max_M = max(s.max_M for s in self.sizes)

        try:
            # CUDA graph static inputs must have stable, private storage.  Do
            # not use the scratch workspace manager here: _forward_prefill()
            # also allocates its gather workspace from that pool, and aliasing
            # graph inputs with runtime KV scratch corrupts replay.
            self.q_padded = torch.empty(
                (max_tokens, padded_heads, head_dim),
                dtype=torch.float16,
                device=device,
            )
            self.indices_padded = torch.empty(
                (max_tokens, 1, max_M),
                dtype=torch.int32,
                device=device,
            )
            self.topk_length_padded = torch.empty(
                (max_tokens,),
                dtype=torch.int32,
                device=device,
            )
            self.kv_padded = torch.empty(
                (PREFILL_CHUNK_SIZE, max_M, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self.attn_sink_workspace = torch.empty(
                tuple(attn_sink.shape),
                dtype=attn_sink.dtype,
                device=device,
            )
            self.output_padded = torch.empty(
                (max_tokens, padded_heads, head_dim),
                dtype=torch.float16,
                device=device,
            )
            self.flash_output_bf16 = torch.empty(
                (max_tokens, padded_heads, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self.attn_sink_workspace.copy_(attn_sink)
            total_bytes = (
                self.q_padded.nelement() * self.q_padded.element_size()
                + self.indices_padded.nelement() * self.indices_padded.element_size()
                + self.topk_length_padded.nelement()
                * self.topk_length_padded.element_size()
                + self.kv_padded.nelement() * self.kv_padded.element_size()
                + self.attn_sink_workspace.nelement()
                * self.attn_sink_workspace.element_size()
                + self.output_padded.nelement() * self.output_padded.element_size()
                + self.flash_output_bf16.nelement()
                * self.flash_output_bf16.element_size()
            )
            logger.info(
                "Prefill CUDA graph workspace allocated: %.2f MB for %d capture sizes",
                total_bytes / (1024 * 1024),
                len(self.sizes),
            )
        except (AssertionError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(
                "Prefill CUDA graph disabled: insufficient GPU memory for "
                "workspace allocation: %s",
                e,
            )
            self.enabled = False
            self.q_padded = None
            self.indices_padded = None
            self.topk_length_padded = None
            self.kv_padded = None
            self.attn_sink_workspace = None
            self.output_padded = None
            self.flash_output_bf16 = None

    def find_capture_size(
        self, num_chunk_tokens: int, M: int
    ) -> PrefillCaptureSize | None:
        if not self.enabled:
            return None
        for s in self.sizes:
            if s.max_tokens >= num_chunk_tokens and s.max_M >= M and s.enabled:
                return s
        return None

    def _prepare_workspace(
        self,
        q_chunk: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        num_chunk_tokens: int,
        capture_size: PrefillCaptureSize,
    ) -> None:
        capture_tokens = capture_size.max_tokens
        capture_M = capture_size.max_M

        # Copy q_chunk → q_padded[:num_chunk_tokens], zero-fill rest
        self.q_padded[:num_chunk_tokens].copy_(q_chunk)
        if capture_tokens > num_chunk_tokens:
            self.q_padded[num_chunk_tokens:capture_tokens].zero_()

        # Copy combined_indices → indices_padded[:num_chunk_tokens], padding
        # the captured graph's static width with invalid slots when needed.
        indices_width = combined_indices.shape[-1]
        self.indices_padded[:num_chunk_tokens, :, :indices_width].copy_(
            combined_indices
        )
        if capture_M > indices_width:
            self.indices_padded[
                :num_chunk_tokens, :, indices_width:capture_M
            ].fill_(-1)
        if capture_tokens > num_chunk_tokens:
            self.indices_padded[num_chunk_tokens:capture_tokens].fill_(-1)

        # combined_lens is per token, not per request.  Keeping stale token
        # lengths here corrupts graph replay outputs.
        self.topk_length_padded[:num_chunk_tokens].copy_(combined_lens)
        if capture_tokens > num_chunk_tokens:
            self.topk_length_padded[num_chunk_tokens:capture_tokens].zero_()

    def _prepare_kv_workspace(
        self,
        kv_flat: torch.Tensor,
        chunk_size: int,
        M: int,
        capture_size: PrefillCaptureSize,
    ) -> None:
        assert self.kv_padded is not None
        capture_M = capture_size.max_M
        kv_source = kv_flat.view(PREFILL_CHUNK_SIZE, M, self.head_dim)
        self.kv_padded[:chunk_size, :M].copy_(kv_source[:chunk_size, :M])
        if capture_M > M:
            self.kv_padded[:chunk_size, M:capture_M].zero_()
        if PREFILL_CHUNK_SIZE > chunk_size:
            self.kv_padded[chunk_size:PREFILL_CHUNK_SIZE].zero_()

    def try_graph_replay(
        self,
        q_chunk: torch.Tensor,
        kv_flat: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output_slice: torch.Tensor,
        num_chunk_tokens: int,
        num_chunk_reqs: int,
        M: int,
        attn_sink: torch.Tensor | None = None,
    ) -> bool:
        capture_size = self.find_capture_size(num_chunk_tokens, M)
        if capture_size is None or capture_size.graph is None:
            self.eager_dispatch_count += 1
            return False

        self._prepare_workspace(
            q_chunk, combined_indices, combined_lens,
            num_chunk_tokens, capture_size,
        )
        self._prepare_kv_workspace(kv_flat, num_chunk_reqs, M, capture_size)
        if attn_sink is not None and self.attn_sink_workspace is not None:
            self.attn_sink_workspace.copy_(attn_sink)

        capture_size.graph.replay()

        output_slice.copy_(self.output_padded[:num_chunk_tokens])
        self.graph_dispatch_count += 1
        if self.graph_dispatch_count == 1:
            logger.info(
                "Prefill CUDA graph replayed first chunk with "
                "tokens=%d, M=%d, capture_tokens=%d, capture_M=%d",
                num_chunk_tokens, M, capture_size.max_tokens, capture_size.max_M,
            )
        return True

    def capture_graphs(
        self,
        flash_mla_sparse_fwd_fn,
        flashmla_bf16_io_fn,
        copy_flashmla_output_fn,
    ) -> None:
        for size in self.sizes:
            if not size.enabled:
                continue
            try:
                ct = size.max_tokens

                # Prepare deterministic workspace content for capture
                self.q_padded[:ct].zero_()
                self.indices_padded[:ct].fill_(-1)
                self.topk_length_padded[:ct].zero_()
                self.kv_padded.zero_()
                self.output_padded[:ct].zero_()
                self.flash_output_bf16[:ct].zero_()

                # Warm up the kernels before capture (required by CUDA graphs)
                flash_q, flash_out = flashmla_bf16_io_fn(
                    self.q_padded[:ct], self.output_padded[:ct]
                )
                flash_mla_sparse_fwd_fn(
                    flash_q,
                    self.kv_padded.view(-1, 1, self.head_dim),
                    self.indices_padded[:ct],
                    self.scale,
                    attn_sink=self.attn_sink_workspace,
                    topk_length=self.topk_length_padded[:ct],
                )
                copy_flashmla_output_fn(flash_out, self.output_padded[:ct])

                # Capture the graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    flash_q, flash_out = flashmla_bf16_io_fn(
                        self.q_padded[:ct], self.output_padded[:ct]
                    )
                    out_tuple = flash_mla_sparse_fwd_fn(
                        flash_q,
                        self.kv_padded.view(-1, 1, self.head_dim),
                        self.indices_padded[:ct],
                        self.scale,
                        attn_sink=self.attn_sink_workspace,
                        topk_length=self.topk_length_padded[:ct],
                    )
                    copy_flashmla_output_fn(out_tuple[0], self.output_padded[:ct])

                size.graph = graph
                logger.info(
                    "Prefill CUDA graph captured for size (tokens=%d, M=%d)",
                    ct, size.max_M,
                )
            except Exception as e:
                logger.warning(
                    "Prefill CUDA graph capture failed for size (tokens=%d, M=%d): %s",
                    size.max_tokens, size.max_M, e,
                )
                size.enabled = False

    def warmup_validate(
        self,
        flash_mla_sparse_fwd_fn,
        flashmla_bf16_io_fn,
        copy_flashmla_output_fn,
    ) -> None:
        for size in self.sizes:
            if not size.enabled or size.graph is None:
                continue
            ct = size.max_tokens

            # Generate random inputs for validation
            q_rand = torch.randn(
                ct, self.padded_heads, self.head_dim,
                dtype=torch.float16, device=self.device,
            )
            indices_rand = torch.full(
                (ct, 1, 1), -1, dtype=torch.int32, device=self.device,
            )
            lens_rand = torch.zeros(ct, dtype=torch.int32, device=self.device)
            kv_rand = torch.randn(
                PREFILL_CHUNK_SIZE,
                size.max_M,
                self.head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            output_eager = torch.zeros(
                ct, self.padded_heads, self.head_dim,
                dtype=torch.float16, device=self.device,
            )

            # --- Eager execution ---
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

            start_event.record()
            flash_q, flash_out = flashmla_bf16_io_fn(q_rand, output_eager)
            out_tuple = flash_mla_sparse_fwd_fn(
                flash_q,
                kv_rand.view(-1, 1, self.head_dim),
                indices_rand,
                self.scale,
                attn_sink=self.attn_sink_workspace,
                topk_length=lens_rand,
            )
            copy_flashmla_output_fn(out_tuple[0], output_eager)
            end_event.record()
            torch.cuda.synchronize()
            eager_us = start_event.elapsed_time(end_event) * 1000.0

            # --- Graph execution ---
            self.q_padded[:ct].copy_(q_rand)
            self.indices_padded[:ct].copy_(indices_rand)
            self.topk_length_padded[:ct].copy_(lens_rand)
            self.kv_padded.copy_(kv_rand)
            self.output_padded[:ct].zero_()

            # The first replay after capture can pay driver-side cold costs.
            # Do not let that one sample decide whether the graph is usable.
            size.graph.replay()
            torch.cuda.synchronize()

            start_event.record()
            size.graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            graph_us = start_event.elapsed_time(end_event) * 1000.0

            size.eager_latency_us = eager_us
            size.graph_latency_us = graph_us

            # Compare outputs
            output_graph = self.output_padded[:ct].clone()
            if not torch.equal(output_eager, output_graph):
                max_diff = (output_eager.float() - output_graph.float()).abs().max().item()
                logger.warning(
                    "Prefill graph output diverges from eager for size "
                    "(tokens=%d, M=%d): max_diff=%.6e — disabling",
                    ct, size.max_M, max_diff,
                )
                size.enabled = False
                continue

            tolerance = 0.05
            if (
                _PREFILL_CUDAGRAPH_PERF_GATE
                and graph_us > eager_us * (1.0 + tolerance)
            ):
                logger.info(
                    "Prefill graph for size (tokens=%d, M=%d) disabled: "
                    "graph %.1fµs vs eager %.1fµs",
                    ct, size.max_M, graph_us, eager_us,
                )
                size.enabled = False
            else:
                logger.info(
                    "Prefill graph for size (tokens=%d, M=%d) validated: "
                    "graph %.1fµs vs eager %.1fµs%s",
                    ct, size.max_M, graph_us, eager_us,
                    "" if _PREFILL_CUDAGRAPH_PERF_GATE
                    else " (perf gate disabled)",
                )
        self.debug_validated = True

    def get_dispatch_stats(self) -> dict[str, int | float]:
        total = self.graph_dispatch_count + self.eager_dispatch_count
        graph_fraction = (
            self.graph_dispatch_count / total if total > 0 else 0.0
        )
        return {
            "graph_dispatch_count": self.graph_dispatch_count,
            "eager_dispatch_count": self.eager_dispatch_count,
            "total_dispatch_count": total,
            "graph_dispatch_fraction": graph_fraction,
        }


_PREFILL_GRAPH_DISPATCHER_CACHE: dict[tuple[object, ...], PrefillGraphDispatcher] = {}
_PREFILL_GRAPH_DISPATCHER_CACHE_LOCK = threading.Lock()


def _get_or_create_prefill_graph_dispatcher(
    *,
    capture_sizes: list[tuple[int, int]],
    padded_heads: int,
    head_dim: int,
    scale: float,
    attn_sink: torch.Tensor,
    device: torch.device,
    flash_mla_sparse_fwd_fn,
    flashmla_bf16_io_fn,
    copy_flashmla_output_fn,
) -> PrefillGraphDispatcher:
    device_key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device()
        if device.type == "cuda" and torch.cuda.is_available()
        else None,
    )
    key = (
        device_key,
        tuple(capture_sizes),
        padded_heads,
        head_dim,
        float(scale),
        tuple(attn_sink.shape),
        str(attn_sink.dtype),
    )

    with _PREFILL_GRAPH_DISPATCHER_CACHE_LOCK:
        dispatcher = _PREFILL_GRAPH_DISPATCHER_CACHE.get(key)
        if dispatcher is not None:
            return dispatcher

        dispatcher = PrefillGraphDispatcher(
            capture_sizes=capture_sizes,
            padded_heads=padded_heads,
            head_dim=head_dim,
            scale=scale,
            attn_sink=attn_sink,
            device=device,
        )
        dispatcher.capture_graphs(
            flash_mla_sparse_fwd_fn,
            flashmla_bf16_io_fn,
            copy_flashmla_output_fn,
        )
        _PREFILL_GRAPH_DISPATCHER_CACHE[key] = dispatcher
        return dispatcher


def _flashmla_bf16_io(
    q: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_q = q if q.dtype is torch.bfloat16 else q.to(torch.bfloat16)
    flash_output = (
        output
        if output.dtype is torch.bfloat16
        else torch.empty_like(output, dtype=torch.bfloat16)
    )
    return flash_q, flash_output


def _copy_flashmla_output(
    flash_output: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if flash_output.data_ptr() != output.data_ptr() or flash_output.dtype != output.dtype:
        output.copy_(flash_output.to(output.dtype))


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream: torch.cuda.Stream | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128
        # SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        assert cap is not None, "DeepseekV4 attention requires a CUDA device"
        self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
        self._tma_aligned_scales = cap.major >= 10

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        self.aux_stream = mla_modules.aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.input", hidden_states)
        with _profile_or_null("wrapper.fused_wqa_wkv", hidden_states):
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.qr", qr)
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.kv", kv)

        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention (inside custom op for torch.compile boundary)
        with _profile_or_null("wrapper.attention_total", hidden_states):
            torch.ops.vllm.deepseek_v4_attention(
                hidden_states,
                qr,
                kv,
                positions,
                o_padded,
                self.layer_name,
            )
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.o_padded", o_padded)
        o = o_padded[:, : self.n_local_heads, :]
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.o", o)
        _trace_layer16_summary(self.prefix, "wrapper.o", o)

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        with _profile_or_null("wrapper.o_inv_rope_fp8_quant", o):
            o_fp8, o_scale = fused_inv_rope_fp8_quant(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim,
                rope_dim=self.rope_head_dim,
                tma_aligned_scales=self._tma_aligned_scales,
            )
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.o_scale", o_scale)
        _trace_layer16_summary(self.prefix, "wrapper.o_scale", o_scale)

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        # O1 fused path: combine `wrapper.o_fp8_einsum` + `wrapper.wo_b` into
        # a single Python-level call, eliminating the intermediate `z` global
        # memory roundtrip. Gated by VLLM_SM70_DEEPSEEK_V4_FUSE_O_WOB; only
        # used on the SM70 software-FP8 fallback path. SM80+ unchanged.
        if (
            _env_flag("VLLM_SM70_DEEPSEEK_V4_FUSE_O_WOB", default=True)
            and _should_use_torch_fp8_einsum_fallback(o_fp8)
        ):
            with _profile_or_null("wrapper.wo_b", o_fp8):
                out = _sm70_fused_o_einsum_wo_b(
                    o_fp8,
                    o_scale,
                    wo_a_fp8,
                    wo_a_scale,
                    self.wo_b,
                    "bhr,hdr->bhd",
                    hidden_states.dtype,
                )
            if _should_clamp_sm70_fp16_attention_output(out):
                _clamp_sm70_fp16_attention_output_(out)
            _trace_nonfinite_tensor(f"{self.prefix}.wrapper.wo_b", out)
            _trace_layer16_summary(self.prefix, "wrapper.wo_b", out)
            return out

        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=hidden_states.dtype,
        )
        with _profile_or_null("wrapper.o_fp8_einsum", z):
            torch.ops.vllm.deepseek_v4_fp8_einsum(
                o_fp8,
                o_scale,
                wo_a_fp8,
                wo_a_scale,
                z,
                "bhr,hdr->bhd",
                list(self._einsum_recipe),
            )
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.z", z)
        _trace_layer16_summary(self.prefix, "wrapper.z", z)

        with _profile_or_null("wrapper.wo_b", z):
            out = self.wo_b(z.flatten(1))
        if isinstance(out, tuple):
            out = out[0]
        if _should_clamp_sm70_fp16_attention_output(out):
            _clamp_sm70_fp16_attention_output_(out)
        _trace_nonfinite_tensor(f"{self.prefix}.wrapper.wo_b", out)
        _trace_layer16_summary(self.prefix, "wrapper.wo_b", out)
        return out

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        with _profile_or_null("impl.q_kv_rmsnorm", qr):
            qr, kv = fused_q_kv_rmsnorm(
                qr,
                kv,
                self.q_norm.weight.data,
                self.kv_norm.weight.data,
                self.eps,
            )
        _trace_nonfinite_tensor(f"{self.prefix}.impl.qr_norm", qr)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.kv_norm", kv)
        with _profile_or_null("impl.q_proj", qr):
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.q_proj", q)

        # Overlap kv_insert with whichever of indexer/compressor is present.
        # Indexer implies compressor; when both exist, compressor rides on the
        # aux stream alongside kv_insert so the heavy indexer owns default.
        if self.indexer is not None:
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def kv_insert_and_compress() -> None:
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                compressor(hidden_states, positions, self.rotary_emb)

            with _profile_or_null("impl.indexer_kv_compress_overlap", q):
                maybe_execute_in_parallel(
                    lambda: indexer(
                        hidden_states, qr, positions, self.indexer_rotary_emb
                    ),
                    kv_insert_and_compress,
                    self.ln_events[0],
                    self.ln_events[1],
                    self.aux_stream,
                )
        elif self.compressor is not None:
            # Compressor on default, kv_insert on aux.
            compressor = self.compressor
            with _profile_or_null("impl.compressor_kv_insert_overlap", q):
                maybe_execute_in_parallel(
                    lambda: compressor(hidden_states, positions, self.rotary_emb),
                    lambda: self._fused_qnorm_rope_kv_insert(
                        q, kv, positions, attn_metadata
                    ),
                    self.ln_events[0],
                    self.ln_events[1],
                    self.aux_stream,
                )
        else:
            # SWA-only layer: no compressor, no overlap.
            with _profile_or_null("impl.kv_insert", q):
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.q_after_insert", q)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            # Reserve _forward_prefill's bf16-gather workspace; the dummy
            # run returns before mla_attn runs, so without this the shared
            # workspace locks below the real prefill size.
            sub = self.mla_attn
            swa_only = sub.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (sub.max_model_len + sub.compress_ratio - 1) // sub.compress_ratio
            )
            M = N + sub.window_size + sub.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            out.zero_()
            return

        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        with _profile_or_null("impl.mla_attn_total", q):
            self.mla_attn(q, kv, positions, output=out)
        _trace_nonfinite_tensor(f"{self.prefix}.impl.mla_out", out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        if _should_use_qnorm_rope_kv_insert_fallback(q):
            # Use the custom-op version when available (opts us out of the
            # torch inductor 2.9 `decompose_triton_kernel_wrapper_functional`
            # pass that mishandles dynamic-shape stride expressions).
            if _sm70_triton_qnorm_rope_kv_insert_op is not None:
                _sm70_triton_qnorm_rope_kv_insert_op(
                    q,
                    kv,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions.to(torch.int64),
                    self.rotary_emb.cos_sin_cache,
                    self.eps,
                    swa_metadata.block_size,
                )
            else:
                _sm70_triton_qnorm_rope_kv_insert(
                    q,
                    kv,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions.to(torch.int64),
                    self.rotary_emb.cos_sin_cache,
                    self.eps,
                    swa_metadata.block_size,
                )
        else:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, qr, kv, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if _should_use_torch_fp8_einsum_fallback(a):
        _sm70_fp8_einsum_bmm(a, a_scale, b, b_scale, out, equation)
        return
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def _should_use_torch_fp8_einsum_fallback(a: torch.Tensor) -> bool:
    if not has_deep_gemm():
        return True
    if not a.is_cuda:
        return False
    return torch.cuda.get_device_capability(a.device)[0] < 8


def _deepseek_v4_fp8_einsum_torch_fallback(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(
            "DeepSeek V4 torch fp8 einsum fallback only supports "
            f"'bhr,hdr->bhd', got {equation!r}."
        )

    groups = a.shape[1]
    hidden = a.shape[2]
    rank = b.shape[1] if b.dim() == 3 else b.shape[0] // groups
    b_3d = b.reshape(groups, rank, hidden)

    a_blocks = a_scale.shape[-1]
    weight_scale_shape = (groups, rank // 128, hidden // 128)
    b_scale_3d = b_scale.reshape(weight_scale_shape)
    a_deq = a.float() * a_scale.repeat_interleave(hidden // a_blocks, dim=-1)
    b_deq = b_3d.float() * b_scale_3d.repeat_interleave(
        128, dim=1
    ).repeat_interleave(128, dim=2)
    result = torch.einsum(equation, a_deq, b_deq)
    out.copy_(result.to(out.dtype))


def _sm70_fp8_einsum_bmm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    """SM70 optimised path: pre-dequant weight to fp16 (cached) + fp16 einsum.

    The wo_a weight (b) is dequantized to fp16 once and cached on the tensor
    (~43 MB extra VRAM).  At runtime, the activation (a) is dequanted to fp16,
    then einsum runs in fp16 for halved bandwidth vs the previous fp32 path.
    Eliminates repeated weight dequant (~1.5x faster than re-dequant every call).
    """
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(
            "SM70 fp8 einsum only supports 'bhr,hdr->bhd', got "
            f"{equation!r}."
        )

    groups = a.shape[1]
    hidden = a.shape[2]
    rank = b.shape[1] if b.dim() == 3 else b.shape[0] // groups

    # Lazily pre-dequant b (weight) to fp16 and cache
    b_f16 = getattr(b, "_sm70_predequant_f16", None)
    if b_f16 is None:
        b_3d = b.reshape(groups, rank, hidden)
        weight_scale_shape = (groups, rank // 128, hidden // 128)
        b_scale_3d = b_scale.reshape(weight_scale_shape)
        b_f16 = (
            b_3d.float()
            * b_scale_3d.repeat_interleave(128, dim=1).repeat_interleave(
                128, dim=2
            )
        ).half().contiguous()  # [groups, rank, hidden]
        b._sm70_predequant_f16 = b_f16  # type: ignore[attr-defined]

    # Dequant a (activation) to fp16
    a_blocks = a_scale.shape[-1]
    a_deq = a.float() * a_scale.repeat_interleave(
        hidden // a_blocks, dim=-1
    )  # [T, G, D] fp32

    # fp16 einsum — halved bandwidth, acceptable precision (rtol < 1e-3)
    result = torch.einsum("bhr,hdr->bhd", a_deq.half(), b_f16)
    out.copy_(result.to(out.dtype))


def _sm70_fused_o_einsum_wo_b(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    wo_b_module: torch.nn.Module,
    equation: str,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """O1 fix: fused SM70 FP8 software-einsum + wo_b RowParallelLinear.

    Mirrors the math of `_sm70_fp8_einsum_bmm` followed by
    ``wo_b(z.flatten(1))`` but eliminates the explicit ``z`` global-memory
    materialization and the cross-Python-call boundary, allowing PyTorch
    to keep the einsum result in caches before the wo_b matmul. Returns
    the per-rank partial output (TP all-reduce contract unchanged).

    Gated by ``VLLM_SM70_DEEPSEEK_V4_FUSE_O_WOB`` (default on).
    """
    if equation != "bhr,hdr->bhd":
        raise RuntimeError(
            "SM70 fused o_einsum+wo_b only supports 'bhr,hdr->bhd', "
            f"got {equation!r}."
        )

    groups = a.shape[1]
    hidden = a.shape[2]
    rank = b.shape[1] if b.dim() == 3 else b.shape[0] // groups

    # Reuse the same lazily cached fp16 weight as `_sm70_fp8_einsum_bmm`
    # so both paths share the (~43 MB) pre-dequant cost.
    b_f16 = getattr(b, "_sm70_predequant_f16", None)
    if b_f16 is None:
        b_3d = b.reshape(groups, rank, hidden)
        weight_scale_shape = (groups, rank // 128, hidden // 128)
        b_scale_3d = b_scale.reshape(weight_scale_shape)
        b_f16 = (
            b_3d.float()
            * b_scale_3d.repeat_interleave(128, dim=1).repeat_interleave(
                128, dim=2
            )
        ).half().contiguous()
        b._sm70_predequant_f16 = b_f16  # type: ignore[attr-defined]

    # R4: fused FP8 a dequant -> FP16 in one Triton pass, skipping the
    # fp32 materialization and `repeat_interleave` scale expansion that
    # the old path used.
    a_fp16 = sm70_fp8_a_dequant_to_fp16(a, a_scale)

    # Chained: einsum -> flatten(1) -> wo_b. Keep result in fp16 to match
    # `out` dtype contract; wo_b applies its own (FP8 or fp16) GEMM.
    z = torch.einsum("bhr,hdr->bhd", a_fp16, b_f16).to(out_dtype)
    out = wo_b_module(z.flatten(1))
    if isinstance(out, tuple):
        out = out[0]
    return out


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        # DeepseekV4 only supports fp8 kv-cache format for now
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 only supports fp8 kv-cache format for now, "
            f"got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for DeepseekV4 for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

        # Prefill CUDA graph dispatcher (partial capture of attention kernel)
        self._prefill_graph_dispatcher: PrefillGraphDispatcher | None = None
        if _PREFILL_CUDAGRAPH_ENABLED:
            N = (
                (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            # Capture token-count buckets, not request-count buckets. A
            # single-request prefill can contain thousands of tokens, while
            # PREFILL_CHUNK_SIZE only bounds how many requests are gathered at
            # once. Override with VLLM_PREFILL_CUDAGRAPH_CAPTURE_TOKENS=...
            # for narrower experiments.
            capture_sizes = _default_prefill_cudagraph_capture_sizes(
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_model_len=self.max_model_len,
                max_M=M,
            )
            self._prefill_graph_dispatcher = _get_or_create_prefill_graph_dispatcher(
                capture_sizes=capture_sizes,
                padded_heads=self.padded_heads,
                head_dim=head_dim,
                scale=scale,
                attn_sink=attn_sink,
                device=attn_sink.device,
                flash_mla_sparse_fwd_fn=flash_mla_sparse_fwd,
                flashmla_bf16_io_fn=_flashmla_bf16_io,
                copy_flashmla_output_fn=_copy_flashmla_output,
            )
            if (
                _PREFILL_CUDAGRAPH_DEBUG
                and not self._prefill_graph_dispatcher.debug_validated
            ):
                self._prefill_graph_dispatcher.warmup_validate(
                    flash_mla_sparse_fwd,
                    _flashmla_bf16_io,
                    _copy_flashmla_output,
                )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    # NOTE: _forward_decode is entirely unaffected by the prefill CUDA graph
    # feature.  PrefillGraphDispatcher operates only within _forward_prefill's
    # chunk loop; the decode path (including the existing CudagraphDispatcher
    # for full-model decode graphs) remains unchanged.
    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                with _profile_or_null("decode.compute_global_topk", q):
                    global_indices, topk_lens = compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        with _profile_or_null("decode.flashmla_bf16_io", q):
            q, flash_output = _flashmla_bf16_io(q, output)
        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        if _should_use_sm70_decode_prefill_fallback(q, swa_only):
            if swa_only:
                swa_topk = swa_indices.shape[-1]
                fallback_kv = _get_decode_prefill_fallback_workspace(
                    (num_decode_tokens, swa_topk, q.shape[-1]),
                    torch.bfloat16,
                    q.device,
                )
                with _profile_or_null("decode.fallback_gather.swa", q):
                    fallback_kv, fallback_indices, fallback_lens = (
                        _gather_decode_prefill_fallback_kv_with_indices_(
                            fallback_kv,
                            self.swa_cache_layer.kv_cache,
                            swa_indices,
                            swa_lens,
                            swa_metadata.block_size,
                        )
                    )
                fallback_topk_length: torch.Tensor | None = fallback_lens
            else:
                assert kv_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                compressed_topk = topk_indices.shape[-1]
                swa_topk = swa_indices.shape[-1]
                total_topk = compressed_topk + swa_topk
                fallback_topk_length = None
                fallback_kv = _get_decode_prefill_fallback_workspace(
                    (num_decode_tokens, total_topk, q.shape[-1]),
                    torch.bfloat16,
                    q.device,
                )
                compressed_kv = fallback_kv[:, :compressed_topk]
                with _profile_or_null("decode.fallback_gather.compressed", q):
                    compressed_kv, compressed_indices, _ = (
                        _gather_decode_prefill_fallback_kv_with_indices_(
                            compressed_kv,
                            kv_cache,
                            topk_indices,
                            topk_lens,
                            attn_metadata.block_size // self.compress_ratio,
                            row_stride=total_topk,
                        )
                    )
                swa_kv = fallback_kv[:, compressed_topk:]
                with _profile_or_null("decode.fallback_gather.swa", q):
                    swa_kv, swa_fallback_indices, _ = (
                        _gather_decode_prefill_fallback_kv_with_indices_(
                            swa_kv,
                            self.swa_cache_layer.kv_cache,
                            swa_indices,
                            swa_lens,
                            swa_metadata.block_size,
                            row_stride=total_topk,
                            offset=compressed_topk,
                        )
                    )
                fallback_indices = torch.cat(
                    (compressed_indices, swa_fallback_indices), dim=-1
                )
            _normalize_flashmla_sm70_prefill_kv_(fallback_kv)
            with _profile_or_null("decode.attn.fallback_sparse_prefill", q):
                flash_output, _, _ = flash_mla_sparse_fwd(
                    q=q.squeeze(1),
                    kv=fallback_kv.view(-1, 1, q.shape[-1]),
                    indices=fallback_indices,
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=fallback_topk_length,
                    out=flash_output,
                )
                _copy_flashmla_output(flash_output, output)
            return

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        with _profile_or_null("decode.attn.direct_flashmla", q):
            out, _ = flash_mla_with_kvcache(
                q=q,
                k_cache=swa_cache,
                block_table=None,
                head_dim_v=512,
                tile_scheduler_metadata=tile_metadata,
                cache_seqlens=None,
                is_fp8_kvcache=True,
                indices=swa_indices,
                topk_length=swa_lens,
                softmax_scale=self.scale,
                attn_sink=self.attn_sink,
                extra_k_cache=kv_cache if not swa_only else None,
                extra_indices_in_kvcache=topk_indices,
                extra_topk_length=topk_lens,
                out=flash_output.unsqueeze(1),
            )
            _copy_flashmla_output(out.squeeze(1), output)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE
        trace_prefill = (
            os.getenv("VLLM_DEEPSEEK_V4_NAN_TRACE", "0") == "1"
            and self.prefix.endswith("layers.1.attn")
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        # Aux stream operations (KV-insert, compressor) launched via
        # maybe_execute_in_parallel in attention_impl have already completed
        # by the time we reach here — event synchronization in that helper
        # ensures the default stream observes their writes.  The chunk loop
        # below therefore runs entirely on the default stream with no
        # cross-stream hazards.
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                with _profile_or_null(
                    "prefill.compressed_gather",
                    q,
                    extra={
                        "chunk_idx": chunk_idx,
                        "num_chunk_tokens": int(chunk_size),
                    },
                ):
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                    )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            with _profile_or_null(
                "prefill.swa_gather",
                q,
                extra={
                    "chunk_idx": chunk_idx,
                    "num_chunk_tokens": int(chunk_size),
                },
            ):
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                )
            kv_chunk = kv[:chunk_size]
            _normalize_flashmla_sm70_prefill_kv_(kv_chunk)
            if trace_prefill:
                _trace_tensor_summary(f"{self.prefix}.prefill.kv", kv_chunk)
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.seq_lens",
                    seq_lens[chunk_start:chunk_end],
                )
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.gather_lens",
                    gather_lens[chunk_start:chunk_end],
                )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            with _profile_or_null(
                "prefill.combine_indices",
                q,
                extra={
                    "chunk_idx": chunk_idx,
                    "num_chunk_tokens": int(chunk_size),
                },
            ):
                combined_indices, combined_lens = combine_topk_swa_indices(
                    topk_indices[query_start:query_end],
                    query_start_loc[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ],
                    seq_lens[chunk_start:chunk_end],
                    gather_lens[chunk_start:chunk_end],
                    self.window_size,
                    self.compress_ratio,
                    top_k,
                    M,
                    N,
                )
            if trace_prefill:
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.combined_indices", combined_indices
                )
                _trace_tensor_summary(
                    f"{self.prefix}.prefill.combined_lens", combined_lens
                )

            output_slice = output[query_start:query_end]
            num_chunk_tokens = query_end - query_start

            # Attempt prefill CUDA graph replay for the attention kernel.
            # KV gather and index combination above remain eager; only the
            # _flashmla_bf16_io + flash_mla_sparse_fwd + _copy_flashmla_output
            # sequence is eligible for graph replay.
            graph_replayed = False
            if self._prefill_graph_dispatcher is not None:
                with _profile_or_null("prefill.flashmla_graph_replay", q):
                    graph_replayed = self._prefill_graph_dispatcher.try_graph_replay(
                        q_chunk=q[query_start:query_end],
                        kv_flat=kv.view(-1, 1, q.shape[-1]),
                        combined_indices=combined_indices.unsqueeze(1),
                        combined_lens=combined_lens,
                        output_slice=output_slice,
                        num_chunk_tokens=num_chunk_tokens,
                        num_chunk_reqs=chunk_size,
                        M=M,
                        attn_sink=self.attn_sink,
                    )

            # Debug mode: run eager too and compare outputs
            if _PREFILL_CUDAGRAPH_DEBUG and graph_replayed:
                graph_output = output_slice.clone()
                q_chunk_dbg, output_chunk_dbg = _flashmla_bf16_io(
                    q[query_start:query_end], output_slice,
                )
                flash_out_dbg, _, _ = flash_mla_sparse_fwd(
                    q=q_chunk_dbg,
                    kv=kv.view(-1, 1, q.shape[-1]),
                    indices=combined_indices.unsqueeze(1),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_length=combined_lens,
                    out=output_chunk_dbg,
                )
                _copy_flashmla_output(flash_out_dbg, output_slice)
                if not torch.equal(graph_output, output_slice):
                    max_diff = (
                        (graph_output.float() - output_slice.float())
                        .abs().max().item()
                    )
                    logger.warning(
                        "Prefill graph debug: output divergence at %s "
                        "chunk %d: max_diff=%.6e — using eager output",
                        self.prefix, chunk_idx, max_diff,
                    )
                else:
                    output_slice.copy_(graph_output)

            if not graph_replayed:
                if _DEEPSEEK_V4_PROFILE_ENABLED or _DEEPSEEK_V4_PROFILE_NVTX:
                    try:
                        combined_lens_max = int(combined_lens.max().item())
                    except Exception:
                        combined_lens_max = None
                else:
                    combined_lens_max = None
                with _profile_or_null(
                    "prefill.flashmla_sparse_fwd",
                    q,
                    extra={
                        "chunk_idx": chunk_idx,
                        "num_chunk_tokens": int(num_chunk_tokens),
                        "combined_lens_max": combined_lens_max,
                    },
                ):
                    q_chunk, output_chunk = _flashmla_bf16_io(
                        q[query_start:query_end],
                        output_slice,
                    )
                    if trace_prefill:
                        _trace_tensor_summary(f"{self.prefix}.prefill.q_chunk", q_chunk)
                    flash_output, max_logits, lse = flash_mla_sparse_fwd(
                        q=q_chunk,
                        kv=kv.view(-1, 1, q.shape[-1]),
                        indices=combined_indices.unsqueeze(1),
                        sm_scale=self.scale,
                        attn_sink=self.attn_sink,
                        topk_length=combined_lens,
                        out=output_chunk,
                    )
                    _copy_flashmla_output(flash_output, output_slice)
                if trace_prefill:
                    _trace_tensor_summary(
                        f"{self.prefix}.prefill.flash_output", flash_output
                    )
                    _trace_tensor_summary(
                        f"{self.prefix}.prefill.max_logits", max_logits
                    )
                    _trace_tensor_summary(f"{self.prefix}.prefill.lse", lse)


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        # O4 fix: Allow override of the indexer top-K via
        # VLLM_DEEPSEEK_V4_INDEXER_TOPK. Must not exceed the model config's
        # index_topk (which fixes the topk_indices_buffer allocation width).
        # Default override value is 0, which falls back to the model config.
        _cfg_topk = config.index_topk
        _override_topk = envs.VLLM_DEEPSEEK_V4_INDEXER_TOPK
        if _override_topk and 0 < _override_topk <= _cfg_topk:
            self.topk_tokens = _override_topk
            if _override_topk != _cfg_topk:
                logger.info_once(
                    "O4: indexer top-K overridden from %d to %d via "
                    "VLLM_DEEPSEEK_V4_INDEXER_TOPK",
                    _cfg_topk,
                    _override_topk,
                )
        else:
            self.topk_tokens = _cfg_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lighening Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(hidden_states, positions, rotary_emb)
        weights, _ = self.weights_proj(hidden_states)
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
