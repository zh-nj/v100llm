# DeepSeek V4 FlashMLA SM70 CUDAGraph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the current DeepSeek V4 Flash branch is using the SM70 FlashMLA implementation from `/mnt/data/apps/FlashMLA`, then make `/mnt/data6/models/DeepSeek-V4-Flash` run correctly and efficiently with decode CUDA Graphs on V100-class GPUs.

**Architecture:** Treat the finished eager DeepSeek V4 SM70 bring-up as the baseline, and add a second gated path for CUDA Graph inference. Keep prefill/model-load correctness conservative, use `FULL_DECODE_ONLY` first so decode can be graphed while prefill remains eager, and only enable faster SM70 paths when exact semantic canaries pass.

**Tech Stack:** Python 3.13, Conda `gptq`, CUDA 12.8, vLLM V1, `CUDAGraphMode.FULL_DECODE_ONLY`, local FlashMLA at `/mnt/data/apps/FlashMLA`, local model `/mnt/data6/models/DeepSeek-V4-Flash`, 8x V100 via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9`.

---

## Current Evidence

- Repository root `/mnt/data/apps/1Cat-vLLM` is on branch `chore/ignore-project-worktrees`; the DeepSeek V4 target worktree is `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split` on branch `feature/vllm-0190-upstream-split`.
- Existing bring-up plan: `docs/superpowers/plans/2026-05-01-deepseek-v4-flashmla-sm70.md`.
- Local FlashMLA source: `/mnt/data/apps/FlashMLA`, branch `feature/sm70-volta-flashmla`, inspected HEAD `a507081`.
- Eager DeepSeek V4 smoke and later CUDA Graph runs are semantically correct with
  the current default `VLLM_SM70_MHC_FAST=1`; setting it to `0` is now a fallback
  A/B mode and materially lowers decode throughput.
- FlashMLA sparse backend and sparse runtime gate already accept SM70, and `_flashmla_C` can be built from the local FlashMLA source.
- CUDA Graph design says FlashMLA sparse supports uniform batches, so the first production target should be `FULL_DECODE_ONLY`: eager prefill plus full CUDA Graph replay for uniform decode.

## File Map

- Modify: `benchmarks/deepseek_v4_flashmla_sm70.py`
  - Add source/build provenance to `--mode inspect`.
  - Add graph-aware stream benchmarking fields: requested graph mode, observed capture evidence, and warmup-vs-measured decode speed.
- Modify: `tests/benchmarks/test_deepseek_v4_flashmla_sm70.py`
  - Unit-test the new inspect/provenance fields and generated-token accounting.
- Create: `tests/v1/cudagraph/test_deepseek_v4_flashmla_sm70_cudagraph.py`
  - Static and lightweight unit coverage that `FlashMLASparseBackend`, `DeepseekV4FlashMLASparseBackend`, and `DeepseekSparseSWABackend` resolve to CUDA Graph support compatible with `FULL_DECODE_ONLY`.
- Modify when capture exposes a blocker: `vllm/model_executor/layers/deepseek_v4_attention.py`
  - Keep SM70 decode fallback workspaces graph-stable and avoid graph-capture-host sync in decode.
- Modify when capture exposes a blocker: `vllm/model_executor/layers/deepseek_compressor.py`
  - Keep prefill-only torch fallbacks outside the decode graph path; replace decode-time host-sync loops only if the capture trace proves they execute during decode capture.
- Modify when performance canary fails: `vllm/model_executor/layers/mhc.py`
  - Keep `VLLM_SM70_MHC_FAST=1` as the current validated default. Use
    `VLLM_SM70_MHC_FAST=0` only for fallback A/B or rollback tests.
- Modify: `docs/models/supported_models.md`
  - Extend the local SM70 note with CUDA Graph launch and acceptance boundaries after successful validation.

## Runtime Contract

Use this environment for all build and runtime checks:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
export FLASH_MLA_SRC_DIR=/mnt/data/apps/FlashMLA
export FLASH_MLA_ENABLE_SM70=1
export FLASH_MLA_DISABLE_SM100=1
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_USE_V1=1
export VLLM_SM70_MHC_FAST=1
```

The initial CUDA Graph server launch must remove `--enforce-eager` and use:

```bash
--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

For `max-num-seqs=1`, capture size `1` is the correct first decode graph key. Broaden to `[1,2,4]` only after the single-sequence path is correct and useful.

## Task 1: Confirm The Active Branch Uses `/mnt/data/apps/FlashMLA`

**Files:**
- Modify: `benchmarks/deepseek_v4_flashmla_sm70.py`
- Modify: `tests/benchmarks/test_deepseek_v4_flashmla_sm70.py`
- Verify: `vllm/_flashmla_C*.so`

- [x] **Step 1: Record current worktree and FlashMLA source provenance**

Run:

```bash
cd /mnt/data/apps/1Cat-vLLM
git worktree list --porcelain
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split rev-parse --abbrev-ref HEAD
git -C /mnt/data/apps/FlashMLA rev-parse --abbrev-ref HEAD
git -C /mnt/data/apps/FlashMLA rev-parse --short HEAD
git -C /mnt/data/apps/FlashMLA log -1 --oneline
```

Expected:

- Target branch is `feature/vllm-0190-upstream-split`.
- FlashMLA branch is `feature/sm70-volta-flashmla`.
- FlashMLA HEAD is recorded in the task log before rebuilding vLLM.

- [x] **Step 2: Extend inspect mode with source/build evidence**

Add these fields to the JSON returned by `benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json`:

```python
"flashmla_source": {
    "path": "/mnt/data/apps/FlashMLA",
    "branch": "<git branch --show-current output>",
    "head": "<git rev-parse --short HEAD output>",
    "has_sm70_sparse_decode_sources": True,
    "has_sm70_sparse_prefill_sources": True
},
"flashmla_runtime": {
    "vllm_file": "<vllm.__file__>",
    "flashmla_core_importable": True,
    "sparse_supported": [True, None]
}
```

Use `subprocess.run(..., check=False, text=True, capture_output=True)` for git provenance, and keep inspect usable even when `_flashmla_C` is not built by returning `flashmla_core_importable: false` plus the import error string.

- [x] **Step 3: Add unit coverage for provenance fields**

Extend `tests/benchmarks/test_deepseek_v4_flashmla_sm70.py` with a monkeypatched test that proves inspect mode reports:

```python
assert report["flashmla_source"]["path"] == "/mnt/data/apps/FlashMLA"
assert "flashmla_runtime" in report
assert "flashmla_core_importable" in report["flashmla_runtime"]
```

- [x] **Step 4: Run focused tests**

Run:

```bash
pytest tests/benchmarks/test_deepseek_v4_flashmla_sm70.py -q
python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json
```

Expected:

- Unit tests pass.
- Inspect reports the target worktree path in `vllm_file`.
- Inspect reports the local FlashMLA source path and HEAD.

- [x] **Step 5: Rebuild and re-import with the local FlashMLA source**

Run:

```bash
python -m pip install -e . --no-build-isolation -v
python - <<'PY'
import vllm
print(vllm.__file__)
import vllm._flashmla_C
print("flashmla_core_imported")
from vllm.v1.attention.ops import flashmla
print("sparse_supported", flashmla.is_flashmla_sparse_supported())
PY
```

Expected:

- Build logs mention `/mnt/data/apps/FlashMLA`.
- `vllm.__file__` starts with `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/`.
- `_flashmla_C` imports successfully.

## Task 2: Establish The CUDA Graph Baseline Without Changing Kernels

**Files:**
- Modify: `benchmarks/deepseek_v4_flashmla_sm70.py`
- Create: `tests/v1/cudagraph/test_deepseek_v4_flashmla_sm70_cudagraph.py`
- Verify: server log for graph capture and replay

- [x] **Step 1: Add static CUDA Graph readiness tests**

Create `tests/v1/cudagraph/test_deepseek_v4_flashmla_sm70_cudagraph.py`:

```python
from unittest.mock import MagicMock

from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import BatchDescriptor
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend


def _make_vllm_config_for_full_decode_only():
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        mode=CompilationMode.NONE,
        cudagraph_capture_sizes=[1],
    )
    compilation_config.max_cudagraph_capture_size = 1
    compilation_config.post_init_cudagraph_sizes()

    config = MagicMock(spec=VllmConfig)
    config.compilation_config = compilation_config
    config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=1)
    config.parallel_config = ParallelConfig()
    config.speculative_config = None
    config.lora_config = None
    return config


def test_deepseek_v4_flashmla_sparse_backends_allow_uniform_decode_cudagraph():
    assert (
        FlashMLASparseBackend.get_builder_cls().get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        DeepseekV4FlashMLASparseBackend.get_builder_cls().get_cudagraph_support(
            None, None
        )
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        DeepseekSparseSWABackend.get_builder_cls().get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_full_decode_only_dispatches_single_token_uniform_decode(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda *args, **kwargs: DeviceCapability(7, 0),
    )
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    dispatcher = CudagraphDispatcher(_make_vllm_config_for_full_decode_only())
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 1)

    mode, desc = dispatcher.dispatch(num_tokens=1, uniform_decode=True)

    assert mode == CUDAGraphMode.FULL
    assert desc == BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True)
```

- [x] **Step 2: Run static graph tests**

Run:

```bash
pytest tests/v1/cudagraph/test_deepseek_v4_flashmla_sm70_cudagraph.py -q
```

Expected: tests pass before any runtime graph capture work begins.

- [x] **Step 3: Launch the eager baseline server for comparison**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9 \
VLLM_USE_V1=1 \
VLLM_SM70_MHC_FAST=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --served-model-name /mnt/data6/models/DeepSeek-V4-Flash \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --enforce-eager \
  --port 18080
```

In a second shell, run:

```bash
python benchmarks/deepseek_v4_flashmla_sm70.py \
  --mode openai-stream \
  --endpoint http://127.0.0.1:18080/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "请只输出这五个字符，不要输出其它内容：ZX-42" \
  --max-tokens 16 \
  --timeout 600
```

Expected:

- `output_preview` is exactly `ZX-42` or has the exact answer followed only by an allowed stop marker.
- `finish_reason=stop`.
- The JSON records `TTFT`, `decode_window_s`, and `decode_tokens_per_s` from streaming timestamps.

- [x] **Step 4: Launch the first CUDA Graph server**

Stop the eager server, then run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9 \
VLLM_USE_V1=1 \
VLLM_SM70_MHC_FAST=1 \
VLLM_LOGGING_LEVEL=DEBUG \
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --served-model-name /mnt/data6/models/DeepSeek-V4-Flash \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --port 18080
```

Expected server log evidence:

- No `--enforce-eager` path.
- The resolved mode remains `FULL_DECODE_ONLY` or records a concrete downgrade reason.
- `Graph capturing finished` appears.
- Capture progress includes a decode/full graph for `num_tokens=1`.

- [x] **Step 5: Run semantic and speed smoke on the graph server**

Run both prompts:

```bash
python benchmarks/deepseek_v4_flashmla_sm70.py \
  --mode openai-stream \
  --endpoint http://127.0.0.1:18080/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "请只输出这五个字符，不要输出其它内容：ZX-42" \
  --max-tokens 16 \
  --timeout 600

python benchmarks/deepseek_v4_flashmla_sm70.py \
  --mode openai-stream \
  --endpoint http://127.0.0.1:18080/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "你好，请用一句话说明你是谁。" \
  --max-tokens 64 \
  --timeout 600
```

Expected:

- `ZX-42` canary remains exact.
- Identity prompt is coherent.
- `decode_tokens_per_s` is not worse than the eager baseline after capture warmup.
- Server logs show graph replay for decode, not only graph capture.

## Task 3: Fix The First CUDA Graph Capture Or Replay Blocker

**Files:**
- Modify as needed: `vllm/model_executor/layers/deepseek_v4_attention.py`
- Modify as needed: `vllm/model_executor/layers/deepseek_compressor.py`
- Modify as needed: `vllm/model_executor/models/deepseek_v4.py`
- Modify as needed: `vllm/model_executor/layers/mhc.py`
- Test: the smallest focused regression for the first failing stack

- [x] **Step 1: Classify the first failure**

Use this failure taxonomy before editing code:

- Graph disabled: config resolves to `NONE` or `PIECEWISE` unexpectedly.
- Capture unsupported: `cudaErrorStreamCaptureUnsupported`, host sync, `.item()`, allocator, event, or NCCL capture failure during `capture_model()`.
- Replay mismatch: graph captures but streaming output diverges from eager canaries.
- Performance miss: graph captures and output is correct, but steady-state decode speed does not improve after warmup.

- [x] **Step 2: Capture the exact first stack and line**

Save the server command, log excerpt, and first stack frame in the task notes. If the stack is in SM70 fallback code, add a focused regression near the existing SM70 test:

- `tests/model_executor/test_deepseek_v4_attention_sm70.py` for decode fallback workspace or FlashMLA prefill fallback shape.
- `tests/model_executor/test_mhc_fallback.py` for mHC fast/fallback semantics.
- `tests/model_executor/test_deepseek_v4_moe.py` for TP all-reduce or MoE output routing.

- [x] **Step 3: Patch only the proven blocker**

Prefer these fixes by failure class:

- For decode fallback allocations, route through `current_workspace_manager().get_simultaneous(...)` and keep tensor shapes tied to captured `BatchDescriptor`.
- For decode-time host sync, replace `.item()`/Python loops with Triton or tensorized code only on the path proven to execute during decode capture.
- For NCCL/all-reduce capture failures, use the existing vLLM graph-capture communicator path instead of introducing a separate synchronization primitive.
- For an mHC semantic mismatch, explicitly set `VLLM_SM70_MHC_FAST=0` as a
  rollback/fallback A/B mode, then repair the fast path under a full-model
  semantic canary before restoring the validated default.

- [x] **Step 4: Re-run the focused test and graph server smoke**

Run:

```bash
pytest tests/model_executor/test_deepseek_v4_attention_sm70.py \
  tests/model_executor/test_mhc_fallback.py \
  tests/model_executor/test_deepseek_v4_moe.py -q
```

Then re-run the `FULL_DECODE_ONLY` server launch and both OpenAI-stream prompts from Task 2.

Expected:

- Focused tests pass.
- CUDA Graph capture succeeds or reaches the next classified blocker.
- Semantic canaries stay correct.

Execution notes:

- First graph launch with only `cudagraph_mode=FULL_DECODE_ONLY` resolved to
  `CompilationMode.VLLM_COMPILE` and failed before capture in Torch Inductor
  (`decompose_triton_kernel_wrapper_functional` assertion). The validated graph
  launch sets `mode=NONE`; full decode CUDA Graphs do not require Inductor.
- The first real capture blocker was `cudaErrorStreamCaptureUnsupported` from
  `vllm/model_executor/layers/sparse_attn_indexer.py` in
  `_fp8_paged_mqa_logits_torch_fallback`, where the SM70 no-DeepGEMM fallback
  used `context_lens_2d[...].item()` during decode capture.
- Patch: route SM70 FP8 paged decode fallback through the existing Triton
  `sm70_fp8_paged_mqa_logits` path and fix its 1D `context_lens` semantics.
  Focused verification:
  `CUDA_VISIBLE_DEVICES=2 pytest tests/kernels/attention/test_sparse_attn_indexer_sm70_fallback.py -q`
  passed with `4 passed`.
- Runtime verification: graph server reached `Application startup complete`;
  logs showed `Capturing CUDA graphs (decode, FULL)`,
  `Graph capturing finished in 2 secs`, and request-time
  `cudagraph_mode: FULL` for
  `BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True)`.

## Task 4: Turn CUDA Graph Correctness Into A Performance Gate

**Files:**
- Modify: `benchmarks/deepseek_v4_flashmla_sm70.py`
- Modify: `docs/models/supported_models.md`
- Optional create: `benchmarks/run_deepseek_v4_flashmla_sm70_cudagraph.sh`

- [x] **Step 1: Add a repeatable benchmark mode or wrapper**

The benchmark must report one JSON object per run with these fields:

```json
{
  "mode": "eager|full_decode_only",
  "prompt_name": "zx42|identity|long_decode",
  "TTFT": 0.0,
  "decode_window_s": 0.0,
  "completion_tokens": 0,
  "decode_tokens_per_s": 0.0,
  "finish_reason": "stop",
  "output_preview": "..."
}
```

Do not use framework throughput fields for decode speed; compute it from streaming first-token and final-token wall-clock timestamps.

- [ ] **Step 2: Measure eager baseline and graph mode after warmup**

Run each prompt twice and keep the second run as steady state:

```bash
python benchmarks/deepseek_v4_flashmla_sm70.py --mode openai-stream \
  --endpoint http://127.0.0.1:18080/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "请只输出这五个字符，不要输出其它内容：ZX-42" \
  --max-tokens 16 --timeout 600

python benchmarks/deepseek_v4_flashmla_sm70.py --mode openai-stream \
  --endpoint http://127.0.0.1:18080/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "请写一段约200字的中文说明，介绍CUDA Graph为什么能降低decode阶段开销。" \
  --max-tokens 256 --timeout 900
```

Expected:

- Eager and graph results both pass semantic checks.
- Graph result records lower steady-state per-token wall time than eager, or the plan records the next bottleneck with evidence.

- [ ] **Step 3: Broaden capture sizes only after single-seq success**

Restart with:

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

Expected:

- `max-num-seqs=1` behavior is unchanged.
- Server log captures only useful graph sizes for the configured scheduler; if extra graph memory is wasted, restore `[1]`.

- [x] **Step 4: Document the support boundary**

Append to `docs/models/supported_models.md`:

```markdown
CUDA Graph note for local SM70: DeepSeek V4 Flash on V100 is validated first
with eager prefill and `FULL_DECODE_ONLY` decode graphs. Use
the default `VLLM_SM70_MHC_FAST=1` path after the exact `ZX-42` canary and
identity smoke both pass. Set `VLLM_SM70_MHC_FAST=0` only for fallback A/B or
rollback comparison, because it materially lowers decode throughput.
```

Run:

```bash
git diff --check docs/models/supported_models.md
```

Expected: no whitespace errors.

## Final Acceptance

- [x] `python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json` proves the current worktree imports vLLM from `feature/vllm-0190-upstream-split` and points FlashMLA provenance at `/mnt/data/apps/FlashMLA`.
- [x] `_flashmla_C` imports after rebuilding with `FLASH_MLA_SRC_DIR=/mnt/data/apps/FlashMLA FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 TORCH_CUDA_ARCH_LIST=7.0`.
- [x] `FULL_DECODE_ONLY` server startup succeeds without `--enforce-eager`, and logs show CUDA Graph capture for decode.
- [x] The `ZX-42` exact prompt passes on the graph server.
- [x] The identity prompt returns coherent text with `finish_reason=stop`.
- [x] Decode tokens/s is computed from streaming timestamps and compared against the eager baseline.
- [ ] Any remaining performance gap is classified by evidence: graph disabled, capture/replay miss, FlashMLA kernel time, mHC fallback time, MoE/all-reduce time, or scheduler/padding overhead.
