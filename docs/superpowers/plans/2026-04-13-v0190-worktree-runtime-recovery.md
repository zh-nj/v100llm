# v0.19.0 Worktree Runtime Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split` self-sufficient for build, install, import, and real inference in `conda gptq` without relying on `PYTHONPATH=/mnt/data/apps/1Cat-vLLM`.

**Architecture:** Recover the worktree in four implementation layers that match the approved spec, then finish with one explicit extended-validation task. First restore a reproducible editable-install path and CUDA platform detection, then realign `EngineArgs` and `VllmConfig` runtime contracts to the `v0.19.0` code layout, then port the remaining `SM70 + FLASH_ATTN + compressed-tensors MoE` runtime behavior from `chore/ignore-project-worktrees`, and finally run worktree-local `AsyncLLM` smokes plus an explicit verification log. Keep the patch surface limited to files that affect `build/import/platform/config/SM70 runtime`; do not use the root repository as a fallback import path at any point.

**Tech Stack:** Python, PyTorch, pytest, CMake/CUDA, vLLM v1 engine, Conda (`gptq`), Tesla V100, local models under `/mnt/data6/models`

---

## File Structure

### Files to create

- `docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`
  - Responsibility: record the literal build/import/test/smoke/benchmark evidence for this worktree only.
- `tests/cuda/scripts/check_disable_nvml_cuda_platform.py`
  - Responsibility: pin the regression where `VLLM_DISABLE_NVML=1` must still resolve `current_platform.device_type == "cuda"` without eagerly initializing CUDA.

### Files to modify

- `setup.py`
  - Responsibility: make the worktree editable install build the extensions expected by the current Python runtime, including flash-attention-aware CMake configuration.
- `csrc/ops.h`
  - Responsibility: keep the exported custom-op declarations in sync with the Python bindings compiled into `vllm._C`.
- `vllm/platforms/__init__.py`
  - Responsibility: keep CUDA platform detection correct when NVML is explicitly disabled.
- `vllm/config/vllm.py`
  - Responsibility: define the runtime config groups that `EngineArgs` and `AsyncLLM` expect in this branch.
- `vllm/config/model.py`
  - Responsibility: keep `ModelConfig` aligned with the newer `arg_utils` and multimodal/runtime callers already present in the worktree.
- `vllm/envs.py`
  - Responsibility: provide the environment validation entry point used by `EngineArgs.create_engine_config()`.
- `vllm/engine/arg_utils.py`
  - Responsibility: convert CLI / dict inputs into a self-consistent `VllmConfig` object and pass the full runtime contract into `AsyncLLM`.
- `vllm/utils/platform_utils.py`
  - Responsibility: expose compute-unit helpers without forcing early CUDA initialization.
- `vllm/model_executor/layers/batch_invariant.py`
  - Responsibility: consume the platform helper API instead of reaching into CUDA properties directly.
- `vllm/platforms/cuda.py`
  - Responsibility: keep SM70 backend priority and runtime defaults aligned with the verified root branch behavior.
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
  - Responsibility: restore the missing SM70 compressed-tensors MoE AWQ compatibility path.
- `tests/cuda/test_platform_no_cuda_init.py`
  - Responsibility: cover CUDA import/platform regressions in subprocesses.
- `tests/engine/test_arg_utils.py`
  - Responsibility: pin CLI/config helper coverage for the runtime config groups.
- `tests/v1/engine/test_engine_args.py`
  - Responsibility: pin `EngineArgs -> VllmConfig` round trips that the current runtime needs.

### Existing regression tests to run unchanged

- `tests/v1/executor/test_executor.py`
- `tests/v1/attention/test_flash_attn_sm70.py`
- `tests/quantization/test_minimax_m2_awq_sm70.py`
- `tests/quantization/test_compressed_tensors.py`

### Scope guard

- Never set `PYTHONPATH=/mnt/data/apps/1Cat-vLLM`.
- Never commit `build-force/` or `install/`.
- Do not port unrelated root-branch changes such as Zen CPU detection or broad gRPC/serving refactors unless a failing test in this plan proves they are required for the acceptance criteria.

## Task 1: Restore Editable Install, `_C` Import, and NVML-Free CUDA Detection

**Files:**
- Create: `docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`
- Create: `tests/cuda/scripts/check_disable_nvml_cuda_platform.py`
- Modify: `tests/cuda/test_platform_no_cuda_init.py`
- Modify: `setup.py`
- Modify: `csrc/ops.h`
- Modify: `vllm/platforms/__init__.py`

- [ ] **Step 1: Create the verification log scaffold**

Create `docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md` with this exact content:

```markdown
# 2026-04-13 v0.19.0 Worktree Runtime Recovery Verification

## Environment

- Worktree: `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split`
- Conda env: `gptq`
- Root `PYTHONPATH` disabled: `yes`

## Build And Import

## Targeted Pytest

## Qwen3.5-27B-AWQ AsyncLLM Smoke

## MiniMax-M2.5-AWQ AsyncLLM Smoke

## Qwen3.5-122B-A10B-AWQ-4bit Extended Validation

## Remaining Gaps
```

- [ ] **Step 2: Add the failing NVML-disabled CUDA platform regression**

Create `tests/cuda/scripts/check_disable_nvml_cuda_platform.py` with this exact content:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that disabling NVML still resolves the CUDA platform."""

import os

for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
    os.environ.pop(key, None)

os.environ["VLLM_DISABLE_NVML"] = "1"

import torch  # noqa: E402

assert not torch.cuda.is_initialized(), "CUDA initialized before import"

from vllm.platforms import current_platform  # noqa: E402

assert current_platform.device_type == "cuda", (
    f"Expected CUDA platform, got {current_platform.device_type!r}"
)
assert not torch.cuda.is_initialized(), (
    "CUDA was initialized while resolving current_platform with VLLM_DISABLE_NVML=1"
)
print("OK")
```

Modify `tests/cuda/test_platform_no_cuda_init.py` so it imports `torch` and adds this test:

```python
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA host")
def test_disable_nvml_still_detects_cuda_platform():
    result = run_script("check_disable_nvml_cuda_platform.py")
    if result.returncode != 0:
        pytest.fail(f"disable-nvml CUDA detection failed:\n{result.stderr}")
```

- [ ] **Step 3: Run the new regression and the current import smoke to prove the worktree is still broken**

Run:

```bash
pytest -q tests/cuda/test_platform_no_cuda_init.py -k disable_nvml_still_detects_cuda_platform
```

Expected:

- The new subprocess test fails with either `current_platform.device_type == ""`, `UnspecifiedPlatform`, or an exception from `cuda_platform_plugin`.

Then run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
unset PYTHONPATH
python -m pip install -e . --no-build-isolation
python - <<'PY'
import vllm
print(vllm.__file__)
import vllm._C
print("imported _C")
PY
```

Expected:

- Either the editable build fails, or `import vllm._C` fails, or the imported path is not the worktree path. Record the literal first failure in the verification log.

- [ ] **Step 4: Port the minimal build/platform code required by the failing checks**

Apply these exact code changes.

In `setup.py`, make the CMake configure step flash-attention-aware:

```python
        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DVLLM_TARGET_DEVICE={}".format(VLLM_TARGET_DEVICE),
            "-DVLLM_BUILD_FLASH_ATTN={}".format(
                "ON" if _should_build_fa2() or _should_build_fa3() else "OFF"
            ),
        ]
```

In `vllm/platforms/__init__.py`, hoist `os` to the module imports and add the non-NVML CUDA path at the start of `cuda_platform_plugin()`:

```python
import os


def cuda_platform_plugin() -> str | None:
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    if os.environ.get("VLLM_DISABLE_NVML") == "1":
        try:
            import torch

            is_cuda = torch.cuda.is_available() and not vllm_version_matches_substr(
                "cpu"
            )
            if is_cuda:
                logger.debug("Confirmed CUDA platform is available without NVML.")
            else:
                logger.debug(
                    "CUDA platform is not available in non-NVML detection path."
                )
        except Exception as e:
            logger.debug(
                "CUDA platform is not available in non-NVML detection path: %s",
                str(e),
            )
        return "vllm.platforms.cuda.CudaPlatform" if is_cuda else None
```

In `csrc/ops.h`, replace the drifted declarations with the signatures currently expected by the Python bindings:

```cpp
void large_context_topk(const torch::Tensor& logits, torch::Tensor& indices,
                        const torch::Tensor& seq_lens,
                        std::optional<torch::Tensor> row_starts = std::nullopt);

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets,
    const bool is_gated);

void get_cutlass_batched_moe_mm_data(torch::Tensor& expert_offsets,
                                     torch::Tensor& problem_sizes1,
                                     torch::Tensor& problem_sizes2,
                                     const torch::Tensor& expert_num_tokens,
                                     const int64_t num_local_experts,
                                     const int64_t padded_m,
                                     const int64_t n, const int64_t k);

void scaled_fp4_quant_out(torch::Tensor const& input,
                          torch::Tensor const& input_sf,
                          bool is_sf_swizzled_layout, torch::Tensor& output,
                          torch::Tensor& output_sf);

std::tuple<torch::Tensor, torch::Tensor> scaled_fp4_quant_func(
    torch::Tensor const& input, torch::Tensor const& input_sf,
    bool is_sf_swizzled_layout);

void selective_scan_fwd(
    const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A,
    const torch::Tensor& B, const torch::Tensor& C,
    const std::optional<torch::Tensor>& D_,
    const std::optional<torch::Tensor>& z_,
    const std::optional<torch::Tensor>& delta_bias_, bool delta_softplus,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& ssm_states, int64_t pad_slot_id, int64_t block_size,
    const std::optional<torch::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx,
    const std::optional<torch::Tensor>& cu_chunk_seqlen,
    const std::optional<torch::Tensor>& last_chunk_indices);
```

Do not port unrelated root-branch changes in this task.

- [ ] **Step 5: Re-run the build/import checks and record the literal output**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
unset PYTHONPATH
rm -rf build-force install
python -m pip install -e . --no-build-isolation
python - <<'PY'
import vllm
print("vllm_path", vllm.__file__)
import vllm._C
print("imported _C")
from vllm.platforms import current_platform
print("device_type", current_platform.device_type)
PY
pytest -q tests/cuda/test_platform_no_cuda_init.py -k "disable_nvml_still_detects_cuda_platform or device_count_respects_env_after_platform_import"
```

Expected:

- `vllm_path` points into `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split`
- `imported _C` prints
- `device_type cuda` prints
- Both selected tests pass

- [ ] **Step 6: Commit the build/import recovery**

```bash
git add \
  docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md \
  tests/cuda/scripts/check_disable_nvml_cuda_platform.py \
  tests/cuda/test_platform_no_cuda_init.py \
  setup.py \
  csrc/ops.h \
  vllm/platforms/__init__.py
git commit -m "build: restore v0190 worktree import path"
```

## Task 2: Realign `EngineArgs`, `VllmConfig`, and Runtime Helper Contracts

**Files:**
- Modify: `vllm/config/vllm.py`
- Modify: `vllm/config/model.py`
- Modify: `vllm/envs.py`
- Modify: `vllm/engine/arg_utils.py`
- Modify: `vllm/utils/platform_utils.py`
- Modify: `vllm/model_executor/layers/batch_invariant.py`
- Modify: `tests/engine/test_arg_utils.py`
- Modify: `tests/v1/engine/test_engine_args.py`

- [ ] **Step 1: Add failing coverage for the missing runtime config groups**

In `tests/engine/test_arg_utils.py`, extend the imports and add this test:

```python
from vllm.config import AttentionConfig, CompilationConfig, VllmConfig, config


def test_get_kwargs_includes_runtime_config_groups():
    kwargs = get_kwargs(VllmConfig)

    for key in (
        "offload_config",
        "kernel_config",
        "reasoning_config",
        "weight_transfer_config",
    ):
        assert key in kwargs

    parsed = kwargs["kernel_config"]["type"]('{"moe_backend":"FLASHINFER-CUTLASS"}')
    assert parsed.moe_backend == "flashinfer_cutlass"
```

In `tests/v1/engine/test_engine_args.py`, add this test:

```python
def test_runtime_config_groups_round_trip_from_engine_args():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        kernel_config={"moe_backend": "FLASHINFER-CUTLASS"},
        reasoning_config={
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "</think>",
        },
        performance_mode="throughput",
        weight_transfer_config={"backend": "ipc"},
        shutdown_timeout=7,
        offload_backend="uva",
        cpu_offload_gb=1.5,
    )

    vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    assert vllm_config.kernel_config.moe_backend == "flashinfer_cutlass"
    assert vllm_config.reasoning_config is not None
    assert vllm_config.reasoning_config.reasoning_end_str == "</think>"
    assert vllm_config.offload_config.offload_backend == "uva"
    assert vllm_config.offload_config.uva.cpu_offload_gb == 1.5
    assert vllm_config.weight_transfer_config is not None
    assert vllm_config.weight_transfer_config.backend == "ipc"
    assert vllm_config.performance_mode == "throughput"
    assert vllm_config.shutdown_timeout == 7
```

- [ ] **Step 2: Run the new coverage to capture the first real contract break**

Run:

```bash
pytest -q tests/engine/test_arg_utils.py -k runtime_config_groups
pytest -q tests/v1/engine/test_engine_args.py -k runtime_config_groups_round_trip_from_engine_args
```

Expected:

- At least one test fails with a missing `VllmConfig` field, a missing `EngineArgs` conversion, or a helper/runtime attribute error.

- [ ] **Step 3: Port the runtime contract code, not ad-hoc compatibility shims**

In `vllm/config/vllm.py`, add the missing runtime config fields:

```python
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from .kernel import KernelConfig
from .offload import OffloadConfig
from .reasoning import ReasoningConfig
from .weight_transfer import WeightTransferConfig

PerformanceMode = Literal["balanced", "interactivity", "throughput"]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    ...
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    offload_config: OffloadConfig = Field(default_factory=OffloadConfig)
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    kernel_config: KernelConfig = Field(default_factory=KernelConfig)
    ...
    ec_transfer_config: ECTransferConfig | None = None
    reasoning_config: ReasoningConfig | None = None
    ...
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    performance_mode: PerformanceMode = "balanced"
    weight_transfer_config: WeightTransferConfig | None = None
    shutdown_timeout: int = Field(default=0, ge=0)
```

In `vllm/config/model.py`, keep `ModelConfig` compatible with the newer callers already present in `arg_utils.py`:

```python
from vllm.config.multimodal import (
    MMCacheType,
    MMEncoderTPMode,
    MMTensorIPC,
    MultiModalConfig,
)

class ModelConfig:
    ...
    io_processor_plugin: str | None = None
    renderer_num_workers: int = 1
    ...
    multimodal_config: MultiModalConfig | None = None
    language_model_only: InitVar[bool] = False
    ...
    video_pruning_rate: InitVar[float | None] = None
    mm_tensor_ipc: InitVar[MMTensorIPC | None] = None
```

In `vllm/envs.py`, add the validation entry point used by `EngineArgs.create_engine_config()`:

```python
def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in environment_variables:
            if hard_fail:
                raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            logger.warning("Unknown vLLM environment variable detected: %s", env)
```

In `vllm/utils/platform_utils.py`, expose compute-unit lookup through the platform layer:

```python
def get_cu_count(device_id: int = 0) -> int:
    """Backwards-compatible alias for the device's compute unit count."""
    return num_compute_units(device_id)


@cache
def num_compute_units(device_id: int = 0) -> int:
    """Get the number of compute units of the current device."""
    from vllm.platforms import current_platform

    return current_platform.num_compute_units(device_id)
```

In `vllm/model_executor/layers/batch_invariant.py`, stop reading CUDA properties directly and read the env flag through `vllm.envs`:

```python
import vllm.envs as envs
from vllm.utils.platform_utils import num_compute_units


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    ...
    NUM_SMS = num_compute_units(a.device.index)
    ...


def vllm_is_batch_invariant() -> bool:
    return getattr(envs, "VLLM_BATCH_INVARIANT", False)
```

In `vllm/engine/arg_utils.py`, keep `EngineArgs` and `create_engine_config()` aligned with those config groups:

```python
from vllm.config import (
    ...
    KernelConfig,
    OffloadConfig,
    PrefetchOffloadConfig,
    ReasoningConfig,
    UVAOffloadConfig,
    WeightTransferConfig,
)
from vllm.config.vllm import OptimizationLevel, PerformanceMode


class EngineArgs:
    ...
    compilation_config: CompilationConfig = get_field(VllmConfig, "compilation_config")
    attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")
    kernel_config: KernelConfig = get_field(VllmConfig, "kernel_config")
    ...
    reasoning_config: ReasoningConfig = get_field(VllmConfig, "reasoning_config")
    ...
    performance_mode: PerformanceMode = VllmConfig.performance_mode
    shutdown_timeout: int = 0
    weight_transfer_config: WeightTransferConfig | None = get_field(
        VllmConfig,
        "weight_transfer_config",
    )

    def __post_init__(self):
        ...
        if isinstance(self.kernel_config, dict):
            self.kernel_config = KernelConfig(**self.kernel_config)
        ...
        if isinstance(self.weight_transfer_config, dict):
            self.weight_transfer_config = WeightTransferConfig(
                **self.weight_transfer_config
            )

    def create_engine_config(...):
        ...
        kernel_config = copy.deepcopy(self.kernel_config)
        ...
        offload_config = OffloadConfig(
            offload_backend=self.offload_backend,
            uva=UVAOffloadConfig(
                cpu_offload_gb=self.cpu_offload_gb,
                cpu_offload_params=self.cpu_offload_params,
            ),
            prefetch=PrefetchOffloadConfig(
                offload_group_size=self.offload_group_size,
                offload_num_in_group=self.offload_num_in_group,
                offload_prefetch_step=self.offload_prefetch_step,
                offload_params=self.offload_params,
            ),
        )
        ...
        config = VllmConfig(
            ...
            offload_config=offload_config,
            attention_config=attention_config,
            kernel_config=kernel_config,
            ...
            reasoning_config=self.reasoning_config,
            ...
            performance_mode=self.performance_mode,
            weight_transfer_config=self.weight_transfer_config,
            shutdown_timeout=self.shutdown_timeout,
        )
```

- [ ] **Step 4: Re-run the runtime contract tests plus one executor smoke**

Run:

```bash
pytest -q tests/engine/test_arg_utils.py -k "runtime_config_groups or unrecognized_env"
pytest -q tests/v1/engine/test_engine_args.py -k "runtime_config_groups_round_trip_from_engine_args or defaults_with_usage_context"
pytest -q tests/v1/executor/test_executor.py -k custom_executor_async
```

Expected:

- All selected tests pass
- No `AttributeError`, `TypeError`, or `Unknown vLLM environment variable` regression beyond the existing intentional validation test

- [ ] **Step 5: Commit the runtime contract alignment**

```bash
git add \
  vllm/config/vllm.py \
  vllm/config/model.py \
  vllm/envs.py \
  vllm/engine/arg_utils.py \
  vllm/utils/platform_utils.py \
  vllm/model_executor/layers/batch_invariant.py \
  tests/engine/test_arg_utils.py \
  tests/v1/engine/test_engine_args.py
git commit -m "fix: realign v0190 runtime config contracts"
```

## Task 3: Recover SM70 Attention and Compressed-Tensors MoE Runtime Behavior

**Files:**
- Modify: `vllm/platforms/cuda.py`
- Modify: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- Test: `tests/v1/attention/test_flash_attn_sm70.py`
- Test: `tests/quantization/test_minimax_m2_awq_sm70.py`
- Test: `tests/quantization/test_compressed_tensors.py`
- Reference: `vllm/model_executor/layers/quantization/awq.py`

- [ ] **Step 1: Run the existing SM70 regressions before touching runtime code**

Run:

```bash
pytest -q tests/v1/attention/test_flash_attn_sm70.py
pytest -q tests/quantization/test_minimax_m2_awq_sm70.py
pytest -q tests/quantization/test_compressed_tensors.py -k test_compressed_tensors_moe_ignore_with_model
```

Expected:

- If the worktree is still missing the root-branch runtime recovery, at least one test fails.
- If the first two suites already pass, treat `test_compressed_tensors_moe_ignore_with_model` as the critical missing regression because `3e0759231` is the known root-only delta.

- [ ] **Step 2: Port the verified SM70 backend selection and compressed-tensors MoE branch**

In `vllm/platforms/cuda.py`, keep SM70 backend ordering aligned with the verified root branch:

```python
        elif device_capability.major == 7:
            # SM70 (V100): Prefer standard FlashAttention when available,
            # keep the V100-specific backend as a fallback.
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASH_ATTN_V100,
                AttentionBackendEnum.TRITON_ATTN,
            ]
```

In `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, restore the missing SM70 AWQ-compatibility path before the Marlin fallback:

```python
            # SM70 (V100): use TurboMind GEMM kernels,
            # Marlin requires SM75+.
            if _is_sm70_available() and weight_quant.num_bits == 4:
                gs = weight_quant.group_size
                moe_cfg = layer.moe_config
                hidden = moe_cfg.hidden_dim
                inter = moe_cfg.intermediate_size_per_partition
                sm70_ok = (
                    gs in (32, 64, 128)
                    and hidden % gs == 0
                    and inter % gs == 0
                    and hidden % 8 == 0
                    and inter % 8 == 0
                    and weight_quant.symmetric
                )
                if sm70_ok:
                    logger.info_once(
                        "Using CompressedTensorsSM70WNA16MoEMethod "
                        "(TurboMind SM70 kernels)"
                    )
                    return CompressedTensorsSM70WNA16MoEMethod(
                        weight_quant, input_quant, layer.moe_config
                    )
                else:
                    logger.warning_once(
                        "SM70 detected but compressed-tensors MoE "
                        "dimensions incompatible with TurboMind "
                        f"(hidden={hidden}, inter={inter}, "
                        f"group_size={gs}, "
                        f"symmetric={weight_quant.symmetric}). "
                        "Falling back to WNA16MoE."
                    )
```

Do not widen `awq.py` in this task unless `tests/quantization/test_minimax_m2_awq_sm70.py` regresses after the compressed-tensors change.

- [ ] **Step 3: Re-run the regressions and one direct backend-priority smoke**

Run:

```bash
pytest -q tests/v1/attention/test_flash_attn_sm70.py
pytest -q tests/quantization/test_minimax_m2_awq_sm70.py
pytest -q tests/quantization/test_compressed_tensors.py -k test_compressed_tensors_moe_ignore_with_model
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm.platforms.cuda import CudaPlatform, _get_backend_priorities

cap = CudaPlatform.get_device_capability()
print("capability", (cap.major, cap.minor))
print("priorities", [x.name for x in _get_backend_priorities(False, cap)])
PY
```

Expected:

- The three pytest invocations pass, or the compressed-tensors integration test skips cleanly because its local model fixture is unavailable
- On V100, the direct smoke prints `capability (7, 0)`
- On V100, the direct smoke prints `priorities ['FLASH_ATTN', 'FLASH_ATTN_V100', 'TRITON_ATTN']`

- [ ] **Step 4: Commit the SM70 runtime recovery**

```bash
git add \
  vllm/platforms/cuda.py \
  vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
git commit -m "fix: recover sm70 moe runtime on v0190 worktree"
```

## Task 4: Prove the Worktree Can Run Real `AsyncLLM` Inference Without Root `PYTHONPATH`

**Files:**
- Modify: `docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`

- [ ] **Step 1: Run the mandatory Qwen3.5-27B-AWQ smoke through `AsyncLLM`**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
unset PYTHONPATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python - <<'PY'
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


async def main():
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="/mnt/data6/models/Qwen3.5-27B-AWQ",
            quantization="awq",
            dtype="half",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            attention_backend="FLASH_ATTN",
            disable_log_stats=True,
        )
    )
    final = None
    try:
        async for out in engine.generate(
            request_id="qwen27-smoke",
            prompt="只回答最终结果：2+2等于几？",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        ):
            final = out
    finally:
        engine.shutdown()

    assert final is not None
    text = final.outputs[0].text.strip()
    finish_reason = final.outputs[0].finish_reason
    print("text:", text)
    print("finish_reason:", finish_reason)
    assert text
    assert finish_reason in ("stop", "length")


asyncio.run(main())
PY
```

Expected:

- The worktree starts its own engine
- The output text is non-empty
- `finish_reason` is `stop` or `length`

- [ ] **Step 2: Run the mandatory MiniMax-M2.5-AWQ smoke through `AsyncLLM`**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
unset PYTHONPATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python - <<'PY'
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


async def main():
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="/mnt/data6/models/MiniMax-M2.5-AWQ",
            quantization="awq",
            dtype="half",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
        )
    )
    final = None
    try:
        async for out in engine.generate(
            request_id="minimax-smoke",
            prompt="只回答最终结果：北京是不是中国的首都？",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        ):
            final = out
    finally:
        engine.shutdown()

    assert final is not None
    text = final.outputs[0].text.strip()
    finish_reason = final.outputs[0].finish_reason
    print("text:", text)
    print("finish_reason:", finish_reason)
    assert text
    assert finish_reason in ("stop", "length")


asyncio.run(main())
PY
```

Expected:

- The engine starts without importing the root source tree
- The output text is non-empty
- `finish_reason` is `stop` or `length`

- [ ] **Step 3: Run the focused pytest suite after the real-model smokes**

Run:

```bash
pytest -q tests/cuda/test_platform_no_cuda_init.py
pytest -q tests/engine/test_arg_utils.py -k "runtime_config_groups or unrecognized_env"
pytest -q tests/v1/engine/test_engine_args.py -k "runtime_config_groups_round_trip_from_engine_args or defaults_with_usage_context"
pytest -q tests/v1/executor/test_executor.py -k custom_executor_async
pytest -q tests/v1/attention/test_flash_attn_sm70.py
pytest -q tests/quantization/test_minimax_m2_awq_sm70.py
pytest -q tests/quantization/test_compressed_tensors.py -k test_compressed_tensors_moe_ignore_with_model
```

Expected:

- All selected tests pass, with at most a clean skip on the model-backed compressed-tensors integration case

- [ ] **Step 4: Append the literal command/results to the verification log**

Append the exact commands you ran and their literal summaries under:

```markdown
## Build And Import

## Targeted Pytest

## Qwen3.5-27B-AWQ AsyncLLM Smoke

## MiniMax-M2.5-AWQ AsyncLLM Smoke
```

Record:

- the imported `vllm.__file__` path
- whether `import vllm._C` succeeded
- the final pytest pass/skip counts
- the generated text and `finish_reason` for both real-model smokes

- [ ] **Step 5: Commit the mandatory verification record**

```bash
git add docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md
git commit -m "docs: record mandatory v0190 worktree smokes"
```

## Task 5: Attempt Extended Qwen122 Validation and 1k/32k Serve Benchmarks

**Files:**
- Modify: `docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`

- [ ] **Step 1: Start a worktree-local Qwen27 server for serving benchmarks**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
unset PYTHONPATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data6/models/Qwen3.5-27B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend FLASH_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 127.0.0.1 \
  --port 8000 \
  > /tmp/v0190-qwen27-server.log 2>&1 &
echo $! > /tmp/v0190-qwen27-server.pid
sleep 30
curl -sf http://127.0.0.1:8000/health
```

Expected:

- `curl` returns success
- `/tmp/v0190-qwen27-server.log` shows the worktree server started without import fallbacks

- [ ] **Step 2: Run 1k and 32k serving benchmarks against that server**

Run:

```bash
mkdir -p /tmp/v0190-qwen27-1k /tmp/v0190-qwen27-32k
vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/completions \
  --model /mnt/data6/models/Qwen3.5-27B-AWQ \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 8 \
  --max-concurrency 1 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/v0190-qwen27-1k
vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/completions \
  --model /mnt/data6/models/Qwen3.5-27B-AWQ \
  --dataset-name random \
  --random-input-len 31744 \
  --random-output-len 128 \
  --num-prompts 2 \
  --max-concurrency 1 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/v0190-qwen27-32k
```

Expected:

- Both runs print `Serving Benchmark Result`
- Both runs report `Mean TTFT (ms)` and token throughput
- Save paths under `/tmp/v0190-qwen27-1k` and `/tmp/v0190-qwen27-32k` are populated

- [ ] **Step 3: Stop the Qwen27 benchmark server cleanly**

Run:

```bash
kill "$(cat /tmp/v0190-qwen27-server.pid)"
```

Expected:

- The server exits and port `8000` is freed

- [ ] **Step 4: Attempt the extended Qwen122 smoke once the mandatory path is already green**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
unset PYTHONPATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python - <<'PY'
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


async def main():
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="/mnt/data6/models/Qwen3.5-122B-A10B-AWQ-4bit",
            quantization="awq",
            dtype="half",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            attention_backend="FLASH_ATTN",
            disable_log_stats=True,
        )
    )
    final = None
    try:
        async for out in engine.generate(
            request_id="qwen122-smoke",
            prompt="只回答最终结果：128+256等于几？",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        ):
            final = out
    finally:
        engine.shutdown()

    assert final is not None
    text = final.outputs[0].text.strip()
    finish_reason = final.outputs[0].finish_reason
    print("text:", text)
    print("finish_reason:", finish_reason)
    assert text
    assert finish_reason in ("stop", "length")


asyncio.run(main())
PY
```

Expected:

- Best case: non-empty output plus `finish_reason stop` or `length`
- If startup fails or time budget is exceeded, stop after the first real failure and record that exact blocker in the verification log instead of iterating blindly

- [ ] **Step 5: Append benchmark/Qwen122 outcomes and commit the final verification doc**

Append under:

```markdown
## Qwen3.5-122B-A10B-AWQ-4bit Extended Validation

## Remaining Gaps
```

Record:

- the 1k and 32k `Mean TTFT (ms)` values from the benchmark summaries
- the 1k and 32k output token throughput values from the benchmark summaries
- whether Qwen122 smoke passed, failed, or was intentionally stopped after the first blocker
- whether the current branch still lacks any prefill/decode chart parity with the root branch

Then commit:

```bash
git add docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md
git commit -m "docs: record extended v0190 worktree verification"
```

## Self-Review

### Spec coverage

- Build/install/import without root `PYTHONPATH`: covered by Task 1 and Task 4
- CUDA platform detection and `_C` import: covered by Task 1
- `AsyncEngineArgs.create_engine_config()` / `AsyncLLM.from_engine_args()` runtime alignment: covered by Task 2 and the executor smoke
- SM70 `FLASH_ATTN` and compressed-tensors MoE runtime recovery: covered by Task 3
- Mandatory real-model validation on `Qwen3.5-27B-AWQ` and `MiniMax-M2.5-AWQ`: covered by Task 4
- Best-effort extended validation on `Qwen3.5-122B-A10B-AWQ-4bit` plus 1k/32k serving benchmarks: covered by Task 5
- Explicit gap recording instead of hand-waving: covered by the verification log updates in Tasks 4 and 5

### Placeholder scan

- No `TODO`
- No `TBD`
- No unresolved file paths
- No “fix later” language
- Every code-changing step includes concrete code
- Every verification step includes concrete commands

### Type consistency

- Runtime config names are used consistently as:
  - `offload_config`
  - `kernel_config`
  - `reasoning_config`
  - `weight_transfer_config`
  - `performance_mode`
  - `shutdown_timeout`
- Mandatory model paths are used consistently as:
  - `/mnt/data6/models/Qwen3.5-27B-AWQ`
  - `/mnt/data6/models/MiniMax-M2.5-AWQ`
  - `/mnt/data6/models/Qwen3.5-122B-A10B-AWQ-4bit`
