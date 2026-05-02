# DeepSeek V4 FlashMLA SM70 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `DeepseekV4ForCausalLM` select and run the local FlashMLA SM70 sparse decode/prefill backend on V100-class GPUs, then validate `/mnt/data6/models/DeepSeek-V4-Flash` with a minimal end-to-end smoke.

**Architecture:** Keep DeepSeek V4's existing model, KV-cache, sparse metadata, and MoE paths intact. The work is a narrow integration layer: allow SM70 sparse FlashMLA gates only after the local FlashMLA SM70 source is compiled into vLLM, split sparse availability from dense FlashMLA extension availability, then verify the exact MODEL1 `fp8_ds_mla` decode/prefill APIs before running the full model.

**Tech Stack:** Python 3.13, Conda `gptq`, CUDA 12.8, CMake, PyTorch custom ops, vLLM V1, local FlashMLA at `/mnt/data/apps/FlashMLA`, local model `/mnt/data6/models/DeepSeek-V4-Flash`, SM70/V100 GPUs via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9`.

---

## Current Evidence

- Target worktree: `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split`
- Target branch: `feature/vllm-0190-upstream-split`
- Current HEAD when this plan was written: `dbdf0d8a6` (`移植 DeepSeek V4 Flash 推理支持`)
- Worktree status when this plan was written: clean
- Local FlashMLA branch: `feature/sm70-volta-flashmla`
- Local FlashMLA HEAD when inspected: `a507081` (`提升 SM70 sparse decode topk 上限`)
- `DeepseekV4ForCausalLM` is already registered in `vllm/model_executor/models/registry.py`.
- DeepSeek V4 attention already returns `DeepseekV4FlashMLASparseBackend`.
- DeepSeek V4 decode already calls `flash_mla_with_kvcache(...)`.
- DeepSeek V4 prefill already calls `flash_mla_sparse_fwd(...)`.
- DeepSeek V4 KV cache already normalizes to `fp8_ds_mla`.
- DeepSeek V4 MODEL1 cache shape is already `584B` per token.
- Static inspect from `/mnt/data/apps/FlashMLA/benchmark/bench_vllm_deepseek_v4_flash_sm70.py --mode inspect --vllm-root /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split --json` reports `end_to_end_smoke_ready=false` because vLLM still blocks SM70 in the FlashMLA sparse backend gate and runtime probe.

## File Map

- Modify: `vllm/v1/attention/backends/mla/flashmla_sparse.py`
  - Allow SM70 for sparse FlashMLA backend selection.
  - Keep supported head sizes, block sizes, and `fp8_ds_mla` cache-shape constraints as the real layout guards.
- Modify: `vllm/v1/attention/ops/flashmla.py`
  - Split sparse FlashMLA availability from dense-extension availability.
  - Allow sparse FlashMLA on SM70 when `_flashmla_C` is importable.
  - Keep dense FlashMLA SM90-only unless its extension is present.
- Modify: `setup.py`
  - Build vLLM FlashMLA extensions on CUDA 12.8 when `FLASH_MLA_ENABLE_SM70=1` is set.
  - Preserve the CUDA 12.9+ default for upstream SM90/SM100 FlashMLA builds.
- Modify: `cmake/external_projects/flashmla.cmake`
  - Add opt-in SM70 support to `SUPPORT_ARCHS`.
  - Include local FlashMLA SM70 sparse decode and sparse prefill sources when `7.0` is in `FLASH_MLA_ARCHS`.
  - Skip the dense FP8 extension target for an SM70-only build, because DeepSeek V4 sparse inference does not need `_flashmla_extension_C`.
- Create: `tests/v1/attention/test_flashmla_sm70_sparse_support.py`
  - Unit-test Python backend/runtime gates without requiring a GPU.
- Create: `tests/build/test_flashmla_sm70_build_contract.py`
  - Static build-contract tests for the CUDA 12.8 opt-in and CMake SM70 source list.
- Create: `benchmarks/deepseek_v4_flashmla_sm70.py`
  - Repo-local inspect and OpenAI-stream smoke helper for this exact integration path.
- Modify: `docs/models/supported_models.md`
  - Add a short local note that SM70 DeepSeek V4 Flash requires the FlashMLA SM70 opt-in build and remains V100-targeted.

## Build Contract

Use this build environment for all tasks that compile FlashMLA inside vLLM:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
export FLASH_MLA_SRC_DIR=/mnt/data/apps/FlashMLA
export FLASH_MLA_ENABLE_SM70=1
export FLASH_MLA_DISABLE_SM100=1
export TORCH_CUDA_ARCH_LIST="7.0"
```

The build must not rely on root-repo `PYTHONPATH`. Use the worktree path as the package source and verify `import vllm; print(vllm.__file__)` after installation.

## Task 1: Add Red Tests For SM70 Sparse FlashMLA Gates

**Files:**
- Create: `tests/v1/attention/test_flashmla_sm70_sparse_support.py`
- Modify: `vllm/v1/attention/backends/mla/flashmla_sparse.py`
- Modify: `vllm/v1/attention/ops/flashmla.py`
- Test: `tests/v1/attention/test_flashmla_sm70_sparse_support.py`

- [x] **Step 1: Write the failing backend/runtime gate tests**

Create `tests/v1/attention/test_flashmla_sm70_sparse_support.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
)
from vllm.v1.attention.ops import flashmla


def test_flashmla_sparse_backend_supports_sm70() -> None:
    assert FlashMLASparseBackend.supports_compute_capability(
        DeviceCapability(7, 0)
    )
    assert DeepseekV4FlashMLASparseBackend.supports_compute_capability(
        DeviceCapability(7, 0)
    )


def test_flashmla_sparse_runtime_probe_accepts_sm70_core_extension(
    monkeypatch,
) -> None:
    monkeypatch.setattr(flashmla, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla, "_flashmla_extension_C_AVAILABLE", False)
    monkeypatch.setattr(
        flashmla.current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 70,
    )

    assert flashmla.is_flashmla_sparse_supported() == (True, None)


def test_flashmla_dense_runtime_probe_stays_rejected_on_sm70(
    monkeypatch,
) -> None:
    monkeypatch.setattr(flashmla, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla, "_flashmla_extension_C_AVAILABLE", True)
    monkeypatch.setattr(
        flashmla.current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 70,
    )

    ok, reason = flashmla.is_flashmla_dense_supported()

    assert not ok
    assert reason == "FlashMLA Dense is only supported on Hopper devices."
```

- [x] **Step 2: Run the tests and verify the expected red state**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
pytest tests/v1/attention/test_flashmla_sm70_sparse_support.py -q
```

Expected:

- `test_flashmla_sparse_backend_supports_sm70` fails because `FlashMLASparseBackend.supports_compute_capability(DeviceCapability(7, 0))` is currently false.
- `test_flashmla_sparse_runtime_probe_accepts_sm70_core_extension` fails because `is_flashmla_sparse_supported()` still rejects SM70 and still depends on `_flashmla_extension_C_AVAILABLE`.
- The dense rejection test should pass or continue failing only if the helper split has not been implemented yet.

- [x] **Step 3: Implement the sparse backend SM70 gate**

In `vllm/v1/attention/backends/mla/flashmla_sparse.py`, replace the current capability check:

```python
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]
```

with:

```python
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [7, 9, 10]
```

- [x] **Step 4: Split core and extension availability in `ops/flashmla.py`**

In `vllm/v1/attention/ops/flashmla.py`, replace `_is_flashmla_available()` with these helpers:

```python
def _is_flashmla_core_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    return True, None


def _is_flashmla_extension_available() -> tuple[bool, str | None]:
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )
    return True, None


def _is_flashmla_available() -> tuple[bool, str | None]:
    is_available, maybe_reason = _is_flashmla_core_available()
    if not is_available:
        return False, maybe_reason
    return _is_flashmla_extension_available()
```

Then change `is_flashmla_dense_supported()` to require both core and extension:

```python
def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None
```

Change `is_flashmla_sparse_supported()` to require only the core sparse extension and to accept SM70:

```python
def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_core_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(70)
        or current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Volta, Hopper and Blackwell "
            "devices.",
        )
    return True, None
```

Finally change the import gate from:

```python
if _is_flashmla_available()[0]:
```

to:

```python
if _is_flashmla_core_available()[0]:
```

- [x] **Step 5: Run the focused gate tests**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
pytest tests/v1/attention/test_flashmla_sm70_sparse_support.py -q
```

Expected: `3 passed`.

- [x] **Step 6: Commit Task 1**

Run:

```bash
git add tests/v1/attention/test_flashmla_sm70_sparse_support.py \
  vllm/v1/attention/backends/mla/flashmla_sparse.py \
  vllm/v1/attention/ops/flashmla.py
git diff --cached --check
git commit -m "支持 SM70 选择 FlashMLA sparse 后端"
```

## Task 2: Add Build Contract Tests For Local FlashMLA SM70

**Files:**
- Create: `tests/build/test_flashmla_sm70_build_contract.py`
- Modify: `setup.py`
- Modify: `cmake/external_projects/flashmla.cmake`
- Test: `tests/build/test_flashmla_sm70_build_contract.py`

- [x] **Step 1: Write static build-contract red tests**

Create `tests/build/test_flashmla_sm70_build_contract.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_setup_allows_flashmla_sm70_on_cuda_128_when_opted_in() -> None:
    text = _read("setup.py")

    assert "FLASH_MLA_ENABLE_SM70" in text
    assert 'Version("12.8")' in text
    assert "_flashmla_sm70_enabled()" in text
    assert "get_nvcc_cuda_version() >= Version(\"12.9\")" in text


def test_flashmla_cmake_declares_sm70_opt_in_arch_and_sources() -> None:
    text = _read("cmake/external_projects/flashmla.cmake")

    assert "FLASH_MLA_ENABLE_SM70" in text
    assert 'list(APPEND SUPPORT_ARCHS "7.0")' in text
    assert "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu" in text
    assert "csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu" in text
    assert "csrc/sm70/prefill/sparse/instantiations/bf16.cu" in text


def test_flashmla_cmake_skips_dense_extension_for_sm70_only_build() -> None:
    text = _read("cmake/external_projects/flashmla.cmake")

    assert "FLASHMLA_BUILD_DENSE_EXTENSION" in text
    assert "add_custom_target(_flashmla_extension_C)" in text


def test_flashmla_cmake_mirrors_sm70_feature_and_tuning_defines() -> None:
    text = _read("cmake/external_projects/flashmla.cmake")

    assert "KERUTILS_ALLOW_SM70_STUB_COMPILE" in text
    assert "FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE" in text
    assert "FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE" in text
```

- [x] **Step 2: Run the tests and verify the expected red state**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
pytest tests/build/test_flashmla_sm70_build_contract.py -q
```

Expected: the tests fail because `setup.py` still requires CUDA 12.9 for FlashMLA, the CMake file still only lists SM90/SM100 sources, and the SM70 feature/tuning defines are not mirrored from the local FlashMLA build.

- [x] **Step 3: Add the setup.py SM70 opt-in helper**

In `setup.py`, add this helper near `get_nvcc_cuda_version()`:

```python
def _flashmla_sm70_enabled() -> bool:
    return os.getenv("FLASH_MLA_ENABLE_SM70", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
```

Then replace the FlashMLA extension condition:

```python
    if envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.9")
    ):
```

with:

```python
    nvcc_cuda_version = get_nvcc_cuda_version() if CUDA_HOME else Version("0")
    build_flashmla = envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME
        and (
            nvcc_cuda_version >= Version("12.9")
            or (_flashmla_sm70_enabled() and nvcc_cuda_version >= Version("12.8"))
        )
    )
    if build_flashmla:
```

- [x] **Step 4: Add SM70 source selection in CMake**

In `cmake/external_projects/flashmla.cmake`, add this after the CUDA 12.8/12.9 `SUPPORT_ARCHS` logic and before `cuda_archs_loose_intersection(...)`:

```cmake
if(DEFINED ENV{FLASH_MLA_ENABLE_SM70} AND "$ENV{FLASH_MLA_ENABLE_SM70}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
    list(APPEND SUPPORT_ARCHS "7.0")
endif()
```

After `message(STATUS "FlashMLA is available at ${flashmla_SOURCE_DIR}")`, add:

```cmake
set(FLASHMLA_HAS_SM70 FALSE)
set(FLASHMLA_HAS_DENSE_EXTENSION_ARCH FALSE)
if("7.0" IN_LIST FLASH_MLA_ARCHS)
    set(FLASHMLA_HAS_SM70 TRUE)
endif()
foreach(_FLASHMLA_ARCH ${FLASH_MLA_ARCHS})
    if("${_FLASHMLA_ARCH}" MATCHES "^(9\\.0a|10\\.0a|10\\.0f)$")
        set(FLASHMLA_HAS_DENSE_EXTENSION_ARCH TRUE)
    endif()
endforeach()
```

Before `set(FlashMLA_SOURCES`, add:

```cmake
    set(FlashMLA_SM70_SOURCES)
    if(FLASHMLA_HAS_SM70)
        list(APPEND FlashMLA_SM70_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm70/decode/dense/instantiations/fp16.cu
            ${flashmla_SOURCE_DIR}/csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu
            ${flashmla_SOURCE_DIR}/csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu
            ${flashmla_SOURCE_DIR}/csrc/sm70/prefill/sparse/instantiations/bf16.cu)
    endif()
```

Add `${FlashMLA_SM70_SOURCES}` to `FlashMLA_SOURCES` immediately after the common SMXX decode sources:

```cmake
        ${FlashMLA_SM70_SOURCES}
```

Replace the unconditional `_flashmla_extension_C` `define_extension_target(...)` block with this guarded block:

```cmake
    set(FLASHMLA_BUILD_DENSE_EXTENSION ${FLASHMLA_HAS_DENSE_EXTENSION_ARCH})
    if(FLASHMLA_BUILD_DENSE_EXTENSION)
        define_extension_target(
            _flashmla_extension_C
            DESTINATION vllm
            LANGUAGE ${VLLM_GPU_LANG}
            SOURCES ${FlashMLA_Extension_SOURCES}
            COMPILE_FLAGS ${VLLM_FLASHMLA_GPU_FLAGS}
            ARCHITECTURES ${VLLM_GPU_ARCHES}
            INCLUDE_DIRECTORIES ${FlashMLA_Extension_INCLUDES}
            USE_SABI 3
            WITH_SOABI)

        target_compile_options(_flashmla_extension_C PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
            $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)
    else()
        message(STATUS "FlashMLA dense extension skipped for CUDA architectures: ${FLASH_MLA_ARCHS}")
        add_custom_target(_flashmla_extension_C)
    endif()
```

- [x] **Step 5: Run the build-contract tests**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
pytest tests/build/test_flashmla_sm70_build_contract.py -q
```

Expected: `4 passed`.

- [x] **Step 6: Commit Task 2**

Run:

```bash
git add tests/build/test_flashmla_sm70_build_contract.py setup.py cmake/external_projects/flashmla.cmake
git diff --cached --check
git commit -m "接入本地 FlashMLA SM70 sparse 构建"
```

## Task 3: Build And Import vLLM With Local FlashMLA SM70

**Files:**
- Modify only if generated by CMake: `vllm/third_party/flashmla/flash_mla_interface.py`
- Verify: `vllm/_flashmla_C*.so`
- Test: focused import commands below

- [x] **Step 1: Clean only stale FlashMLA build outputs from this worktree**

Run:

```bash
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
find . -maxdepth 3 -name '_flashmla*.so' -print
```

Expected: list any previous `_flashmla` shared objects. Remove only stale files in this worktree if they exist:

```bash
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
find ./vllm -maxdepth 1 -name '_flashmla*.so' -delete
```

- [x] **Step 2: Build and install the worktree with local FlashMLA**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
export FLASH_MLA_SRC_DIR=/mnt/data/apps/FlashMLA
export FLASH_MLA_ENABLE_SM70=1
export FLASH_MLA_DISABLE_SM100=1
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=16
python -m pip install -e . --no-build-isolation -v
```

Expected:

- CMake prints `FlashMLA is available at /mnt/data/apps/FlashMLA`.
- CMake prints `FlashMLA CUDA architectures: 7.0`.
- `_flashmla_C` is compiled.
- `_flashmla_extension_C` may be skipped for an SM70-only build.
- Installation completes without using the root checkout as `PYTHONPATH`.

- [x] **Step 3: Verify imports come from this worktree**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python - <<'PY'
import vllm
print(vllm.__file__)
import vllm._flashmla_C
print("flashmla_core_imported")
from vllm.v1.attention.ops import flashmla
print("sparse_supported", flashmla.is_flashmla_sparse_supported())
print("dense_supported", flashmla.is_flashmla_dense_supported())
PY
```

Expected:

- `vllm.__file__` starts with `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/`.
- `flashmla_core_imported` is printed.
- On a visible SM70 device, `sparse_supported (True, None)` is printed.
- Dense may report unsupported if `_flashmla_extension_C` is not built or the visible GPU is not Hopper.

- [x] **Step 4: Run the focused Python tests after installation**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
pytest tests/v1/attention/test_flashmla_sm70_sparse_support.py \
  tests/build/test_flashmla_sm70_build_contract.py -q
```

Expected: all focused tests pass.

- [x] **Step 5: Commit generated vendored interface only if CMake changed it**

Run:

```bash
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
git status --short vllm/third_party/flashmla/flash_mla_interface.py
```

If the file changed because CMake vendored the local FlashMLA interface, run:

```bash
git add vllm/third_party/flashmla/flash_mla_interface.py
git diff --cached --check
git commit -m "同步本地 FlashMLA Python 接口"
```

If the file did not change, record no commit for this step.

## Task 4: Add A vLLM-Local DeepSeek V4 FlashMLA SM70 Probe

**Files:**
- Create: `benchmarks/deepseek_v4_flashmla_sm70.py`
- Test: `python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json`

- [x] **Step 1: Create the inspect and OpenAI-stream helper**

Create `benchmarks/deepseek_v4_flashmla_sm70.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/mnt/data6/models/DeepSeek-V4-Flash")


CHECKS = [
    ("registry", "vllm/model_executor/models/registry.py", "DeepseekV4ForCausalLM"),
    (
        "backend",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "DeepseekV4FlashMLASparseBackend",
    ),
    (
        "decode_api",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "flash_mla_with_kvcache",
    ),
    (
        "prefill_api",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "flash_mla_sparse_fwd",
    ),
    (
        "cache_dtype",
        "vllm/model_executor/layers/deepseek_v4_attention.py",
        "fp8_ds_mla",
    ),
    (
        "sm70_backend_gate",
        "vllm/v1/attention/backends/mla/flashmla_sparse.py",
        "major in [7, 9, 10]",
    ),
    (
        "sm70_runtime_gate",
        "vllm/v1/attention/ops/flashmla.py",
        "is_device_capability_family(70)",
    ),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def inspect_static() -> dict[str, Any]:
    checks = []
    for name, relpath, needle in CHECKS:
        path = ROOT / relpath
        checks.append(
            {
                "name": name,
                "path": str(path),
                "needle": needle,
                "ok": path.is_file() and needle in _read(path),
            }
        )

    model_cfg_path = MODEL_PATH / "config.json"
    model_cfg = json.loads(_read(model_cfg_path)) if model_cfg_path.is_file() else {}
    model_checks = {
        "model_path": str(MODEL_PATH),
        "config_exists": model_cfg_path.is_file(),
        "architecture": model_cfg.get("architectures", [None])[0],
        "model_type": model_cfg.get("model_type"),
        "head_dim": model_cfg.get("head_dim"),
        "qk_rope_head_dim": model_cfg.get("qk_rope_head_dim"),
        "index_topk": model_cfg.get("index_topk"),
        "compress_ratios": sorted(set(model_cfg.get("compress_ratios", []))),
    }
    model_ok = (
        model_checks["config_exists"]
        and model_checks["architecture"] == "DeepseekV4ForCausalLM"
        and model_checks["model_type"] == "deepseek_v4"
        and model_checks["head_dim"] == 512
        and model_checks["qk_rope_head_dim"] == 64
        and model_checks["index_topk"] <= 8192
        and set(model_checks["compress_ratios"]).issubset({0, 4, 128})
    )

    return {
        "root": str(ROOT),
        "checks": checks,
        "model": model_checks,
        "static_ready": all(check["ok"] for check in checks) and model_ok,
    }


def parse_sse_line(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return {"done": True}
    return json.loads(payload)


def run_openai_stream(args: argparse.Namespace) -> int:
    endpoint = args.endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    start = time.perf_counter()
    first_token_time = None
    completion_tokens = None
    finish_reason = None
    pieces: list[str] = []
    delta_chunks = 0

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                chunk = parse_sse_line(raw_line)
                if chunk is None:
                    continue
                if chunk.get("done"):
                    break
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                content = (choice.get("delta") or {}).get("content") or ""
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    delta_chunks += 1
                    pieces.append(content)
    except urllib.error.URLError as exc:
        print(json.dumps({"request_failed": str(exc)}, indent=2))
        return 1

    end = time.perf_counter()
    generated = completion_tokens if completion_tokens is not None else delta_chunks
    decode_window_s = None if first_token_time is None else max(end - first_token_time, 1e-9)
    decode_tokens_per_s = None if decode_window_s is None else generated / decode_window_s
    print(
        json.dumps(
            {
                "TTFT": None if first_token_time is None else first_token_time - start,
                "total_s": end - start,
                "completion_tokens": completion_tokens,
                "delta_chunks_when_usage_missing": delta_chunks,
                "decode_window_s": decode_window_s,
                "decode_tokens_per_s": decode_tokens_per_s,
                "finish_reason": finish_reason,
                "output_preview": "".join(pieces)[: args.preview_chars],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "openai-stream"], default="inspect")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--prompt", default="你好，请用一句话说明你是谁。")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--preview-chars", type=int, default=400)
    args = parser.parse_args()

    if args.mode == "inspect":
        report = inspect_static()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"root: {report['root']}")
            print(f"static_ready: {report['static_ready']}")
            for check in report["checks"]:
                print(f"- {'ok' if check['ok'] else 'missing'}: {check['name']}")
            print(f"model: {report['model']}")
        return 0 if report["static_ready"] else 2
    return run_openai_stream(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Run inspect before committing**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json
```

Expected after Tasks 1-2: JSON prints `"static_ready": true`.

- [x] **Step 3: Commit Task 4**

Run:

```bash
git add benchmarks/deepseek_v4_flashmla_sm70.py
git diff --cached --check
git commit -m "新增 DeepSeek V4 FlashMLA SM70 检查脚本"
```

## Task 5: Run Sparse FlashMLA Kernel Smokes Through vLLM

**Files:**
- Verify: `tests/kernels/attention/test_flashmla_sparse.py`
- Verify: `tests/v1/attention/test_sparse_mla_backends.py`

- [x] **Step 1: Run the low-level sparse FlashMLA smoke on a V100**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/kernels/attention/test_flashmla_sparse.py::test_sparse_flashmla_decode_smoke -q
```

Expected: the decode smoke passes or reaches a concrete `_flashmla_C::sparse_decode_fwd` kernel error. If it reaches a kernel error, record the full stack and compare against `/mnt/data/apps/FlashMLA/benchmark/bench_sm70_sparse_decode.py --cases quick`.

- [x] **Step 2: Run the sparse prefill smoke on a V100**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/kernels/attention/test_flashmla_sparse.py::test_sparse_flashmla_prefill_smoke -q
```

Expected: the prefill smoke passes. If the test fails because it calls the old name `flash_mla_sparse_prefill`, change that test call to `flash_mla_sparse_fwd` and rerun this same command.

- [x] **Step 3: Run a small v1 sparse MLA backend case**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/v1/attention/test_sparse_mla_backends.py::test_sparse_backend_decode_correctness \
  -q -k 'FlashMLA and small_prefill and fp8_ds_mla and tensor_parallel_size1 and block_size64'
```

Expected: the selected FlashMLA sparse backend case passes or reveals the first vLLM metadata mismatch after the Python gates are open.

- [x] **Step 4: Commit any test compatibility fix from this task**

If Step 2 required renaming `flash_mla_sparse_prefill` to `flash_mla_sparse_fwd`, run:

```bash
git add tests/kernels/attention/test_flashmla_sparse.py
git diff --cached --check
git commit -m "修复 FlashMLA sparse prefill 测试入口名"
```

If no file changed, record no commit for this task.

## Task 6: Run DeepSeek V4 Flash End-To-End Smoke

**Files:**
- Verify: `benchmarks/deepseek_v4_flashmla_sm70.py`
- Runtime target: `/mnt/data6/models/DeepSeek-V4-Flash`

Current status as of 2026-05-01 14:23 CST:

- Server startup on `CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9` succeeds on port
  `18080`: `DeepseekV4ForCausalLM` is resolved, `kv_cache_dtype` is normalized
  to `fp8_ds_mla`, FlashMLA sparse block size is forced to 64, 46 checkpoint
  shards load, KV cache initializes, and Uvicorn reaches application startup.
- Fixed one concrete SM70 cache bug found after the first semantic failure:
  `indexer_k_quant_and_cache` wrote correct scales but zero FP8 value bytes on
  SM70, and `quantize_and_insert_k_cache` still used Triton `tl.float8e4nv`.
  SM70 now uses torch correctness fallbacks for DeepSeek V4 attention/indexer
  K-cache quant insert and indexer gather.
- **Root cause found and fixed (2026-05-01 22:40 CST):**
  `DeepseekV4MoE.forward` and `_forward_fused_moe` were missing
  `tensor_model_parallel_all_reduce` on the non-MegaMoE (FusedMoE) path.
  MegaMoE handles reduce internally, but FusedMoE's `reduce_results` defaults
  to `False`, so MoE output was never synchronized across TP ranks. Each rank
  had only its partial sum, causing hidden-state divergence from layer 0 onward
  and garbled output. Fix: add `experts.maybe_all_reduce_tensor_model_parallel`
  after MoE forward, matching the DeepSeek V2 implementation.
- OpenAI stream now returns semantically correct output:
  `你好！我是DeepSeek，由深度求索公司创造的AI助手，乐于为你提供热情、细腻的帮助。`
  `TTFT=2.4s`, `decode_tokens_per_s=1.6`, `finish_reason=stop`,
  `completion_tokens=25`.
- Latest OpenAI-stream result after the cache fallback fix: `TTFT=2.688s`,
  `completion_tokens=29`, `decode_tokens_per_s=1.351`, `finish_reason=stop`;
  `output_preview` still contains `Banay...<|end_of_repo_name|>...`-style
  invalid content.
- Quick LM-head check after the failed smoke: `sm70_f16_prepare` +
  `sm70_f16_gemm_out` matches `x @ weight.T` on random fp16 shapes
  (`max_diff <= 0.03125` for the checked shapes), so the default SM70 LM-head
  fast path is not the current lead suspect.
- **MHC fast-path semantic regression isolated (2026-05-02 14:38 CST):**
  `08398d167` made the no-TileLang/no-DeepGEMM SM70 mHC fused Triton path the
  default. With that fast path enabled, the exact prompt
  `请只输出这五个字符，不要输出其它内容：ZX-42` starts with `已为您...` and mutates
  `42` into `4.2`; the identity prompt also reaches `finish_reason=length`.
  Restarting the same 8xV100 server with `VLLM_SM70_MHC_FAST=0` restores
  `ZX-42` with EOS at logprob 0.0 and restores the identity smoke to
  `finish_reason=stop`. The fused mHC path is now opt-in only until its
  full-model semantics are fixed.

- [x] **Step 1: Start the OpenAI-compatible server on the requested GPUs**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9 \
VLLM_USE_V1=1 \
VLLM_LOGGING_LEVEL=DEBUG \
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --served-model-name /mnt/data6/models/DeepSeek-V4-Flash \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --enforce-eager \
  --port 8000
```

Expected server log evidence:

- Resolved architecture is `DeepseekV4ForCausalLM`.
- KV cache dtype is normalized to `fp8_ds_mla`.
- `V4_FLASHMLA_SPARSE` or `DeepseekV4FlashMLASparseBackend` appears in backend selection or debug logs.
- `_flashmla_C` imports successfully.
- No fallback to a root checkout appears in Python file paths.

- [x] **Step 2: Run OpenAI-stream smoke and compute decode throughput from streaming timestamps**

In a second shell:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
python benchmarks/deepseek_v4_flashmla_sm70.py \
  --mode openai-stream \
  --endpoint http://127.0.0.1:8000/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "你好，请用一句话说明你是谁。" \
  --max-tokens 32 \
  --timeout 600
```

Expected:

- JSON includes `TTFT`, `decode_tokens_per_s`, `finish_reason`, and `output_preview`.
- `decode_tokens_per_s` is computed from first-token-to-end wall-clock time, not from framework throughput fields.
- `finish_reason` is `stop` or `length`.
- The output preview is semantically coherent for a one-sentence identity prompt.

- [x] **Step 3: If startup fails, classify the first blocker**

Use this classification:

- Build/import blocker: `_flashmla_C` missing, CMake did not compile SM70 sources, or `vllm.__file__` points outside the worktree.
- Gate blocker: backend or runtime probe still says SM70 is unsupported.
- Kernel blocker: stack enters `_flashmla_C::sparse_decode_fwd` or `_flashmla_C::sparse_prefill_fwd` and fails inside FlashMLA.
- Model/runtime blocker: failure happens in loader, MoE, quantization, scheduler, memory planning, or MTP before FlashMLA sparse kernels are called.

Record the first failing stack frame and the exact launch command before changing code.

- [x] **Step 4: Commit the smoke helper result documentation**

If the smoke succeeds, append the command and JSON result to `docs/models/supported_models.md` under the `DeepseekV4ForCausalLM` note added in Task 7. If it fails, append only the first blocker classification and command to the same note.

Run:

```bash
git add docs/models/supported_models.md
git diff --cached --check
git commit -m "记录 DeepSeek V4 Flash SM70 smoke 结果"
```

## Task 7: Document The SM70 DeepSeek V4 Support Boundary

**Files:**
- Modify: `docs/models/supported_models.md`
- Test: `git diff --check`

- [x] **Step 1: Add a support-boundary note**

In `docs/models/supported_models.md`, near the `DeepseekV4ForCausalLM` row or immediately below the table, add:

```markdown
> Local SM70 note: `DeepseekV4ForCausalLM` on V100-class GPUs requires the
> FlashMLA SM70 opt-in build (`FLASH_MLA_SRC_DIR=/mnt/data/apps/FlashMLA`,
> `FLASH_MLA_ENABLE_SM70=1`, `FLASH_MLA_DISABLE_SM100=1`,
> `TORCH_CUDA_ARCH_LIST=7.0`). The supported SM70 attention path is the
> sparse FlashMLA path for `fp8_ds_mla` MODEL1 layout; dense FlashMLA remains
> Hopper-only.
```

- [x] **Step 2: Run markdown diff hygiene**

Run:

```bash
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
git diff --check docs/models/supported_models.md
```

Expected: no whitespace errors.

- [x] **Step 3: Commit Task 7**

Run:

```bash
git add docs/models/supported_models.md
git diff --cached --check
git commit -m "说明 DeepSeek V4 Flash 的 SM70 支持边界"
```

## Final Verification Bundle

Run this bundle before reporting the branch ready:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split
pytest tests/v1/attention/test_flashmla_sm70_sparse_support.py \
  tests/build/test_flashmla_sm70_build_contract.py -q
python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/kernels/attention/test_flashmla_sparse.py::test_sparse_flashmla_decode_smoke \
  tests/kernels/attention/test_flashmla_sparse.py::test_sparse_flashmla_prefill_smoke -q
git diff --check
```

Expected:

- Focused Python tests pass.
- Inspect reports `"static_ready": true`.
- Sparse decode and sparse prefill smokes pass on V100 or stop at a concrete FlashMLA kernel issue with a captured stack.
- `git diff --check` reports no whitespace errors.

The branch is ready for the full model smoke only after the final verification bundle passes or the remaining failure is classified as a model/runtime blocker unrelated to FlashMLA sparse selection.
