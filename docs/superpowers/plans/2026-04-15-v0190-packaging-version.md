# v0.19.0 默认打包版本固定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `feature/vllm-0190-upstream-split` 默认构建出的包版本固定为 `0.19.0`，且运行时和 CLI 看到相同版本。

**Architecture:** 只修改 `setup.py` 的默认版本生成路径，保留 `VLLM_VERSION_OVERRIDE` 的显式覆盖能力，并继续让 `setuptools_scm` 写出 `vllm/_version.py`。回归测试复用现有 `tests/test_version.py` 和 `tests/test_embedded_commit.py`，避免为这次发布整理引入新的运行时行为。

**Tech Stack:** Python 3.13, setuptools, setuptools_scm, pytest, editable vLLM build in `conda gptq`

---

## File Map

- Modify: `setup.py`
  - 固定默认打包版本为 `0.19.0`
  - 保留 `VLLM_VERSION_OVERRIDE` 的高优先级覆盖
  - 删除默认路径上的 CUDA/ROCm/precompiled/empty 版本后缀拼接
- Modify: `tests/test_version.py`
  - 增加对精确版本字符串 `0.19.0` 和 `__version_tuple__[:3] == (0, 19, 0)` 的回归断言
- Verify: `tests/test_embedded_commit.py`
  - 继续确认运行时不会退回 `dev`

## Task 1: 固定默认版本并补回归测试

**Files:**
- Modify: `setup.py:983-1018`
- Modify: `tests/test_version.py:1-36`
- Test: `tests/test_version.py`
- Test: `tests/test_embedded_commit.py`

- [ ] **Step 1: 在 `tests/test_version.py` 写出失败回归**

```python
def test_release_version_matches_v0190():
    assert version.__version__ == "0.19.0"
    assert version.__version_tuple__[:3] == (0, 19, 0)
```

- [ ] **Step 2: 运行测试确认当前默认版本不是 `0.19.0`**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
unset PYTHONPATH && \
python setup.py --version && \
pytest -q tests/test_version.py::test_release_version_matches_v0190 \
tests/test_embedded_commit.py::test_embedded_commit_defined'
```

Expected:

- `python setup.py --version` 输出不是精确的 `0.19.0`
- 新增测试失败，报 `version.__version__` 与 `0.19.0` 不一致

- [ ] **Step 3: 最小修改 `setup.py`，让默认版本固定为 `0.19.0`**

```python
DEFAULT_VLLM_RELEASE_VERSION = "0.19.0"


def get_vllm_version() -> str:
    if env_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        print(f"Overriding VLLM version with {env_version} from VLLM_VERSION_OVERRIDE")
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        return get_version(write_to="vllm/_version.py")

    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = DEFAULT_VLLM_RELEASE_VERSION
    return get_version(write_to="vllm/_version.py")
```

Implementation notes:

- 删除当前默认路径中对 `cuda_version`、`rocm_version`、`precompiled`、`empty` 的后缀追加代码
- 不改 `setup()`、`project.name`、CLI 名称或任何推理逻辑

- [ ] **Step 4: 复跑版本测试，确认回归转绿**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
unset PYTHONPATH && \
python setup.py --version && \
pytest -q tests/test_version.py tests/test_embedded_commit.py'
```

Expected:

- `python setup.py --version` 输出精确 `0.19.0`
- `tests/test_version.py` 通过
- `tests/test_embedded_commit.py` 通过

- [ ] **Step 5: 提交版本固定改动**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  setup.py \
  tests/test_version.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "build: 固定默认打包版本为 0.19.0"
```

## Task 2: 验证运行时与 CLI 版本一致

**Files:**
- Modify: none
- Test: `tests/test_version.py`
- Test: `tests/test_embedded_commit.py`

- [ ] **Step 1: 检查运行时版本字符串**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
unset PYTHONPATH && \
python -c "import vllm; print(vllm.__version__)"'
```

Expected:

- 输出精确 `0.19.0`

- [ ] **Step 2: 检查 CLI `--version`**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
unset PYTHONPATH && \
python -m vllm.entrypoints.cli.main --version'
```

Expected:

- 输出包含 `0.19.0`
- 不报 `PackageNotFoundError`

- [ ] **Step 3: 记录最终验证结果，作为发布前口径**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
unset PYTHONPATH && \
python setup.py --version && \
python -c "import vllm; print(vllm.__version__)" && \
python -m vllm.entrypoints.cli.main --version'
```

Expected:

- 三条输出都为 `0.19.0`
