# 1Cat-vLLM Upstream Split and v0.19.0 Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `1Cat-vLLM` on top of upstream `vLLM v0.19.0`, manage `lmdeploy/` as an upstream `v0.12.1` subtree, and preserve only two replayable 1Cat customization commits for future upgrades.

**Architecture:** Implement the work in two phases inside the existing isolated worktree. First reconstruct the current public product as `upstream vLLM v0.15.0` plus `upstream lmdeploy v0.12.1` plus two isolated local deltas, then replace the root baseline with upstream `vLLM v0.19.0` and replay only the vLLM-side local delta while keeping `lmdeploy/` pinned as subtree content.

**Tech Stack:** Git worktrees, local upstream mirrors at `/mnt/data/apps/vllm` and `/mnt/data/apps/lmdeploy`, `git subtree`, Python packaging, CMake/CUDA integration, Conda (`gptq`), `pytest`, `compileall`.

---

### Task 1: Repair The Baseline Verification Environment

**Files:**
- Create: `docs/upstream-sync/verification-2026-04-07.md`
- Test: `tests/test_version.py`
- Test: `tests/test_envs.py`
- Test: `tests/utils_/test_import_utils.py`

- [ ] **Step 1: Write the initial verification log**

```markdown
# 2026-04-07 Verification Log

## Environment

- Worktree: `feature/vllm-0190-upstream-split`
- Conda env: `gptq`
- Known pre-plan gap: `pytest` import failed in `tests/conftest.py` because
  `tblib` was missing; runs outside `gptq` also failed because `torch` was not
  installed.

## Baseline Smoke

- Command:
- Result:
- Notes:
```

- [ ] **Step 2: Install the missing import-time dependency in `gptq`**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python -m pip install tblib
python - <<'PY'
import torch
import tblib
print(torch.__version__)
print(tblib.__version__)
PY
```

Expected: two version strings print and there is no `ModuleNotFoundError`.

- [ ] **Step 3: Re-run the lightweight baseline smoke tests**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
pytest tests/test_version.py tests/test_envs.py tests/utils_/test_import_utils.py -q
```

Expected: the run gets past `tests/conftest.py` import without `No module named 'torch'` or `No module named 'tblib'`. If a new failure appears, record the literal first failure instead of guessing.

- [ ] **Step 4: Record the actual baseline result in the verification log**

Append the literal command and literal terminal result you saw:

```markdown
## Baseline Smoke

- Command: `source /home/z/anaconda3/etc/profile.d/conda.sh && conda activate gptq && pytest tests/test_version.py tests/test_envs.py tests/utils_/test_import_utils.py -q`
- Result: `Write the exact first PASS summary or the exact first failing line you saw.`
- Notes: `Record whether the run fully passed or what the next real blocker is.`
```

- [ ] **Step 5: Commit the verification log**

```bash
git add docs/upstream-sync/verification-2026-04-07.md
git commit -m "docs: record baseline verification status"
```

### Task 2: Add Upstream Sync Docs And Audit Manifests

**Files:**
- Create: `docs/design/upstream_sync.md`
- Create: `docs/upstream-sync/root-v015-files.txt`
- Create: `docs/upstream-sync/lmdeploy-v0121-files.txt`
- Modify: `.git/config` (local-only remote aliases and snapshot refs)
- Test: `docs/design/upstream_sync.md`
- Test: `docs/upstream-sync/root-v015-files.txt`
- Test: `docs/upstream-sync/lmdeploy-v0121-files.txt`

- [ ] **Step 1: Write the upstream sync guide**

~~~markdown
# Upstream Sync Workflow

## Baselines

- Root upstream baseline: `vLLM v0.19.0`
- Vendored subtree baseline: `lmdeploy v0.12.1`
- Local replay commits:
  - `1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations`
  - `1cat(lmdeploy): turbomind SM70 kernel customizations`

## Local Mirrors Used In This Workspace

- `vllm-upstream` -> `/mnt/data/apps/vllm`
- `lmdeploy-upstream` -> `/mnt/data/apps/lmdeploy`

## Future Upgrade Commands

```bash
git fetch vllm-upstream --tags
git subtree pull --prefix=lmdeploy /mnt/data/apps/lmdeploy v0.12.1 --squash
```

For root `vLLM` upgrades, import the new root snapshot and then replay the
single `1cat(vllm)` commit with a `git diff phase1/1cat-vllm^ phase1/1cat-vllm | git apply --3way`
style flow.

Do not fetch `lmdeploy` tags into the `1Cat-vLLM` repository itself, because
`vLLM` and `lmdeploy` reuse the same `v0.*` tag names.
~~~

- [ ] **Step 2: Add local upstream remotes and snapshot refs**

Run:

```bash
git remote remove vllm-upstream 2>/dev/null || true
git remote remove lmdeploy-upstream 2>/dev/null || true
git remote add vllm-upstream /mnt/data/apps/vllm
git remote add lmdeploy-upstream /mnt/data/apps/lmdeploy
git fetch vllm-upstream --tags
git branch -f snapshot/1cat-current origin/main
git branch -f snapshot/plan-start HEAD
```

Expected: `snapshot/1cat-current` points at `origin/main`, both local remotes exist in `.git/config`, and no `lmdeploy` tags are imported into the `1Cat-vLLM` tag namespace.

- [ ] **Step 3: Generate fresh upstream snapshots and audit manifests**

Run:

```bash
rm -rf /tmp/1cat-vllm-v015 /tmp/1cat-vllm-v019 /tmp/1cat-lmdeploy-v0121
mkdir -p /tmp/1cat-vllm-v015 /tmp/1cat-vllm-v019 /tmp/1cat-lmdeploy-v0121
git -C /mnt/data/apps/vllm archive v0.15.0 | tar -x -C /tmp/1cat-vllm-v015
git -C /mnt/data/apps/vllm archive v0.19.0 | tar -x -C /tmp/1cat-vllm-v019
git -C /mnt/data/apps/lmdeploy archive v0.12.1 | tar -x -C /tmp/1cat-lmdeploy-v0121
diff -qr /tmp/1cat-vllm-v015 . \
  --exclude=.git \
  --exclude=.worktrees \
  --exclude=docs/superpowers \
  --exclude=docs/upstream-sync \
  --exclude=docs/design/upstream_sync.md \
  --exclude=lmdeploy \
  --exclude=build \
  --exclude=dist-cu128-sm70 \
  --exclude=vllm.egg-info \
  --exclude=测试结果 \
  2>/dev/null | sed 's#^#- #' > docs/upstream-sync/root-v015-files.txt
diff -qr /tmp/1cat-lmdeploy-v0121 lmdeploy \
  2>/dev/null | sed 's#^#- #' > docs/upstream-sync/lmdeploy-v0121-files.txt
```

Expected: both text files are populated and describe the current local delta against the intended upstream baselines.

- [ ] **Step 4: Verify the manifest boundaries**

Run:

```bash
rg -n '^- .*lmdeploy/' docs/upstream-sync/root-v015-files.txt
rg -n 'mma.h|dispatch_cache.cu|gemm.cu|sm70_884_16.cu|sm70_884_4.cu' docs/upstream-sync/lmdeploy-v0121-files.txt
wc -l docs/upstream-sync/lmdeploy-v0121-files.txt
```

Expected: the first command returns no output, the second command prints the five known kernel files, and the line count stays small enough to audit manually.

- [ ] **Step 5: Commit the docs and manifests**

```bash
git add docs/design/upstream_sync.md docs/upstream-sync/root-v015-files.txt docs/upstream-sync/lmdeploy-v0121-files.txt
git commit -m "docs: add upstream sync manifests"
```

### Task 3: Import The Upstream v0.15.0 Root Baseline

**Files:**
- Modify: `.buildkite/`
- Modify: `.dockerignore`
- Modify: `.github/CODEOWNERS`
- Modify: `.pre-commit-config.yaml`
- Modify: `.gitignore`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `benchmarks/`
- Modify: `mkdocs.yaml`
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `cmake/`
- Modify: `csrc/`
- Modify: `docker/`
- Modify: `docs/`
- Modify: `examples/`
- Modify: `requirements/`
- Modify: `tests/`
- Modify: `vllm/`
- Test: `docs/upstream-sync/root-v015-files.txt`

- [ ] **Step 1: Replace the repository root with the upstream `v0.15.0` snapshot**

Run:

```bash
rsync -a --delete \
  --exclude '.git' \
  --exclude '.worktrees' \
  --exclude 'docs/superpowers' \
  --exclude 'docs/upstream-sync' \
  --exclude 'docs/design/upstream_sync.md' \
  --exclude 'lmdeploy' \
  --exclude 'build' \
  --exclude 'dist-cu128-sm70' \
  --exclude 'vllm.egg-info' \
  --exclude '测试结果' \
  /tmp/1cat-vllm-v015/ ./
```

Expected: root-level product files now match upstream `v0.15.0`, while `lmdeploy/` and planning docs remain untouched.

- [ ] **Step 2: Reapply the local worktree ignore rule**

Run:

```bash
grep -qxF '.worktrees/' .gitignore || printf '\n.worktrees/\n' >> .gitignore
```

Expected: `.gitignore` keeps the project-local worktree ignore line after the upstream import.

- [ ] **Step 3: Inspect the changed paths before committing**

Run:

```bash
git status --short
git diff --name-only -- . ':(exclude)lmdeploy' ':(exclude)docs/superpowers' ':(exclude)docs/upstream-sync' ':(exclude)docs/design/upstream_sync.md'
```

Expected: the diff is root-only and does not include `lmdeploy/`.

- [ ] **Step 4: Commit the upstream root baseline**

```bash
git add . ':(exclude)docs/superpowers'
git commit -m "upstream(vllm): import v0.15.0 root baseline"
```

- [ ] **Step 5: Tag the clean root baseline**

```bash
git tag -f phase1/upstream-vllm-v015 HEAD
```

### Task 4: Replace Vendored lmdeploy With A v0.12.1 Subtree

**Files:**
- Modify: `lmdeploy/`
- Test: `lmdeploy/src/turbomind/kernels/gemm/gemm.cu`

- [ ] **Step 1: Remove the old vendored `lmdeploy/` tree**

```bash
git rm -r lmdeploy
git commit -m "chore: remove vendored lmdeploy before subtree import"
```

- [ ] **Step 2: Add the subtree import from the local upstream mirror**

```bash
git subtree add --prefix=lmdeploy /mnt/data/apps/lmdeploy v0.12.1 --squash -m "upstream(lmdeploy): add v0.12.1 subtree"
```

Expected: Git creates a single subtree import commit rooted at `lmdeploy/`.

- [ ] **Step 3: Verify the subtree matches upstream `v0.12.1` exactly**

Run:

```bash
diff -qr /tmp/1cat-lmdeploy-v0121 lmdeploy
```

Expected: no output.

- [ ] **Step 4: Tag the subtree baseline**

```bash
git tag -f phase1/upstream-lmdeploy-v0121 HEAD
```

- [ ] **Step 5: Confirm the subtree commit touched only `lmdeploy/`**

Run:

```bash
git show --name-only --format='' HEAD | sort -u
```

Expected: every listed path starts with `lmdeploy/`.

### Task 5: Reapply The Current 1Cat Root And vLLM Customizations

**Files:**
- Modify: `.buildkite/`
- Modify: `.dockerignore`
- Modify: `.github/CODEOWNERS`
- Modify: `.pre-commit-config.yaml`
- Modify: `README.md`
- Modify: `OPEN_SOURCE_SM70_GUIDE.md`
- Modify: `CMakeLists.txt`
- Modify: `benchmarks/`
- Modify: `docs/`
- Modify: `examples/`
- Modify: `mkdocs.yaml`
- Modify: `setup.py`
- Modify: `pyproject.toml`
- Modify: `docker/`
- Modify: `cmake/`
- Modify: `csrc/`
- Modify: `requirements/`
- Modify: `tests/`
- Modify: `vllm/`
- Test: `docs/upstream-sync/root-v015-files.txt`

- [ ] **Step 1: Export the current root-only delta as a replay patch**

Run:

```bash
git diff --binary HEAD snapshot/1cat-current -- \
  . \
  ':(exclude)lmdeploy' \
  ':(exclude).worktrees' \
  ':(exclude)docs/superpowers' \
  ':(exclude)docs/upstream-sync' \
  ':(exclude)docs/design/upstream_sync.md' \
  > /tmp/1cat-vllm-phase1.patch
```

Expected: `/tmp/1cat-vllm-phase1.patch` is non-empty.

- [ ] **Step 2: Apply the replay patch onto the clean `v0.15.0` plus subtree baseline**

Run:

```bash
git apply --3way /tmp/1cat-vllm-phase1.patch
```

Expected: patch application succeeds or leaves only root-level conflicts to resolve. Do not continue until there are no unresolved markers.

- [ ] **Step 3: Verify that the staged delta does not include `lmdeploy/`**

Run:

```bash
git diff --name-only -- . | rg '^lmdeploy/' && exit 1 || true
```

Expected: no output.

- [ ] **Step 4: Commit the isolated root customization layer**

```bash
git add .buildkite .dockerignore .github/CODEOWNERS .pre-commit-config.yaml README.md OPEN_SOURCE_SM70_GUIDE.md CMakeLists.txt benchmarks docs examples mkdocs.yaml setup.py pyproject.toml docker cmake csrc requirements tests vllm
git commit -m "1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations"
```

- [ ] **Step 5: Tag the phase-1 replay unit**

```bash
git tag -f phase1/1cat-vllm HEAD
```

### Task 6: Reapply The Current lmdeploy Kernel Customizations And Validate Phase 1

**Files:**
- Modify: `lmdeploy/src/turbomind/kernels/core/mma.h`
- Modify: `lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu`
- Modify: `lmdeploy/src/turbomind/kernels/gemm/gemm.cu`
- Modify: `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu`
- Modify: `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`
- Modify: `docs/upstream-sync/verification-2026-04-07.md`
- Test: `docs/upstream-sync/verification-2026-04-07.md`

- [ ] **Step 1: Export the lmdeploy-only replay patch**

Run:

```bash
git diff --binary HEAD snapshot/1cat-current -- \
  lmdeploy/src/turbomind/kernels/core/mma.h \
  lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu \
  lmdeploy/src/turbomind/kernels/gemm/gemm.cu \
  lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu \
  lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu \
  > /tmp/1cat-lmdeploy-phase1.patch
```

Expected: `/tmp/1cat-lmdeploy-phase1.patch` is non-empty and only references the five known kernel files.

- [ ] **Step 2: Apply the lmdeploy replay patch and verify the file scope**

Run:

```bash
git apply --3way /tmp/1cat-lmdeploy-phase1.patch
git diff --name-only -- lmdeploy
```

Expected: the second command lists only the five kernel files above.

- [ ] **Step 3: Commit the isolated lmdeploy customization layer**

```bash
git add lmdeploy/src/turbomind/kernels/core/mma.h \
  lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu \
  lmdeploy/src/turbomind/kernels/gemm/gemm.cu \
  lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu \
  lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu
git commit -m "1cat(lmdeploy): turbomind SM70 kernel customizations"
```

- [ ] **Step 4: Tag the phase-1 lmdeploy replay unit**

```bash
git tag -f phase1/1cat-lmdeploy HEAD
```

- [ ] **Step 5: Verify the phase-1 tree still matches the current product**

Run:

```bash
git diff --stat snapshot/1cat-current -- \
  . \
  ':(exclude).gitignore' \
  ':(exclude).worktrees' \
  ':(exclude)docs/superpowers' \
  ':(exclude)docs/upstream-sync' \
  ':(exclude)docs/design/upstream_sync.md'
python -m compileall vllm lmdeploy/lmdeploy tests examples docker
```

Expected: the diff command prints nothing, and `compileall` completes without syntax errors.

- [ ] **Step 6: Record the phase-1 verification result and commit it**

Append the actual outputs to `docs/upstream-sync/verification-2026-04-07.md`, then run:

```bash
git add docs/upstream-sync/verification-2026-04-07.md
git commit -m "docs: record phase1 equivalence status"
git tag -f phase1/clean-baseline HEAD
```

### Task 7: Import The Upstream v0.19.0 Root Baseline

**Files:**
- Modify: `.buildkite/`
- Modify: `.dockerignore`
- Modify: `.github/CODEOWNERS`
- Modify: `.pre-commit-config.yaml`
- Modify: `.gitignore`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `benchmarks/`
- Modify: `mkdocs.yaml`
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `cmake/`
- Modify: `csrc/`
- Modify: `docker/`
- Modify: `docs/`
- Modify: `examples/`
- Modify: `requirements/`
- Modify: `tests/`
- Modify: `vllm/`
- Test: `CMakeLists.txt`

- [ ] **Step 1: Export the phase-1 vLLM replay patch that will be ported to `v0.19.0`**

Run:

```bash
git diff --binary phase1/1cat-vllm^ phase1/1cat-vllm -- \
  . \
  ':(exclude)lmdeploy' \
  ':(exclude).worktrees' \
  ':(exclude)docs/superpowers' \
  ':(exclude)docs/upstream-sync' \
  ':(exclude)docs/design/upstream_sync.md' \
  > /tmp/1cat-vllm-v019-replay.patch
```

Expected: `/tmp/1cat-vllm-v019-replay.patch` is non-empty.

- [ ] **Step 2: Replace the repository root with the upstream `v0.19.0` snapshot**

Run:

```bash
rsync -a --delete \
  --exclude '.git' \
  --exclude '.worktrees' \
  --exclude 'docs/superpowers' \
  --exclude 'docs/upstream-sync' \
  --exclude 'docs/design/upstream_sync.md' \
  --exclude 'lmdeploy' \
  --exclude 'build' \
  --exclude 'dist-cu128-sm70' \
  --exclude 'vllm.egg-info' \
  --exclude '测试结果' \
  /tmp/1cat-vllm-v019/ ./
```

Expected: root files now match upstream `v0.19.0`, while `lmdeploy/` and planning docs remain untouched.

- [ ] **Step 3: Reapply the local worktree ignore rule**

Run:

```bash
grep -qxF '.worktrees/' .gitignore || printf '\n.worktrees/\n' >> .gitignore
```

Expected: `.gitignore` still ignores `.worktrees/`.

- [ ] **Step 4: Commit the upstream `v0.19.0` root baseline**

```bash
git add . ':(exclude)docs/superpowers'
git commit -m "upstream(vllm): import v0.19.0 root baseline"
```

- [ ] **Step 5: Verify that this import did not touch `lmdeploy/`**

Run:

```bash
git show --name-only --format='' HEAD | rg '^lmdeploy/' && exit 1 || true
```

Expected: no output.

### Task 8: Replay And Port The Final 1Cat vLLM Customization Layer

**Files:**
- Modify: `.buildkite/`
- Modify: `.dockerignore`
- Modify: `.github/CODEOWNERS`
- Modify: `.pre-commit-config.yaml`
- Modify: `README.md`
- Modify: `OPEN_SOURCE_SM70_GUIDE.md`
- Modify: `CMakeLists.txt`
- Modify: `benchmarks/`
- Modify: `docs/`
- Modify: `examples/`
- Modify: `mkdocs.yaml`
- Modify: `setup.py`
- Modify: `pyproject.toml`
- Modify: `docker/`
- Modify: `cmake/`
- Modify: `csrc/`
- Modify: `requirements/`
- Modify: `tests/`
- Modify: `vllm/`
- Modify: `docs/design/upstream_sync.md`
- Test: `docs/design/upstream_sync.md`

- [ ] **Step 1: Apply the saved phase-1 replay patch onto the `v0.19.0` root**

Run:

```bash
git apply --3way /tmp/1cat-vllm-v019-replay.patch || true
git diff --name-only --diff-filter=U | tee /tmp/1cat-v019-conflicts.txt
```

Expected: either the patch applies cleanly, or `/tmp/1cat-v019-conflicts.txt` lists the exact files that still need manual merge work.

- [ ] **Step 2: Restore the uniquely 1Cat-owned files before merging the remaining conflicts**

Run:

```bash
git checkout phase1/1cat-vllm -- \
  OPEN_SOURCE_SM70_GUIDE.md \
  docker/Dockerfile.sm70-wheel \
  docker/entrypoint.sm70.sh \
  csrc/quantization/awq/awq_sm70_gemm.cu \
  csrc/quantization/awq/tm_registry_sm70.cu
git diff --binary phase1/1cat-vllm^ phase1/1cat-vllm -- \
  .buildkite .dockerignore .github/CODEOWNERS .pre-commit-config.yaml CMakeLists.txt README.md benchmarks docs examples mkdocs.yaml setup.py pyproject.toml cmake csrc requirements tests vllm \
  > /tmp/1cat-v019-guided-merge.patch
```

Expected: the unique files are present again, and `/tmp/1cat-v019-guided-merge.patch` is available as a reference while resolving the remaining `v0.19.0` conflicts.

- [ ] **Step 3: Update the upstream sync guide to the final repository shape**

Replace `docs/design/upstream_sync.md` with:

~~~markdown
# Upstream Sync Workflow

## Current Baselines

- Root upstream baseline: `vLLM v0.19.0`
- Vendored subtree baseline: `lmdeploy v0.12.1`

## Replay Commits

- `1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations`
- `1cat(lmdeploy): turbomind SM70 kernel customizations`

## Local Mirror Commands

```bash
git fetch vllm-upstream --tags
git subtree pull --prefix=lmdeploy /mnt/data/apps/lmdeploy v0.12.1 --squash
git diff final/1cat-vllm^ final/1cat-vllm -- \
  . ':(exclude)lmdeploy' | git apply --3way
```

Use the subtree command only for `lmdeploy/`. Use the root import plus replay
flow only for `vLLM` upgrades.
~~~

- [ ] **Step 4: Verify that the final local delta still excludes `lmdeploy/`**

Run:

```bash
git diff --name-only -- . | rg '^lmdeploy/' && exit 1 || true
```

Expected: no output.

- [ ] **Step 5: Commit the final replayed root customization layer**

```bash
git add .buildkite .dockerignore .github/CODEOWNERS .pre-commit-config.yaml README.md OPEN_SOURCE_SM70_GUIDE.md CMakeLists.txt benchmarks docs examples mkdocs.yaml setup.py pyproject.toml docker cmake csrc requirements tests vllm docs/design/upstream_sync.md
git commit -m "1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations"
git tag -f final/1cat-vllm HEAD
```

### Task 9: Run Final Verification And Record Remaining Gaps

**Files:**
- Modify: `docs/upstream-sync/verification-2026-04-07.md`
- Test: `tests/test_version.py`
- Test: `tests/test_envs.py`
- Test: `tests/utils_/test_import_utils.py`

- [ ] **Step 1: Run syntax and import smoke checks on the final tree**

Run:

```bash
python -m compileall vllm lmdeploy/lmdeploy tests examples docker
python - <<'PY'
import vllm
import vllm.version
print(vllm.version.__version__)
PY
python -m vllm.entrypoints.openai.api_server --help >/tmp/1cat-v019-api-help.txt
```

Expected: `compileall` succeeds, the Python snippet prints a version string, and the API help command exits successfully.

- [ ] **Step 2: Re-run the lightweight pytest smoke tests in `gptq`**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
pytest tests/test_version.py tests/test_envs.py tests/utils_/test_import_utils.py -q
```

Expected: the same lightweight suite from Task 1 passes on the upgraded tree. If it fails, record the literal first failure.

- [ ] **Step 3: Run one build-path smoke command**

Run:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python -m build --wheel --no-isolation --outdir /tmp/1cat-v019-wheel-smoke
ls -1 /tmp/1cat-v019-wheel-smoke
```

Expected: one wheel file is created in `/tmp/1cat-v019-wheel-smoke`.

- [ ] **Step 4: Run the conditional SM70 smoke command if GPU access is available**

Run:

```bash
nvidia-smi -L
```

If the machine exposes an SM70 GPU, then run one documented help-or-startup smoke path such as:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python -m vllm.entrypoints.openai.api_server --model /models/Qwen3.5-27B-AWQ --help
```

Expected: the command reaches argument parsing without crashing. If no SM70 GPU or model path is available, record that as an environment-limited gap instead of claiming success.

- [ ] **Step 5: Record the final verification results**

Append the literal commands and literal results to `docs/upstream-sync/verification-2026-04-07.md`, including:

```markdown
## Final Verification

- `compileall`:
- `import smoke`:
- `api_server --help`:
- `pytest smoke`:
- `wheel build`:
- `SM70 smoke`:
```

- [ ] **Step 6: Commit the verification log**

```bash
git add docs/upstream-sync/verification-2026-04-07.md
git commit -m "docs: record v0.19.0 upgrade verification"
```
