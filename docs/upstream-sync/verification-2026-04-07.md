# 2026-04-07 Verification Log

## Environment

- Worktree: `feature/vllm-0190-upstream-split`
- Conda env: `gptq`
- Known pre-plan gap: `pytest` import failed in `tests/conftest.py` because
  `tblib` was missing; runs outside `gptq` also failed because `torch` was not
  installed.

## Baseline Smoke

- Command: `source /home/z/anaconda3/etc/profile.d/conda.sh && conda activate gptq && pytest tests/test_version.py tests/test_envs.py tests/utils_/test_import_utils.py -q`
- Result: `ModuleNotFoundError: No module named 'vllm._C'`
- Notes:
  - `torch==2.9.1+cu128` imports successfully in `gptq`.
  - `tblib==3.2.2` imports successfully in `gptq`.
  - The next real blocker is that the local C/CUDA extension has not been built
    for this worktree yet, so `tests/conftest.py` cannot finish importing the
    CUDA platform path.

## Final Verification

- Branch: `feature/vllm-0190-upstream-split-clean`
- `git describe --tags --always --dirty`:
  - Result: `v0.0.2-13-g459f99ce8`
- `python -m compileall vllm csrc tests examples docker`:
  - Result: `EXIT:0`
- `python - <<'PY' ... import vllm; import vllm.version; print(vllm.version.__version__)`:
  - Result: printed `dev`
  - Notes: emitted a runtime warning that `vllm._version` is not present in the
    source tree yet.
- `python -m vllm.entrypoints.openai.api_server --help`:
  - Result: import traceback at `vllm/platforms/cuda.py:16`
  - First failure: `ModuleNotFoundError: No module named 'vllm._C'`
- `pytest tests/test_version.py tests/test_envs.py tests/utils_/test_import_utils.py -q`:
  - Result: import traceback at `vllm/platforms/cuda.py:16`
  - First failure: `ModuleNotFoundError: No module named 'vllm._C'`
- `python -m build --wheel --no-isolation --outdir /tmp/1cat-v019-wheel-smoke`:
  - Result: packaging advanced past the earlier tag-parse failure, but stopped on
    missing build dependency
  - First failure: `ERROR Missing dependencies: torch==2.10.0`
- `nvidia-smi -L`:
  - Result: visible GPUs include ten `Tesla PG503-216` devices and one
    `NVIDIA GeForce RTX 3080 Ti Laptop GPU`
- Summary:
  - The upgraded clean-history branch is syntactically valid and the tag/layout
    issue that broke `setuptools_scm` has been removed.
  - Remaining blockers are environment/build related:
    `vllm._C` has not been built for this worktree, and the `gptq` environment
    does not satisfy the wheel build's `torch==2.10.0` requirement.
