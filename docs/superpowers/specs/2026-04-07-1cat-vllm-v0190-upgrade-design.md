# 1Cat-vLLM v0.19.0 Upgrade and Upstream Split Design

## Goal

Restructure `1Cat-vLLM` so the repository can be upgraded against upstream
projects with clear boundaries, while preserving the current `0.0.2` feature
set, build paths, runtime defaults, docs, and release behavior for the SM70 /
Qwen3.5 / AWQ product direction.

The end state must satisfy two requirements at the same time:

1. The repository root is upgraded from upstream `vLLM v0.15.0` lineage to
   upstream `vLLM v0.19.0`.
2. The repository history exposes the remaining 1Cat-specific work as two
   independent commits:
   - `1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations`
   - `1cat(lmdeploy): turbomind SM70 kernel customizations`

## Current Baseline

The current `1Cat-vLLM` repository is a single repo with vendored upstream
source trees:

- The repository root is structurally much closer to upstream `vLLM v0.15.0`
  than to `v0.19.0`.
- `lmdeploy/` is vendored source rather than a subtree today.
- The vendored `lmdeploy/` tree matches upstream `lmdeploy v0.12.1` closely,
  with the 1Cat-specific kernel changes concentrated in:
  - `lmdeploy/src/turbomind/kernels/core/mma.h`
  - `lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu`
  - `lmdeploy/src/turbomind/kernels/gemm/gemm.cu`
  - `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu`
  - `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`

The current repository also contains release-facing assets outside `vllm/`
itself, including `README.md`, `docker/`, `CMakeLists.txt`, packaging files,
and benchmark artifacts. Those assets are part of the supported public product
and must continue to work after the upgrade.

## Non-Goals

This design does not include an `lmdeploy` version upgrade. For this cycle,
`lmdeploy` is being converted to subtree management and its local modifications
are being isolated into one replayable commit. That keeps the `vLLM` major
upgrade separate from a second independent upstream jump.

This design also does not attempt to preserve the exact existing commit graph.
The objective is a new clean branch with maintainable upstream boundaries, not a
lossless historical replay of the current `main`.

## Repository Shape After The Work

The repository keeps the same external layout:

- `vllm/` remains the root project package.
- `lmdeploy/` remains at `1Cat-vLLM/lmdeploy`.
- Public docs and Docker entrypoints remain in the repository root where users
  expect them.

The change is in how those directories are sourced and maintained:

- The repository root is explicitly aligned to upstream `vLLM v0.19.0`.
- `lmdeploy/` is reintroduced as `git subtree` content rooted at upstream
  `lmdeploy v0.12.1`.
- 1Cat-specific logic is no longer spread across mixed vendor-import commits.
  It is instead carried by the two dedicated commits named above.

## Commit Model

The target branch will be organized so the upstream and local layers are easy to
understand and reapply.

Required logical layers:

1. A clean upstream-aligned repository root based on `vLLM v0.19.0`.
2. A clean subtree import of `lmdeploy v0.12.1` under `lmdeploy/`.
3. One 1Cat commit for repository-root and vLLM-facing customizations.
4. One 1Cat commit for `lmdeploy/` kernel customizations.

The exact number of support commits before those layers can vary if needed for
mechanical setup, but the finished branch must preserve the two local
customization commits as isolated replay units.

Commit ownership rules:

- Changes under `lmdeploy/` that represent local TurboMind kernel behavior stay
  in `1cat(lmdeploy)`.
- Root-level packaging, build integration, runtime defaults, Docker files,
  README updates, benchmarks, release scripts, and any vLLM-side integration
  code stay in `1cat(vllm)`.
- If upstream `v0.19.0` already contains behavior that previously required a
  1Cat patch, that delta is dropped from `1cat(vllm)` instead of preserved
  redundantly.

## Migration Strategy

The migration is split into two phases so history cleanup does not get mixed
with semantic upgrade work.

### Phase 1: Reconstruct a Clean Baseline Equivalent To Current Behavior

Build a new branch in an isolated worktree.

1. Start from the current repository and preserve the project-local worktree
   ignore rule in a support commit.
2. Reconstruct the current repository as explicit layers rather than a monolith:
   - restore an upstream `vLLM v0.15.0` root baseline
   - isolate the current non-`lmdeploy/` 1Cat delta into one vLLM-side patch
   - reintroduce `lmdeploy/` from upstream `v0.12.1` as subtree content
   - isolate the current `lmdeploy/` kernel delta into one patch
3. Verify that the resulting tree is functionally equivalent to the current
   `main`, except for expected metadata differences caused by subtree setup and
   branch-only cleanup.

This phase exists to answer one question before the actual upgrade: can the
current product be expressed as upstream baselines plus two local deltas? If the
answer is no, the repository is still not in an upgradeable shape.

### Phase 2: Upgrade The vLLM Root To Upstream v0.19.0

Once the clean baseline exists, move the repository root from the `v0.15.0`
lineage to upstream `v0.19.0`.

1. Replace the root baseline with upstream `v0.19.0`.
2. Replay `1cat(vllm)` onto the new root.
3. Resolve API, build, and runtime conflicts introduced by the `v0.15.0` to
   `v0.19.0` jump.
4. Keep `lmdeploy/` on top of subtree `v0.12.1` plus `1cat(lmdeploy)` unless a
   strictly required interface adjustment is needed for the root upgrade.

The replay is semantic, not mechanical. If upstream deleted or replaced old
interfaces, the new implementation still belongs in `1cat(vllm)` as long as it
serves the same 1Cat-specific responsibility.

## Conflict Handling Rules

Conflicts must be resolved by responsibility, not by file location alone.

Rules:

- `lmdeploy/` subtree content is authoritative for vendored upstream
  `lmdeploy`.
- Only the `1cat(lmdeploy)` commit may carry local changes to vendored
  TurboMind kernel behavior.
- Root-level changes that wire `lmdeploy` into the vLLM build remain part of
  `1cat(vllm)`, even when they mention `lmdeploy`.
- Root-level docs, release assets, benchmark assets, and Docker paths remain
  part of `1cat(vllm)`.
- If a root-level file must change only because a file moved or an interface
  changed in upstream `v0.19.0`, it still belongs to `1cat(vllm)` if the
  purpose is preserving 1Cat behavior.

This avoids the common failure mode where integration code gets split across the
wrong local commit and future cherry-picks become misleading.

## File Ownership Map

The following areas are expected to be owned by the `1cat(vllm)` commit when
their contents differ from upstream:

- `README.md`
- `OPEN_SOURCE_SM70_GUIDE.md`
- `CMakeLists.txt`
- `setup.py`
- `pyproject.toml`
- `docker/`
- `benchmarks/`
- root `csrc/`, `cmake/`, `requirements/`, `tests/`, and `vllm/` changes needed
  to preserve SM70, AWQ, Triton attention, packaging, and runtime behavior
- release-oriented assets such as `dist-cu128-sm70/` handling and source-build
  vendor integration

The following areas are expected to be owned by the `1cat(lmdeploy)` commit when
they differ from upstream `lmdeploy v0.12.1`:

- `lmdeploy/src/turbomind/kernels/core/mma.h`
- `lmdeploy/src/turbomind/kernels/gemm/dispatch_cache.cu`
- `lmdeploy/src/turbomind/kernels/gemm/gemm.cu`
- `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu`
- `lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`

If additional `lmdeploy/` files are found to require change during replay, they
must be justified as part of the same kernel-support responsibility before being
added to the local commit.

## Validation Plan

Validation is staged so failures are attributable.

### Baseline Validation

Before semantic upgrade work, confirm the isolated worktree can at least reach a
known baseline environment.

Observed current gap:

- The current `gptq` conda environment does not provide a complete test
  baseline for this repository. Lightweight pytest startup already fails during
  `tests/conftest.py` import because `tblib` is missing. Earlier runs outside
  that environment also failed because `torch` was not installed.

This gap does not block writing the design or implementation plan, but it must
be treated as an environment prerequisite before any passing-test claim is made.

### Phase 1 Validation

- Tree-diff the reconstructed baseline against current `main`, allowing only the
  expected structural differences introduced by subtree metadata and the new
  support commit.
- Run a minimal import/build smoke test to ensure the repository still imports
  and configures after history cleanup.
- If the environment is repaired, rerun a lightweight pytest subset to validate
  that the cleanup phase did not change behavior.

### Phase 2 Validation

After the root upgrade to `v0.19.0`:

- Verify Python package import paths still work.
- Verify build configuration still discovers and wires `lmdeploy` correctly.
- Verify Docker and documented entrypoints still map to the supported public
  install and serving paths.
- Run at least one source-build or wheel-build smoke path.
- Run a non-GPU or low-dependency pytest subset where possible.
- If GPU access and dependencies are available, run an SM70-focused smoke test
  that exercises the intended Qwen3.5 / AWQ / attention path at least once.

If a GPU smoke test cannot be executed in the available environment, the branch
must record that as a verification gap instead of claiming full runtime success.

## Risks And Mitigations

### Risk: The v0.15.0 To v0.19.0 Jump Removes Old Integration Points

Mitigation:

- Replay the local vLLM patch semantically instead of preserving dead code.
- Keep all upgrade-only compatibility rewrites inside `1cat(vllm)`.

### Risk: Cleanup Work Accidentally Changes Behavior Before The Upgrade

Mitigation:

- Separate the cleanup phase from the upgrade phase.
- Require tree-level comparison against current `main` before accepting the new
  clean baseline.

### Risk: lmdeploy Upgrade Pressure Expands Scope

Mitigation:

- Lock `lmdeploy` to subtree `v0.12.1` for this cycle.
- Only touch `lmdeploy/` beyond the known kernel delta if the root upgrade
  forces interface compatibility work.

### Risk: Environment Gaps Hide Regressions

Mitigation:

- Record missing dependencies explicitly.
- Do not claim passing verification until the environment can run the selected
  validation commands successfully.

## Deliverables

The finished branch must provide:

- A new isolated upgrade branch developed in project-local `.worktrees/`.
- A repository root aligned to upstream `vLLM v0.19.0`.
- `lmdeploy/` managed as subtree content while keeping the same external path.
- Two isolated local customization commits for future replay.
- Updated docs and build/release assets that continue to describe the public
  `1Cat-vLLM-0.0.2` style SM70 runtime story.
- A validation record that clearly distinguishes completed checks from remaining
  environment-limited gaps.

## Follow-Up Implementation Constraint

The implementation plan must preserve the two local customization commits as
named deliverables. Any task decomposition that would split those local changes
across multiple final commits is invalid for this project, even if the work is
performed in smaller intermediate steps during development.
