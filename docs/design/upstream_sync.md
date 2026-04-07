# Upstream Sync Workflow

## Current Baselines

- Root upstream baseline: `vLLM v0.19.0`
- Vendored subtree baseline: `lmdeploy v0.12.1`
- Local replay commits:
  - `1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations`
  - `1cat(lmdeploy): turbomind SM70 kernel customizations`

## Local Mirrors Used In This Workspace

- `vllm-upstream` -> `/mnt/data/apps/vllm`
- `lmdeploy-upstream` -> `/mnt/data/apps/lmdeploy`

## Replay Commits

- `1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations`
- `1cat(lmdeploy): turbomind SM70 kernel customizations`

## Future Upgrade Commands

```bash
git fetch vllm-upstream --tags
git subtree pull --prefix=lmdeploy /mnt/data/apps/lmdeploy v0.12.1 --squash
git diff final/1cat-vllm^ final/1cat-vllm -- \
  . ':(exclude)lmdeploy' | git apply --3way
```

Use the subtree command only for `lmdeploy/`. Use the root import plus replay
flow only for `vLLM` upgrades.

For `lmdeploy` upgrades, do not fetch its tags into the `1Cat-vLLM` repository,
because `vLLM` and `lmdeploy` reuse tag names like `v0.10.0` and `v0.12.0`.
Operate against the on-disk mirror at `/mnt/data/apps/lmdeploy` directly
instead. If that mirror needs refreshing, do it outside this repository
workflow.
