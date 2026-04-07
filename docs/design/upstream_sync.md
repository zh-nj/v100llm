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
single `1cat(vllm)` commit with a
`git diff phase1/1cat-vllm^ phase1/1cat-vllm | git apply --3way` style flow.

For `lmdeploy` upgrades, do not fetch its tags into the `1Cat-vLLM` repository,
because `vLLM` and `lmdeploy` reuse tag names like `v0.10.0` and `v0.12.0`.
Operate against the on-disk mirror at `/mnt/data/apps/lmdeploy` directly
instead. If that mirror needs refreshing, do it outside this repository
workflow.
