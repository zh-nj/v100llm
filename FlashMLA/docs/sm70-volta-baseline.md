# FlashMLA SM70 Baseline

Date: 2026-04-28

Branch: `feature/sm70-volta-flashmla`

Base commit: `a6ec2ba`

Workspace status at start:

```text
?? .codex
?? docs/sm70-volta-flashmla-design.md
?? docs/sm70-volta-flashmla-tasks.md
```

Hardware observed with `nvidia-smi -L`:

- GPU 0-5,7,9: `Tesla PG503-216`
- GPU 6,8: `Tesla V100-SXM2-32GB`
- GPU 10: `NVIDIA GeForce RTX 3080 Ti Laptop GPU`

CUDA compiler:

```text
nvcc release 12.8, V12.8.93
```

Python environments checked:

```text
system python: torch not installed
/home/z/anaconda3/envs/gptq/bin/python: torch 2.10.0+cu128, CUDA 12.8, torch.cuda.is_available() == False
/home/z/anaconda3/envs/lmdeploy/bin/python: torch 2.8.0+cu128, CUDA 12.8, torch.cuda.is_available() == False
```

Runtime recheck after SM70 build work:

```text
/home/z/anaconda3/envs/gptq/bin/python: torch 2.10.0+cu128, CUDA 12.8, torch.cuda.is_available() == True, device_count == 11
CUDA_VISIBLE_DEVICES=6 maps cuda:0 to Tesla V100-SXM2-32GB / SM70
```

Verification note:

- Build/static tests can run in the current shell.
- CUDA runtime tests now run in the `gptq` environment when a V100 is selected with `CUDA_VISIBLE_DEVICES=6` or `8`.
- Dense decode SM90 baseline tests are not meaningful on the V100 target; SM90/SM100 regression should be run on matching hardware.

SM90/SM100 regression command baseline:

```bash
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_decoding.py
python tests/test_flash_mla_sparse_prefill.py
python tests/test_fmha_sm100.py
```
