# v100llm (vLLM 0.19.0 for V100/SM70)

`v100llm` 是一个面向 `Tesla V100 / SM70` 的 `vLLM 0.19.0` 定向分支。这个仓库的重点不是重新定义
`vllm` 的 Python API，而是在保持现有 `vllm` 包和 CLI 习惯不变的前提下，恢复并验证 V100 上真实需要的
推理路径，包括 `FLASH_ATTN`、`AWQ`、`compressed-tensors MoE` 和 MiniMax/Qwen3.5 相关运行时链路。

`v100llm` is a `vLLM 0.19.0` fork focused on `Tesla V100 / SM70`. The repository keeps the existing
`vllm` package and CLI interface for compatibility, while restoring and validating the inference paths
that matter on V100, including `FLASH_ATTN`, `AWQ`, `compressed-tensors MoE`, and the runtime paths
used by recent Qwen3.5 and MiniMax models.

## Highlights / 项目亮点

- 基于 upstream `vLLM 0.19.0`，但面向 `V100 / SM70` 做了运行时恢复与兼容性补充。
- 仓库名称是 `v100llm`，但安装后的 Python 包名和 CLI 仍然是 `vllm`，便于兼容现有脚本与调用方式。
- 当前默认对外打包版本固定为 `0.19.0`；`python setup.py --version`、`vllm.__version__` 和 CLI `--version` 已对齐。
- 已在实际 V100 环境验证 `Qwen3.5-27B-AWQ`、`Qwen3.5-122B-A10B-AWQ-4bit`、`MiniMax-M2.5-AWQ`、
  `MiniMax-M2.7-AWQ-4bit` 的关键推理路径。

## Quick Start / 快速开始

当前推荐路径是源码安装，优先面向需要直接在 `V100 / SM70` 上构建和运行这个仓库的工程用户。

```bash
conda create -n v100llm python=3.13 -y
conda activate v100llm

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Recommended on V100 when you have a local flash-attention-v100 source tree.
export VLLM_FLASH_ATTN_SRC_DIR=/path/to/flash-attention-v100

python -m pip install -e . --no-build-isolation
python -m vllm.entrypoints.cli.main --version
```

期望版本输出：

```text
0.19.0
```

Additional notes:

- 如果你的机器是混合 GPU 环境，建议显式设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- 这个仓库名叫 `v100llm`，但安装后的包名和 CLI 仍然是 `vllm`。
- 更细的 `flash-attention-v100` 安装与升级说明见：
  [`docs/open_source_sm70_flash_attn_install_upgrade_zh.md`](docs/open_source_sm70_flash_attn_install_upgrade_zh.md)

## Validated Models and Paths / 已验证模型与路径

以下内容来自当前 worktree 的实测验证，不是理论支持列表。

| Model | Hardware | Validated Path | Result |
| --- | --- | --- | --- |
| `Qwen3.5-27B-AWQ` | `4x V100` | `AWQ` + `AsyncLLM` + serve benchmark | smoke 通过，`1k/32k` serve benchmark 已记录 |
| `Qwen3.5-122B-A10B-AWQ-4bit` | `4x V100` | `compressed-tensors` 自动识别 + `SM70` TurboMind MoE | 真实生成通过 |
| `MiniMax-M2.5-AWQ` | `8x V100` | `AWQ` + `SM70` MoE warmup | AsyncLLM smoke 通过 |
| `MiniMax-M2.7-AWQ-4bit` | `8x V100` | `compressed-tensors -> AWQ` on `SM70` | AsyncLLM smoke 通过 |

详细命令、日志和验证记录见：

- [`docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`](docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md)

## What Changed vs Upstream vLLM / 与上游差异

这个仓库仍然以 upstream `vLLM 0.19.0` 为基础，但当前分支额外收敛了几类与 `V100 / SM70` 直接相关的内容：

- `SM70` 上的 attention/backend 选择与 `FLASH_ATTN` 路径恢复
- `AWQ` 和 `compressed-tensors MoE` 在 `SM70` 上的运行时兼容
- `fused_moe`、量化辅助函数、`_C` 绑定与入口契约恢复
- `Qwen3.5` 和 `MiniMax` 相关真实推理链路打通

这些差异的目标是让当前分支像组合分支一样能在 V100 上完成实际推理，而不是通过绕开原有模型路径来“跑通一次”。

## Current Status and Limits / 当前状态与限制

- 这是一个偏 `Tesla V100 / SM70` 的定向 fork，不承诺对所有 GPU 平台都有同样的优化效果。
- 当前 README 首页只保留已验证的模型和路径，不包含尚未复核的 benchmark 或通用性能结论。
- 某些环境下 CLI 或启动日志里仍可能出现与混合 GPU、Triton 可选内核相关的 warning；这不等同于功能失效。
- 与根分支的完整 prefill/decode 图表对齐尚未在当前验证周期内全部补齐。

## Docs and Verification / 文档与验证

- V100/SM70 的 `flash-attention-v100` 安装与升级说明：
  [`docs/open_source_sm70_flash_attn_install_upgrade_zh.md`](docs/open_source_sm70_flash_attn_install_upgrade_zh.md)
- 当前运行时恢复验证记录：
  [`docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`](docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md)
- 文档站入口：
  [`docs/README.md`](docs/README.md)

## Build and Release Notes / 构建与发布说明

- 当前默认对外打包版本固定为 `0.19.0`
- 仓库品牌名为 `v100llm`
- 安装后的包名和 CLI 仍为 `vllm`
- 常用检查命令：

```bash
python setup.py --version
python -c "import vllm; print(vllm.__version__)"
python -m vllm.entrypoints.cli.main --version
```

以上三条当前都应输出：

```text
0.19.0
```

## License / 许可证

本仓库沿用 upstream `vLLM` 的许可证体系，详见 [`LICENSE`](LICENSE)。

## Acknowledgements / 致谢

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
