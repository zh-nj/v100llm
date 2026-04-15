# v100llm README Refresh Design

日期：2026-04-15

## 1. 背景

当前仓库已经从旧的 `1Cat-vLLM-0.0.2` 发布状态推进到了基于 upstream
`vLLM 0.19.0` 的 `V100 / SM70` 定向分支，并且已经完成了关键运行态恢复与
实机验证：

- `Qwen3.5-27B-AWQ`
- `Qwen3.5-122B-A10B-AWQ-4bit`
- `MiniMax-M2.5-AWQ`
- `MiniMax-M2.7-AWQ-4bit`

旧版 README 属于 `0.0.2` 时代的发布文案，混合了旧版本号、旧发布素材、旧安装
口径和当时的对外叙述方式，已经不适合作为新的 GitHub 项目首页。

当前仓库又已经删除了旧 README，因此需要补一份新的仓库首页文档，既服务首次
访问 GitHub 的外部用户，也为当前工程用户提供足够清晰的安装、验证和文档入口。

## 2. 目标

新的 `README.md` 需要同时满足以下目标：

1. 明确项目定位为：
   - `v100llm (vLLM 0.19.0 for V100/SM70)`
2. 以中英双语方式提供首页说明，但不做逐段完全镜像式翻译堆叠。
3. 优先服务外部首次访问用户，让对方能快速理解：
   - 这是什么项目
   - 与 upstream `vLLM 0.19.0` 的关系
   - 适用硬件与主要使用场景
4. 保留工程用户需要的关键入口：
   - 源码安装方式
   - 已验证模型与路径
   - 验证记录和补充文档位置
5. 与当前仓库真实状态保持一致，不继续沿用已失效的 `0.0.2`、旧图片、旧发布
   文案或未经验证的性能承诺。

## 3. 非目标

本次 README 刷新明确不做以下事情：

- 不重写整个文档站
- 不新增新的模型支持承诺
- 不编造尚未验证的 benchmark 数字
- 不把 README 写成完整安装手册，详细信息仍应留在 `docs/`
- 不修改 Python 包名、模块名或 CLI 名称
- 不在 README 中引入与当前仓库状态不一致的 GitHub 发布流程描述

## 4. 目标读者

README 需要同时覆盖两类读者，但优先级不同：

### 4.1 一级读者：外部首次访问用户

他们需要在 GitHub 首页快速获得以下信息：

- 项目名称与定位
- 这个项目是否适合自己的硬件
- 最快的安装方式是什么
- 当前已经验证过哪些模型和路径

### 4.2 二级读者：工程使用者与维护者

他们还需要：

- 确认项目与 upstream 的差异点
- 找到验证记录
- 找到更细的安装与设计文档

因此 README 结构应该先回答“这是什么”和“怎么开始”，再给工程细节入口。

## 5. 推荐结构

README 采用“发布型首页 + 工程验证入口”的结构。

建议章节如下：

1. 标题与一句话定位
2. 中文简介
3. English summary
4. Highlights / 项目亮点
5. Quick Start（源码安装优先）
6. Validated Models and Paths
7. What Changed vs Upstream vLLM
8. Current Status and Limits
9. Docs and Verification
10. Build and Release Notes
11. License / Acknowledgements（简短）

## 6. 文案策略

### 6.1 标题

README 顶部标题固定为：

`v100llm (vLLM 0.19.0 for V100/SM70)`

### 6.2 双语策略

采用“中文优先一段 + 英文总结一段”的方式，而不是每个小节整段重复双语。

原因：

- GitHub 首页可读性更好
- 不会让 README 变成过长的镜像文本
- 中文用户仍能直接获得完整信息
- 英文用户也能在首页快速理解项目定位和关键入口

### 6.3 语气

README 应保持以下语气：

- 直接
- 工程化
- 不夸大
- 不写营销式承诺

避免：

- “最强”“领先”“极致”之类不可验证措辞
- 未经当前分支验证的数据结论
- 针对未来发布的猜测

## 7. Quick Start 设计

README 的默认 quick start 采用源码安装路径，而不是 wheel-first 路径。

推荐展示：

1. 创建并激活环境
2. 安装 PyTorch/CUDA 依赖的最小前提
3. 执行：

```bash
python -m pip install -e . --no-build-isolation
```

4. 最小版本检查：

```bash
python -m vllm.entrypoints.cli.main --version
```

需要明确：

- 当前默认对外打包版本为 `0.19.0`
- 这个仓库仍然主要面向 `V100 / SM70` 使用场景
- 实际 GPU 运行时建议显式设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`

README 中不需要把所有系统依赖和 FlashAttention 构建细节全部展开，可把深度说明
链接到现有文档：

- `docs/open_source_sm70_flash_attn_install_upgrade_zh.md`

## 8. 已验证内容设计

README 中的“Validated Models and Paths”只列当前已有证据支持的内容：

- Qwen3.5-27B-AWQ
- Qwen3.5-122B-A10B-AWQ-4bit
- MiniMax-M2.5-AWQ
- MiniMax-M2.7-AWQ-4bit

并简要说明：

- `Qwen27`：AsyncLLM smoke + serve benchmark
- `Qwen122`：compressed-tensors 自动识别路径验证
- `MiniMax M2.5`：8-GPU AWQ smoke
- `MiniMax M2.7`：8-GPU compressed-tensors -> AWQ 路径 smoke

详细日志和命令不应堆在 README 首页，统一链接到：

- `docs/upstream-sync/verification-2026-04-13-v0190-worktree-runtime-recovery.md`

## 9. Upstream 差异说明

README 中应有一个简短章节解释这个仓库与 upstream `vLLM 0.19.0` 的关系。

建议只保留高层描述：

- 基于 upstream `vLLM 0.19.0`
- 面向 `V100 / SM70` 做了运行时恢复和兼容性补充
- 覆盖 `AWQ`、`compressed-tensors MoE`、`FlashAttention on V100` 等关键路径

不在 README 首页写入过多实现细节；实现细节继续留在验证文档和计划文档中。

## 10. 风险与限制

README 需要主动写明以下限制：

- 这是一个偏 `V100 / SM70` 的定向 fork
- 不保证对所有 GPU 平台提供同等优化效果
- 某些 CLI 启动日志中仍可能出现与当前环境相关的 warning，但不等同于功能失效
- 更细的性能对比与完整 benchmark 数据目前不在 README 首页展开

## 11. 验收口径

新的 `README.md` 完成后，应满足以下验收标准：

1. `pyproject.toml` 的 `readme = "README.md"` 再次有实际文件可用
2. GitHub 首页可在第一页回答：
   - 项目是什么
   - 针对什么硬件
   - 怎么安装
   - 看哪里了解验证细节
3. 文案不再引用旧版 `0.0.2` 发布内容
4. 文案不再引用已删除的图片或失效素材
5. README 与当前版本口径一致：
   - 对外打包版本 `0.19.0`
   - 项目定位 `v100llm (vLLM 0.19.0 for V100/SM70)`
