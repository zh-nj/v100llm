# feature/vllm-0190-upstream-split 运行态恢复与可用化设计

日期：2026-04-13

## 1. 背景与目标

本设计只覆盖 `.worktrees/vllm-0190-upstream-split` 这条升级分支的“运行态恢复”问题，不讨论重新拆分历史、重建新 worktree，或把根仓库整体替换成该分支。

当前事实有三条：

1. `feature/vllm-0190-upstream-split` 的历史目标是把仓库根从旧 `vLLM` 基线升级到 `v0.19.0`，同时保留 1Cat 的 `SM70 / Qwen3.5 / AWQ / MiniMax / lmdeploy` 定制能力。
2. 根仓库 `/mnt/data/apps/1Cat-vLLM` 上的 `chore/ignore-project-worktrees@3e0759231d9cef70e0dc53fe1b8e57ac70c83699` 已经能在 `conda gptq` 下真实运行 `Qwen3.5-27B-AWQ`、`Qwen3.5-122B-A10B-AWQ-4bit`、`MiniMax-M2.5-AWQ` 推理与 benchmark。
3. `feature/vllm-0190-upstream-split` 目前仍处于“安装可部分完成，但运行态不自洽”的状态，必须借助根仓库 `PYTHONPATH=/mnt/data/apps/1Cat-vLLM` 才能绕过问题，不满足目标口径。

本次工作的唯一目标是：

- 把 `feature/vllm-0190-upstream-split` 推进到“自己可 build / install / import / inference”的可用状态；
- 运行时不再依赖根仓库源码树逃逸；
- 保持其“基于 upstream `vLLM v0.19.0` 的升级分支”定位不变。

## 2. 验收口径

本次“可用”按强口径验收，必须同时满足以下条件：

1. 在 `.worktrees/vllm-0190-upstream-split` 中，使用 `conda gptq` 能完成可重复的 build / install。
2. 不设置 `PYTHONPATH=/mnt/data/apps/1Cat-vLLM`，直接从该 worktree 的源码或安装产物完成：
   - `import vllm`
   - `import vllm._C`
   - CUDA 平台探测与 `AsyncEngineArgs.create_engine_config()` 正常
3. 至少完成一条真实 `AsyncLLM` 推理 smoke。
4. 最好复现此前 benchmark 口径：
   - `Qwen3.5-27B-AWQ`
   - `Qwen3.5-122B-A10B-AWQ-4bit`
   - `MiniMax-M2.5-AWQ`
   - 输出 `1k / 32k` 的 `prefill tokens/s`、`decode tokens/s`、`TTFT`、`finish_reason`、语义质量结论
5. 如果时间或初始化成本导致三组模型无法全部完成，最低要求是：
   - `Qwen3.5-27B-AWQ`
   - `MiniMax-M2.5-AWQ`
   两条代表路径成功跑通，并明确记录其余验证缺口。

## 3. 当前状态与根因判断

### 3.1 分支关系

`feature/vllm-0190-upstream-split` 不是一条空壳分支。它已经包含：

- `upstream(vllm): import v0.19.0 root baseline`
- `upstream(lmdeploy): add v0.12.1 subtree`
- `1cat(vllm): SM70/Qwen3.5/AWQ/runtime/build customizations`
- `1cat(lmdeploy): turbomind SM70 kernel customizations`
- MiniMax 相关多轮 SM70 / AWQ / compressed-tensors MoE 适配

同时，根仓库分支 `chore/ignore-project-worktrees` 还额外包含一个对运行态至关重要的提交：

- `3e0759231 Fix SM70 compressed-tensors MoE AWQ compatibility`

换句话说，`feature/vllm-0190-upstream-split` 并不是“没有实现”，而是“升级线和可运行线已经发生分叉，而且分叉后没有重新收敛到一个自洽的运行态”。

### 3.2 观察到的真实失效模式

实际排查表明，这条 worktree 当前不是单一 build 错误，而是“新旧接口混搭”导致的系统性失配。已观察到的断点包括：

- `arg_utils.py` 依赖 `PerformanceMode`、`kernel_config`、`reasoning_config` 等新字段，但 `config/vllm.py` 没有完整对应实现。
- `envs.py` 缺少 `validate_environ()` 等运行时入口。
- `platforms/__init__.py` 中的 CUDA 平台探测被本地兼容修改破坏，出现 `UnboundLocalError` 并退化成 `UnspecifiedPlatform`。
- `ModelConfig`、`batch_invariant`、`platform_utils`、`fused_moe.config` 等模块缺失根仓库当前运行态已依赖的符号。
- 这些问题串联起来后，导致 `AsyncEngineArgs.create_engine_config()`、模型 registry inspect、MoE 路由初始化等多个阶段连续失败。

因此，本问题的本质不是“某个符号没定义”，而是：

- `feature/vllm-0190-upstream-split` 上的运行时接口层处于半迁移状态；
- 根仓库可运行实现中的一部分修复还未被该 worktree 吸收；
- 局部试探性补洞已经证明，继续随机修补会不断遇到下一处断点，缺乏闭环。

## 4. 非目标

本次设计明确不做以下事情：

- 不重新设计 `1Cat-vLLM` 的仓库结构。
- 不把根仓库整体 fast-forward 成该 worktree。
- 不升级 `lmdeploy` 到 `v0.12.1` 之外的新版本。
- 不新增与当前目标无关的模型能力。
- 不借“推进到可用”为名，对整个 `v0.19.0` 代码树做无边界重构。

本次只关心让现有升级分支恢复成一个可安装、可导入、可推理、可 benchmark 的运行态。

## 5. 候选方案

### 方案 A：在当前 worktree 上继续逐个补洞

做法：

- 保持当前分支不变；
- 按遇到的报错顺序逐个补 `config/envs/platform/model_executor/fused_moe` 缺口。

优点：

- 改动看起来最小；
- 不需要主动分析两条分支的系统差异。

缺点：

- 已证实失败风险高；
- 每修一处都会暴露新的跨模块失配；
- 很容易得到一组“刚好跑过当前命令、但整体仍不自洽”的脆弱补丁。

### 方案 B：以当前 worktree 为主体，用根仓库可运行实现做对照移植

做法：

- 继续以 `feature/vllm-0190-upstream-split` 为目标分支；
- 以根仓库 `chore/ignore-project-worktrees@3e0759231` 为运行态 oracle；
- 系统性对照两条线在 `SM70/AWQ/FLASH_ATTN_V100/compressed-tensors MoE` 相关实现上的差异；
- 将已验证有效的行为迁移回 worktree，并按 `v0.19.0` 当前接口重新落位。

优点：

- 既保住 `v0.19.0` 升级分支语义，又利用了现成可运行实现；
- 目标明确，不是盲修；
- 最符合本次“推进到可用”的实际需求。

缺点：

- 需要做一轮系统性接口收敛，而不是单点热修；
- 实施成本高于方案 A。

### 方案 C：丢弃当前运行态，重新从 upstream `v0.19.0` 重放所有 1Cat 改动

做法：

- 另起一条新分支；
- 重新从 upstream `v0.19.0` 开始，把根仓库可运行实现和当前 feature 上的 MiniMax / SM70 改动全部重放一遍。

优点：

- 历史最干净；
- 最容易得到理想化的补丁层次。

缺点：

- 成本最高；
- 极易重复劳动；
- 对当前“尽快把 worktree 推进到可用”的目标不经济。

## 6. 推荐方案

推荐采用方案 B。

原因：

1. 当前 worktree 已经承载了升级线的大部分历史语义，不值得推倒重来。
2. 根仓库分支已经给出了“这台机器、这个 conda 环境、这组模型、这套 SM70 patch”下的有效运行参考。
3. 当前障碍主要集中在运行时接口和适配层，而不是模型支持从零开始缺失。
4. 用可运行实现做 oracle，可以把工作从“猜哪里坏了”变成“找哪部分还没收敛到可运行状态”。

## 7. 详细设计

### 7.1 分层推进模型

实施必须分三层推进，而且每层都要在进入下一层前完成验证。

#### 第一层：构建与导入层

目标：

- worktree 在 `conda gptq` 下能稳定 build / install；
- `vllm` 的安装产物与源码路径一致；
- `import vllm`、`import vllm._C` 成功；
- CUDA 平台识别正常，不再出现 `UnspecifiedPlatform` 或 `device_type=''`。

主要关注文件范围：

- `setup.py`
- `pyproject.toml`
- `csrc/ops.h`
- `vllm/__init__.py`
- `vllm/platforms/__init__.py`
- `vllm/platforms/cuda.py`

这一层不追求真实推理，只确保运行态入口完整。

#### 第二层：运行时接口对齐层

目标：

- `AsyncEngineArgs.create_engine_config()` 成功；
- `AsyncLLM.from_engine_args()` 能启动 engine；
- registry inspect、model config、env 校验、MoE 路由初始化不再在导入期失败。

主要关注文件范围：

- `vllm/engine/arg_utils.py`
- `vllm/config/vllm.py`
- `vllm/config/model.py`
- `vllm/envs.py`
- `vllm/model_executor/layers/batch_invariant.py`
- `vllm/utils/platform_utils.py`
- `vllm/model_executor/layers/fused_moe/**`

这一层的核心不是“把所有新接口照抄回来”，而是确保 worktree 的 `v0.19.0` 代码树形成一套自洽的运行时契约。

#### 第三层：SM70 运行态恢复层

目标：

- 恢复与根仓库当前一致的关键行为：
  - `FLASH_ATTN_V100` / `FLASH_ATTN` on SM70
  - AWQ dense fast path
  - compressed-tensors MoE AWQ 兼容路径
  - MiniMax 的 SM70 AWQ MoE 路径
- 完成至少一条真实推理 smoke；
- 以 benchmark 形式验证代表模型。

主要关注文件范围：

- `vllm/platforms/cuda.py`
- `vllm/vllm_flash_attn/**`
- `vllm/model_executor/layers/quantization/**`
- `vllm/model_executor/layers/fused_moe/**`
- `benchmarks/**`
- `tests/benchmarks/**`

这一层才是最终“可用”的决定层。

### 7.2 对照移植规则

根仓库 `chore/ignore-project-worktrees@3e0759231` 被视为“运行行为 oracle”，但不意味着直接整分支合并。迁移时必须遵守以下规则：

1. 只迁移与本次目标直接相关的行为：
   - SM70
   - AWQ
   - FLASH_ATTN_V100 / FLASH_ATTN on V100
   - compressed-tensors MoE AWQ compatibility
   - MiniMax-M2.5-AWQ
2. 每个迁移点都要按 `v0.19.0` 的当前代码结构重新落位，不能盲目覆盖 worktree 文件。
3. 如果根仓库修复依赖的是更老或更新的接口，需要先判断该行为在 `v0.19.0` 上的等价承载点，再实施改写。
4. 对照移植以“恢复行为”为目标，不以“让 diff 最小”或“保持字面一致”为目标。

### 7.3 当前实验性修改的处置原则

当前 worktree 已经存在一组未提交的实验性修改和构建产物，包括但不限于：

- `setup.py`
- `csrc/ops.h`
- `vllm/config/model.py`
- `vllm/config/vllm.py`
- `vllm/envs.py`
- `vllm/platforms/__init__.py`
- `vllm/platforms/cuda.py`
- `vllm/model_executor/layers/batch_invariant.py`
- `vllm/utils/platform_utils.py`
- `build-force/`
- `install/`

这些内容不能被默认视为最终实现，也不能在没有验证的前提下整体提交。实施时必须：

- 逐项复查其是否属于最终方案；
- 对保留项进行正式化整理与验证；
- 对无效试探性改动及时清理；
- 对生成目录与源码修复严格分离。

### 7.4 验证策略

验证必须沿实施分层同步推进。

#### 构建/导入验证

- `python -c "import vllm; print(vllm.__file__)"`
- `python -c "import vllm._C"`
- `python -c "from vllm.platforms import current_platform; print(current_platform.device_type)"`

#### 运行时接口验证

- `AsyncEngineArgs.create_engine_config()`
- `AsyncLLM.from_engine_args()` 最小实例化
- 最小 prompt 的单案生成

#### 最终运行验证

- `Qwen3.5-27B-AWQ`
- `Qwen3.5-122B-A10B-AWQ-4bit`
- `MiniMax-M2.5-AWQ`

统一输出：

- `prefill tokens/s`
- `decode tokens/s`
- `TTFT`
- `finish_reason`
- 语义质量结论

若某一模型因初始化成本过高未在本轮完成，必须在验证记录中明确写出：

- 未完成原因
- 已完成到哪一步
- 为什么不影响“最低口径”的可用结论，或为什么仍构成缺口

## 8. 风险与缓解

### 风险 1：worktree 的接口失配范围比当前观察到的更广

缓解：

- 坚持“分层推进 + 每层验证”；
- 不在未通过上一层验证前进入真实 benchmark。

### 风险 2：根仓库的可运行实现并非完全兼容 `v0.19.0`

缓解：

- 以行为迁移而不是文件覆盖为准；
- 必须在 `v0.19.0` 语义下寻找等价落点。

### 风险 3：MiniMax / compressed-tensors MoE 在 `v0.19.0` 上仍有额外性能退化

缓解：

- 把“跑通”与“性能最优”分开判断；
- 首先达到可用；
- 对缺少 tuned config 或 fallback path 的情况单独记录。

### 风险 4：构建环境污染导致“看似修好，实际依赖外部路径”

缓解：

- 最终验收时禁止使用根仓库 `PYTHONPATH`；
- 用 worktree 自身路径和安装产物重新验证 import 与 inference。

## 9. 最终交付物

本次设计对应的最终交付物应包括：

1. 一个可重复 build / install 的 `feature/vllm-0190-upstream-split` worktree。
2. 一组最小且可解释的运行态修复提交，不再依赖试探性本地改动。
3. 一份验证记录，明确：
   - build / import 结果
   - 至少一条真实推理 smoke
   - benchmark 覆盖情况
   - 未覆盖项与原因

只有当该 worktree 在不借助根仓库 `PYTHONPATH` 的条件下完成上述验证，才能被认定为“推进到可用”。
