# 1Cat-vLLM v0.19.1 + Transformers 5.5.4 升级设计

日期：2026-04-20

## 1. 背景与目标

当前 `feature/vllm-0190-upstream-split` worktree 已经在 `vLLM 0.19.0` 基线上完成了一批本地化改造，重点包括：

- `SM70 / V100` attention 后端适配
- `Gemma4` 在 `SM70` 上的真实加载与推理打通
- `compressed-tensors / AWQ` 的 `SM70` 路径接入
- 面向 V100 用户的安装与升级文档整理

与此同时，上游 `/mnt/data/apps/vllm` 已经来到 `v0.19.1`，其依赖基线也已经放宽到 `transformers >= 4.56.0, != 5.0.*, != 5.1.*, != 5.2.*, != 5.3.*, != 5.4.*, != 5.5.0`，测试锁版本为 `5.5.3`。

本次工作的目标是：

1. 将当前 worktree 的 `vllm` 基线从 `0.19.0` 升级到上游 `0.19.1`
2. 在此基础上，将 `transformers` 提升到 `5.5.4`
3. 保留并校正当前分支已经实现的 `SM70 / Gemma4 / FA2 hdim512 / AWQ` 路径
4. 让 `Gemma4` 的 tokenizer / processor / serve 链路在 `transformers 5.5.4` 下重新进入可验证状态

本次升级的重点不是“完全重新做一条新分支”，而是基于当前已经跑通的 `SM70` 实现，吸收上游 `0.19.1` 的基础升级，并把 `transformers` 切到一个更适合 `Gemma4` 的版本。

## 2. 非目标

本次设计明确不做以下事情：

- 不升级 `lmdeploy` 的 vendored 基线
- 不重构现有 `SM70` 实现的总体架构
- 不把本地 `SM70` 改造重写成另一套全新后端体系
- 不尝试一次性对齐上游所有最新提交，只以 `v0.19.1` 已提交内容为基线
- 不把本次升级扩展为新的模型接入需求或新的量化格式支持

## 3. 当前基线与约束

### 3.1 当前 worktree 基线

当前 worktree 的关键事实如下：

- `setup.py` 默认发布版本仍是 `0.19.0`
- `requirements/common.txt` 仍限制为 `transformers >= 4.56.0, < 5`
- `requirements/test.in` / `requirements/test.txt` 仍锁在 `transformers 4.57.5`
- 当前本地环境已经在 `4.57.6` 下暴露出 `Gemma4` tokenizer 的兼容性问题

### 3.2 上游参考基线

上游仓库路径为 `/mnt/data/apps/vllm`，参考基线是 `v0.19.1` tag，而不是其工作区中的未提交状态。

特别说明：

- `/mnt/data/apps/vllm` 当前存在未跟踪文件 `vllm/v1/attention/backends/volta_fa2.py`
- 该文件不属于本次“上游基线”的可信输入
- 本次同步只以 `v0.19.1` 已提交内容为准

### 3.3 现有本地改造的重要性

当前分支已经有一批与 `SM70` 强绑定的本地改造，至少包含：

- `CUDA / attention backend` 的 `SM70` 能力放宽与优先级调整
- 与 `flash-attention-v100` 的集成
- `SM70` 上 `dense AWQ` 的 warmup 和 fallback 逻辑
- `Gemma4` 的 `head_dim=512` attention 实测打通
- 对应的 `SM70` 文档与测试补充

这些改造不是“可随意覆盖的临时 patch”，而是本次升级必须保护的行为面。

## 4. 候选方案

### 方案 A：全量回到上游 `v0.19.1`，然后重做全部 `SM70` 改造

优点：

- 与上游形态最接近
- 历史包袱最少

缺点：

- 风险最高
- 现有已跑通的 `Gemma4 SM70` 路径容易被打散
- 回归成本大，不适合当前“先稳住可用链路”的目标

### 方案 B：只改依赖版本，按报错逐步打补丁

优点：

- 表面改动最小
- 起步快

缺点：

- 容易遗留隐性不一致
- 很难说明哪些行为是来自 `0.19.1`，哪些只是临时兼容
- 后续维护成本高

### 方案 C：以当前 worktree 为主体，定向同步 `v0.19.1` 的基础升级，再保留并校正现有 `SM70` 实现

优点：

- 最大限度复用当前已经验证过的 `SM70` 工作成果
- 能把升级范围控制在“版本基线 + HF 兼容 + 现有本地行为回归”
- 更符合当前分支已经承担的产品化责任

缺点：

- 需要逐项判断哪些上游改动该吸收，哪些本地改动必须保留
- 需要做一轮针对 `transformers 5.5.4` 的兼容回归

## 5. 推荐方案

推荐采用方案 C。

核心策略如下：

1. 以当前 `feature/vllm-0190-upstream-split` 为实现主体
2. 将基础版本号、requirements 和必要的 `transformers / huggingface_hub` 兼容逻辑，同步到上游 `v0.19.1`
3. 将测试锁版本从上游 `5.5.3` 再进一步提升到 `5.5.4`
4. 保留当前本地 `SM70` 相关实现，并针对新版依赖做定向校正
5. 用真实 `Gemma4` 模型重新验证 `serve` 和 `smoke` 路径，确认升级不是“仅能 import”

## 6. 详细设计

### 6.1 升级边界

本次升级分为三层：

#### 第一层：基础版本与依赖基线

目标文件预计包括：

- `setup.py`
- `requirements/common.txt`
- `requirements/test.in`
- `requirements/test.txt`

预期动作：

1. 将默认发布版本从 `0.19.0` 提升到 `0.19.1`
2. 将主依赖约束调整到与上游 `v0.19.1` 一致
3. 将测试锁版本从当前 `4.57.5` 提升到 `5.5.4`
4. 评估并按需同步 `tokenizers`、`huggingface_hub` 及相关测试依赖版本

#### 第二层：`transformers 5.5.4` / `huggingface_hub` 兼容

目标不是全仓“为了 v5 而重构”，而是只同步和本次升级强相关的兼容路径。

关注范围包括：

- tokenizer / processor 初始化路径
- `AutoTokenizer` / `AutoProcessor` 加载
- `huggingface_hub 1.x` 相关 API 和异常类型
- `HF_TRANSFER`、缓存变量和下载路径相关逻辑
- 测试里对 `special tokens` 两套语义的兼容判断

这部分的原则是：

- 优先吸收上游 `v0.19.1` 已有实现
- 只有当本地 `SM70` 路径被上游实现直接冲突时，才在本分支做最小修正

#### 第三层：保留并校正本地 `SM70` 行为

保留范围包括但不限于：

- `vllm/platforms/cuda.py`
- `SM70` attention backend 选择逻辑
- `flash-attention-v100` 接入路径
- `compressed-tensors` 下的 `SM70 AWQ` 路径
- `Gemma4` 的 `hdim512` attention 可运行性
- 已新增的 `SM70` 文档与测试

这层的原则是：

- 保持当前已经验证过的行为优先
- 不用“更像上游”去覆盖“已经证实对 V100 有效”的本地逻辑

### 6.2 实施顺序

推荐按以下顺序实施：

1. 先同步版本号与 requirements
2. 再同步 `transformers 5.x` / `huggingface_hub` 相关兼容代码
3. 然后修正因升级引发的本地 `SM70` 冲突
4. 最后做测试与真实模型回归

顺序原因：

- 先把依赖基线对齐，后续错误才具备可解释性
- 先处理 HF 兼容，再看 `Gemma4` tokenizer / processor 是否自然恢复
- 将 `SM70` 修复放在后面，便于区分“依赖升级问题”和“本地后端问题”

### 6.3 `Gemma4` 特别关注点

本次升级中，`Gemma4` 是必须显式验证的重点模型。

原因有三点：

1. 模型 `config.json` 声明的 `transformers_version` 已经是 `5.5.0.dev0`
2. 当前 `4.57.6` 下的 tokenizer / processor 链路存在已知兼容性问题
3. 当前分支已经在 `SM70 + Gemma4 + FA2 hdim512 + compressed-tensors` 组合上完成过真实 smoke

因此，`Gemma4` 在本次升级中承担双重职责：

- 既是 `SM70` 推理回归对象
- 也是 `transformers 5.5.4` 兼容性回归对象

### 6.4 风险控制

本次升级的主要风险如下：

1. 上游 `0.19.1` 的基础变更与本地 `SM70` 改造发生冲突
2. `huggingface_hub 1.x` 带来的异常类型或环境变量变化影响下载逻辑
3. `transformers 5.5.4` 引入的 tokenizer / processor 语义变化影响现有 serve 路径
4. 测试锁版本变化引发与 `4.57.x` 不同的行为差异

控制策略：

1. 以当前 worktree 为主体，不做全量覆盖式升级
2. 所有依赖相关改动先与上游 `v0.19.1` 对齐，再做最小本地增量
3. 用已有 `SM70` 测试和真实 `Gemma4` smoke 作为硬回归门槛
4. 将 `serve` 验证纳入验收，而不是只做离线 `LLM` smoke

## 7. 验证口径

本次升级必须至少通过以下四类验证。

### 7.1 依赖与基础导入验证

最小要求：

- `import vllm`
- `import transformers`
- `AutoTokenizer.from_pretrained(...)`
- `AutoProcessor.from_pretrained(...)`

目标不是覆盖所有模型，而是确认本次升级后的基础依赖组合可工作。

### 7.2 `SM70` attention 测试

至少重跑：

- `tests/v1/attention/test_flash_attn_sm70.py`

目标是确认：

- `SM70` 仍能正确选择本地预期的 attention backend
- `FA2 hdim512` 相关路径没有因依赖升级失效

### 7.3 真实 `Gemma4` 2-GPU smoke

验证对象：

- `/mnt/data6/models/gemma-4-31B-it-AWQ-4bit`

最小要求：

- 2 张 `SM70 / V100` GPU
- `tensor_parallel_size=2`
- `compressed-tensors` 真实加载
- 至少一次真实生成

目标是确认：

- 模型仍能完成加载、warmup、attention、decode 全链路
- 当前本地 `SM70` dense AWQ 和 `hdim512 FA2` 路径未被破坏

### 7.4 `serve / OpenAI API` 验证

最小要求：

- 启动 API server
- 用真实请求完成一次文本生成
- 验证 `finish_reason`
- 验证服务端 detokenize / tokenizer 可用性

对于 `Gemma4`，这里尤其重要，因为它直接回答本次升级是否解决了当前 tokenizer 不可用的问题。

## 8. 成功标准

本次升级完成后，应同时满足以下条件：

1. 仓库基础版本为 `0.19.1`
2. `transformers` 依赖和测试锁版本为 `5.5.4`
3. 当前本地 `SM70` attention / AWQ / Gemma4 路径继续可用
4. `Gemma4` 的 tokenizer / processor / serve 链路至少恢复到可验证状态
5. 现有面向 `SM70` 的文档与测试仍与实际实现一致

## 9. 实施边界

这次工作应作为一次“基础升级 + 本地能力回归”的集中整理，不应顺手混入以下内容：

- 新模型接入
- 新量化格式适配
- 与 `lmdeploy` 无关的额外内核实验
- 无关的文档重写
- 与 `SM70 / transformers 5.5.4 / v0.19.1` 无关的仓库整理

最终产物应是一个边界清楚、可验证、后续便于继续迭代的 `v0.19.1 + transformers 5.5.4 + SM70` 工作基线。
