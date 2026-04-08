# MiniMax-M2.5-AWQ 在 1Cat-vLLM 上复用 TurboMind SM70 AWQ 路径的设计

日期：2026-04-08

## 1. 背景与目标

本设计只覆盖以下范围：

- 目标模型：`/mnt/data6/models/MiniMax-M2.5-AWQ`
- 目标硬件：`SM70 / V100`
- 目标能力：文本推理主链路

明确不包含：

- tool parser
- reasoning parser
- GGUF
- 多模态
- 把 TurboMind 作为整套推理引擎接入 1Cat-vLLM

本次工作的本质不是“给 1Cat 新增 MiniMaxM2 模型支持”。`1Cat-vLLM` 已经具备 `MiniMaxM2ForCausalLM` 的模型注册、层实现、权重装载和标准推理路径。真正要补的是：

1. 把 `lmdeploy/TurboMind` 对 `MiniMax-M2.5-AWQ` 的量化语义迁移到 1Cat 现有的 vLLM 执行框架。
2. 让 `SM70` 上可量化的 MiniMax experts 复用 1Cat 已接好的 TurboMind AWQ/MoE kernel。
3. 避免把本应保留为 `fp16` 的模块错误送入 `int4` 路径。

## 2. 问题定义

### 2.1 目标模型的关键约束

`/mnt/data6/models/MiniMax-M2.5-AWQ/config.json` 表明：

- `architectures = ["MiniMaxM2ForCausalLM"]`
- `quant_method = "awq"`
- `bits = 4`
- `group_size = 128`
- `zero_point = true`
- `modules_to_not_convert = ["self_attn", "block_sparse_moe.gate", "model.layers.0."]`

这组配置的语义不是“整模型全量 AWQ”，而是：

- `self_attn` 不量化
- `block_sparse_moe.gate` 不量化
- `model.layers.0.` 整层不量化，意味着第 0 层 experts 也必须保持原始高精度
- 其余专家层可以走 AWQ int4

### 2.2 TurboMind 对 MiniMax-M2.5-AWQ 的真实处理方式

`/mnt/data/apps/lmdeploy/lmdeploy/turbomind/deploy/converter.py` 中把 `MiniMaxM2ForCausalLM` 放进 `_moe_only_quant_archs`，说明：

- attention 保持原生 dtype
- 只有 MoE experts 走量化

同一段逻辑还会根据 `modules_to_not_convert` 为每一层计算 experts 的实际权重类型。对于 `MiniMax-M2.5-AWQ` 这类模型，TurboMind 的语义是：

- 某些层 experts 为 `fp16`
- 某些层 experts 为 `int4`

也就是说，TurboMind 不是“统一给所有 experts 上 int4 kernel”，而是“按层执行 mixed expert weight type”。

### 2.3 1Cat 当前最可能的错误点

`vllm/model_executor/layers/quantization/awq.py` 当前逻辑中：

- `LinearBase` 已支持 `modules_to_not_convert`
- 但 `FusedMoE` 的 `SM70` 分支没有在选择 `AWQSM70MoEMethod` 之前先检查 skip 规则

这会导致 `MiniMax-M2.5-AWQ` 的 `model.layers.0.block_sparse_moe.experts` 存在被错误量化的风险。

因此，本设计的核心不是新增 kernel，而是修正量化决策与模块选择，使其与 TurboMind 对 MiniMax-M2.5-AWQ 的语义一致。

## 3. 现有可复用基础

### 3.1 1Cat 已有的 MiniMaxM2 推理路径

`1Cat-vLLM` 已具备：

- `MiniMaxM2ForCausalLM` 模型入口
- `MiniMaxM2Attention` 的标准 attention 路径
- `MiniMaxM2MoE` 的 `FusedMoE` 路径
- experts 权重映射与加载
- `MiniMaxText01RMSNormTP` 的 TP QK norm

这意味着本次不需要从 `lmdeploy` 复制 MiniMax 模型结构。

### 3.2 1Cat 已有的 SM70 AWQ/TurboMind 底座

`1Cat-vLLM` 已具备：

- TurboMind `SM70` GEMM 源码并入 vLLM 扩展
- `awq_gemm_sm70`
- `awq_moe_*_sm70`
- `sm70_f16_gemm`
- `AWQSM70MoEMethod`
- `SM70 AWQ` warmup 与 LUT import/export

因此 MiniMax 的正确移植方式应是：

- 继续让 vLLM 负责模型执行、调度、KV cache 与 attention backend
- 仅在需要的 MoE experts 层复用 TurboMind `SM70` int4 kernel

## 4. 设计目标

本次设计要求落地后满足以下条件：

1. `MiniMax-M2.5-AWQ` 能在 `SM70` 上完成文本推理主链路。
2. `self_attn` 继续走非量化路径。
3. `block_sparse_moe.gate` 继续走非量化路径。
4. `model.layers.0` 的 experts 保持非量化路径。
5. 其余满足 `SM70` 约束的 experts 走 `AWQSM70MoEMethod`。
6. 行为与 TurboMind 对该模型的量化语义一致。
7. 不引入 MiniMax 专属 attention custom op，不复制 TurboMind 的整套 MiniMax runtime。

## 5. 可选方案

### 方案 A：最小语义对齐方案

做法：

- 保持 `vllm/model_executor/models/minimax_m2.py` 主体不变
- 在 `AWQConfig.get_quant_method()` 中为 `FusedMoE` 补齐 `modules_to_not_convert` 跳过逻辑
- 明确支持 `model.layers.N.` 这种按层跳过的 MiniMax-M2.5-AWQ 语义
- 跳过层返回 `UnquantizedFusedMoEMethod`
- 其余层在 `SM70` 上走 `AWQSM70MoEMethod`

优点：

- 改动最小
- 与现有 vLLM 量化框架一致
- 与 `AWQMarlinConfig` 的 MoE skip 语义对齐
- 不复制 MiniMax 专属执行逻辑

缺点：

- 逻辑放在通用 AWQ 层，MiniMax 特例不够显式

### 方案 B：新增 MiniMax 专用 AWQ 规划层

做法：

- 新增 `MiniMaxM2AWQConfig` 或 `MiniMaxM2QuantPlanner`
- 在模型构建前先产出每层 expert weight type
- 再据此选择 `UnquantizedFusedMoEMethod` 或 `AWQSM70MoEMethod`

优点：

- MiniMax 语义更直观
- 后续对 MiniMax 家族扩展更清晰

缺点：

- 引入新的量化配置分支
- 与现有 AWQ 框架重叠
- 对当前需求过重

### 方案 C：抽象为通用 moe-only quant planner

做法：

- 参考 TurboMind converter，把 `moe-only quant + per-layer expert weight type` 抽象成一层通用计划器
- 供 `MiniMaxM2`、`Qwen3Next`、`Qwen3.5-MoE` 等模型共用

优点：

- 长期结构最好
- 更接近 TurboMind 当前策略

缺点：

- 范围明显扩大
- 涉及更多模型回归
- 不适合作为 MiniMax-M2.5-AWQ 首次落地方案

## 6. 推荐方案

推荐采用方案 A。

原因：

- 当前 1Cat 对 `MiniMaxM2` 的模型实现已经足够完整，问题不在模型结构层。
- `SM70` AWQ/TurboMind kernel 已在仓库内打通，问题不在 kernel 供给层。
- 真正缺的是 `FusedMoE` 的量化决策与 TurboMind MiniMax 语义不一致。
- 先把量化决策修正为正确语义，再决定是否有必要抽象成通用 planner，风险和收益比最好。

## 7. 详细设计

### 7.1 总体调用路径

目标路径如下：

`config.json`
-> `AWQConfig / modules_to_not_convert`
-> `MiniMaxM2ForCausalLM`
-> `MiniMaxM2DecoderLayer`
-> `MiniMaxM2Attention` 与 `MiniMaxM2MoE`
-> `FusedMoE quant_method` 按层选择
-> `UnquantizedFusedMoEMethod` 或 `AWQSM70MoEMethod`
-> logits

其中：

- attention 继续走现有 vLLM 推理流程
- 只有 experts 的执行方法需要按层切换

### 7.2 量化决策层

主修改点在：

- `vllm/model_executor/layers/quantization/awq.py`

需要新增或补齐的行为：

1. `FusedMoE` 分支在选择 `AWQSM70MoEMethod` 前，先执行与 `AWQMarlinConfig` 一致的 skip 判断。
2. 对命中 `modules_to_not_convert` 的 `FusedMoE`，返回 `UnquantizedFusedMoEMethod`。
3. 对未命中的 `FusedMoE`，继续现有 `SM70 compatibility` 判断，再进入 `AWQSM70MoEMethod` 或 fallback。

重点是支持以下匹配语义：

- `self_attn`
- `block_sparse_moe.gate`
- `model.layers.0.`

其中最后一个不对应单独的 `LinearBase`，而是会影响该层的 `FusedMoE` 整体选择。

### 7.3 模型层

主目标是“不改结构，只验证语义已正确接入”。

涉及文件：

- `vllm/model_executor/models/minimax_m2.py`

预期保持不变的部分：

- `MiniMaxM2Attention`
- `MiniMaxM2MoE`
- expert 权重装载映射
- `MiniMaxText01RMSNormTP` 与 partial rotary 路径

只在必要时补小型辅助逻辑，例如：

- 如果 prefix 命名与 `modules_to_not_convert` 的匹配粒度不一致，则在最小范围内补 prefix 规范化

但设计上默认不把模型类作为主改动点。

### 7.4 TurboMind 语义对齐边界

需要对齐的 TurboMind 语义：

- `MiniMaxM2ForCausalLM` 是 `moe-only quant`
- `modules_to_not_convert` 决定部分层 experts 保持高精度

不需要对齐或复制的部分：

- TurboMind MiniMax 专属 attention runtime
- TurboMind converter 全流程
- TurboMind 作为整体推理后端
- Qwen3.5 类似的模型 custom op 路径

### 7.5 Warmup

主目标是确保：

- 命中 `AWQSM70MoEMethod` 的 MiniMax experts 层仍然进入现有 `SM70 AWQ warmup`
- 未量化的层不参与 AWQ warmup

预期不新增 MiniMax 专属 warmup 文件，复用：

- `vllm/model_executor/warmup/awq_sm70_warmup.py`

只有当现有 warmup 遍历条件误把未量化 MoE 层当作 SM70 AWQ 层时，才做修复。

## 8. 文件级改动规划

### 必改

- `vllm/model_executor/layers/quantization/awq.py`

### 大概率新增测试

- `tests/.../test_minimax_m2_awq_sm70.py`
- 或放入现有 quantization / model_executor 测试目录中，取决于仓库现有组织方式

### 可能只读验证，不改代码

- `vllm/model_executor/models/minimax_m2.py`
- `vllm/model_executor/warmup/awq_sm70_warmup.py`

## 9. 验证策略

### 9.1 单元级验证

至少覆盖以下断言：

1. `self_attn` 命中 `modules_to_not_convert` 时仍为非量化。
2. `block_sparse_moe.gate` 仍为非量化。
3. `model.layers.0.block_sparse_moe` 返回 `UnquantizedFusedMoEMethod`。
4. 非第 0 层、满足 `SM70` 约束的 `block_sparse_moe` 返回 `AWQSM70MoEMethod`。
5. 若维度不兼容 `SM70` kernel，仍按现有逻辑 fallback，不破坏现有行为。

### 9.2 模型级 smoke test

目标模型：

- `/mnt/data6/models/MiniMax-M2.5-AWQ`

建议至少验证：

- `SM70` 上模型可成功加载
- TP=1 和 TP=2 至少覆盖一种
- 能完成一次最小生成
- 日志中未出现“layer 0 experts 进入 SM70 AWQ path”的错误信号

### 9.3 回归边界

至少确认不破坏：

- 已有 `Qwen3.5 AWQ SM70`
- 非 MiniMax 的普通 AWQ linear 路径
- 非 `SM70` 的 AWQ MoE 路径

## 10. 风险

### 风险 1：prefix 匹配不准

如果 `modules_to_not_convert` 与 `FusedMoE` 实例使用的 prefix 不完全一致，可能导致：

- 本应 skip 的层未 skip
- 本应量化的层被错误跳过

这是本次实现的首要风险。

### 风险 2：按层 mixed expert type 的语义只做了 skip，没有抽象成通用计划

对 `MiniMax-M2.5-AWQ` 来说足够，但后续如果引入更多 “部分层 experts 为 fp16” 的模型，通用性会不足。

### 风险 3：config 字段与实际 runtime 语义不完全一致

例如 `config.json` 中的 `qk_norm_type` 与 TurboMind 对 MiniMax 的解释并不完全相同。当前设计默认沿用 1Cat 已存在的 `MiniMaxText01RMSNormTP` 路径，不把 QK norm 作为本轮主风险项，但 smoke test 仍需关注输出稳定性。

## 11. 非目标

以下内容不在本轮实现范围内：

- 让 MiniMax 在 1Cat 上走 TurboMind attention runtime
- 复制 lmdeploy 的 converter / deploy 目录
- 重构 1Cat 的全局 AWQ 体系
- 为所有 moe-only-quant 模型一次性抽象通用 planner

## 12. 实施顺序建议

建议按以下顺序实施：

1. 在 `awq.py` 为 `FusedMoE` 增加 skip 逻辑。
2. 写单测固定 `MiniMax-M2.5-AWQ` 的语义。
3. 验证 `SM70` 上非第 0 层命中 `AWQSM70MoEMethod`。
4. 跑目标模型最小 smoke test。
5. 如果验证表明后续还有更多同类模型，再考虑抽象通用 `moe-only quant planner`。

## 13. 最终结论

把 TurboMind 支持的 `MiniMax-M2.5-AWQ` 推理迁移到 `1Cat-vLLM` 的正确路径，不是复制 TurboMind 的 MiniMax 模型执行，而是：

- 保留 1Cat 现有的 `MiniMaxM2` vLLM 推理主链
- 把 TurboMind 的 `moe-only quant + per-layer skip` 语义移植到 1Cat 的 AWQ 量化决策层
- 在需要量化的 experts 层复用现有 `SM70` TurboMind AWQ MoE kernel

一句话概括：

`MiniMaxM2 model stays in vLLM; MiniMax-M2.5-AWQ expert quant semantics come from TurboMind.`
