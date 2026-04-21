# SM70 FP8 Runtime Decode Dense Linear Design

日期：2026-04-21

## 1. 目标

为 `sm70 / V100` 增加一条新的 `dense Linear` 推理路径，使序列化
`block-FP8` checkpoint 可以在以下约束下运行：

- 权重必须以 `FP8 + scale` 压缩态常驻显存
- 运行时只允许使用小型临时 workspace
- 不允许把整层权重重打包成 `AWQ / INT4 / 常驻 FP16`
- 必须复用现有 `SM70` 的 `f16 GEMM` 栈
- 第一里程碑必须覆盖普通 dense `Linear` 以及 fused/merged 线性层

本次设计只覆盖 `dense Linear`。`MoE / KV cache / attention` 留到下一轮。

## 2. 非目标

本次明确不做以下事情：

- 不实现“真 FP8 Tensor Core GEMM”
- 不把 `sm70` 宣布为通用 `fp8` 平台
- 不扩展到 `MoE`
- 不扩展到 `KV cache / attention`
- 不接受 `FP8 -> AWQ` 的离线或加载期重打包方案
- 不把整层权重在 runtime 全量解码成临时 `FP16` 矩阵

## 3. 约束与背景

现有仓库在 `sm70` 上可复用的 dense 计算栈主要有两类：

1. `AWQ` 的 `awq_sm70_prepare / awq_gemm_sm70`
2. `f16` 的 `sm70_f16_prepare / sm70_f16_gemm_out`

其中：

- `AWQ` 路径要求常驻 `int4` 权重，不满足“FP8 常驻”的新约束
- `sm70_f16` 路径要求权重已转换成 TurboMind 期望的 `f16` layout

因此，新方案的核心不是再做一种新的终态权重格式，而是：

- 常驻 `FP8` 权重
- runtime 按小 panel 解码并转换成临时 `TM-f16` panel
- 立刻复用现有 `sm70_f16 GEMM`

## 4. 候选方案

### 方案 A：FP8 -> AWQ/int4 重打包

优点：

- 改动小
- 直接复用现有 `awq_sm70` kernel

缺点：

- 违反“FP8 常驻”约束
- 是有损二次量化
- 与后续 `MoE / KV cache` 方案不一致

结论：不采用。

### 方案 B：保留原始 checkpoint 布局，runtime 直接按原布局解码

优点：

- 加载期改动最少

缺点：

- runtime 地址计算复杂
- fused/merged 层尾块与逻辑分片处理复杂
- 很难稳定复用 `sm70_f16` 的 tile/workspace 约定

结论：不作为第一版方案。

### 方案 C：加载期做无损 SM70 友好重排，runtime 小 panel decode/pack

优点：

- 满足 `FP8` 常驻
- workspace 可以限制在 panel 级
- 便于统一支持普通 dense 与 fused/merged 线性层
- 可以复用现有 `sm70_f16` 的 dispatch、workspace 和 warmup 机制

缺点：

- 需要新增 `sm70` custom op
- 需要定义新的常驻布局和 panel workspace 契约

结论：采用。

## 5. 推荐方案

第一里程碑新增 `Fp8SM70RuntimeDecodeLinearMethod`，只在以下条件下选中：

- 当前设备 capability 为 `70`
- quantization 为序列化 `fp8`
- 权重为 block quant
- activation scheme 为 `dynamic`
- 当前层属于 `LinearBase`

该方法不用于 `MoE` 或 `Attention`。`sm70` 上的序列化 `FP8 MoE` 继续明确报错。

## 6. 设计细节

### 6.1 Python 层方法选择

`Fp8Config.get_quant_method()` 增加一条新的窄路径：

- 对 `LinearBase`：
  - 满足 `sm70 + serialized fp8 + block quant + dynamic act` 时返回
    `Fp8SM70RuntimeDecodeLinearMethod`
  - 否则继续走现有 `Fp8LinearMethod / Fp8OnlineLinearMethod`
- 对 `FusedMoE`：
  - `sm70 + serialized fp8` 继续报错，不做伪支持
- 对 `Attention`：
  - 本轮不改

全局 capability gate 也只为这条窄路径放行，不能把 `FP8` 的最低能力要求整体降到 `70`。

### 6.2 支持范围

第一里程碑支持：

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `RowParallelLinear`
- `MergedColumnParallelLinear`

这意味着以下 fused/merged 形态必须一起支持：

- `QKV`
- `gate_up`
- `in_proj_qkvz`
- 其他通过 `LinearBase + output_partition_sizes` 表示的 packed dense 权重

本设计不需要为 fused/merged 层单独写另一套 kernel。它们在 runtime 仍被视为
“一个物理矩阵”，只是 Python 层保留逻辑 shard 描述用于加载、校验和尾块处理。

### 6.3 常驻显存布局

权重常驻格式仍是 `FP8 + block scale`，但允许在加载后做无损规范化与重排。

第一版定义的常驻状态如下：

- `layer._sm70_fp8_weight`
  - `torch.float8_e4m3fn`
  - 逻辑上仍表示 `[out, in]`
  - 加载后允许做 padding 和按输出 panel 对齐的重排
- `layer._sm70_fp8_scale`
  - `torch.float32`
  - 保持 block scale 语义
- `layer._sm70_fp8_meta`
  - 小型元数据张量或 Python attrs
  - 记录 `block_n`, `block_k`, `panel_n`, `padded_out`, `padded_in`,
    `logical_widths`

第一版不把常驻格式转成 `AWQ/int4`，也不保存整层 `TM-f16` 权重。

加载期 prepare 允许做的事情：

- `fn/fnuz` 规范化
- `K/N` 维 padding
- 输出维按 panel 对齐
- 为 fused/merged 层记录 logical shard offsets

加载期 prepare 不允许做的事情：

- 量化位宽改变
- 常驻 `FP16/BF16` 整层副本
- 依赖 runtime 输入 batch shape 的任何缓存

### 6.4 Runtime panel decode 策略

为了满足“小 workspace”，第一版只沿 `N` 维切 panel，不沿 `K` 维分块累加。

原因：

- 现有 `sm70_f16_gemm_out` 适合直接消费完整 `K` 维权重
- 若同时切 `K` 维，需要在输出上显式累加，复杂度和风险都会上升
- 仅沿 `N` 维切 panel 已经能把 workspace 控制到远小于整层大小

第一版的 runtime 算法是：

1. 将输入 reshape 成 `M x K`
2. 按 `panel_n` 遍历输出列块
3. 从常驻 `FP8` 权重中取当前 panel 对应的 `FP8 block`
4. 用对应的 block scale 解码成临时 `f16` panel
5. 把该 `f16 panel` 转换成 `sm70_f16 GEMM` 所需的 `TM` layout
6. 直接调用现有 `SM70 f16 GEMM`
7. 把结果写入输出张量的对应列区域

这样 workspace 大小约为：

- `panel_weight_fp16 = panel_n x K`
- `panel_weight_tm = panel_n x K` 的 TurboMind layout

而不是整层 `N x K`。

### 6.5 Panel 大小与块形状

第一版只支持 checkpoint 中最常见的 `weight_block_size = [128, 128]`。

原因：

- 当前目标模型就是 `128x128`
- 现有序列化 FP8 模型主要使用该配置
- `panel_n` 与 `block_n` 对齐后，decode 与 tail 处理最简单

因此第一版约束为：

- `block_n = 128`
- `block_k = 128`
- `panel_n` 为 `128` 的整数倍
- 默认首选 `panel_n = 128`

如果后续验证显示 `panel_n = 256` 更好，可以作为调优项，而不是第一版的接口分叉点。

### 6.6 新 custom op 形态

第一版不暴露“decode op + f16 prepare op + gemm op”三段式 Python 拼装。

原因：

- Python 循环 panel 会带来高额调度开销
- `sm70_f16_prepare` 当前是面向整层静态权重缓存的接口，不适合 transient panel
- panel decode、TM pack、GEMM dispatch 本身需要共享同一套 workspace 与 stream 语义

因此新增单一 custom op：

- `sm70_fp8_runtime_gemm_out(...)`

职责：

- 读取常驻 `FP8` 权重与 scale
- 在 CUDA 内部完成 panel decode
- 在 CUDA 内部完成 panel 到 `TM-f16` layout 的临时转换
- 复用现有 `sm70_f16 GEMM` 的 dispatch / workspace / warmup 机制
- 将结果直接写入输出张量

Python 层只负责：

- 选择 quant method
- 保存常驻权重与 meta
- 调用新的 custom op

### 6.7 与现有 SM70 f16 GEMM 的复用边界

“复用现有 `SM70 f16 GEMM`”在第一版中的含义是：

- 复用 TurboMind GEMM 本体
- 复用现有 dispatch policy 选择逻辑
- 复用 GEMM workspace 管理
- 复用 warmup / LUT cache 机制

第一版不要求：

- 必须在 Python 层直接调用 `sm70_f16_gemm_out`
- 必须复用 `sm70_f16_prepare` 的整层缓存接口

也就是说，新 op 允许在 C++/CUDA 内部调用与 `sm70_f16_gemm_out` 相同的底层 GEMM 逻辑，
但以“transient panel”而不是“整层预转换权重”的方式提供输入。

### 6.8 Fused/Merged 层处理

fused/merged 层仍按一个物理权重处理，不为每个 logical shard 独立启动 kernel。

需要保证的行为：

- `MergedColumnParallelLinear` 的 `output_partition_sizes` 在 prepare 后仍被保留
- tail panel 可以跨 shard 边界
- mixed precision packed 层仍沿用现有 `modules_to_not_convert` / packed mapping 逻辑

特别是 `Qwen3.5` 这类模型：

- 如果 checkpoint 通过 `modules_to_not_convert` 要求把 `in_proj_b`/`in_proj_a`
  从 `in_proj_qkvz` 中拆出来，模型侧 packed mapping 仍然应该优先尊重拆分
- 新的 runtime decode 路径只处理“最终被建模为一个 quantized dense linear”的物理层

### 6.9 错误处理

第一版显式拒绝以下情况：

- 非 `sm70`
- 非序列化 `FP8`
- 非 block quant
- `activation_scheme = static`
- `weight_block_size != [128, 128]`
- `MoE`
- `KV cache / attention`

错误信息必须直接说明：

- 当前限制是什么
- 当前层类型是什么
- 下一步应走哪个现有后端或为什么仍不支持

### 6.10 测试设计

第一里程碑测试分三层：

#### 单元测试

覆盖：

- `Fp8Config` 在 `sm70` 上选择新 method
- capability gate 只对新窄路径放行
- fused/merged dense 层仍能走同一选择逻辑
- `modules_to_not_convert` 别名兼容不回退

#### 算子集成测试

覆盖：

- fake/custom op 注册
- `panel_n=128` 下的输出形状正确
- 多 panel 输出拼接正确
- tail panel 正确
- fused/merged 物理层输出形状正确

#### 真实模型 smoke

使用：

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `CUDA_VISIBLE_DEVICES=2` 或 `3/4/5` 中空闲卡
- `/mnt/data6/models/Qwen3.5-0.8B-FP8`

目标：

- 至少验证模型可构建、可加载、可跑一次短生成
- 证明新路径不再依赖 `AWQ` 常驻重打包

## 7. 里程碑切分

### 第一里程碑

- `dense Linear` 路径选通
- 新 resident layout
- 新 runtime decode custom op
- 普通 dense 与 fused/merged dense 层可运行
- `Qwen3.5-0.8B-FP8` smoke 通过

### 第二里程碑

- `MoE`
- `KV cache / attention`

## 8. 风险

主要风险有四个：

1. panel decode + TM pack 的 runtime 开销过高，导致吞吐明显低于预期
2. fused/merged 层的 tail panel 与 logical shard 边界处理出错
3. 新 op 如果直接复用现有 `sm70_f16` 内部逻辑不当，可能破坏已有 `AWQ/f16` 路径的 LUT 或 workspace 行为
4. `Qwen3.5` 这类 mixed precision packed mapping 模型，模型侧拆分逻辑与 quant config 字段别名可能再次出现不一致

第一版实现必须优先控制正确性与接口稳定性，再看性能调优。

## 9. 决策总结

本次采用的方案是：

- `FP8` 权重常驻显存
- 加载期仅做无损规范化和 SM70 友好重排
- runtime 仅沿 `N` 维做小 panel decode
- panel 在临时 workspace 中解码成 `f16` 并转换成 `TM` layout
- 复用现有 `SM70 f16 GEMM`
- 第一版仅覆盖 `dense/fused/merged Linear`

这条方案满足当前用户约束，也为下一轮 `MoE / KV cache` 保留了一致的“压缩态常驻 + runtime 临时 decode”架构方向。
