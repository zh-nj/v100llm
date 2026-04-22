# SM70 FP8 Direct MoE Design

日期：2026-04-22

## 1. 目标

为 `sm70 / V100 / Tesla PG503-216` 增加可复用的 FP8 MoE expert 推理路径，使
`/mnt/data6/models/Qwen3.6-35B-A3B-FP8` 的 language model MoE experts 能在以下约束下运行：

- expert 权重常驻显存必须保持 `FP8 + block scale` 压缩态
- 不允许把 FP8 expert 权重转换成 AWQ/int4 或常驻 FP16/BF16
- runtime 不允许全量展开 expert 权重，只允许使用现有 MoE activation/output workspace
- 正式路径必须兼容 CUDA graph，目标配置为 `cudagraph_mode=full_and_piecewise`
- 复用现有 SM70 TurboMind GEMM、StridedPtr、MoE permute/unpermute 和 warmup 机制
- 第一轮覆盖 text generate 所需的 `language_model` routed MoE experts

## 2. 非目标

本轮不做以下事情：

- 不优化 visual encoder 的 FP8/非 FP8 模块
- 不优化 MTP
- 不扩展 KV cache FP8
- 不实现 SM70 原生 FP8 Tensor Core GEMM
- 不做 FP8 到 AWQ/int4 的二次量化
- 不把 MoE expert 作为 dense layer 逐 expert Python 循环长期执行
- 不在第一版实现 expert parallel all-to-all 的新通信后端

## 3. 目标模型事实

目标模型配置：

- `model_type=qwen3_5_moe`
- text config: `model_type=qwen3_5_moe_text`
- `hidden_size=2048`
- `num_hidden_layers=40`
- `num_experts=256`
- `num_experts_per_tok=8`
- `moe_intermediate_size=512`
- FP8 quantization: `quant_method=fp8`, `activation_scheme=dynamic`, `fmt=e4m3`
- `weight_block_size=[128,128]`

实际 expert checkpoint 形状示例：

- `gate_proj.weight`: `[512, 2048]`, `float8_e4m3fn`
- `gate_proj.weight_scale_inv`: `[4, 16]`, `bfloat16` on disk, loaded into float scale parameter
- `up_proj.weight`: `[512, 2048]`, `float8_e4m3fn`
- `up_proj.weight_scale_inv`: `[4, 16]`
- `down_proj.weight`: `[2048, 512]`, `float8_e4m3fn`
- `down_proj.weight_scale_inv`: `[16, 4]`

vLLM `SharedFusedMoE` loads gate/up into `w13_weight` with physical shape:

- `w13_weight`: `[local_experts, 2 * intermediate_per_partition, hidden]`
- `w2_weight`: `[local_experts, hidden, intermediate_per_partition]`

For TP=1, Qwen3.6-35B-A3B has:

- `w13`: `[256, 1024, 2048]`
- `w2`: `[256, 2048, 512]`

For TP=2:

- `w13`: `[256, 512, 2048]`
- `w2`: `[256, 2048, 256]`

All these dimensions remain multiples of `128`, so direct FP8 SM70 GEMM is viable.

## 4. Candidate 方案

### 方案 A：FP8 Direct MoE GEMM

加载后把每个 expert 的 FP8 weight 无损转换到 TurboMind SM70 e4m3 packed layout，把 block scale 转成 TurboMind group scale layout。runtime 使用 grouped/batched `Gemm::Run`，权重 operand 为 `kFloat8_e4m3`，`quant_b={kK,128}`。

优点：

- 满足 FP8 常驻约束
- 不做有损二次量化
- 复用 dense FP8 direct GEMM 的 converter 和 TurboMind kernel registry
- 复用 AWQ SM70 MoE 的 dispatch、StridedPtr、permute/unpermute、workspace、warmup
- CUDA graph 友好

缺点：

- 需要新增 C++ op 和 Python quant method
- 加载期需要为 `[E,N,K]` expert weight 做一次 packed layout prepare

结论：采用。

### 方案 B：FP8 Runtime Decode MoE

runtime 按 active experts 把 FP8 临时 decode/pack 到 FP16 workspace，再调用 SM70 f16 GEMM。

优点：

- 与最早 dense runtime decode 方案概念一致
- 可以覆盖部分非 128x128 layout

缺点：

- top_k=8 会放大 decode/pack 开销
- decode token 会被 active expert 数量拖慢
- CUDA graph workspace 管理更复杂

结论：只作为后续 fallback，不作为 Qwen3.6 第一版主路径。

### 方案 C：FP8 -> AWQ/int4 MoE

加载期把 expert FP8 重量化成 AWQ/int4，直接走现有 AWQ SM70 MoE。

优点：

- 改动最少

缺点：

- 违反 FP8 常驻要求
- 有二次量化损失
- 不能作为通用 FP8 模型基础设施

结论：不采用。

## 5. 推荐架构

新增 `Fp8SM70DirectMoEMethod`，只在以下条件同时满足时选中：

- 当前设备 capability 为 `70`
- layer 是 `FusedMoE`
- checkpoint 是 serialized FP8
- activation scheme 是 `dynamic`
- weight block size 是 `[128,128]`
- expert `hidden/intermediate_per_partition` 维度均可被 `128` 整除

该方法与 dense `Fp8SM70RuntimeDecodeLinearMethod` 并列存在：

- dense Linear 继续走现有 SM70 FP8 direct GEMM
- routed MoE experts 走新的 SM70 FP8 direct MoE GEMM
- shared expert 是普通 dense MLP，继续由 dense FP8 direct path 覆盖
- MoE gate/router 是 BF16/FP16 dense，不进入 FP8 expert path

## 6. 常驻权重格式

加载完成后，MoE layer 保留以下常驻 tensors：

- `layer.w13_tm_weight`
  - dtype: `torch.float8_e4m3fn`
  - shape: `[E, N13, K13]`
  - 内容是 TurboMind SM70 e4m3 packed layout
- `layer.w13_tm_scales`
  - dtype: `torch.float16`
  - shape: `[E, K13 / 128, N13]`
  - 内容是 TurboMind group scale layout
- `layer.w2_tm_weight`
  - dtype: `torch.float8_e4m3fn`
  - shape: `[E, N2, K2]`
- `layer.w2_tm_scales`
  - dtype: `torch.float16`
  - shape: `[E, K2 / 128, N2]`

加载后删除原始 unprepared `w13_weight/w2_weight` 和 float32 block scale 参数，避免额外常驻显存。最终常驻仍是 `FP8 + scale` 压缩态，不存在整 expert FP16 权重副本。

## 7. C++ Custom Ops

### 7.1 `sm70_fp8_moe_direct_prepare`

新增 3D prepare op：

```text
sm70_fp8_moe_direct_prepare(
  weight,
  weight_scale,
  block_n,
  block_k,
  interleave_gated_silu,
)
  -> prepared_weight, prepared_scale, prepared_meta
```

输入：

- `weight`: `[E, N, K]`, `float8_e4m3fn`
- `weight_scale`: `[E, N / 128, K / 128]`, float32-compatible
- `block_n=128`
- `block_k=128`
- `interleave_gated_silu`: whether to interleave gate/up output rows before packing

输出：

- `prepared_weight`: `[E, N, K]`, `float8_e4m3fn`
- `prepared_scale`: `[E, K / 128, N]`, `float16`
- `prepared_meta`: int64 `[N, K, group_size, k_ld, q_ld]`

实现复用 dense `sm70_fp8_direct_prepare` 的内部 primitives：

- `extend_fp8_e4m3_to_u16`
- `expand_block_fp8_scales_to_group_half`
- `pack_sm70_fp8_weight_into`
- `pack_sm70_fp8_scales_into`

For `w13`, `interleave_gated_silu=True` makes the packed output order match
TurboMind `Epilogue::kGatedSilu`: `[gate0, up0, gate1, up1, ...]`.
The prepare op must interleave both the FP8 weight rows and the expanded
per-output scale columns before packing. Interleaving only the weight would
apply incorrect block scales. For `w2`, `interleave_gated_silu=False`.

prepare 在 C++ 内循环 experts，避免 Python 端 40 层 * 256 experts * 2 matrices 的高额调度开销。

### 7.2 `sm70_fp8_moe_gemm_out`

新增 grouped/batched MoE GEMM op：

```text
sm70_fp8_moe_gemm_out(
  out,
  sorted_input,
  expert_offsets,
  strided_ptrs_w,
  strided_ptrs_s,
  num_experts,
  k,
  n,
  group_size,
  gated_silu=False,
)
```

它复用 `awq_moe_gemm_sm70_out` 的结构，但修改：

- weight converter: `GetConverters(kHalf, kFloat8_e4m3, kHalf, grouped=true, sm=70)`
- `desc_B.type = kFloat8_e4m3`
- scale descriptor dtype: `kUint16`
- `op.quant_b = {QuantType::kK, 128}`
- `op.batch_dim = 0`
- `desc_A/desc_D.offsets = expert_offsets`
- `desc_B/desc_V.ld = 0` 使用 StridedPtr

`gated_silu=True` 用于 `w13`，输出写入 `[total_slots, intermediate]`。`gated_silu=False` 用于 `w2`，输出写入 `[total_slots, hidden]`。

### 7.3 StridedPtr

现有 `awq_moe_build_strided_ptrs()` 本质只依赖 tensor base pointer、expert stride、`k_ld/q_ld`，不依赖 AWQ dtype。第一版直接复用该 op，并在 Python 侧以 FP8 tensors 构造：

- `w13_strided_ptrs_w`
- `w13_strided_ptrs_s`
- `w2_strided_ptrs_w`
- `w2_strided_ptrs_s`

后续可增加别名 `sm70_moe_build_strided_ptrs()`，但不是第一版必要项。

## 8. Python Quant Method

新增 `Fp8SM70DirectMoEMethod(FusedMoEMethodBase)`。

### 8.1 `create_weights`

沿用 `Fp8MoEMethod.create_weights()` 的参数命名和 loader 约定：

- `w13_weight`
- `w2_weight`
- `w13_weight_scale_inv`
- `w2_weight_scale_inv`

scale 参数用 float32 存储，checkpoint 的 bf16 scale 由 loader copy 到 float32 参数中。

### 8.2 `process_weights_after_loading`

处理步骤：

1. 校验 `weight_block_size == [128,128]`
2. 校验 `w13/w2` 的 `N/K` 均为 128 的整数倍
3. 调用 `ops.sm70_fp8_moe_direct_prepare(..., interleave_gated_silu=True)` 准备 w13
4. 调用 `ops.sm70_fp8_moe_direct_prepare(..., interleave_gated_silu=False)` 准备 w2
5. 将输出注册为 non-trainable parameters
6. 调用 `ops.awq_moe_build_strided_ptrs()` 构造 batched GEMM pointer tables
7. 删除原始 `w13_weight/w2_weight/w13_weight_scale_inv/w2_weight_scale_inv`
8. 预分配 decode 常用 MoE buffers

### 8.3 `apply`

runtime 复用 AWQ SM70 MoE 的高层 flow：

1. router 得到 `topk_weights/topk_ids`
2. `_moe_C.moe_permute()` 按 expert 对 token 展开和排序
3. `sm70_fp8_moe_gemm_out(... gated_silu=True)` 计算 w13
4. `sm70_fp8_moe_gemm_out(... gated_silu=False)` 计算 w2
5. `_moe_C.moe_unpermute()` 按 top-k weight 合并回 token 输出

第一版要求 `sm70_batched_ready=True`。如果 StridedPtr 构造失败，直接报错，不回退到慢速 Python per-expert loop。这样可以避免误以为正式路径满足 CUDA graph。

## 9. CUDA Graph 和 Workspace

正式验证必须使用：

```json
{"cudagraph_mode":"full_and_piecewise"}
```

MoE runtime 不在 CUDA graph capture 中分配新的 persistent tensors：

- decode 常用 token 数使用 layer 内预分配 buffers
- larger prefill 如超过 persistent buffer，可走普通临时 tensor；CUDA graph capture sizes 覆盖 decode/microbatch 形状
- GEMM workspace 复用现有 per-stream `WorkspaceHolder`
- LUT/tuning 通过 warmup 在 capture 前完成

warmup 扩展 `awq_sm70_warmup.py`：

- 统计 `direct FP8 MoE shapes`
- 对每个 MoE layer 分别 warm up w13/w2
- log 示例：

```text
Warming up SM70 AWQ/FP8 kernels (..., X direct FP8 MoE shapes)
SM70 AWQ/FP8 warmup finished (..., Y direct FP8 MoE calls)
```

## 10. Tensor Parallel 和 Expert Parallel

第一版支持 TP=1/2，只要求 local shard 维度满足 128 对齐。

TP 行为：

- `w13` 是 column-parallel expert projection，`N13=2*intermediate/tp`
- `w2` 是 row-parallel expert projection，`K2=intermediate/tp`
- 输出 reduce 仍由现有 `FusedMoE` / runner / tensor parallel 逻辑处理

EP 行为：

- 使用现有 `FusedMoE` 的 local expert mapping
- `num_experts` 传入 local physical expert 数
- 第一版不新增 all-to-all kernel

如果某个 TP/EP 配置导致 `N/K` 非 128 对齐，`Fp8SM70DirectMoEMethod` 必须在加载后明确报错，不能静默落到错误 kernel。

## 11. 测试计划

### Unit tests

- `Fp8Config.get_quant_method()` 在 SM70 serialized block FP8 `FusedMoE` 上返回 `Fp8SM70DirectMoEMethod`
- 非 SM70 或非 block `[128,128]` 不选中该方法
- `create_weights()` 创建正确 dtype/shape 的 `w13/w2` 和 scale 参数
- `process_weights_after_loading()` 后只保留 FP8 packed weight 和 half scale，不保留 FP16 full expert
- unsupported shape 明确报错

### CUDA kernel tests

- `sm70_fp8_moe_direct_prepare()` 对 `[E,N,K]` 输出 meta/shape/dtype 正确
- `sm70_fp8_moe_gemm_out()` 数值对齐 torch reference：
  - `w13`: `silu(x @ gate.T) * (x @ up.T)`
  - `w2`: `hidden @ down.T`
- top-k weighted unpermute 后输出对齐 reference MoE
- CUDA graph capture/replay 覆盖 `M=1` decode shape

### Model smoke

使用 `/mnt/data6/models/Qwen3.6-35B-A3B-FP8`：

- 启动服务确认不再报 `sm70 serialized FP8 MoE is not supported yet`
- 日志确认 direct dense FP8 和 direct MoE FP8 warmup
- 日志确认 `CUDAGraphMode.FULL_AND_PIECEWISE`
- `/v1/models` 返回目标 model
- 简短生成输出语义正常

### 性能验证

沿用现有测速格式，至少输出：

- `prefill tokens/s`
- `decode tokens/s`
- `TTFT`
- `finish_reason`
- 语义质量结论

输入长度覆盖：

- 1k
- 32k

## 12. 风险和缓解

### 风险：prepare 启动时间过长

Qwen3.6 有 40 层、256 experts、每层两组 expert matrices。使用 3D C++ prepare 一次处理一组 `[E,N,K]`，减少 Python 调度开销。后续如仍慢，可增加 fused w13/w2 prepare。

### 风险：FP8 scale on disk 是 bf16

loader 参数使用 float32，checkpoint bf16 scale copy 到 float32 后进入 prepare；C++ prepare 接受 float32 scale，并输出 half scale layout 给 TurboMind。

### 风险：single-token decode overhead

第一版先复用 batched grouped GEMM。若 profiling 显示 `M=1, top_k=8, E=256` 下 StridedPtr/offset 开销明显，再参考 AWQ `single_token_compact` 增加 FP8 compact active-expert path。

### 风险：CUDA graph capture 中出现隐式分配

所有 small decode buffers 在 `process_weights_after_loading()` 预分配；GEMM workspace 和 LUT 通过 warmup 在 capture 前准备。测试必须包含 CUDA graph capture/replay。

### 风险：与现有 FP8 MoE backend selection 冲突

SM70 path 不走 `select_fp8_moe_backend()`，避免进入 Triton/Cutlass/DeepGEMM 的 SM80+ 假设。其他平台仍走原有 `Fp8MoEMethod`。

## 13. 成功标准

第一版完成时必须满足：

- Qwen3.6-35B-A3B-FP8 在 SM70 上能启动
- routed MoE experts 使用 FP8 compressed resident direct GEMM
- 不存在常驻 FP16/BF16 expert weight 副本
- CUDA graph 模式为 `full_and_piecewise`
- 1k/32k 生成语义正确，`finish_reason` 合理
- 相关 unit/kernel/warmup tests 通过
