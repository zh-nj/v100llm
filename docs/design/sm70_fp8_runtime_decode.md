# SM70 FP8 Runtime Decode Inference Guide

本文梳理 SM70/V100 上 FP8 模型的推理、数据准备和计算流程，并给出后续适配其它 FP8 模型时的工程流程。这里的 SM70 FP8 路线不是原生 FP8 Tensor Core 计算，而是：

```text
FP8 权重常驻显存压缩态
  -> runtime 按 panel 临时 decode/pack 到 workspace
  -> 复用 SM70 f16 GEMM
  -> 输出 fp16 activation
```

核心目标是节省权重显存，同时保证 SM70 可运行。不要把它理解成 Hopper/Ada 上的原生 FP8 W8A8 GEMM。

## 适用范围

当前路线优先覆盖 dense Linear，包括 `QKVParallelLinear`、`MergedColumnParallelLinear`、`ColumnParallelLinear`、`RowParallelLinear`、`ReplicatedLinear` 等 dense/fused/merged 线性层。

已验证或应优先复用的量化入口：

- vLLM `fp8` quantization path。
- `compressed-tensors` 的 `float-quantized` FP8 checkpoint。
- 权重 dtype 为 `torch.float8_e4m3fn`。
- scale strategy 为 tensor、channel 或 block。

不应默认纳入同一个适配闭环的内容：

- KV cache FP8。它是 cache dtype/attention 问题，不是 weight-only FP8 dense GEMM 问题。
- MoE expert kernel。MoE 可以复用部分底层 primitive，但路由、expert batching、workspace 粒度不同，必须单独验证。
- 视觉塔或其它被 checkpoint quantization `ignore` 的模块。这些模块通常仍是 BF16/FP16 常驻，不能用文本侧 FP8 显存收益推断整模型显存。

## SM70 约束

SM70 没有可用于 vLLM FP8 GEMM 的原生 FP8 Tensor Core 路径。直接加载 FP8 checkpoint 后，如果沿用普通 FP8 W8A8 scheme，通常会遇到 capability gate、kernel 不支持或错误 fallback。

因此 SM70 的正确原则是：

- checkpoint 中的 FP8 weight 保持 FP8 常驻，不在加载后全量展开成 FP16。
- 每次 Linear forward 时，只把当前 weight panel decode 成 FP16 临时面板。
- decode 后立即 pack 成 SM70 f16 GEMM 可复用的布局。
- workspace 预分配或缓存复用，避免每层/每 token 反复大分配。
- activation 仍按 FP16/BF16 输入处理，SM70 路线本质是 W8A16。

## 关键代码路径

通用 dense helper：

- `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py`
- `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode_linear.py`

custom op：

- `csrc/quantization/awq/awq_sm70_gemm.cu`
- `vllm/_custom_ops.py`

quantization 接入：

- `vllm/model_executor/layers/quantization/fp8.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_sm70_fp8.py`

验证入口：

- `tests/quantization/test_fp8_sm70.py`
- `tests/quantization/test_compressed_tensors_sm70.py`

## 数据准备流程

### 1. 识别 checkpoint 量化格式

先看 `config.json`：

```bash
jq '.quantization_config // .quantization' /path/to/model/config.json
```

重点确认：

- `quant_method` 是否为 `compressed-tensors` 或 vLLM 已支持的 `fp8`。
- `format` 是否为 `float-quantized`。
- `weights.type` 是否为 `float`。
- `weights.num_bits` 是否为 `8`。
- `weights.strategy` 是 `tensor`、`channel` 还是 `block`。
- `input_activations` 是否存在。SM70 dense runtime-decode 路径不会使用原生 FP8 activation GEMM，因此通常会把 `input_scale` 清空或忽略。
- `ignore` 列表中是否包含 vision tower、lm_head、router、embedding、MoE expert 等模块。

不要只看模型名里有 `FP8` 就假设全模型都是 FP8。很多 VL/embedding checkpoint 会把视觉塔、merger、`lm_head` 保持 BF16。

### 2. 统计实际权重构成

适配新模型前先统计 safetensors：

```python
from pathlib import Path
from safetensors import safe_open

root = Path("/path/to/model")
by_dtype = {}
by_prefix = {}

for path in sorted(root.glob("*.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            nbytes = t.numel() * t.element_size()
            by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + nbytes
            prefix = key.split(".", 2)[:2]
            group = ".".join(prefix)
            by_prefix[group] = by_prefix.get(group, 0) + nbytes

print("by dtype")
for k, v in sorted(by_dtype.items()):
    print(k, round(v / 1024**3, 3), "GiB")

print("by prefix")
for k, v in sorted(by_prefix.items()):
    print(k, round(v / 1024**3, 3), "GiB")
```

这个统计决定后续显存预期。如果 FP8 权重只有文本 backbone，而视觉塔和 `lm_head` 都是 BF16，那么服务显存下限一定高于“纯 8B FP8 文本模型”。

### 3. scheme 选择

SM70 上不能直接使用要求 `min_capability >= 75/89` 的通用 FP8 scheme。适配逻辑应在量化配置解析阶段根据 `current_platform.get_device_capability().to_int() == 70` 选择 SM70 scheme。

对 `compressed-tensors`：

- W8A8 FP8 和 W8A16 FP8 在 SM70 上都应路由到 `CompressedTensorsSM70Fp8`。
- `CompressedTensorsSM70Fp8.get_min_capability()` 返回 `70`。
- 如果不是 FP8 `float-quantized`，不要误路由到 FP8 runtime decode。

对 vLLM `fp8`：

- SM70 dense Linear 应使用 `Fp8SM70RuntimeDecodeLinearMethod` 或等价 helper。
- merged Linear 必须保留 `logical_widths`，避免输出 padding 后形状错误。

### 4. process_weights_after_loading

加载权重后，按 strategy 做最小必要整理：

- tensor scale：整理成单 scale。
- channel scale：整理成 1D scale。
- block scale：保留 block scale，并记录 `weight_block_size`。
- weight 保持 `torch.float8_e4m3fn`。
- scale 通常保持 `torch.float32`，除非走 direct block path 需要其它 layout。
- `input_scale` 对 SM70 f16 GEMM 路径不参与计算，通常置空。

完成后调用通用 helper：

```python
prepare_sm70_fp8_runtime_decode_layer(
    layer,
    weight=layer.weight,
    weight_scale=layer.weight_scale,
    weight_block_size=weight_block_size,
    direct_block_gemm_enabled=...,
)
```

runtime-decode path 的 prepare 只应记录：

- FP8 resident weight。
- scale。
- prepared meta。
- workspace meta。
- logical output width。
- merged/fused Linear 的 logical widths。

它不应生成全量 FP16 prepared weight。

## Runtime 计算流程

SM70 dense FP8 forward 的主流程：

```text
input x: fp16/bf16 activation
  -> reshape 到 2D contiguous
  -> allocate fp16 out_padded
  -> 获取共享 workspace
  -> 对 N 维按 panel 切分 weight
  -> FP8 panel + scale decode 到 fp16 decoded_panel
  -> decoded_panel pack 成 SM70 f16 GEMM layout
  -> sm70 f16 GEMM 写入 out_padded 对应列
  -> slice logical output width
  -> add bias
  -> reshape 回原 batch 形状
```

对应 custom op：

- `sm70_fp8_prepare()`：runtime path 的轻量 prepare。对 channel/tensor path 不做全量 FP16 展开。
- `sm70_fp8_runtime_gemm_out()`：按 panel decode/pack/GEMM。
- `sm70_fp8_direct_prepare()`：block 128x128 direct path 的 prepare。
- `sm70_fp8_direct_gemm_out()`：direct path GEMM。

注意 direct path 和 runtime path 的显存语义不同：

- runtime path：权重 FP8 常驻，FP16 只在 workspace 中按 panel 临时存在。
- direct block path：可能生成额外 prepared copy，只应在确认显存和性能收益后开启。

默认适配新模型时应先走 runtime path。只有当模型是 block FP8、`block_n == block_k == 128`、性能瓶颈明确且显存允许时，再评估 direct path。

## Workspace 策略

workspace 由 `decoded_panel`、`packed_panel` 和少量 meta buffer 组成。

关键约束：

- workspace 按 device 和 K 维度共享缓存。
- 不是每层保存一份完整 FP16 权重。
- panel size 根据 `logical_n`、`logical_k` 和预算选择。
- 当前 runtime workspace 预算是 64MiB 量级，用于控制单个临时面板。

适配新模型时必须检查：

- 最大 `logical_k` 是否导致 workspace 过大。
- merged Linear 的 padded output 是否显著大于 logical output。
- batch/prompt 较大时 `out_padded` 临时输出是否成为主要 transient。
- CUDA graph capture 是否额外保留 workspace 或 activation。

## 显存分析方法

不要只看整卡 `nvidia-smi` 总占用。正确流程是：

1. 启动前记录整卡和进程占用。
2. 启动服务后记录 `VLLM::EngineCore` 或 worker PID 的 `used_memory`。
3. 看 vLLM 日志中的 `Model loading took ... GiB memory`。
4. 看 `Available KV cache memory` 和 `GPU KV cache size`。
5. 跑 text/image/video 或 prefill/decode 请求后再次记录进程占用。
6. 停服务确认显存回落。

常用命令：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
  --format=csv,noheader,nounits | sort
ps -fp <pid>
```

如果模型进程本身只占 10GiB，但整卡显示 31GiB，通常是同卡已有其它服务。先清理目标 GPU，再判断模型显存。

显存构成通常包括：

- FP8/BF16/FP16 权重。
- 未量化模块，例如 vision tower、merger、`lm_head`。
- KV cache。embedding/pooling runner 也可能初始化 KV cache。
- encoder cache 和 multimodal profiling/warmup。
- CUDA context、NCCL、allocator reserve。
- torch.compile 和 CUDA graph capture 的 transient/reserved 内存。
- runtime decode workspace 和临时 `out_padded`。

## CUDA Graph 和 eager

开发阶段可以用 `--enforce-eager` 定位 shape、scale、语义和显存问题。

正式服务应验证 CUDA graph 配置，但要注意：

- generation runner 通常可请求 `FULL_AND_PIECEWISE`。
- pooling/embedding runner 当前会把 full cudagraph modes 降级为 `PIECEWISE`。
- graph capture 会增加启动耗时和少量显存。
- compile 阶段可能有 transient peak，不应只按 ready 后显存估算启动峰值。

推荐记录日志中的最终配置，而不是只记录启动参数：

```text
compilation_config.cudagraph_mode
Capturing CUDA graphs ...
Graph capturing finished ...
```

## 适配其它 FP8 模型的流程

### Step 1: 模型结构和量化配置审计

确认：

- architecture 是否已有 vLLM 模型类。
- dense Linear 类型是否走 vLLM 标准 LinearBase。
- 是否存在自定义 remote-code Linear。
- quantization config 是否为支持的 FP8 格式。
- `ignore` 列表里哪些模块未量化。
- 是否有 MoE、vision tower、audio encoder、cross-attention、multi-token heads 等非 dense text path。

输出一份预期：

```text
text dense FP8: yes/no
vision/audio/module ignored: yes/no
moe experts: yes/no
lm_head ignored: yes/no
expected resident weight GiB: ...
expected non-FP8 GiB: ...
```

### Step 2: scheme 路由

如果是 `compressed-tensors` FP8：

- 在 `compressed_tensors.py` 中确认 SM70 capability 走 `CompressedTensorsSM70Fp8`。
- 补单测确保 capability 70 不再触发 min capability 75/89 报错。

如果是 vLLM 原生 `fp8`：

- 确认 Linear method 选择 `Fp8SM70RuntimeDecodeLinearMethod`。
- merged Linear 需要覆盖 `logical_widths`。

如果是新 quantization provider：

- 不要在模型类里硬编码。
- 先抽象成 scheme/helper，再让模型复用。
- 保持“模型结构适配”和“SM70 FP8 dense primitive”分层。

### Step 3: 权重 prepare 红测

为新格式加最小测试：

- capability 70 下选择 SM70 FP8 scheme。
- `process_weights_after_loading()` 后 `layer.weight.dtype` 仍为 FP8。
- `layer.weight_scale` shape 符合 runtime op。
- `layer.input_scale is None` 或不参与 SM70 f16 GEMM。
- `_sm70_fp8_runtime_prepared` 为真。
- 没有生成全量 FP16 resident weight。

### Step 4: apply/GEMM 红测

mock custom op 验证：

- `apply()` 调到 `sm70_fp8_runtime_gemm_out()`。
- 输入 reshape 后 K 维正确。
- 输出按 logical width slice。
- bias 只加到 logical output。
- workspace cache 可复用，batch 增大不会因同一层反复分配全量 weight。

### Step 5: 真机加载冒烟

先用单卡、低并发、短上下文：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
vllm serve /path/to/model \
  --host 127.0.0.1 \
  --port <port> \
  --served-model-name <name> \
  --dtype float16 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --enforce-eager
```

对 VL/embedding 模型再限制：

```bash
--runner pooling \
--convert embed \
--limit-mm-per-prompt '{"image":1,"video":1}' \
--media-io-kwargs '{"video":{"num_frames":4}}'
```

验证日志必须出现：

```text
Resolved architecture: ...
quantization=compressed-tensors 或 quantization=fp8
Model loading took ... GiB memory
SM70 AWQ/FP8 warmup finished (... runtime-decode dense calls ...)
```

### Step 6: 功能和语义验证

generation 模型：

- 1k 输入 prefill。
- 32k 输入 prefill，如果目标支持长上下文。
- decode throughput。
- TTFT。
- `finish_reason`。
- 简单语义正确性。

embedding/VL 模型：

- `LLM.embed()` 离线。
- `/v1/embeddings` 在线。
- text/image/video/image+text/video+text。
- embedding dim。
- finite/non-zero norm。
- match similarity > mismatch similarity。

MoE 模型：

- dense path 先过。
- expert path 单独验证 routed experts。
- top-k、expert map、padding、workspace 不与 dense path 混淆。

### Step 7: graph 模式验证

eager 只用于开发验证。正式结论必须补 graph 模式：

```bash
--compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'
```

然后记录最终是否被降级：

- generation runner：期望保持 `FULL_AND_PIECEWISE` 或按 attention backend 合理 fallback。
- pooling runner：当前会降级到 `PIECEWISE`，这是 runner 级限制。

### Step 8: 性能和显存归因

输出至少包括：

- 模型权重 resident GiB。
- 进程 ready 后显存。
- 首次请求后显存。
- prefill tokens/s。
- decode tokens/s。
- TTFT。
- finish_reason。
- 语义质量结论。
- CUDA graph 最终模式。

如果性能明显低于预期，先区分：

- attention backend 是否是 FA2/FA_V100/TRITON。
- decode 是否被 Python/调度/小 batch 开销主导。
- FP8 runtime decode 是否每 token 重复 decode 大量 panel。
- direct path 是否适合该模型。
- 是否还有 BF16 未量化大模块。
- 是否 TP/PP 切分导致通信吞吐下降。

## 常见问题

### 模型名是 FP8，但显存不低

先检查 `ignore` 列表和 safetensors dtype。VL 模型经常只量化文本 Linear，视觉塔和 `lm_head` 仍是 BF16。

### 整卡显示 30GiB，但模型进程只有 10GiB

这是其它进程占用。用 `--query-compute-apps` 看 PID，不要只看整卡总量。

### capability 70 被 min capability 拦住

说明 scheme 没有路由到 SM70 FP8 path。应在 quantization config 解析处修路由，而不是在模型类里绕过检查。

### 输出维度不对

优先检查 merged/fused Linear 的 `logical_widths` 和 `output_size_per_partition`。SM70 path 可能有 padded output，必须 slice 回 logical width。

### 语义明显错误

优先检查：

- scale axis 是否和 weight layout 一致。
- channel scale 是否 squeeze 成 custom op 期望的一维。
- block scale row/col 是否和 `(N, K)` 对齐。
- prompt/template 是否符合模型要求。
- BF16 到 FP16 fallback 是否可接受。

### 请求后显存持续增长

区分正常 cache 和泄漏：

- prefix cache / MM cache 命中会保留数据。
- torch allocator reserved 不会立即还给 driver。
- 如果进程 `used_memory` 随相同请求无限增长，检查 workspace cache key 是否包含不稳定维度。

## 新模型适配完成标准

一个新的 SM70 FP8 模型适配完成，至少应满足：

- capability 70 下能选择 SM70 FP8 dense scheme。
- FP8 权重常驻，未全量 FP16 展开。
- runtime workspace 可复用。
- dense/fused/merged Linear 输出 shape 正确。
- 单卡 eager 能加载并完成语义请求。
- 正式 graph 配置能启动，并记录最终 graph mode。
- 显存归因清楚，能解释权重、KV、workspace、未量化模块和 graph 开销。
- 有针对该 quantization 格式的单测。
- 有至少一个真实 checkpoint 的在线或离线冒烟。

## 推荐最小验证矩阵

| 场景 | 必测项 |
| --- | --- |
| dense text generation | 1k prefill、短 decode、语义、finish_reason |
| long context generation | 32k prefill、TTFT、OOM 边界 |
| embedding | offline `LLM.embed()`、online `/v1/embeddings` |
| VL embedding | text/image/video/image+text/video+text |
| MoE FP8 | routed expert 正确性、top-k、专家显存、decode throughput |
| graph mode | eager 对照、最终 cudagraph mode、capture 显存 |
| memory | checkpoint dtype 统计、进程显存、请求后显存、释放后回落 |

