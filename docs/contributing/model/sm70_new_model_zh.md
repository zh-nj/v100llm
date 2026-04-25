# SM70 / V100 新模型接入指南

这份文档用于在 `1Cat-vLLM` 中评估和接入新的 `SM70 / Tesla V100` 模型。它补充通用的 [Basic Model](basic.md)、[Registering a Model](registration.md) 和 [Unit Testing](tests.md) 指南，重点覆盖 V100 上容易出问题的环节：AWQ/ MoE 量化分派、TurboMind SM70 kernel、attention backend、KV cache、构建产物和实机 smoke test。

## 适用范围

优先按这份指南处理以下模型：

- 需要在 `Tesla V100 / SM70` 上运行的模型。
- AWQ 4-bit dense 模型。
- AWQ 4-bit MoE 模型。
- 使用 compressed-tensors WNA16 MoE、且希望复用 SM70 AWQ kernel 的模型。
- 需要在 V100 上确认 `TRITON_ATTN`、`FLASH_ATTN_V100`、MLA、GDN 或 Mamba-like attention 行为的模型。

不建议把这份指南理解成“所有量化格式都能在 V100 上跑”。GPTQ/AWQ Marlin、FP8/FP4、DeepGEMM、FlashInfer 相关路径通常带有 SM75、SM80 或 SM90 假设，新增模型时必须逐项验证。

## 总体流程

新增 SM70 模型时，按下面顺序推进：

1. 收集模型配置和权重格式。
2. 判断模型是否可以复用 Transformers backend、现有 vLLM 模型，或必须新增原生模型实现。
3. 计算 tensor parallel 后的 dense linear 和 MoE expert 维度，确认是否满足 SM70 kernel 约束。
4. 确认量化分派会进入 V100 可执行路径。
5. 确认 attention backend、KV cache dtype 和 serve 参数。
6. 确认当前 wheel 或源码构建包含 `sm_70` 和 TurboMind SM70 AWQ sources。
7. 补单元测试和 V100 实机 smoke test。
8. 把目标模型的已验证启动参数写入部署文档或 README。

## 1. 模型信息预检

先从目标模型目录或 Hugging Face repo 中收集这些文件：

- `config.json`
- `generation_config.json`
- tokenizer 文件和 chat template
- `quant_config.json`、`quantize_config.json`、`compression_config` 或 `quantization_config`
- safetensors index 和权重文件

重点字段：

| 类别 | 需要确认的字段 |
| --- | --- |
| 架构 | `architectures`、`model_type`、是否已有 vLLM 实现 |
| dense 尺寸 | `hidden_size`、`intermediate_size`、`num_hidden_layers` |
| attention | `num_attention_heads`、`num_key_value_heads`、`head_dim`、rope、sliding window、MLA/GDN/Mamba |
| MoE | `num_experts`、`num_experts_per_tok`、`top_k`、expert intermediate size |
| 量化 | bits、group size、zero point、symmetric/asymmetric、target modules、ignored modules |
| 多模态 | vision/video processor、projector、`limit-mm-per-prompt` 需求 |

如果模型只是 architecture alias 不同，但结构与现有模型一致，优先走最小接入：注册 architecture 或补 config convertor。只有在 attention、MLP、MoE、norm、position embedding 或权重布局不同的时候，才新增模型文件。

## 2. 架构接入路径选择

### 路径 A：Transformers backend 可直接加载

先用最小参数尝试：

```bash
vllm serve /path/to/model --dtype float16
```

如果能走通，再检查 V100 量化和 runtime 参数。这个路径通常不需要新增 `vllm/model_executor/models/*.py`。

### 路径 B：复用现有 vLLM 模型

适用于模型结构与现有实现一致，但 `architectures` 名称不同的情况。通常需要改：

- [`vllm/model_executor/models/registry.py`](../../../vllm/model_executor/models/registry.py)
- [`tests/models/registry.py`](../../../tests/models/registry.py)
- supported model 文档

### 路径 C：新增原生 vLLM 模型

如果必须新增模型文件，应遵守通用模型接入规则：

- 所有 vLLM module 构造时带 `prefix`。
- 使用 `QKVParallelLinear`、`MergedColumnParallelLinear`、`RowParallelLinear`、`ColumnParallelLinear` 等 TP-aware layer。
- `load_weights()` 正确处理 QKV 合并、gate/up 合并、expert weights、TP shard。
- 量化层必须能通过 `prefix` 匹配 `modules_to_not_convert` 或 compression target。

SM70 场景里，`prefix` 不只是命名问题。它会影响部分层跳过量化、MoE expert 是否进入 `AWQSM70MoEMethod`，以及 mixed quant 模型能否正确加载。

## 3. SM70 量化路径检查

### 3.1 Dense AWQ

本分支把 AWQ 最低算力放宽到 SM70，普通 dense AWQ linear 在加载后会提前转换成 TurboMind 格式，forward 时调用 SM70 GEMM：

- [`AWQConfig.get_min_capability()`](../../../vllm/model_executor/layers/quantization/awq.py)
- `AWQLinearMethod.process_weights_after_loading()`
- `ops.awq_sm70_prepare`
- `ops.awq_gemm_sm70`

新增模型需要确认：

- AWQ bits 是 `4`。
- activation dtype 是 `float16`。
- group size 是 `32`、`64` 或 `128`。
- dense linear 的 input/output shard 后仍满足 pack factor 和 group size 对齐。
- `modules_to_not_convert` 没有误跳过本应量化的 dense layer。

如果 dense linear 的 `input_size_per_partition % group_size != 0`，当前 AWQ weight 创建会报错。这通常说明 TP size 过大、group size 与 shard 不匹配，或模型权重布局需要特殊 mapper。

### 3.2 AWQ MoE

MoE AWQ 是 SM70 新模型最容易出错的部分。当前 `AWQConfig.get_quant_method()` 在 `FusedMoE` 分支会先处理 skip 规则，再判断是否为 SM70。若是 V100，会尝试使用 [`AWQSM70MoEMethod`](../../../vllm/model_executor/layers/quantization/awq_sm70_moe.py)。

进入 SM70 MoE AWQ 的基本条件：

- checkpoint 是 AWQ 4-bit。
- `group_size` 可适配为 `32`、`64` 或 `128`。
- `hidden_dim % 8 == 0`。
- `intermediate_size_per_partition % 8 == 0`。
- `hidden_dim` 和 `intermediate_size_per_partition` 能被有效 group size 整除。
- scale dtype 能以 `float16` 加载。

本分支支持一种常见适配：checkpoint group size 是 `128`，但 TP 分片后的 expert intermediate size 只适合 `64` 或 `32`。此时会用较小的 effective group size，并重复 checkpoint scales/zero-points。相关逻辑在 `AWQConfig._get_sm70_moe_group_size()` 和 `AWQSM70MoEMethod` 的 weight loader 中。

新增 MoE 模型时，必须逐层确认这些点：

- `FusedMoE` prefix 是否与 checkpoint 的 ignored modules 匹配。
- layer 级别 skip，例如 `model.layers.0.`，是否在进入 SM70 method 前生效。
- `w13` 和 `w2` 的 expert 权重形状是否与 vLLM 的 `FusedMoE` 约定一致。
- `num_experts`、`top_k`、routing weights 是否与现有 fused MoE kernel 支持范围一致。
- 不兼容层是否 fallback 到 `MoeWNA16` 或 unquantized method，而不是落到 Marlin。

### 3.3 Compressed-tensors WNA16 MoE

部分 compressed-tensors MoE 可以复用 SM70 AWQ MoE kernel。选择逻辑在 [`compressed_tensors_moe.py`](../../../vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py)。

需要满足：

- 当前 device capability 是 `(7, 0)`。
- 4-bit WNA16。
- symmetric quantization。
- group size 是 `32`、`64` 或 `128`。
- hidden 和 intermediate shard 满足 8 对齐与 group size 对齐。

如果这些条件不满足，应该接受显式 fallback，并为 fallback 行为写单测。

### 3.4 不应默认启用的路径

新增 V100 模型时，不要未经验证就引入这些路径：

- AWQ Marlin 或 GPTQ Marlin：通常要求 SM75+。
- FP8/FP4 MoE：很多 kernel 面向 Ampere/Hopper/Blackwell。
- DeepGEMM：通常不是 V100 目标。
- FlashInfer MoE/attention：需要单独确认 SM70 支持。

## 4. Attention 和 KV cache

V100 上的公开默认建议是显式使用：

```bash
--attention-backend TRITON_ATTN
```

CUDA platform 对 SM70 的自动优先级是 `FLASH_ATTN`、`FLASH_ATTN_V100`、`TRITON_ATTN`，但生产和复现文档应优先固定 `TRITON_ATTN`，避免 optional flash 模块或特性 fallback 造成结果不稳定。相关代码在 [`vllm/platforms/cuda.py`](../../../vllm/platforms/cuda.py) 和 [`vllm/v1/attention/backends/registry.py`](../../../vllm/v1/attention/backends/registry.py)。

新增模型需要确认：

- head dim 是否被选中的 backend 支持。
- sliding window、prefix/chunked prefill、MLA、GDN、Mamba-like attention 是否触发专门 backend。
- 多模态 encoder attention 是否单独选择 backend。
- CUDA graph capture size 是否与 decode 路径匹配。
- KV cache dtype 是否适合 V100。

当前 V100 Triton 路径下：

- `fp8_e4m3` KV cache 不作为可用选项。
- `fp8_e5m2` 只能作为实验选项，不要和 `--calculate-kv-scales` 混用。
- 默认优先用 fp16 KV cache 证明模型可用，再做 KV cache 压缩实验。

## 5. 构建产物检查

SM70 AWQ 不只是 Python 分派。当前源码构建必须把 TurboMind SM70 kernel 编进去。

检查点：

- `lmdeploy/src/turbomind` 存在。
- `CUDA_ARCHS` 或构建配置包含 `7.0`。
- [`CMakeLists.txt`](../../../CMakeLists.txt) 中的 TurboMind SM70 AWQ source 被启用。
- [`csrc/torch_bindings.cpp`](../../../csrc/torch_bindings.cpp) 注册了 `awq_sm70_prepare`、`awq_gemm_sm70`、`awq_moe_gemm_sm70` 等 op。
- 构建出的 wheel 与部署环境 CUDA/Python/glibc 兼容。

如果新增模型不需要新 kernel，尽量不要改 `csrc/`。如果确实需要新 op，必须同步完成：

- CMake source 列表。
- CUDA arch 7.0 gencode。
- torch binding schema。
- Python 调用封装。
- CPU-side unit test 和 V100 smoke test。

## 6. 运行参数基线

V100 上新增模型先用保守参数跑通，再逐步放宽上下文和并发。

推荐基线：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --skip-mm-profiling \
  --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000
```

根据显存和模型大小调整：

| 场景 | 建议 |
| --- | --- |
| 双卡 16GB V100 | 优先 TP2，`max-num-seqs` 从 1 或 2 开始 |
| 四卡 16GB V100 | 优先 TP4，先确认 MoE shard 维度是否仍满足 group size |
| 长上下文 | 降低 `max-num-seqs`，先验证 startup + first request |
| 多模态 | 先关闭 vision/video，文本链路稳定后再打开 |
| 首请求很慢 | V100 首请求可能花 1 到 3 分钟编译 kernel 和建图，不代表稳定速度 |

已验证的 Qwen3.5 V100 参考参数见仓库根目录 [`README.md`](../../../README.md#public-runtime-defaults-for-v100-16-gb-reference-systems)。

## 7. 测试策略

### 7.1 静态和 CPU-side 单测

新增或扩展单测，至少覆盖：

- registry 能识别新 architecture。
- dummy weights 能初始化模型。
- `modules_to_not_convert` 映射到 vLLM prefix 后仍正确。
- SM70 mock capability 下，dense AWQ 返回预期 method。
- SM70 mock capability 下，MoE AWQ 兼容层返回 `AWQSM70MoEMethod`。
- skip 层返回 `UnquantizedFusedMoEMethod`。
- 维度不兼容层 fallback 到预期 method。
- checkpoint group size 与 effective group size 不一致时，scale/zero-point 重复逻辑正确。

可以参考：

- [`tests/quantization/test_awq_sm70.py`](../../../tests/quantization/test_awq_sm70.py)
- [`tests/quantization/test_awq_sm70_compat.py`](../../../tests/quantization/test_awq_sm70_compat.py)
- [`tests/quantization/test_minimax_m2_awq_sm70.py`](../../../tests/quantization/test_minimax_m2_awq_sm70.py)
- [`tests/v1/attention/test_flash_attn_sm70.py`](../../../tests/v1/attention/test_flash_attn_sm70.py)
- [`tests/v1/attention/test_triton_attn_sm70_tuning.py`](../../../tests/v1/attention/test_triton_attn_sm70_tuning.py)

### 7.2 V100 实机 smoke test

实机 smoke 至少包括：

1. `python -m vllm.entrypoints.openai.api_server ...` 可以完成启动。
2. `/v1/models` 返回目标 served model。
3. 一个短 prompt 可以生成。
4. 一个较长 prompt 可以完成 prefill。
5. 并发 2 到 4 个短请求不崩溃。
6. `nvidia-smi` 显示显存没有持续增长。
7. 日志中没有意外 fallback 到不支持的 Marlin/FlashInfer 路径。
8. 对 MoE 模型，确认兼容 experts 进入 `AWQSM70MoEMethod`，跳过层保持 unquantized 或预期 fallback。

### 7.3 正确性和性能

smoke 通过后再做：

- 与 HF/Transformers 的短样本 logprobs 或文本输出对比。
- 长上下文启动和首请求验证。
- warmup 后吞吐和 decode speed benchmark。
- TP2/TP4 对比。
- 不同 `max-num-seqs` 和 `max-num-batched-tokens` 组合对比。

不要用首个真实请求的耗时评价 V100 稳态性能。

## 8. 常见失败模式

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| AWQ weight shape 不对齐 | TP shard 后不能被 group size 或 pack factor 整除 | 降低 TP、调整 effective group size、或增加模型专属 mapper |
| MoE layer 误进入 SM70 AWQ | `modules_to_not_convert` 没映射到 vLLM prefix | 修正 mapper 或 prefix，补 skip 单测 |
| fallback 到 Marlin 后报错 | SM70 不支持该 Marlin 路径 | 在 SM70 分支显式选择 `AWQSM70MoEMethod`、`MoeWNA16` 或 unquantized |
| 启动找不到 `awq_sm70_prepare` | wheel 没编进 SM70 op | 重新构建，确认 `CUDA_ARCHS=7.0` 和 TurboMind sources |
| 首请求超慢 | 编译 kernel、建图、调优缓存 | 预热后再测性能，必要时导入/导出 SM70 GEMM cache |
| `fp8_e4m3` KV cache 出错 | V100 Triton 路径当前不支持 | 改用 fp16 KV cache 或实验性 `fp8_e5m2` |
| 多模态显存不足 | vision/video processor 和 KV cache 抢显存 | 先文本模式，设置 `--skip-mm-profiling` 和 `--limit-mm-per-prompt` |

## 9. 交付清单

新增 SM70 模型完成前，至少留下这些信息：

- 模型路径或 HF repo id。
- 使用的接入路径：Transformers backend、registry alias、或原生 vLLM model。
- 量化类型和实际 method：dense AWQ、AWQ SM70 MoE、compressed-tensors SM70 MoE、fallback。
- TP size、`max-model-len`、`max-num-seqs`、`max-num-batched-tokens`。
- attention backend 和 KV cache dtype。
- 通过的单测列表。
- V100 实机 smoke 命令和结果。
- 已知限制，例如不支持 vision、只验证 text-only、长上下文只验证 startup + first request。

## 10. 最小接入检查表

```text
[ ] config/tokenizer/quantization 配置已收集
[ ] 已判断 Transformers backend / registry alias / 原生模型路径
[ ] TP 后 dense linear 维度满足 AWQ 对齐
[ ] TP 后 MoE expert 维度满足 SM70 group size 和 8 对齐
[ ] modules_to_not_convert 已映射到 vLLM prefix
[ ] dense AWQ 在 SM70 上会调用 awq_sm70_prepare / awq_gemm_sm70
[ ] MoE AWQ 兼容层会进入 AWQSM70MoEMethod
[ ] 不兼容层有明确 fallback
[ ] attention backend 固定为 TRITON_ATTN 或有明确验证依据
[ ] KV cache dtype 未使用 V100 不支持的 fp8_e4m3
[ ] wheel/source build 包含 sm_70 和 TurboMind SM70 sources
[ ] 单元测试覆盖量化分派和 skip 规则
[ ] V100 实机 smoke 通过
[ ] 部署参数写入文档
```
