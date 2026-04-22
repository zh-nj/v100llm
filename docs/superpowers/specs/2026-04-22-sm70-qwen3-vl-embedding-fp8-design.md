# SM70 Qwen3-VL-Embedding FP8 Design

日期：2026-04-22

## 1. 目标

为 `sm70 / V100 / Tesla PG503-216` 增加对
`/mnt/data6/models/Qwen3-VL-Embedding-8B-FP8` 的单卡优先推理支持，使其在当前
vLLM 分支上满足以下条件：

- 单卡 `V100 32GB` 为主部署目标
- `text / image / video / image+text / video+text` embedding 都可用
- 同时支持离线 `LLM(..., runner="pooling")` 和在线 `vllm serve` 的
  `/v1/embeddings`
- 在线接口只支持 `messages[].content[]` 形态的多模态 embedding 请求
- 文本 backbone 的 FP8 权重在显存中保持压缩常驻态
- SM70 上只允许在 kernel 内或临时 workspace 中 runtime decode / pack，
  不允许把权重常驻展开成 FP16/BF16
- 复用当前分支已有的 SM70 FP8 dense linear 通用基础设施，而不是为
  `Qwen3-VL-Embedding-8B-FP8` 单独分叉一套实现

## 2. 非目标

本轮不做以下事情：

- 不新增专用的 `Qwen3VLForEmbedding` 大型模型分支实现
- 不把多模态 embedding 扩展到第二套在线协议，例如把图像或视频塞进 `input`
- 不实现 Cohere `/v2/embed` 的视频多模态兼容
- 不修改视觉塔量化策略，不把视觉 encoder 改成 SM70 FP8 路径
- 不优化 generate 路径、KV cache、spec decode、MoE 或 MTP
- 不把 `TP=2` 作为默认运行形态，只把它作为显卡拥挤时的验证兜底
- 不在第一轮追求高并发或极限长上下文吞吐，首要目标是单卡稳定可用和语义正确

## 3. 目标模型事实

目标模型 `config.json` 关键信息：

- `architectures = ["Qwen3VLForConditionalGeneration"]`
- `model_type = "qwen3_vl"`
- `dtype = "bfloat16"`
- `text_config.hidden_size = 4096`
- `text_config.num_hidden_layers = 36`
- `vision_config.depth = 27`
- `vision_config.hidden_size = 1152`
- `vision_config.out_hidden_size = 4096`
- `quantization_config.quant_method = "compressed-tensors"`
- `quantization_config.quantization_status = "compressed"`
- `config_groups.group_0.targets = ["Linear"]`
- `weights.num_bits = 8`, `weights.type = "float"`
- `input_activations.num_bits = 8`, `input_activations.dynamic = true`

量化 ignore 列表显示：

- 文本 dense linear 是 FP8 压缩权重
- `model.visual.blocks.*`、`model.visual.merger.*`、
  `model.visual.deepstack_merger_list.*` 明确在 ignore 列表中
- `lm_head` 也在 ignore 列表中

因此本轮的正确工程判断是：

- 文本 backbone 走 SM70 FP8 通用路径
- 视觉塔与视觉 merger 保持高精度
- 这个模型不是“整模型都要适配 FP8”，而是“Qwen3-VL 多模态 embedding 模型中，
  文本主干 FP8 在 SM70 上可推理”

## 4. 现状结论

### 4.1 当前分支的真实入口

当前分支不是旧式 `--task embed` 语义，而是：

- 运行 embedding 服务应使用 `--runner pooling`
- 对非 pooling 原生模型，`--convert auto` 会在 `runner="pooling"` 时默认解析为
  `embed`
- 运行时模型类由 `as_embedding_model(...)` 自动包装

这意味着首轮实现不应该围绕一个新的 CLI 或新任务入口展开，而应该直接复用当前的
 pooling runner。

### 4.2 当前分支的真实路由能力

当前代码已经具备以下基础能力：

- `/v1/embeddings` 已支持 `EmbeddingChatRequest`
- `EmbeddingChatRequest` 直接走 `messages[].content[]`
- `chat_utils.py` 已支持 `image_url` 与 `video_url`
- `examples/pooling/embed/vision_embedding_online.py` 中已经存在
  `run_qwen3_vl()`，说明 `Qwen3-VL` 的在线 embedding 请求格式已经有现成模式

因此首轮不是要“发明”多模态 embedding 接口，而是要把
 `Qwen3-VL-Embedding-8B-FP8` 在该接口上跑通，并补齐视频覆盖与 SM70 验证。

### 4.3 真实 bring-up 探测结果

对本地模型做了无代码改动的启动探测，命令形态为：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split \
/home/z/anaconda3/envs/gptq/bin/vllm serve \
  /mnt/data6/models/Qwen3-VL-Embedding-8B-FP8 \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --dtype float16 \
  --limit-mm-per-prompt '{"image":2,"video":1}' \
  --enforce-eager
```

探测结果：

- 服务入口已正确解析到 `Resolved architecture: Qwen3VLForConditionalGeneration`
- pooling/embed 适配链路可达
- 失败点不是模型架构不支持，而是所选 GPU 当时空闲显存不足

这说明首轮方案不需要优先解决“模型无法转换为 embedding”的问题，而是要解决：

- 单卡显存预算
- `Qwen3-VL` 多模态 embedding 请求的单卡默认配置
- SM70 FP8 通用路径在该模型上的稳定落地

## 5. 方案比较

### 方案 A：复用 `Qwen3VLForConditionalGeneration` + pooling/embed 自动包装

做法：

- 保持模型主类为 `Qwen3VLForConditionalGeneration`
- 服务使用 `--runner pooling`
- 通过 `--convert auto` 或显式 `--convert embed` 进入 embedding 模式
- 复用现有 `qwen3_vl.py` 多模态处理与当前分支的 SM70 FP8 dense linear 路径

优点：

- 改动最小
- 直接继承 `Qwen3-VL` 的 text/image/video processor 能力
- 复用当前 `/v1/embeddings + messages[]` 在线协议
- 不引入新的模型分叉，后续别的 `Qwen3-VL-*FP8` 可直接复用
- 更符合“做可复用底层方案”的要求

缺点：

- 需要补测试与示例，确保 `Qwen3-VL-Embedding` 的 prompt 组织方式被固定
- 如果后续发现该模型存在特殊 embedding 语义，仍可能需要增加薄层适配

结论：采用。

### 方案 B：新增专用 `Qwen3VLForEmbedding`

做法：

- 参照 `LlamaNemotronVLForEmbedding`，新建一个专用 embedding 模型类
- 单独控制 pooler、processor 或权重映射

优点：

- 行为显式
- 如果 checkpoint 有特别的 embedding 头，可以单独处理

缺点：

- 与 `qwen3_vl.py` 重复度高
- 后续 upstream 演进容易漂移
- 对“所有 FP8 Qwen3-VL 模型可复用”帮助较小

结论：不作为第一选择，只在发现自动包装路线有结构性缺陷时再考虑。

### 方案 C：Transformers fallback / remote-code 专用实现

优点：

- 可能更快 bring-up

缺点：

- 容易绕开当前分支的 SM70 FP8 主路径
- 运行性能和长期维护都差
- 与“通用 SM70 FP8 基础设施”目标相违背

结论：不采用。

## 6. 推荐架构

首轮推荐架构如下：

### 6.1 模型层

- 继续使用 `Qwen3VLForConditionalGeneration`
- 通过 `runner="pooling"` + `convert="embed"` 自动包装为 embedding 模型
- 不新建专用模型大类
- 只有在 bring-up 发现 `qwen3_vl.py` 对 embedding/pooling 有真实缺口时，
  才补一个薄层修正，而不是重写模型

### 6.2 量化层

- 文本 backbone 中被量化为 FP8 的 dense linear 继续复用当前分支 SM70 通用路径
- 保持 `FP8 常驻 + runtime 临时 decode/pack + 复用 SM70 f16 GEMM`
- 不允许把权重常驻解压为 FP16/BF16
- 如果 `Qwen3-VL` 文本主干里出现当前通用路径尚未覆盖的 merged/fused 线性层，
  修复点必须放在通用 SM70 FP8 基础设施，而不是写 `qwen3_vl` 特判

### 6.3 多模态层

- image / video 继续走 `Qwen3-VL` 现有 processor
- 视觉 encoder、视觉 merger 保持高精度
- 多模态输入协议统一为：
  - `messages[].content[].type = text`
  - `messages[].content[].type = image_url`
  - `messages[].content[].type = video_url`

### 6.4 在线服务层

- 只支持 `/v1/embeddings`
- 只支持 chat-style `messages`
- 不扩展第二套 `input` 多模态协议
- 首轮只要求 OpenAI 形态的 embedding API 可用，不扩展 Cohere 视频路径

## 7. 单卡运行策略

### 7.1 默认部署画像

默认部署目标是：

- `1x V100 32GB`
- `--runner pooling`
- `--convert embed`
- 单请求为主，低并发优先

### 7.2 单卡标准档

单卡标准档作为首轮默认建议，目标是先稳定可用：

- `max_model_len = 8192`
- `max_num_seqs = 1`
- `limit_mm_per_prompt = {"image": 2, "video": 1}`
- `media_io_kwargs = {"video": {"num_frames": 8}}`
- `gpu_memory_utilization = 0.75 ~ 0.80`
- 首轮验证优先 `enforce_eager=True` 做 bring-up，再收敛到图模式

理由：

- embedding 服务不需要 generate 的长 decode cache
- 对单卡 V100 来说，视频帧数和多模态 token 展开是最大显存风险
- `8192 + 8 frames + max_num_seqs=1` 是更稳的起点

### 7.3 单卡高载档

单卡高载档只在显卡空闲充分时启用，不作为首轮默认值：

- 允许提高 `max_model_len`
- 允许提高 `video.num_frames`
- 允许放大 `max_num_seqs`

高载档的数值在实现验证后再固定到 README / 示例命令，不作为当前设计文档默认承诺。

### 7.4 TP=2 的角色

- `TP=2` 不作为推荐部署
- 只在单卡显卡被其他进程占用、无法完成功能验证时临时使用
- 所有对外结论以单卡配置为准

## 8. 接口设计

### 8.1 在线接口

请求路径：

```text
POST /v1/embeddings
```

请求体主形态：

```json
{
  "model": "qwen3-vl-embed-8b-fp8-sm70",
  "messages": [
    {
      "role": "system",
      "content": [
        {"type": "text", "text": "Represent the user's input."}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "A cat standing in the snow."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": ""}
      ]
    }
  ],
  "encoding_format": "float",
  "continue_final_message": true,
  "add_special_tokens": true
}
```

说明：

- `Qwen3-VL-Embedding` 的 embedding 质量依赖 chat-style 组织方式
- 首轮示例和测试会固定使用 system/user/assistant 三段式模板
- text-only、image-only、video-only 也统一走这套结构

### 8.2 离线接口

离线接口继续使用：

```python
LLM(model=..., runner="pooling")
```

以及：

```python
llm.embed([...])
```

离线输入同样以 chat-style prompt / 多模态数据为主，不额外发明独立协议。

## 9. 代码边界

预计首轮只在以下区域落地：

- `tests/test_config.py`
  - 锁定 `Qwen3-VL` 在 `runner="pooling"` 下解析到 embedding 适配路径
- `tests/entrypoints/pooling/embed/`
  - 新增 `Qwen3-VL-Embedding-8B-FP8` 的 text/image/video 在线测试
- `examples/pooling/embed/vision_embedding_online.py`
  - 扩充 `run_qwen3_vl()` 为 text/image/video/image+text/video+text 示例
- 必要时：
  - `vllm/model_executor/models/qwen3_vl.py`
  - 或当前分支已有的 SM70 FP8 通用基础设施

不应在首轮触碰以下区域：

- 新增一整套 `qwen3_vl_embedding.py`
- 新增模型私有 CUDA kernel 分支
- 改写通用 OpenAI embedding API 协议

## 10. 验证矩阵

### 10.1 配置与模型选择

验证内容：

- `runner="pooling"` 时，`Qwen3VLForConditionalGeneration` 可被正确包装为 embedding 模型
- `convert="auto"` 不会退化成错误路径

### 10.2 离线 embedding

至少覆盖：

- text
- image
- video
- image + text
- video + text

断言：

- 输出向量维度为 `4096`
- 向量中无 `NaN` / `Inf`
- 范数非零
- 同一图像加上不同文本后向量会变化
- 简单匹配样本相似度高于明显不匹配样本

### 10.3 在线 `/v1/embeddings`

至少覆盖：

- `messages` text-only
- `messages` image-only
- `messages` video-only
- `messages` image+text
- `messages` video+text

断言：

- `EmbeddingResponse.data[0].embedding` 长度为 `4096`
- `usage.prompt_tokens` 合理非零
- 服务返回成功，无协议校验错误

### 10.4 单卡实机验证

使用本地模型：

- `/mnt/data6/models/Qwen3-VL-Embedding-8B-FP8`

验证项目：

- 单卡服务可启动
- `/v1/models` 可见目标 model id
- 5 类 embedding 请求都能返回
- 语义质量基本正确

如果实机环境当时显卡被占满，允许临时改用 `TP=2` 做功能验证，但最终需要回到单卡结论。

## 11. 风险

### 11.1 单卡显存风险

首要风险不是 FP8 权重本体，而是：

- 视频帧采样
- 长上下文
- 并发批量

因此首轮默认配置必须保守。

### 11.2 Prompt 组织风险

`Qwen3-VL-Embedding` 不是“任何输入文本都直接拿最后一层就行”的模型。
如果在线示例和测试没有固定它期望的三段式 chat 结构，可能出现：

- 请求成功
- 向量维度正常
- 但语义质量漂移

因此示例和测试必须固定 prompt 组织方式。

### 11.3 通用路径覆盖风险

如果 `Qwen3-VL` 文本主干中存在当前 SM70 FP8 通用路径未覆盖的 merged/fused linear，
bring-up 会卡在量化层。
此时修复必须回到通用 SM70 FP8 基础设施，而不是在模型文件中写特殊分支。

### 11.4 视频在线测试空缺风险

当前仓内已有视频 chat/generation 测试，但 embedding 视频 coverage 较少。
这不是协议层问题，而是测试覆盖空洞，首轮需要主动补齐。

## 12. 后续阶段划分

第一阶段：

- 跑通单卡 text/image/video embedding
- 补在线与离线测试
- 固定单卡默认运行参数

第二阶段：

- 收敛 CUDA graph 配置
- 进一步扩大视频帧数、上下文长度和并发窗口
- 评估是否需要补额外的通用 SM70 FP8 fused/merged linear 支持

第三阶段：

- 把这套路径外推到其他 `Qwen3-VL-*FP8` 模型
- 视需要再扩展非主协议或更高载配置
