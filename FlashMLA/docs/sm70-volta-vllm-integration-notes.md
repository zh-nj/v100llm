# FlashMLA SM70 vLLM DeepSeek V4 集成说明

## 静态检查快照

- 日期：2026-04-29
- FlashMLA 工作区：`/mnt/data/apps/FlashMLA`
- vLLM 目标源码：`/mnt/data/apps/vllm`
- vLLM 分支：`woosuk/dsv4-sync`
- vLLM 工作区提示：存在未跟踪文件 `vllm/v1/attention/backends/volta_fa2.py`，本次只读分析未修改 vLLM。

## 当前结论

当前 `/mnt/data/apps/vllm` 已有 DeepSeek V4 主模型入口，`vllm/model_executor/models/registry.py` 注册了 `DeepseekV4ForCausalLM`。DeepSeek V4 MLA 运行路径位于 `vllm/model_executor/layers/deepseek_v4_attention.py`，并强制选择 `DeepseekV4FlashMLASparseBackend`，KV cache dtype 会转成 `fp8_ds_mla`。

这条路径和 FlashMLA SM70 sparse alpha 的 decode 侧接口基本对齐：vLLM decode 调用 `flash_mla_with_kvcache`，传入 `is_fp8_kvcache=True`、SWA `indices/topk_length`、`attn_sink`，并在 C4A/C128A 层把压缩 KV 作为 `extra_k_cache`、`extra_indices_in_kvcache`、`extra_topk_length` 传给 FlashMLA。DeepSeek V4 cache layout 是 MODEL1 风格，每 token `584B`，语义 `D_qk=512`、`D_v=512`、`h_kv=1`。

当前端到端 SM70 仍有明确外部阻断：

1. vLLM 选择门还没有放开 SM70。`DeepseekV4FlashMLASparseBackend` 继承的 `supports_compute_capability` 当前只允许 `major in [9, 10]`，`vllm/v1/attention/ops/flashmla.py` 的 `is_flashmla_sparse_supported()` 也只接受 `is_device_capability_family(90)` 或 `is_device_capability_family(100)`。

DeepSeek V4 prefill 路径会调用 `flash_mla_sparse_fwd`。FlashMLA 侧已在 2026-04-29 增加 SM70 BF16 SIMT fast path，并把 BF16 KV staged 到 FP16 shared K/V tile，因此这个 API 边界不再是 FlashMLA 侧硬阻断；它仍不是最终 MMA_884 性能路径。

## vLLM SM70 gate 放开任务

`benchmark/bench_vllm_deepseek_v4_flash_sm70.py --mode inspect --json` 现在输出 `sm70_blockers` 与 `end_to_end_smoke_ready`。当前 `/mnt/data/apps/vllm` 静态检查中，DeepSeek V4 FlashMLA 调用链存在，但 `end_to_end_smoke_ready=false`，需要先拆解以下 vLLM 侧任务：

1. `vllm_flashmla_sparse_backend_sm70_gate`：放开 `DeepseekV4FlashMLASparseBackend.supports_compute_capability`，允许满足 FlashMLA SM70 sparse decode/prefill 支持矩阵的 SM70 设备选择该 backend。
2. `vllm_flashmla_sparse_runtime_sm70_gate`：放开 `vllm/v1/attention/ops/flashmla.py` 的 `is_flashmla_sparse_supported()`，让运行时 probe 能识别 SM70，同时对未支持 dtype/layout 保持清晰错误。

在这两项完成前，OpenAI streaming smoke 不能命中 SM70 FlashMLA sparse backend；这属于 vLLM 集成 gate，不应归因于 FlashMLA kernel。

## vLLM 调用路径

### Model / Backend

- `vllm/model_executor/models/registry.py`
  - 注册 `DeepseekV4ForCausalLM`。
- `vllm/model_executor/layers/deepseek_v4_attention.py`
  - `DeepseekV4MLAAttention.get_attn_backend()` 返回 `DeepseekV4FlashMLASparseBackend`。
  - `get_kv_cache_spec()` 返回 `MLAAttentionSpec`，关键字段为 `num_kv_heads=1`、`head_size=self.head_dim`、`dtype=torch.uint8`、`cache_dtype_str="fp8_ds_mla"`、`alignment=576`、`model_version="deepseek_v4"`。
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
  - `DeepseekV4FlashMLASparseBackend.get_kv_cache_shape()` 在 `fp8_ds_mla` 下返回 `(num_blocks, block_size, 584)`。

### Decode

`DeepseekV4MLAAttention._forward_decode()` 的 FlashMLA 调用是：

```python
flash_mla_with_kvcache(
    q=q.unsqueeze(1),
    k_cache=swa_cache,
    block_table=None,
    head_dim_v=512,
    tile_scheduler_metadata=tile_metadata,
    cache_seqlens=None,
    is_fp8_kvcache=True,
    indices=swa_indices,
    topk_length=swa_lens,
    softmax_scale=self.scale,
    attn_sink=self.attn_sink,
    extra_k_cache=kv_cache if not swa_only else None,
    extra_indices_in_kvcache=topk_indices,
    extra_topk_length=topk_lens,
    out=output.unsqueeze(1),
)
```

这对应 FlashMLA SM70 sparse alpha 已覆盖的能力：MODEL1 FP8 sparse layout、`H_q=64/128`、`topk_length`、`extra_kv`、`extra_topk_length`、`attn_sink`。

### Prefill

`DeepseekV4MLAAttention._forward_prefill()` 会先将压缩 KV 与 SWA KV gather/dequant 到 BF16 workspace，再调用：

```python
flash_mla_sparse_fwd(
    q=q[query_start:query_end],
    kv=kv.view(-1, 1, q.shape[-1]),
    indices=combined_indices.unsqueeze(1),
    sm_scale=self.scale,
    attn_sink=self.attn_sink,
    topk_length=combined_lens,
    out=output[query_start:query_end],
)
```

这不是 SM70 sparse decode alpha；它走 FlashMLA sparse prefill API。当前 FlashMLA SM70 的 `csrc/api/sparse_fwd.h` 会分派到 `Fwd_Sm70_Bf16_Impl`，进入 `csrc/sm70/prefill/sparse/*` 的 BF16 SIMT fast path。该 path 支持 `D_qk in {512,576}`、`D_v=512`、`H_q in {64,128}`、`topk <= 8192`、`topk_length` 和 `attn_sink`，并使用默认 `FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE=32` 的 FP16 shared K/V staging。

## Fallback 评估

如果目标 vLLM 分支必须在 SM70 上跑 DeepSeek V4 prompt prefill，优先级建议如下：

1. FlashMLA correctness-first sparse prefill：已在 FlashMLA 内新增 SM70 sparse prefill SIMT fast path，只承诺正确性和可诊断性能指标。它让 vLLM 调用点保持不变。
2. vLLM 侧 fallback：如果后续发现 FlashMLA prefill SIMT path 对真实 prompt 太慢，prefill 阶段可以继续使用 vLLM 已有 gather/dequant workspace，并接一个不依赖 FlashMLA sparse prefill 的 BF16/SIMT attention fallback；decode 阶段命中 FlashMLA SM70 sparse alpha。
3. Decode-only smoke：构造无 prefill 或预填充 KV 的内部测试路径，只验证 `flash_mla_with_kvcache` 命中 SM70 sparse alpha。它不能代表真实 prompt 端到端可用。

不建议在 FlashMLA 内把 sparse 输入自动降成 dense，因为 sparse API 只有索引和量化 KV，不能恢复完整 dense KV 语义。

## 验证入口

静态检查：

```bash
/home/z/anaconda3/envs/gptq/bin/python benchmark/bench_vllm_deepseek_v4_flash_sm70.py \
  --mode inspect \
  --vllm-root /mnt/data/apps/vllm
```

OpenAI 兼容流式 benchmark 入口需要先由目标 vLLM 分支启动服务，并确保 vLLM 已能选择 SM70 FlashMLA sparse backend：

```bash
/home/z/anaconda3/envs/gptq/bin/python benchmark/bench_vllm_deepseek_v4_flash_sm70.py \
  --mode openai-stream \
  --endpoint http://127.0.0.1:8000/v1 \
  --model /path/to/deepseek-v4-model \
  --prompt "Hello" \
  --max-tokens 32
```

该入口记录 `TTFT`、`decode_tokens_per_s`、`finish_reason`、输出 token 计数，以及静态 gate 检查结果。真实 smoke 报告还需要附上 vLLM 启动命令、模型路径、GPU mask、FlashMLA kernel 命中日志和首个失败点或成功输出。
