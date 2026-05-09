# DeepSeek V4 Flash Prefill/Decode 实验分析与效率审计

日期：2026-05-05
工作区：`/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split`
分支：`feature/vllm-0190-upstream-split`
模型：`/mnt/data6/models/DeepSeek-V4-Flash`
FlashMLA 源码：`/mnt/data/apps/FlashMLA`

## 1. 审计目标

本分析围绕 DeepSeek V4 Flash 在 SM70/V100 上的真实推理路径，拆解 prefill 和 decode 两条路径的流程环节、函数调用、数据组织和计算方法，目标是形成可以执行的效率审计框架，并给出后续优化方向。

本轮审计不只看符号是否存在，而是按“模型入口 -> attention wrapper -> KV/cache 写入 -> sparse index -> FlashMLA attention -> O projection -> MoE/mHC”的执行顺序对齐源码和实验观测点。

## 2. 当前证据基线

低成本 inspect 在 `gptq` 环境下通过：

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json
```

关键结果：

| 项目 | 结果 |
|---|---|
| `DeepseekV4ForCausalLM` registry | ok |
| vLLM prefill API | `flash_mla_sparse_fwd` ok |
| vLLM decode API | `flash_mla_with_kvcache` ok |
| KV cache dtype | `fp8_ds_mla` ok |
| DeepSeek V4 cache shape | `584` bytes/token ok |
| SM70 backend gate | `major in [7, 9, 10]` ok |
| SM70 runtime gate | `is_device_capability_family(70)` ok |
| `vllm._flashmla_C` import | ok in `gptq` |
| sparse FlashMLA supported | true |
| FlashMLA source branch/head | `feature/sm70-volta-flashmla @ a507081` |
| SM70 sparse decode/prefill source | present |
| model config | `DeepseekV4ForCausalLM`, `head_dim=512`, `qk_rope_head_dim=64`, `index_topk=512`, `compress_ratios=[0,4,128]` |

上一轮未启动 584B 模型 server，因此只形成了源码路径 + inspect 证据。本轮补充了实际推理实验，复用当前已加载的 8xV100 OpenAI server：

```text
pid=18714
endpoint=http://127.0.0.1:18080/v1
CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9
VLLM_SM70_MHC_FAST=0  # historical fallback run; current validated default is 1
max_model_len=4096
max_num_seqs=1
compilation_config={"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}
```

注意：本轮 server 的 `max_model_len=4096`，所以实测覆盖到 3.3k prompt tokens 附近；未冒充 8k/32k 长上下文结果。

## 3. 总体执行图

DeepSeek V4 每层 attention 的稳定分界点是：

```text
DeepseekV4ForCausalLM
  -> DeepseekV4Model
    -> DeepseekV4DecoderLayer
      -> mHC pre(attn)
      -> DeepseekV4Attention
        -> DeepseekV4MultiHeadLatentAttentionWrapper.forward()
          -> fused_wqa_wkv(hidden_states) -> qr, kv
          -> torch.ops.vllm.deepseek_v4_attention(...)
            -> attention_impl(...)
              -> fused_q_kv_rmsnorm(qr, kv)
              -> wq_b(qr) -> q
              -> indexer/compressor/KV insert 并行或串行
              -> DeepseekV4MLAAttention.forward(...)
                -> split decode tokens before prefill tokens
                -> _forward_prefill(...)
                -> _forward_decode(...)
          -> fused_inv_rope_fp8_quant(o)
          -> torch.ops.vllm.deepseek_v4_fp8_einsum(...)
          -> wo_b(...)
      -> mHC post(attn)
      -> DeepseekV4MoE
      -> mHC pre/post(ffn)
```

prefill 和 decode 的分流发生在 `DeepseekV4MLAAttention.forward()`：

| 分流依据 | 数据区间 | 调用 |
|---|---|---|
| `num_prefills > 0` | `q[num_decode_tokens:]` | `_forward_prefill(...)` |
| `num_decodes > 0` | `q[:num_decode_tokens]` | `_forward_decode(...)` |

vLLM 侧负责 batch/metadata、KV cache、SWA/compressed indices、fallback workspace、head padding 和图捕获边界；FlashMLA 侧负责 sparse prefill/decode attention kernel。

## 4. 数据组织

### 4.1 核心张量

| 名称 | 形状/含义 | 主要使用点 |
|---|---|---|
| `hidden_states` | `[T, hidden_size]` | attention 输入、compressor 输入 |
| `qr` | `[T, q_lora_rank]` | Q LoRA low-rank 分支 |
| `kv` | `[T, head_dim=512]` | MLA latent K/V，写入 paged cache |
| `q` | `[T, n_local_heads, 512]`，随后 pad 到 64/128 heads | FlashMLA query |
| `o_padded` | `[T, padded_heads, 512]` | FlashMLA 输出 buffer |
| `o` | `[T, n_local_heads, 512]` | O projection 输入 |
| `topk_indices_buffer` | `[max_num_batched_tokens, index_topk]` int32 | C4A indexer runtime top-k |
| `attn_sink` | `[padded_heads]` fp32 | FlashMLA sink bias |

### 4.2 KV cache 格式

DeepSeek V4 Flash 的主 KV cache 使用 `fp8_ds_mla`，每 token `584` bytes：

| 区域 | 字节 | 说明 |
|---|---:|---|
| NoPE FP8 数据 | 448 | 7 个 64-element quant block，float8 e4m3fn |
| RoPE BF16 数据 | 128 | 64 个 BF16 RoPE elements |
| UE8M0 scale | 8 | 7 个真实 scale byte + 1 pad |

逻辑 head_dim 是 512，其中 NoPE 448，RoPE 64。cache block layout 是 token data 区和 scale 区分离，读 decode cache 时需要按 `block_idx`、`pos_in_block`、`TOKEN_DATA_BYTES=576`、`SCALE_DIM=8` 计算字节偏移。

### 4.3 层类型

`compress_ratio` 决定层的 attention 数据组织：

| 层类型 | `compress_ratio` | cache/indices |
|---|---:|---|
| SWA-only | `<=1` | 只使用 sliding-window cache 和 `decode_swa_indices` |
| C4A | `4` | runtime indexer 写 `topk_indices_buffer`，再和 SWA indices 合并 |
| C128A | `128` | metadata build 阶段预计算 top-k，decode/prefill 直接读取 |

prefill 会把 compressed KV pool、SWA window 和当前 batch token 空间合并到一个 chunk-local `M = N + window_size + max_num_batched_tokens` workspace；decode 在 SM70 fallback 下也会把 paged FP8 cache gather 成 BF16 workspace，再复用 prefill-style sparse kernel。

## 5. Prefill 路径

### 5.1 函数调用链

```text
DeepseekV4MultiHeadLatentAttentionWrapper.forward
  -> attention_impl
    -> fused_q_kv_rmsnorm
    -> wq_b
    -> maybe_execute_in_parallel
      -> indexer(hidden_states, qr, positions, indexer_rotary_emb)   # C4A
      -> _fused_qnorm_rope_kv_insert(q, kv, positions, metadata)
      -> compressor(hidden_states, positions, rotary_emb)            # C4A/C128A
    -> pad q heads to 64/128
    -> DeepseekV4MLAAttention.forward
      -> _forward_prefill
        -> allocate kv workspace [PREFILL_CHUNK_SIZE, M, 512] bf16
        -> chunk loop, PREFILL_CHUNK_SIZE=4 requests
          -> dequantize_and_gather_k_cache(compressed cache)          # C4A/C128A
          -> dequantize_and_gather_k_cache(SWA cache)
          -> combine_topk_swa_indices(...)
          -> optional PrefillGraphDispatcher.try_graph_replay(...)
          -> _flashmla_bf16_io(q_chunk, output_slice)
          -> flash_mla_sparse_fwd(q, kv, indices, topk_length, attn_sink)
          -> _copy_flashmla_output(...)
```

### 5.2 计算方法

Prefill 的主要计算分四类：

| 阶段 | 计算方法 | 主要代价 |
|---|---|---|
| Q/KV projection | GEMM + RMSNorm + RoPE | 大矩阵吞吐，通常 compute-bound |
| Indexer | FP8/quantized MQA logits -> ReLU weighted score -> top-k | C4A 特有，依赖 cache layout 和 top-k width |
| Compressor | BF16 GEMM 得到 KV/score state，按 compress boundary 做 softmax weighted sum、RMSNorm、RoPE、FP8 quant、cache write | 长 prompt 下可能成为 prefill 关键开销 |
| Sparse attention | `flash_mla_sparse_fwd` 对 chunk 内 token 做 sparse QK、softmax/LSE、PV | topk + SWA width 决定 KV 读量和计算量 |

当前 prefill 的动态性来自：

| 动态项 | 影响 |
|---|---|
| 每个请求的 `seq_lens` 和 `gather_lens` | KV gather 形状、SWA window 长度变化 |
| chunk 内 token 数 `num_chunk_tokens` | FlashMLA prefill Q shape 变化 |
| C4A/C128A/SWA-only 层混合 | compressed region `N`、top-k width、indexer 是否执行变化 |

### 5.3 Prefill 效率审计点

| 审计点 | 观测指标 | 判断 |
|---|---|---|
| chunk 粒度 | `num_chunks`、每 chunk `num_chunk_tokens`、`M` | chunk 太小会增加 launch/metadata 比例；chunk 太大会放大 workspace |
| KV gather | `dequantize_and_gather_k_cache` 时间、读字节、scatter pattern | 长 prompt 下 HBM 带宽敏感 |
| `combine_topk_swa_indices` | CPU/GPU 时间、输出 `combined_lens` 分布 | 合并后 topk 宽度决定 attention 计算量 |
| compressor | firing token 数、compress_ratio boundary、kernel 时间 | C4A/C128A prefill 重点 |
| `flash_mla_sparse_fwd` | kernel 时间、topk width、LSE/max logits | FlashMLA SM70 sparse prefill 主优化对象 |
| BF16/FP16 转换 | `_flashmla_bf16_io`、`_copy_flashmla_output` | 若 attention kernel 变快，转换开销会浮现 |

## 6. Decode 路径

### 6.1 函数调用链

```text
DeepseekV4MultiHeadLatentAttentionWrapper.forward
  -> attention_impl
    -> fused_q_kv_rmsnorm
    -> wq_b
    -> indexer/KV insert/compressor
    -> DeepseekV4MLAAttention.forward
      -> _forward_decode
        -> build topk_indices/topk_lens
          -> C4A: compute_global_topk_indices_and_lens(topk_indices_buffer, ...)
          -> C128A: attn_metadata.c128a_global_decode_topk_indices
        -> q/output bf16 IO
        -> if SM70 fallback:
          -> _gather_decode_prefill_fallback_kv_with_indices_(compressed cache)
          -> _gather_decode_prefill_fallback_kv_with_indices_(SWA cache)
          -> torch.cat(compressed_indices, swa_indices)
          -> flash_mla_sparse_fwd(...)   # prefill-style kernel with T=1 decode
        -> else:
          -> flash_mla_with_kvcache(...)
```

当前 `_should_use_sm70_decode_prefill_fallback()` 对 CUDA capability `< 8` 返回 true，因此 SM70 decode 默认仍走 gather-to-BF16 + `flash_mla_sparse_fwd` fallback。直接 `flash_mla_with_kvcache` 分支存在，但在 SM70 上不会被这个 guard 命中。

### 6.2 计算方法

Decode 是单 token/小 batch 重复路径，关键不是总 FLOPs，而是每 token 每层的 kernel 数、host sync、cache 读放大和图捕获效率。

| 阶段 | 计算方法 | 主要代价 |
|---|---|---|
| Sparse indexer | `sm70_fp8_paged_mqa_logits` 从 paged FP8 cache 读 K，解码 FP8，按 head 点积、ReLU、加权 | decode top-k 选择，cache 随机读 |
| KV insert | SM70 Triton qnorm + GPT-J RoPE + FP8 quant + paged cache write | 每层每 token 固定开销 |
| SM70 fallback gather | `_gather_decode_kv_triton_kernel` 读 584B token cache，FP8->FP32->BF16，RoPE BF16 copy | 把 sparse paged cache 物化成 BF16 workspace |
| Attention | fallback 下调用 `flash_mla_sparse_fwd`，直接路径调用 `flash_mla_with_kvcache` | fallback 多一次 materialization；直接路径应更高效 |
| O projection | inverse RoPE + FP8 quant + FP8 einsum + `wo_b` | 若 FP32 pre-dequant/torch einsum存在，会成为 decode 带宽/launch 瓶颈 |
| mHC/MoE | mHC pre/post、MoE routing/expert GEMM | 每层固定成本，语义安全优先于默认启用快路径 |

### 6.3 Decode 效率审计点

| 审计点 | 观测指标 | 判断 |
|---|---|---|
| CUDA graph 状态 | log 是否有 graph capture/replay；`mode=FULL_DECODE_ONLY` | decode 图生效前，launch overhead 会掩盖 kernel 优化收益 |
| fallback 命中率 | `_should_use_sm70_decode_prefill_fallback` 命中次数 | 若总是命中，FlashMLA SM70 direct decode 还未成为主路径 |
| fallback gather | compressed/SWA 两次 gather 时间、BF16 workspace 字节 | 是当前 attention 路径最大结构性浪费之一 |
| `flash_mla_sparse_fwd` fallback | T=1 sparse prefill kernel 时间 | prefill kernel 复用 decode 可能不适合 per-token 低延迟 |
| direct `flash_mla_with_kvcache` | 是否进入 SM70 sparse decode kernel、split/combine 时间 | 目标路径，需要 correctness + throughput 双验证 |
| indexer | `sm70_fp8_paged_mqa_logits` 时间、top-k 宽度、context_len | 长上下文 decode 的 cache 读放大会在这里出现 |
| mHC | `VLLM_SM70_MHC_FAST=0/1` A/B 的 exact canary 和 token/s | 快路径曾有语义风险，必须先证正确 |
| O projection | `_sm70_fp8_einsum_bmm` 时间、weight cache dtype/size | T=1 decode 对内存读和 launch 敏感 |

## 7. 实验测量方案

### 7.1 基线检查

每次实验先记录：

```bash
git rev-parse --show-toplevel --abbrev-ref HEAD HEAD
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && conda activate gptq && python benchmarks/deepseek_v4_flashmla_sm70.py --mode inspect --json'
nvidia-smi -L
```

必须保存字段：

| 字段 | 用途 |
|---|---|
| worktree/branch/HEAD | 防止 root checkout 和 `.worktrees` 混淆 |
| FlashMLA source branch/head | 判断 SM70 kernel 是否是预期源码 |
| `flashmla_core_importable` | 判断 `_flashmla_C` 是否加载 |
| `sparse_supported` | 判断 runtime gate |
| model config | 判断 cache shape/topk/compress ratio 是否匹配 |

### 7.2 语义 smoke

最小语义门禁：

```bash
python benchmarks/deepseek_v4_flashmla_sm70.py \
  --mode openai-stream \
  --endpoint http://127.0.0.1:23333/v1 \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --prompt "请只输出这五个字符，不要输出其它内容：ZX-42" \
  --max-tokens 8
```

记录：

| 字段 | 要求 |
|---|---|
| `output_preview` | 必须精确包含或等于 `ZX-42`，不能用乱码通过 |
| `finish_reason` | 优先 `stop`，`length` 只作为测速样本 |
| `TTFT` | prefill + 首 token 开销 |
| `decode_tokens_per_s` | 必须用 stream timestamps 自算，不能直接复用框架吞吐 |

### 7.3 Prefill micro profile

建议增加/使用 CUDA event 或 Nsight Systems range，按 chunk 记录：

| Range 名称 | 包含函数 | 输出 |
|---|---|---|
| `prefill.kv_gather.compressed` | `dequantize_and_gather_k_cache(compressed)` | us、bytes、seq_lens |
| `prefill.kv_gather.swa` | `dequantize_and_gather_k_cache(SWA)` | us、bytes、gather_lens |
| `prefill.combine_indices` | `combine_topk_swa_indices` | us、combined_lens histogram |
| `prefill.flashmla_sparse_fwd` | `_flashmla_bf16_io` + `flash_mla_sparse_fwd` + copy | us、topk width、tokens |
| `prefill.compressor` | `DeepseekCompressor.forward` kernels | us、firing tokens |
| `prefill.indexer` | `DeepseekV4Indexer.forward` | us、top-k width |

测试矩阵：

| Case | 目的 |
|---|---|
| 1k prompt, single request | 低上下文启动和短 prefill |
| 8k/32k prompt, single request | 长上下文 KV gather 和 FlashMLA sparse prefill |
| mixed batch, ragged prompt | chunk 动态形状和 metadata 成本 |
| C4A-heavy trace | indexer + compressor 成本 |
| SWA-only trace | attention lower bound |

### 7.4 Decode micro profile

按 decode iteration 记录：

| Range 名称 | 包含函数 | 输出 |
|---|---|---|
| `decode.indexer` | `sm70_fp8_paged_mqa_logits` + topk | us、context_len、topk |
| `decode.kv_insert` | `_sm70_triton_qnorm_rope_kv_insert` | us |
| `decode.fallback_gather.compressed` | `_gather_decode_prefill_fallback_kv_with_indices_` | us、BF16 bytes |
| `decode.fallback_gather.swa` | 同上 | us、BF16 bytes |
| `decode.attn.fallback` | `flash_mla_sparse_fwd` with T=1 | us |
| `decode.attn.direct` | `flash_mla_with_kvcache` | us、split/combine |
| `decode.o_proj` | `fused_inv_rope_fp8_quant` + FP8 einsum + `wo_b` | us |
| `decode.mhc` | mHC pre/post | us、fast/fallback mode |
| `decode.moe` | routing + expert GEMM | us、expert distribution |

测试矩阵：

| Case | 目的 |
|---|---|
| graph off, 1k prompt | eager decode kernel boundary |
| graph on, 1k prompt | launch overhead 是否消除 |
| graph on, 32k prompt | long-context cache read 放大 |
| `VLLM_SM70_MHC_FAST=0/1` | mHC 快路径语义和速度 A/B |
| direct decode opt-in vs fallback | FlashMLA direct SM70 decode 是否值得切默认 |

## 8. 本轮实测结果

### 8.1 测量方法

本轮每条请求前后各抓一次 `/metrics`，用 delta 计算该请求独占的 vLLM phase 时间；同时用 OpenAI streaming 的首个 generated delta 到结束的 wall-clock 计算 decode tokens/s。

保存的核心字段：

| 字段 | 来源 | 说明 |
|---|---|---|
| `prompt_tokens` / `completion_tokens` | OpenAI `usage`，缺失时用 metrics delta | 真实 token 数 |
| `TTFT_s` | 客户端流式时间戳 | 从发送请求到首个 generated delta |
| `engine_prefill_s` | `vllm:request_prefill_time_seconds_sum` delta | vLLM RUNNING/PREFILL phase |
| `engine_decode_s` | `vllm:request_decode_time_seconds_sum` delta | vLLM RUNNING/DECODE phase |
| `client_decode_tokens_per_s` | 手算 streaming 时间 | `completion_tokens / (end - first_generated_delta)` |
| `engine_prefill_tokens_per_s` | metrics delta | `prompt_tokens / engine_prefill_s` |
| `prefix_cache_hits` | metrics delta | 本轮均为 0，避免 prefix cache 污染 |

### 8.2 端到端样本

| Case | prompt tokens | completion tokens | TTFT s | engine prefill s | prefill tok/s | client decode tok/s | engine decode tok/s | total s | finish | 语义 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `semantic_zx42` | 19 | 4 | 0.537 | 0.521 | 36.48 | 13.84 | 13.75 | 0.826 | `stop` | `ZX-42` exact |
| `short_cn_qa` | 36 | 64 | 0.707 | 0.642 | 56.08 | 10.44 | 10.44 | 6.840 | `length` | 中文输出连贯 |
| `ctx_approx_1k_retry` | 887 | 64 | 13.480 | 13.463 | 65.88 | 9.82 | 9.82 | 19.994 | `length` | 英文摘要连贯 |
| `ctx_approx_3k` | 2867 | 64 | 53.234 | 53.197 | 53.89 | 9.24 | 9.23 | 60.164 | `length` | 英文摘要连贯 |
| `ctx_near_4k_retry` | 3327 | 32 | 57.436 | 57.402 | 57.96 | 9.33 | 9.32 | 60.864 | `length` | 英文摘要连贯 |

另一次更长 prompt 尝试被 server 返回 HTTP 400，原因是超过当前 `max_model_len=4096`；该样本只作为边界证明，不纳入性能表。

### 8.3 定量结论

1. `TTFT` 与 `engine_prefill_s` 高度贴合。887 tokens 时 `TTFT=13.480s`、prefill phase `13.463s`；3327 tokens 时 `TTFT=57.436s`、prefill phase `57.402s`。在 `max_num_seqs=1` 的空闲服务上，TTFT 基本可以当作 prefill 时间代理。
2. prefill 吞吐在 887-3327 prompt tokens 间约 `54-66 tok/s`。短 prompt 的固定开销明显，长 prompt 后每 token 成本没有线性恶化到失控，但绝对 TTFT 已经成为主要用户可见延迟。
3. decode 吞吐在有效样本中约 `9.2-10.4 tok/s`，4-token canary 的 `13.8 tok/s` 不适合作为稳态值。prompt 从 36 增到 3327 tokens 后，decode 从 `10.44` 降到 `9.33 tok/s`，长上下文 cache/indexer/attention 读放大带来约 `10-12%` 下降，但不是本轮最大瓶颈。
4. 对 3327 prompt + 32 decode 的请求，prefill 占 `57.40 / 60.84 = 94.4%` engine inference 时间；decode 只占 `5.6%`。在当前 4k 内实验里，优化 TTFT/prefill 比继续只抠单 token decode 更直接影响端到端时延。
5. 所有样本 `prefix_cache_hits=0`、`prompt_tokens_cached=0`，因此上表不是 prefix cache 命中后的虚高吞吐。
6. 当前进程按 `FULL_DECODE_ONLY` 配置启动，但本轮无法从已存在的 pts 日志重新截取 `Capturing CUDA graphs` 文本；本轮只把它作为 graph-configured server 测速。若要把 graph capture 作为新证据，需要重启 server 并保存启动日志。

### 8.4 对优化方向的影响

| 优化点 | 本轮数字支撑 | 优先级调整 |
|---|---|---|
| prefill KV gather / sparse prefill / compressor-indexer 分解 | 887 tokens 已需 `13.46s`，3327 tokens 需 `57.40s` | 从 P1 上调为和 decode direct path 并列的 P0 审计项 |
| decode fallback elimination | 长上下文 decode 约 `9.2-9.8 tok/s`，且随 prompt 增长下降约 `10-12%` | 仍是 P0，但端到端收益要和 prefill 比例分开估算 |
| mHC fast path | fallback `VLLM_SM70_MHC_FAST=0` 和 fast `=1` canary 均 exact、自然语言连贯 | 当前 validated default 为 `1`；`0` 仅用于 fallback A/B 或 rollback |
| FULL_DECODE_ONLY graph | historical fallback mHC graph server decode 稳态约 `9-10 tok/s`；当前 `MHC_FAST=1` graph 复测为 `15.8-19.2 tok/s` | 后续 graph 性能报告必须同时记录 `VLLM_SM70_MHC_FAST` |
| 8k/32k 长上下文 | 当前 server 只到 `max_model_len=4096` | 需要新 server 或更高 KV 预算后单独测，不从 4k 外推 |

### 8.5 Prefill CUDA Graph 实验与 phase trace

本轮继续实测了自定义 prefill CUDA graph。启动配置：

```bash
VLLM_PREFILL_CUDAGRAPH=1
VLLM_PREFILL_CUDAGRAPH_DEBUG=0
VLLM_PREFILL_CUDAGRAPH_CAPTURE_TOKENS=32,256,1024,2048
VLLM_DEEPSEEK_V4_PROFILE=1
VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH=/tmp/deepseek_v4_phase_profile_live.jsonl
--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

实现侧新增的 raw trace 是 opt-in 的逐 CUDA event JSONL；聚合时按 TP worker/pid 分组，报告 `max_rank_ms`，避免把 8 个并行 rank 的耗时直接相加成 wall time。raw trace 有文件 IO 和同步开销，所以本节只用于阶段占比和热点定位，端到端吞吐仍以 8.2 中无 raw trace 的结果为准。

启动证据：

| 证据 | 结果 |
|---|---|
| prefill graph workspace | 每 rank、每类 `M` 只分配一次：`M=6272` 约 `457.51 MB`，`M=3200` 约 `421.51 MB`，`M=2208` 约 `409.88 MB` |
| prefill graph capture | `M=6272/3200/2208` 均捕获 `32/256/1024/2048` token 桶 |
| decode graph capture | 官方 `FULL_DECODE_ONLY` 仍输出 `Graph capturing finished in 2 secs, took 0.11 GiB` |
| 语义 smoke | graph-only ZX-42 请求输出精确 `ZX-42` |

阶段聚合结果：

| Case | 状态 | prompt tokens | engine prefill s | decode tok/s | 最大热点 `max_rank_ms` | 结论 |
|---|---|---:|---:|---:|---|---|
| `semantic_zx42_phase` | 成功，`ZX-42` exact | 19 | 7.321 | 13.18 | `impl.kv_insert` 3298 ms；`prefill.flashmla_graph_replay` 2519 ms | 短 prompt 命中 32 桶也明显反优化 |
| `ctx_approx_1k_phase` | 成功，输出连贯 | 887 | 79.515 | 10.00 | `prefill.flashmla_graph_replay` 78181 ms；`wrapper.wo_b` 8115 ms；`impl.indexer_kv_compress_overlap` 418 ms | 1K prompt 几乎全部时间被静态 graph replay 吃掉 |
| `ctx_near_4k_phase` | 未完成；EngineCore fatal | server dump: 3493 | 无完整 metrics | 无 | partial trace 中 `prefill.flashmla_graph_replay` 已达 27923 ms；随后 `RPC call to sample_tokens timed out` | 2048 静态桶在长 prompt 下可拖死服务 |

关键判断：

1. 当前 prefill CUDA graph 不能作为默认优化打开。它语义上已经能通过短 canary，但性能模型不成立：graph 绑定静态 `tokens x M`，小 prompt/1K prompt 都被迫执行过大的 `flash_mla_sparse_fwd` 图。
2. 1K 样本里 `prefill.flashmla_graph_replay` 单 rank `78.18s`，占 `engine_prefill_s=79.52s` 的绝对主体；KV gather、combine indices、compressor/indexer 在该形态下不是主瓶颈。
3. 近 4K 样本在 `prompt_token_ids_len=3493`、首个 `2048` token chunk 上进入长时间执行，最终触发 EngineCore `sample_tokens` RPC timeout。该样本不能纳入吞吐均值，但足以作为“prefill graph 当前不具备可用性”的失败证据。
4. 现阶段优化方向应回到 eager prefill 的真实热点：先保留 `VLLM_PREFILL_CUDAGRAPH=0`，用 raw trace/NVTX 在 eager prefill 下拆 `flash_mla_sparse_fwd`、KV gather、indexer/compressor 和 O projection；prefill graph 只有在 FlashMLA sparse prefill 支持更细粒度动态形状或低开销 bucketing 后才值得重启。

## 9. 优化方向

### P0: Prefill 长上下文 phase 分解

现状：4k 内实测显示 prefill/TTFT 是端到端主耗时，3327 prompt tokens 时 prefill phase `57.40s`，占 engine inference `94.4%`。prefill CUDA graph 实验进一步证明当前 graph bucketing 不是可用优化：887 tokens 时 `prefill.flashmla_graph_replay` 单 rank `78.18s`，近 4K 样本触发 EngineCore timeout。

目标：保持 prefill graph 默认关闭，给 eager prefill 路径加可开关的 CUDA event/NVTX/raw trace，拆出 KV gather、indexer、compressor、`flash_mla_sparse_fwd` 和 O projection 的层级占比。

验收：

| 验收项 | 标准 |
|---|---|
| phase 覆盖 | 887/2867/3327 token 三档均能输出 prefill 子阶段表 |
| overhead | profiling gate 默认关闭，开启后语义不变 |
| 决策 | 能指出 TTFT 主要来自 FlashMLA sparse prefill、compressor/indexer、KV gather 还是 projection |

### P0: Prefill CUDA Graph 先冻结为实验路径

现状：`VLLM_PREFILL_CUDAGRAPH=1` 可完成 capture，并在短 canary 上保持语义，但在 887 tokens 上显著慢于 eager prefill，近 4K 样本触发 EngineCore timeout。

冻结条件：

| 条件 | 标准 |
|---|---|
| 默认行为 | `VLLM_PREFILL_CUDAGRAPH` 不设置时不改变 eager prefill |
| 实验开关 | 仅显式设置 `VLLM_PREFILL_CUDAGRAPH=1` 才进入该路径 |
| 重启条件 | FlashMLA sparse prefill 支持低浪费动态 shape 或新的 exact-shape capture 策略 |
| 回归门槛 | ZX-42 exact、887/3k prompt 不慢于 eager、长 prompt 不触发 EngineCore timeout |

### P0: Decode attention fallback elimination

现状：SM70 decode guard 对 capability `<8` 固定命中 fallback，流程是 paged FP8 cache -> BF16 workspace -> `flash_mla_sparse_fwd`。

目标：让 SM70 主路径进入 FlashMLA 的 `flash_mla_with_kvcache` sparse decode kernel，并保留 fallback 作为语义安全降级。

验收：

| 验收项 | 标准 |
|---|---|
| correctness | `ZX-42` exact canary，first-token logprob 与 fallback 对齐 |
| path proof | trace 显示进入 `_flashmla_C` sparse decode SM70 path |
| speed | decode token/s 高于 fallback，32k 不退化 |
| graph | `FULL_DECODE_ONLY` capture/replay 仍成立 |

### P0: mHC fast path 默认口径与 fallback A/B

现状：`VLLM_SM70_MHC_FAST` 当前 validated default 为开启；显式设置为 `0`
只用于 fallback A/B 或 rollback。旧实验记录里的 `MHC_FAST=0` 数字不能作为当前
默认 decode 性能基线。

目标：保持 SM70 mHC cuBLAS fp16 GEMM + fused Triton path 的默认语义和速度回归门，
同时保留 `VLLM_SM70_MHC_FAST=0` 作为明确的 fallback 对照。

验收：

| 验收项 | 标准 |
|---|---|
| exact canary | `ZX-42` 不漂移 |
| identity smoke | 自然语言 prompt 不乱码 |
| numeric A/B | mHC fast 与 fallback 在代表层输出误差受控 |
| performance | decode 每 token 稳态下降，且 graph capture 不破坏 |

### P0: Decode O projection bandwidth audit

目标：确认 `_sm70_fp8_einsum_bmm` 是否仍存在 FP32 pre-dequant weight cache 和 T=1 torch einsum 带宽浪费。

方向：

| 方向 | 预期收益 |
|---|---|
| FP16 pre-dequant cache | 降低每层 weight 读带宽 |
| SM70 Triton/CUDA grouped matmul | 减少 small GEMM launch 与 torch overhead |
| CUDA graph 内固定 workspace | 减少分配和 launch 抖动 |

### P1: Prefill compressor/indexer 长 prompt 审计

目标：在 8k/32k prompt 下量化 C4A/C128A compressor、indexer 和 KV gather 的比例。

方向：

| 方向 | 预期收益 |
|---|---|
| compressor firing-token fused path 全面接入主 forward | 消除 Python/token-loop 风险 |
| C4A indexer top-k 宽度/块访问重排 | 降低 cache 随机读 |
| compressed/SWA gather 合并或 shared staging | 降低 HBM 往返 |

### P1: FlashMLA SM70 sparse prefill tuning

FlashMLA 侧已有 SM70 sparse prefill BF16 fast path，并有 `mma884_online`、K tile、CTA threads 等调参空间。

方向：

| 方向 | 预期收益 |
|---|---|
| `K_TILE=16/32` 与 128/256 threads 分矩阵复扫 | 找到长 prompt / max_topk 最优点 |
| shared K/V staging bank conflict 检查 | 降低 memory stall |
| topk + SWA combined width 分布驱动调度 | 减少无效 KV 计算 |

### P2: Prefill CUDA graph 保持谨慎

prefill 是动态形状路径，完整 graph capture 收益通常不如 decode 明确。当前可以保留 `PrefillGraphDispatcher` 作为实验路径，但默认优化优先级应低于 decode direct path、mHC 和 prefill FlashMLA/compressor。

验收重点不是“能 capture”，而是：

| 验收项 | 标准 |
|---|---|
| padding overhead | padding 后总时间低于 eager |
| semantic parity | debug mode eager/graph 输出一致 |
| memory footprint | workspace 增量不挤压 KV cache |

### 9.1 当前落地状态（2026-05-06）

| 优化项 | 状态 | 落地内容 | 下一道门槛 |
|---|---|---|---|
| P0 prefill phase 分解 | [x] 已实测归档 | `VLLM_DEEPSEEK_V4_PROFILE=1` 记录 CUDA event/Prometheus/raw JSONL，`phase-summary` 按最慢 rank 聚合；无 prefix cache fallback trace 显示 `prefill.flashmla_sparse_fwd` 最慢 rank `114531.6ms`，远高于 `wrapper.wo_b` `14815.5ms`、indexer/compressor overlap `7499.7ms` | 优化主方向转向 FlashMLA sparse prefill；vLLM 侧继续保留 raw trace 作回归门 |
| P0 prefill CUDA graph 冻结 | [x] 默认冻结 | `VLLM_PREFILL_CUDAGRAPH` 默认仍为 `0`；graph dispatcher 只在显式开关下创建，且使用私有 graph workspace、per-token lens 和 runtime KV workspace | 只有 graph 在 887/3k 不慢于 eager 且长 prompt 不 timeout 后才考虑默认 |
| P0 decode fallback elimination | [x] opt-in + live A/B | `VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE=1` 可进入 `decode.attn.direct_flashmla`；fresh eager raw trace 记录 `decode.attn.direct_flashmla`，无 profiler graph server 的 1012/2892/3492 token decode 分别比 fallback 快约 `9.9%/4.5%/4.6%` | 默认仍保留 fallback；direct path 可作为下一轮更长上下文候选 |
| P0 mHC fast path | [x] live gate 通过并设为当前默认 | `VLLM_SM70_MHC_FAST=1` 单层 numeric A/B 通过，full-model `ZX-42` exact，自然语言 smoke 连贯；旧 1012 token run 中 fast 为 `15.98 tok/s`，fallback 为 `9.98 tok/s`；2026-05-09 复测 897/2825/3328 prompt tokens 时 fast 为 `19.21/15.80/15.95 tok/s`，fallback 为 `11.03/9.88/9.99 tok/s` | 后续测速默认必须保持 `MHC_FAST=1`，只有 rollback/A-B 才显式设 `0` |
| P0 decode O projection | [x] 第一阶段实测 | `_sm70_fp8_einsum_bmm` 已使用 `_sm70_predequant_f16` 持久 cache；fallback profile 中 `wrapper.o_fp8_einsum` 最慢 rank `303.6ms`，`wrapper.wo_b` `14815.5ms`，低于 sparse prefill 主热点 | grouped matmul 仍可做，但优先级低于 FlashMLA sparse prefill 和 mHC opt-in 扩大验证 |
| P1 compressor/indexer 长 prompt 审计 | [x] 4k 内审计完成 | phase range 覆盖 `impl.indexer_kv_compress_overlap`、`impl.compressor_kv_insert_overlap`、`prefill.compressed_gather`、`prefill.swa_gather`、`prefill.combine_indices`；4k 内 indexer/compressor 是第二梯队，不是主瓶颈 | 需要更大 `max_model_len` server 跑 8k/32k，确认长上下文是否转为主瓶颈 |
| P1 FlashMLA sparse prefill tuning | [~] 当前 build 已实测 | vLLM 侧已证明 `prefill.flashmla_sparse_fwd` 是主热点；只读 FlashMLA benchmark 当前默认 `256/32/mma884_online`，`max_topk=8192` 为 `21580.799us`、48 registers、0 spills、25% static occupancy | 完整 `K_TILE=16/32` 与 `CTA=128/256` sweep 需要重编 `/mnt/data/apps/FlashMLA`，当前仓库有 decode 侧未提交改动，暂不触碰 |
| P2 prefill graph 谨慎策略 | [x] 已固化 | graph replay 被独立标记为 `prefill.flashmla_graph_replay`，可直接和 eager `prefill.flashmla_sparse_fwd` 对比；`VLLM_PREFILL_CUDAGRAPH_PERF_GATE=1` 可在 debug validation 中禁用慢 bucket | 保持实验路径，不再把“能 capture”当成完成标准 |

### 9.2 实测验收记录（2026-05-06 后续）

无 profiler、`--no-enable-prefix-caching`、`FULL_DECODE_ONLY`、8xV100、`VLLM_PREFILL_CUDAGRAPH=0`：

| Case | prompt tokens | path | TTFT s | decode tok/s | finish | 语义 |
|---|---:|---|---:|---:|---|---|
| `ZX-42` | 19 | fallback | 0.646 | 13.89 | stop | exact |
| `ZX-42` | 19 | direct decode | 1.435 | 15.69 | stop | exact |
| 1k | 1012 | fallback | 15.616 | 9.98 | length | 连贯 |
| 1k | 1012 | direct decode | 15.653 | 10.97 | length | 连贯 |
| 3k | 2892 | fallback | 50.278 | 9.44 | length | 连贯 |
| 3k | 2892 | direct decode | 50.200 | 9.87 | length | 连贯 |
| near-4k | 3492 | fallback | 60.777 | 9.72 | length | 连贯 |
| near-4k | 3492 | direct decode | 61.367 | 10.17 | length | 连贯 |
| 1k | 1012 | `VLLM_SM70_MHC_FAST=1` | 15.277 | 15.98 | length | 连贯 |

### 9.3 CUDA Graph decode 速度回归排查（2026-05-09）

同一份代码、同一 `FULL_DECODE_ONLY` graph、同一 `/tmp/dsv4_cudagraph_speed.py`
脚本下，只改变 `VLLM_SM70_MHC_FAST`：

| Case | prompt tokens | `MHC_FAST=1` decode tok/s | `MHC_FAST=0` decode tok/s | 结论 |
|---|---:|---:|---:|---|
| `ZX-42` | 19 | 31.06 | 16.10 | 两者 canary exact，短输出受固定开销影响较大 |
| short CN | 12 | 26.68 | 13.24 | 两者连贯 |
| 1k | 897 | 19.21 | 11.03 | 速度下降来自 fallback mHC |
| 3k | 2825 | 15.80 | 9.88 | 复现此前约 10 tok/s 低位 |
| near-4k | 3328 | 15.95 | 9.99 | 复现此前约 10 tok/s 低位 |

两轮都确认 graph capture 生效：`Capturing CUDA graphs (decode, FULL)`，
`Graph capturing finished ... took 0.11 GiB`，prefix cache hits/queries 均为 0。
因此 2026-05-09 看到的“默认路径约 10 tok/s”不是 MoE early shared-experts
overlap 的默认分支泄漏；根因是测速启动口径显式关闭了 mHC fast path。

Direct decode first-token logprob gate：fallback 首 token `-`，logprob `-1.83347`；direct 首 token同为 `-`，logprob `-1.85339`。fresh eager/path proof trace：`/tmp/deepseek_v4_phase_direct_eager_registry_20260506_1029.jsonl` 中出现 `decode.attn.direct_flashmla`，最慢 rank `38.435ms`；启动日志不再出现 custom DeepSeek V4 env unknown warning。

FlashMLA sparse prefill 当前 build 只读 benchmark，命令使用 `PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313`、单 V100、`--warmup 1 --runs 1`：

| Case | topk | features | mean us | max_abs_out | registers | spills | shared bytes | path |
|---|---:|---|---:|---:|---:|---:|---:|---|
| quick | 64 | `-` | 171.008 | 0.000128843 | 48 | 0 | 35088 | `256/32/mma884_online` |
| quick | 64 | `attn_sink` | 211.968 | 0.000247553 | 48 | 0 | 39184 | `256/32/mma884_online` |
| quick | 128 | `topk_length` | 582.656 | 0.00025019 | 48 | 0 | 39184 | `256/32/mma884_online` |
| quick | 128 | `attn_sink+topk_length` | 859.136 | 0.000125986 | 48 | 0 | 35088 | `256/32/mma884_online` |
| max_topk | 8192 | `attn_sink` | 21580.799 | 0.000122488 | 48 | 0 | 39184 | `256/32/mma884_online` |

## 10. 结论

1. 当前 worktree 的 DeepSeek V4 Flash 路径静态和 `gptq` runtime inspect 已经满足分析前置条件：模型、registry、FlashMLA sparse gate、`fp8_ds_mla` 和 local FlashMLA SM70 source 都对齐。
2. 本轮实测补齐了上一轮缺口：在 `FULL_DECODE_ONLY` 配置、8xV100、`max_model_len=4096` 下，fallback/direct/mHC-fast 都跑了 `ZX-42` semantic gate；direct 和 mHC-fast 都保持 exact canary，mHC-fast 自然语言 smoke 也连贯。
3. prefill/TTFT 仍是当前 4k 内最直接的端到端瓶颈：1012/2892/3492 prompt tokens 的 fallback TTFT 分别为 `15.616/50.278/60.777s`，direct decode 基本不改变 TTFT。
4. direct decode opt-in 有稳定但有限收益：1012/2892/3492 prompt tokens 的 decode 从 `9.98/9.44/9.72 tok/s` 提到 `10.97/9.87/10.17 tok/s`；fresh trace 证明路径进入 `decode.attn.direct_flashmla`。
5. `VLLM_SM70_MHC_FAST=1` 的收益更明显：单层 numeric A/B、full-model canary 和中文 identity smoke 通过；2026-05-09 的 `FULL_DECODE_ONLY` 复测显示 897/2825/3328 prompt tokens 的 decode 为 `19.21/15.80/15.95 tok/s`，而 fallback `MHC_FAST=0` 为 `11.03/9.88/9.99 tok/s`。当前默认测速口径应保持 `MHC_FAST=1`。
6. 当前实测没有 8k/32k，因为现有 server 限制为 `max_model_len=4096`。下一轮若要回答长上下文效率，需要重启更大 KV 预算的 server，并同时保存 CUDA Graph capture 日志。
7. 后续实验报告应同时输出 `TTFT`、手算 `decode_tokens_per_s`、`finish_reason`、语义结论、graph 状态、prefill/decode micro range 分解，否则无法判断时间究竟花在 attention、KV、mHC、MoE 还是 launch/capture 上。
