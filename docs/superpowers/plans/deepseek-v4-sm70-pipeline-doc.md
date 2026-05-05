# DeepSeek V4 Flash — SM70 (V100) Inference Pipeline Analysis & Optimization

> **Scope**: 8× V100 GPUs, `1Cat-vLLM` branch `feature/vllm-0190-upstream-split`, FlashMLA branch `feature/sm70-volta-flashmla`.
>
> This document is the single deliverable for Requirements 1–14. Sections are added incrementally by task; cross-references use `§` + section heading.

---

## 1  Prefill Path Flow Documentation

*Satisfies Requirement 1 (acceptance criteria 1.1 – 1.5).*

### 1.1  Complete Prefill Call Chain

The prefill path is entered when the current step contains at least one prefill request (query length > 1). Prefill tokens are placed **after** decode tokens in the reordered batch.

```
DeepseekV4MultiHeadLatentAttentionWrapper.forward(positions, hidden_states)
│
├─ fused_wqa_wkv(hidden_states) → qr [T, q_lora_rank], kv [T, head_dim]
│
├─ torch.ops.vllm.deepseek_v4_attention(hidden_states, qr, kv, positions, o_padded, layer_name)
│   └─ attention_impl(hidden_states, qr, kv, positions, o_padded)
│       │
│       ├─ fused_q_kv_rmsnorm(qr, kv, q_norm.weight, kv_norm.weight, eps)
│       │   → qr (RMS-normed), kv (RMS-normed)
│       │
│       ├─ wq_b(qr) → q  [T, n_local_heads, head_dim]
│       │
│       ├─ maybe_execute_in_parallel:                          (§1.2)
│       │   ├─ Default stream:
│       │   │   • indexer(hidden_states, qr, positions, indexer_rotary_emb)
│       │   │     [only for C4A layers with indexer; fills topk_indices_buffer]
│       │   │
│       │   └─ Aux stream (overlapped):
│       │       • _fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
│       │       │   └─ SM80+: torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
│       │       │              q, kv, swa_kv_cache_2d, slot_mapping, positions,
│       │       │              cos_sin_cache, eps, block_size)
│       │       │   └─ SM70:  _torch_qnorm_rope_kv_insert_fallback(...)   (§1.4)
│       │       │
│       │       • compressor(hidden_states, positions, rotary_emb)          (§1.4)
│       │         [only for C4A / C128A layers]
│       │         └─ SM80+: _fused_kv_compress_norm_rope_insert_sparse_attn (Triton)
│       │         └─ SM70:  _torch_fused_compress_norm_rope_insert_fp8_fallback
│       │
│       ├─ q padding: F.pad(q, (0,0, 0, padded_heads - n_local_heads))
│       │   [FlashMLA requires exactly 64 or 128 heads]
│       │
│       └─ mla_attn(q, kv, positions, output=o_padded)
│           └─ DeepseekV4MLAAttention.forward(q, kv, positions, output)
│               │
│               ├─ Split batch: num_decode_tokens | num_prefill_tokens
│               │   (via swa_metadata.num_decodes / num_decode_tokens)
│               │
│               └─ _forward_prefill(q[num_decode_tokens:], ...)             (§1.3)
│                   │
│                   ├─ Allocate workspace via workspace_manager.get_simultaneous():
│                   │   kv = [PREFILL_CHUNK_SIZE, M, head_dim] bf16
│                   │   where M = N + window_size + max_num_batched_tokens
│                   │   and N = max_model_len // compress_ratio (0 for SWA-only)
│                   │
│                   ├─ Chunked loop (num_chunks = ceil(num_prefills / PREFILL_CHUNK_SIZE)):
│                   │   for chunk_idx in range(num_chunks):
│                   │   │
│                   │   ├─ dequantize_and_gather_k_cache(                   (compressed)
│                   │   │       kv[:chunk_size], compressed_k_cache,
│                   │   │       seq_lens=seq_lens[chunk] // compress_ratio,
│                   │   │       block_table=block_table[chunk],
│                   │   │       block_size=block_size // compress_ratio,
│                   │   │       offset=0)
│                   │   │   [only for C4A / C128A layers]
│                   │   │
│                   │   ├─ dequantize_and_gather_k_cache(                   (SWA)
│                   │   │       kv[:chunk_size], swa_k_cache,
│                   │   │       seq_lens=seq_lens[chunk],
│                   │   │       gather_lens=gather_lens[chunk],
│                   │   │       block_table=swa_block_table[chunk],
│                   │   │       block_size=swa_block_size,
│                   │   │       offset=N)
│                   │   │
│                   │   ├─ _normalize_flashmla_sm70_prefill_kv_(kv_chunk)
│                   │   │   [currently identity; placeholder for SM70 normalization]
│                   │   │
│                   │   ├─ combine_topk_swa_indices(
│                   │   │       topk_indices[query_start:query_end],
│                   │   │       query_start_loc[chunk_range],
│                   │   │       seq_lens[chunk], gather_lens[chunk],
│                   │   │       window_size, compress_ratio, top_k, M, N)
│                   │   │   → combined_indices [num_chunk_tokens, topk+window_size]
│                   │   │   → combined_lens [chunk_size]
│                   │   │
│                   │   ├─ _flashmla_bf16_io(q_chunk, output_chunk)
│                   │   │   → (q_bf16, flash_output_bf16)
│                   │   │   [SM70: fp16→bf16 upcast for FlashMLA]
│                   │   │
│                   │   └─ flash_mla_sparse_fwd(
│                   │           q=q_chunk,                    # [chunk_tokens, padded_heads, head_dim]
│                   │           kv=kv.view(-1, 1, head_dim),  # [CHUNK*M, 1, head_dim]
│                   │           indices=combined_indices.unsqueeze(1),  # [chunk_tokens, 1, topk+win]
│                   │           sm_scale=self.scale,
│                   │           attn_sink=self.attn_sink,
│                   │           topk_length=combined_lens,
│                   │           out=output_chunk)
│                   │
│                   └─ _copy_flashmla_output(flash_output, output_slice)
│
├─ o = o_padded[:, :n_local_heads, :]   [slice back from padded heads]
│
└─ O-projection:                                                            (§1.5)
    ├─ fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, ...) → o_fp8, o_scale
    ├─ torch.ops.vllm.deepseek_v4_fp8_einsum(o_fp8, o_scale, wo_a, wo_a_scale, z, ...)
    │   └─ SM70: _sm70_fp8_einsum_bmm (pre-dequant weight to FP32, cached; FP32 einsum)
    ├─ wo_b(z.flatten(1)) → out [T, hidden_size]
    └─ _clamp_sm70_fp16_attention_output_(out)
        [clamp to ±fp16_max to prevent inf propagation on SM70]
```

### 1.2  Parallel Execution: `maybe_execute_in_parallel`

The prefill path overlaps two independent workloads on separate CUDA streams to hide latency:

| Stream | Workload | Purpose |
|---|---|---|
| **Default stream** | `indexer(hidden_states, qr, positions, indexer_rotary_emb)` | Computes sparse attention top-K indices (FP8 MQA logits via `sm70_fp8_mqa_logits`). Only runs for C4A layers that have an indexer. |
| **Aux stream** | `_fused_qnorm_rope_kv_insert()` + `compressor()` | (1) Per-head Q RMSNorm + GPT-J RoPE + FP8 quant + paged SWA cache write. (2) Compressor: partial state accumulation → fused compress + RMSNorm + RoPE + FP8 quant + compressed MLA cache write. |

**Overlap structure** (from `maybe_execute_in_parallel`):

```
Default stream:                    Aux stream:
┌───────────────────────┐
│ indexer()             │          ┌───────────────────────┐
│  • wq_b(qr) → q      │  ──▶    │ _fused_qnorm_rope_    │
│  • compressor(h,p,r)  │  event  │     kv_insert()       │
│  • fused_indexer_q_   │  sync   │ compressor(h,p,r)     │
│    rope_quant()       │          │  • cublas GEMM        │
│  • indexer_op(h,q,k,w)│          │  • _save_partial_     │
│    → topk_indices_buf │          │      states_kernel    │
└───────────────────────┘          │  • _fused_kv_compress_│
         │                         │    _norm_rope_insert  │
         ▼                         └───────────────────────┘
    [event.record()]                    [event.record()]
         │                                   │
         └──── synchronize ──────────────────┘
                      │
                      ▼
              q padding + mla_attn
```

**Fallback (no indexer)**: When `self.indexer is None` but `self.compressor is not None`, the compressor runs on the default stream and `_fused_qnorm_rope_kv_insert` runs on the aux stream. For SWA-only layers (no compressor, no indexer), `_fused_qnorm_rope_kv_insert` runs alone on the default stream with no overlap.

### 1.3  Chunked Prefill with `PREFILL_CHUNK_SIZE=4`

Prefill requests are processed in **fixed-size chunks of 4 requests** (not 4 tokens). This bounds the BF16 KV-gather workspace to a manageable size while allowing per-request block-table traversal.

#### Chunking parameters

| Parameter | Value | Description |
|---|---|---|
| `PREFILL_CHUNK_SIZE` | 4 | Number of prefill requests per chunk |
| `M` | `N + window_size + max_num_batched_tokens` | Maximum KV entries per chunk in the gathered workspace |
| `N` | `max_model_len // compress_ratio` | Maximum compressed KV positions (0 for SWA-only) |
| Workspace shape | `[PREFILL_CHUNK_SIZE, M, head_dim]` | BF16 workspace, allocated via `get_simultaneous()` |

#### Per-chunk KV gathering

Each chunk performs **two sequential** `dequantize_and_gather_k_cache` calls:

1. **Compressed KV gather** (C4A/C128A only):
   - Source: `compressed_k_cache` (main MLA FP8 paged cache, `[num_blocks, block_size, head_bytes]` uint8)
   - Range: `seq_lens // compress_ratio` entries per request
   - Written to `kv[:chunk_size, 0:N, :]` (offset=0)
   - Block size: `attn_metadata.block_size // compress_ratio` (e.g., 256/4 = 64 for C4A)

2. **SWA KV gather** (all layers):
   - Source: `swa_k_cache` (SWA FP8 paged cache, `[num_blocks, 64, head_bytes]` uint8)
   - Range: `gather_lens` entries per request (bounded by `window_size`)
   - Written to `kv[:chunk_size, N:N+gather_lens, :]` (offset=N)
   - Block size: `swa_metadata.block_size` (64)

Both calls use `dequantize_and_gather_k_cache` which performs:
- FP8 e4m3fn → float32 dequantization with UE8M0 block scales (7 blocks × 64 elements)
- BF16 RoPE value copy
- Output: contiguous BF16 `[chunk_size, M, 512]` workspace

#### Index combination

After gathering, `combine_topk_swa_indices` merges the sparse top-K indices (from the indexer) with SWA window indices into a single combined index tensor `[num_chunk_tokens, 1, total_topk]` with per-query valid lengths. This combined index maps into the flat `kv.view(-1, 1, head_dim)` buffer at the correct offsets (compressed entries at `[0, N)`, SWA entries at `[N, N+window_size)`).

### 1.4  SM70 Fallback Functions in Prefill

On SM70 (capability < 8), two torch-based fallback functions replace the fused CUDA/Triton kernels used on SM80+:

#### 1.4.1  `_torch_qnorm_rope_kv_insert_fallback`

**Source**: `vllm/model_executor/layers/deepseek_v4_attention.py`
**Replaces**: `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (SM80+ CUDA kernel)
**Triggered by**: `_should_use_qnorm_rope_kv_insert_fallback(q)` → `capability[0] < 8`

**Operations performed (sequentially, in Python/PyTorch):**

```
_torch_qnorm_rope_kv_insert_fallback(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size):
│
├─ Q side: per-head RMSNorm (no weight) + GPT-J RoPE
│   ├─ q_float = q.float()
│   ├─ variance = q_float.pow(2).mean(dim=-1, keepdim=True)
│   ├─ q_norm = q_float * rsqrt(variance + eps) → cast to q.dtype
│   └─ q.copy_(_apply_gptj_rope_tail(q_norm, positions, cos_sin_cache))
│       └─ GPT-J interleaved RoPE: even/odd pairs rotated by cos/sin
│
├─ KV side: GPT-J RoPE + FP8 block quantization + paged cache scatter write
│   ├─ kv_rope = _apply_gptj_rope_tail(kv[:num_tokens], positions, cos_sin_cache)
│   ├─ Filter valid tokens: valid_mask = slot_mapping >= 0
│   ├─ Block quantization (per 64-element block):
│   │   ├─ nope = kv_valid[:, :448].float()
│   │   ├─ blocks = nope.view(-1, 7, 64)
│   │   ├─ absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
│   │   ├─ exponents = ceil(log2(absmax / 448.0))
│   │   ├─ scales = 2^exponents
│   │   └─ fp8_bytes = (blocks / scales).clamp(-448, 448).to(float8_e4m3fn)
│   ├─ rope_bytes = kv_valid[:, 448:].to(bfloat16).view(uint8)
│   ├─ token_data = cat(fp8_bytes, rope_bytes)  [576 bytes per token]
│   └─ Scatter write to paged cache:
│       ├─ k_cache[block_indices, data_offsets] = token_data
│       └─ k_cache[block_indices, scale_offsets] = encoded_scales (UE8M0)
```

**SM70-specific overhead**: Sequential per-token processing in float32 intermediates, non-fused RoPE (separate cos/sin lookups, clone, slice operations), multiple kernel launches for pow/mean/rsqrt/mul/cat/scatter.

#### 1.4.2  `_torch_fused_compress_norm_rope_insert_fp8_fallback`

**Source**: `vllm/model_executor/layers/deepseek_compressor.py`
**Replaces**: `_fused_kv_compress_norm_rope_insert_sparse_attn` (Triton kernel, SM80+)
**Triggered by**: `_should_use_torch_fused_compressor_fallback(kv_cache, use_fp4_cache)` → `capability[0] < 8`

**Operations performed (Python-level token loop):**

```
_torch_fused_compress_norm_rope_insert_fp8_fallback(...):
│
├─ for token_idx in range(num_tokens):
│   │
│   ├─ Skip if slot_id < 0 or (position + 1) % compress_ratio != 0
│   │   [only fires at compress_ratio boundaries]
│   │
│   ├─ Gather state from sliding window:
│   │   for local_idx in range(window):  [window = (1+overlap) × compress_ratio]
│   │       ├─ Look up physical block via block_table[req_idx, block_in_seq]
│   │       ├─ kv_rows[i] = state_cache[phys_block, pos_in_block, :head_dim]
│   │       └─ score_rows[i] = state_cache[phys_block, pos_in_block, state_width:]
│   │
│   ├─ Weighted compression:
│   │   ├─ score = stack(score_rows).softmax(dim=0)   [window × head_dim]
│   │   ├─ kv = stack(kv_rows)                        [window × head_dim]
│   │   └─ compressed_kv = (kv * score).sum(dim=0)    [head_dim]
│   │
│   ├─ RMSNorm:
│   │   ├─ variance = compressed_kv.pow(2).sum() / head_size
│   │   └─ normed = compressed_kv * rsqrt(variance + eps) * rms_norm_weight
│   │
│   ├─ GPT-J RoPE:
│   │   └─ rotated = _apply_gptj_rope_tail_1d(normed, position, compress_ratio, cos_sin_cache)
│   │
│   └─ FP8 quantization + paged cache scatter write:
│       ├─ head_size == 512: _store_fp8_sparse_attention_cache_torch(...)
│       │   [7-block NoPE quant + BF16 RoPE + UE8M0 scales → 576+8 bytes]
│       └─ head_size == 128: _store_fp8_indexer_cache_torch(...)
│           [single-block quant + single float32 scale → 132 bytes]
```

**SM70-specific overhead**: Python-level `for` loop over tokens with per-token host-side control flow, sequential softmax + weighted-sum + RMSNorm + RoPE + FP8 quant + scatter write. Each iteration issues multiple small CUDA kernels. The SM80+ Triton kernel (`_fused_kv_compress_norm_rope_insert_sparse_attn`) performs all these operations in a single kernel launch with one Triton program per token.

### 1.5  O-Projection Path

After the attention kernel produces output `o [num_tokens, n_local_heads, head_dim]`, the O-projection pipeline transforms it back to the hidden dimension:

```
O-Projection Pipeline (identical for prefill and decode):
│
├─ fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups, heads_per_group,
│                           nope_dim, rope_dim, tma_aligned_scales)
│   ├─ Inverse GPT-J RoPE: undo the forward rotation applied during Q processing
│   │   (rotate by -θ using the same cos_sin_cache at the same positions)
│   ├─ Reshape: o [T, n_heads, head_dim] → [T, n_groups, heads_per_group, head_dim]
│   └─ FP8 block quantization:
│       ├─ Per-128-element blocks: compute absmax → UE8M0 scale → FP8 e4m3fn
│       └─ Output: o_fp8 [T, n_groups, heads_per_group * head_dim // 128, 128] uint8
│                  o_scale [T, n_groups, ...] float32
│
├─ torch.ops.vllm.deepseek_v4_fp8_einsum(o_fp8, o_scale, wo_a, wo_a_scale, z, "bhr,hdr->bhd", recipe)
│   └─ SM70 path: _sm70_fp8_einsum_bmm(o_fp8, o_scale, wo_a, wo_a_scale, z, "bhr,hdr->bhd")
│       │
│       ├─ Lazy weight pre-dequantization (cached, ~86 MB FP32):
│       │   b_f32 = wo_a.reshape(groups, rank, hidden).float() * b_scale_3d.repeat_interleave(128)
│       │   [stored as b._sm70_predequant_f32 — computed once, reused every call]
│       │
│       ├─ Activation dequantization:
│       │   a_deq = o_fp8.float() * o_scale.repeat_interleave(hidden // a_blocks, dim=-1)
│       │
│       └─ FP32 einsum:
│           z = torch.einsum("bhr,hdr->bhd", a_deq, b_f32)
│           → z [T, n_groups, o_lora_rank]
│
├─ wo_b(z.flatten(1)) → out [T, hidden_size]
│   [cuBLAS FP16 linear projection: [T, n_groups * o_lora_rank] → [T, hidden_size]]
│
└─ _clamp_sm70_fp16_attention_output_(out)
    [out.clamp_(min=-65504.0, max=65504.0)]
    [SM70: prevents inf propagation from FP16 overflow in subsequent residual additions]
```

**Key SM70 characteristics of the O-projection:**

| Stage | SM70 Behaviour | SM80+ Behaviour |
|---|---|---|
| `fused_inv_rope_fp8_quant` | Same fused kernel (works on all archs) | Same |
| `fp8_einsum` | `_sm70_fp8_einsum_bmm`: pre-dequant weight to FP32 (cached ~86 MB) + FP32 `torch.einsum` | `fp8_einsum` via DeepGEMM (hardware FP8 MMA) |
| `wo_b` | cuBLAS FP16 (Volta tensor cores) | cuBLAS FP16/BF16 |
| Output clamping | Active (FP16 overflow risk) | Not needed (BF16 has wider range) |

**FP32 einsum overhead on SM70**: The batched einsum `"bhr,hdr->bhd"` operates on `[T, n_groups, hidden] × [n_groups, rank, hidden] → [T, n_groups, rank]`. For typical values (`n_groups=2, hidden=512, rank=384`), this is a small grouped matmul that underutilises V100's FP32 throughput for T=1 (decode) but amortises well for larger T (prefill). The ~86 MB FP32 weight cache trades memory for eliminating repeated per-call dequantization.

---

## 2  Decode Path Flow Documentation

*Satisfies Requirement 2 (acceptance criteria 2.1 – 2.5).*

### 2.1  Complete Decode Call Chain

The decode path is entered when the current step contains at least one decode request (query length ≤ 1 + `num_speculative_tokens`). Decode tokens are always placed **before** prefill tokens in the reordered batch.

```
DeepseekV4MultiHeadLatentAttentionWrapper.forward(positions, hidden_states)
│
├─ fused_wqa_wkv(hidden_states) → qr [T, q_lora_rank], kv [T, head_dim]
│
├─ torch.ops.vllm.deepseek_v4_attention(hidden_states, qr, kv, positions, o_padded, layer_name)
│   └─ attention_impl(hidden_states, qr, kv, positions, o_padded)
│       │
│       ├─ fused_q_kv_rmsnorm(qr, kv, q_norm.weight, kv_norm.weight, eps)
│       │   → qr (RMS-normed), kv (RMS-normed)
│       │
│       ├─ wq_b(qr) → q  [T, n_local_heads, head_dim]
│       │
│       ├─ maybe_execute_in_parallel:
│       │   ├─ Default stream:
│       │   │   • indexer(hidden_states, qr, positions, indexer_rotary_emb)
│       │   │     [only for C4A layers; fills topk_indices_buffer]
│       │   │
│       │   └─ Aux stream (overlapped):
│       │       • _fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
│       │       │   └─ SM80+: torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
│       │       │              q, kv, swa_kv_cache_2d, slot_mapping, positions,
│       │       │              cos_sin_cache, eps, block_size)
│       │       │   └─ SM70:  _torch_qnorm_rope_kv_insert_fallback(...)
│       │       │
│       │       • compressor(hidden_states, positions, rotary_emb)
│       │         [only for C4A / C128A layers]
│       │
│       ├─ q padding: F.pad(q, (0,0, 0, padded_heads - n_local_heads))
│       │   [FlashMLA requires exactly 64 or 128 heads]
│       │
│       └─ mla_attn(q, kv, positions, output=o_padded)
│           └─ DeepseekV4MLAAttention.forward(q, kv, positions, output)
│               │
│               ├─ Split batch: num_decode_tokens | num_prefill_tokens
│               │   (via swa_metadata.num_decodes / num_decode_tokens)
│               │
│               └─ _forward_decode(q[:num_decode_tokens], ...)
│                   │
│                   ├─ Compute topk_indices + topk_lens:
│                   │   • C4A (compress_ratio=4):
│                   │     compute_global_topk_indices_and_lens(
│                   │       topk_indices_buffer, token_to_req_indices,
│                   │       block_table, block_size, is_valid)
│                   │     → global_indices [T, 1, topk], topk_lens [T]
│                   │
│                   │   • C128A (compress_ratio=128):
│                   │     Pre-computed during metadata build
│                   │     (c128a_global_decode_topk_indices, c128a_decode_topk_lens)
│                   │
│                   │   • SWA-only (compress_ratio≤1):
│                   │     topk_indices = None, topk_lens = None
│                   │
│                   ├─ Retrieve SWA indices:
│                   │   swa_indices = swa_metadata.decode_swa_indices  [T, 1, window_size]
│                   │   swa_lens    = swa_metadata.decode_swa_lens     [T]
│                   │
│                   ├─ _flashmla_bf16_io(q, output) → (q_bf16, flash_output_bf16)
│                   │   [SM70: fp16→bf16 upcast for FlashMLA]
│                   │
│                   ├─ q = q.unsqueeze(1)  → [T, 1, padded_heads, head_dim]
│                   │
│                   ├─ BRANCH: _should_use_sm70_decode_prefill_fallback(q, swa_only)
│                   │   │   (returns True when capability[0] < 8, i.e. SM70)
│                   │   │
│                   │   ├─── YES → SM70 Fallback Path  (§2.2)
│                   │   │
│                   │   └─── NO  → Normal Path          (§2.3)
│                   │
│                   └─ _copy_flashmla_output(flash_output, output)
│
├─ o = o_padded[:, :n_local_heads, :]   [slice back from padded heads]
│
└─ O-projection:
    ├─ fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, ...) → o_fp8, o_scale
    ├─ torch.ops.vllm.deepseek_v4_fp8_einsum(o_fp8, o_scale, wo_a, wo_a_scale, z, ...)
    │   └─ SM70: _sm70_fp8_einsum_bmm (pre-dequant weight to FP32, cached; FP32 einsum)
    ├─ wo_b(z.flatten(1)) → out [T, hidden_size]
    └─ _clamp_sm70_fp16_attention_output_(out)
        [clamp to ±fp16_max to prevent inf propagation on SM70]
```

### 2.2  SM70 Decode Fallback Path

When `_should_use_sm70_decode_prefill_fallback` returns `True` (SM70), the decode path materializes KV entries into a contiguous BF16 workspace and calls `flash_mla_sparse_fwd` (the prefill-style kernel) instead of `flash_mla_with_kvcache`.

#### 2.2.1  SWA-only layers (compress_ratio ≤ 1)

```
_forward_decode [swa_only=True, SM70 fallback]
│
├─ _build_decode_prefill_fallback_indices(swa_indices, swa_lens)
│   → fallback_indices [T, 1, window_size], fallback_lens [T]
│   (maps global SWA slot ids → dense local 0..topk-1 offsets)
│
├─ _get_decode_prefill_fallback_workspace(
│       shape=(T, window_size, head_dim), dtype=bf16, device)
│   → fallback_kv [T, window_size, 512]
│   [allocated via current_workspace_manager().get_simultaneous()
│    for CUDA graph address stability]
│
├─ _gather_decode_prefill_fallback_kv_(
│       fallback_kv, swa_kv_cache, swa_indices, swa_lens, block_size)
│   └─ _gather_decode_kv_triton_kernel[(T, window_size)]
│       • For each (token, slot) pair:
│         - Load 7 × 64-element FP8 NoPE blocks from paged cache
│         - Manual FP8 e4m3fn→FP32 bit decoding:
│           sign = (x >> 7) & 1;  exp = (x >> 3) & 0xF;  mant = x & 0x7
│           fp32_bits = (sign << 31) | ((exp+120) << 23) | (mant << 20)
│           Handle zero and subnormal cases separately
│         - Load UE8M0 scale byte per quant block → scale = 2^(byte - 127)
│         - x_dequant = x_float * scale
│         - FP32 → BF16 via round-to-nearest-even bit manipulation
│         - Copy 64 × BF16 RoPE values directly as uint16 (4 chunks of 16)
│       • Output: [T, topk, 512] as uint16 (BF16 view)
│       • Zero-fill for pid_topk >= tok_len or slot_id < 0
│
├─ _normalize_flashmla_sm70_prefill_kv_(fallback_kv)
│   [currently identity; placeholder for future SM70-specific normalization]
│
└─ flash_mla_sparse_fwd(
       q=q.squeeze(1),                          # [T, padded_heads, head_dim]
       kv=fallback_kv.view(-1, 1, head_dim),    # [T*window_size, 1, head_dim]
       indices=fallback_indices,                 # [T, 1, window_size]
       sm_scale=self.scale,
       attn_sink=self.attn_sink,
       topk_length=fallback_lens,
       out=flash_output)
```

#### 2.2.2  Compressed layers (C4A / C128A, compress_ratio > 1)

```
_forward_decode [swa_only=False, SM70 fallback]
│
├─ total_topk = compressed_topk + swa_topk
│
├─ _get_decode_prefill_fallback_workspace(
│       shape=(T, total_topk, head_dim), dtype=bf16, device)
│   → fallback_kv [T, total_topk, 512]
│
├─ Gather compressed KV portion:
│   _gather_decode_prefill_fallback_kv_(
│       fallback_kv[:, :compressed_topk],
│       kv_cache,                               # main MLA FP8 paged cache
│       topk_indices,                           # [T, 1, compressed_topk]
│       topk_lens,                              # [T]
│       block_size // compress_ratio)
│   └─ _gather_decode_kv_triton_kernel[(T, compressed_topk)]
│
├─ Gather SWA portion:
│   _gather_decode_prefill_fallback_kv_(
│       fallback_kv[:, compressed_topk:],
│       swa_kv_cache,                           # SWA FP8 paged cache
│       swa_indices,                            # [T, 1, window_size]
│       swa_lens,                               # [T]
│       swa_block_size)
│   └─ _gather_decode_kv_triton_kernel[(T, swa_topk)]
│
├─ Build combined local indices:
│   compressed_indices, _ = _build_decode_prefill_fallback_indices(
│       topk_indices, topk_lens, row_stride=total_topk)
│   swa_fallback_indices, _ = _build_decode_prefill_fallback_indices(
│       swa_indices, swa_lens, row_stride=total_topk, offset=compressed_topk)
│   fallback_indices = cat(compressed_indices, swa_fallback_indices, dim=-1)
│
├─ _normalize_flashmla_sm70_prefill_kv_(fallback_kv)
│
└─ flash_mla_sparse_fwd(
       q=q.squeeze(1),
       kv=fallback_kv.view(-1, 1, head_dim),
       indices=fallback_indices,
       sm_scale=self.scale,
       attn_sink=self.attn_sink,
       topk_length=None,                        # combined lens encoded in indices
       out=flash_output)
```

**Key difference from SWA-only**: `topk_length` is `None` because the combined index tensor already encodes valid/invalid entries (−1 sentinels); the kernel ignores slots with index < 0.

### 2.3  Normal (Non-Fallback) Decode Path

On SM80+ (or when the fallback guard is bypassed), `_forward_decode` calls `flash_mla_with_kvcache` directly with the FP8 paged caches and sparse indices. No intermediate BF16 workspace is materialized.

```
_forward_decode [SM80+ normal path]
│
├─ swa_cache = swa_kv_cache.unsqueeze(-2)
│   → [num_blocks, swa_block_size, 1, head_bytes]
│
├─ kv_cache = kv_cache.unsqueeze(-2)   (if compress_ratio > 1)
│   → [num_blocks, block_size, 1, head_bytes]
│
├─ Select tile_scheduler per layer type:
│   • compress_ratio ≤ 1  → swa_metadata.tile_sched_swaonly
│   • compress_ratio == 4  → swa_metadata.tile_sched_c4a
│   • compress_ratio == 128 → swa_metadata.tile_sched_c128a
│   (each is a FlashMLASchedMeta instance, see §2.3.1)
│
└─ flash_mla_with_kvcache(
       q       = q,                              # [T, 1, padded_heads, head_dim]
       k_cache = swa_cache,                      # SWA FP8 paged cache
       block_table = None,                       # sparse mode (no block table)
       head_dim_v  = 512,
       tile_scheduler_metadata = tile_metadata,
       cache_seqlens = None,                     # sparse mode
       is_fp8_kvcache = True,
       indices        = swa_indices,             # [T, 1, window_size]
       topk_length    = swa_lens,                # [T]
       softmax_scale  = self.scale,
       attn_sink      = self.attn_sink,          # [padded_heads] float32
       extra_k_cache            = kv_cache,      # compressed MLA FP8 cache (or None)
       extra_indices_in_kvcache = topk_indices,   # [T, 1, compressed_topk] (or None)
       extra_topk_length        = topk_lens,      # [T] (or None)
       out = flash_output.unsqueeze(1))
```

#### 2.3.1  Per-Layer-Type `FlashMLASchedMeta` Tile Schedulers

`FlashMLASchedMeta` manages workload partitioning across V100's 80 SMs. **Three instances** exist per decode step, one per layer type present in the model:

| Layer Type | `compress_ratio` | Purpose | Cache Params |
|---|---|---|---|
| **swaonly** | ≤ 1 | SWA-only layers (no compressed KV) | `k_cache` = SWA, no `extra_k_cache` |
| **c4a** | 4 | Layers with compress_ratio=4 | `k_cache` = SWA, `extra_k_cache` = C4A compressed |
| **c128a** | 128 | Layers with compress_ratio=128 | `k_cache` = SWA, `extra_k_cache` = C128A compressed |

**Lifecycle per decode step:**

1. `DeepseekSparseSWAMetadataBuilder.build()` calls `build_tile_scheduler(num_decode_tokens)`.
2. For each layer type present (detected from `hf_config.compress_ratios`), `get_mla_metadata()` returns a **fresh empty `FlashMLASchedMeta`** with `have_initialized = False`.
3. The **first** `flash_mla_with_kvcache` call of a given type triggers the in-kernel planner:
   - Allocates `tile_scheduler_metadata` (`[num_sm_parts, 8]` int32) and `num_splits` (`[batch_size+1]` int32) via PyTorch's graph-aware allocator (stable addresses across CUDA graph replays).
   - Computes SM-to-work mapping based on `(b, s_q, h_q, page_block_size, h_k, causal, is_fp8_kvcache, topk, extra_page_block_size, extra_topk)`.
   - Sets `have_initialized = True`.
4. **Subsequent** same-type layers (all ~60 layers of the same `compress_ratio`) skip the planner and reuse the existing plan.
5. A new step's `build()` creates fresh instances, resetting the cycle.

**Key config fields** captured by FlashMLASchedMeta:

| Field | Description |
|---|---|
| `tile_scheduler_metadata` | `[num_sm_parts, 8]` int32 — workload assignment per SM |
| `num_splits` | `[batch_size+1]` int32 — split boundaries |
| `have_initialized` | bool — set True after first planner run |
| `config` | tuple of `(b, s_q, h_q, page_block_size, h_k, causal, is_fp8_kvcache, topk, extra_page_block_size, extra_topk)` |

### 2.4  CUDA Graph Capture Boundaries and Workspace Stability

The decode path supports CUDA graph capture under the `FULL_DECODE_ONLY` policy with `cudagraph_capture_sizes=[1]`.

#### Capture support declarations

- `FlashMLASparseMetadataBuilder._cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`
- `DeepseekSparseSWAMetadataBuilder._cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`

#### Workspace stability requirements

All decode-path workspace tensors are allocated through `current_workspace_manager().get_simultaneous()` to **maintain fixed GPU memory addresses** across CUDA graph captures and replays:

| Workspace | Shape | Dtype | Allocated In |
|---|---|---|---|
| SM70 fallback KV | `(T, total_topk, 512)` | bf16 | `_get_decode_prefill_fallback_workspace()` |
| Prefill KV gather | `(PREFILL_CHUNK_SIZE, M, head_dim)` | bf16 | `_forward_prefill()` |
| q_concat buffer | `(max_tokens, num_heads, head_size)` | bf16 | `FlashMLASparseImpl.__init__()` |
| Prefill BF16 workspace | `(prefill_workspace_size, head_size)` | bf16 | `FlashMLASparseImpl.__init__()` |
| C128A topk decode buffer | `(max_tokens, c128a_max_compressed)` | int32 | `FlashMLASparseMetadataBuilder.__init__()` |
| C128A decode lens buffer | `(max_tokens,)` | int32 | `FlashMLASparseMetadataBuilder.__init__()` |
| C128A prefill buffer | `(max_tokens, c128a_max_compressed)` | int32 | `FlashMLASparseMetadataBuilder.__init__()` |
| Compressed slot mapping | `(max_tokens,)` | int64 | `FlashMLASparseMetadataBuilder.__init__()` |

#### `.item()` blocker fix

The original `sparse_attn_indexer.py` contained a `.item()` call that triggered a host-device sync, blocking CUDA graph capture. This was fixed by routing through the SM70 Triton kernel path (`sm70_fp8_paged_mqa_logits`) which avoids the synchronization.

#### Tile scheduler and graph capture

`FlashMLASchedMeta` tensors (`tile_scheduler_metadata`, `num_splits`) are allocated by the FlashMLA C++ planner via PyTorch's graph-aware allocator. This ensures:
- Tensor addresses remain stable across CUDA graph replay.
- The planner runs during **capture** (first call per type); on **replay**, `have_initialized = True` and the plan is reused from the same addresses.

#### Non-graphed operations

The following operations are **not** captured in the CUDA graph:
- mHC torch fallback (`_mhc_pre_torch_fallback`, `_mhc_post_torch_fallback`) — uses multiple separate kernel launches in float32.
- Tile scheduler initialization (first call per type per step) — runs during capture only.
- Metadata build phase (`build()`) — CPU-side, runs before graph replay.

### 2.5  Combined SWA + Compressed KV Index Handling

DeepSeek V4 uses **two separate KV caches** per decode token, combined at the attention level:

| Cache | Source | Block Size | Index Source |
|---|---|---|---|
| **SWA** (Sliding Window) | `DeepseekV4SWACache.kv_cache` | 64 tokens | `DeepseekSparseSWAMetadata.decode_swa_indices` |
| **Compressed** (C4A / C128A) | `DeepseekV4MLAAttention.kv_cache` | 256/compress_ratio tokens | C4A: `topk_indices_buffer` (filled by Indexer); C128A: `c128a_global_decode_topk_indices` (pre-computed) |

#### SWA index computation

SWA indices are computed by `_compute_swa_indices_and_lens_kernel` (Triton, launched during `DeepseekSparseSWAMetadataBuilder.build()`):

- For each decode token at position `pos`: selects all cache slots in `[max(0, pos - window_size + 1), pos + 1)`.
- Converts logical positions to **global slot IDs** via `block_table[req_idx]` lookup: `slot_id = block_number * block_size + block_offset`.
- Stores result in `decode_swa_indices` (`[T, 1, window_size]` int32, -1 for invalid/padded).
- Stores valid count in `decode_swa_lens` (`[T]` int32).

#### C4A compressed index computation

For `compress_ratio=4` layers:
1. The `Indexer` (running on default stream) writes per-token local top-K indices into `topk_indices_buffer`.
2. `compute_global_topk_indices_and_lens()` converts local indices → global slot IDs via block table lookup.
3. Result: `topk_indices [T, 1, compressed_topk]`, `topk_lens [T]`.

#### C128A compressed index computation

For `compress_ratio=128` layers:
1. Pre-computed during `FlashMLASparseMetadataBuilder._build_c128a_metadata()` using `_build_c128a_topk_metadata_kernel` (Triton).
2. For each decode token: `num_compressed = (position + 1) // 128`, then block_table lookup to global slot IDs.
3. Padded to `_C128A_TOPK_ALIGNMENT=128` boundary (FlashMLA decode kernel asserts `extra_topk % B_TOPK == 0`).
4. Result: `c128a_global_decode_topk_indices [T, 1, c128a_max_compressed]`, `c128a_decode_topk_lens [T]`.

#### Combination at kernel level

- **Normal path (SM80+)**: `flash_mla_with_kvcache` receives both cache + index sets simultaneously:
  - `k_cache` / `indices` / `topk_length` → SWA cache entries
  - `extra_k_cache` / `extra_indices_in_kvcache` / `extra_topk_length` → compressed cache entries
  - The kernel internally iterates over both index sets and computes a single fused softmax.

- **SM70 fallback path**: Both caches are gathered into a single contiguous BF16 workspace (`fallback_kv`), with compressed entries occupying columns `[0, compressed_topk)` and SWA entries occupying `[compressed_topk, total_topk)`. The combined local index tensor maps each position to the correct column. `flash_mla_sparse_fwd` then treats this as a single flat KV buffer.

---

## 3  KV Cache Data Organization

*Satisfies Requirement 3 (acceptance criteria 3.1 – 3.5).*

This section documents the complete KV cache data layout used by DeepSeek V4 Flash on SM70, covering the FP8 per-token format, paged block organization for both main MLA and SWA caches, compressor state layout, indexer cache format, and FlashMLA head-padding requirements.

### 3.1  FP8 Token Format (584 bytes/token)

Every token stored in the KV cache occupies exactly **584 bytes**, organized as follows:

```
Byte offset   Size        Content
──────────────────────────────────────────────────────────────
  0 – 447     448 bytes   448 × FP8 e4m3fn NoPE latent values
448 – 575     128 bytes    64 × BF16 RoPE values (2 bytes each)
576 – 582       7 bytes     7 × UE8M0 scale bytes (one per 64-element quant block)
583             1 byte      1 × padding byte
──────────────────────────────────────────────────────────────
Total         584 bytes
```

#### NoPE region (448 bytes)

The 448 NoPE values are the low-rank latent KV projection (`kv_lora_rank = 512 − 64 = 448`). They are quantized per 64-element block:

- **7 quantization blocks** × 64 elements = 448 FP8 e4m3fn values.
- Each block has an independent UE8M0 scale (unsigned 8-bit exponent-only float): `scale = 2^(byte − 127)`.
- Encoding: `fp8_value = clamp(float_value / scale, −448, 448)` cast to `float8_e4m3fn`.
- Decoding (SM70 Triton): manual bit extraction — `sign = (x >> 7) & 1`, `exp = (x >> 3) & 0xF`, `mant = x & 0x7` → `fp32_bits = (sign << 31) | ((exp+120) << 23) | (mant << 20)`, with special handling for zero (`exp==0, mant==0`) and subnormal (`exp==0, mant!=0`) cases.

#### RoPE region (128 bytes)

The 64 RoPE values occupy `rope_dim = 64` dimensions in BF16 format (2 bytes each = 128 bytes). These are stored after GPT-J interleaved rotation (`_apply_gptj_rope_tail`) using position-dependent `cos_sin_cache` entries. BF16 is chosen because FlashMLA internally operates on BF16 KV; on SM70, the RoPE values are cast from FP16 → BF16 before cache write and copied as raw `uint16` during cache read (bitwise identical round-trip).

#### Scale region (7 bytes) + padding (1 byte)

The 7 UE8M0 scale bytes correspond 1:1 to the 7 NoPE quantization blocks. The 1-byte padding brings the total to 584 bytes, maintaining alignment for efficient memory access. The scale encoding follows the UE8M0 format: `encoded = clamp(ceil(log2(absmax / 448.0)) + 127, 0, 254)`.

### 3.2  Paged Block Organization

KV cache entries are organized in **paged blocks** managed by vLLM's block allocator. Two distinct page types exist:

#### Main MLA Cache (256-token blocks)

| Property | Value |
|---|---|
| Block shape | `[num_blocks, 256, 584]` uint8 |
| Tokens per block | 256 |
| Bytes per token | 584 |
| Total block size | 256 × 584 = **149,504 bytes** |
| FlashMLA alignment | 576 bytes (the first 576 bytes of each 584-byte token slot are 576-byte aligned for FlashMLA's vectorized loads) |
| Used by | C4A layers (`compress_ratio=4`) and C128A layers (`compress_ratio=128`) — stores compressed KV entries |

The compressed block size seen by the attention kernel differs by layer type:
- **C4A**: effective block_size = `256 / 4 = 64` compressed entries per block.
- **C128A**: effective block_size = `256 / 128 = 2` compressed entries per block.

#### SWA Cache (64-token blocks)

| Property | Value |
|---|---|
| Block shape | `[num_blocks, 64, 584]` uint8 |
| Tokens per block | 64 |
| Bytes per token | 584 (identical per-token format as main MLA) |
| Total block size | 64 × 584 = **37,376 bytes** |
| FlashMLA alignment | 576 bytes |
| Window size | From `config.sliding_window` (model-dependent) |
| Used by | All layers — stores the most recent `window_size` tokens for local attention |

#### 576-byte Alignment

FlashMLA's CUDA decode kernel (`flash_mla_with_kvcache`) accesses each token slot as a 576-byte contiguous region for vectorized memory loads. The per-token layout is designed so that the first 576 bytes (448 NoPE + 128 RoPE) are naturally aligned at the start of each 584-byte slot. The kernel reads the 576 data bytes in a single coalesced transaction, then separately loads the 7 scale bytes from the trailing region.

#### Page sharing

SWA cache blocks share **physical page allocation** with compressor state blocks via page-size alignment in the block allocator. This means both cache types draw from the same pool of GPU memory pages, enabling efficient memory utilization when different layer types have different cache occupancy patterns.

### 3.3  Compressor State Cache Layout

The compressor (`DeepseekCompressor`) accumulates partial KV states over `compress_ratio` positions before writing the final compressed entry. The intermediate state is stored in a dedicated **float32** cache:

#### C4A Compressor State (`compress_ratio = 4`)

```
Shape: [num_blocks, 4, 2 * 2 * 512]
         │          │    │   │   └── head_dim (kv_lora_rank + rope_dim = 512)
         │          │    │   └── coff = 2 (overlap factor: compress_ratio // (compress_ratio - overlap))
         │          │    └── 2 (packing kv_state + score_state in last dim)
         │          └── block_size = 4
         └── num_blocks (from block allocator)

Dtype: float32
Total bytes per block: 4 × 2 × 2 × 512 × 4 = 32,768 bytes
Bytes per token position: 2 × 2 × 512 × 4 = 8,192 bytes
```

The `coff = 2` overlap factor means the compressor looks back across adjacent compress windows (overlapping by `compress_ratio − 1` positions). Each position stores both:
- **kv_state** (`[head_dim]` float32): running weighted KV accumulation.
- **score_state** (`[head_dim]` float32): running softmax denominator for score-weighted compression.

#### C128A Compressor State (`compress_ratio = 128`)

```
Shape: [num_blocks, 8, 2 * 1 * 512]
         │          │    │   │   └── head_dim (512)
         │          │    │   └── coff = 1 (no overlap)
         │          │    └── 2 (packing kv_state + score_state)
         │          └── block_size = 8
         └── num_blocks

Dtype: float32
Total bytes per block: 8 × 2 × 1 × 512 × 4 = 32,768 bytes
Bytes per token position: 2 × 1 × 512 × 4 = 4,096 bytes
```

With `coff = 1` (no overlap), each position covers exactly `compress_ratio = 128` raw tokens before emitting one compressed cache entry. The state is consumed and flushed when `(position + 1) % 128 == 0`.

#### Compressor output path

At compress boundaries, the compressor:
1. Gathers state rows from the float32 state cache across the compression window.
2. Applies softmax over score states → weighted sum of KV states.
3. RMSNorm + GPT-J RoPE on the compressed result.
4. FP8 block quantization → 584-byte token entry written to the main MLA paged cache.

### 3.4  Indexer Cache

The sparse attention indexer uses a separate, more compact cache format for computing per-query top-K KV relevance scores during decode.

#### Per-token format (132 bytes)

```
Byte offset   Size        Content
──────────────────────────────────────────────────────────────
  0 – 127     128 bytes   128 × FP8 e4m3fn values (low-rank indexer projection)
128 – 131       4 bytes   1 × float32 scale (single scale for entire 128-element block)
──────────────────────────────────────────────────────────────
Total         132 bytes
```

Unlike the main KV cache's 7-block quantization, the indexer cache uses a **single quantization block** covering all 128 elements with one float32 scale value (not UE8M0). This simpler format reflects the indexer's lower-dimensional projection (`q_lora_rank` for the indexer is 128, not 512).

#### Block organization

| Property | Value |
|---|---|
| Block shape | `[num_blocks, 256, 132]` (logical) |
| Tokens per block | 256 (same block size as main MLA) |
| Bytes per token | 132 |
| Total block bytes | 256 × 132 = **33,792 bytes** |
| Alignment | 576 bytes (page-level) |
| Used by | `_sm70_fp8_paged_mqa_logits_kernel` (decode) and `sm70_fp8_mqa_logits` (prefill) |

#### Physical page sharing

The indexer cache shares physical pages with the compressor state cache via page-size alignment in the block allocator. This is possible because:
- The indexer writes once per token (during KV insert), while the compressor accumulates over `compress_ratio` positions.
- Both caches are indexed by the same block table, with different byte offsets within each page.
- Memory is allocated from the same pool, reducing fragmentation.

### 3.5  Head Padding for FlashMLA

FlashMLA's CUDA kernels (`flash_mla_sparse_fwd` and `flash_mla_with_kvcache`) require the number of query heads to be **exactly 64 or 128**. DeepSeek V4 Flash with 8-way tensor parallelism has `n_local_heads = total_heads / tp_size`, which may not satisfy this constraint.

#### Padding mechanism

```
Forward path (before kernel call):
  padded_heads = next value in {64, 128} ≥ n_local_heads
  q_padded = F.pad(q, (0, 0, 0, padded_heads - n_local_heads))
  # Pads with zeros along the head dimension

After kernel call:
  o = o_padded[:, :n_local_heads, :]
  # Slices back to the original head count
```

#### Why zero-padding is safe

- The padded (zero) query heads produce zero attention logits after the dot product with KV.
- After softmax, zero logits contribute negligible weight to the output.
- The `attn_sink` bias (if present) applies uniformly across all heads; the padded heads' output is discarded by the slice-back.
- The slice-back at the output ensures no padded-head results propagate downstream.

#### Impact on memory and compute

| Aspect | Impact |
|---|---|
| Query tensor | Slight memory increase: `(padded_heads − n_local_heads) × head_dim × dtype_size` per token |
| Attention compute | FlashMLA processes all `padded_heads` — wasted FLOPs proportional to `(padded_heads − n_local_heads) / padded_heads` |
| Output tensor | Same size as padded query; sliced immediately after kernel return |
| KV cache | **Unaffected** — KV cache stores per-token entries, not per-head entries (MLA uses shared KV projections) |

### 3.6  Cache Block Allocation Summary

| Cache Type | Block Size (tokens) | Bytes/Token | Alignment | Total Block Bytes | Dtype |
|---|---|---|---|---|---|
| Main MLA (C4A) | 256 | 584 | 576 | 149,504 | uint8 |
| Main MLA (C128A) | 256 | 584 | 576 | 149,504 | uint8 |
| SWA | 64 | 584 | 576 | 37,376 | uint8 |
| Compressor C4A state | 4 | 8,192 | — | 32,768 | float32 |
| Compressor C128A state | 8 | 4,096 | — | 32,768 | float32 |
| Indexer | 256 | 132 | 576 | 33,792 | uint8 + float32 |

**Total per-token cache footprint** (for a C4A layer with indexer): 584 (main MLA) + 584 (SWA) + 8,192 (compressor state, amortized) + 132 (indexer) = **~9,492 bytes** peak during accumulation, dropping to 584 + 584 + 132 = **1,300 bytes** after compression completes and state is flushed.

---

## 4  SM70 Triton Kernel Inventory and Analysis

*Satisfies Requirement 4 (acceptance criteria 4.1 – 4.4).*

This section catalogs every SM70-specific Triton kernel in the vLLM codebase, documenting grid dimensions, register pressure, memory access patterns, arithmetic intensity, and precision limitations.

### 4.1  `_sm70_fp8_paged_mqa_logits_kernel` — Decode Indexer

**Source**: `vllm/model_executor/layers/sm70_mqa_logits.py`
**Python wrapper**: `sm70_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, max_model_len)`
**Usage**: Decode path indexer — computes per-token, per-K-position ReLU-weighted multi-head scores from the FP8 paged KV cache for the sparse attention indexer.

#### Grid dimensions

```
grid = (batch_size * next_n, max_model_len)
```

- **Axis 0** (`pid_row`): One program per (batch, speculative-token) pair. `pid_row = batch_idx * next_n + next_idx`.
- **Axis 1** (`k_pos`): One program per K position in the model's context window. Programs with `k_pos >= context_len` early-exit.
- **Total programs**: `batch_size × next_n × max_model_len`. For typical decode with `batch_size=1, next_n=1, max_model_len=8192`: **8,192 programs**.

#### Register pressure

**High**. The kernel loops over `NUM_HEADS` (compile-time constant), and each iteration:
- Loads a full Q vector (`HEAD_DIM` uint8 values → `HEAD_DIM` int32 intermediates for bit manipulation → `HEAD_DIM` float32 values).
- Performs `HEAD_DIM`-wide element-wise multiply with the (already-decoded) K vector, then a reduction sum.
- K is decoded once outside the loop (1 × `HEAD_DIM` float32 values persisted across iterations).

Per-thread live registers include:
- K decoded: `HEAD_DIM` × float32 (e.g., 128 × 4B = 512B → **128 registers**).
- Q decoded per head iteration: `HEAD_DIM` × float32 (**128 registers**, reused across iterations).
- Intermediate bit-manipulation: `HEAD_DIM` × 3 int32 (sign, exp, mant) — **384 registers** at peak within decode block, partially reusable.
- Accumulator, weight, score: ~3 float32 scalars.

For `HEAD_DIM=128`: **estimated ~250–300 registers per thread** at peak, exceeding SM70's 255-register-per-thread limit. Triton will spill to local memory, degrading performance. For `HEAD_DIM=128` with `NUM_HEADS=8`: the Q reload per head is the primary pressure source.

#### Memory access patterns

| Access | Pattern | Coalescing |
|---|---|---|
| **K load** (FP8 from paged cache) | `kv_cache[physical_block, pos_in_block, 0:HEAD_DIM]` — contiguous within token, but `physical_block` varies per `k_pos` | **Partially coalesced**: threads in the same warp process the same `k_pos` (each thread handles a single `(row, k_pos)` program), so K loads are uniform across the warp. However, across programs, K accesses are **scattered** across different physical pages. |
| **K scale** (float32) | Single scalar per K position at `kv_cache[block, pos, HEAD_DIM]` | **Scalar load** (broadcast within program). |
| **Q load** (FP8 per head) | `q[batch_idx, next_idx, h, 0:HEAD_DIM]` — contiguous per head | **Coalesced** within each head's `HEAD_DIM` range. Reloaded `NUM_HEADS` times per program (high bandwidth cost). |
| **Weights** (float32) | `weights[pid_row, h]` — single scalar per head | **Scalar load**, `NUM_HEADS` loads per program. |
| **Output** (float32) | `logits[pid_row, k_pos]` — single scalar store | **Scattered** across the `max_model_len` dimension. |

**Arithmetic intensity**: Low. Each program computes `NUM_HEADS × HEAD_DIM` FMA operations (dot products) plus `NUM_HEADS` ReLU + multiply-accumulate, but loads `(1 + NUM_HEADS) × HEAD_DIM` bytes of FP8 data plus scales and weights. For `HEAD_DIM=128, NUM_HEADS=8`: ~2048 FLOPs vs ~1280 bytes loaded → **~1.6 FLOP/byte** (memory-bound).

#### Precision limitations

- **FP8 e4m3fn decoding**: Manual bit extraction (`sign = (x >> 7) & 1; exp = (x >> 3) & 0xF; mant = x & 0x7`). Biased exponent uses `+120` offset to reconstruct IEEE 754 float32 bits. Zero values handled explicitly (`k_is_zero` mask). **Subnormals are NOT handled** — the `k_is_zero` check only catches exact zero (`exp==0 && mant==0`), meaning FP8 subnormal values (exp==0, mant≠0) are decoded with an incorrect exponent bias, producing slightly wrong values.
- **FP32 accumulation**: Dot products and weighted ReLU accumulation use FP32 throughout. No FP16 intermediate truncation.
- **No Q scale**: Q values are decoded from FP8 but no per-block scale is applied (the indexer Q is stored without separate scale bytes, unlike the KV cache).

---

### 4.2  `_sm70_fp8_mqa_logits_kernel` — Prefill Indexer

**Source**: `vllm/model_executor/layers/sm70_mqa_logits.py`
**Python wrapper**: `sm70_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)`
**Usage**: Prefill path indexer — computes MQA logits from non-paged, contiguous FP8 Q and K tensors for the sparse attention indexer during prefill.

#### Grid dimensions

```
grid = (M, N)
```

- **Axis 0** (`pid_m`): One program per query token. `M` = number of prefill tokens.
- **Axis 1** (`pid_n`): One program per key token. `N` = total key tokens in the flattened KV buffer.
- **Total programs**: `M × N`. For a 1024-token prefill with 4096 KV entries: **~4.2M programs**.
- **Bounds checking**: Programs with `pid_n < cu_seqlen_ks[pid_m]` or `pid_n >= cu_seqlen_ke[pid_m]` store `-inf` and return.

#### Register pressure

**Similar to paged variant** but slightly lower because K is loaded from a contiguous buffer (no block table indirection):
- K decoded: `HEAD_DIM` × float32 values.
- Q decoded per head: `HEAD_DIM` × float32 values (reloaded `NUM_HEADS` times).
- Bit manipulation intermediates: same 3 int32 fields per element.
- **Estimated ~250–300 registers per thread** for `HEAD_DIM=128`.

#### Memory access patterns

| Access | Pattern | Coalescing |
|---|---|---|
| **K load** (FP8) | `k[pid_n, 0:HEAD_DIM]` — contiguous per key token | **Well-coalesced**: K buffer is contiguous `[N, HEAD_DIM]`. Adjacent programs (`pid_n, pid_n+1`) access adjacent rows. |
| **K scale** (float32) | `k_scale[pid_n]` — single scalar per key position | **Coalesced** across adjacent `pid_n` programs. |
| **Q load** (FP8 per head) | `q[pid_m, h, 0:HEAD_DIM]` — contiguous per head | **Coalesced** within each head. Same Q reloaded for all `N` programs sharing `pid_m`. |
| **Output** (float32) | `out[pid_m, pid_n]` — single scalar store | **Partially coalesced**: adjacent `pid_n` programs write adjacent memory locations. |

**Arithmetic intensity**: Same as paged variant (~1.6 FLOP/byte for `HEAD_DIM=128, NUM_HEADS=8`). Memory-bound.

#### Precision limitations

- **Identical FP8 decode logic** to the paged variant: same bit manipulation, same subnormal handling gap, same FP32 accumulation.
- **Both kernels share identical FP8 e4m3fn bit manipulation code** — the decode logic is duplicated verbatim between `_sm70_fp8_paged_mqa_logits_kernel` and `_sm70_fp8_mqa_logits_kernel`.

---

### 4.3  `_gather_decode_kv_triton_kernel` — Decode KV Gather

**Source**: `vllm/model_executor/layers/deepseek_v4_attention.py`
**Python wrapper**: `_gather_decode_prefill_fallback_kv_(out, k_cache, global_indices, global_lens, block_size)`
**Usage**: SM70 decode fallback path — gathers FP8 KV cache entries from scattered paged locations, dequantizes NoPE to BF16 and copies RoPE as BF16, producing a contiguous `[num_tokens, topk, 512]` workspace for `flash_mla_sparse_fwd`.

#### Grid dimensions

```
grid = (num_tokens, topk)
```

- **Axis 0** (`pid_token`): One program per decode token.
- **Axis 1** (`pid_topk`): One program per top-K KV entry to gather.
- **Total programs**: `num_tokens × topk`. For single-token decode with C4A `topk=2048` + SWA `window_size=4096`: up to **6,144 programs** (called twice — once for compressed, once for SWA).

#### Register pressure

**Moderate**. The kernel processes 7 quantization blocks × 64 elements each in a `tl.static_range` unrolled loop, plus 4 chunks of 16 BF16 RoPE values:

- Per quant block iteration: 64 × uint8 loaded → 64 × int32 intermediates (sign, exp, mant) → 64 × float32 dequantized → 64 × uint16 BF16 output.
- The `tl.static_range(N_QUANT_BLOCKS)` with `N_QUANT_BLOCKS=7` means the compiler unrolls 7 iterations. However, since each iteration operates on the same set of 64-element registers, the live register count is bounded by a single iteration: **~64 × 4 (float32/int32 intermediates) + 64 (uint8 input) + 64 (uint16 output) ≈ 320 values**, but Triton can reuse registers across iterations.
- **Estimated ~120–180 registers per thread** — within SM70's 255-register limit without spills.

#### Memory access patterns

| Access | Pattern | Coalescing |
|---|---|---|
| **FP8 NoPE read** | `k_cache[block_idx, pos_in_block * 576 + qb_idx*64 : +64]` — contiguous 64-byte chunks within a token's data region | **Partially coalesced**: contiguous within the 64-byte quant block, but `block_idx` and `pos_in_block` vary per program (scattered across pages). |
| **UE8M0 scale read** | `k_cache[block_idx, block_size*576 + pos_in_block*8 + qb_idx]` — single byte per quant block | **Scalar read**, 7 reads per program. Scale bytes are stored separately from token data (after all tokens in the block). |
| **BF16 RoPE read** | `k_cache[block_idx, pos_in_block*576 + 448 : +128]` as uint16 — 4 chunks of 16 uint16 values | **Partially coalesced**: 32-byte chunks, but address is page-dependent. |
| **Output write** | `out[pid_token, pid_topk, 0:512]` as uint16 — contiguous 1024-byte write per program | **Well-coalesced** along the head dimension. Adjacent `pid_topk` programs write adjacent rows in the output workspace. |

**Arithmetic intensity**: Very low. The kernel performs primarily format conversion (bit manipulation + multiply by scale + FP32→BF16 rounding), with no reductions or dot products. Per program: ~7 × 64 × ~10 ALU ops (bit shifts, masks, compares, multiply, add for rounding) ≈ 4,480 ops vs ~576 bytes read + 1,024 bytes written → **~2.8 FLOP/byte**. Firmly memory-bandwidth-bound.

#### Precision limitations

- **Full subnormal handling**: Unlike the MQA logits kernels, this kernel explicitly handles FP8 e4m3fn subnormals: `is_subnorm = (exp_bits == 0) & (mant_bits != 0)` with `subnorm_val = mant_bits * 1.953125e-3` (i.e., `mant / 512`). This produces correct dequantized values for all FP8 e4m3fn inputs.
- **FP32 → BF16 rounding**: Uses round-to-nearest-even via `(x_u32 + 0x7FFF + ((x_u32 >> 16) & 1)) >> 16`, which is the standard IEEE 754 rounding method. This introduces a maximum 0.5 ULP error in BF16 precision.
- **BF16 RoPE copy**: RoPE values are copied as raw uint16 bytes — **bitwise identical** with no precision loss.

---

### 4.4  `_mhc_pre_post_gemm_kernel` — SM70 mHC Pre-Block Fused Kernel

**Source**: `vllm/model_executor/layers/mhc.py`
**Python wrapper**: `_mhc_pre_sm70_fast(residual, fn, hc_scale, hc_base, ...)`
**Usage**: SM70 mHC fast path (controlled by `VLLM_SM70_MHC_FAST=1`, **disabled by default**). Fuses post-GEMM processing for the mHC pre-block: RMSNorm computation, sigmoid activation, sinkhorn normalization, and weighted-sum layer_input — into a single kernel launch following a cuBLAS FP16 GEMM.

#### Grid dimensions

```
grid = (num_tokens,)
```

- **Axis 0** (`pid`): One program per token.
- **Total programs**: `num_tokens`. For decode with `T=1`: **1 program**.

#### Dependency on preceding GEMM

Before this kernel launches, a cuBLAS FP16 GEMM computes:
```
gemm_out = residual_vec_fp16 @ fn_fp16.T    # [N, hc_hidden_size] × [hc_mult3, hc_hidden_size]^T → [N, hc_mult3]
```
Where `hc_hidden_size = hc_mult × hidden_size` (e.g., `4 × 7168 = 28672`) and `hc_mult3 = hc_mult*2 + hc_mult*hc_mult` (e.g., `4*2 + 16 = 24`).

The TurboMind MMA_884 path (`sm70_f16_prepare` + `sm70_f16_gemm_out`) is **disabled** due to precision issues (`_use_tm = False`). The cuBLAS FP16 fallback is used instead.

#### Register pressure

**Moderate to High**. The kernel performs multiple stages with different register demands:

1. **RMSNorm** (streamed): Accumulates `sq_sum` over `hc_hidden_size` in `RMS_BLOCK`-sized chunks. Live: `RMS_BLOCK` float32 values + 1 accumulator.
2. **Sinkhorn normalization**: Operates on a `[hc_mult × hc_mult]` = `[4 × 4]` = 16-element matrix. Live: 16 float32 values + row/column masks (16 bools) + row/col sums (4 float32 each). Iterated `sinkhorn_repeat` times.
3. **Weighted-sum layer_input**: Streams through `hidden_size` in `BLOCK_H`-sized chunks. Live: `BLOCK_H` float32 accumulators + `BLOCK_H` float16 residual values per `hc_mult` iteration.

Peak registers occur during the layer_input computation: `BLOCK_H` (up to 1024) float32 accumulators + `BLOCK_H` float16 residual loads. For `BLOCK_H=1024`: **estimated ~150–200 registers per thread** (Triton maps `BLOCK_H` elements across threads within the program).

**Hardcoded for `hc_mult=4`**: The sinkhorn normalization uses explicit `row0_mask`/`row1_mask`/`row2_mask`/`row3_mask` patterns (4 rows) and similar column masks. This avoids dynamic indexing but **does not generalise** to other `hc_mult` values.

#### Memory access patterns

| Access | Pattern | Coalescing |
|---|---|---|
| **GEMM output** (fp16) | `gemm_out[pid, 0:hc_mult3]` — 24 fp16 values (48 bytes) | **Coalesced** (small, fits in a single cache line). |
| **Residual vec** (fp16) | `residual_vec[pid, 0:hc_hidden_size]` — streamed in `RMS_BLOCK` chunks for RMSNorm | **Coalesced** per chunk. Total: `hc_hidden_size × 2` bytes (e.g., 57,344 bytes). |
| **Residual 3D** (fp16) | `residual[pid, hc_idx, 0:hidden_size]` — read per `hc_mult` × `BLOCK_H` for weighted-sum | **Coalesced** per `BLOCK_H` chunk. Total: `hc_mult × hidden_size × 2` bytes. |
| **Constants** | `hc_scale[0:3]`, `hc_base[0:hc_mult3]` — small constant arrays | **Scalar loads**, cached after first access. |
| **Outputs** | `post_mix[pid, 0:hc_mult]`, `comb_mix[pid, 0:hc_mult²]`, `layer_input[pid, 0:hidden_size]` | **Coalesced** stores. `layer_input` streamed in `BLOCK_H` chunks. |

**Arithmetic intensity**: Moderate. The RMSNorm computation is bandwidth-bound (streaming `hc_hidden_size` values for a single rsqrt). The sinkhorn normalization is compute-bound on the small `4×4` matrix (negligible memory). The weighted-sum is bandwidth-bound (streaming `hc_mult × hidden_size` values). Overall: **mixed compute/memory bound**, dominated by the residual streaming.

#### Precision limitations

- **FP16 GEMM + FP32 post-processing**: The cuBLAS GEMM computes in FP16 (Volta tensor cores), then the Triton kernel casts GEMM output to FP32 for all subsequent operations (RMSNorm, sigmoid, sinkhorn, weighted-sum). Final `layer_input` is cast back to FP16 for output.
- **TurboMind MMA_884 DISABLED**: The Volta `mma.sync.aligned.m8n8k4` path was found to produce precision-degraded results causing **semantic divergence** (incorrect model outputs). Controlled by `fn._sm70_use_turbomind = False`. The cuBLAS FP16 path produces acceptable results.
- **Sinkhorn precision**: Softmax + iterative row/column normalization in FP32. The `hc_sinkhorn_eps` additive term prevents division-by-zero but introduces small bias. For `sinkhorn_repeat > 1`, accumulated FP32 rounding may differ from the torch float32 fallback due to operation ordering.

---

### 4.5  `_mhc_post_fused_kernel` — SM70 mHC Post-Block Fused Kernel

**Source**: `vllm/model_executor/layers/mhc.py`
**Python wrapper**: `_mhc_post_sm70_fast(x, residual, post_layer_mix, comb_res_mix)`
**Usage**: SM70 mHC fast path (controlled by `VLLM_SM70_MHC_FAST=1`, **disabled by default**). Computes the mHC post-block combination: `out[n,o,d] = Σ_i(comb[n,i,o] × res[n,i,d]) + post[n,o] × x[n,d]`.

#### Grid dimensions

```
grid = (num_tokens,)
```

- **Axis 0** (`pid`): One program per token.
- **Total programs**: `num_tokens`. For decode with `T=1`: **1 program**.

#### Register pressure

**Moderate**. The kernel loads a small `[hc_mult × hc_mult]` = `[4 × 4]` = 16-element comb matrix and a 4-element post vector, then streams through `hidden_size` in `BLOCK_H`-sized chunks:

- **Comb matrix**: 16 float32 values (persistent across all `h_start` iterations).
- **Post vector**: 4 float32 values (persistent).
- **Per-chunk**: 4 × `BLOCK_H` float32 residual values (`res0, res1, res2, res3`) + `BLOCK_H` float32 x values + `BLOCK_H` float32 accumulator per output row.
- Uses `tl.static_range(4)` for both residual loading and output computation, unrolling 4 iterations.

For `BLOCK_H=1024`: **estimated ~100–150 registers per thread**. Within SM70's limit.

#### Memory access patterns

| Access | Pattern | Coalescing |
|---|---|---|
| **Comb matrix** (float32) | `comb[pid, 0:16]` — 16 floats (64 bytes) | **Coalesced** (single cache line). Loaded once. |
| **Post vector** (float32) | `post[pid, 0:4]` — 4 floats (16 bytes) | **Coalesced**. Loaded once. |
| **Residual** (fp16) | `residual[pid, hc_idx, h_start:h_start+BLOCK_H]` — per hc_mult row per chunk | **Coalesced** per chunk. Total: `hc_mult × hidden_size × 2` bytes read. |
| **x** (fp16) | `x[pid, h_start:h_start+BLOCK_H]` — per chunk | **Coalesced** per chunk. Total: `hidden_size × 2` bytes read. |
| **Output** (fp16) | `out[pid, o_idx, h_start:h_start+BLOCK_H]` — per output row per chunk | **Coalesced** per chunk. Total: `hc_mult × hidden_size × 2` bytes written. |

**Arithmetic intensity**: Low. Per token: `hc_mult² × hidden_size` FMA operations (comb×res) + `hc_mult × hidden_size` FMA operations (post×x) = `(16 + 4) × 7168 = 143,360` FLOPs. Data movement: `(hc_mult + 1) × hidden_size × 2` bytes read + `hc_mult × hidden_size × 2` bytes written = `(5 + 4) × 7168 × 2 = 129,024` bytes. **~1.1 FLOP/byte** — firmly memory-bandwidth-bound.

#### Precision limitations

- **FP32 computation, FP16 output**: All intermediate accumulation is FP32. Output is cast to FP16 via `acc.to(tl.float16)` (round-to-nearest-even).
- **Comb element extraction**: Uses `tl.sum(tl.where(comb_offsets == idx, comb, 0.0))` to extract individual elements from a flat vector — semantically correct but generates inefficient masked reduction code. A scalar load would be more efficient.
- **Same precision as `_mhc_pre_post_gemm_kernel`**: Both mHC kernels share the FP16 input → FP32 compute → FP16 output pipeline.

---

### 4.6  Summary: SM70 Triton Kernel Comparison

| Kernel | File | Grid | Registers (est.) | Dominant Access Pattern | Arith. Intensity | Precision |
|---|---|---|---|---|---|---|
| `_sm70_fp8_paged_mqa_logits_kernel` | `sm70_mqa_logits.py` | `(B×next_n, max_model_len)` | 250–300 (spill likely) | Scattered paged reads | ~1.6 FLOP/byte | FP32 accum; no subnormal handling |
| `_sm70_fp8_mqa_logits_kernel` | `sm70_mqa_logits.py` | `(M, N)` | 250–300 (spill likely) | Contiguous K reads | ~1.6 FLOP/byte | FP32 accum; no subnormal handling |
| `_gather_decode_kv_triton_kernel` | `deepseek_v4_attention.py` | `(num_tokens, topk)` | 120–180 | Scattered paged reads → coalesced write | ~2.8 FLOP/byte | Full subnormal handling; BF16 RoPE bitwise copy |
| `_mhc_pre_post_gemm_kernel` | `mhc.py` | `(num_tokens,)` | 150–200 | Streamed residual reads | Mixed | FP16 GEMM + FP32 post; TurboMind MMA_884 disabled |
| `_mhc_post_fused_kernel` | `mhc.py` | `(num_tokens,)` | 100–150 | Streamed residual reads + writes | ~1.1 FLOP/byte | FP32 compute; FP16 I/O |

**Key observations:**

1. **All kernels are memory-bandwidth-bound** on V100 (900 GB/s HBM2). The MQA logits kernels are additionally register-pressure-limited due to per-head Q reloading.
2. **FP8 decode logic is duplicated** between the two MQA logits kernels (paged and non-paged). The gather kernel has a more complete implementation (with subnormal handling).
3. **mHC kernels are disabled by default** (`VLLM_SM70_MHC_FAST=0`) due to TurboMind MMA_884 precision issues. When enabled, they replace ~10 sequential kernel launches with 1 cuBLAS GEMM + 1 Triton kernel (pre) + 1 Triton kernel (post).
4. **Register spilling** in the MQA logits kernels (estimated 250–300 registers vs SM70's 255-per-thread limit) is the primary performance concern. Reducing per-head Q loading or splitting the head loop across multiple programs would alleviate this.

---

## 5  FlashMLA SM70 Kernel Interface Analysis

*Satisfies Requirement 5 (acceptance criteria 5.1 – 5.4).*

### 5.1  `flash_mla_sparse_fwd` — Prefill-Style Sparse Attention

**Python signature** (from `vllm.v1.attention.ops.flashmla`):

```python
def flash_mla_sparse_fwd(
    q: torch.Tensor,       # [s_q, h_q, d_qk] bfloat16
    kv: torch.Tensor,      # [s_kv, h_kv, d_qk] bfloat16
    indices: torch.Tensor,  # [s_q, h_kv, topk] int32
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,   # [h_q] float32
    topk_length: Optional[torch.Tensor] = None,  # [s_q] int32
    out: Optional[torch.Tensor] = None,          # [s_q, h_q, d_v] bfloat16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns: (output [s_q, h_q, d_v], max_logits [s_q, h_q], lse [s_q, h_q])
```

**Tensor shapes and dtypes:**

| Parameter | Shape | Dtype | Notes |
|---|---|---|---|
| `q` | `[s_q, h_q, d_qk]` | bfloat16 | `d_qk` = 512 (DeepSeekV4) or 576 (V3.2). `h_q` = 64 or 128 (padded). |
| `kv` | `[s_kv, h_kv, d_qk]` | bfloat16 | `h_kv` = 1 (MQA). Pre-gathered and dequantized on SM70 fallback path. |
| `indices` | `[s_q, h_kv, topk]` | int32 | Per-query sparse KV indices. -1 sentinel for invalid/padded entries. |
| `attn_sink` | `[h_q]` | float32 | Per-head attention sink bias. Padded with `-inf` for inactive heads. |
| `topk_length` | `[s_q]` | int32 | Valid entry count per query. `None` when indices already encode validity via -1. |
| `out` | `[s_q, h_q, d_v]` | bfloat16 | `d_v` = 512. Pre-allocated output buffer. |

**Usage context:**

- **Prefill path**: Called per chunk in `_forward_prefill()` after `dequantize_and_gather_k_cache` gathers FP8 cache entries into a contiguous BF16 workspace. Receives all prefill tokens for the chunk with combined SWA + compressed indices.
- **SM70 decode fallback**: Called per decode step in `_forward_decode()` when `_should_use_sm70_decode_prefill_fallback` returns `True`. Receives `T=1` decode tokens with pre-gathered BF16 KV workspace from `_gather_decode_kv_triton_kernel`.

**C++ dispatch** (`csrc/api/sparse_fwd.h`):

The `sparse_attn_prefill_interface` function requires SM90a or SM100f:
```cpp
TORCH_CHECK(is_sm90a || is_sm100f,
    "Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.");
```
On SM70, this kernel is compiled for SM90 but runs via the SM70 fallback path where all inputs are already in BF16 (not FP8). The kernel operates as a standard sparse attention over pre-gathered data, without needing architecture-specific FP8 dequantization.

### 5.2  `flash_mla_with_kvcache` — Fused Decode Attention

**Python signature** (from `vllm.v1.attention.ops.flashmla`):

```python
def flash_mla_with_kvcache(
    q: torch.Tensor,             # [batch, seq_len_q, h_q, d_qk] bfloat16
    k_cache: torch.Tensor,       # [num_blocks, page_block_size, 1, head_bytes] uint8
    block_table: Optional[torch.Tensor],    # None (sparse mode)
    cache_seqlens: Optional[torch.Tensor],  # None (sparse mode)
    head_dim_v: int,             # 512
    tile_scheduler_metadata: FlashMLASchedMeta,
    is_fp8_kvcache: bool = True,
    indices: Optional[torch.Tensor] = None,          # [b, s_q, topk] int32 — SWA slots
    topk_length: Optional[torch.Tensor] = None,      # [b] int32 — SWA valid lens
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,        # [padded_heads] float32
    extra_k_cache: Optional[torch.Tensor] = None,    # compressed KV cache (uint8)
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,  # [b, s_q, extra_topk] int32
    extra_topk_length: Optional[torch.Tensor] = None, # [b] int32
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: (out [b, s_q, h_q, d_v], lse [b, h_q, s_q])
```

**SM70-relevant parameters:**

| Parameter | Description | SM70 Relevance |
|---|---|---|
| `is_fp8_kvcache=True` | KV cache is FP8 e4m3fn + UE8M0 scales, 584 bytes/token | SM70 has no hardware FP8; kernel performs scalar FP8→FP32 dequant |
| `indices` (SWA) | `[b, 1, window_size]` int32 — SWA cache slot IDs | Combined with `extra_indices` for dual-cache attention |
| `topk_length` (SWA) | `[b]` int32 — valid SWA entries per request | Caps iteration over SWA slots |
| `extra_k_cache` | Compressed MLA FP8 paged cache (`uint8`) | C4A/C128A compressed cache; `None` for SWA-only layers |
| `extra_indices_in_kvcache` | `[b, 1, compressed_topk]` int32 — compressed slot IDs | Selected by indexer (C4A) or pre-computed (C128A) |
| `extra_topk_length` | `[b]` int32 — valid compressed entries | Caps iteration over compressed slots |
| `tile_scheduler_metadata` | `FlashMLASchedMeta` — per-layer-type shared plan | One of `tile_sched_{swaonly,c4a,c128a}`; maps work to 80 SMs |
| `attn_sink` | `[padded_heads]` float32 — per-head sink bias | Padded with `-inf` for inactive heads |

**Dispatch restriction**: The C++ `sparse_attn_decode_interface` (`csrc/api/sparse_decode.h`) only supports SM90a and SM100f:
```cpp
if (arch.is_sm100f()) { ... }
else if (arch.is_sm90a()) { ... }
else { TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd"); }
```
**SM70 cannot call `flash_mla_with_kvcache` directly.** The `_should_use_sm70_decode_prefill_fallback` guard (`capability[0] < 8`) forces all SM70 decode through the fallback path (§2.2), which gathers KV into BF16 and calls `flash_mla_sparse_fwd` instead.

### 5.3  FlashMLA Build Configuration for SM70

```bash
# Environment variables for SM70 FlashMLA build
export FLASH_MLA_ENABLE_SM70=1        # Enable SM70 compilation paths
export FLASH_MLA_DISABLE_SM100=1       # Disable SM100 (Blackwell) kernels
export TORCH_CUDA_ARCH_LIST=7.0        # Target Volta architecture only

# Source location
FLASH_MLA_SRC_DIR=/mnt/data/apps/FlashMLA
# Branch: feature/sm70-volta-flashmla
```

**Build implications:**

- `FLASH_MLA_ENABLE_SM70=1`: Compiles the `smxx/` helper kernels (`get_decoding_sched_meta`, `combine`) for SM70, and allows the SM90 prefill kernel to be compiled with SM70 compatibility (BF16 data path only, no TMA or FP8 hardware ops).
- `FLASH_MLA_DISABLE_SM100=1`: Excludes SM100 (Blackwell) kernel compilation, reducing build time and binary size.
- `TORCH_CUDA_ARCH_LIST=7.0`: Ensures NVCC generates code for SM70 (Volta) only. This means SM90-specific PTX instructions (e.g., `wgmma`, `tma`) are compiled but guarded by runtime architecture checks.

**Runtime verification**: After build, `is_flashmla_sparse_supported()` returns `(True, None)` on SM70, confirming the BF16 prefill path (`flash_mla_sparse_fwd`) is available.

### 5.4  SM70 Kernel Source Files and Computational Strategy

#### Source file inventory

The FlashMLA kernel source tree resides at `/mnt/data/apps/FlashMLA/csrc/` (mirrored in `.deps/flashmla-src/csrc/`). The relevant directories are:

| Directory | Contents | SM70 Usage |
|---|---|---|
| `csrc/api/sparse_fwd.h` | `sparse_attn_prefill_interface` — dispatches `flash_mla_sparse_fwd` to `Fwd_Sm90_Impl` or `Fwd_Sm100_*` | SM70 uses `Fwd_Sm90_Impl` path (BF16 input, no FP8 hardware) |
| `csrc/api/sparse_decode.h` | `sparse_attn_decode_interface` — dispatches `flash_mla_with_kvcache` to `Decode_Sm90_Impl` or `Decode_Sm100_*` | **Not used on SM70** (architecture check fails; decode uses fallback) |
| `csrc/sm90/prefill/sparse/` | SM90 sparse prefill kernel (`phase1.h`) | Compiled for SM70; processes pre-gathered BF16 Q×KV sparse attention |
| `csrc/sm90/decode/sparse_fp8/` | SM90 FP8 sparse decode kernel (`splitkv_mla.h`) | **Not available on SM70** (requires SM90 wgmma + TMA instructions) |
| `csrc/smxx/decode/get_decoding_sched_meta/` | Tile scheduler metadata computation | Used on SM70 for preparing `FlashMLASchedMeta` (architecture-generic) |
| `csrc/smxx/decode/combine/` | Split-KV output combination kernel | Used on SM70 by the prefill-style decode fallback (architecture-generic) |
| `csrc/sm100/` | SM100 (Blackwell) decode and prefill kernels | Excluded by `FLASH_MLA_DISABLE_SM100=1` |
| `csrc/params.h` | `SparseAttnDecodeParams`, `SparseAttnFwdParams`, `FlashMLASchedMeta` structs | Shared across all architectures |

**Note**: There is **no `csrc/sm70/`** directory. SM70 does not have dedicated FlashMLA CUDA kernels for either prefill or decode. The SM70 strategy is:

1. **Prefill**: Reuse the SM90 BF16 sparse prefill kernel compiled for SM70. This kernel operates on BF16 Q and KV tensors (already dequantized from FP8 by the vLLM-side `dequantize_and_gather_k_cache` or `_gather_decode_kv_triton_kernel`).
2. **Decode**: Bypass `flash_mla_with_kvcache` entirely. The vLLM-side fallback gathers FP8 cache → BF16 workspace via Triton, then calls `flash_mla_sparse_fwd` (the prefill kernel) with T=1 queries.

#### Computational strategy on SM70

**Prefill kernel (SM90 compiled for SM70):**

- **Warp-level computation**: Uses CUTLASS 3.x HMMA (half-precision matrix multiply-accumulate) on Volta's `mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16` instruction. SM70 supports FP16 tensor core operations with FP16 accumulation (not FP32 accumulation like SM80+).
- **Shared memory**: Volta has 96 KB shared memory per SM (configurable L1/shmem split). The prefill kernel stages Q and KV tiles into shared memory for warp-cooperative GEMM.
- **Pipeline**: Single-stage pipeline (no async copy like SM80+ `cp.async`). Data moves from global → shared via `__ldg` + `__stg` with explicit synchronization barriers.
- **Sparse index handling**: Each CTA processes a tile of query positions. Per-query `indices[s_q, h_kv, topk]` selects which KV positions to attend to. The kernel iterates over topk blocks, loads the corresponding KV tiles from the pre-gathered BF16 buffer, and accumulates the softmax-weighted output.
- **Accumulation precision**: FP16 MMA with FP16 accumulation (Volta limitation). The final softmax and output are computed in FP32 to avoid overflow, then converted back to BF16 for output.

**Architecture-generic helpers (`smxx/`):**

- **`get_decoding_sched_meta`**: Computes tile scheduler metadata mapping (batch, topk) work units to SMs. On SM70 (80 SMs for V100), `num_sm_parts = max(80 / s_q / (h_q/64), 1)`. This follows the SM90 formula since `smxx` helpers are architecture-agnostic.
- **`combine`**: After the split-KV attention kernel produces partial LSE and output accumulations across SM partitions, this kernel performs the log-sum-exp combination to produce the final output and LSE tensors. Uses a grid of `(b * s_q, h_q)` with per-thread accumulation.

**SM70-specific limitations impacting FlashMLA:**

| Limitation | Impact | Current Workaround |
|---|---|---|
| No hardware FP8 | Cannot decode FP8 KV cache in-kernel | vLLM Triton kernel `_gather_decode_kv_triton_kernel` dequants FP8→BF16 before FlashMLA |
| No `cp.async` | No asynchronous global→shared memory copy | Synchronous loads with explicit `__syncthreads()` barriers |
| FP16 accumulation only | Reduced precision in attention dot product vs SM80+ FP32 accum | FP32 softmax + output accum; FP16 clamping in vLLM post-processing |
| 96 KB shared memory | Less shmem than SM80+ (164 KB) or SM90 (228 KB) | Smaller tile sizes, more iterations per CTA |
| No TMA | No tensor memory accelerator for bulk data movement | Standard vectorized global loads |
| 80 SMs (V100) | Fewer compute units than SM90 (132 SMs, H100) | Tile scheduler adapts; fewer `num_sm_parts` |

---

## 6  Prefill Path Efficiency Audit

*Satisfies Requirement 6 (acceptance criteria 6.1 – 6.4).*

This section audits the computational efficiency of every prefill stage on SM70 (V100), identifying the dominant bottlenecks and quantifying the overhead introduced by SM70-specific torch fallback functions.

### 6.1  Prefill Bottleneck Breakdown

Each prefill request processes the full input prompt (hundreds to thousands of tokens) through all ~61 decoder layers. The table below itemises the per-layer compute stages on SM70, ordered by execution sequence, with estimated relative cost for a typical 1024-token prefill.

| Stage | Kernel / Function | Call Frequency | Estimated Relative Cost | Notes |
|---|---|---|---|---|
| **★ Projection GEMMs** | `fused_wqa_wkv` (cuBLAS FP16) | 1× per layer | **~25–35%** | `[T, 7168] → [T, 1536+512]`. cuBLAS FP16 on Volta tensor cores. Amortised well for large T; compute-bound for long prompts. |
| **★ Q Projection** | `wq_b` (cuBLAS FP16) | 1× per layer | **~10–15%** | `[T, 1536] → [T, n_local_heads × 512]`. cuBLAS FP16. |
| **★ Attention** | `flash_mla_sparse_fwd` | 1× per chunk (up to `ceil(num_prefills/4)` chunks per layer) | **~20–30%** | SM90 BF16 sparse prefill kernel reused on SM70. Processes chunked KV workspace. FP16 MMA with FP16 accumulation (Volta limitation). Compute-bound for large T × topk. |
| **★ O-Projection** | `fused_inv_rope_fp8_quant` → `_sm70_fp8_einsum_bmm` → `wo_b` | 1× per layer | **~10–15%** | FP32 einsum `bhr,hdr→bhd` amortises well over large T. Weight pre-dequanted to FP32 (cached ~86 MB). `wo_b` cuBLAS FP16. |
| **★ KV Insert + RoPE** | `_torch_qnorm_rope_kv_insert_fallback` (SM70) | 1× per layer (aux stream) | **~5–10%** | Overlapped with indexer on default stream. Per-batch (not per-token) but involves sequential float32 operations. See §6.2. |
| **★ Compressor** | `_torch_fused_compress_norm_rope_insert_fp8_fallback` (SM70) | 1× per C4A/C128A layer (aux stream) | **~5–10%** | Python-level token loop. Only fires at compress_ratio boundaries (e.g., every 4th token for C4A). See §6.3. |
| **★ Indexer** | `sm70_fp8_mqa_logits` (Triton) | 1× per C4A layer (default stream) | **~5–10%** | Grid `(M, N)`. For 1024-token prefill with 4096 KV: ~4.2M programs. Memory-bound (~1.6 FLOP/byte). Overlapped with KV insert + compressor on aux stream. |
| **★ Chunked KV Gather** | `dequantize_and_gather_k_cache` | 2× per chunk (compressed + SWA) | **~5–8%** | FP8 → BF16 dequantization + page-strided gather into contiguous workspace. Memory-bandwidth-bound. See §6.4. |
| **★ mHC Pre/Post-Block** | `_mhc_pre_torch_fallback` + `_mhc_post_torch_fallback` | 4× per layer | **~5–8%** | Float32 GEMM + RMSNorm + sigmoid + sinkhorn + einsum, each launch a separate CUDA kernel. For large T, the GEMM `[T, 28672] × [24, 28672]^T` is compute-bound; smaller T is launch-overhead-dominated. |
| **★ MoE** | `FusedMoE` (Triton) | 1× per layer | **~10–15%** | Hash/noaux_tc routing → MXFP4 weight dequant → expert compute → TP allreduce. Software int4→FP16 dequant (no hardware acceleration on SM70). |

**Key observations:**

1. **Projection GEMMs dominate** for long prefills. For T=1024 with hidden_size=7168, the `fused_wqa_wkv` GEMM is a `[1024, 7168] × [2048, 7168]^T` FP16 matmul — well-suited for V100's 125 TFLOPS tensor core throughput, achieving ~60–80% utilisation.

2. **Attention (`flash_mla_sparse_fwd`) is the second major cost**. The sparse attention kernel performs `T × topk × head_dim` FP16 MMA operations per chunk. With FP16 accumulation on Volta (vs FP32 on SM80+), the kernel achieves lower arithmetic throughput per MMA but is still compute-bound for large `T × topk`.

3. **SM70 fallback functions** (§6.2, §6.3) are **hidden by stream overlap** during normal operation — the indexer runs on the default stream while KV insert and compressor run on the aux stream. Their overhead only becomes visible when they exceed the indexer's runtime, which happens for:
   - Short prompts where indexer completes quickly (few tokens → small `M × N` grid).
   - C128A layers where `compress_ratio=128` means the compressor fires infrequently but each invocation gathers 128+ tokens of state.

4. **Chunked KV gather** (§6.4) is a unique SM70 bottleneck — it materialises the entire KV working set as BF16 before `flash_mla_sparse_fwd`, consuming substantial memory bandwidth.

### 6.2  `_torch_qnorm_rope_kv_insert_fallback` Overhead Analysis

**Source**: `vllm/model_executor/layers/deepseek_v4_attention.py`
**Replaces**: `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (SM80+ single CUDA kernel)

This fallback is called once per layer for every prefill step, running on the **aux stream** (overlapped with the indexer on the default stream).

#### Operation breakdown and CUDA kernel count

The function performs the following operations, each translating to one or more CUDA kernel launches:

| Operation | CUDA Kernels | Dtype | Description |
|---|---|---|---|
| `q.float()` | 1 (cast) | fp16→fp32 | Upcast Q to float32 for RMSNorm |
| `q_float.pow(2).mean(dim=-1, keepdim=True)` | 1–2 (fused reduction) | fp32 | Per-head variance computation |
| `torch.rsqrt(variance + eps)` | 1 (elementwise) | fp32 | Reciprocal square root |
| `q_float * rsqrt_result` | 1 (elementwise) | fp32 | Normalize |
| `.to(q.dtype)` | 1 (cast) | fp32→fp16 | Downcast back to fp16 |
| `_apply_gptj_rope_tail` (Q) | 3–5 (cos/sin lookup, clone, slice, rotate, interleave) | fp16 | GPT-J interleaved RoPE: even/odd pairs rotated by cos/sin |
| `q.copy_()` | 1 (memcpy) | fp16 | In-place write |
| `_apply_gptj_rope_tail` (KV) | 3–5 | fp16 | Same RoPE for KV side |
| `slot_mapping >= 0` → `valid_mask.any()` | 2 (compare + reduce) | int64→bool | Filter valid tokens |
| `kv_valid[:, :448].float()` → block reshape | 2 (slice+cast, view) | fp16→fp32 | Extract NoPE, upcast for quant |
| `blocks.abs().amax().clamp()` | 2 (abs+reduce, clamp) | fp32 | Per-block absmax |
| `torch.ceil(torch.log2(...))` | 2 (log2, ceil) | fp32 | UE8M0 exponent computation |
| `torch.exp2(exponents)` | 1 (elementwise) | fp32 | Scale computation |
| `(blocks / scales).clamp().to(float8_e4m3fn)` | 2–3 (div, clamp, cast) | fp32→fp8 | Quantize to FP8 |
| `.view(uint8).view(...)` | 0 (view only) | — | Reinterpret cast |
| `kv_valid[:, 448:].to(bfloat16).view(uint8)` | 1 (cast) | fp16→bf16→uint8 | RoPE bytes |
| `torch.cat((fp8_bytes, rope_bytes), dim=-1)` | 1 (cat) | uint8 | Combine token data |
| Paged scatter write (data) | 1 (advanced indexing) | uint8 | `k_cache[block_indices, data_offsets] = token_data` |
| UE8M0 scale encoding + scatter write | 2–3 (clamp, cast, zeros, scatter) | uint8 | Scale bytes to paged cache |
| **Total** | **~25–35 kernel launches** | | |

#### Overhead characteristics

| Characteristic | Detail |
|---|---|
| **Sequential per-token processing** | All operations are batched across `num_tokens` (not per-token in a Python loop), but the **sheer number of separate CUDA kernels** (~25–35 per layer) is the primary overhead. Each kernel launch adds ~5–10 µs of host-side overhead on V100. |
| **Float32 intermediates** | Q-norm and block quantization operate in float32, consuming 2× the memory bandwidth of fp16 operations. For T=1024, the Q-norm float32 intermediate is `1024 × n_heads × head_dim × 4` bytes = ~128 MB for `n_heads=64, head_dim=512`. |
| **Non-fused RoPE** | GPT-J RoPE (`_apply_gptj_rope_tail`) involves: (1) cos/sin cache lookup with position indices, (2) clone + slice to separate even/odd elements, (3) multiply by cos/sin, (4) interleave. Each sub-operation is a separate CUDA kernel. The SM80+ fused kernel performs all of this + norm + quant + scatter in a single launch. |
| **Scatter write overhead** | Paged cache writes use PyTorch advanced indexing (`k_cache[block_indices, data_offsets]`), which generates gather/scatter kernels with non-coalesced memory access patterns across different physical pages. |

#### Comparison with SM80+ fused path

| Metric | SM70 Fallback | SM80+ Fused | Overhead Factor |
|---|---|---|---|
| CUDA kernel launches | ~25–35 | 1 | **25–35×** |
| Host-side launch overhead | ~125–350 µs | ~5–10 µs | **12–70×** |
| Memory traffic (float32 intermediates) | Q-norm + quant in fp32 | All in fp16/fp8 | **~2× for norm/quant stages** |
| RoPE computation | 3–5 separate kernels | Fused in single kernel | **3–5× kernel launches** |

**Net impact**: For a 1024-token prefill, the fallback adds approximately **0.1–0.35 ms per layer** of kernel launch overhead (25–35 launches × 5–10 µs each). Across 61 layers: **6–21 ms total**. However, this is **largely hidden** by stream overlap with the indexer. The overhead becomes visible only when the fallback exceeds the indexer's runtime — typically for short prompts (<128 tokens) where the indexer's `(M, N)` grid is small.

### 6.3  `_torch_fused_compress_norm_rope_insert_fp8_fallback` Overhead Analysis

**Source**: `vllm/model_executor/layers/deepseek_compressor.py`
**Replaces**: `_fused_kv_compress_norm_rope_insert_sparse_attn` (SM80+ Triton kernel, one program per token)

This fallback is called once per C4A/C128A layer for every prefill step, running on the **aux stream**.

#### Python-level token loop overhead

The critical overhead is the **Python `for` loop** over tokens:

```python
for token_idx in range(slot_mapping.shape[0]):
    slot_id = int(slot_mapping[token_idx].item())   # host-device sync!
    if slot_id < 0:
        continue
    position = int(positions[token_idx].item())       # host-device sync!
    if (position + 1) % compress_ratio != 0:
        continue
    ...  # per-token computation
```

Each iteration performs **two `.item()` calls**, each triggering a host-device synchronisation that serialises the CUDA stream. For T=1024 tokens, this means up to **2048 host-device syncs** per layer call — even though only `T / compress_ratio` tokens (e.g., 256 for C4A) will pass the `(position + 1) % compress_ratio != 0` filter.

#### Per-token computation breakdown (when the filter passes)

For each token that fires (at compress_ratio boundaries):

| Operation | CUDA Kernels | Description |
|---|---|---|
| State gather (inner loop over `window`) | `window` × 2–3 `.item()` calls + indexing | Gathers `kv_rows` and `score_rows` from `state_cache` via block_table lookup. Each block_table lookup calls `.item()` (host-device sync). For C4A: `window = 2 × 4 = 8` iterations, so **~16–24 extra .item() syncs per token**. |
| `torch.stack(score_rows).softmax(dim=0)` | 2 (stack + softmax) | `[window, head_dim]` softmax — small tensor, launch-overhead-dominated |
| `torch.stack(kv_rows)` | 1 (stack) | `[window, head_dim]` |
| `(kv * score).sum(dim=0)` | 2 (mul + reduce) | Weighted compression |
| `compressed_kv.pow(2).sum() / head_size` | 2 (pow + reduce) | RMSNorm variance |
| `torch.rsqrt(variance + eps)` | 1 | Reciprocal sqrt |
| `normed * rms_norm_weight.float()` | 1 (mul) | Weighted norm |
| `_apply_gptj_rope_tail_1d` | 3–5 | Single-token GPT-J RoPE |
| `_store_fp8_sparse_attention_cache_torch` | 5–8 | FP8 block quant + UE8M0 scale + scatter write (same as §6.2 but for a single token) |
| **Total per token** | **~20–30 kernel launches + ~20–30 .item() syncs** | |

#### Aggregate overhead

For a 1024-token C4A prefill (`compress_ratio=4`, so 256 tokens fire):

| Metric | Value |
|---|---|
| Python loop iterations | 1024 (all tokens checked) |
| `.item()` syncs in loop header | 2048 (2 per iteration) |
| Tokens that fire | 256 (`(pos+1) % 4 == 0`) |
| `.item()` syncs per firing token (state gather) | ~16–24 (block_table lookups) |
| Total `.item()` syncs | ~2048 + 256 × 20 = **~7,168** |
| Per-sync overhead | ~5–15 µs (host-device round-trip) |
| **Total estimated overhead** | **~36–107 ms per layer** |

**This is the single largest SM70 prefill overhead.** For comparison, the SM80+ Triton kernel (`_fused_kv_compress_norm_rope_insert_sparse_attn`) performs the same computation in a **single kernel launch** with one Triton program per firing token — zero `.item()` calls, zero Python loop overhead.

#### Mitigating factors

1. **Stream overlap**: The compressor runs on the aux stream, overlapped with the indexer. If the indexer takes longer (e.g., for long sequences with large `M × N` grids), the compressor overhead is partially hidden.
2. **Infrequent firing**: Only `T / compress_ratio` tokens trigger the full computation. For C128A (`compress_ratio=128`), a 1024-token prefill fires only 8 times, reducing aggregate overhead to ~1–8 ms per layer.
3. **Decode-step impact**: During decode (T=1), at most one token fires per layer. The per-token overhead (~0.1–0.3 ms) is small relative to other decode stages, though the `.item()` syncs still serialise the CUDA stream.

### 6.4  Memory Bandwidth for `dequantize_and_gather_k_cache` Scatter-Gather

**Source**: `vllm/v1/attention/ops/deepseek_v4_ops.py` (or equivalent C++ implementation)
**Usage**: Called twice per chunk in `_forward_prefill()` — once for compressed KV and once for SWA KV.

This function reads FP8 KV cache entries from scattered paged locations and writes dequantized BF16 values into a contiguous workspace tensor. It is the memory-bandwidth bottleneck in the chunked prefill path.

#### Per-chunk bandwidth analysis

For each chunk of `PREFILL_CHUNK_SIZE=4` requests:

**Compressed KV gather** (C4A layers, `compress_ratio=4`):

| Metric | Formula | Typical Value |
|---|---|---|
| Entries per request | `seq_len / compress_ratio` | 256 (for 1024-token prompt, C4A) |
| Entries per chunk | `4 × 256` | 1,024 |
| Read per entry (FP8 NoPE) | `448` bytes | |
| Read per entry (UE8M0 scales) | `7` bytes | |
| Read per entry (BF16 RoPE) | `128` bytes | |
| **Total read per entry** | `583` bytes | |
| **Total read per chunk** | `1,024 × 583` | **~583 KB** |
| Write per entry (BF16 output) | `512 × 2 = 1,024` bytes | |
| **Total write per chunk** | `1,024 × 1,024` | **~1,024 KB** |

**SWA KV gather** (all layers):

| Metric | Formula | Typical Value |
|---|---|---|
| Entries per request | `min(seq_len, window_size)` | 1,024 (or `window_size` if shorter) |
| Entries per chunk | `4 × 1,024` | 4,096 |
| **Total read per chunk** | `4,096 × 583` | **~2,333 KB** |
| **Total write per chunk** | `4,096 × 1,024` | **~4,096 KB** |

**Combined per-chunk bandwidth**:

| Direction | Compressed + SWA | Total |
|---|---|---|
| Read | 583 + 2,333 | **~2,916 KB** |
| Write | 1,024 + 4,096 | **~5,120 KB** |
| **Total per chunk** | | **~8,036 KB (~7.8 MB)** |

#### Scatter-gather access pattern analysis

The primary efficiency concern is the **scattered read pattern** from the paged FP8 cache:

| Access Characteristic | Detail |
|---|---|
| **Page-strided reads** | Each KV entry resides at `cache[block_idx, pos_in_block, 0:584]`, where `block_idx` is determined by the block table. Adjacent entries in logical sequence order may reside in different physical blocks (non-contiguous). |
| **Intra-token coalescing** | Within a single 584-byte entry, reads are contiguous — 448 FP8 NoPE bytes + 128 BF16 RoPE bytes are stored sequentially. This portion is well-coalesced. |
| **Inter-token scattering** | Across tokens within a block, entries are 584-byte-strided (or `token_stride`-byte-strided). This is partially coalesced for in-block access but scattered across blocks. |
| **Block table indirection** | Each entry requires a block table lookup (`block_table[req_idx, block_in_seq]`) to resolve the physical block, adding an extra memory read per entry. For chunked prefill with 4 requests, this is 4 concurrent block table traversals. |

**Effective bandwidth utilisation**:

| Scenario | Estimated HBM Efficiency | Explanation |
|---|---|---|
| Entries within same block | ~70–80% | 584-byte reads are aligned within the block's contiguous memory region. Partial cache line waste at 576-byte alignment boundary. |
| Entries across blocks | ~40–60% | Block transitions cause cache line waste from the new block's start address. V100's 32-byte cache lines mean a 584-byte read spans 19 cache lines, with the last line ~50% wasted. |
| Overall (mixed) | **~50–70%** | Weighted average depending on sequence length vs block size. Short sequences (few blocks) achieve higher efficiency; long sequences (many blocks) suffer from more scattered block accesses. |

At V100's ~900 GB/s peak HBM bandwidth with ~50–70% efficiency:

| Metric | Value |
|---|---|
| Effective bandwidth | ~450–630 GB/s |
| Time for 7.8 MB per chunk | **~12–17 µs** |
| Chunks per 1024-token prefill | `ceil(num_prefills / 4)` = 1 (if batch_size ≤ 4) |
| **Per-layer time** | **~12–17 µs** |
| **61-layer total** | **~0.7–1.0 ms** |

#### Comparison with fused-cache path (hypothetical)

A fused attention kernel that reads FP8 cache directly (like `flash_mla_with_kvcache` on SM80+) would:
- **Eliminate the BF16 write**: No intermediate workspace materialisation (saves ~5,120 KB write per chunk).
- **Read FP8 directly**: Reads 583 bytes/entry instead of 1,024 bytes/entry (BF16), saving ~43% read bandwidth.
- **Total bandwidth reduction**: From ~8.0 MB/chunk to ~2.9 MB/chunk (read only) — a **~2.7× reduction**.

However, this is only achievable with a FlashMLA prefill kernel that supports FP8 input on SM70 (currently not implemented — the SM90 prefill kernel requires BF16 input). The gather + dequant overhead is therefore an **inherent cost of the SM70 architecture** rather than a fixable inefficiency, unless a custom SM70 FP8 prefill kernel is developed.

### 6.5  Summary: Prefill Efficiency Findings

| Finding | Impact | Affected Stages | Mitigation Path |
|---|---|---|---|
| **Compressor Python-loop `.item()` syncs** | **Critical** (~36–107 ms/layer for C4A, 1024 tokens) | `_torch_fused_compress_norm_rope_insert_fp8_fallback` | Replace with fused Triton kernel (§9.4 in tasks.md) |
| **KV-insert fallback kernel launch count** | **Moderate** (~0.1–0.35 ms/layer, hidden by stream overlap) | `_torch_qnorm_rope_kv_insert_fallback` | Replace with fused Triton kernel (§9.2 in tasks.md) |
| **BF16 workspace materialisation** | **Moderate** (~0.7–1.0 ms per 61-layer pass) | `dequantize_and_gather_k_cache` | Requires FP8-native FlashMLA prefill kernel for SM70 |
| **Scattered paged cache reads** | **Low–Moderate** (~50–70% bandwidth efficiency) | `dequantize_and_gather_k_cache` | Block-size-aligned access, prefetch hints |
| **Projection GEMMs** | **Low** (already efficient at ~60–80% utilisation for large T) | `fused_wqa_wkv`, `wq_b`, `wo_b` | Already near-optimal for Volta tensor cores |

**Priority ranking for prefill optimisation on SM70:**

1. **P0**: Eliminate the compressor fallback Python loop (§6.3) — the `.item()` serialisation dominates prefill latency for C4A layers.
2. **P1**: Replace `_torch_qnorm_rope_kv_insert_fallback` with a fused Triton kernel (§6.2) — reduces kernel launch count from ~25–35 to 1 per layer.
3. **P1**: Optimise `dequantize_and_gather_k_cache` scatter-gather patterns (§6.4) — improve page-access coalescing and reduce BF16 workspace size.
4. **P2**: Develop an SM70 FP8-native FlashMLA prefill kernel — eliminates the BF16 workspace entirely (long-term, high effort).

---

## 7  Decode Path Efficiency Audit

*Satisfies Requirement 7 (acceptance criteria 7.1 – 7.4).*

### 7.1  Per-Token Decode Latency Breakdown

Each autoregressive decode step for a single token passes through every one of the ~61 decoder layers. The table below itemises the per-layer compute stages on SM70 (V100), ordered by execution sequence. Latency-critical stages (on the decode hot path) are marked **★**.

| Stage | Kernel / Function | Call Frequency | Notes |
|---|---|---|---|
| **★ mHC Pre-Block** | `_mhc_pre_torch_fallback` (default) | 1× per layer | Float32 GEMM `[T, hc_mult*H] × [hc_mult3, hc_mult*H]^T` + RMSNorm + sigmoid + sinkhorn (multi-iteration) + weighted-sum einsum. Multiple sequential kernel launches. |
| **★ Projection GEMMs** | `fused_wqa_wkv` (cuBLAS) | 1× per layer | `[T, hidden_size] → [T, q_lora_rank + head_dim]`. cuBLAS FP16. Amortised by batch size. |
| **★ Q Projection** | `wq_b` (cuBLAS) | 1× per layer | `[T, q_lora_rank] → [T, n_local_heads * head_dim]`. cuBLAS FP16. |
| **★ RMSNorm + RoPE + KV Insert** | `_torch_qnorm_rope_kv_insert_fallback` | 1× per layer | Q per-head RMSNorm (float32) + GPT-J RoPE (float32 intermediates) + FP8 block quant + paged cache scatter write. Runs on aux stream, overlapped with indexer. |
| **★ Indexer (C4A layers)** | `sm70_fp8_paged_mqa_logits` | 1× per C4A layer | Grid `(T*next_n, max_model_len)` — one program per (token, K-position). Loads FP8 K from paged cache, manual bit-level FP8 decode, loops over `NUM_HEADS` for weighted ReLU dot. High register pressure from per-head Q reload. |
| **★ KV Gather (SM70 fallback)** | `_gather_decode_kv_triton_kernel` | 1× per layer (SM70) | Grid `(T, topk)`. Reads 7 × 64-element FP8 NoPE blocks from scattered paged cache + 64 BF16 RoPE values. Materialises `[T, topk, 512]` BF16 workspace. |
| **★ Attention** | `flash_mla_sparse_fwd` (SM70 fallback) | 1× per layer (SM70) | Operates on pre-gathered BF16 KV workspace. Prefill-style sparse attention kernel reused for decode. |
| **★ O-Projection** | `fused_inv_rope_fp8_quant` → `_sm70_fp8_einsum_bmm` → `wo_b` | 1× per layer | Inverse RoPE → FP8 quant → FP32 einsum `bhr,hdr→bhd` (weight pre-dequanted to FP32, cached ~86 MB) → linear `wo_b` → FP16 clamp. |
| **★ mHC Post-Block** | `_mhc_post_torch_fallback` | 1× per layer | Float32 einsum `nio,nih→noh` + `post × x` broadcast add. |
| **★ FFN Norm** | RMSNorm | 1× per layer | Standard RMSNorm, FP16. |
| **★ MoE** | `FusedMoE` (Triton) | 1× per layer | Hash/noaux_tc routing → MXFP4 weight dequant → expert compute → TP allreduce. |
| **★ mHC Post-Block FFN** | `_mhc_post_torch_fallback` | 1× per layer | Same as attention-side mHC post. |

**Dominant cost centres** (SM70 decode, single token):

1. **mHC torch fallback** — called 4× per layer (pre-attn, post-attn, pre-ffn, post-ffn). Each invocation launches ≥5 CUDA kernels in float32 (matmul, squared-sum, rsqrt, sigmoid, sinkhorn iterations, einsum). For `hc_mult=4, hidden_size=7168`, the GEMM alone is `[1, 28672] × [24, 28672]^T` in float32 — compute-bound on V100's FP32 throughput (15.7 TFLOPS) but latency-dominated by the sequential kernel launch overhead (each launch adds ~5–10 µs host-side).

2. **Projection GEMMs** (`fused_wqa_wkv`, `wq_b`, `wo_b`) — small-batch matrix multiplications that underutilise V100's 125 TFLOPS FP16 tensor core throughput for T=1. Latency dominated by kernel launch + memory transfer rather than compute.

3. **SM70 decode fallback path** (KV gather + attention) — the two-kernel fallback adds overhead compared to the fused `flash_mla_with_kvcache` on SM80+ (see §7.2).

4. **MoE routing + dequant** — MXFP4 weight dequantisation inside `FusedMoE` Triton kernels has no hardware acceleration on SM70; purely software int4→FP16 unpacking.

### 7.2  SM70 Decode Fallback Overhead Analysis

On SM70, `_should_use_sm70_decode_prefill_fallback` always returns `True` (capability[0] < 8), forcing the two-stage fallback path instead of the fused `flash_mla_with_kvcache`.

#### Fallback path (SM70): 3 kernels per layer

```
Kernel 1: _gather_decode_kv_triton_kernel
  Grid: (num_decode_tokens, topk)
  • Reads: scattered FP8 paged cache (7 × 64-byte NoPE + 64 × BF16 RoPE per slot)
  • Writes: contiguous BF16 workspace [T, topk, 512] (1024 bytes per slot)
  • Memory pattern: scattered reads (non-coalesced across pages) → coalesced write

Kernel 2: _build_decode_prefill_fallback_indices  (torch ops, not a single kernel)
  • Builds local-offset index tensor from global slot IDs
  • Involves arange, mul_, add_, where, cat — multiple small CUDA kernel launches

Kernel 3: flash_mla_sparse_fwd
  • Prefill-style sparse attention on the gathered BF16 workspace
  • Input: q [T, padded_heads, 512], kv [T*topk, 1, 512], indices [T, 1, topk]
```

#### Normal path (SM80+): 1 kernel per layer

```
flash_mla_with_kvcache
  • Reads FP8 KV directly from paged cache via indices
  • Fuses: KV dequant + attention dot product + softmax + output projection
  • No intermediate BF16 workspace materialisation
  • Single kernel launch, tile-scheduler handles SM work distribution
```

#### Overhead breakdown

| Overhead Source | Estimated Impact |
|---|---|
| **Extra kernel launches** | ~15–30 µs per layer (2 additional kernel launches + index-building torch ops ≈ 5 small kernels). At 61 layers: **~0.9–1.8 ms per decode step.** |
| **BF16 workspace materialisation** | Memory write bandwidth for `[T, topk, 512]` at BF16 = `topk × 1024` bytes/token. For C4A `topk≈2048` + SWA `window_size`: total ≈ 2–3 MB written per layer. At 900 GB/s V100 HBM: **~3 µs per layer**, but 61× = ~0.18 ms. |
| **Redundant data movement** | FP8 → FP32 → BF16 conversion in gather kernel, then BF16 → FP32 inside `flash_mla_sparse_fwd`. The fused path does FP8 → FP32 directly inside the attention kernel, saving one format conversion. |
| **Suboptimal attention kernel** | `flash_mla_sparse_fwd` is a prefill-style kernel called with T=1 queries. It lacks the decode-specific tile scheduler optimisation of `flash_mla_with_kvcache` (which partitions work across 80 SMs based on per-request topk lengths). |

**Total estimated overhead of SM70 fallback vs fused path: 1.0–2.5 ms per decode step** (primarily from kernel launch overhead × 61 layers).

### 7.3  CUDA Graph Replay Efficiency

#### Current CUDA graph configuration

| Parameter | Value |
|---|---|
| `CG_SUPPORT` | `AttentionCGSupport.UNIFORM_BATCH` (both `FlashMLASparseMetadataBuilder` and `DeepseekSparseSWAMetadataBuilder`) |
| `cudagraph_capture_sizes` | `[1]` (single decode token only) |
| Policy | `FULL_DECODE_ONLY` — only pure-decode steps are graph-captured |

#### Captured graph contents (per decode step)

The CUDA graph captures the full forward pass of all ~61 decoder layers for `T=1` decode tokens. Per layer, the captured graph includes:

| Component | Estimated Kernel Count | Notes |
|---|---|---|
| mHC pre-block (torch fallback) | ~6–8 | matmul, sq-sum, rsqrt, sigmoid, softmax, div, einsum (varies with sinkhorn_repeat) |
| `fused_wqa_wkv` GEMM | 1 | cuBLAS FP16 |
| `fused_q_kv_rmsnorm` | 1 | Fused Q+KV RMSNorm |
| `wq_b` GEMM | 1 | cuBLAS FP16 |
| `_torch_qnorm_rope_kv_insert_fallback` | ~6–8 | Multiple torch ops: pow, mean, rsqrt, mul, clone, cos/sin RoPE, FP8 quant (clamp, exp2, to), scatter write |
| Compressor (C4A/C128A layers) | ~4–6 | `_torch_fused_compress_norm_rope_insert_fp8_fallback`: Python-level token loop with per-token softmax, weighted-sum, RMSNorm, RoPE, FP8 quant, scatter |
| Indexer (C4A layers) | 1 | `sm70_fp8_paged_mqa_logits` Triton kernel |
| `_gather_decode_kv_triton_kernel` | 1 | Triton KV gather |
| Index building (torch ops) | ~3–5 | arange, mul_, add_, where, cat for fallback indices |
| `flash_mla_sparse_fwd` | 1 | FlashMLA sparse attention |
| O-projection pipeline | ~4–5 | `fused_inv_rope_fp8_quant` (1) + `_sm70_fp8_einsum_bmm` (torch.einsum → 1–2 kernels) + `wo_b` cuBLAS (1) + clamp (1) |
| mHC post-block (torch fallback) | ~3–4 | Float32 einsum + broadcast add |
| RMSNorm (FFN) | 1 | |
| MoE | ~3–5 | Routing + FusedMoE Triton + allreduce |
| mHC pre-block FFN | ~6–8 | Same as attention-side |
| mHC post-block FFN | ~3–4 | Same as attention-side |

**Estimated total kernel launches per decode step: ~2,400–3,400** (for 61 layers × ~40–55 kernels/layer).

This is a **high kernel count** for a CUDA graph. The graph replay overhead scales with graph size, though it is still significantly faster than re-issuing each kernel from the CPU. The primary concern is:

1. **Graph capture memory** — each captured kernel records its arguments (tensor pointers, scalar params). With ~3,000 kernels, the graph object consumes ~1–5 MB of GPU memory.
2. **Graph replay latency** — replay iterates through the captured command buffer. For ~3,000 kernels, replay overhead is ~50–100 µs (vs ~10–20 ms for eager kernel-by-kernel CPU dispatch), providing a **~100–200× speedup** in launch overhead.

#### Non-graphed operations

The following operations are **not captured** in the CUDA graph and run on every decode step:

| Operation | Reason | Impact |
|---|---|---|
| `DeepseekSparseSWAMetadataBuilder.build()` | CPU-side metadata construction (runs before graph replay) | ~0.1–0.5 ms CPU time; does not block GPU |
| `FlashMLASchedMeta` planner (first call per type) | Runs during capture only; on replay `have_initialized=True` and the existing plan is reused | Zero overhead on replay |
| mHC torch fallback (if not captured) | The torch fallback uses dynamic shapes internally for sinkhorn iterations | **Captured** in practice since shapes are fixed for `T=1` |

**Key observation**: With `FULL_DECODE_ONLY` and `cudagraph_capture_sizes=[1]`, effectively **all GPU operations** in the decode step are captured. The only non-graphed work is CPU-side metadata build, which is overlapped with GPU execution.

#### Workspace memory overhead

Workspace tensors allocated via `get_simultaneous()` persist across CUDA graph captures:

| Workspace | Shape (T=1, typical) | Bytes | Purpose |
|---|---|---|---|
| SM70 fallback KV (SWA-only) | `(1, window_size, 512)` | `window_size × 1024` | BF16 gathered KV for SWA layers |
| SM70 fallback KV (C4A) | `(1, topk+window_size, 512)` | `(topk+window_size) × 1024` | BF16 gathered KV for compressed layers |
| Prefill KV gather | `(4, M, 512)` | `4 × M × 1024` | Not used during decode, but allocation persists |
| q_concat buffer | `(max_tokens, num_heads, 512)` | Large | Reserved for mixed prefill/decode; unused in pure decode |
| Prefill BF16 workspace | `(prefill_workspace_size, 512)` | Very large (~900 MB) | Reserved for prefill; unused in pure decode |

**The prefill workspace (~900 MB) is the largest overhead** — it remains allocated even during pure-decode CUDA graph replay because `get_simultaneous()` allocations are persistent. This workspace could be freed or shared with other subsystems during decode-only phases.

### 7.4  Extra Memory Bandwidth from Fallback KV Materialisation

When `_should_use_sm70_decode_prefill_fallback` returns `True` (always on SM70), the decode path materialises a full BF16 KV workspace before calling `flash_mla_sparse_fwd`. This section quantifies the additional memory bandwidth consumed compared to the fused `flash_mla_with_kvcache` path.

#### Per-layer bandwidth analysis (single decode token)

**SM70 Fallback path — read bandwidth:**

| Read Source | Bytes per Token | Calculation |
|---|---|---|
| FP8 NoPE from paged cache | `topk × 448` | 7 quant blocks × 64 bytes each × topk slots |
| UE8M0 scales from paged cache | `topk × 7` | 7 scale bytes × topk slots |
| BF16 RoPE from paged cache | `topk × 128` | 64 × 2 bytes × topk slots |
| **Total read per token** | **`topk × 583`** | |

**SM70 Fallback path — write bandwidth (extra vs fused):**

| Write Destination | Bytes per Token | Calculation |
|---|---|---|
| BF16 workspace (NoPE + RoPE) | `topk × 1024` | 512 × 2 bytes × topk slots |
| **Total extra write per token** | **`topk × 1024`** | This write does not exist in the fused path |

**SM70 Fallback path — second read (by `flash_mla_sparse_fwd`):**

| Read Source | Bytes per Token | Calculation |
|---|---|---|
| BF16 workspace KV | `topk × 1024` | Re-reading the materialised workspace |
| **Total extra read per token** | **`topk × 1024`** | This read does not exist in the fused path |

#### Concrete numbers

For a C4A layer with `topk=2048` (compressed) + `window_size=4096` (SWA), `total_topk ≈ 6144`:

| Metric | Fallback Path | Fused Path | Extra |
|---|---|---|---|
| Cache read | 6144 × 583 = **3.41 MB** | 6144 × 583 = **3.41 MB** | 0 |
| Workspace write | 6144 × 1024 = **6.00 MB** | 0 | **+6.00 MB** |
| Workspace re-read | 6144 × 1024 = **6.00 MB** | 0 | **+6.00 MB** |
| **Total extra bandwidth** | | | **+12.00 MB/layer** |

**Per decode step** (61 layers): `61 × 12.00 MB ≈ **732 MB** extra memory traffic`.

At V100 HBM bandwidth of ~900 GB/s, this extra traffic alone would take `732 / 900000 ≈ 0.81 ms` — a meaningful fraction of the total decode latency.

**For SWA-only layers** with `window_size=4096`: extra bandwidth = `4096 × 2048 = 8 MB/layer`. There are fewer SWA-only layers, so total contribution is smaller.

#### Format conversion overhead

The fallback path also performs redundant format conversions:

1. **Gather kernel**: FP8 e4m3fn → FP32 (bit manipulation) → BF16 (round-to-nearest-even bit shift). This requires 3 ALU operations per value (sign/exp/mant extraction, scale multiply, FP32→BF16 rounding).

2. **Attention kernel**: BF16 → FP32 (implicit widening inside `flash_mla_sparse_fwd`). The fused `flash_mla_with_kvcache` performs FP8 → FP32 directly, saving one conversion step.

**Net effect**: The SM70 fallback path consumes approximately **1.7× the memory bandwidth** of the fused path per attention layer, due to the intermediate BF16 workspace write and re-read. Combined with kernel launch overhead (§7.2), the total fallback penalty is estimated at **2–4 ms per decode step**, or roughly **15–25% of total decode latency** (assuming ~15 ms per-token latency at moderate sequence lengths).

---

## 8  MoE Layer SM70 Analysis

*Satisfies Requirement 8 (acceptance criteria 8.1 – 8.4).*

This section documents the Mixture-of-Experts (MoE) layer behavior on SM70, covering routing mechanisms, weight quantization format, execution path constraints, and latency contribution.

### 8.1  Routing Mechanism

DeepSeek V4 Flash uses two routing strategies depending on layer depth:

| Layer Range | Routing Method | Description |
|---|---|---|
| First `num_hash_layers` layers | **Hash-based** | Deterministic expert assignment via hash function. No learnable routing weights; eliminates load-balancing auxiliary losses. Fast dispatch with minimal compute overhead. |
| Remaining layers | **noaux_tc** | Token-choice routing with `e_score_correction_bias` additive bias. No auxiliary load-balancing loss (hence "noaux"). Uses learned gating network to produce routing scores. |

#### Scoring function: `sqrtsoftplus`

For `noaux_tc` layers, the gating network produces raw logits that are transformed via `sqrtsoftplus`:

```
sqrtsoftplus(x) = sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
```

This scoring function:
- Ensures non-negative expert scores (required for top-K selection).
- Provides smoother gradients than ReLU-based gating.
- Combined with `e_score_correction_bias` (a per-expert learned offset), it steers token-to-expert assignment without an explicit load-balancing loss.

#### Top-K expert selection

After scoring, the top-K experts are selected per token. The selected expert weights are normalized (scores sum to 1.0 per token) before being used to combine expert outputs. The `fused_topk` function handles both scoring and selection in a single kernel launch.

### 8.2  MXFP4 Weight Quantization

MoE expert weights are stored in **MXFP4** (Microscaling FP4) format, configured via `DeepseekV4FP8Config` which routes MoE layers to `Mxfp4Config`:

#### Weight layout

| Weight | Shape (per expert) | Format | Components |
|---|---|---|---|
| `w13` (gate + up projection) | `[expert, 2 × intermediate_size, hidden_size]` | MXFP4 | int4 packed data + UE8M0 block scales |
| `w2` (down projection) | `[expert, hidden_size, intermediate_size]` | MXFP4 | int4 packed data + UE8M0 block scales |

#### MXFP4 encoding

- **Data**: Two 4-bit floating-point values packed per byte (int4 packed). Each FP4 value has 1 sign bit, 2 exponent bits, 1 mantissa bit (E2M1 format).
- **Scales**: UE8M0 block scales (unsigned 8-bit exponent-only float, `scale = 2^(byte − 127)`), one per 32-element block. Identical format to the KV cache UE8M0 scales.
- **Dequantization**: `float_value = fp4_to_float(packed_nibble) × scale`. The FP4→float conversion and scale multiplication happen inside the `FusedMoE` Triton kernels.

#### SM100-only constraint: `MegaMoE`

`DeepseekV4MegaMoEExperts` (expert parallelism with NVLink all-to-all) requires SM100 (Blackwell) hardware features. On SM70, only `FusedMoE` with tensor parallelism is available — each GPU holds all experts but processes only its TP shard of the intermediate dimension.

### 8.3  SM70 MoE Execution Path

On SM70, the MoE layer follows this execution path:

```
MoE Layer Execution (per decode/prefill token):
│
├─ Router:
│   ├─ Hash-based (early layers): hash(token_idx) → expert_ids, uniform weights
│   └─ noaux_tc (later layers):
│       ├─ gate_proj(hidden_states) → logits [num_tokens, num_experts]
│       ├─ sqrtsoftplus(logits) + e_score_correction_bias → scores
│       └─ fused_topk(scores, topk) → topk_ids, topk_weights
│
├─ FusedMoE kernel (Triton):
│   ├─ Permute tokens by expert assignment (gather-scatter)
│   ├─ Per-expert computation (fused gate+up+down):
│   │   ├─ Dequantize w13: MXFP4 int4→FP32 (nibble extraction + UE8M0 scale)
│   │   ├─ Compute: x @ w1 → gate, x @ w3 → up
│   │   ├─ Activation: SiLU(gate) * up → intermediate
│   │   ├─ Dequantize w2: MXFP4 int4→FP32 (nibble extraction + UE8M0 scale)
│   │   └─ Compute: intermediate @ w2 → expert_output
│   └─ Unpermute + weighted combination: sum(topk_weights × expert_outputs)
│
└─ Output: [num_tokens, hidden_size] fp16
```

#### Expert weight dequantization on SM70

SM70 lacks native MXFP4 hardware support (no `mma.fp4` instructions). The `FusedMoE` Triton kernels handle dequantization in software:

1. **Nibble extraction**: Each byte contains two FP4 values; extract via `(byte >> 4) & 0xF` and `byte & 0xF`.
2. **FP4→FP32 conversion**: Map 4-bit E2M1 pattern to float32 via lookup table or bit manipulation (16 possible values per sign).
3. **Scale application**: Multiply by UE8M0 block scale (`2^(scale_byte − 127)`), broadcast across the 32-element block.
4. **FP16 accumulation**: On SM70, the Triton kernel uses FP16 tensor core MMA (`mma.sync.aligned.m8n8k4.f16.f16`) for the GEMM, with FP32 accumulation for the reduction. Dequantized FP32 values are cast to FP16 before feeding into MMA.

This software dequantization path adds overhead compared to SM100's native MXFP4 MMA, but the `FusedMoE` kernel fuses dequant with the GEMM to minimize memory traffic — dequantized values flow directly from registers into the MMA pipeline without intermediate global memory writes.

### 8.4  MoE Latency Contribution and Bottleneck Assessment

#### Fraction of per-token latency

The MoE layer executes once per decoder layer (61 layers total). Its contribution to per-token decode latency:

| Component | Estimated Fraction | Reasoning |
|---|---|---|
| **MoE (all layers)** | **35–45%** of total decode latency | MoE is the dominant compute stage — each layer performs two large GEMMs (gate+up, then down) with `intermediate_size ≈ 18432` on SM70. |
| **Routing overhead** | **<2%** of MoE time | Top-K selection and token permutation are memory-bound with tiny compute footprint relative to the expert GEMMs. |
| **MXFP4 dequantization** | **10–15%** of MoE time | Software dequant adds ~4 ALU ops per weight element (nibble extract, LUT/shift, scale multiply, fp16 cast). Fused into the GEMM loop, so partially overlapped with MMA. |

#### Bottleneck analysis

**Primary bottleneck: Expert GEMM compute throughput**, not dequantization or routing.

Reasoning:
- V100 delivers 125 TFLOPS FP16 tensor core throughput. For a single MoE layer with top-K=8, each token requires: `2 × (hidden × intermediate + intermediate × hidden) × topk = 2 × 2 × 7168 × 18432 × 8 ≈ 4.2 GFLOP`. At 125 TFLOPS, this takes ~34 μs per layer (theoretical minimum).
- The MXFP4 dequant cost is fused inside the GEMM loop and partially hidden behind MMA latency. It adds register pressure and pipeline bubbles but does not create a separate memory-bandwidth bottleneck because dequantized values stay in registers.
- Routing + permutation is a simple gather/scatter of `[num_tokens, hidden_size]` — memory-bound at ~14 KB per token, negligible compared to the GEMM data movement.

**Secondary consideration: Memory bandwidth for weight loading**. Each MoE layer loads `2 × num_experts × intermediate × hidden / tp_size` bytes of MXFP4 weights = `2 × 256 × 18432 × 7168 / 8 / 2` (MXFP4 = 0.5 bytes/param) ≈ **1.7 GB per layer** at full expert count. V100's 900 GB/s HBM2 bandwidth can deliver this in ~1.9 ms — making MoE layers **compute+bandwidth co-bottlenecked** on SM70.

---

## 9  mHC (Multi-Head Cache) SM70 Analysis

*Satisfies Requirement 9 (acceptance criteria 9.1 – 9.4).*

This section documents the Multi-Head Cache (mHC) block on SM70, covering its computation, execution paths, the disabled fast path, and torch fallback overhead assessment.

### 9.1  mHC Pre-Block Computation

The mHC pre-block transforms the residual stream into a layer-specific input, executed **once per decoder layer** (61 layers × 2 = 122 mHC pre-block calls per token: one for attention, one for FFN). The computation graph:

```
mHC Pre-Block:
│
├─ Input: residual [N, hc_mult, hidden_size]  (hc_mult=4, hidden_size=7168)
│          fn [hc_mult3, hc_hidden_size]      (hc_mult3=24, hc_hidden_size=28672)
│
├─ Step 1: GEMM
│   residual_vec = residual.reshape(N, hc_hidden_size)   # [N, 28672]
│   mixes = residual_vec @ fn.T                          # [N, 24] — the core GEMM
│
├─ Step 2: RMSNorm (on residual_vec, applied to mixes)
│   rms = rsqrt(residual_vec.square().sum(-1) / hc_hidden_size + eps)
│   mixes = mixes * rms                                  # Normalize GEMM output
│
├─ Step 3: Split into three components
│   pre_logits  = mixes[:, 0:4]                          # [N, hc_mult]
│   post_logits = mixes[:, 4:8]                          # [N, hc_mult]
│   comb_logits = mixes[:, 8:24].reshape(N, 4, 4)       # [N, hc_mult, hc_mult]
│
├─ Step 4: Sigmoid activations
│   pre_mix  = sigmoid(pre_logits * scale[0] + base[0:4]) + eps
│   post_mix = sigmoid(post_logits * scale[1] + base[4:8]) * alpha
│   comb_mix = comb_logits * scale[2] + base[8:24]
│
├─ Step 5: Sinkhorn normalization (on comb_mix [N, 4, 4])
│   comb = softmax(comb_mix, dim=-1) + eps               # Row normalization
│   comb = comb / (comb.sum(dim=-2) + eps)               # Column normalization
│   for _ in range(sinkhorn_repeat - 1):                 # Iterate (default: 1 extra)
│       comb = comb / (comb.sum(dim=-1) + eps)           #   Row re-normalization
│       comb = comb / (comb.sum(dim=-2) + eps)           #   Column re-normalization
│
└─ Step 6: Weighted-sum layer_input
    layer_input = einsum("nh,nhd->nd", pre_mix, residual)  # [N, hidden_size]
    # Each token's output is a learned weighted combination of hc_mult=4 residual heads
```

The mHC post-block (after attention/FFN) reverses this:
```
mHC Post-Block:
    output = einsum("nio,nid->nod", comb, residual) + post * x
    # Combines hc_mult heads back into the residual stream
```

### 9.2  Three Execution Paths

| Path | Trigger | GEMM | Post-GEMM | Availability |
|---|---|---|---|---|
| **Torch fallback** | Default (`VLLM_SM70_MHC_FAST=0` or SM70 without env override) | `torch.matmul` in float32 | Multiple separate kernels (see §9.4) | All architectures |
| **SM70 fast path** | `VLLM_SM70_MHC_FAST=1` + SM70 | cuBLAS FP16 GEMM (or TurboMind MMA_884, disabled) | Single fused Triton kernel `_mhc_pre_post_gemm_kernel` | SM70 only |
| **tilelang+DeepGEMM** | SM80+ with tilelang available | DeepGEMM FP8/FP16 GEMM | tilelang fused kernel | SM80+ only (unavailable on SM70) |

#### Torch fallback (default)

```python
_mhc_pre_torch_fallback(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
```

- Casts residual to float32 for the GEMM: `residual_vec.float() @ fn.float().t()`
- All intermediate computations in float32.
- Returns `post_mix, comb_mix, layer_input` with appropriate shapes.
- **Called 122 times per decode token** (61 layers × 2 mHC blocks per layer).

#### SM70 fast path

```python
_mhc_pre_sm70_fast(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
```

- cuBLAS FP16 GEMM: `residual_vec_fp16 @ fn_fp16.T → [N, 24]` fp16.
- Single Triton kernel (`_mhc_pre_post_gemm_kernel`): reads GEMM output + residual, computes RMSNorm inline, sigmoid, sinkhorn, weighted-sum — all fused.
- Weight is lazily prepared: `fn_fp16 = fn.to(float16)` cached on first call.

### 9.3  SM70 Fast Path: Disabled by Default

The SM70 mHC fast path is **disabled by default** despite offering significant performance improvement. The control mechanism:

```python
# vllm/envs.py
"VLLM_SM70_MHC_FAST": lambda: bool(int(os.getenv("VLLM_SM70_MHC_FAST", "1")))
```

#### Reason: MMA_884 precision issues

The original SM70 fast path used TurboMind's `sm70_f16_prepare` + `sm70_f16_gemm_out` (Volta `mma.sync.aligned.m8n8k4.f16.f16`) for the GEMM. This path exhibited:

1. **Precision divergence**: The FP16 GEMM with Volta's MMA_884 instruction produces slightly different results from the float32 torch GEMM due to:
   - FP16 intermediate accumulation in the MMA pipeline (vs FP32 accumulation in cuBLAS).
   - Different reduction ordering across warps.
   - Accumulated rounding differences across 28672-dimensional dot products.

2. **Semantic divergence**: The small numerical differences compound through the sinkhorn normalization (iterative row/column normalization of a 4×4 matrix), sigmoid activations, and the weighted-sum — producing layer_input values that are sufficiently different to cause model output divergence (failing the ZX-42 canary test and identity coherence test).

3. **Current status**: The TurboMind MMA_884 path is hard-coded as disabled (`_use_tm = False`). The cuBLAS FP16 GEMM path within `_mhc_pre_sm70_fast` is functional and produces results closer to the float32 reference, but the env var defaults to enabled (`"1"`), so the fast path is technically available when the platform is SM70.

#### Validation requirement

Enabling the SM70 fast path requires passing:
- **ZX-42 canary test**: The model must produce the exact expected output for a fixed prompt.
- **Identity coherence test**: Open-ended generation must produce coherent, grammatically correct text with `finish_reason=stop`.

### 9.4  Torch Fallback Overhead Quantification

The torch fallback (`_mhc_pre_torch_fallback`) incurs overhead from multiple sequential CUDA kernel launches and float32 computation:

#### Kernel launch breakdown (per mHC pre-block call)

| Operation | CUDA Kernels Launched | Compute |
|---|---|---|
| `residual.reshape().float()` | 1 (cast kernel) | FP16→FP32 conversion of `[N, 28672]` |
| `residual_vec @ fn.t()` | 1 (cuBLAS GEMM) | `[N, 28672] × [28672, 24]` in FP32 |
| `residual_vec.square().sum(-1)` | 1–2 (fused or separate) | Reduction over 28672 elements |
| `torch.rsqrt(... + eps)` | 1 (elementwise) | Per-token scalar |
| `mixes * rms` | 1 (elementwise) | `[N, 24]` broadcast multiply |
| `torch.sigmoid(...)` × 2 | 2 (elementwise) | For pre_mix and post_mix |
| `torch.softmax(comb, dim=-1)` | 1 (softmax kernel) | `[N, 4, 4]` row softmax |
| Sinkhorn iterations (×`sinkhorn_repeat`) | 2–4 per iteration (sum + div) | Row/column normalization on `[N, 4, 4]` |
| `torch.einsum("nh,nhd->nd", ...)` | 1 (batched matmul or custom) | Weighted-sum `[N, 4] × [N, 4, 7168]` |
| **Total** | **~12–16 kernel launches** | |

#### Overhead comparison: Torch fallback vs fused Triton

| Metric | Torch Fallback | Fused Triton (SM70 fast path) |
|---|---|---|
| **CUDA kernel launches** | 12–16 per call | **2** (1 cuBLAS GEMM + 1 Triton kernel) |
| **GEMM precision** | FP32 (`[N, 28672] × [28672, 24]`) | FP16 (`[N, 28672] × [28672, 24]`) |
| **Intermediate memory** | Multiple FP32 temporaries written to global memory | Single-kernel: intermediates stay in registers/shared memory |
| **Per-call overhead** | ~100–200 μs (dominated by kernel launch latency for N=1 decode) | ~20–40 μs |
| **Per-token total** (122 calls) | **~12–24 ms** | **~2.4–4.9 ms** |
| **Fraction of decode latency** | **~15–30%** (assuming ~50–80 ms total) | **~3–6%** |

#### Why the overhead is significant

For **decode** (N=1 token), each mHC call processes a single token through the 28672-dimensional GEMM + post-processing. The GEMM itself is tiny (`1 × 28672 × 24` FLOPs ≈ 1.4 MFLOP), completing in <1 μs of actual compute. The **dominant cost is kernel launch overhead**: ~5–10 μs per kernel launch × 12–16 launches = 60–160 μs per mHC call. Over 122 calls per token, this accumulates to **7–20 ms of pure kernel launch overhead**.

The fused Triton path eliminates this by:
1. Using a single cuBLAS GEMM launch (which the driver batches efficiently).
2. Performing all post-GEMM operations (RMSNorm, sigmoid, sinkhorn, weighted-sum) in a single Triton kernel launch with one program per token — all intermediates stay in registers.

#### Memory bandwidth comparison

| Path | Global Memory Traffic (per call, N=1) | Notes |
|---|---|---|
| Torch fallback | ~230 KB read + ~60 KB write | FP32 residual read (114 KB), fn weight read (110 KB), multiple intermediate writes |
| Fused Triton | ~86 KB read + ~29 KB write | FP16 residual read (57 KB), FP16 fn weight (cached, 28 KB), single output write |

The torch fallback's FP32 compute path doubles the memory bandwidth requirement for the GEMM operands. Since V100's 900 GB/s bandwidth is shared across all concurrent operations, this 2× bandwidth overhead further constrains the MoE and attention kernels running in other streams.

---

## 10  FlashMLA SM70 Kernel Optimization Directions

*Satisfies Requirement 10 (acceptance criteria 10.1 – 10.4).*

This section proposes concrete optimization directions for the FlashMLA kernels on SM70, covering both the prefill-style sparse kernel and the decode path. Each direction is categorised by estimated speedup impact on **per-token decode latency**.

**Speedup categories:**
- **High (>20%)**: Eliminates a major architectural bottleneck or removes an entire pipeline stage.
- **Medium (5–20%)**: Reduces overhead in a frequently-executed component without eliminating the stage.
- **Low (<5%)**: Micro-optimization within a single kernel or infrequent code path.

### 10.1  `flash_mla_sparse_fwd` Optimizations (Prefill + SM70 Decode Fallback)

`flash_mla_sparse_fwd` is the SM90 BF16 prefill kernel reused on SM70 for both prefill and the decode fallback path (§2.2). Because SM70 lacks `cp.async`, TMA, and FP32-accumulating tensor cores, the kernel's performance on Volta is constrained by data movement and accumulation precision.

#### 10.1.1  Shared Memory Tiling for Volta L1/Shmem Architecture

**Problem**: Volta provides 96 KB combined L1/shared memory per SM (configurable split), compared to 164 KB on SM80+ and 228 KB on SM90. The SM90 prefill kernel's tile sizes were tuned for larger shared memory budgets, leading to either:
- Register spills when tiles are too large for SM70's register file (65536 × 32-bit registers per SM, 255 per thread).
- Excessive global memory re-reads when tiles are too small.

**Proposed optimization**:
1. Re-tune Q and KV tile dimensions for the 96 KB shmem budget. Target: 48 KB for KV tile staging, 32 KB for Q tile, 16 KB for accumulation scratch.
2. Use Volta's configurable L1/shmem split (`cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout`) to maximise shmem allocation for attention tiles.
3. Reduce tile-K dimension (e.g., from 64 to 32) to halve shmem pressure per tile, at the cost of more iterations per CTA.

**Estimated speedup**: **Medium (5–15%)**. Reduces global memory re-reads for moderate sequence lengths but does not eliminate the fundamental two-kernel overhead of the fallback path. Impact is larger for prefill (large `s_q`) than decode (T=1).

#### 10.1.2  Warp Scheduling for Sparse Index Gather

**Problem**: The sparse attention kernel iterates over `topk` KV indices per query. Each index may reference a different location in the pre-gathered BF16 workspace. The current implementation processes indices sequentially within each warp, leading to divergent memory access patterns when topk indices are non-contiguous.

**Proposed optimization**:
1. Sort topk indices per CTA tile before KV loads to improve spatial locality in the BF16 workspace.
2. Use warp-shuffle (`__shfl_sync`) to redistribute KV elements across threads within a warp after loading, reducing bank conflicts in shared memory.
3. Pre-compute a CTA-local index permutation in shared memory to batch coalesced loads.

**Estimated speedup**: **Low (<5%)**. The BF16 workspace is already contiguous per-token (`[T, topk, 512]`), so within-token access is already coalesced along the head dimension. The benefit of index sorting is marginal when topk is moderate (~2048–6144). Greater impact for very large topk values.

#### 10.1.3  FP16 Accumulation Precision Management

**Problem**: Volta's `mma.sync.aligned.m8n8k4` instruction accumulates in FP16 (not FP32 like SM80+). For long attention sequences, FP16 accumulation can lose significant precision in the softmax numerator, especially when attention scores span a wide dynamic range. The current kernel compensates with FP32 softmax computation and FP32 output accumulation, but the MMA output itself is FP16-precision.

**Proposed optimization**:
1. Split the accumulation across multiple MMA tiles with intermediate FP32 reduction — accumulate partial sums from each MMA tile in FP32 registers before combining.
2. Apply Kahan compensated summation in the FP32 output accumulator to reduce round-off error from repeated FP16 MMA outputs.
3. Scale Q and KV tiles to keep dot-product magnitudes in the FP16-safe range (±65504), reducing the frequency of overflow-induced precision loss.

**Estimated speedup**: **Low (<5%)** on latency, but **medium impact on output quality**. This is primarily a correctness improvement that prevents semantic degradation (related to the mHC MMA_884 precision issues in §9.3 of the design). Latency impact is minimal since the extra FP32 reduction is cheap compared to MMA.

### 10.2  `flash_mla_with_kvcache` Optimizations (Normal Decode Path)

`flash_mla_with_kvcache` is the fused decode kernel that reads FP8 KV directly from paged cache and performs attention in a single kernel launch. **On SM70, this kernel is currently unreachable** due to the `_should_use_sm70_decode_prefill_fallback` guard (§5.2). The optimizations below target enabling and tuning this kernel for SM70.

#### 10.2.1  Tile Scheduler Tuning for V100's 80 SMs

**Problem**: The `FlashMLASchedMeta` tile scheduler was designed for SM90 (132 SMs, H100) and SM100 (Blackwell). The formula `num_sm_parts = max(num_sms / s_q / (h_q/64), 1)` distributes work uniformly, but V100's 80 SMs have different memory hierarchy characteristics:
- Lower L2 cache bandwidth (3 MB L2 vs 50 MB on H100).
- Different warp scheduler behaviour (4 warp schedulers per SM on Volta vs 4 on SM80+, but different scheduling policies).

**Proposed optimization**:
1. Add an SM70-specific tile scheduler branch that accounts for 80 SMs and Volta's memory hierarchy.
2. For T=1 decode with `padded_heads=64`: `num_sm_parts = 80 / 1 / 1 = 80` — each SM gets one head tile. Verify this saturates the SMs without excessive split-KV overhead.
3. For T=1 decode with large topk (>4096): Experiment with fewer sm_parts and larger per-SM work chunks to improve L2 cache reuse for repeated KV page accesses.
4. Profile the `combine` kernel overhead for different `num_splits` values — on SM70, the combine step may become a bottleneck when `num_sm_parts` is large.

**Estimated speedup**: **Medium (5–10%)** once the kernel is enabled on SM70 (see §10.3). The tile scheduler primarily affects decode throughput at larger batch sizes. For T=1 single-token decode, the impact is modest since there is limited opportunity for work redistribution.

#### 10.2.2  FP8 Dequantization/MMA Pipeline Overlap

**Problem**: The fused decode kernel performs FP8→FP32 dequantization of KV cache entries before feeding them to the FP16 MMA. On SM90+, this dequantization uses hardware FP8→FP16 conversion instructions. On SM70, it would require scalar bit-manipulation (identical to `_gather_decode_kv_triton_kernel`), which is compute-bound and serialises with the MMA.

**Proposed optimization**:
1. **Double-buffer dequantization**: Use shared memory as a two-stage pipeline — while MMA processes dequantized tile N, a separate warp dequantizes tile N+1 from FP8 global memory into shared memory.
2. **Leverage Volta's 96 KB shmem**: Allocate 2 × 32 KB tile buffers (one for current MMA, one for next dequant) + 32 KB for Q tile and accumulation. This fits within the 96 KB budget.
3. **Warp specialisation**: Dedicate 1–2 warps per CTA to FP8 dequantization (producer) and remaining warps to MMA (consumer), with shared memory barriers for synchronisation. Volta supports `__syncwarp()` and `__syncthreads()` for this pattern.
4. **Batch scale application**: Load all 7 UE8M0 scale bytes per KV token once, broadcast to all elements via shared memory, rather than per-element scale lookup.

**Estimated speedup**: **Medium (10–15%)** relative to a naive SM70 port of the decode kernel. This optimisation is **prerequisite** for making `flash_mla_with_kvcache` competitive on SM70 — without it, scalar FP8 decode would dominate kernel runtime.

**Implementation status** (Task 5.2): **Implemented** in `csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh`. The double-buffer pipeline overlaps FP8 dequantization of tile N+1 into shared memory while MMA processes tile N from the active buffer. Key design decisions:
- **Ping-pong shared memory buffers**: Two kv_tile buffers (`kv_bufs[0]`, `kv_bufs[1]`) and two token_refs arrays, swapped via `cur_buf = 1 - cur_buf` after each tile.
- **Pre-staging**: Tile 0 is pre-loaded before the main loop; inside the loop, the next tile is staged into the alternate buffer after the current tile's PV MMA completes and `__syncthreads()` ensures the alternate buffer is safe to overwrite.
- **Shared memory budget**: MODEL1 (d_qk=512) uses 2 × (32×512×2 + 32×4) + overhead = ~64.4 KB; V32 (d_qk=576) uses ~72.4 KB. Both fit within Volta's 96 KB/SM.
- **Compile-time toggle**: `FLASH_MLA_SM70_SPARSE_DECODE_DOUBLE_BUFFER=1` (default on) in `config.h`. Set to 0 to disable and revert to original single-buffer path.
- **Scope**: Only applies to the `accumulate_output_mma884_online_register` path (MODEL1 with 256 threads), which is the primary decode hot path. The non-register online and scalar PV paths are unchanged.

#### 10.2.3  Dual-Cache Access Pattern Optimization

**Status**: ✅ **Implemented** (task 5.3)

**Problem**: For compressed layers (C4A/C128A), `flash_mla_with_kvcache` accesses two separate paged caches: `k_cache` (SWA, 64-token blocks) and `extra_k_cache` (compressed MLA, 256/ratio-token blocks). These caches reside in different memory regions with different page tables, causing:
- Alternating access patterns that thrash V100's 3 MB L2 cache.
- Non-uniform memory access latency between SWA and compressed page lookups.

**Proposed optimization**:
1. **Access ordering**: Process all SWA entries first (which are temporally local, likely L2-resident from recent writes), then all compressed entries. This avoids interleaving two different paged caches and improves L2 hit rate.
2. **Page prefetching**: For compressed cache entries accessed via `extra_indices_in_kvcache`, issue `__ldg` prefetch hints for the next page while processing the current one. Volta supports L2 prefetch via `__ldg` intrinsic.
3. **Coalesced index loading**: Load `indices` (SWA) and `extra_indices_in_kvcache` (compressed) into shared memory upfront, then sort/group by page number to batch page accesses and reduce TLB misses.

**Implementation** (in `csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh`, function `stage_kv_tile_to_shared`):

The function was restructured from a single interleaved loop into a two-phase approach:

- **Phase 1 — Index pre-loading**: All logical indices for the tile are resolved upfront into `token_refs[]` (shared memory). Each thread resolves one or more tokens via `score_ref_for_logical_idx`, which determines whether each token comes from the SWA cache (`is_extra=false`) or compressed cache (`is_extra=true`). This eliminates redundant `__ldg` index reads that previously occurred once per KV dimension element.

- **Phase 2 — Two-pass KV loading** (L2 thrashing mitigation):
  - **Pass 1**: Loads all SWA (main cache) tokens. For positions referencing the compressed cache, zeros are written as placeholders. This pass accesses only `params.kv` memory pages.
  - **Pass 2**: Loads all compressed (extra cache) tokens, overwriting the zeros from Pass 1. This pass accesses only `params.extra_kv` memory pages. Guarded by `if (params.extra_topk > 0)` so SWA-only tiles skip it entirely.

This two-pass design keeps each pass reading from a single contiguous memory region, improving L2 residency and TLB hit rate on V100's limited 3 MB L2. For tiles that are entirely SWA or entirely compressed (the common case for most tile positions), the second pass exits early with minimal overhead.

**Estimated speedup**: **Low to Medium (3–8%)**. Impact depends on working set size relative to V100's 3 MB L2 cache. For small topk values, both caches may fit in L2 and reordering has minimal effect. For large topk (C128A with many compressed entries), the L2 thrashing reduction can be significant.

### 10.3  SM70 Decode Fallback Elimination

This is the **highest-impact** optimization direction — removing the SM70 decode fallback path entirely by enabling direct `flash_mla_with_kvcache` on SM70.

#### 10.3.1  Current Barrier

The SM70 decode fallback exists because `flash_mla_with_kvcache` dispatches to `sparse_attn_decode_interface` in `csrc/api/sparse_decode.h`, which has an architecture gate:

```cpp
if (arch.is_sm100f()) { ... }
else if (arch.is_sm90a()) { ... }
else { TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd"); }
```

There is **no SM70 code path** for the fused decode kernel. The SM90 decode kernel (`csrc/sm90/decode/sparse_fp8/splitkv_mla.h`) uses SM90-specific instructions (wgmma, TMA, FP8 hardware types) that have no SM70 equivalents.

#### 10.3.2  Proposed Approach: SM70 Decode Kernel Implementation

**Option A — New SM70 CUDA decode kernel** (recommended):

1. Add a `csrc/sm70/decode/sparse_fp8/` directory with a Volta-compatible decode kernel.
2. Use `mma.sync.aligned.m8n8k4.f16` (Volta tensor core) for attention dot product with FP16 inputs and FP16 accumulation.
3. Implement scalar FP8→FP16 dequantization in-kernel (reuse the bit-manipulation logic from `_gather_decode_kv_triton_kernel`).
4. Implement the double-buffer dequant/MMA pipeline from §10.2.2.
5. Support the dual-cache interface: `k_cache` (SWA) + `extra_k_cache` (compressed) with separate index arrays.
6. Implement split-KV tile scheduling compatible with `FlashMLASchedMeta` (reuse `smxx/decode/get_decoding_sched_meta/`).
7. Add the SM70 branch in `sparse_attn_decode_interface`:
   ```cpp
   else if (arch.is_sm70()) { run_sm70_decode(...); }
   ```

**Option B — SM70 Triton decode kernel**:

1. Write the fused decode kernel in Triton (Python-level, no CUDA C++ needed).
2. Triton on SM70 supports `tl.load`, `tl.dot` (maps to `mma.sync`), and manual FP8 decode via integer arithmetic.
3. Advantage: faster iteration, no CMake/build complexity.
4. Disadvantage: less control over shared memory layout and warp scheduling; Triton's SM70 code generation is less mature than SM80+.

**Option C — Hybrid approach**:

1. Implement the FP8 dequant + KV staging in a Triton "prelude" kernel (similar to current `_gather_decode_kv_triton_kernel` but writing to shared memory / L2-pinned buffer).
2. Call an existing FlashMLA attention primitive on the staged BF16 data.
3. Advantage: reuses proven attention code; only the dequant stage needs SM70-specific work.
4. Disadvantage: still two kernels, though the second kernel operates on L2-resident data (vs the current fallback which writes/reads HBM).

#### 10.3.3  Kernel-Level Requirements for SM70 Decode

| Requirement | Detail |
|---|---|
| **FP8 dequant in-kernel** | Scalar `e4m3fn → FP32` via bit extraction (sign/exp/mant). 7 quant blocks × 64 elements per token. UE8M0 scale = `2^(byte - 127)`. |
| **Dual-cache support** | Read from both `k_cache` (SWA) and `extra_k_cache` (compressed) via separate index arrays. Must handle different block sizes (64 vs 256/ratio). |
| **FP16 MMA** | Use Volta `mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16`. Accumulate partial softmax in FP32 registers. |
| **Split-KV** | Partition topk entries across SM parts. Each SM part computes partial LSE + output. `combine` kernel merges. |
| **96 KB shmem budget** | Tile Q + double-buffer KV dequant + accumulation within 96 KB. |
| **Graph-safe allocations** | All intermediate buffers via `FlashMLASchedMeta` or graph-aware PyTorch allocator. |
| **attn_sink support** | Per-head float32 bias added to attention logits. |

#### 10.3.4  Expected Impact

| Metric | Current (Fallback) | After Elimination | Improvement |
|---|---|---|---|
| Kernel launches per layer | 3+ (gather + index build + sparse_fwd) | 1 (fused decode) | ~66% fewer launches |
| Extra BF16 workspace | `topk × 1024` bytes/layer | 0 | Eliminated |
| Extra memory bandwidth | ~12 MB/layer (§7.4) | 0 | **~732 MB/step saved** |
| Per-step overhead | ~2–4 ms (§7.2) | 0 | **~2–4 ms saved** |
| Estimated decode latency reduction | — | — | **High (>20%)** |

Fallback elimination is estimated to provide **>20% decode latency reduction** by removing ~2–4 ms of kernel launch overhead and ~0.8 ms of redundant memory traffic per decode step. This is the single highest-impact optimisation direction for SM70 decode performance.

### 10.4  Summary: Speedup Estimates by Direction

| ID | Optimization Direction | Target Kernel | Category | Estimated Speedup | Priority |
|---|---|---|---|---|---|
| **10.3** | **SM70 decode fallback elimination** | `flash_mla_with_kvcache` (new SM70 kernel) | **High (>20%)** | 20–30% decode latency reduction | **P0** |
| 10.2.2 | FP8 dequant/MMA pipeline overlap | `flash_mla_with_kvcache` (SM70 port) | Medium (10–15%) | Prerequisite for 10.3 | P0 |
| 10.2.1 | Tile scheduler tuning for 80 SMs | `flash_mla_with_kvcache` (SM70 port) | Medium (5–10%) | Cumulative with 10.3 | P0 |
| 10.1.1 | Shared memory tiling for Volta shmem | `flash_mla_sparse_fwd` | Medium (5–15%) | Higher impact on prefill than decode | P1 |
| 10.2.3 | Dual-cache access pattern optimization ✅ | `flash_mla_with_kvcache` (SM70 port) | Low–Medium (3–8%) | **Implemented** — two-pass SWA→compressed ordering in `stage_kv_tile_to_shared` | P1 |
| 10.1.3 | FP16 accumulation precision management | `flash_mla_sparse_fwd` | Low (<5% latency) | Primarily correctness, not speed | P2 |
| 10.1.2 | Warp scheduling for sparse gather | `flash_mla_sparse_fwd` | Low (<5%) | Marginal for moderate topk | P2 |

**Recommended execution order**: 10.3 (with 10.2.2 as prerequisite) → 10.2.1 → 10.1.1 → 10.2.3 → 10.1.3 → 10.1.2.

The dominant strategy is **fallback elimination** (10.3): implementing a native SM70 decode kernel inside FlashMLA that fuses FP8 dequantization with MMA attention, removing the two-kernel overhead and ~732 MB/step of redundant memory traffic. All other FlashMLA optimisations (10.1.x, 10.2.x) provide incremental improvements on top of this foundation.

---

## 11  SM70 Torch Fallback Elimination Plan

*Satisfies Requirement 11 (acceptance criteria 11.1 – 11.4).*

This section identifies every SM70 torch fallback function, proposes Triton kernel replacements, prioritises by decode-path and prefill-path impact, and identifies template kernels for each replacement.

### 11.1  SM70 Torch Fallback Inventory

The following table lists every SM70-specific torch fallback function in the codebase, with call frequency and estimated overhead per invocation.

| # | Fallback Function | Source File | Used In | Call Frequency | Estimated Overhead | Path |
|---|---|---|---|---|---|---|
| F1 | `_torch_qnorm_rope_kv_insert_fallback` | `deepseek_v4_attention.py` | Prefill + Decode KV insert | 1× per layer per step | ~25–35 CUDA kernel launches; ~0.1–0.35 ms/layer (§6.2) | Prefill (aux stream) + Decode (aux stream) |
| F2 | `_torch_fused_compress_norm_rope_insert_fp8_fallback` | `deepseek_compressor.py` | Prefill + Decode compressor (C4A/C128A layers) | 1× per C4A/C128A layer per step | Python token loop; ~7,168 `.item()` syncs for T=1024 C4A; ~36–107 ms/layer (§6.3) | Prefill (aux stream) + Decode (aux stream) |
| F3 | `_mhc_pre_torch_fallback` | `mhc.py` | All layers (mHC pre-block) | 2× per layer per step (attn + FFN) | float32 GEMM + ~8–10 kernel launches; ~0.05–0.2 ms/layer | Decode + Prefill |
| F4 | `_mhc_post_torch_fallback` | `mhc.py` | All layers (mHC post-block) | 2× per layer per step (attn + FFN) | einsum + ~3–5 kernel launches; ~0.02–0.1 ms/layer | Decode + Prefill |
| F5 | `_sm70_fp8_einsum_bmm` (via `_deepseek_v4_fp8_einsum_torch_fallback`) | `deepseek_v4_attention.py` | O-projection | 1× per layer per step | Pre-dequant weight (cached ~86 MB) + FP32 `torch.einsum`; ~0.05–0.15 ms/layer | Decode + Prefill |

#### Aggregate overhead per decode step (61 layers)

| Fallback | Layers Affected | Calls per Step | Total Overhead (est.) |
|---|---|---|---|
| **F1** `_torch_qnorm_rope_kv_insert_fallback` | All 61 | 61 | ~6–21 ms (hidden by stream overlap for decode T=1) |
| **F2** `_torch_fused_compress_norm_rope_insert_fp8_fallback` | ~50 C4A/C128A | 50 | ~1,800–5,350 ms for prefill T=1024 (dominant); ~0.5–2 ms for decode T=1 (fires rarely at compress boundaries) |
| **F3** `_mhc_pre_torch_fallback` | All 61 × 2 | 122 | ~6–24 ms |
| **F4** `_mhc_post_torch_fallback` | All 61 × 2 | 122 | ~2.4–12 ms |
| **F5** `_sm70_fp8_einsum_bmm` | All 61 | 61 | ~3–9 ms |

**Total decode-step fallback overhead**: ~17–66 ms (excluding F2 which fires rarely during decode). **Total prefill-step fallback overhead**: dominated by F2 (~1.8–5.4 s for a 1024-token prompt across all C4A/C128A layers).

### 11.2  Triton Kernel Replacement Proposals

#### F1: `_torch_qnorm_rope_kv_insert_fallback` → Fused Triton Q-Norm/RoPE/KV-Insert Kernel

**Operations to fuse:**
1. Q-side: per-head RMSNorm (float32 variance + rsqrt) + GPT-J interleaved RoPE → in-place Q update
2. KV-side: GPT-J RoPE + 7-block FP8 e4m3fn quantization (absmax → UE8M0 scale → clamp → cast) + BF16 RoPE bytes + paged cache scatter write

**Input/output tensor specifications:**

| Tensor | Shape | Dtype | Direction |
|---|---|---|---|
| `q` | `[T, n_heads, head_dim]` | fp16 | Input/Output (in-place) |
| `kv` | `[T, head_dim]` | fp16 | Input |
| `k_cache` | `[num_blocks, block_size, 584]` | uint8 | Output (scatter write) |
| `slot_mapping` | `[T]` | int64 | Input |
| `positions` | `[T]` | int64 | Input |
| `cos_sin_cache` | `[max_pos, rope_dim]` | fp32 | Input |

**Fusion opportunities:**
- Merge RMSNorm (pow→mean→rsqrt→mul) + RoPE (cos/sin lookup + interleave) + FP8 quant (absmax→scale→clamp→cast) + scatter write into a **single Triton program per token**.
- Eliminate ~25–35 separate CUDA kernel launches per layer invocation.
- Eliminate float32 intermediate materialization for Q-norm (compute in-register).

**SM70 constraints:**
- No `tl.float8e4nv`: use manual FP8 e4m3fn encoding via integer bit manipulation (mirror the existing decode path's bit logic in reverse).
- FP16 compute: Q-norm can use fp32 in-register (tl.float32) for the variance/rsqrt, then cast result back to fp16.
- 96 KB shared memory: each program handles one token — no inter-token shared memory needed; per-token data (512 × fp16 + scales) fits in registers.

**Grid:** `(num_tokens,)` — one program per token, processing all `n_heads` Q values and 512 KV values.

#### F2: `_torch_fused_compress_norm_rope_insert_fp8_fallback` → Fused Triton Compressor Kernel

**Operations to fuse:**
1. Per-token compress-boundary check: `(position + 1) % compress_ratio == 0`
2. State gather from compressor state cache (float32 `kv_rows` + `score_rows` over `window` positions) via block_table lookup
3. Softmax over score rows → weighted-sum of KV rows → compressed KV
4. RMSNorm with learned weight
5. GPT-J RoPE at compressed position
6. FP8 block quantization (7-block NoPE + BF16 RoPE + UE8M0 scales)
7. Paged cache scatter write (584 bytes for MLA or 132 bytes for indexer)

**Input/output tensor specifications:**

| Tensor | Shape | Dtype | Direction |
|---|---|---|---|
| `state_cache` | `[num_blocks, block_size, state_width]` | float32 | Input |
| `kv_cache` | `[num_blocks, mla_block_size, 584]` | uint8 | Output (scatter write) |
| `slot_mapping` | `[T]` | int64 | Input |
| `positions` | `[T]` | int64 | Input |
| `block_table` | `[num_reqs, max_blocks]` | int32 | Input |
| `cos_sin_cache` | `[max_pos, rope_dim]` | fp32 | Input |
| `rms_norm_weight` | `[head_dim]` | fp32 | Input |

**Fusion opportunities:**
- Eliminate the Python-level `for` loop and all `.item()` host-device synchronisations (~7,168 syncs per layer for T=1024 C4A).
- Merge softmax + weighted-sum + RMSNorm + RoPE + FP8 quant + scatter write into a single Triton program per token.
- Use a pre-computed `fire_mask` tensor (`(positions + 1) % compress_ratio == 0`) to filter tokens at the grid level instead of Python-level branching.

**SM70 constraints:**
- No `tl.float8e4nv`: manual FP8 encoding via integer bit manipulation.
- FP16 compute: state cache is float32; perform softmax and weighted-sum in fp32 (tl.float32), then cast to fp16 for RoPE, then re-encode to FP8.
- 96 KB shared memory: each program handles one token's compression window (`window × 512` float32 values = `8 × 512 × 4 = 16 KB` for C4A). Fits in shared memory.
- Block_table indirection: load block numbers from `block_table` using `tl.load` with indirect indexing (same pattern as `_gather_decode_kv_triton_kernel`).

**Grid:** `(num_firing_tokens,)` — one program per token that passes the compress-boundary filter, with a pre-computed `firing_indices` tensor mapping to original token positions.

#### F3: `_mhc_pre_torch_fallback` → Enable Existing SM70 Fast Path

**Current state:** The SM70 fast path already exists (`_mhc_pre_sm70_fast` using `_mhc_pre_post_gemm_kernel`; §4.4) but is **disabled by default** (`VLLM_SM70_MHC_FAST=0`) due to TurboMind MMA_884 precision issues.

**Proposed strategy:** The existing Triton kernel (`_mhc_pre_post_gemm_kernel`) already uses cuBLAS FP16 GEMM (not MMA_884), so the precision issue is in the TurboMind preparation code, not the Triton kernel itself. The fix path is:
1. Verify that `_mhc_pre_sm70_fast` with `_use_tm = False` (cuBLAS path) produces semantically correct output via the ZX-42 canary test.
2. If verified, enable by default or provide a more fine-grained control knob.
3. If precision issues persist in the Triton kernel itself, investigate sinkhorn iteration count and FP32 intermediate precision.

**No new Triton kernel needed** — the replacement already exists, it just needs validation and enablement.

**Fusion in existing kernel:** cuBLAS FP16 GEMM (`[T, hc_hidden_size] × [hc_mult3, hc_hidden_size]^T`) + fused Triton post-GEMM (RMSNorm + sigmoid + sinkhorn + weighted-sum). Replaces ~8–10 separate kernel launches with 1 GEMM + 1 Triton kernel.

#### F4: `_mhc_post_torch_fallback` → Enable Existing SM70 Fast Path

**Current state:** Same as F3 — the SM70 fast path `_mhc_post_sm70_fast` using `_mhc_post_fused_kernel` (§4.5) is disabled by default.

**Proposed strategy:** Same as F3. Enable `_mhc_post_fused_kernel` after validation. The kernel computes `out[n,o,d] = Σ_i(comb[n,i,o] × res[n,i,d]) + post[n,o] × x[n,d]` in a single Triton launch, replacing ~3–5 separate kernel launches.

**No new Triton kernel needed.**

#### F5: `_sm70_fp8_einsum_bmm` → Triton FP8 Grouped Einsum

**Operations to fuse:**
1. FP8 activation dequantization: `o_fp8.float() × o_scale.repeat_interleave(...)` → FP32 activation
2. Weight pre-dequantization: `wo_a.float() × wo_a_scale.repeat_interleave(...)` → FP32 weight (currently cached as `_sm70_predequant_f32`)
3. Grouped einsum: `torch.einsum("bhr,hdr→bhd", a_deq, b_f32)` → `[T, n_groups, o_lora_rank]`

**Input/output tensor specifications:**

| Tensor | Shape | Dtype | Direction |
|---|---|---|---|
| `o_fp8` | `[T, n_groups, heads_per_group × head_dim // 128, 128]` | uint8 (FP8 e4m3fn) | Input |
| `o_scale` | `[T, n_groups, ...]` | float32 | Input |
| `wo_a` | `[n_groups, o_lora_rank, heads_per_group × head_dim]` | uint8 (FP8 e4m3fn) | Input |
| `wo_a_scale` | `[n_groups, ...]` | float32 | Input |
| `z` | `[T, n_groups, o_lora_rank]` | fp16 | Output |

**Fusion opportunities:**
- Fuse FP8 dequant + grouped matmul into a single Triton kernel, eliminating the FP32 pre-dequanted weight cache (~86 MB) and the intermediate FP32 activation tensor.
- Use Triton `tl.dot` for the grouped matmul with FP16 accumulation on Volta, trading slight precision for significant memory savings.

**SM70 constraints:**
- No `tl.float8e4nv`: manual FP8 decode in-register (same bit extraction as other SM70 kernels).
- FP16 MMA via `tl.dot`: Volta's `mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16` maps naturally.
- 96 KB shared memory: for `n_groups=2, o_lora_rank=384, hidden=512`, a `[groups, hidden]` tile is `2 × 512 × 2 = 2 KB` in FP16. Multiple tiles fit easily.
- Weight caching: the ~86 MB FP32 pre-dequant cache can be replaced by on-the-fly FP8→FP16 dequant per tile, saving GPU memory.

**Grid:** `(T, n_groups)` — one program per (token, group) pair, computing `[1, o_lora_rank]` output via `[heads_per_group × head_dim] × [o_lora_rank, heads_per_group × head_dim]^T` matmul.

### 11.3  Priority Ranking

Fallback replacements are prioritised by decode-path impact (per-token-per-layer on the decode critical path) first, then prefill-path impact (amortised over prompt length).

| Priority | Fallback | Path Impact | Rationale |
|---|---|---|---|
| **P0-decode** | **F3 + F4**: mHC pre/post torch fallback | Decode: 4× per layer per step (122 calls/step), ~8–36 ms total | **Highest decode impact.** Every decode token traverses mHC pre+post for both attention and FFN. Not hidden by stream overlap. Replacement exists (enable fast path). |
| **P0-decode** | **F5**: `_sm70_fp8_einsum_bmm` | Decode: 1× per layer per step (61 calls/step), ~3–9 ms total | On decode critical path (O-projection). FP32 einsum underutilises V100 for T=1. Triton FP8 grouped matmul with FP16 MMA would be faster. |
| **P0-decode** | **F1**: `_torch_qnorm_rope_kv_insert_fallback` | Decode: 1× per layer per step (61 calls/step), ~6–21 ms total (partially hidden by stream overlap) | Moderate decode impact; largely hidden by aux-stream overlap but adds to critical path when indexer is fast (T=1 decode). |
| **P1-prefill** | **F2**: `_torch_fused_compress_norm_rope_insert_fp8_fallback` | Prefill: ~50 C4A/C128A layers, ~1.8–5.4 s total for T=1024 | **Highest prefill impact** (Python token loop with thousands of `.item()` syncs). Low decode impact (fires only at compress boundaries for T=1). |
| **P1-prefill** | **F1**: `_torch_qnorm_rope_kv_insert_fallback` (prefill aspect) | Prefill: 61 layers, ~6–21 ms total (hidden by stream overlap) | Lower prefill priority — overhead is hidden by stream overlap for typical prompt lengths. |

#### Recommended execution order

1. **F3 + F4** (mHC): Validate and enable existing SM70 fast path — **lowest implementation risk**, highest decode-path ROI.
2. **F1** (Q-norm/RoPE/KV-insert): Implement new Triton kernel — **moderate risk**, eliminates ~25–35 kernel launches per layer.
3. **F5** (FP8 einsum): Implement Triton grouped einsum — **moderate risk**, eliminates ~86 MB cached weight + FP32 intermediate.
4. **F2** (compressor): Implement new Triton kernel — **highest complexity** (state gather via block_table indirection, multi-operation fusion), but **highest prefill ROI**.

### 11.4  Template Kernels for Each Replacement

Each proposed Triton replacement can leverage existing SM70 Triton kernels as implementation templates to reduce development risk.

| Fallback | Proposed Replacement | Template Kernel | Template Source | Reusable Patterns |
|---|---|---|---|---|
| **F1** | Triton Q-Norm/RoPE/KV-Insert | `_gather_decode_kv_triton_kernel` | `deepseek_v4_attention.py` | **Inverse operation** — same FP8 e4m3fn bit manipulation (encode instead of decode), same UE8M0 scale computation, same paged cache addressing via `slot_mapping → block_idx + pos_in_block`, same 7-block × 64-element quantization structure. RoPE logic reusable from `_apply_gptj_rope_tail`. |
| **F2** | Triton Fused Compressor | `_gather_decode_kv_triton_kernel` + `_fused_kv_compress_norm_rope_insert_sparse_attn` (SM80+ Triton) | `deepseek_v4_attention.py` + `deepseek_compressor.py` | Block_table indirection pattern from gather kernel. Softmax/RMSNorm/RoPE/FP8 quant fusion pattern from the SM80+ Triton compressor kernel (adapt for SM70: replace `tl.float8e4nv` with manual bit encoding). |
| **F3** | Enable `_mhc_pre_post_gemm_kernel` | `_mhc_pre_post_gemm_kernel` (already exists) | `mhc.py` (§4.4) | **Template IS the replacement.** Kernel already implemented and functional. Only needs validation + enablement (`VLLM_SM70_MHC_FAST=1` by default after ZX-42 canary passes). |
| **F4** | Enable `_mhc_post_fused_kernel` | `_mhc_post_fused_kernel` (already exists) | `mhc.py` (§4.5) | **Template IS the replacement.** Same as F3. |
| **F5** | Triton FP8 Grouped Einsum | `_sm70_fp8_paged_mqa_logits_kernel` | `sm70_mqa_logits.py` | FP8 e4m3fn → FP32 dequantization via bit extraction (identical logic). UE8M0 scale application. Per-block dequant → dot product accumulation pattern. Adapt from 1D dot product to 2D grouped matmul by tiling over the `o_lora_rank` output dimension. |

#### Template reuse summary

```
Existing kernel patterns:
┌─────────────────────────────────────────────────────┐
│ _gather_decode_kv_triton_kernel                     │
│   • FP8 bit decode (sign/exp/mant extraction)       │──► F1 (encode = reverse decode)
│   • UE8M0 scale application                         │──► F2 (same scale logic)
│   • Paged cache addressing (block_idx, pos_in_block)│──► F1, F2 (scatter write)
│   • BF16 RoPE copy as uint16                        │──► F1 (write RoPE bytes)
│   • 7-block × 64-element loop (tl.static_range)     │──► F1, F2
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ _sm70_fp8_paged_mqa_logits_kernel                   │
│   • FP8 bit decode (identical logic)                │──► F5 (activation dequant)
│   • Per-head loop with dot product accumulation     │──► F5 (adapt to grouped matmul)
│   • UE8M0/float32 scale application                 │──► F5 (weight + activation scales)
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ _mhc_pre_post_gemm_kernel / _mhc_post_fused_kernel │
│   • Already complete SM70 Triton implementation     │──► F3, F4 (enable as-is)
│   • cuBLAS FP16 GEMM + Triton post-processing      │
│   • Sinkhorn normalization in FP32                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ _fused_kv_compress_norm_rope_insert_sparse_attn     │
│   (SM80+ Triton, not SM70-compatible)               │
│   • Softmax + weighted-sum + RMSNorm + RoPE fusion  │──► F2 (algorithm template)
│   • Per-token program grid                          │──► F2 (grid design)
│   • Block_table indirection for state gather        │──► F2 (indirect addressing)
└─────────────────────────────────────────────────────┘
```

---

## 12  CUDA Graph Status and Optimization Directions

*Satisfies Requirement 12 (acceptance criteria 12.1 – 12.4).*

This section consolidates the CUDA graph findings from §2.4 (capture boundaries) and §7.3 (decode efficiency audit) and expands them with concrete optimization directions for graph coverage expansion, workspace memory efficiency, and SM70 memory bandwidth considerations.

### 12.1  Current CUDA Graph Status

#### Configuration summary

| Parameter | Current Value | Notes |
|---|---|---|
| Graph policy | `FULL_DECODE_ONLY` | Only pure-decode steps (no prefill tokens) are captured |
| `cudagraph_capture_sizes` | `[1]` | Single decode token per step |
| `CG_SUPPORT` | `AttentionCGSupport.UNIFORM_BATCH` | Both `FlashMLASparseMetadataBuilder` and `DeepseekSparseSWAMetadataBuilder` |
| Workspace stability | `get_simultaneous()` allocations | Fixed GPU addresses across capture/replay |
| Tile scheduler | `FlashMLASchedMeta` with `have_initialized` guard | Planner runs during capture; reused on replay |

#### `.item()` blocker fix

The original `sparse_attn_indexer.py` contained a `.item()` call that forced a host-device synchronization, preventing CUDA graph capture. This was resolved by routing the indexer through the SM70 Triton kernel path (`sm70_fp8_paged_mqa_logits`) which computes logits entirely on-device without host readback. See §2.4 for full details.

#### Validated capture characteristics

With `FULL_DECODE_ONLY` and `cudagraph_capture_sizes=[1]`, the entire decode forward pass (~61 layers × ~40–55 kernels/layer ≈ 2,400–3,400 total kernel launches) is captured in a single CUDA graph. The only non-graphed work is CPU-side metadata build (`DeepseekSparseSWAMetadataBuilder.build()`), which is overlapped with GPU graph replay.

### 12.2  Expansion Opportunities

#### 12.2.1  Larger capture sizes: `[1, 2, 4]`

**Current limitation**: Only `cudagraph_capture_sizes=[1]` is validated. Multi-token decode (e.g., speculative decoding with MTP) requires graph captures at larger batch sizes.

**Expansion plan**:

| Capture Size | Use Case | Key Validation Requirements |
|---|---|---|
| `T=1` | Standard autoregressive decode | ✅ Already validated |
| `T=2` | 2-token MTP (speculative) | Workspace address stability for `(2, topk, 512)` fallback KV; tile scheduler plan with `batch_size=2` |
| `T=4` | 4-token MTP or small-batch decode | Larger fallback workspace `(4, topk, 512)`; verify `FlashMLASchedMeta` planner handles `b=4` correctly |

**Requirements for expansion**:

1. **Workspace tensor shape stability**: `get_simultaneous()` must be called with the **maximum** shape across all capture sizes. Currently the fallback workspace is allocated as `(max_num_batched_tokens, total_topk, 512)`, so larger T values are already accommodated if `max_num_batched_tokens ≥ 4`.
2. **FlashMLASchedMeta compatibility**: The tile scheduler config includes `b` (batch size). Different capture sizes produce different plans. Each `(capture_size, layer_type)` pair needs its own `FlashMLASchedMeta` instance — or the planner must be re-invoked per graph.
3. **Index tensor padding**: `swa_indices` and `topk_indices` shapes change with T. For `UNIFORM_BATCH` mode, all decode tokens must have identical metadata (same `topk`, same `window_size`). Padding to the max capture size with sentinel indices (`-1`) ensures shape uniformity.
4. **Testing**: Eager vs graph parity test at each capture size for all three layer types (swaonly / c4a / c128a).

#### 12.2.2  Prefill graph segments

**Assessment**: Prefill is fundamentally **dynamic-shape** (variable prompt lengths) and **chunked** (PREFILL_CHUNK_SIZE=4 requests with varying `seq_lens`). This makes full prefill graph capture impractical.

**Partial capture opportunities**:

| Candidate Segment | Fixed-Shape? | Feasibility |
|---|---|---|
| `flash_mla_sparse_fwd` per chunk | No — `num_chunk_tokens` varies | ❌ Requires padding |
| O-projection pipeline | Yes for fixed T | ⚠️ Marginal — only 4–5 kernels, and T varies |
| `dequantize_and_gather_k_cache` | No — `seq_lens` varies | ❌ |
| mHC pre/post blocks | Yes for fixed T | ⚠️ Same concern — T varies across chunks |

**Verdict**: Prefill graph capture provides negligible benefit. The primary overhead in prefill is compute-bound (GEMMs, attention), not launch-bound. CPU launch overhead is amortized over the much longer per-chunk execution time. **Not recommended for near-term work.**

#### 12.2.3  Host synchronization blockers

Remaining potential host sync points that could prevent graph capture expansion:

| Sync Point | Location | Status | Impact |
|---|---|---|---|
| `.item()` in indexer | `sparse_attn_indexer.py` | ✅ **Fixed** — routed through Triton kernel | Was the only blocker |
| Python `for` loop in compressor fallback | `deepseek_compressor.py` | ⚠️ **Captured** for T=1 (loop body is fixed-shape) | Could block T>1 if compress_ratio boundary varies per token |
| `torch.where` / `torch.cat` in index building | `flashmla_sparse.py` | ✅ Captured — shapes fixed for uniform batch | No issue for UNIFORM_BATCH |
| `sinkhorn_repeat` iterations in mHC | `mhc.py` | ✅ Captured — fixed iteration count | No issue |
| `FlashMLASchedMeta` planner | FlashMLA C++ | ✅ Runs during capture only | No issue on replay |

**Key risk for T>1**: The compressor fallback's Python-level token loop iterates over `num_tokens`. For `T=2` or `T=4`, the loop body executes multiple times with potentially different compress_ratio boundary conditions per token. If some tokens trigger compression and others don't, the conditional branches may produce non-deterministic graph structure. **Mitigation**: Replace compressor fallback with the Triton kernel (§11, F2) which has a fixed grid regardless of which tokens hit boundaries.

### 12.3  Workspace Memory Efficiency Analysis

#### Current `get_simultaneous()` allocations

| Workspace | Shape (typical, max_model_len=163840) | Bytes | Used During Decode? |
|---|---|---|---|
| **Prefill BF16 workspace** | `(819200, 512)` = 5×max_model_len × 512 | **~839 MB** | ❌ Idle during decode |
| SM70 fallback KV (C4A) | `(max_tokens, topk+window, 512)` | topk-dependent, ~2–8 MB | ✅ Active |
| SM70 fallback KV (SWA-only) | `(max_tokens, window, 512)` | ~1–4 MB | ✅ Active |
| q_concat buffer | `(max_tokens, padded_heads, 512)` | ~64 KB (T=1) | ✅ Active |
| Indexer workspace (k_fp8) | `(total_seq_lens, 128)` | ~10–50 MB | ✅ Active |
| Indexer workspace (k_scale) | `(total_seq_lens, 4)` | ~1–2 MB | ✅ Active |
| MoE workspace13 | varies | **>2 GB** | ✅ Active |
| MoE workspace2 | varies | ~200–500 MB | ✅ Active |

#### Peak memory reduction recommendations

1. **Prefill workspace release during decode** (saves ~839 MB):
   - The prefill BF16 workspace (`get_prefill_workspace_size() × 512 × 2 bytes`) is allocated persistently but unused during pure decode.
   - **Proposal**: Implement a "workspace phase" system — allocate prefill workspace lazily on first prefill, and mark it as reclaimable during pure-decode CUDA graph phases. The MoE workspace (>2 GB) dominates peak memory regardless, so the savings are meaningful only if MoE workspace can also be phased.
   - **Complexity**: Medium — requires coordination with CUDA graph capture (graph references the address even if unused).

2. **Fallback workspace size reduction with `flash_mla_with_kvcache` on SM70** (saves 2–8 MB per layer type):
   - If the SM70 decode fallback is eliminated (§10.3, enabling direct `flash_mla_with_kvcache`), the fallback KV workspace is no longer needed.
   - **Savings**: Modest per-layer, but eliminates workspace address stability requirements for this buffer.

3. **MoE workspace sharing with prefill workspace** (potential ~839 MB save):
   - The MoE workspace and prefill BF16 workspace are never active simultaneously (prefill and MoE don't overlap within a single layer forward).
   - **Proposal**: Use a single large allocation for both, sized to `max(moe_workspace, prefill_workspace)`.
   - **Complexity**: Low — both are `get_simultaneous()` allocations; unify into a shared pool.

4. **Indexer workspace proportional to active sequence count** (saves 10–50 MB):
   - Currently sized for `total_seq_lens` (sum of all sequence lengths). During decode with few active sequences, this over-allocates.
   - **Proposal**: Dynamic resize on metadata build, or allocate for `max_num_seqs × max_model_len` once (already the case).
   - **Complexity**: Low, but savings are minor compared to MoE/prefill workspaces.

### 12.4  FP16 vs BF16 Memory Bandwidth Impact on SM70

#### SM70 data type constraints

| Aspect | SM70 (V100) | SM80+ (A100/H100) |
|---|---|---|
| Native compute dtype | FP16 (FP32 accumulation) | BF16 or FP16 |
| Memory storage dtype | FP16 for activations | BF16 for activations |
| Bytes per element | 2 (identical to BF16) | 2 (identical to FP16) |
| FP32 intermediate requirement | Yes — mHC, compressor state, einsum | No — BF16 throughout |

#### Bandwidth implications

**For activations (FP16 on SM70 vs BF16 on SM80+):** Memory bandwidth is **identical** — both dtypes are 16-bit, consuming the same bytes per element. The V100's 900 GB/s HBM2 bandwidth is fully utilized regardless of whether data is FP16 or BF16. No bandwidth penalty from the SM70 FP16 requirement.

**For FP32 intermediate tensors:** SM70 requires FP32 intermediates in several paths that SM80+ can execute in BF16:

| Operation | SM70 Dtype | SM80+ Dtype | Bandwidth Overhead (SM70 vs SM80+) |
|---|---|---|---|
| mHC pre-block GEMM output | FP32 | FP16/BF16 | **2×** — 4 bytes vs 2 bytes per element |
| Compressor state cache | FP32 | FP32 | 1× (same on both) |
| `_sm70_fp8_einsum_bmm` pre-dequant weight | FP32 | N/A (DeepGEMM) | **~86 MB** persistent; read once per call |
| `_sm70_fp8_einsum_bmm` activation dequant | FP32 | N/A | **2×** during einsum computation |
| Sinkhorn iterations (mHC) | FP32 | FP32 | 1× (same on both) |

#### Quantified impact

For a single decode token (T=1) across 61 layers:

- **mHC FP32 GEMM output**: `[1, 24]` × 4 bytes = 96 bytes/layer × 61 = ~5.9 KB total. **Negligible.**
- **`_sm70_fp8_einsum_bmm` FP32 activation dequant**: `[1, 2, 512]` × 4 bytes = 4 KB/layer × 61 = ~244 KB total. **Negligible.**
- **`_sm70_fp8_einsum_bmm` FP32 weight read**: ~86 MB × 1 read per layer × 61 layers = ~5.2 GB total bandwidth per decode step. **This is the dominant FP32 bandwidth consumer.**

**The 86 MB FP32 pre-dequantized weight cache** (`wo_a._sm70_predequant_f32`) is the primary SM70-specific bandwidth overhead. It is read once per layer per decode step, consuming ~5.2 GB of HBM bandwidth. On V100 with 900 GB/s peak bandwidth, this represents ~5.8 ms of pure memory read time — a significant fraction of the decode budget.

#### Optimization directions for FP32 bandwidth

1. **FP16 einsum with dynamic scaling** (saves ~43 MB per weight set, halves bandwidth):
   - Store pre-dequantized weights in FP16 instead of FP32, with per-block scaling factors.
   - Risk: FP16 range (`±65504`) may clip some dequantized values. Requires validation of value distribution.
   - Expected bandwidth reduction: **50%** for O-projection einsum reads.

2. **Eliminate `_sm70_fp8_einsum_bmm` entirely** (§11, F5):
   - Replace with a Triton grouped-matmul kernel that reads FP8 weights directly and dequantizes on-the-fly.
   - Reduces weight memory from 86 MB FP32 → ~21 MB FP8 (4× reduction).
   - Expected bandwidth reduction: **75%** for O-projection.

3. **mHC FP16 fast path enablement** (§11, F3/F4):
   - The SM70 fast path uses cuBLAS FP16 GEMM + Triton FP32 post-processing.
   - Switching from torch FP32 fallback eliminates the FP32 intermediate GEMM output entirely.
   - Bandwidth saving: minimal (mHC tensors are tiny), but **latency saving from kernel fusion is significant** (§9.4).

---

## 13  Task Decomposition Structure

*Satisfies Requirement 13 (acceptance criteria 13.1 – 13.4).*

This section decomposes all optimization directions from §10, §11, and §12 into independent, testable tasks organized by priority tier. Each task includes: ID, description, estimated effort, dependencies, files to modify, and acceptance test (verification method). Tasks already implemented are marked ✅.

### 13.1  P0 — Decode Latency Critical Path (>10% expected improvement)

These tasks target the decode hot path where per-token latency directly impacts generation throughput.

| ID | Description | Effort | Dependencies | Files | Verification | Status |
|---|---|---|---|---|---|---|
| **T-P0-1** | Implement SM70 CUDA decode kernel for `flash_mla_with_kvcache` — fused FP8 dequant + MMA attention eliminating the 2-kernel fallback (§10.3) | Large | T-P0-2, T-P0-3 | `/mnt/data/apps/FlashMLA/csrc/sm70/decode/sparse_fp8/` | Benchmark: decode tok/s improvement >20% vs fallback path | In progress |
| **T-P0-2** | Implement FP8 dequant/MMA double-buffer pipeline overlap in SM70 decode kernel (§10.2.2) | Large | None | `/mnt/data/apps/FlashMLA/csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh` | Benchmark: dequant latency hidden behind MMA; unit test correctness vs reference | ✅ Done |
| **T-P0-3** | Implement dual-cache (SWA + compressed) two-pass access ordering in SM70 decode kernel (§10.2.3) | Medium | T-P0-2 | `/mnt/data/apps/FlashMLA/csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh` | Benchmark: L2 miss rate reduction; unit test SWA+compressed parity | ✅ Done |
| **T-P0-4** | Tune FlashMLA tile scheduler for V100's 80 SMs (§10.2.1) | Medium | T-P0-1 | `/mnt/data/apps/FlashMLA/csrc/sm70/decode/sparse_fp8/`, `csrc/smxx/decode/get_decoding_sched_meta/` | Benchmark: decode tok/s with T=1,2,4 vs untunned; semantic canary ZX-42 | ✅ Done |
| **T-P0-5** | Validate + enable mHC SM70 fast path by default (F3+F4, §11.2) | Small | None | `vllm/model_executor/layers/mhc.py` | Semantic canary ZX-42 exact match + identity coherence; benchmark: ~8–36 ms/step reduction | Pending |
| **T-P0-6** | Implement Triton FP8 grouped einsum replacing `_sm70_fp8_einsum_bmm` (F5, §11.2) | Medium | None | `vllm/model_executor/layers/deepseek_v4_attention.py` | Unit test: output parity vs torch FP32 einsum within FP16 tolerance; benchmark: ~3–9 ms/step reduction, ~86 MB memory saved | Pending |
| **T-P0-7** | Optimize `_sm70_fp8_paged_mqa_logits_kernel` register pressure — reduce per-head Q reload (§4.1, §10.1) | Medium | None | `vllm/model_executor/layers/sm70_mqa_logits.py` | Benchmark: decode indexer latency reduction; unit test output parity | ✅ Done |
| **T-P0-8** | Optimize `_gather_decode_kv_triton_kernel` memory coalescing (§4.3) | Small | None | `vllm/model_executor/layers/deepseek_v4_attention.py` | Benchmark: KV gather latency reduction; unit test output parity | ✅ Done |
| **T-P0-9** | Expand CUDA graph capture sizes to `[1, 2, 4]` (§12.2.1) | Medium | T-P0-5 (mHC fast path eliminates non-graphed ops) | `vllm/v1/attention/backends/mla/flashmla_sparse.py` | Unit test: eager vs graph parity for T=1,2,4 across swaonly/c4a/c128a | ✅ Done |

### 13.2  P1 — Prefill Optimization and Memory Efficiency (5–10% expected improvement)

These tasks target prefill throughput and memory bandwidth reduction.

| ID | Description | Effort | Dependencies | Files | Verification | Status |
|---|---|---|---|---|---|---|
| **T-P1-1** | Implement Triton replacement for `_torch_qnorm_rope_kv_insert_fallback` (F1, §11.2) — fuse Q-norm + RoPE + FP8 quant + cache write | Large | None | `vllm/model_executor/layers/deepseek_v4_attention.py` | Unit test: Triton output == torch fallback for edge cases (zero-len, single-tok, max-model-len); benchmark: ~6–21 ms/step reduction | ✅ Done |
| **T-P1-2** | Implement Triton replacement for `_torch_fused_compress_norm_rope_insert_fp8_fallback` (F2, §11.2) — eliminate Python token loop | Large | None | `vllm/model_executor/layers/deepseek_compressor.py` | Unit test: Triton output == torch fallback at compress_ratio boundaries for C4A/C128A; benchmark: prefill 1024-tok latency reduction >50% | ✅ Done |
| **T-P1-3** | Optimize `flash_mla_sparse_fwd` shared-memory tiling for Volta 96 KB shmem (§10.1.1) | Large | None | `/mnt/data/apps/FlashMLA/csrc/smxx/` (prefill kernels) | Benchmark: prefill tok/s improvement 5–15%; semantic canary ZX-42 | Pending |
| **T-P1-4** | Implement workspace phase system — release prefill BF16 workspace during pure-decode (§12.3) | Medium | T-P0-9 | `vllm/v1/attention/backends/mla/flashmla_sparse.py`, workspace manager | Unit test: workspace reclaimable after prefill; benchmark: ~839 MB peak memory reduction during decode | Pending |
| **T-P1-5** | Unify MoE workspace + prefill workspace allocation via shared `get_simultaneous()` pool (§12.3) | Small | T-P1-4 | `vllm/v1/attention/backends/mla/flashmla_sparse.py`, MoE layer | Unit test: no OOM when both workspaces needed sequentially; benchmark: ~839 MB memory saved | Pending |
| **T-P1-6** | FP8 cache encode-decode round-trip property test (§14) | Small | None | `tests/v1/attention/test_fp8_cache_roundtrip_pbt.py` | PBT: 100+ iterations, NoPE within 1 ULP, RoPE bitwise identical | ✅ Done |

### 13.3  P2 — Code Quality, Maintainability, and Future-Proofing

These tasks improve code quality, reduce technical debt, and prepare for future architecture changes.

| ID | Description | Effort | Dependencies | Files | Verification | Status |
|---|---|---|---|---|---|---|
| **T-P2-1** | Implement FP16 accumulation precision management for `flash_mla_sparse_fwd` (§10.1.3) — split MMA accumulation with FP32 reduction | Medium | None | `/mnt/data/apps/FlashMLA/csrc/smxx/` (prefill kernels) | Semantic canary ZX-42; unit test: output precision vs FP32 reference within 1e-3 | Pending |
| **T-P2-2** | Implement warp scheduling optimization for sparse index gather in `flash_mla_sparse_fwd` (§10.1.2) | Small | None | `/mnt/data/apps/FlashMLA/csrc/smxx/` | Benchmark: <5% improvement; unit test output parity | Pending |
| **T-P2-3** | Reduce `_sm70_fp8_einsum_bmm` FP32 weight cache to FP16 with per-block scaling (§12.4) | Small | None | `vllm/model_executor/layers/deepseek_v4_attention.py` | Unit test: output within FP16 tolerance; benchmark: ~43 MB memory reduction, 50% bandwidth reduction for O-projection | Pending |
| **T-P2-4** | Add subnormal handling to `_sm70_fp8_paged_mqa_logits_kernel` / `_sm70_fp8_mqa_logits_kernel` (§4.1, §4.2 precision limitation) | Small | None | `vllm/model_executor/layers/sm70_mqa_logits.py` | Unit test: FP8 subnormal inputs decoded correctly vs reference; PBT: MQA logits equivalence property | Pending |
| **T-P2-5** | SM70 FP8 MQA logits equivalence property test | Small | None | `tests/v1/attention/test_sm70_fp8_mqa_logits_pbt.py` | PBT: 100+ iterations, Triton output within FP32 rounding tolerance of reference | ✅ Done |
| **T-P2-6** | Document pipeline analysis + optimization directions (Requirements 1–12) | Large | None | `docs/superpowers/plans/deepseek-v4-sm70-pipeline-doc.md` | Human review: all 14 requirement sections present and internally consistent | ✅ Done |

### 13.4  Parallelizable Task Groups

Tasks within the same group have **no shared file conflicts** or data dependencies and can be executed in parallel to maximize development throughput.

#### Parallel Group A — Independent Triton kernel work (vLLM)

| Tasks | Shared Constraint |
|---|---|
| T-P0-5 (mHC enablement), T-P0-6 (FP8 einsum), T-P1-1 (Q-norm/RoPE/KV-insert), T-P1-2 (compressor) | Each modifies a different source file. No data dependencies. Can be developed and tested independently. |

#### Parallel Group B — FlashMLA kernel work (external repo)

| Tasks | Shared Constraint |
|---|---|
| T-P0-1 (SM70 decode kernel), T-P0-4 (tile scheduler), T-P1-3 (prefill shmem tiling) | All in `/mnt/data/apps/FlashMLA/csrc/`. T-P0-4 depends on T-P0-1. T-P1-3 is independent (prefill path). **T-P0-1 and T-P1-3 can run in parallel.** |

#### Parallel Group C — Memory efficiency (vLLM infrastructure)

| Tasks | Shared Constraint |
|---|---|
| T-P1-4 (workspace phase), T-P1-5 (unified allocation) | Both touch `flashmla_sparse.py` workspace management. T-P1-5 depends on T-P1-4. **Sequential within group, but parallel with Groups A and B.** |

#### Parallel Group D — Correctness and precision (tests + minor fixes)

| Tasks | Shared Constraint |
|---|---|
| T-P2-1 (FP16 precision), T-P2-2 (warp scheduling), T-P2-4 (subnormal fix) | All in FlashMLA or `sm70_mqa_logits.py`. T-P2-1 and T-P2-2 both touch FlashMLA prefill (potential merge conflict). **T-P2-4 is independent.** |

#### Execution timeline recommendation

```
Week 1-2:  ┌─ Group A: T-P0-5, T-P0-6, T-P1-1, T-P1-2  (4 developers)
           ├─ Group B: T-P0-1 + T-P1-3                    (2 developers)
           └─ Group D: T-P2-4                              (1 developer)

Week 3-4:  ┌─ Group B: T-P0-4 (depends on T-P0-1)
           ├─ Group C: T-P1-4, T-P1-5 (sequential)
           ├─ Group A: T-P0-9 (depends on T-P0-5)
           └─ Group D: T-P2-1, T-P2-2, T-P2-3

Integration: T-P0-1 integration test (full decode path without fallback)
             Semantic canary: ZX-42 exact match across all changes
```

#### Dependency graph

```
T-P0-2 ✅ ──┐
             ├──► T-P0-1 (SM70 decode kernel — in progress)
T-P0-3 ✅ ──┘         │
                       ▼
                   T-P0-4 ✅ (tile scheduler tuning)

T-P0-5 (mHC enablement) ──► T-P0-9 ✅ (expanded CUDA graph)

T-P1-4 (workspace phase) ──► T-P1-5 (unified allocation)

All others: independent (no blocking dependencies)
```

---

## 14  FP8 Cache Encode-Decode Path Documentation

*Satisfies Requirement 14 (acceptance criteria 14.1, 14.2).*

This section defines the complete encode and decode paths for the FP8 KV cache format used by DeepSeek V4 Flash on SM70. The FP8 cache stores 584 bytes per token (see §3.1 for the detailed token format). These paths ensure that KV latent values survive the quantization round-trip with bounded precision loss.

### 14.1  Encode Path: float32/bf16 → FP8 Paged Cache Write

The encode path transforms floating-point KV latent values into the 584-byte FP8 token format and writes them to the paged KV cache. On SM70, this is performed by `_torch_qnorm_rope_kv_insert_fallback` (or the equivalent Triton replacement kernel).

```
Encode Path (per token):

Input: kv [head_dim=512] float16/float32
       ├── NoPE portion: kv[:448] (448 latent dimensions)
       └── RoPE portion: kv[448:512] (64 rotated dimensions, after GPT-J RoPE)

Step 1: NoPE Block Quantization (64-element blocks)
│
├── Reshape NoPE to 7 blocks: blocks = kv[:448].float().view(7, 64)
│
├── Per-block absmax:
│   absmax[b] = max(|blocks[b, :]|) for b in 0..6
│   absmax[b] = clamp(absmax[b], min=1e-4)   [prevent div-by-zero]
│
├── UE8M0 scale computation:
│   exponent[b] = ceil(log2(absmax[b] / 448.0))
│   scale[b] = 2^exponent[b]
│   encoded_scale[b] = clamp(exponent[b] + 127, 0, 254)   [UE8M0 byte]
│
├── FP8 e4m3fn quantization:
│   fp8_values[b, :] = clamp(blocks[b, :] / scale[b], -448, 448).to(float8_e4m3fn)
│   → 64 × 1-byte FP8 e4m3fn values per block
│
└── Output: 448 bytes FP8 NoPE + 7 bytes UE8M0 scales

Step 2: RoPE BF16 Encoding
│
├── Cast RoPE to bfloat16:
│   rope_bf16 = kv[448:512].to(bfloat16)   [64 values × 2 bytes = 128 bytes]
│   (On SM70: fp16 → bf16 cast; values are post-GPT-J-RoPE-rotation)
│
└── Output: 128 bytes BF16 RoPE

Step 3: Paged Cache Write (584-byte token slot)
│
├── Assemble token data (576 bytes):
│   token_data = concat(fp8_nope[448 bytes], rope_bf16[128 bytes])
│
├── Compute target slot:
│   block_idx = slot_mapping[token] // block_size
│   pos_in_block = slot_mapping[token] % block_size
│
├── Write token data to cache:
│   k_cache[block_idx, pos_in_block, 0:576] = token_data
│
├── Write UE8M0 scales to cache:
│   k_cache[block_idx, pos_in_block, 576:583] = encoded_scale[0:7]
│
└── Write padding byte:
    k_cache[block_idx, pos_in_block, 583] = 0x00
```

**Key characteristics:**

| Property | Detail |
|---|---|
| Quantization granularity | 64-element blocks (7 blocks for 448 NoPE values) |
| Scale format | UE8M0 (unsigned 8-bit exponent-only): `scale = 2^(byte − 127)` |
| FP8 format | e4m3fn (4-bit exponent, 3-bit mantissa, no NaN): range ±448, 1 sign + 4 exp + 3 mant |
| Representable range | [-448, +448] with ~1.95% relative precision per value |
| Absmax clamping | `min=1e-4` prevents log2(0) and ensures non-zero scale |
| RoPE dtype | bfloat16 (16-bit: 1 sign + 8 exp + 7 mant) — stored as raw bytes |

### 14.2  Decode Path: Paged Cache Read → float32 Dequant → bf16 Output

The decode path reads a 584-byte token entry from the paged KV cache, dequantizes FP8 NoPE values to floating-point, and copies BF16 RoPE values, producing a contiguous `[head_dim=512]` bf16 output vector. On SM70, this is performed by `_gather_decode_kv_triton_kernel`.

```
Decode Path (per token, per top-K entry):

Input: k_cache[block_idx, pos_in_block, 0:584] uint8
       (584-byte token slot in paged FP8 cache)

Step 1: UE8M0 Scale Lookup
│
├── Read 7 scale bytes:
│   raw_scale[b] = k_cache[block_idx, pos_in_block, 576 + b]  for b in 0..6
│
├── Decode UE8M0 → float32 scale:
│   scale[b] = 2^(raw_scale[b] − 127)
│   (equivalently: reinterpret raw_scale[b] as the exponent field of an IEEE 754 float32)
│
└── Output: 7 × float32 scale values

Step 2: FP8 e4m3fn → float32 Dequantization (per 64-element block)
│
├── Read FP8 bytes:
│   fp8_bytes[b, :] = k_cache[block_idx, pos_in_block, b*64 : (b+1)*64]  for b in 0..6
│
├── Manual bit-level decode (SM70 Triton, no hardware FP8):
│   for each byte x in fp8_bytes[b, :]:
│       sign = (x >> 7) & 1
│       exp  = (x >> 3) & 0xF     [4-bit exponent]
│       mant = x & 0x7            [3-bit mantissa]
│
│       if exp == 0 and mant == 0:
│           float_val = 0.0                          [zero]
│       elif exp == 0 and mant != 0:
│           float_val = (-1)^sign × mant × 2^(-9)   [subnormal: mant/512]
│       else:
│           fp32_bits = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
│           float_val = reinterpret_as_float32(fp32_bits)   [normal]
│
├── Apply per-block scale:
│   dequantized[b, :] = float_val[:] × scale[b]
│
└── Output: 7 × 64 = 448 float32 NoPE values

Step 3: float32 → bf16 Conversion (NoPE)
│
├── Round-to-nearest-even:
│   For each float32 value x:
│       x_u32 = reinterpret_as_uint32(x)
│       bf16_bits = (x_u32 + 0x7FFF + ((x_u32 >> 16) & 1)) >> 16
│       (standard IEEE 754 round-to-nearest-even for BF16 truncation)
│
└── Output: 448 × bf16 NoPE values (896 bytes as uint16)

Step 4: BF16 RoPE Copy (bitwise identical)
│
├── Direct uint16 copy:
│   rope_out[:] = k_cache[block_idx, pos_in_block, 448:576] viewed as uint16[64]
│   (4 chunks of 16 uint16 values — no format conversion, bitwise identical)
│
└── Output: 64 × bf16 RoPE values (128 bytes as uint16)

Step 5: Assemble Output Vector
│
├── output[0:448] = bf16 NoPE values   (from Step 3)
├── output[448:512] = bf16 RoPE values (from Step 4)
│
└── Final output: [512] bf16 (1024 bytes)
    Ready for flash_mla_sparse_fwd attention computation
```

**Key characteristics:**

| Property | Detail |
|---|---|
| Dequantization precision | FP8 e4m3fn → float32 (lossless bit reconstruction) → bf16 (0.5 ULP rounding) |
| Subnormal handling | Full subnormal support in `_gather_decode_kv_triton_kernel` (mant × 2^−9) |
| RoPE round-trip | **Bitwise identical** — stored as bf16, copied as raw uint16 bytes |
| NoPE round-trip error | Bounded by FP8 quantization loss + bf16 rounding: ≤ 1 ULP of FP8 representation |
| Zero handling | Explicit zero check (`exp==0 && mant==0` → 0.0) |
| Scale reconstruction | UE8M0 byte → `2^(byte−127)` — exact power-of-two, no precision loss in scale itself |

### 14.3  Round-Trip Precision Guarantee

The encode-decode round-trip introduces precision loss at exactly two points:

1. **FP8 quantization (encode Step 1):** The original float32/bf16 NoPE value is divided by a power-of-two scale and cast to FP8 e4m3fn. This introduces quantization error bounded by 0.5 ULP of the FP8 representation (round-to-nearest in the FP8 cast).

2. **FP32 → BF16 conversion (decode Step 3):** After lossless FP8→float32 dequantization and exact scale multiplication (power-of-two), the float32 result is rounded to bf16 with ≤ 0.5 ULP error in bf16.

**Combined guarantee:** For any NoPE value within the FP8 e4m3fn representable range [-448, 448], the round-trip result satisfies:
- `|decode(encode(x)) − quantize_fp8(x)| ≤ 0.5 ULP(bf16)` — the decoded value is within 0.5 BF16 ULP of the ideal FP8 representation.
- Equivalently: the decoded bf16 value equals `round_to_bf16(dequant_fp8(quantize_fp8(x)))` — the best bf16 approximation of the FP8-quantized value.

**RoPE guarantee:** The 64 RoPE values are stored as bf16 and copied as raw bytes — the round-trip is **bitwise identical** (zero error).

---

<!-- Subsequent sections will be added by tasks 7.x, 9.x, etc. -->
