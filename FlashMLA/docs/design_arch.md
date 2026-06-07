
## Architecture

### System overview (prefill execution path on SM70)

```
hidden_states
  -> DeepseekV4MLAAttention.forward
       -> compressed_gather / swa_gather  (prefill only)
       -> wrapper.fused_wqa_wkv             (FP8 quant + QKV projection)
       -> impl.q_kv_rmsnorm
       -> decoder.mhc_pre_attn              (MHC pre-attention sentinel)
       -> wrapper.attention_total
            -> torch.ops.vllm.deepseek_v4_attention (custom op, inductor-shielded)
                 -> impl.mla_attn_total
                      -> prefill.flashmla_sparse_fwd   <-- 94.1% of attention
       -> wrapper.o_inv_rope_fp8_quant     (fused_inv_rope_fp8_quant custom op)
       -> wrapper.wo_b (= _sm70_fused_o_einsum_wo_b under VLLM_SM70_DEEPSEEK_V4_FUSE_O_WOB=1)
       -> TP all-reduce (unchanged)
  -> DeepseekV4MoE
       -> decoder.moe.experts
```

All **bold** custom-op boundaries above are already shielded from inductor
fusion via `direct_register_custom_op` (pattern in
`vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`).
The two paths that are NOT yet shielded and are candidate Round-1 fix
targets:

- `_sm70_fused_o_einsum_wo_b` -- called directly as a Python function (not
  through `torch.ops.vllm.*`) at
  `deepseek_v4_attention.py:~2368`. Its internal body does
  `sm70_fp8_a_dequant_to_fp16(a, a_scale)` followed by a Python-level
  `torch.einsum` / `flatten` / `wo_b(module)` chain. Inductor sees each
  intermediate as a regular torch tensor and can attempt to fuse them.
- `_sm70_fp8_einsum_bmm` fallback path -- called through
  `torch.ops.vllm.deepseek_v4_fp8_einsum` (already shielded) BUT the
  `b._sm70_predequant_f16` cached fp16 weight attribute is populated inside
  the Python function body on first call; the cache-miss branch does a
  `b.reshape(...).float() * b_scale_3d.repeat_interleave(128,dim=1).repeat_interleave(128,dim=2)`
  which triggers a `repeat_interleave`-fused kernel in inductor. On dynamo
  fullgraph capture the shape is not known at trace time and inductor
  generates a kernel typed `fp8e4nv` because `b` is `torch.float8_e4m3fn`.

The Round 1 design therefore targets the `_sm70_fused_o_einsum_wo_b` /
`_sm70_fp8_einsum_bmm` functions: move the fp8-touching tensor materialization
into a `torch.ops.vllm.sm70_fused_o_einsum_wo_b` custom op with a registered
fake impl, so inductor treats the whole op as opaque. This keeps inductor's
fusion enabled for every non-fp8 path in the program. Exact op-shape and
selection rule below.

### FlashMLA SM70 sparse-prefill kernel -- current vs. target grid

Current launch at `/mnt/data/apps/FlashMLA/csrc/sm70/prefill/sparse/fwd.cuh:1136`:

```cpp
kernel<<<dim3(params.s_q, params.h_q), dim3(NUM_THREADS), smem_size, params.stream>>>(params);
```

- `NUM_THREADS = 256` (from `config.h::FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS`)
- `K_TILE = 16` (post-K_TILE=16 rebuild; was 32 pre-rebuild; 96 KB shmem
  budget on V100 fits 20 KB per block = 4 blocks/SM occupancy)
- `grid = (s_q=2048, h_q=64)` on the 2048-token canonical prompt -- **one block
  per (query, head) pair**. 131,072 blocks across 80 SMs means the kernel runs
  as a fine-grained roundrobin where each block computes a single output query
  for a single head. The 256 threads per block reduce one query over `topk`
  KV tokens in tiles of `K_TILE` -- this is a good thread/reduction layout but
  a bad grid layout because it trades launch amortization for a simplified
  reduction.

Target launch (Round 2, modeled on FA-Bond-vLLM per
`/mnt/data/apps/FlashMLA/docs/三方 flash-attention-v100 深度对比分析.md`):

```cpp
// BLOCK_M = 32 queries per block, one head per block
kernel_bm32<<<dim3(ceil_div(s_q, 32), h_q), dim3(512), smem_size, stream>>>(params);
```

- `BLOCK_M = 32` queries batched into one block. Arithmetic intensity rises
  from O(1) Q-fetch-per-K-fetch to O(32) Q-fetch-per-K-fetch, because the same
  K-tile is now reused across 32 queries via shmem.
- Grid shrinks 32x: `(64 query-blocks, 64 heads) = 4096 blocks` -- still enough
  to saturate 80 SMs.
- Threads per block: 512 (matches FA-Bond-vLLM `16 warps x 32 threads` for
  hdim=128; SM70 can run 2 CTAs/SM at 512 threads provided shmem fits).
- Shmem target: <=48 KB per block -> 2 CTAs/SM (from 4 CTAs/SM at K_TILE=16).
  Occupancy drops 2x but arithmetic intensity rises 32x, so net throughput
  is predicted at ~8-16x the current kernel per the reference doc's 540%
  decode speedup number.

The S/P/O storage choice (following FA-Bond-vLLM reference design):

- S (QK scores, fp32): shmem, `sS[BLOCK_M, BLOCK_N]` stride `N_STRIDE + PAD`.
- P (softmax probs, fp16): shmem, union-overlaid with S via
  `P_STRIDE = N_STRIDE * 2` so P reads the same memory viewed as fp16.
- O (acc, fp32): shmem, `sO[BLOCK_M, D_V]` stride `D_STRIDE + PAD`.
- Online softmax: `THREADS_PER_ROW = 512 / 32 = 16` lanes per row reduction
  via `__shfl_down_sync` with `THREADS_PER_ROW` mask. This is intentionally
  different from the current kernel's single-warp reduction.
