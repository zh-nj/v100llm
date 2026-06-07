#pragma once

namespace sm70::prefill::sparse {

#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS
#define FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS 256
#endif

#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE
#define FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE 32
#endif

#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK
#define FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK 0
#endif

#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV
#define FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV 0
#endif

#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE
#define FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE 1
#endif

// Round 3 of the deepseek-v4-flash-prefill-throughput spec:
// batch HEADS_PER_BLOCK heads-of-the-same-query into one CUDA block
// so the KV tile is staged to shared memory ONCE per block and
// reused across all HEADS_PER_BLOCK heads. DeepSeek V4 MLA has
// h_kv=1 with h_q=64; all 64 heads share the same kv_head_idx and
// therefore the same top-K KV page set. HEADS_PER_BLOCK must divide
// h_q and must be <= q_heads_per_kv (= h_q / h_kv). V100 96 KB
// shmem budget allows HPB up to ~8 at K_TILE=16; HPB=4 gives 4x KV
// reuse at 3 CTAs/SM (from 4 CTAs/SM at HPB=1). Allowed: {1, 2, 4, 8}.
#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK
#define FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK 1
#endif

static constexpr int NUM_THREADS = FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS;
static constexpr int K_TILE = FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE;
static constexpr bool USE_MMA_884_QK = FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK != 0;
static constexpr bool USE_MMA_884_PV = FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV != 0;
static constexpr bool USE_MMA_884_ONLINE = FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE != 0;
static constexpr int HEAD_DIM_V = 512;
static constexpr int MAX_TOPK_FALLBACK = 8192;
static constexpr int HEADS_PER_BLOCK = FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK;
static_assert(HEADS_PER_BLOCK == 1 || HEADS_PER_BLOCK == 2 ||
              HEADS_PER_BLOCK == 4 || HEADS_PER_BLOCK == 8,
              "HEADS_PER_BLOCK must be in {1, 2, 4, 8}");

#ifndef FLASH_MLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER
#define FLASH_MLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER 0
#endif

static constexpr bool USE_DOUBLE_BUFFER = FLASH_MLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER != 0;

}  // namespace sm70::prefill::sparse
