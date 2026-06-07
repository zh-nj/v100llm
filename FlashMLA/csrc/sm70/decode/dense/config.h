#pragma once

namespace sm70::dense {

#ifndef FLASH_MLA_SM70_H_TILE
#define FLASH_MLA_SM70_H_TILE 4
#endif

#ifndef FLASH_MLA_SM70_CTA_THREADS
#define FLASH_MLA_SM70_CTA_THREADS 256
#endif

#ifndef FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE
#define FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE 0
#endif

#ifndef FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE
#define FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE 0
#endif

static constexpr int H_TILE = FLASH_MLA_SM70_H_TILE;
static constexpr int K_TILE = 64;
static constexpr int MMA_884_ONLINE_AUTO_K_TILE = 0;
static constexpr int MMA_884_ONLINE_K_TILE = FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE;
static constexpr int D_TILE = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;
static constexpr int HEAD_DIM_V = 512;
static constexpr int NUM_THREADS = FLASH_MLA_SM70_CTA_THREADS;
static constexpr bool USE_MMA_884_ONLINE = FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE != 0;
static constexpr int MMA_884_REGISTER_OUTPUT_CTA_THREADS = 256;
static constexpr int MMA_884_ONLINE_SCALAR_COUNT = 5;
static constexpr int MMA_884_ONLINE_REDUCE_SCRATCH = NUM_THREADS / 32;
static constexpr int MAX_DV_PER_THREAD = (HEAD_DIM_V + NUM_THREADS - 1) / NUM_THREADS;

}  // namespace sm70::dense
