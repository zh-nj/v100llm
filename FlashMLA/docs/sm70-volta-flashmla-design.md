# FlashMLA SM70 (Volta) 适配设计文档

## 背景与目标

目标是在 SM70 / Volta GPU（主要按 V100 处理）上，为 vLLM 推理 DeepSeek V4 Flash 所依赖的 FlashMLA 提供可编译、可运行、可验证的后端。这个工作不是把现有 SM90/SM100 kernel 放开到 `sm_70`，而是新增一套 Volta 专用实现，并保持 FlashMLA 现有 Python/C++ API 形状不变。

本设计使用的本地依据：

- FlashMLA 当前源码：`setup.py` 只生成 `sm_90a` / `sm_100f`，`csrc/api/common.h` 只识别 SM90/SM100，dense decode、sparse decode、sparse prefill 入口均无 SM70 分支。
- FlashMLA 文档：dense decode 的核心形状固定在 `head_dim_qk in {512, 576}`、`head_dim_v = 512`、`page_block_size = 64`；sparse FP8 V3.2 格式是每 token 656 bytes。
- `/mnt/data/apps/flash-attention-v100/docs/volta.md`：Volta 的核心 Tensor Core 指令是 warp 级 `mma.sync.aligned.m8n8k4`，按 quadpair 映射，且不能依赖 `ldmatrix` / TMA / warpgroup。
- `/mnt/data/apps/flash-attention-v100`：提供 `compute_70/sm_70` 构建、Volta WMMA/共享内存/softmax 组织经验，但其 MHA head dim 上限和形状不直接覆盖 FlashMLA MLA。
- `/mnt/data/apps/lmdeploy` TurboMind：提供更接近本项目的 SM70 分派与 MMA_884 attention 经验，尤其是 `src/turbomind/kernels/core/mma.h`、`attention_config.h`、`mainloop_sm70.h`、`impl_884_decode.h`、`decoding_config.h`。
- 两份预研报告：作为方向参考，但本设计只采用能被上述本地代码和 Volta 文档支撑的结论。

## 结论

推荐路线是：

1. 新增 `csrc/sm70/...` 后端，不改现有 SM90/SM100 代码路径的行为。
2. 第一阶段只承诺 dense decode 的 FP16 快路径，使用 Volta `mma.sync.aligned.m8n8k4` / MMA_884 组织，复用现有 scheduler 和 combine 的接口约定。
3. 第二阶段补 sparse decode，外部仍接受 FlashMLA 当前 FP8 KV cache 格式，内部按 tile gather + dequantize 到 FP16；当前默认路径已推进到 online MMA_884 QK/PV 主循环，`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=0` 保留 warp-SIMT 回退，发布级性能化仍继续推进。
4. Sparse prefill 先提供 SM70 BF16 fast path，并已把默认路径推进到 FP16 shared K/V tile 上的 online MMA_884 QK/PV 主循环；`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=0` 保留 warp-SIMT 回退。完整 vLLM DeepSeek V4 Flash 集成仍放到后续里程碑。FlashMLA 只是 vLLM 路径中的一个后端，端到端可用还取决于 vLLM 目标分支的 DeepSeek V4 registry、KV cache、MoE/量化和 fallback 策略。

## 当前实现状态

> 2026-04-30 更新：SM70 sparse decode 已将 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` 作为默认 online alpha。该路径隐式启用 QK/PV MMA_884，将 tile QK、online softmax/LSE 更新和 PV 累加放入同一 tile loop，去掉全量 scores/LSE/PV 两阶段边界。最新默认 256-thread path 又新增 model-specific hybrid accumulator 与 parallel online softmax reduction：V32 使用 shared output accumulator 避免 register spill，ptxas 为 80 registers、0 spill；MODEL1 继续使用 register output accumulator，ptxas 为 97 registers、0 spill。shared bytes 为 `39220/33076`，quick 为 `233.473/408.952/237.247/858.323 us`，long 为 `3667.897/7065.823 us`。`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=0` 保留 `warp_simt_qk+simt_pv` 回退。

> 2026-04-30 更新：SM70 sparse decode 的 total topk 上限已从 2048 提到 8192。当前 FlashMLA 侧通过 `topk + extra_topk <= 8192` 的 API/dispatch 约束，V100 correctness 子集覆盖了纯 V32 `topk=8192 + attn_sink`，以及 MODEL1 `4096+4096` 的 `topk_length + extra_kv + extra_topk_length + attn_sink` 路径；`benchmark/bench_sm70_sparse_decode.py --cases max_topk` 会打印 `splitkv_us/combine_us`、`online_path=mma884_online`、`compute_path=mma884_online` 和 runtime K tile 信息，用于防止 extra cache、split-K/combine 或 sink 支持在扩大上限时回退。

> 2026-04-30 更新：SM70 dense decode 已新增 `FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE=1` online alpha。该路径把 QK `mma_m8n8k4_row_col`、online softmax/LSE 更新和 PV `mma_m8n8k4_row_row` 放入同一 K tile loop，`FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE` 支持 `{0,16,32,64}`，其中 `0` 表示 runtime auto dispatch。随后 dense online 又完成两步主路径性能化：256-thread 专用 register output accumulator 去掉 512 维 FP32 shared output accumulator，parallel tile softmax reduction 用 warp scratch 替代 thread0 串行 tile softmax 扫描。fresh ptxas 为 113 registers、0 spill；K_TILE=64 shared bytes 为 `65844/74036`，静态 occupancy 降到 `1 block/SM, 12.5%`。V100 smoke 显示 K_TILE=64 quick 为 `247.136/200.288/205.056/195.264 us`，long `S_k=16384` 为 `633.728/693.504 us`，仍慢于默认 scalar score-cache 的 `400.672/476.864 us`，因此 dense MMA_884 online 保持 opt-in alpha，不切默认。

> 2026-04-30 后续更新：dense MMA_884 默认路径选择已推进到 `K_TILE=0` runtime-auto 双层分派。当前 auto 会选择 K64 online kernel；一次长上下文 D512 K16 探针虽然把 shared bytes 降到 `16500`、occupancy 提到 `25.0%`，但 `S_k=4096/8192/16384` 退到约 `281.941/532.139/1007.957 us`，慢于 K64 auto。最终 V100 online auto correctness 通过，quick 为 `113.664/128.000/156.331/163.157 us`，long `S_k=16384` 为 `685.739/652.629 us`；随后重建回默认 scalar score-cache，dense FP16 correctness 通过，quick 为 `98.304/98.645/98.304/108.885 us`，long `S_k=16384` 为 `349.525/383.317 us`。结论：runtime-auto 机制与显式 K16/K32/K64 编译选项保留，但 dense 默认仍保持 scalar score-cache，online 不再作为当前默认候选。

截至 2026-04-30，仓库内已经具备 SM70 构建开关、`Arch::is_sm70()`、SM70 common primitive header、correctness-first FP16 dense decode alpha、correctness-first sparse FP8 decode alpha，以及 SM70 sparse prefill BF16 fast path。dense alpha 使用 `DenseAttnDecodeParams`、`block_table`、`cache_seqlens` 和现有 Python API，已经接入现有 scheduler，并以默认 `dense::H_TILE=4`、`dense::NUM_THREADS=256` 做 q-seq tile launch；未切分请求直接写最终 `out/lse`，切分请求写 `o_accum/lse_accum` 并交给 `smxx::decode::run_flash_mla_combine_kernel` 合并。当前 scalar dense path 已按 `K_TILE=64` 在 shared memory 缓存 QK score，并用在线 softmax 更新每线程负责的 V accumulator，避免 PV 阶段为每个 `d_v` 重复计算 QK。`FLASH_MLA_SM70_H_TILE` 和 `FLASH_MLA_SM70_CTA_THREADS` 可在 build time 覆盖默认值，用于扫描 `{4,8,16} x {128,256}`。

SM70 sparse decode 增加 `Decode_Sm70_Fp8_Dequant_Impl` 和 `csrc/sm70/decode/sparse_fp8/...`，支持 V32 (`D_qk=576`) 和 MODEL1 (`D_qk=512`)、`H_q=64/128`、`topk + extra_topk <= 8192`、`topk_length`、`extra_kv`、`extra_topk_length`、`attn_sink`，并已接入 `smxx::decode` scheduler/combine。默认路径已切到 `256/32/mma884_online`：`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` 隐式启用 QK/PV MMA_884，并在同一 tile loop 内完成 QK、online softmax/LSE 更新和 PV 累加；`=0` 保留 `warp_simt_qk+simt_pv` shared-staging 回退。

SM70 sparse prefill fast path 位于 `csrc/sm70/prefill/sparse/*`，支持 BF16 `q/kv`、`D_qk in {512,576}`、`D_v=512`、`H_q=64/128`、`topk <= 8192`、`topk_length`、`attn_sink`。当前默认路径已切到 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=1` + `FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE=32`：每个 CTA 负责一个 `(q, head)`，将 BF16 KV staged 为 FP16 shared tile，并在同一 K tile loop 内完成 QK MMA_884、online softmax/LSE 更新和 PV MMA_884 累加。`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=0` 保留为 `warp_simt_qk+simt_pv` shared-staging 回退；回退路径默认 256-thread ptxas 最大为 29 registers、0 spill，online 默认 build 为 48 registers、0 spill。`benchmark/bench_sm70_sparse_prefill.py` 通过 `qk_path`、`pv_path`、`online_path`、`compute_path` 字段区分回退两阶段路径与默认 `mma884_online`；本轮 `K_TILE=16/32` 复扫后，`K_TILE=16` 虽提高 occupancy，但 `max_topk=8192` 约 `29200.224 us`，慢于 `K_TILE=32` 的约 `21693.184 us`，因此默认保持 `256/32/mma884_online`。

`benchmark/bench_sm70_sparse_decode.py` 现在作为 sparse FP8 decode alpha 的独立性能脚手架，覆盖 `H_q=64/128`、V32/MODEL1、`topk_length`、`extra_kv`、`extra_topk_length` 和 `attn_sink`，输出 `splitkv_us`、`combine_us`、TFLOPS、GB/s、correctness diff、shared bytes、CTA threads、静态 `active_blocks_per_sm/theoretical_occupancy_pct`、`qk_path`、`pv_path`、`online_path`、`compute_path` 与 ptxas 记录字段。2026-04-29 后，sparse scalar fallback 已新增 `SPARSE_K_TILE=32` 的 FP16 shared K/V staging：按 tile 将 V32/MODEL1 dequant 后的 K/V 写入 `cutlass::half_t` shared layout，QK score 使用 warp-parallel dot reduction 从 shared tile 读取，PV accumulator 也从 shared tile 读取。该回退路径 256-thread sm_70 ptxas 显示 V32/MODEL1 均为 64 registers、0 spill、16B stack frame，quick benchmark 约 `263.271/415.559/288.004/982.230 us`。2026-04-30 已补 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=1 FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=1` 两阶段 opt-in path：256-thread V32/MODEL1 ptxas 为 64 registers、20/28B spill stores，128-thread 探针为 95/96 registers、0 spill；V100 quick case 约 `649.319/1077.890/736.043/2565.772 us`。该组合慢于回退路径，因此只保留为验证脚手架；默认发布候选转向 fused/online Volta MMA_884 主循环。

2026-04-30 后，sparse decode benchmark 还会输出 `online_path` 与 `compute_path`。`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` 会启用 compact shared online layout；默认路径已切到 `256/32/mma884_online`。最新 256-thread 默认 path 改为 model-specific hybrid accumulator：V32 保留 shared output accumulator 以消掉 register spill，MODEL1 使用 register output accumulator 去掉 512 维 FP32 shared buffer，并用 parallel online softmax reduction 替换 thread0 串行 tile softmax。V32/MODEL1 shared bytes 为 `39220/33076`，静态 occupancy 仍为 `2 blocks/SM, 25.0%`。ptxas 事实是 V32 `80 registers、0 spill`，MODEL1 `97 registers、0 spill`；128-thread fallback 探针为 V32 72 registers、4B/8B spill，MODEL1 80 registers、8B/4B spill。V100 quick 为 `233.473/408.952/237.247/858.323 us`，long 为 `3667.897/7065.823 us`，max LSE diff `<= 9.53674e-07`。该改动把上一版 V32 的 `128 registers、28B spill` 压回无 spill，同时保留 MODEL1 register accumulator 收益。

同日后续的 CTA/K tile 复扫改为默认路径内的 V32-only runtime adaptive K tile dispatch：短 topk 继续调度 `K_TILE=32` online kernel；V32 `topk + extra_topk >= 512` 时调度同源 `K_TILE=16` online kernel，使 long V32 shared bytes 降到 `20660`、occupancy 提到 `3 blocks/SM, 37.5%`，V100 long 约 `3243.768 us`，max out/lse diff 为 `6.10352e-05/9.53674e-07`。MODEL1 由于 `97 registers` 限制仍保持 K32 kernel，long 约 `7094.224 us`；一次全局 K16 探针显示 MODEL1 long 会退到约 `8963.554 us`，所以 dispatch 明确限定在 V32。该策略保留 quick case 的 `runtime_k_tile=32`，同时把长 V32 的 K16 shared-footprint 收益并入默认 online path。

SM70 sparse prefill 的 fast path 也有独立 build-time tuning 开关：`FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS` 允许 `{128,256}`，默认 `256`；`FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE` 允许 `{16,32}`，默认 `32`。2026-04-29 早期 quick scan 中 128-thread 版本可编译运行，但 4 个 quick case 均不优于默认 256-thread 版本；shared-staging 之后 warp-SIMT 回退路径 256-thread 为 28/29 registers、0 spill，128-thread 单文件编译最大为 32 registers、0 spill。MMA_884 QK 为 40 registers、0 spill，MMA_884 QK/PV 与 online 路径均为 48 registers、0 spill。本轮复扫后默认切到 `256/32/mma884_online`；`K_TILE=16` 作为后续短上下文/occupancy 探针保留，但不作为默认。

2026-04-30 后续更新：dense decode 与 sparse decode 的 MMA_884 online path 已共用 `csrc/sm70/common/softmax.h` 中的 warp-scratch block reduction 和 parallel online-softmax tile update。dense path 使用不跳过 `-inf` 的连续 KV tile 版本，sparse decode 使用跳过 invalid token `-inf` 的版本；这样后续调 CTA/K tile、register accumulator 和 q-head tile 主循环时，row max/row sum/online scale 语义不会在两条 decode 路径里继续分叉。

同一轮还新增了 `csrc/sm70/common/mma_884_attention.h`，把 dense/sparse decode 共享的 Volta attention primitive 从各自 `.cuh` 中抽出来：`mma884_accumulate_qk` / `mma884_accumulate_pv` 封装底层 `mma_m8n8k4_row_col/row_row`，并统一 row0 score 写回、probability fragment 装载和 row0 PV 输出映射。dense 与 sparse decode 仍各自负责 Q/K/V 装载、token 有效性和 scheduler 分派，但完整 sparse decode `mma884_online` 主循环现在和 dense online path 共享同一组 QK/PV fragment accumulate 与 row 映射 helper。该整理已通过默认 SM70 build、sparse decode V100 13/13 correctness、dense FP16 correctness 和 sparse quick benchmark 复核。

尚未完成的关键性能路径包括：prefill online 默认路径的更大长上下文矩阵与 CTA 复扫、dense decode online alpha 继续压低 register/shared 占用直到可挑战默认 scalar score-cache，以及 q-head tile 级别的完整 MMA_884 主循环整理。dense MMA_884 的本轮默认路径选择已经收敛为继续保留 scalar score-cache 默认、online 保持 opt-in。SM70/Volta dense decode 明确保持 FP16-only，不支持 BF16 compat。

SM70 sparse prefill 的下一阶段参考本地两套代码：`/mnt/data/apps/lmdeploy/src/turbomind/kernels/attention/attention_config.h` 中的 `SM70_PREFILL_USE_MMA_884` 配置、`impl_884.h` 中的 `mma_m8n8k4_row_col/row_row` QK/PV 主循环、`mainloop_sm70.h` 中的 shared K/V load + softmax + PV 流水，以及 `/mnt/data/apps/flash-attention-v100/docs/volta.md` 中对 Volta `m8n8k4`、quadpair 映射和无 `ldmatrix` 的约束说明。FlashMLA 后续不直接改这些参考仓库，而是在 `csrc/sm70/prefill/sparse/*` 内实现等价的 BF16-to-FP16 staging 和 MMA_884 tiled path。

vLLM DeepSeek V4 集成边界已做静态核对。当前 `/mnt/data/apps/vllm` 的 `DeepseekV4MLAAttention` decode 路径调用 sparse `flash_mla_with_kvcache`，传入 `fp8_ds_mla` KV cache、MODEL1 584B layout、`indices/topk_length`、`attn_sink` 和可选 `extra_k_cache/extra_topk_length`，与 SM70 sparse FP8 decode alpha 的接口能力对齐。Prompt prefill 会调用 `flash_mla_sparse_fwd`；FlashMLA 侧已有 SM70 BF16 fast path v1，但真实端到端仍受 vLLM FlashMLA sparse backend gate 限制，该 gate 当前只允许 SM90/SM100。集成说明和 smoke/benchmark 入口见 `docs/sm70-volta-vllm-integration-notes.md` 与 `benchmark/bench_vllm_deepseek_v4_flash_sm70.py`。

## 非目标

- 不尝试在 SM70 上复用 SM90 TMA/GMMA/warpgroup kernel。
- 不在第一版实现 SM90 sparse FP8 kernel 的 Hopper crossover/cluster shared memory 技术；Volta 没有对应硬件能力。
- 不在第一版保证 BF16 Tensor Core 等价。SM70 Tensor Core 快路径以 FP16 输入、FP32 累加为核心；sparse prefill v1 先用 BF16 SIMT fast path，后续 MMA_884 版本需要把 BF16 staged 为 FP16。
- 不修改 flash-attention-v100 或 lmdeploy 参考仓库。

## 当前 FlashMLA 分层

FlashMLA 现有路径可以分成四层：

| 层级 | 当前文件 | 现状 | SM70 改造方式 |
|---|---|---|---|
| Python API | `flash_mla/flash_mla_interface.py` | `get_mla_metadata()` 延迟初始化，`flash_mla_with_kvcache()` 分 dense / sparse | 保持不变 |
| C++ API | `csrc/api/*.h` | 架构分派只覆盖 SM90/SM100 | 增加 `Arch::is_sm70()` 和 SM70 impl |
| 公共 decode 辅助 | `csrc/smxx/decode/*` | scheduler / combine 与具体 SM 解耦程度较高 | 优先复用，若 `combine` 在 SM70 编译或 PDL 行为有问题再加 fallback |
| 计算 kernel | `csrc/sm90/*`, `csrc/sm100/*` | 依赖 TMA、GMMA、warpgroup、SM90/SM100 Cutlass/CuTe 布局 | 新增 `csrc/sm70/*` |

## 硬件约束

SM70 必须按 Volta 处理：

- Tensor Core 主基元：`mma.sync.aligned.m8n8k4.*.f32.f16.f16.f32`。
- 执行粒度：warp 级，硬件按 4 个 quadpair 组织，每个 quadpair 计算 8x8x4 子块。
- 输入类型：Tensor Core 快路径是 FP16 输入、FP32 累加。
- 不可依赖：`ldmatrix`、`cp.async`、TMA、GMMA、warpgroup、distributed shared memory、CTA cluster barrier。
- 共享内存上限按 V100 约 96 KiB 设计；寄存器压力比 SM90 更紧。

这决定了 SM70 后端需要自己的数据搬运、shared memory layout、fragment 装载和计算 mainloop。

## 数据组织

### Dense Decode

Dense decode 保持当前 FlashMLA 语义：

- `q`: `[B, S_q, H_q, D_qk]`
- `k_cache`: `[num_blocks, page_block_size, H_kv, D_qk]`
- `block_table`: `[B, max_num_blocks]`
- `cache_seqlens`: `[B]`
- `D_qk in {512, 576}`
- `D_v = 512`
- `page_block_size = 64`
- 输出 `out`: `[B, S_q, H_q, D_v]`
- 输出 `lse`: `[B, H_q, S_q]`

SM70 dense 快路径建议先限定：

- `q.dtype == k_cache.dtype == fp16`
- `H_q % H_kv == 0`
- `page_block_size == 64`
- `S_q` 支持 1、2、4，但首版以 decode/MTP 的 `S_q <= 4` 为优化目标

BF16 输入在 SM70 上不能假装等价。设计上提供两档：

- alpha：SM70 dense fast path 只接受 FP16，BF16 给出清晰错误。
- beta：为 BF16 输入分配 FP16 scratch，转换后进入同一 FP16 kernel，输出按原 dtype 语义返回。

### Sparse Decode

Sparse decode 保持当前接口：

- `q`: `[B, S_q, H_q, D_qk]`, 当前要求 BF16
- `kv`: `[num_blocks, page_block_size, H_kv, bytes_per_token]`
- `indices`: `[B, S_q, topk]`
- `topk_length`, `attn_sink`, `extra_kv`, `extra_indices`, `extra_topk_length` 语义保持

优先支持 V3.2 / V32 格式：

- `D_qk = 576`
- `D_v = 512`
- 每 token 656 bytes
- NoPE: 512 个 `float8_e4m3`
- scales: 4 个 `float32`，每 128 个 NoPE 元素一个 scale
- RoPE: 64 个 `bfloat16`

SM70 内部不使用原生 FP8/BF16 Tensor Core。sparse alpha 的数据流为：

1. 按 `indices` gather 当前 topk block/token。
2. 用 128-bit vector load 读取 NoPE / scale / RoPE 分段。
3. 在寄存器或 shared memory staging 中做 `fp8_e4m3 -> fp16`、`bf16 -> fp16`。
4. 将重组后的 K tile 写入 SM70 计算友好的 shared layout。
5. QK/PV 使用 dense SM70 MMA 核心。

MODEL1 格式（`D_qk = 512`，NoPE 448，RoPE 64，e8m0 scale）已纳入 correctness-first sparse alpha。当前 SM70 scalar path 按每 block 的 576B token data 区和 block 尾部 8B-per-token scale 区做显式寻址。

## SM70 后端结构

建议新增目录：

```text
csrc/sm70/common/
  arch.h
  mma_884.h
  layout.h
  load_store.h
  softmax.h
  fp8_dequant.h
  bf16_cast.h
csrc/sm70/decode/dense/
  config.h
  splitkv_mla_sm70.h
  splitkv_mla_sm70.cuh
  instantiations/fp16.cu
  instantiations/bf16_compat.cu
csrc/sm70/decode/sparse_fp8/
  config.h
  dequant.h
  splitkv_mla_sm70_sparse.h
  splitkv_mla_sm70_sparse.cuh
  instantiations/v32_fp8.cu
  instantiations/model1_fp8.cu
```

`mma_884.h` 以 TurboMind 的 `mma_m8n8k4_row_col` / `mma_m8n8k4_row_row` 为参考，实现 FlashMLA 自己的最小包装。注意 license 和来源说明写入文件头。

## Dense Decode 计算设计

SM90 dense decode 使用 64 query-head tile、TMA 加载 64x64 KV tile、WGMMA 执行 QK/PV。SM70 不能照搬这个 tile，因为 V100 共享内存和寄存器不足。

SM70 dense decode 推荐初版 tile：

| 维度 | 建议初值 | 说明 |
|---|---:|---|
| CTA threads | 256 | 2026-04-29 quick scan 中 256 全面快于 128 |
| H tile | 4 | score-cache 后的 `H_q=8` tile case 中 4/256 明显快于 8/256 和 16/256 |
| K token tile | 64 | 与 FlashMLA page block size 对齐 |
| D tile | 64 | 576 拆成 9 个 64 维 tile，512 拆成 8 个 |
| V tile | 512 完整保留，必要时调小 H tile | 避免重复 QK/softmax |
| 累加 | FP32 | `lse` 和 online softmax 保持稳定 |

当前 scalar dense alpha 的 tile/thread 是 build-time 参数，不是运行时参数：

- `FLASH_MLA_SM70_H_TILE`：允许 `{4,8,16}`，默认 `4`。
- `FLASH_MLA_SM70_CTA_THREADS`：允许 `{128,256}`，默认 `256`。

当前 sparse decode alpha 的 shared K/V tile 也是 build-time 参数：

- `FLASH_MLA_SM70_SPARSE_DECODE_K_TILE`：允许 `{16,32,64}`，默认 `32`。

2026-04-29 在 score-cache streaming 优化前完成 quick scan，case 为 `B=1,S_q=1,H_q=1,H_kv=1,S_k in {20,140},D in {512,576}`，`warmup=0,runs=1`：

| H tile | CTA threads | S_k=20 D512 us | S_k=20 D576 us | S_k=140 D512 us | S_k=140 D576 us |
|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 2785.472 | 3012.736 | 8290.528 | 9332.032 |
| 4 | 256 | 1683.520 | 1803.744 | 4739.200 | 5253.376 |
| 8 | 128 | 2713.216 | 2979.520 | 8321.152 | 9272.384 |
| 8 | 256 | 1596.224 | 1741.920 | 4812.448 | 5269.376 |
| 16 | 128 | 2724.064 | 3003.168 | 8291.264 | 9288.224 |
| 16 | 256 | 1600.736 | 1736.576 | 4722.816 | 5296.448 |

该表用于选择 scalar alpha 的初始默认值；其中 `H_q=1` 不充分覆盖 `H_TILE` 敏感性，后续若继续做 q-head tile 调参，应增加 `q_seq_per_hk > 1` 的 benchmark case。

score-cache streaming 优化后，默认 `4/256` 的当前复核为：

- quick：`S_k=20,D512` `215.232 us`，`S_k=20,D576` `186.464 us`，`S_k=140,D512` `172.800 us`，`S_k=140,D576` `166.080 us`。
- long：`S_k=4096`、`warmup=3,runs=5`，D512 `116.326 us`，D576 `112.230 us`。
- long smoke：`S_k={4096,8192,16384}`、`warmup=0,runs=1` 已跑通 D512/D576 验收集合；该 smoke 只证明可运行与数值边界，不替代上面的稳定性能基线。
- tile (`H_q=8,H_kv=1`)：`S_k=140,D512` `311.296 us`，`S_k=140,D576` `330.547 us`，`S_k=4096,D512` `445.440 us`，`S_k=4096,D576` `487.219 us`。
- ptxas：64 registers、0 spill stores、0 spill loads；dynamic shared memory `1280B` for default `4/256`.

score-cache 后的 `H_q=8` / `q_seq_per_hk=8` H-tile 复扫如下，均为 CTA threads 256、`warmup=3,runs=5`：

| H tile | S_k=140 D512 us | S_k=140 D576 us | S_k=4096 D512 us | S_k=4096 D576 us |
|---:|---:|---:|---:|---:|
| 4 | 311.296 | 330.547 | 445.440 | 487.219 |
| 8 | 592.896 | 634.266 | 613.376 | 661.094 |
| 16 | 591.258 | 638.771 | 616.243 | 661.914 |

主循环：

1. CTA 负责一个 `(scheduler partition, kv_head, q_seq_per_hk tile)`。`q_seq_per_hk = S_q * H_q / H_kv` 已把原始 q-head 维折叠进 dense API 的 tile 维度，因此这是当前实现中等价且可验证的 q-head tile 语义；batch 由 `DecodingSchedMeta` partition 在 CTA 内遍历。
2. 根据 `DecodingSchedMeta` 遍历本 partition 内的 batch/split 与 KV page。
3. 对每个 page：
   - global -> shared 手工加载 K 的 64-token tile，使用 padding/swizzle 降低 bank conflict。
   - 对 `D_qk` 按 64 维 tile 循环，使用 `mma_m8n8k4_row_col` 累加 QK scores。
   - 做 online softmax，维护 `row_max` / `row_sum` / `lse`。
   - 对 V 使用 `mma_m8n8k4_row_row` 或等价布局计算 P·V，更新 O accumulator。
4. 写 `o_accum` 和 `lse_accum`。
5. 多 split 时调用现有 `smxx::decode::run_flash_mla_combine_kernel` 合并。

关键点：

- K 和 V 逻辑上来自同一 `k_cache`，但 PV 只使用前 `D_v=512` 维。
- 如果 ptxas 显示 O accumulator register spill，先降低 H tile；只有在 H tile 仍不够时，才考虑 split V 维。split V 会重复 QK，代价大。
- `q_seq_per_hk = S_q * H_q / H_kv` 的 reshape 语义保留，与现有 dense API 一致。

## Sparse Decode 计算设计

Sparse SM70 后端分两步落地。

### Sparse Alpha: FP8 Dequant Path

只支持：

- `D_qk in {576, 512}`
- `D_v = 512`
- `H_kv = 1`
- `H_q in {64, 128}`
- V32 656-byte KV cache
- MODEL1 584-byte logical row / block-tail scale KV cache
- `attn_sink` 支持
- `topk_length`
- `extra_kv` / `extra_indices`
- `extra_topk_length`

数据流：

```text
indices/topk
  -> gather token address
  -> load NoPE 512B + scales 16B + RoPE 128B
  -> fp8_e4m3/bf16 software dequant to fp16
  -> shared K tile [64, 576], V view [64, 512]
  -> SM70 QK/PV core (default warp-SIMT, opt-in MMA_884 QK or QK+PV)
  -> o_accum/lse_accum
  -> smxx combine
```

dequant 设计：

- 不依赖 CUDA FP8 hardware conversion。
- 以 `uint8_t` 读取 e4m3，位运算重建 FP16。
- scale 先读 FP32；可在寄存器中转 half 或保持 FP32 乘后转 half。
- RoPE BF16 采用显式 `bf16 -> fp16`，溢出时做饱和或按测试策略报错。
- 优先使用 128-bit vector load，保证 656-byte stride 下的访问尽量合并。

### Sparse Prefill BF16 Fast Path

SM70 sparse prefill 已从早期 BF16 SIMT v1 推进到默认 online MMA_884 fast path，同时保留 warp-SIMT 回退供定位和兼容：

- 输入限定 BF16 `q/kv`，支持 `D_qk in {512,576}`、`D_v=512`、`H_q in {64,128}`、`topk <= 8192`、`topk_length`、`attn_sink`。
- 一个 CTA 负责一个 `(q, head)`；CTA 线程按 `K_TILE` staged BF16 KV 到 FP16 shared K/V tile。默认 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=1` 会在同一 tile loop 内做 QK MMA_884、online softmax/LSE 更新和 PV MMA_884 累加。
- `FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS` 允许 `{128,256}`，默认 `256`；`FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE` 允许 `{16,32}`，默认 `32`。本轮复扫后 `K_TILE=16` 虽提高 occupancy，但 `max_topk=8192` 慢于 `K_TILE=32`，因此默认保持 `256/32/mma884_online`。
- `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK` 允许 `{0,1}`，默认 `0`；设为 `1` 时把 QK score 切到 Volta `mma_m8n8k4_row_col` opt-in 路径。当前 QK-only opt-in ptxas 为 40 registers/0 spill，quick benchmark 约 `259.776-1163.776 us`。
- `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV` 允许 `{0,1}`，默认 `0`；与 `USE_MMA_884_QK=1` 组合后用 Volta `mma_m8n8k4_row_row` 计算 PV row0 输出，形成 `mma884_qk+mma884_pv` 两阶段路径。当前组合 opt-in ptxas 为 48 registers/0 spill，quick benchmark 约 `368.896-1757.184 us`。
- `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=0` 会强制使用 `warp_simt_qk+simt_pv` shared-staging 回退；该回退路径的 QK score 和 PV accumulator 都从 shared tile 读取，thread 0 按 topk 顺序做确定性 LSE/sink。
- `benchmark/bench_sm70_sparse_prefill.py --cases max_topk` 固定 `topk=8192` 边界 smoke；默认 online 路径当前资源形态约为 `39184B` dynamic shared、`2 blocks/SM`、`25.0%` 静态 occupancy。
- 后续性能化应扩大默认 online 路径的长上下文矩阵并复扫 CTA threads，同时继续整理 dense/sparse decode 的 fused/online MMA_884 主循环边界。

### Sparse Beta: Feature Completion

补齐：

- `H_q=64/128` 双路径性能调优。当前已有 `benchmark/bench_sm70_sparse_decode.py` 固定 alpha 基线，并能用 `qk_path`、`pv_path`、`online_path`、`compute_path` 区分默认 `mma884_online`、回退 `warp_simt_qk+simt_pv`、opt-in `mma884_qk+simt_pv` 与 opt-in `mma884_qk+mma884_pv`；后续优化集中在默认 online 路径的 CTA/K tile 复扫和 shared layout 整理。
- `FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS` 允许 `{128,256}`，默认保持 `256`；shared staging 后 128-thread 手工编译显示 MODEL1 registers 低于默认 256，online MMA_884 默认化后仍需要重新做 128/256 runtime quick/long 复扫。
- `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK` 允许 `{0,1}`，默认 `0`；设为 `1` 时只替换 QK score 计算，当前 ptxas 为 80 registers/0 spill，PV accumulator 仍为 SIMT。
- `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV` 允许 `{0,1}`，默认 `0`；与 `USE_MMA_884_QK=1` 组合后用 Volta `mma_m8n8k4_row_row` 计算 PV row0 输出，形成 `mma884_qk+mma884_pv` 两阶段路径。当前组合 256-thread ptxas 为 V32 64 registers、20B spill stores，MODEL1 64 registers、28B spill stores；quick benchmark 约 `649.319-2565.772 us`，慢于默认路径。
- `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE` 允许 `{0,1}`，默认 `1`；设为 `1` 时隐式启用 QK/PV MMA_884，并把 tile scores、online softmax、output accumulator 与 PV 累加放在同一 tile loop。设为 `0` 时回到 `warp_simt_qk+simt_pv` shared-staging 回退。当前 online 256-thread ptxas 为 V32/MODEL1 80 registers、0 spill；本轮 `K_TILE=16/32` 复扫后选择 `32` 作为默认，`16` 仅保留为长上下文探针。

## Dispatcher 与构建

`setup.py` 增加可控 SM70 编译目标：

- 建议新增 `FLASH_MLA_ENABLE_SM70=1`。
- 默认是否启用 SM70 由维护策略决定。若主线环境可能进入 CUDA 13，则默认不启用更安全。
- SM70 构建建议使用 CUDA 12.x，尤其是仍可离线编译 Volta 的工具链。

`csrc/api/common.h` 增加：

```cpp
bool is_sm70() const {
    return major == 7 && minor == 0;
}
```

`csrc/api/dense_decode.h`：

- SM90a 保持原路径。
- SM70 走 `sm70::run_flash_splitkv_mla_kernel<half>` 或 BF16 compat。
- 其他架构保持原错误。

`csrc/api/sparse_decode.h`：

- SM100/SM90 保持原路径。
- SM70 在 sparse alpha 完成后新增 `Decode_Sm70_Fp8_Dequant_Impl`。
- 未实现 feature 组合必须给清晰错误，不允许静默落入错误 kernel。

## 回退策略

第一版回退策略应显式、可诊断：

| 输入组合 | 行为 |
|---|---|
| SM70 dense FP16, supported shape | 走 SM70 dense fast path |
| SM70 dense BF16, BF16 compat 未启用 | 报错，提示使用 FP16 或启用 compat |
| SM70 sparse V32 FP8, alpha supported feature | 走 SM70 sparse dequant path |
| SM70 sparse prefill BF16, supported shape | 走 SM70 BF16 SIMT fast path v1 |
| SM70 sparse unsupported feature | 报错并列出 `arch=sm70`、模型格式和缺失 feature |
| 非 SM70/SM90/SM100 | 保持原行为 |

不要在 FlashMLA 内部自动把 sparse 降成 dense，除非 vLLM 上层明确提供完整 dense KV cache。FlashMLA sparse 输入只有 `indices` 和量化 KV，不能凭空恢复 dense 语义。

## 测试与验收

测试分四层：

1. 静态/构建测试：确认 `sm_70` cubin 生成，SM90/SM100 source 不受影响。
2. 单 kernel 数值测试：使用 `tests/test_flash_mla_dense_decoding.py` 和 `tests/ref.py` 的 PyTorch 参考，先跑 FP16 dense。
3. Sparse dequant 数值测试：复用 `tests/quant.py` 和 `tests/test_flash_mla_sparse_decoding.py`，单独比较 dequant 后参考值。
4. 集成测试：目标 vLLM 分支中跑 DeepSeek V4 Flash 最小 prompt，记录是否实际命中 FlashMLA SM70 后端。

建议最小验收：

- `python tests/test_flash_mla_dense_decoding.py --dtype fp16` 在 V100 上通过一个小矩阵子集。
- `cuobjdump` 能看到 `sm_70` 目标。
- SM90/SM100 原测试不因 dispatcher 改动回归。
- sparse alpha 至少通过 V32/MODEL1、`topk=64/576/2048/8192`、`4096+4096` extra cache、`attn_sink`、`topk_length`、`extra_topk_length`、`H_q=64/128`、`S_q=1/2` 的正确性测试。
- sparse decode alpha 可用 `benchmark/bench_sm70_sparse_decode.py` 输出 H_q=64/128、V32/MODEL1、feature 组合的 us、TFLOPS、GB/s、registers、spills、静态 occupancy、`qk_path`、`pv_path`、`online_path`、`compute_path` 和 correctness diff；`--cases max_topk` 固定验证 8192 级别边界。
- sparse prefill fast path 通过 `tests/test_flash_mla_sparse_prefill.py` 的 SM70 子集，并可用 `benchmark/bench_sm70_sparse_prefill.py` 输出 us、TFLOPS、GB/s、registers、spills、静态 occupancy 和 correctness diff。

## 性能观察项

必须记录：

- kernel time：splitkv 和 combine 分开。
- decode tokens/s 或每 step us。
- achieved TFLOPS / GB/s。
- ptxas register count / spill。
- occupancy。
- topk / seqlen / batch / H tile / dtype。

性能基线：

- FlashMLA SM90/SM100 原路径：只作为绝对上界参考，不作为 V100 SLA。
- vLLM/Torch fallback：判断 SM70 后端是否有工程价值。
- TurboMind V100 attention：判断 MMA_884 路线是否合理。
- flash-attention-v100：参考 Volta MHA 行为和调参，不直接当 MLA 基线。

## 风险与开放问题

- `D_v=512` 的 O accumulator 对 SM70 寄存器压力很高，H tile 需要通过 ptxas 和实测决定。
- BF16 输入是否必须无损保留还未确认；若业务允许 FP16，第一版难度明显下降。
- FP8 sparse dequant 在 SM70 上很可能 dequant-bound；需要用 ILP 和加载/计算交织隐藏部分开销。
- vLLM DeepSeek V4 Flash decode 的实际调用形状、KV cache dtype 和 MODEL1 layout 已在 `/mnt/data/apps/vllm` 静态确认；FlashMLA 侧已具备 `flash_mla_sparse_fwd` SM70 BF16 fast path v1，后续真实 smoke 前仍需在 vLLM 侧放开 SM70 FlashMLA gate。
- CUDA 13 及之后对 Volta 离线编译支持收缩，SM70 构建应独立钉住工具链。

## 推荐里程碑

1. M1：SM70 dense FP16 decode 可编译、可跑、正确。
2. M2：SM70 dense 支持 split-K combine、MTP `S_q>1`、基础性能调优。
3. M3：BF16 compat 或明确 FP16-only 部署策略。
4. M4：SM70 sparse V32 FP8 dequant decode alpha。
5. M5：Sparse feature 补齐：SM70 sparse 性能路径。
6. M6：vLLM DeepSeek V4 Flash 端到端 smoke + benchmark 报告。
