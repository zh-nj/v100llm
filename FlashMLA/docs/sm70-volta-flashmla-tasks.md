# FlashMLA SM70 (Volta) 实现任务文档

> **For agentic workers:** 按任务顺序推进。每个任务完成后先跑对应验证，再进入下一任务。不要修改 `/mnt/data/apps/flash-attention-v100` 或 `/mnt/data/apps/lmdeploy`，它们只作为参考。

**Goal:** 为 FlashMLA 增加 SM70 / Volta 后端，先打通 dense decode，再打通 sparse FP8 dequant decode，保持现有 Python API 不变。

**Architecture:** 新增 `csrc/sm70/...` 后端，复用 `csrc/smxx/decode` scheduler/combine，C++ API dispatcher 按架构选择 SM90/SM100/SM70 实现。SM70 kernel 使用 Volta `mma.sync.aligned.m8n8k4`、手工 global/shared load、online softmax 和 FP16 Tensor Core 快路径。

**Tech Stack:** CUDA C++、PyTorch C++ extension、CUTLASS bfloat16/half 类型、FlashMLA 现有 Python tests、V100/SM70 实机验证。

## 最新验证记录

- 2026-04-28：`pytest -q tests/test_sm70_static_contract.py` 通过，19 passed。
- 2026-04-28：`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 python -m pip install -v . --no-build-isolation` 在 `gptq` 环境通过，wheel 可安装。
- 2026-04-28：`CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 python tests/test_flash_mla_dense_decoding.py --dtype fp16` 在 V100 上通过 SM70 alpha 子集，覆盖 `B={1,2}`、`S_q={1,2}`、`S_k={20,140}`、`D_qk={512,576}`、`H_q/H_kv={(1,1),(8,1),(8,2)}`、causal/non-causal。
- 2026-04-28：SM70 dense alpha 保持 FP16-only；BF16 在 Volta 上不支持，dispatcher 给出明确错误。
- 2026-04-28：`CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 python benchmark/bench_sm70_dense_decode.py --cases long --warmup 0 --runs 1` 在 V100 上通过 `S_k=4096` 最小 long case；当前 scalar alpha 很慢，`D=512` 约 `181539 us`，`D=576` 约 `204050 us`。
- 2026-04-28：`CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 SM70 sparse FP8 alpha 子集，13/13 correctness，覆盖 V32/MODEL1、`H_q={64,128}`、`S_q={1,2}`、`topk={64,576,2048}`、`topk_length`、`extra_kv`、`extra_topk_length`、`attn_sink` 开关。
- 2026-04-28：SM70 sparse prefill 调用会明确报错：`SM70 sparse prefill is not supported; SM70 support currently targets sparse decode/dense decode only.`
- 2026-04-29：`pytest -q tests/test_sm70_static_contract.py` 通过，20 passed，新增覆盖 SM70 dense scheduler/split accumulator 静态契约。
- 2026-04-29：`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace` 通过；SM70 dense q-seq tile ptxas 记录为 48 registers、0 spill stores、0 spill loads、1024B dynamic shared memory。
- 2026-04-29：`CUDA_VISIBLE_DEVICES=6 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_dense_decoding.py --dtype fp16` 通过 SM70 dense q-seq tile correctness 子集。
- 2026-04-29：`CUDA_VISIBLE_DEVICES=6 /home/z/anaconda3/envs/gptq/bin/python benchmark/bench_sm70_dense_decode.py --cases long --warmup 0 --runs 1` 通过真实 split-K/combine long case；`S_k=4096` 下 `D=512` 约 `4829.920 us`，`D=576` 约 `5377.920 us`，max abs diff 分别为 `4.76837e-07/0` 与 `2.38419e-07/9.53674e-07`，记录 `h_tile=8`、`cta_threads=256`。
- 2026-04-29：`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过，21 passed，新增覆盖 `FLASH_MLA_SM70_H_TILE` / `FLASH_MLA_SM70_CTA_THREADS` build-time tuning contract。
- 2026-04-29：score-cache 前完成 SM70 dense quick tuning scan：`H_TILE in {4,8,16}`、`CTA threads in {128,256}`；256 threads 全面快于 128，临时默认保持 `H_TILE=8`、`CTA threads=256`。默认配置 ptxas：48 registers、0 spill stores、0 spill loads。
- 2026-04-29：默认 `H_TILE=8`、`CTA threads=256` 重建后，`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_dense_decoding.py --dtype fp16` 在 SM70 上通过。
- 2026-04-29：默认 `H_TILE=8`、`CTA threads=256` long benchmark：`CUDA_VISIBLE_DEVICES=5 ... benchmark/bench_sm70_dense_decode.py --cases long --warmup 3 --runs 5`；`S_k=4096` 下 `D=512` 约 `4713.267 us`，`D=576` 约 `5201.510 us`，max abs diff 分别为 `4.76837e-07/0` 与 `2.38419e-07/9.53674e-07`。
- 2026-04-29：SM70 dense scalar path 改为 `K_TILE=64` shared score-cache + online softmax/PV accumulator，避免 PV 阶段重复 QK；`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过，22 passed。
- 2026-04-29：score-cache 版本默认 `8/256` build 通过；ptxas：64 registers、0 spill stores、0 spill loads，dynamic shared memory 1280B。
- 2026-04-29：score-cache 版本 `CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_dense_decoding.py --dtype fp16` 在 SM70 上通过。
- 2026-04-29：score-cache 版本默认 `8/256` benchmark：quick case `S_k=20/140` 已降到约 `147-196 us`；long case `S_k=4096` 下 `D=512` 约 `127.386 us`，`D=576` 约 `116.941 us`，max abs diff 分别为 `2.38419e-07/0` 与 `0/9.53674e-07`。
- 2026-04-29：新增 benchmark `--cases tile`，使用 `H_q=8,H_kv=1` 覆盖 `q_seq_per_hk=8`，用于 post-score-cache H tile 调参。
- 2026-04-29：score-cache 后复扫 `H_TILE in {4,8,16}`、`CTA threads=256` 的 tile case；`4/256` 明显优于 `8/256` 和 `16/256`，因此默认切到 `H_TILE=4`、`CTA threads=256`。
- 2026-04-29：默认 `4/256` build/correctness/static contract 均通过；ptxas：64 registers、0 spill stores、0 spill loads，dynamic shared memory 1280B；`tests/test_sm70_static_contract.py` 为 22 passed。
- 2026-04-29：默认 `4/256` benchmark：quick case `S_k=20/140` 为约 `166-215 us`；long case `S_k=4096` 下 `D=512` 约 `116.326 us`，`D=576` 约 `112.230 us`；tile case `H_q=8,S_k=4096` 下 `D=512` 约 `445.440 us`，`D=576` 约 `487.219 us`。
- 2026-04-29：新增 vLLM DeepSeek V4 集成说明与 smoke/benchmark 脚手架；`/home/z/anaconda3/envs/gptq/bin/python benchmark/bench_vllm_deepseek_v4_flash_sm70.py --mode inspect --vllm-root /mnt/data/apps/vllm` 通过静态检查，确认当前 vLLM decode 调用 `flash_mla_with_kvcache` sparse FP8 路径，prefill 调用 `flash_mla_sparse_fwd`，且 vLLM gate 仍只放行 SM90/SM100。
- 2026-04-29：`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过，23 passed，新增覆盖 vLLM 集成文档和 benchmark 脚手架契约。
- 2026-04-29：新增 SM70 sparse prefill BF16 correctness fallback；`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace` 通过，sm_70 ptxas 为 27 registers、0 spill stores、0 spill loads；敏感 seed 14 连续 20 次 LSE max diff 稳定在 `9.536743e-07`。该实现随后已升级为 parallel-score SIMT fast path。
- 2026-04-29：`CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 SM70 子集，32/32 correctness cases passed，覆盖 `D_qk={512,576}`、`H_q={64,128}`、`S_q={1,3}`、`topk={64,128}`、`topk_length`、`attn_sink`。
- 2026-04-29：按用户指定测试 mask `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5` 复核 SM70 runtime；`tests/test_flash_mla_sparse_prefill.py` 通过 32/32，`tests/test_flash_mla_sparse_decoding.py` 通过 13/13，`tests/test_flash_mla_dense_decoding.py --dtype fp16` 通过，`pytest -q tests/test_sm70_static_contract.py` 更新后为 25 passed。
- 2026-04-29：按用户要求将 SM70 sparse prefill 从 correctness fallback 升级为 BF16 SIMT fast path v1：CTA 线程并行填充 topk QK score，thread 0 按 topk 顺序做确定性 LSE/sink，输出维度并行写回。`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace` 通过，sm_70 ptxas 为 26 registers、0 spill stores、0 spill loads。
- 2026-04-29：`benchmark/bench_sm70_sparse_prefill.py` 已更新为 SM70 sparse prefill BF16 fast path quick benchmark；同一 mask 下 `--cases quick --warmup 0 --runs 1` 通过，quick case 约 `136.000-422.560 us`，max LSE diff `<= 9.53674e-07`，`pytest -q tests/test_sm70_static_contract.py` 更新后为 26 passed。
- 2026-04-29：补齐 sparse feature gate 的可诊断错误契约；`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过，28 passed；`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace` 通过；`CUDA_VISIBLE_DEVICES=5 ... topk=2049` 负向 smoke 在 SM70 上报出 `arch=sm70, model=MODEL1, missing_feature=TOTAL_TOPK_GT_2048`。
- 2026-04-29：新增 SM70 sparse prefill CTA threads build-time tuning：`FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS` 允许 `{128,256}`，默认 `256`。`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过，29 passed；128/256 两个配置均可 build，ptxas 分别约 27/26 registers、0 spill stores、0 spill loads；quick scan 显示 256 默认整体更优。
- 2026-04-29：新增 `benchmark/bench_sm70_sparse_decode.py`，为 SM70 sparse FP8 decode alpha 输出 H_q=64/128 性能基线；`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py -k sparse_decode_benchmark` 通过；`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python benchmark/bench_sm70_sparse_decode.py --cases quick --warmup 0 --runs 1` 通过，覆盖 V32/MODEL1、`topk_length`、`extra_kv`，max LSE diff `4.76837e-07`。
- 2026-04-29：SM70 sparse decode no-split 行现在同时镜像写入 split-compatible `o_accum/lse_accum`：`out` 仍写 sink 后 BF16，`o_accum` 写 sink 前 FP32 raw accumulator，`lse_accum` 写 log2 LSE。`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace` 通过，ptxas 为 40 registers、0 spill stores、0 spill loads；`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 通过 13/13 correctness。
- 2026-04-29：SM70 sparse decode 解除 `num_sm_parts=1` 限制，`Decode_Sm70_Fp8_Dequant_Impl::get_meta()` 按 SM 数、`S_q`、head group 生成 scheduler partition；kernel grid 第三维使用 `params.num_sm_parts`，split 行写 `o_accum/lse_accum`，no-split 行保持直接写最终 `out/lse` 并镜像 accumulator。`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 通过 13/13；`benchmark/bench_sm70_sparse_decode.py --cases long --warmup 0 --runs 1` 通过并捕获 combine，V32 long 约 `2573.062 us`、combine `8.896 us`，MODEL1 long 约 `4517.676 us`、combine `10.048 us`。ptxas 当前 V32 为 48 registers、0 spill，MODEL1 为 48 registers、12B spill stores/loads。
- 2026-04-29：完成 SM70 sparse decode CTA threads 128/256 复扫，并修正 benchmark 的 per-CTA resource 报告。128-thread build 通过，V32/MODEL1 均为 64 registers、0 spill；quick benchmark 约 `356.446/328.444/411.835/522.074 us`，max LSE diff `<= 9.53674e-07`。默认 256-thread 重建后 quick benchmark 约 `285.342/299.963/339.963/679.833 us`，输出 `cta_threads=256`、V32 48 registers/0 spill、MODEL1 48 registers/12B spill；128 只改善 H_q=128 MODEL1 extra/topk case，因此默认保持 256。
- 2026-04-29：三个 SM70 benchmark 均新增静态 occupancy 输出列：`active_blocks_per_sm` 与 `theoretical_occupancy_pct`，计算基于 V100 每 SM `65536` registers、`96KiB` shared、`2048` threads、`64` warps 上限。`pytest -q tests/test_sm70_static_contract.py` 通过，34 passed；V100 quick smoke 显示默认 dense 为 `4 blocks/SM, 50.0%`，sparse decode 为 `5 blocks/SM, 62.5%`，sparse prefill 为 `8 blocks/SM, 100.0%`。
- 2026-04-29：扩展 `benchmark/bench_sm70_dense_decode.py --cases long` 到验收集合 `S_k={4096,8192,16384}`，并补静态契约防止回退；`pytest -q tests/test_sm70_static_contract.py` 通过，35 passed。V100 smoke：`CUDA_VISIBLE_DEVICES=5 ... --cases long --warmup 0 --runs 1` 覆盖 D512/D576，全 case max LSE diff `<= 9.53674e-07`，默认 `4/256` 静态 occupancy 为 `4 blocks/SM, 50.0%`。
- 2026-04-29：新增 SM70 sparse prefill `max_topk` benchmark smoke，覆盖 `S_q=1,S_kv=8192,topk=8192,H_q=64,D_qk=576,attn_sink` 支持边界，并补静态契约防止 benchmark case 回退。`CUDA_VISIBLE_DEVICES=5 ... benchmark/bench_sm70_sparse_prefill.py --cases max_topk --warmup 0 --runs 1` 通过，约 `12461.184 us`，max out/lse/max-logits diff 为 `0.000122201/4.76837e-06/0`，shared bytes `32776`，静态 occupancy `2 blocks/SM, 25.0%`。
- 2026-04-29：SM70 sparse decode scalar alpha 新增 `SPARSE_K_TILE=32` 的 FP32 shared K/V staging，QK score 与 PV accumulator 均从 shared tile 读取，为后续 FP16/MMA_884 主循环铺路；`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py -k stages_kv` 通过，随后 `FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 ... setup.py build_ext --inplace` 通过。默认 256-thread ptxas：V32 64 registers/0 spill，MODEL1 77 registers/0 spill；手工 128-thread sm_70 编译：V32 64 registers/0 spill，MODEL1 72 registers/0 spill。
- 2026-04-29：shared staging 后 `CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 通过 13/13 correctness；`benchmark/bench_sm70_sparse_decode.py --cases quick --warmup 0 --runs 1` 通过，quick case 约 `347.778/1032.065/643.391/2269.001 us`，max LSE diff `<= 9.53674e-07`，V32/MODEL1 shared bytes 为 `75264/67584`，静态 occupancy 均为 `1 block/SM, 12.5%`。
- 2026-04-29：按用户要求继续推进更大性能化步骤，SM70 sparse decode shared K/V staging 从 FP32 切到 FP16 (`cutlass::half_t`)；`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py -k stages_kv` 通过，`FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace` 通过。默认 256-thread ptxas 保持 V32 64 registers/0 spill、MODEL1 77 registers/0 spill；手工 128-thread sm_70 编译保持 V32 64 registers/0 spill、MODEL1 72 registers/0 spill。
- 2026-04-29：FP16 staging 后 `CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 通过 13/13 correctness；`benchmark/bench_sm70_sparse_decode.py --cases quick --warmup 0 --runs 1` 通过，quick case 约 `356.256/566.171/364.958/1266.427 us`，max LSE diff `<= 9.53674e-07`，V32/MODEL1 shared bytes 降为 `38400/34816`，静态 occupancy 提升到 `2 blocks/SM, 25.0%`。
- 2026-04-29：新增 SM70 sparse decode shared K/V `K_TILE` build-time tuning：`FLASH_MLA_SM70_SPARSE_DECODE_K_TILE` 允许 `{16,32,64}`，默认保持 `32`。`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py -k 'k_tile_is_build_time_tunable or stages_kv'` 通过，随后完成 `K_TILE=16/64` 两档 rebuild + V100 quick scan。`K_TILE=16` shared bytes `19968/18432`、occupancy `50.0%/37.5%`，quick 为 `407.648/370.494/404.926/1146.905 us`；`K_TILE=64` shared bytes `75264/67584`、occupancy `12.5%`，quick 为 `302.240/897.144/565.466/1987.316 us`。`K_TILE=32` 在短/长 quick case 间更均衡，且 shared/occupancy 不走极端，因此作为默认值。
- 2026-04-29：默认 `K_TILE=32` 重新 build 回工作区后复核通过：`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 为 37 passed，`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 为 13/13 correctness，`CUDA_VISIBLE_DEVICES=5 ... benchmark/bench_sm70_sparse_decode.py --cases quick --warmup 0 --runs 1` quick 为 `356.831/567.228/363.740/1265.526 us`，max LSE diff `<= 9.53674e-07`。
- 2026-04-29：SM70 sparse prefill SIMT path 新增 FP16 shared K/V staging 和 `FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE` build-time tuning，允许 `{16,32}`、默认 `32`；QK score 和 PV accumulator 都从 shared tile 读取，为后续 MMA_884 主循环铺路。
- 2026-04-29：默认 `FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE=32` build 通过；SM70 prefill ptxas 为 28 registers、0 spill stores、0 spill loads；手工 128-thread 单文件编译为 30 registers、0 spill stores、0 spill loads。
- 2026-04-29：`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过，38 passed；`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32 correctness。
- 2026-04-29：shared-staging sparse prefill quick benchmark 通过：`--cases quick --warmup 0 --runs 1` 结果约 `286.912/296.192/745.952/1241.568 us`，max LSE diff `<= 9.53674e-07`，shared bytes 约 `33160-37512`，静态 occupancy `2 blocks/SM, 25.0%`。
- 2026-04-29：shared-staging sparse prefill `max_topk` smoke 通过：`S_q=1,S_kv=8192,topk=8192,H_q=64,D_qk=576,attn_sink` 约 `40683.041 us`，max out/lse/max-logits diff `0.000122201/4.76837e-06/0`，shared bytes `69768`，静态 occupancy `1 block/SM, 12.5%`。
- 2026-04-29：SM70 sparse decode 与 sparse prefill 的 QK score 计算升级为 warp-parallel dot reduction：每个 warp 负责一个 staged token，lane 按 `dim += 32` 分摊 512/576 维并用 `__shfl_down_sync` 规约；该步骤仍是 SIMT 路径，不等同于完成 MMA_884。
- 2026-04-29：warp-parallel QK 后默认 `256/32` build 通过；SM70 sparse decode V32/MODEL1 ptxas 均为 64 registers、0 spill stores、0 spill loads、16B stack frame；SM70 sparse prefill ptxas 最大为 29 registers、0 spill stores、0 spill loads。128-thread 探针 build 也通过：decode V32/MODEL1 均 64 registers、0 spill，prefill 最大 32 registers、0 spill。
- 2026-04-29：warp-parallel QK 后 `PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 通过 13/13 correctness；同样导入路径下 `tests/test_flash_mla_sparse_prefill.py` 通过 32/32 correctness。
- 2026-04-29：warp-parallel QK 后 quick benchmark：sparse decode `--cases quick --warmup 0 --runs 1` 约 `263.420/424.186/288.091/982.255 us`，max LSE diff `<= 9.53674e-07`；sparse prefill quick 约 `230.688/204.320/563.776/860.384 us`，max LSE diff `<= 9.53674e-07`。
- 2026-04-29：新增 SM70 sparse prefill opt-in MMA_884 QK-score path：`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=1` build 通过，SM70 prefill ptxas 为 40 registers、0 spill stores、0 spill loads；`tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32 correctness。该路径仅替换 staged K tile 的 QK score 计算，PV accumulator 仍为 SIMT，默认构建仍使用 `warp_simt_qk`。
- 2026-04-29：opt-in `mma884_qk` quick benchmark：`CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=1 ... benchmark/bench_sm70_sparse_prefill.py --cases quick --warmup 0 --runs 1` 通过，结果约 `259.776/259.840/795.232/1163.776 us`，max LSE diff `<= 9.53674e-07`，输出 `qk_path=mma884_qk`、40 registers、0 spill、静态 occupancy `25.0%`。
- 2026-04-30：新增 SM70 sparse prefill opt-in MMA_884 PV path：`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV=1` 用 `mma_m8n8k4_row_row` 在确定性 LSE/sink 后计算 PV row0 输出；与 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=1` 组合后形成 `mma884_qk+mma884_pv` 两阶段 Tensor Core 路径。组合 build ptxas 为 48 registers、0 spill；V100 上 `tests/test_flash_mla_sparse_prefill.py` 通过 32/32，quick benchmark 约 `368.896/380.704/1158.272/1757.184 us`，max LSE diff `<= 9.53674e-07`；默认 `QK=0/PV=0` rebuild 后 correctness 仍通过 32/32。
- 2026-04-29：最终工作区已重建回默认 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=0`；`pytest -q tests/test_sm70_static_contract.py` 通过 41 passed，`git diff --check` 通过。默认 sparse prefill build 中 SM70 ptxas 为 28/29 registers、0 spill，`tests/test_flash_mla_sparse_prefill.py` 通过 32/32；默认 quick benchmark 输出 `qk_path=warp_simt_qk`、29 registers，结果约 `208.128/225.536/566.400/861.792 us`。
- 2026-04-30：本轮最终工作区已重建回默认 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=0 FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV=0`；`/home/z/anaconda3/envs/gptq/bin/python -m pytest -q tests/test_sm70_static_contract.py` 通过 42 passed，默认 `tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32。
- 2026-04-30：新增 SM70 sparse prefill opt-in MMA_884 online alpha：`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=1` 会启用 QK/PV MMA_884，并把 QK tile scores、online softmax 更新、PV 累加放入同一 tile 流水，去掉全量 topk scores shared buffer。online build ptxas 为 48 registers、0 spill；V100 上 `tests/test_flash_mla_sparse_prefill.py` 通过 32/32。quick benchmark 约 `201.344/199.712/586.560/878.272 us`，max LSE diff `<= 9.53674e-07`；`max_topk` smoke (`topk=8192`) 约 `21713.312 us`，max out/lse/max-logits diff 为 `0.000122488/1.90735e-06/9.53674e-07`，shared bytes `39184`，静态 occupancy `2 blocks/SM, 25.0%`。
- 2026-04-30：SM70 sparse prefill online alpha 完成默认路径选择：复扫 `K_TILE=16/32` 后，`K_TILE=16` 虽将 shared bytes 降到 `18576/20624`、occupancy 提到 `50.0-62.5%`，但 `max_topk=8192` 约 `29200.224 us`；`K_TILE=32` quick 约 `200.960/200.032/578.560/875.616 us`，`max_topk=8192` 约 `21693.184 us`，max out/lse/max-logits diff 为 `0.000122488/1.90735e-06/9.53674e-07`。因此默认切到 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=1` + `K_TILE=32`，`=0` 保留为 warp-SIMT 回退。
- 2026-04-30：按新默认值重新执行 `FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 /home/z/anaconda3/envs/gptq/bin/python setup.py build_ext --inplace --force`，编译参数已隐式带 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=1/PV=1/ONLINE=1`，sm70 sparse prefill ptxas 为 48 registers、0 spill；不带 online 环境变量的 `tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32，quick benchmark 输出 `qk_path=mma884_qk`、`pv_path=mma884_pv`、`online_path=mma884_online`，约 `211.584/198.720/585.952/877.504 us`；`max_topk=8192` 约 `21685.345 us`。
- 2026-04-30：新增 SM70 sparse decode opt-in MMA_884 QK-score path：`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=1` 复用 `mma_m8n8k4_row_col` 在 FP16 shared K tile 上计算 QK score，默认仍保持 `warp_simt_qk`，PV accumulator 仍为 SIMT。opt-in build 通过，SM70 sparse decode V32/MODEL1、H64/H128 ptxas 为 80 registers、0 spill、16B stack frame；V100 上 `tests/test_flash_mla_sparse_decoding.py` 通过 13/13。quick benchmark 输出 `qk_path=mma884_qk`，约 `328.260/540.390/354.433/1234.887 us`，max LSE diff `<= 9.53674e-07`，shared bytes 为 `38400/34816`，静态 occupancy `2 blocks/SM, 25.0%`。
- 2026-04-30：新增 SM70 sparse decode opt-in MMA_884 PV path：`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=1` 复用 `mma_m8n8k4_row_row` 在 scores/LSE/sink 后计算 PV row0 输出；与 QK opt-in 组合后形成 `mma884_qk+mma884_pv` 两阶段路径。组合 256-thread build 中 V32 为 64 registers、20B spill stores/16B spill loads，MODEL1 为 64 registers、28B spill stores/loads；128-thread 探针为 V32 95 registers/0 spill、MODEL1 96 registers/0 spill。V100 上 `tests/test_flash_mla_sparse_decoding.py` 通过 13/13；quick benchmark 输出 `qk_path=mma884_qk`、`pv_path=mma884_pv`，约 `649.319/1077.890/736.043/2565.772 us`，max LSE diff `<= 9.53674e-07`。当前慢于默认路径，保持 opt-in。
- 2026-04-30：本轮最终工作区已重建回默认 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=0 FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=0`；默认 build 通过，SM70 sparse decode V32/MODEL1 ptxas 回到 64 registers、0 spill、16B stack frame；`tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 13/13，`tests/test_sm70_static_contract.py` 通过 45 passed，`git diff --check` 通过。默认 quick benchmark 输出 `qk_path=warp_simt_qk`、`pv_path=simt_pv`、64 registers，约 `263.271/415.559/288.004/982.230 us`，max LSE diff `<= 9.53674e-07`。
- 2026-04-30：新增 SM70 sparse decode opt-in MMA_884 online alpha：`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` 会隐式启用 QK/PV MMA_884，并把 tile QK、online softmax/LSE 更新、PV 累加放入同一 tile loop；shared layout 改为 tile scores、5 个 online scalar、FP32 output accumulator、token refs 和 FP16 K/V tile，不再保留全量 scores/LSE/PV 两阶段边界。256-thread online build 中 V32/MODEL1 均为 80 registers、0 spill；128-thread 探针中 V32 为 80 registers、0 spill，MODEL1 为 72 registers、0 spill。V100 上 online `tests/test_flash_mla_sparse_decoding.py` 通过 13/13；quick benchmark 输出 `online_path=mma884_online`、`compute_path=mma884_online`，约 `258.409/444.682/249.544/872.820 us`，max LSE diff `<= 9.53674e-07`，shared bytes 为 `39188/35092`，静态 occupancy `2 blocks/SM, 25.0%`。随后工作区已重建回默认 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=0`，默认 sparse decode correctness 仍通过 13/13，`tests/test_sm70_static_contract.py` 通过 46 passed。
- 2026-04-30：SM70 sparse decode online alpha 完成默认路径选择：复扫 `K_TILE=16/32` 后，`K_TILE=16` 将 shared bytes 降到 `20628/18580`、occupancy 提到 `3 blocks/SM, 37.5%`，long 约 `3566.486/6337.001 us`，但 quick V32 明显退到 `357.980/585.269 us`；`K_TILE=32` quick 约 `250.749/442.712/251.163/872.054 us`，long 约 `3917.942/7315.127 us`，max LSE diff `<= 9.53674e-07`。因此默认切到 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` + `K_TILE=32`，`=0` 保留为 warp-SIMT 回退，`K_TILE=16` 保留为长上下文探针。
- 2026-04-30：SM70 sparse decode 默认 online path 继续推进主循环：256-thread 路径新增 register output accumulator，移除 512 维 FP32 shared accumulator，并把 thread0 串行 tile softmax 改为 parallel online softmax reduction。默认 build ptxas：V32 `128 registers、28B spill`，MODEL1 `97 registers、0 spill`；128-thread fallback 探针为 V32 `72 registers、4B/8B spill`，MODEL1 `80 registers、8B/4B spill`。shared bytes 降到 `37172/33076`，静态 occupancy 仍为 `2 blocks/SM, 25.0%`；V100 correctness 通过 13/13，quick 为 `244.773/430.044/237.506/857.661 us`，long 为 `3846.971/7086.606 us`，max LSE diff `<= 9.53674e-07`。该步骤有小幅收益但暴露 V32 spill；下一条 hybrid accumulator 记录已处理该问题。
- 2026-04-30：SM70 sparse decode 默认 online path 继续推进到 model-specific hybrid accumulator：V32 回退 shared output accumulator 以消掉 register spill，MODEL1 继续使用 256-thread register output accumulator；两者仍共享 MMA_884 online QK/PV 与 parallel online softmax reduction。fresh default build ptxas：V32 `80 registers、0 spill`，MODEL1 `97 registers、0 spill`；shared bytes 为 `39220/33076`，静态 occupancy 仍为 `2 blocks/SM, 25.0%`。V100 correctness 通过 13/13，quick 为 `233.473/408.952/237.247/858.323 us`，long 为 `3667.897/7065.823 us`，max LSE diff `<= 9.53674e-07`。这一步保留 MODEL1 register accumulator 收益，同时把上一版 V32 `128 registers、28B spill` 压回无 spill。
- 2026-04-30：SM70 sparse decode 默认 online path 完成 V32-only runtime adaptive K tile 双 kernel dispatch：默认仍以 `FLASH_MLA_SM70_SPARSE_DECODE_K_TILE=32` 构建，短 topk 走 K32 online kernel；V32 `topk + extra_topk >= 512` 时调度 K16 online kernel，把 long V32 shared bytes 从 `39220` 降到 `20660`、静态 occupancy 从 `25.0%` 提到 `37.5%`，long V32 约 `3243.768 us`，max out/lse diff `6.10352e-05/9.53674e-07`。MODEL1 因 97 registers 限制仍保持 K32，long 约 `7094.224 us`，避免全局 K16 的退化。fresh build 中 V32 K16/K32 均为 `80 registers、0 spill`，MODEL1 K32 为 `97 registers、0 spill`；sparse decode correctness 通过 13/13，quick 仍全部输出 `runtime_k_tile=32`。
- 2026-04-30：新增 SM70 dense decode opt-in MMA_884 online alpha：`FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE=1` 会把 QK `mma_m8n8k4_row_col`、online softmax 更新和 PV `mma_m8n8k4_row_row` 放入同一 K tile loop；`FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE` 允许 `{0,16,32,64}`，其中 `0` 为 runtime-auto。随后 dense online 完成 256-thread register output accumulator 与 parallel tile softmax reduction 两个主路径优化。fresh K_TILE=64 build 为 113 registers、0 spill，shared bytes `65844/74036`，静态 occupancy `1 block/SM, 12.5%`；V100 quick 为 `247.136/200.288/205.056/195.264 us`，long `S_k=16384` 为 `633.728/693.504 us`。同一默认 scalar score-cache build 通过 dense FP16 correctness，quick 为 `209.504/178.464/160.416/154.656 us`，long `S_k=16384` 为 `400.672/476.864 us`，因此 dense online 仍保持 opt-in alpha，不切默认。
- 2026-04-30：完成 dense/sparse decode 共享 Volta MMA_884 attention primitive 整理：新增 `csrc/sm70/common/mma_884_attention.h`，把 QK/PV fragment accumulate、row0 score 写回、probability fragment 装载和 row0 PV 输出映射从 dense/sparse `.cuh` 中抽到公共 helper；dense 与 sparse decode 均通过该 helper 调用底层 `mma_m8n8k4_row_col/row_row`。验证：`tests/test_sm70_static_contract.py` 通过 57 passed；默认 SM70 build 通过，dense `64 registers/0 spill`，sparse decode V32 `80 registers/0 spill`、MODEL1 `97 registers/0 spill`；V100 上 `tests/test_flash_mla_sparse_decoding.py` 通过 13/13，`tests/test_flash_mla_dense_decoding.py --dtype fp16` 通过，sparse quick benchmark 仍输出 `compute_path=mma884_online`、`runtime_k_tile=32`。
- 2026-04-30：dense MMA_884 默认路径选择推进到 `K_TILE=0` runtime-auto dispatch。`FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE=0` 会在 launch 侧选择 online kernel；当前 auto 保守选择 K64，显式 K16/K32/K64 仍可用于 build-time 探针。V100 online auto correctness 通过，quick 为 `113.664/128.000/156.331/163.157 us`，long `S_k=16384` 为 `685.739/652.629 us`；K16 长上下文 D512 探针虽把 occupancy 提到 `25.0%`，但 `S_k=4096/8192/16384` 退到 `281.941/532.139/1007.957 us`。最终重建回默认 scalar score-cache，dense FP16 correctness 通过，quick 为 `98.304/98.645/98.304/108.885 us`，long `S_k=16384` 为 `349.525/383.317 us`。结论：保留 runtime-auto/显式 K-tile 机制，但 dense 默认继续使用 scalar score-cache，online 保持 opt-in alpha。
- 2026-04-30：dense CTA launch contract 已显式整理为 `DenseCtaTileCoord{q_seq_per_hk_tile_idx, kv_head_idx, partition_idx}`，把旧任务中的 q-head tile 字面语义固定到当前 API 的 `q_seq_per_hk = S_q * H_q / H_kv` tile 上，batch/split 继续由 scheduler partition 在 CTA 内遍历。验证：先让 `tests/test_sm70_static_contract.py -k dense_cta_tile_contract` 失败在缺少新契约，再补实现与文档；默认 SM70 build 通过，dense sm_70 ptxas 仍为 `64 registers、0 spill`；`CUDA_VISIBLE_DEVICES=6 /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_dense_decoding.py --dtype fp16` 通过。
- 2026-04-30：SM70 sparse decode total topk 上限从 2048 提到 8192，并同步 API error、benchmark、runtime correctness 子集和文档契约。默认 SM70 build 通过，sparse decode V32 `80 registers、0 spill`，MODEL1 `97 registers、0 spill`；`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/data/apps/FlashMLA /home/z/anaconda3/envs/gptq/bin/python tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 15/15 correctness，新增覆盖 `topk=8192 + attn_sink` 和 MODEL1 `4096+4096 + topk_length + extra_kv + extra_topk_length + attn_sink`。
- 2026-04-30：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/mnt/data/apps/FlashMLA /home/z/anaconda3/envs/gptq/bin/python benchmark/bench_sm70_sparse_decode.py --cases max_topk --warmup 0 --runs 1` 在 V100 上通过。V32 `8192` case 输出 `splitkv_us=11661.135`、`combine_us=50.465`、`features=v32+attn_sink`、`runtime_k_tile=16`、`online_path=mma884_online`；MODEL1 `4096+4096` case 输出 `splitkv_us=11608.072`、`combine_us=46.466`、`features=model1+attn_sink+topk_length+extra_kv+extra_topk_length`、`runtime_k_tile=32`。vLLM gate 和端到端 smoke 暂未改动，按当前指令留作后续任务。

---

## 任务 0：建立基线与约束

**Files:**

- Read: `setup.py`
- Read: `csrc/api/common.h`
- Read: `csrc/api/dense_decode.h`
- Read: `csrc/api/sparse_decode.h`
- Read: `tests/test_flash_mla_dense_decoding.py`
- Read: `tests/test_flash_mla_sparse_decoding.py`
- Read: `/mnt/data/apps/flash-attention-v100/docs/volta.md`
- Read: `/mnt/data/apps/lmdeploy/src/turbomind/kernels/core/mma.h`
- Read: `/mnt/data/apps/lmdeploy/src/turbomind/kernels/attention/attention_config.h`
- Read: `/mnt/data/apps/lmdeploy/src/turbomind/kernels/attention/impl_884.h`
- Read: `/mnt/data/apps/lmdeploy/src/turbomind/kernels/attention/impl_884_decode.h`
- Read: `/mnt/data/apps/lmdeploy/src/turbomind/kernels/attention/mainloop_sm70.h`

- [x] 记录当前 `git status --short`，确认只在 FlashMLA 工作区改动。
- [x] 确认目标 GPU 是 `major=7, minor=0`，优先按 V100 设计。
- [x] 确认 CUDA 工具链能离线编译 `sm_70`。
- [x] 记录当前 FlashMLA SM90/SM100 测试命令，作为回归基线。

**验收：**

- 形成一页本地记录：当前 commit、CUDA 版本、PyTorch 版本、GPU 名称、目标 dtype。

## 任务 1：增加 SM70 构建和架构识别

**Files:**

- Modify: `setup.py`
- Modify: `csrc/api/common.h`
- Test: `tests/test_sm70_static_contract.py`

- [x] 在 `setup.py` 增加 `FLASH_MLA_ENABLE_SM70` 环境变量。
- [x] 当 `FLASH_MLA_ENABLE_SM70=1` 时追加 `-gencode arch=compute_70,code=sm_70`。
- [x] 在 `csrc/api/common.h` 增加 `Arch::is_sm70()`。
- [x] 增加静态测试，检查 `get_arch_flags()` 在启用 SM70 时包含 `sm_70`。
- [x] 保持默认 SM90/SM100 行为不变。

**验证命令：**

```bash
FLASH_MLA_ENABLE_SM70=1 FLASH_MLA_DISABLE_SM100=1 python -m pip install -v . --no-build-isolation
```

**验收：**

- build log 中出现 `arch=compute_70,code=sm_70`。
- 不启用 `FLASH_MLA_ENABLE_SM70` 时原构建参数不变。

## 任务 2：新增 SM70 公共 primitive

**Files:**

- Create: `csrc/sm70/common/mma_884.h`
- Create: `csrc/sm70/common/layout.h`
- Create: `csrc/sm70/common/load_store.h`
- Create: `csrc/sm70/common/softmax.h`
- Create: `csrc/sm70/common/bf16_cast.h`
- Create: `csrc/sm70/common/fp8_dequant.h`

- [x] 从 TurboMind 的 `mma_m8n8k4_row_col` / `mma_m8n8k4_row_row` 提炼最小 inline PTX wrapper，并写明来源。
- [x] 实现 lane id、quadpair 位置映射、`Array<half, 4>` / `Array<float, 8>` fragment 类型。
- [x] 实现 128-bit global load、shared store/load、padding/swizzle helper。
- [x] 实现 row-wise online softmax helper，输出 `row_max`、`row_sum`、`lse`。
- [x] 实现 `bf16 -> fp16` 显式转换 helper。
- [x] 实现 `fp8_e4m3 + scale -> fp16` helper，先覆盖 V32 的 512 NoPE + 4 scales。

**验收：**

- 这些 header 可以在 `sm_70` 编译单元中 include。
- `mma_884.h` 不被 SM90/SM100 编译路径 include。

## 任务 3：实现 SM70 dense decode FP16 alpha

**Files:**

- Create: `csrc/sm70/decode/dense/config.h`
- Create: `csrc/sm70/decode/dense/splitkv_mla_sm70.h`
- Create: `csrc/sm70/decode/dense/splitkv_mla_sm70.cuh`
- Create: `csrc/sm70/decode/dense/instantiations/fp16.cu`
- Modify: `setup.py`
- Modify: `csrc/api/dense_decode.h`
- Test: `tests/test_flash_mla_dense_decoding.py`

- [x] 定义 dense SM70 config：`H_TILE in {4,8,16}`、`K_TILE=64`、`D_TILE=64`、`D_qk in {512,576}`、`D_v=512`。
- [x] 实现 FP16 dense kernel，输入输出使用 `DenseAttnDecodeParams`。
- [x] 每个 CTA 处理一个 `(scheduler partition, kv_head, q_seq_per_hk tile)`，其中 `q_seq_per_hk = S_q * H_q / H_kv` 保留原 dense API 的 q-head tile 语义，batch/split 由 scheduler partition 在 CTA 内遍历。
- [x] 在 scalar alpha 中每个 CTA 处理一个 `(q_seq, kv_head, scheduler partition)` 并遍历 partition 内请求。
- [x] 将 scalar alpha 的 launch shape 调整为 build-time tunable `dense::H_TILE` q-seq tile。
- [x] 根据 `block_table` 和 `cache_seqlens` 遍历 KV page。
- [x] 新增 opt-in dense MMA_884 online alpha：QK 使用 `mma_m8n8k4_row_col`，PV 使用 `mma_m8n8k4_row_row`，并用 online softmax tile loop 写回 `out/lse` 或 `o_accum/lse_accum`。
- [ ] 将 dense MMA_884 online alpha 性能化到可替换默认 scalar score-cache 路径。
- [x] 将 dense MMA_884 online path 的 parallel online-softmax tile update 接到 `csrc/sm70/common/softmax.h`，与 sparse decode online path 共用同一份 warp-scratch block reduction。
- [x] 增加 dense MMA_884 online `K_TILE=0` runtime-auto dispatch，记录 K16/K64 探针结果，并完成本轮默认路径选择：继续保留 scalar score-cache 默认、online 保持 opt-in。
- [x] 写出 `o_accum`、`lse_accum`，维持现有 combine 所需布局。
- [x] 在 `dense_decode.h` 中 SM70 + FP16 分派到新 kernel。
- [x] BF16 在此任务中先给出清晰错误。

> 进展记录（2026-04-28）：已落地 correctness-first FP16 dense alpha，当前 kernel 直接写最终 `out/lse` 并在 SM70 上固定 `num_sm_parts=1`，用于验证 API/shape/数值路径；MMA 主循环、`o_accum/lse_accum`、split-K/combine 仍未完成。
> 进展记录（2026-04-29）：dense alpha 已接入 scheduler，kernel grid 第三维使用 `params.num_sm_parts`。未切分请求仍直接写最终 `out/lse`，切分请求写 `o_accum/lse_accum` 并由现有 combine 合并；当前仍是 scalar FP32 QK/PV，尚未进入 Volta MMA 主循环或多 q-head tile。
> 进展记录（2026-04-29）：dense alpha 已改为 build-time tunable `dense::H_TILE` 的 q-seq tile launch，grid.x 为 `ceil(q_seq_per_hk / H_TILE)`，CTA 内顺序处理 tile 内有效 q_seq。该步骤为后续真正 q-head tile/MMA 主循环铺路；当前仍是逐 row scalar 计算。
> 进展记录（2026-04-30）：dense kernel 的 CTA launch contract 已显式整理为 `DenseCtaTileCoord{q_seq_per_hk_tile_idx, kv_head_idx, partition_idx}`。旧任务中的 `(batch, kv_head, q_head_tile, split)` 字面目标按当前 API 收敛为 `(scheduler partition, kv_head, q_seq_per_hk tile)`：`q_seq_per_hk` 已折叠 `S_q * H_q / H_kv`，因此保留 q-head tile 语义；batch/split 继续由 `DecodingSchedMeta` partition 驱动，避免破坏现有 split-K/combine 数据流。
> 进展记录（2026-04-30）：dense decode 新增 `FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE=1` opt-in online alpha，shared layout 为 tile scores、5 个 online scalar、FP32 output accumulator 和 FP16 K/V tile；每个 K tile 先用 `mma_m8n8k4_row_col` 计算 QK row0 scores，再在线更新 `row_max/row_sum`、缩放 output accumulator，并用 `mma_m8n8k4_row_row` 累加 PV。`FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE` 允许 `{0,16,32,64}`，其中 `0` 为 runtime-auto。随后新增 256-thread register output accumulator 路径，去掉 512 维 FP32 shared output accumulator，并把 tile softmax 更新改为 parallel tile softmax reduction。fresh K_TILE=64 build 为 113 registers、0 spill；V100 quick 为 `247.136/200.288/205.056/195.264 us`，long `S_k=16384` 为 `633.728/693.504 us`。同一默认 scalar score-cache build 的 quick 为 `209.504/178.464/160.416/154.656 us`，long `S_k=16384` 为 `400.672/476.864 us`，因此保留为 opt-in alpha，不切默认。
> 进展记录（2026-04-30）：dense MMA_884 online 的 parallel online-softmax tile update 已从 dense 本地 helper 移到 `csrc/sm70/common/softmax.h`，并以 `compute_online_softmax_tile_parallel<false>` 调用；后续 dense 默认化继续集中在 register/shared 占用和 K tile 性能，不再维护一份与 sparse decode 分叉的 row-max/row-sum 更新逻辑。
> 进展记录（2026-04-30）：dense MMA_884 online 已新增 `FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE=0` runtime-auto dispatch，launch 侧保留 K16/K32/K64 三个 specialization。当前 auto 选择 K64；V100 online auto correctness 通过，quick 为 `113.664/128.000/156.331/163.157 us`，long `S_k=16384` 为 `685.739/652.629 us`。K16 长上下文 D512 探针虽将 shared bytes 降到 `16500`、occupancy 提到 `25.0%`，但 `S_k=4096/8192/16384` 退到 `281.941/532.139/1007.957 us`；默认 scalar score-cache 重建后 correctness 通过，quick 为 `98.304/98.645/98.304/108.885 us`，long `S_k=16384` 为 `349.525/383.317 us`。因此本轮默认路径选择结论是保留 scalar 默认，online auto 作为 opt-in 调参机制。

**验证命令：**

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_flash_mla_dense_decoding.py --dtype fp16
```

**验收：**

- V100 上至少通过小规模 dense correctness 子集：`B in {1,2}`、`S_q in {1,2}`、`S_k in {20,140,4096}`、`H_q/H_kv` 的代表组合。
- SM90/SM100 dense 原路径不回归。

## 任务 4：split-K、combine 和性能调参

**Files:**

- Modify: `csrc/sm70/decode/dense/splitkv_mla_sm70.cuh`
- Possibly Modify: `csrc/smxx/decode/combine/combine.cu`
- Test: `tests/test_flash_mla_dense_decoding.py`
- Create: `benchmark/bench_sm70_dense_decode.py`

- [x] 验证 `csrc/smxx/decode/get_decoding_sched_meta` 在 SM70 构建和运行可用。
- [x] 验证现有 `smxx::decode::run_flash_mla_combine_kernel` 在 SM70 上可用。
- [x] 如果 combine 因 PDL 或编译目标失败，增加 SM70-safe compile guard。
- [x] 解除 SM70 dense alpha 的单 split 限制，验证真实 split-K/combine long case。
- [x] 记录 `ptxas` register count、spill、shared memory usage。
- [x] 记录当前 q-seq tile 配置：`H_TILE=4`、`CTA threads=256`。
- [x] 增加 build-time tuning 开关：`FLASH_MLA_SM70_H_TILE`、`FLASH_MLA_SM70_CTA_THREADS`。
- [x] 扫描 `H_TILE in {4,8,16}`、`CTA threads in {128,256}`。
- [x] 用 `K_TILE=64` shared score-cache + online softmax/PV accumulator 去掉 scalar PV 阶段重复 QK。
- [x] 使用 `q_seq_per_hk > 1` 的 score-cache benchmark 重扫 `H_TILE` 敏感性。
- [x] 输出 dense decode benchmark 表：us、TFLOPS、GB/s、registers、spill、静态 occupancy。
- [x] 扩展 dense long benchmark 到验收集合 `S_k={4096,8192,16384}` 并在 V100 上跑通 smoke。
- [x] 对 dense MMA_884 online alpha 扫描 `K_TILE=32/16/64`，记录 shared bytes、静态 occupancy、quick/long correctness/perf，并确认当前不切默认。

> 进展记录（2026-04-29）：真实 split-K/combine long case 已通过，`S_k=4096` 下 `D=512` 约 `4829.920 us`，`D=576` 约 `5377.920 us`。ptxas 当前为 48 registers、0 spill stores、0 spill loads；后续性能任务进入 Volta MMA 主循环。
> 进展记录（2026-04-29）：已在 score-cache streaming 优化前通过 build-time tuning 扫描六组 `H_TILE x CTA threads`。quick case 结果如下，单位 us，case 为 `B=1,S_q=1,H_q=1,H_kv=1`：
>
> | H_TILE | CTA threads | S_k=20 D512 | S_k=20 D576 | S_k=140 D512 | S_k=140 D576 |
> |---:|---:|---:|---:|---:|---:|
> | 4 | 128 | 2785.472 | 3012.736 | 8290.528 | 9332.032 |
> | 4 | 256 | 1683.520 | 1803.744 | 4739.200 | 5253.376 |
> | 8 | 128 | 2713.216 | 2979.520 | 8321.152 | 9272.384 |
> | 8 | 256 | 1596.224 | 1741.920 | 4812.448 | 5269.376 |
> | 16 | 128 | 2724.064 | 3003.168 | 8291.264 | 9288.224 |
> | 16 | 256 | 1600.736 | 1736.576 | 4722.816 | 5296.448 |
>
> 结论：128 threads 明显较慢；`8/256` 与 `16/256` 长 K 同档，短 K 下 `8/256` 略优，因此该阶段临时保持 `H_TILE=8`、`CTA threads=256`。注意该 tuning 通过编译宏生效，切换配置需要重新 build extension。该 scan 的 `H_q=1` 不充分覆盖 `H_TILE` 敏感性，score-cache 后需要增加 `q_seq_per_hk > 1` cases。
> 进展记录（2026-04-29）：score-cache streaming 版本已通过默认 `8/256` build/correctness/benchmark。quick case：`S_k=20,D512` `196.352 us`，`S_k=20,D576` `170.656 us`，`S_k=140,D512` `147.744 us`，`S_k=140,D576` `167.904 us`；long case：`S_k=4096,D512` `127.386 us`，`S_k=4096,D576` `116.941 us`。ptxas 从 48 registers 提升到 64 registers，但仍为 0 spill；shared memory 从 1024B 提升到 1280B。
> 进展记录（2026-04-29）：新增 `--cases tile` 后重扫 `H_q=8,H_kv=1` 的 H tile 敏感性，CTA threads 固定 256：
>
> | H_TILE | S_k=140 D512 | S_k=140 D576 | S_k=4096 D512 | S_k=4096 D576 |
> |---:|---:|---:|---:|---:|
> | 4 | 311.296 | 330.547 | 445.440 | 487.219 |
> | 8 | 592.896 | 634.266 | 613.376 | 661.094 |
> | 16 | 591.258 | 638.771 | 616.243 | 661.914 |
>
> 结论：`4/256` 在 q-seq tile case 上明显更好，且 `H_q=1` long case 也略快于 `8/256`，因此当前默认切换为 `H_TILE=4`、`CTA threads=256`。最终默认 quick：`S_k=20,D512` `215.232 us`，`S_k=20,D576` `186.464 us`，`S_k=140,D512` `172.800 us`，`S_k=140,D576` `166.080 us`；最终默认 long：`S_k=4096,D512` `116.326 us`，`S_k=4096,D576` `112.230 us`。
> 进展记录（2026-04-29）：`bench_sm70_dense_decode.py --cases long` 已扩展到 `S_k={4096,8192,16384}`，并通过 V100 smoke（`warmup=0,runs=1`，用于可运行性验收，不替代稳定性能基线）。本次结果：`S_k=4096` D512/D576 `268.832/202.624 us`，`S_k=8192` D512/D576 `264.384/276.512 us`，`S_k=16384` D512/D576 `423.040/455.328 us`；max LSE diff `<= 9.53674e-07`，默认 `4/256` 静态 occupancy `4 blocks/SM, 50.0%`。

**验收：**

- 长上下文 dense case 可运行：`S_k in {4096,8192,16384}`。
- 无明显 local memory spill；若有 spill，文档记录采用的降 H tile 策略。

## 任务 5：BF16 输入兼容

**Files:**

- Create: `csrc/sm70/decode/dense/instantiations/bf16_compat.cu`
- Modify: `csrc/api/dense_decode.h`
- Test: `tests/test_flash_mla_dense_decoding.py`

- [x] 明确 BF16 策略：SM70/Volta 不支持 BF16 dense decode，保持 FP16-only。
- [x] 不再为 dense BF16 输入分配 FP16 scratch 或 kernel 内转换；该 compat 路径按用户决策关闭。
- [x] 不再比较 dense BF16 compat 与 PyTorch FP32 reference 的误差；验收改为 FP16-only 快速失败。
- [x] 如果误差不可接受，保留 FP16-only 并在错误信息中写清楚。

> 进展记录（2026-04-28）：用户明确 SM70 支持 FP16、不支持 BF16；因此 BF16 scratch/compat 路径不再作为当前目标，SM70 BF16 输入应快速失败并给出可诊断错误。

**验收：**

- `python tests/test_flash_mla_dense_decoding.py --dtype bf16` 的指定子集通过，或明确关闭 BF16 compat 并给出可诊断错误。

## 任务 6：实现 SM70 sparse V32 FP8 dequant decode alpha

**Files:**

- Create: `csrc/sm70/decode/sparse_fp8/config.h`
- Create: `csrc/sm70/decode/sparse_fp8/dequant.h`
- Create: `csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.h`
- Create: `csrc/sm70/decode/sparse_fp8/splitkv_mla_sm70_sparse.cuh`
- Create: `csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu`
- Modify: `setup.py`
- Modify: `csrc/api/sparse_decode.h`
- Test: `tests/test_flash_mla_sparse_decoding.py`
- Test: `tests/quant.py`

- [x] 增加 `Decode_Sm70_Fp8_Dequant_Impl`，支持 `HEAD_64`、`HEAD_128`、`HEAD_DIM_576`、`V32_KVCACHE_FORMAT`、`ATTN_SINK`。
- [x] 按 `indices` gather topk token，处理 `-1` invalid entry。
- [x] 实现 V32 656-byte token layout dequant：512 FP8 NoPE、4 FP32 scales、64 BF16 RoPE。
- [x] Dequant 后写入 SM70 计算友好的 shared layout。
- [x] 复用 dense QK/PV 核心，形成完整 sparse decode MMA_884 QK/PV 主循环。
- [x] 在 no-split SM70 sparse alpha 中镜像写入 split-compatible `o_accum/lse_accum`，固定 raw accumulator 与 LSE log2 布局契约。
- [x] 解除 SM70 sparse alpha 的单 split 限制，按 scheduler partition 写 split accumulator 并由现有 combine 合并。
- [x] SM70 sparse alpha 对未声明 sparse feature 保持 required-feature 检查。
- [x] 将 QK score 计算从每 token 单线程串行扫维度改为 warp-parallel dot reduction，继续复用 FP16 shared K/V staging 和现有 softmax/PV accumulator。
- [x] 新增 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK` opt-in 两阶段验证路径，用 Volta `mma_m8n8k4_row_col` 计算 staged K tile 的 QK score；当前默认已由 online 路径隐式启用 QK/PV MMA_884，`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=0` 仍可回退到 `warp_simt_qk+simt_pv`。
- [x] 新增 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV` opt-in 路径，用 Volta `mma_m8n8k4_row_row` 在 LSE/sink 后计算 PV row0 输出；与 QK opt-in 组合后形成 `mma884_qk+mma884_pv` 两阶段路径。
- [x] 新增 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE` online alpha，在现有 FP16 shared K/V staging 基础上把 QK、online softmax/LSE 更新和 PV 放进同一 tile 流水，减少两阶段 scores/LSE/PV shared 边界。
- [x] 在 sparse decode online alpha 基础上复扫 `K_TILE=16/32` 的 quick/long 矩阵，并将 SM70 sparse decode 默认路径切到 `256/32/mma884_online`。
- [x] 将 sparse decode 默认 online path 的 256-thread output accumulator 下沉到 register fragments，并把 tile online softmax 更新改为 parallel online softmax reduction；随后改成 V32 shared / MODEL1 register 的 model-specific hybrid accumulator，消掉 V32 spill。
- [x] 将 sparse decode online path 的 parallel online-softmax tile update 接到 `csrc/sm70/common/softmax.h`，使用 `compute_online_softmax_tile_parallel<true>` 保留 invalid token `-inf` 跳过语义，并与 dense decode 共用 warp-scratch reduction。
- [x] 在 sparse decode 默认 online path 中加入 V32-only runtime adaptive K tile 双 kernel dispatch：短 topk 继续走 `SPARSE_K_TILE=32`，V32 `topk + extra_topk >= 512` 时调度 `K_TILE=16` online kernel，MODEL1 保持 `K_TILE=32` 以避免 register-limited 路径退化。

> 进展记录（2026-04-28）：已落地 correctness-first SM70 sparse FP8 alpha。当前 kernel 使用 scalar FP32 QK/PV、shared score buffer，并直接写最终 `out/lse`；最初支持 V32/MODEL1、`topk + extra_topk <= 2048`、`H_q=64/128`、`D_qk=576/512`、`topk_length`、`extra_kv`、`extra_topk_length`、`attn_sink`，但尚未进入 Volta MMA/shared-K tiled 主循环，也未写 `o_accum/lse_accum` 或走 split-K combine。后续 2026-04-30 已将当前上限提升到 `topk + extra_topk <= 8192`。
> 进展记录（2026-04-29）：SM70 sparse alpha 的 no-split 行已开始镜像 split accumulator 布局：最终 `out` 仍保持 sink 后 BF16 输出，`o_accum` 写 sink 前 FP32 raw accumulator，`lse_accum` 写 log2 LSE，供后续真正 split-K/combine 复用。该步骤在当时不等同于完成 split-K：该阶段 `Decode_Sm70_Fp8_Dequant_Impl::get_meta()` 仍返回 `num_sm_parts=1`，真正多 split 调度和 combine 仍待实现。验证：build 通过，ptxas 为 40 registers、0 spill；`tests/test_flash_mla_sparse_decoding.py` 通过 13/13。
> 进展记录（2026-04-29）：SM70 sparse alpha 已接入 `smxx::decode` scheduler/combine，`get_meta()` 不再固定 `num_sm_parts=1`；kernel 按 `(s_q, h_q, partition)` launch，partition 内根据 `DecodingSchedMeta` 处理 topk block 范围。未切分请求继续直接写最终 `out/lse` 并镜像 accumulator；切分请求写 `o_accum/lse_accum`，由 `run_flash_mla_combine_kernel<bf16>` 合并。当前仍是 scalar FP32 QK/PV + shared score/token-ref buffer，不等同于完成 shared layout 或 Volta MMA_884。验证：build 通过；`tests/test_flash_mla_sparse_decoding.py` 通过 13/13；ptxas V32 为 48 registers、0 spill，MODEL1 为 48 registers、12B spill。
> 进展记录（2026-04-29）：SM70 sparse decode 已新增 `SPARSE_K_TILE=32` 的 shared K/V staging，按 tile 将 V32/MODEL1 dequant 后的 K/V 写入计算友好的 shared layout；QK score 与 PV accumulator 均改为从 shared tile 读取。随后 shared staging 从 FP32 切到 FP16 (`cutlass::half_t`)，把 V32/MODEL1 shared bytes 从 `75264/67584` 降到 `38400/34816`，静态 occupancy 从 `1 block/SM, 12.5%` 提升到 `2 blocks/SM, 25.0%`。该步骤完成 FP16 shared-layout 数据流，但仍不是最终 MMA_884 主循环，也未复用 dense MMA 核心。验证：build 通过；`tests/test_flash_mla_sparse_decoding.py` 通过 13/13；quick benchmark 通过。默认 256-thread ptxas：V32 64 registers/0 spill，MODEL1 77 registers/0 spill。
> 进展记录（2026-04-29）：SM70 sparse decode 的 QK score 已改为 warp-parallel dot reduction。每个 warp 处理一个 staged token，lane 按 `dim += 32` 分摊 QK 维度并用 `__shfl_down_sync` 规约，替代旧的每 token 单线程串行扫 512/576 维。该路径仍是 SIMT scalar，不是 MMA_884；PV accumulator 仍复用 FP16 shared K/V tile。默认 256-thread build 中 V32/MODEL1 均为 64 registers、0 spill stores、0 spill loads、16B stack frame；`tests/test_flash_mla_sparse_decoding.py` 通过 13/13，quick benchmark 约 `263.420/424.186/288.091/982.255 us`。
> 进展记录（2026-04-30）：新增 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=1` opt-in QK-score 路径，复用 `csrc/sm70/common/mma_884.h` 的 `mma_m8n8k4_row_col`，在 FP16 shared K tile 上每个 warp group 计算 8x8x4 QK fragment 并写回 row0 score。该路径只替换 QK score 计算，softmax/sink 与 PV accumulator 保持现有 SIMT 语义，因此不等同于完整 QK/PV MMA_884 主循环。opt-in build 中 V32/MODEL1、H64/H128 sm_70 ptxas 均为 80 registers、0 spill stores、0 spill loads、16B stack frame；`tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 13/13；quick benchmark 输出 `qk_path=mma884_qk`，约 `328.260/540.390/354.433/1234.887 us`，max LSE diff `<= 9.53674e-07`。
> 进展记录（2026-04-30）：新增 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=1` opt-in PV 路径，复用 `csrc/sm70/common/mma_884.h` 的 `mma_m8n8k4_row_row`，在确定性 scores/LSE/sink 之后把 probability tile 与 FP16 shared V tile 做 PV row0 输出。`FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=1 FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=1` 组合后，QK 和 PV 都走 Volta MMA_884，但中间仍保留 scores/LSE 边界，因此这是两阶段验证路径，不等同于完整 fused/online 主循环。组合 256-thread build 中 V32 为 64 registers、20B spill stores、16B spill loads、16B stack frame，MODEL1 为 64 registers、28B spill stores、28B spill loads、24B stack frame；128-thread 探针 build 中 V32/MODEL1 分别为 95/96 registers、0 spill。`tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 13/13；quick benchmark 输出 `qk_path=mma884_qk`、`pv_path=mma884_pv`，约 `649.319/1077.890/736.043/2565.772 us`，max LSE diff `<= 9.53674e-07`。当前组合慢于 warp-SIMT 回退和 online 默认候选，保留为 opt-in 验证路径。
> 进展记录（2026-04-30）：新增 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` online alpha。该路径隐式启用 QK/PV MMA_884，shared memory 改为 tile scores、5 个 online scalar、FP32 output accumulator、token refs 和 FP16 K/V tile；每个 K tile 内先用 `mma_m8n8k4_row_col` 生成 tile scores，再在线更新 `row_max/row_sum`、缩放 output accumulator，随后用 `mma_m8n8k4_row_row` 累加 PV。online 256-thread build 中 V32/MODEL1 均为 80 registers、0 spill；128-thread 探针中 V32 为 80 registers、0 spill，MODEL1 为 72 registers、0 spill。`tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 13/13；quick benchmark 输出 `online_path=mma884_online`、`compute_path=mma884_online`，约 `258.409/444.682/249.544/872.820 us`，max LSE diff `<= 9.53674e-07`，shared bytes `39188/35092`，静态 occupancy `2 blocks/SM, 25.0%`。随后复扫 `K_TILE=16/32` 的 quick/long 矩阵，`K_TILE=16` long 更快但 short V32 明显退化，最终默认切到 `FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=1` + `K_TILE=32`，`=0` 保留为 warp-SIMT 回退。
> 进展记录（2026-04-30）：默认 sparse decode online path 已新增 256-thread register output accumulator 与 parallel online softmax reduction；随后改为 model-specific hybrid accumulator，V32 使用 shared output accumulator，MODEL1 使用 register output accumulator。fresh default build 中 V32 为 `80 registers、0 spill`，MODEL1 为 `97 registers、0 spill`；shared bytes 为 `39220/33076`，静态 occupancy 仍为 `2 blocks/SM, 25.0%`。V100 correctness 通过 13/13；quick 为 `233.473/408.952/237.247/858.323 us`，long 为 `3667.897/7065.823 us`。这一步是真正默认主路径优化：保留 MODEL1 register accumulator 收益，并把上一版 V32 `128 registers、28B spill` 压回无 spill。
> 进展记录（2026-04-30）：sparse decode MMA_884 online 的 parallel online-softmax tile update 已从 sparse 本地 helper 移到 `csrc/sm70/common/softmax.h`，并以 `compute_online_softmax_tile_parallel<true>` 调用。`<true>` 保留 sparse invalid token 的 `-inf` 跳过语义，dense decode 使用 `<false>` 的连续 KV tile 语义；两条 decode path 后续复扫 CTA/K tile 时共用同一份 row max、row sum 和 online scale 更新实现。
> 进展记录（2026-04-30）：默认 sparse decode online path 新增 V32-only runtime adaptive K tile 双 kernel dispatch。默认 build 仍保留 `FLASH_MLA_SM70_SPARSE_DECODE_K_TILE=32`，短 topk 走 K32 online kernel；V32 `topk + extra_topk >= 512` 时调度 K16 online kernel，MODEL1 继续使用 K32。fresh build 中 V32 K16/K32 均为 `80 registers、0 spill`，MODEL1 K32 为 `97 registers、0 spill`；`tests/test_flash_mla_sparse_decoding.py` 在 V100 上通过 13/13。quick benchmark 全部 `runtime_k_tile=32`，约 `234.168/407.472/237.911/858.075 us`；long V32 `runtime_k_tile=16`，shared bytes `20660`、occupancy `37.5%`、约 `3243.768 us`；long MODEL1 保持 `runtime_k_tile=32`，shared bytes `33076`、occupancy `25.0%`、约 `7094.224 us`。一次全局 runtime tile=16 探针曾让 MODEL1 long 退到约 `8963.554 us`，因此默认 dispatch 限定为 V32。
> 进展记录（2026-04-30）：dense/sparse decode 的 Volta MMA_884 QK/PV 主循环已经共用 `csrc/sm70/common/mma_884_attention.h`。该 header 封装 `mma884_accumulate_qk`、`mma884_accumulate_pv`、row0 score store、probability fragment 装载和 row0 PV 输出映射；dense 与 sparse 仍保留各自的 Q/K/V 数据装载和 scheduler 分派，但不再各自维护一份底层 MMA fragment accumulate / row0 映射逻辑。默认 SM70 build 与 V100 sparse decode 13/13、dense FP16 correctness 均已复核，sparse quick benchmark 仍命中 `mma884_online`。

**验证命令：**

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_flash_mla_sparse_decoding.py
```

**验收：**

- Sparse correctness 子集通过：V32/MODEL1、`D_qk in {576,512}`、`H_q in {64,128}`、`S_q in {1,2}`、`topk in {64,576,2048}`。
- 8192 级别边界通过：V32 `topk=8192 + attn_sink`，MODEL1 `4096+4096 + topk_length + extra_kv + extra_topk_length + attn_sink`。
- `attn_sink` 的输出缩放与 reference 一致。
- `extra_kv` 和 `extra_topk_length` 的输出与 reference 一致。

## 任务 7：补齐 sparse feature matrix

**Files:**

- Modify: `csrc/sm70/decode/sparse_fp8/*`
- Modify: `csrc/api/sparse_decode.h`
- Test: `tests/test_flash_mla_sparse_decoding.py`

- [x] 支持 `topk_length`。
- [x] 支持 `extra_kv`、`extra_indices`、`extra_topk_length`。
- [x] 支持 MODEL1 FP8 sparse layout：`D_qk=512`、NoPE 448、RoPE 64、e8m0 scales。
- [x] 支持 `H_q=64` 和 `H_q=128` 的 alpha 性能路径，并用 `bench_sm70_sparse_decode.py` 固定 quick 基线。
- [x] 增加 `FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS` build-time tuning 开关，修正 sparse decode benchmark 的 per-CTA resource 报告，并完成 128/256 quick scan；默认保持 256。
- [x] 增加 `FLASH_MLA_SM70_SPARSE_DECODE_K_TILE` build-time tuning 开关，允许 `{16,32,64}`，完成 FP16 shared staging 后的 rebuild + V100 quick scan；默认保持 `32`。
- [x] 扩展 required feature 检查，确保 unsupported feature 不会静默落错 kernel。
- [x] 让 unsupported sparse 组合的错误正文直接包含架构、模型格式和缺失 feature；generic feature gate 异常正文包含 `missing_features`、`required_features`、`supported_features`。

**验收：**

- `tests/test_flash_mla_sparse_decoding.py` 的 correctness cases 在 SM70 上通过已声明支持的组合。
- unsupported 组合错误信息包含架构、模型格式和缺失 feature。

> 进展记录（2026-04-29）：新增 `benchmark/bench_sm70_sparse_decode.py`，独立记录 SM70 sparse decode FP8 alpha 的 H_q=64/128、V32/MODEL1 和 feature 组合性能。`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python benchmark/bench_sm70_sparse_decode.py --cases quick --warmup 0 --runs 1` 结果如下：
>
> | B | S_q | S_kv | topk | extra_topk | H_q | D_qk | features | us | splitkv us | combine us | TFLOPS | GB/s | max out diff | max lse diff |
> |---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
> | 1 | 1 | 512 | 64 | 0 | 64 | 576 | v32 | 284.125 | 281.629 | 1.600 | 0.031370 | 0.638 | 0.00012207 | 4.76837e-07 |
> | 1 | 2 | 512 | 64 | 0 | 128 | 576 | v32+attn_sink+topk_length | 300.091 | 297.755 | 1.472 | 0.077964 | 2.042 | 0.000244141 | 4.76837e-07 |
> | 1 | 1 | 512 | 64 | 64 | 64 | 512 | model1+attn_sink+extra_kv | 338.364 | 332.796 | 4.608 | 0.049583 | 0.605 | 7.62939e-06 | 9.53674e-07 |
> | 1 | 2 | 512 | 64 | 64 | 128 | 512 | model1+attn_sink+topk_length+extra_kv+extra_topk_length | 683.254 | 677.654 | 4.768 | 0.085175 | 0.945 | 0.00012207 | 9.53674e-07 |
>
> 结论：当前 H_q=64/128 都已命中 SM70 alpha kernel，并已通过 scheduler/combine 跑通 scalar split-K；但仍是 scalar FP32 QK/PV + shared score/token-ref buffer。sm_70 ptxas 目前 V32 为 48 registers、0 spill，MODEL1 为 48 registers、12B spill。后续性能提升需要转入 shared layout + MMA_884，并处理 MODEL1 spill，而不是继续只调 Python harness。
>
> 进展记录（2026-04-29）：shared staging 前新增 `FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS` 后已完成 128/256 quick scan，并修正 benchmark 资源字段为按 `(CTA threads, model)` 报告。shared staging 后资源形态已改变：手工 128-thread sm_70 编译显示 V32 64 registers/0 spill、MODEL1 72 registers/0 spill；默认 256-thread build 显示 V32 64 registers/0 spill、MODEL1 77 registers/0 spill。当前默认仍保持 256，后续进入 FP16/MMA_884 后需要重新做 128/256 runtime scan。
> 进展记录（2026-04-29）：FP16 shared staging 后新增 `FLASH_MLA_SM70_SPARSE_DECODE_K_TILE`，允许 `{16,32,64}`，默认 `32`。quick scan 结果显示：`K_TILE=16` 把 V32/MODEL1 shared bytes 降到 `19968/18432`，occupancy 提升到 `50.0%/37.5%`，但短 h64 case 变慢；`K_TILE=64` 只改善短 V32 case，shared bytes 回到 `75264/67584` 且 occupancy 降到 `12.5%`，h128/MODEL1 明显退化；`K_TILE=32` 保持 `38400/34816` shared bytes 与 `25.0%` occupancy，在四个 quick case 间更均衡，因此作为当前默认。后续 MMA_884 主循环落地后需要重新扫描该参数。

## 任务 8：prefill 快路径与 fallback 策略

**Files:**

- Modify: `csrc/api/sparse_fwd.h`
- Possibly Create: `csrc/sm70/prefill/sparse/*`
- Documentation: `README.md`

- [x] 先不要承诺发布级 SM70 sparse prefill MMA 快路径；用户要求后已新增 BF16 SIMT fast path v1。
- [x] 早期在 `sparse_fwd.h` 中为 SM70 给出清晰错误；现已替换为 SM70 BF16 fast path 分派。
- [x] 如果 vLLM 目标分支必须调用 sparse prefill，新增 BF16 SIMT 方案评估。
- [x] 在 FlashMLA 内新增 SM70 sparse prefill BF16 SIMT fast path v1，不修改 vLLM。
- [x] 增加 SM70 sparse prefill 小矩阵 correctness 子集。
- [x] 将支持矩阵写入 README 或新文档。
- [x] 增加 SM70 sparse prefill fast path benchmark 脚手架，输出 us、TFLOPS、GB/s、寄存器、spill 和静态 occupancy。
- [x] 增加 `max_topk` benchmark smoke，覆盖 SM70 sparse prefill `topk=8192` 支持边界。
- [x] 本地参考 `/mnt/data/apps/lmdeploy` TurboMind 与 `/mnt/data/apps/flash-attention-v100` 的 SM70 prefill/MMA 组织，明确后续 MMA_884 shared-K 路线。
- [x] 增加 `FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS` build-time tuning 开关，默认 256，并完成 128/256 quick scan。
- [x] 增加 `FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE` build-time tuning 开关，允许 `{16,32}`，默认 `32`。
- [x] 将 SM70 sparse prefill SIMT path 的 QK/PV 输入切到 FP16 shared K/V tile staging，保留确定性 LSE/sink 语义。
- [x] 将 SM70 sparse prefill QK score 计算改为 warp-parallel dot reduction，减少每个 topk token 单线程串行扫 512/576 维的瓶颈。
- [x] 新增 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK` opt-in 两阶段验证路径，用 Volta `mma_m8n8k4_row_col` 计算 staged K tile 的 QK score；当前默认已由 online 路径隐式启用 QK/PV MMA_884，`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=0` 仍可回退到 `warp_simt_qk+simt_pv`。
- [x] 新增 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV` opt-in 路径，用 Volta `mma_m8n8k4_row_row` 在 LSE/sink 后计算 PV row0 输出；与 QK opt-in 组合后形成 `mma884_qk+mma884_pv` 两阶段路径。
- [x] 新增 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE` opt-in alpha，在现有 FP16 shared K/V staging 基础上把 QK、online softmax/LSE 更新和 PV 放进同一 tile 流水，减少两阶段 scores/LSE/PV shared 边界。
- [x] 在 sparse prefill online alpha 基础上复扫 `K_TILE=16/32` 与 `max_topk=8192`，并将 SM70 sparse prefill 默认路径切到 `256/32/mma884_online`。
- [x] 完成 dense/sparse decode online path 的主循环复用边界整理：公共 parallel online-softmax tile update 进入 `csrc/sm70/common/softmax.h`。
- [x] 完成 sparse decode online 默认路径的 CTA/K tile 后续复扫与 V32-only runtime adaptive K tile dispatch。
- [x] 继续推进 dense 的 MMA_884 默认路径选择。

> 进展记录（2026-04-29）：静态检查 `/mnt/data/apps/vllm` 后确认 DeepSeek V4 prefill 路径会调用 `flash_mla_sparse_fwd`，因此 FlashMLA 侧需要提供 SM70 prefill 路径。评估记录见 `docs/sm70-volta-vllm-integration-notes.md`；当前 FlashMLA 已有 BF16 SIMT fast path v1，端到端仍受 vLLM gate 限制。
> 进展记录（2026-04-29）：已在 FlashMLA 内落地 SM70 sparse prefill BF16 SIMT fast path v1，路径为 `csrc/sm70/prefill/sparse/*`，dispatcher `Fwd_Sm70_Bf16_Impl` 接入 `sparse_fwd.h`。当前实现每个 CTA 负责一个 `(q, head)`，CTA 线程并行计算 topk QK score，thread 0 按 topk 顺序做确定性 LSE/sink，其他线程并行写 `D_v` 输出。支持 `D_qk in {512,576}`、`D_v=512`、`H_q in {64,128}`、`topk <= 8192`、`topk_length`、`attn_sink`。这是可运行 fast path v1，但还不是最终 TurboMind 风格 MMA_884 版本。
> 进展记录（2026-04-29）：参考 `/mnt/data/apps/lmdeploy/src/turbomind/kernels/attention/attention_config.h` 的 `SM70_PREFILL_USE_MMA_884`、`impl_884.h` 的 `mma_m8n8k4_row_col/row_row` QK/PV 主循环、`mainloop_sm70.h` 的 K/V shared load + softmax + PV 流水，以及 `/mnt/data/apps/flash-attention-v100/docs/volta.md` 的 Volta `m8n8k4`/quadpair/无 `ldmatrix` 约束。后续真正 MMA fast path 应按这些参考，把 BF16 q/kv staged 为 FP16 shared layout，再进入 MMA_884 QK/PV。
> 进展记录（2026-04-29）：`benchmark/bench_sm70_sparse_prefill.py` 用于固定 fast path 的性能/数值基线。按 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5` 运行 quick case，结果如下：
>
> | S_q | S_kv | topk | H_q | D_qk | features | us | TFLOPS | GB/s | max out diff | max lse diff |
> |---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
> | 1 | 128 | 64 | 64 | 512 | - | 136.000 | 0.061681 | 62.645 | 0.000122074 | 4.76837e-07 |
> | 1 | 128 | 64 | 128 | 576 | attn_sink | 150.176 | 0.118699 | 120.554 | 0.00024391 | 4.76837e-07 |
> | 3 | 256 | 128 | 64 | 576 | topk_length | 210.784 | 0.106372 | 108.354 | 0.000244141 | 9.53674e-07 |
> | 3 | 256 | 128 | 128 | 512 | attn_sink+topk_length | 422.560 | 0.137102 | 138.963 | 0.000122033 | 9.53674e-07 |
>
> 进展记录（2026-04-29）：新增 `FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS` 后对 128/256 threads 做 quick scan。128-thread build 通过，ptxas 约 27 registers、0 spill；默认 256-thread build 通过，ptxas 约 26 registers、0 spill。`CUDA_VISIBLE_DEVICES=5 /home/z/anaconda3/envs/gptq/bin/python benchmark/bench_sm70_sparse_prefill.py --cases quick --warmup 0 --runs 1` 的对比如下，单位 us：
>
> | CTA threads | S_q=1 H_q=64 D512 | S_q=1 H_q=128 D576 sink | S_q=3 H_q=64 D576 topk_len | S_q=3 H_q=128 D512 sink+topk_len |
> |---:|---:|---:|---:|---:|
> | 128 | 165.952 | 194.016 | 241.504 | 457.216 |
> | 256 | 137.600 | 127.136 | 211.648 | 424.608 |
>
> 结论：128 threads 只缩小了 CTA 并行度，没有改善当前 SIMT QK/PV 结构；默认保持 256。该开关保留给后续 shared-K / MMA_884 重构前后的寄存器与 occupancy 复扫。
> 进展记录（2026-04-29）：新增 `--cases max_topk` 后已在 V100 上跑通 `S_q=1,S_kv=8192,topk=8192,H_q=64,D_qk=576,attn_sink`。该 case 输出约 `12461.184 us`，max out/lse/max-logits diff 为 `0.000122201/4.76837e-06/0`；shared score buffer 增至 `32776B`，静态 occupancy 降为 `2 blocks/SM, 25.0%`。这验证的是当前 BF16 SIMT fast path 的支持边界与资源形态，不代表最终 MMA_884 prefill 性能。
> 进展记录（2026-04-29）：SM70 sparse prefill SIMT path 已新增 `K_TILE=32` 的 FP16 shared K/V staging；每个 topk tile 将 BF16 `kv` staged 为 `cutlass::half_t` shared layout，QK score 和 PV accumulator 都从 shared tile 读取。默认 256-thread build 的 ptxas 为 28 registers、0 spill；128-thread 单文件编译为 30 registers、0 spill。`tests/test_sm70_static_contract.py` 为 38 passed，`tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32 correctness。该步骤完成 shared-K 数据流，但仍是 SIMT scalar QK/PV，不等同于完成 TurboMind 风格 MMA_884。
>
> | Case | mean us | max out diff | max lse diff | shared bytes | occupancy |
> |---|---:|---:|---:|---:|---:|
> | quick S_q=1 H_q=64 D512 | 286.912 | 0.000122074 | 4.76837e-07 | 33160 | 25.0% |
> | quick S_q=1 H_q=128 D576 sink | 296.192 | 0.00024391 | 4.76837e-07 | 37256 | 25.0% |
> | quick S_q=3 H_q=64 D576 topk_len | 745.952 | 0.000244141 | 9.53674e-07 | 37512 | 25.0% |
> | quick S_q=3 H_q=128 D512 sink+topk_len | 1241.568 | 0.000122033 | 9.53674e-07 | 33416 | 25.0% |
> | max_topk S_q=1 H_q=64 D576 sink | 40683.041 | 0.000122201 | 4.76837e-06 | 69768 | 12.5% |
>
> 进展记录（2026-04-29）：SM70 sparse prefill 的 QK score 已改为 warp-parallel dot reduction。每个 warp 负责一个 staged topk token，lane 按 `dim += 32` 分摊 BF16 q 与 FP16 staged KV 的 dot，保留 thread 0 确定性 LSE/sink 和现有 PV 写回语义。默认 256-thread build 的 ptxas 最大为 29 registers、0 spill；128-thread 探针最大为 32 registers、0 spill。`PYTHONPATH=/mnt/data/apps/FlashMLA/build/lib.linux-x86_64-cpython-313 CUDA_VISIBLE_DEVICES=5 ... tests/test_flash_mla_sparse_prefill.py` 通过 32/32；quick benchmark 约 `230.688/204.320/563.776/860.384 us`，max LSE diff `<= 9.53674e-07`。
> 进展记录（2026-04-29）：新增 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=1` opt-in QK-score 路径，复用 `csrc/sm70/common/mma_884.h` 的 `mma_m8n8k4_row_col`，在 FP16 shared K tile 上每个 warp group 计算 8x8x4 QK fragment 并写回 row0 score。该路径只替换 QK score 计算，thread 0 的确定性 LSE/sink 与 PV SIMT accumulator 保持不变，因此不等同于完整 TurboMind 风格 QK/PV MMA_884 主循环。opt-in build ptxas 为 40 registers、0 spill；`tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32。
>
> | qk_path | S_q | S_kv | topk | H_q | D_qk | features | mean us | registers | occupancy | max lse diff |
> |---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
> | mma884_qk | 1 | 128 | 64 | 64 | 512 | - | 259.776 | 40 | 25.0% | 9.53674e-07 |
> | mma884_qk | 1 | 128 | 64 | 128 | 576 | attn_sink | 259.840 | 40 | 25.0% | 4.76837e-07 |
> | mma884_qk | 3 | 256 | 128 | 64 | 576 | topk_length | 795.232 | 40 | 25.0% | 9.53674e-07 |
> | mma884_qk | 3 | 256 | 128 | 128 | 512 | attn_sink+topk_length | 1163.776 | 40 | 25.0% | 9.53674e-07 |
>
> 进展记录（2026-04-30）：新增 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV=1` opt-in PV 路径，复用 `csrc/sm70/common/mma_884.h` 的 `mma_m8n8k4_row_row`，在已计算好的概率 tile 与 FP16 shared V tile 上做 PV 并写回 row0 输出。`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=1 FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV=1` 组合后，QK 和 PV 都走 Volta MMA_884，但中间仍保留 scores/LSE 边界，因此这是两阶段验证路径，不等同于 TurboMind 风格 fused/online 主循环。组合 build ptxas 为 48 registers、0 spill；`tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32，默认 `QK=0/PV=0` rebuild 后也通过 32/32；`tests/test_sm70_static_contract.py` 为 42 passed。
>
> | compute_path | S_q | S_kv | topk | H_q | D_qk | features | mean us | registers | occupancy | max out diff | max lse diff |
> |---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
> | mma884_qk+mma884_pv | 1 | 128 | 64 | 64 | 512 | - | 368.896 | 48 | 25.0% | 0.000129193 | 9.53674e-07 |
> | mma884_qk+mma884_pv | 1 | 128 | 64 | 128 | 576 | attn_sink | 380.704 | 48 | 25.0% | 0.000245683 | 4.76837e-07 |
> | mma884_qk+mma884_pv | 3 | 256 | 128 | 64 | 576 | topk_length | 1158.272 | 48 | 25.0% | 0.000255115 | 9.53674e-07 |
> | mma884_qk+mma884_pv | 3 | 256 | 128 | 128 | 512 | attn_sink+topk_length | 1757.184 | 48 | 25.0% | 0.000129573 | 9.53674e-07 |
>
> 进展记录（2026-04-30）：新增 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=1` opt-in online alpha。该路径隐式启用 MMA_884 QK/PV，shared memory 改为 tile scores、4 个 online scalar、FP32 output accumulator、token refs 和 FP16 K/V tile，不再按 `topk` 分配全量 scores buffer；每个 K tile 中先用 `mma_m8n8k4_row_col` 计算 tile scores，再在线更新 `row_max/row_sum`、缩放 output accumulator，随后用 `mma_m8n8k4_row_row` 累加 PV。online build ptxas 为 48 registers、0 spill；`tests/test_flash_mla_sparse_prefill.py` 在 V100 上通过 32/32。quick benchmark 如下：
>
> | compute_path | S_q | S_kv | topk | H_q | D_qk | features | mean us | registers | shared bytes | occupancy | max out diff | max lse diff |
> |---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
> | mma884_online | 1 | 128 | 64 | 64 | 512 | - | 201.344 | 48 | 35088 | 25.0% | 0.000128843 | 9.53674e-07 |
> | mma884_online | 1 | 128 | 64 | 128 | 576 | attn_sink | 199.712 | 48 | 39184 | 25.0% | 0.000247553 | 4.76837e-07 |
> | mma884_online | 3 | 256 | 128 | 64 | 576 | topk_length | 586.560 | 48 | 39184 | 25.0% | 0.000250190 | 4.76837e-07 |
> | mma884_online | 3 | 256 | 128 | 128 | 512 | attn_sink+topk_length | 878.272 | 48 | 35088 | 25.0% | 0.000125986 | 9.53674e-07 |
>
> `max_topk` smoke (`S_q=1,S_kv=8192,topk=8192,H_q=64,D_qk=576,attn_sink`) 也通过，约 `21713.312 us`，max out/lse/max-logits diff 为 `0.000122488/1.90735e-06/9.53674e-07`，shared bytes `39184`，静态 occupancy `2 blocks/SM, 25.0%`。
>
> 进展记录（2026-04-30）：在 online alpha 上复扫 `K_TILE=16/32` 后完成默认路径选择。`K_TILE=16` quick 为 `257.536/260.032/457.184/861.408 us`，shared bytes `18576-20624`，occupancy `50.0-62.5%`，但 `max_topk=8192` 为 `29200.224 us`；`K_TILE=32` quick 为 `200.960/200.032/578.560/875.616 us`，`max_topk=8192` 为 `21693.184 us`。因此默认切到 `FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=1` + `K_TILE=32`，`FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=0` 仍可强制回退到 `warp_simt_qk+simt_pv`。

**验收：**

- 用户在 SM70 上调用已声明支持的 BF16 sparse prefill 子集时命中 SM70 BF16 fast path 并得到正确输出；超出支持范围时得到可操作错误，而不是 SM90/SM100 误导信息。

## 任务 9：vLLM DeepSeek V4 Flash 集成验证

**Files:**

- Create: `benchmark/bench_vllm_deepseek_v4_flash_sm70.py`
- Documentation: `docs/sm70-volta-vllm-integration-notes.md`

- [x] 确认目标 vLLM 分支中 DeepSeek V4 Flash 实际调用的 FlashMLA API、dtype、block size、KV cache dtype。
- [x] 确认是否调用 dense decode、sparse decode、sparse prefill。
- [x] 新增静态 inspect 与 OpenAI streaming smoke/benchmark 脚手架，记录 `TTFT`、`decode_tokens_per_s`、`finish_reason`。
- [ ] 跑最小 prompt smoke，记录是否命中 `sm70` FlashMLA kernel。
- [ ] 记录端到端 decode tokens/s、TTFT、finish_reason、显存占用。
- [x] 若 vLLM 还被 FlashInfer、MoE、量化或 scheduler 阻断，将其拆成独立任务，不把问题归咎于 FlashMLA kernel。
- [ ] vLLM SM70 gate 放开任务：将 `DeepseekV4FlashMLASparseBackend.supports_compute_capability` 从 SM90/SM100-only 放开到满足 FlashMLA SM70 sparse 支持矩阵的 SM70。
- [ ] vLLM SM70 runtime probe 任务：将 `is_flashmla_sparse_supported()` 从 SM90/SM100-only 放开到 SM70，并保留对未支持 dtype/layout 的清晰错误。

> 进展记录（2026-04-29）：`/mnt/data/apps/vllm` 当前分支为 `woosuk/dsv4-sync`、commit `f481dcb7a`。静态路径显示 `DeepseekV4ForCausalLM` 已注册；`DeepseekV4MLAAttention` 强制选择 `DeepseekV4FlashMLASparseBackend`，KV cache dtype 转为 `fp8_ds_mla`。Decode 侧调用 sparse `flash_mla_with_kvcache`，MODEL1 584B layout 与 SM70 sparse alpha 对齐；prefill 侧调用 `flash_mla_sparse_fwd`，当前 FlashMLA 已提供 SM70 BF16 fast path v1，但 vLLM 侧 `supports_compute_capability` 和 `is_flashmla_sparse_supported()` 仍需放开 SM70 后才能做真实端到端 smoke。
> 进展记录（2026-04-30）：`benchmark/bench_vllm_deepseek_v4_flash_sm70.py --mode inspect --json` 已新增 `sm70_blockers` 与 `end_to_end_smoke_ready` 字段。当前 `/mnt/data/apps/vllm` inspect 显示 DeepSeek V4 FlashMLA API 路径存在，但 `end_to_end_smoke_ready=false`，阻断项为 `vllm_flashmla_sparse_backend_sm70_gate` 和 `vllm_flashmla_sparse_runtime_sm70_gate`。这两个阻断已拆成独立 vLLM gate/runtime probe 任务；在它们完成前，不把 OpenAI smoke 无法命中 SM70 FlashMLA 归因到 FlashMLA kernel。

**验收：**

- 有一份端到端 smoke 报告，明确：启动命令、模型路径、GPU mask、FlashMLA kernel 命中情况、首个失败点或成功输出。

## 任务 10：CI、文档和发布边界

**Files:**

- Modify: `README.md`
- Modify: `docs/sm70-volta-flashmla-design.md`
- Modify: `docs/sm70-volta-flashmla-tasks.md`
- Possibly Modify: CI files if present

- [x] README 支持矩阵增加 SM70 preview 行。
- [x] 记录推荐工具链：CUDA 12.x、PyTorch 版本、V100。
- [x] 记录已支持和未支持功能。
- [x] 记录性能不是对齐 H100/SM90，而是对齐 Volta 原生能力。
- [x] 增加回归命令列表。

**验收：**

- 新用户能按文档知道 SM70 支持范围、构建方式、测试命令和当前限制。

## 阶段性验收矩阵

| 里程碑 | 必须通过 | 可延期 |
|---|---|---|
| M1 dense alpha | SM70 build、FP16 dense correctness、小规模 long context | BF16、sparse |
| M2 dense beta | split-K combine、MTP `S_q>1`、性能报告 | vLLM 端到端 |
| M3 sparse alpha | V32/MODEL1 FP8 dequant、attn_sink、topk/extra_kv 基础正确性 | SM70 sparse 性能路径 |
| M4 sparse beta | SM70 sparse 性能路径、sparse prefill 性能化 | 发布级 prefill 性能 |
| M5 integration | vLLM DeepSeek V4 Flash smoke、kernel 命中、指标报告 | 发布级性能优化 |

## 关键风险检查点

- ptxas register spill：如果出现，优先降低 H tile，不要先 split V。
- BF16：不要默认认为 V100 能高效 BF16 Tensor Core。
- FP8：不要依赖 Hopper FP8 conversion；SM70 路径使用 byte-level dequant。
- sparse fallback：FlashMLA 内部不能凭 sparse 输入自动恢复 dense 全 KV 语义。
- 参考仓库：flash-attention-v100 和 lmdeploy 是只读参考，不要直接改。
