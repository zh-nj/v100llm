三方 flash-attention-v100 深度对比分析
先把三者的"血统"定位清楚，再往下比：

代号	路径	作者/出处	定位
FA-Dao	/mnt/data/apps/fa2/flash-attention-v100/	Tri Dao 官方库裁剪到 SM70（vLLM Team 维护）	用 CuTe/CUTLASS 还原 FA2 全套
FA-Bond-pure	/mnt/data/apps/flash-attention-v100/	D.Skryabin / @ai_bond007，github.com/ai-bond/flash-attention-v100	纯 WMMA 写的 FA2，dense 前反向，零依赖
FA-Bond-vLLM	/mnt/data/apps/fa2/1Cat-vLLM/flash-attention-v100/	同一 Bond 代码再分叉（同一版本号 26.02、同作者）	在 Bond-pure 基础上追加 paged KV 和 decode 特化，为 vLLM 特化
三者用表快速对位：

FA-Dao	FA-Bond-pure	FA-Bond-vLLM
源码量（核心）	~60 个 TU + CUTLASS	2316 行 核心代码	~4000 行（Bond-pure + paged）
依赖	CUTLASS/CuTe submodule	仅 mma.h（WMMA）	仅 mma.h
MMA 指令	SM70_8x8x4_F32F16F16F32_TN（mma.sync）	wmma.m16n16k16.f32.f32	wmma.m16n16k16.f32.f32
head dim	32 / 64 / 96 / 128 / 192 / 256 / 512	16 / 32 / 64 / 128 / 256	16 / 32 / 64 / 128 / 256（decode 另加 80/96/112）
反向	有（CuTe，完整 dropout/alibi）	有（独特的 dQ+dKV 融合单 kernel）	有（同 Bond-pure）
Paged KV	dense kernel 内分支	❌ 不支持	✅ prefill_paged + decode_paged 独立 kernel
Decode 特化	❌（走 splitkv）	❌	✅（split-partition + reduce，CUDA core）
MQA/GQA	✅ h_h_k_ratio	❌ 严格要求 H==H_KV	✅ kv_group_size
dropout / alibi / softcap / sliding / rotary	✅ 全部	❌ TORCH_CHECK 拒绝	❌ TORCH_CHECK 拒绝
vLLM 集成	改一行 supports_compute_capability	纯函数库（训练用）	需要写 vLLM backend
下面按你关心的维度深入。

1. 代码抽象层
1.1 FA-Dao
Flash_fwd_kernel_traits<Headdim,kBlockM,kBlockN,kCtaWarps> 一次生成：CuTe TiledMma、swizzled SmemLayoutQ/KV、GmemTiledCopyQKV（带内联 PTX 的 SM70_LDG_GLOBAL_CG_128b）、SmemLayoutVtransposed。 kernel 主体 compute_attn_1rowblock 走 FA2 原汁原味：Q 加载一次 → 反向遍历 KV → online softmax 全在寄存器 → convert_layout_C_to_A_v2 把 P 从 C-fragment shuffle 到 A-fragment 直接进下一个 GEMM。

1.2 FA-Bond-pure —— 三方里最干净的一个
看它的头文件分层：

00_volta_const.cuh        # 47 行，SM70 硬件常量 + GemmType 位字段枚举
01_forward_config.cuh     # 54 行，forward block/warp 表
01_backward_config.cuh    # 105 行，backward 的 DQ/DKV 双配置
02_wmma.cuh               # 389 行，通用 WMMA 算子（GEMM_SCORES / GEMM_GRADIENTS / EPILOGUE / DOT_PRODUCT）
fused_mma.h               # 501 行，m16n16k16 / m32n8k16 / m8n32k16 三规格 WMMA asm 封装
最值得注意的是 GemmType（00_volta_const.cuh 里的位字段枚举）：

enum class GemmType : uint8_t {
    sQ_KT     = 0b101,  // APPLY_MASK=1, A=row, B=col  (Q @ K^T)
    dOV_dOVT  = 0b100,  // no mask,      A=row, B=col  (dO @ V^T)
    dO_PV     = 0b001,  // accumulate=1, A=row, B=row  (P @ V)
    dV_PTdO   = 0b011,  // accumulate=1, A=col, B=row  (P^T @ dO)
    dK_dSTQ   = 0b011,
    dQ_dSK    = 0b001,
    write_dO  = 0b001,  // NORMALIZE=1  (fwd output)
    write_dQ  = 0b000,  // NORMALIZE=0
    write_dKV = 0b010,  // DUAL_OUTPUT=1 (dK+dV 同时写)
    ...
};
WMMA_GEMM_SCORES<GemmType::sQ_KT, D, IS_CAUSAL, ...> 模板解码这些位，用 if constexpr 选 row/col layout 和 mask 分支。整个 8-次 GEMM 用同一个模板函数，前反向共用一套 SCORES/GRADIENTS/EPILOGUE 原语。

这种写法比 Bond-vLLM 的"forward 和 backward kernel 互相复制粘贴"工程风格规整得多，也比 FA-Dao 的 CuTe DSL 易读得多。代价是功能面窄。

1.3 FA-Bond-vLLM
基本是 Bond-pure 的 fork，但 退化了抽象：forward 和 paged_forward 两个 kernel 都是几百行的大函数，fused_mma.h 里保留的是原始 WMMA PTX 封装（load_matrix_sync / mma_sync 手写 inline asm），没有 Bond-pure 的 WMMA_GEMM_* 宏。原因是 paged 路径要在 load 阶段做 page_table 解算、单页/双页/通用三档分支，用模板封装会把信号弄丢。所以 Bond-vLLM 的 dense forward 基本是把 Bond-pure 的原语内联展开，再在 KV 加载处插上分页逻辑。

2. 前向主循环流程
三者的 online FA2 骨架是相同的：Q 驻留 shmem → for n_block → load K → QK → softmax + rescale sO → load V → PV → next。差别全在 "S/P/O 放哪 + 怎么拿出来用"。

2.1 S/P/O 的存放位置
S（QK 结果 fp32）	P（softmax fp16）	acc_o（fp32）	软 max 归约
FA-Dao	寄存器（acc_s，Tensor<Engine>）	寄存器（rP），再经 convert_layout_C_to_A_v2 用 shfl_sync 重排到 A-fragment 形状	寄存器（acc_o，整个 kernel）	warp 内 3 步 __shfl_xor_sync（sm70_row_allreduce_8）
FA-Bond-pure	shmem (sS, fp32)	shmem (sP, fp16，stride = 2×N_STRIDE)	shmem (sO, fp32)	一行 Q = 多个 lane，组内 __shfl_down_sync + __shfl_sync(row_leader)
FA-Bond-vLLM	shmem (sS)	shmem (sP，stride = N_STRIDE)	shmem (sO)	同 Bond-pure
关键差异：

FA-Dao 的 P 从未落过 shmem，Tensor Core 在 QK 和 PV 间几乎无缝转，但付出大量寄存器（acc_s + acc_o 每线程十几到几十个 fp32），容易 spill——CLAUDE.md 里明确提到这个分支叫 debug/sm70-linear16-spill-only。
Bond 两版都选了 "shmem 存 S/P/O" 的保守路线，寄存器压力极小（每线程常驻的 acc_frag.x[8] = 8 fp32 + half_buffer[20] 暂存），代价是 PV 要额外 wmma.load.a 一次 P。
Bond-pure 的 P_STRIDE = N_STRIDE * 2 是个我在 Bond-vLLM 里没看到的优化：
static constexpr int N_STRIDE = BLOCK_N + PAD + (((BLOCK_N+PAD)%32==0) ? 1 : 0);
static constexpr int P_STRIDE = N_STRIDE * 2;   // P 是 fp16 而 S 是 fp32，但共用同一块 shmem
union { float s[BLOCK_M * N_STRIDE]; half p[BLOCK_M * P_STRIDE] } 共用 32KB 左右内存。当 S 写完后 P 从同一物理地址"按 fp16 视角"读，因为 fp16 是 2 字节、fp32 是 4 字节，所以 P_STRIDE = 2 * N_STRIDE 正好让 P 矩阵在物理上的行距与 S 一致，WMMA load.a 读 P 时可以直接用相同的 row 走到不同行。这是只有在 S 和 P shmem 复用时才需要的小技巧，Bond-vLLM 里 paged 版直接让 S 和 P 用同一个 stride，多占 shmem。
2.2 Bank-conflict 消除策略
手段	细节
FA-Dao	Swizzle<3,3,3> XOR	CuTe 在列索引上做 8×8 XOR 混洗，严格无 conflict
FA-Bond-pure	双段 PAD	D_STRIDE = D + PAD + (((D+PAD)%64==0) ? 1 : 0) ，当对齐到 64 时再额外 +1 打破 32-bank 对齐
FA-Bond-vLLM	单段 PAD	Q_STRIDE = D + PAD，简单的 (8 - D%32 + 32) % 32，不加 "+1"
Bond-pure 这个"打破 64 对齐额外加 1"的技巧是比 Bond-vLLM 更讲究的：当 D=64 时 (D+PAD)%64 == 0 就会多加 1，让相邻行的 bank 错位。vLLM 分叉反而没带这个 tweak，可能是为了配合 paged KV 读取时整齐的 uint4 向量化。

2.3 Grid & 并行度
grid	并行元素
FA-Dao	(num_m_block, b, h)	num_m_block × b × h，可选 splitkv 多路径 combine
FA-Bond-pure	(num_m_block, 1, b*h)	num_m_block × b*h
FA-Bond-vLLM	同 Bond-pure；decode 另有 (b, h, num_partitions) + reduce 阶段	prefill 同；decode 显式把 KV 切 partition 并发
三者都把 b*h 展平到 z 维，方便一次扫完。FA-Dao 的 splitkv kernel + combine kernel 是为长 KV 做的二次切分；FA-Bond-vLLM 只在 decode（Q=1）时做分区；FA-Bond-pure 根本不切。

3. 反向路径（这是 Bond-pure 最大的亮点）
3.1 FA-Dao
标准 FA2 反向，按 (seqlen_q, seqlen_k) 两轴各跑一个 kernel（或同 kernel 分两 pass）：

flash_bwd_preprocess_kernel.h：先算 do * o 的行和 D_i；
flash_bwd_kernel.h：主反向，dropout、alibi、softcap 全支持。 每个 flash_bwd_hdim{X}_[causal]_sm70.cu 一个 TU。
3.2 FA-Bond-pure —— 单 kernel 两相合并
最让我眼前一亮的设计：

template<int D, bool IS_CAUSAL>
__global__ void flash_attention_backward_kernel(...) {
    if (blockIdx.y == 0) {
        // PHASE 1: dQ  - 块大小 Config::DQ::BLOCK_M, Config::DQ::BLOCK_N
        ...
    }
    else if (blockIdx.y == 1) {
        // PHASE 2: dKV - 块大小 Config::DKV::BLOCK_M, Config::DKV::BLOCK_N
        ...
    }
}
// launcher:
const dim3 grid(max(grid_dq, grid_dkv), 2, B*H);
一次 kernel launch 同时驱动 dQ 和 dKV 两种不同分块形状。SmemLayout 里是一个巨型 union：

union PhaseMem {
    struct DQ_Phase { sK, sV, sdO, sQ, sS, union{sdOV,sdS}, sdQ; } dq;
    struct DKV_Phase { sK, sV, union{sdO,sQ}, union{sS,sP}, union{sdOV,sdS}, sdK, sdV; } dkv;
} phase;
两相不同时激活，TOTAL_SMEM = max(sizeof(DQ_Phase), sizeof(DKV_Phase))。好处是：

单次 launch 省去 API overhead（对小 shape 很明显）；
两相 BLOCK_M/BLOCK_N 可以独立最优化（dQ 用 BLOCK_M=32, BLOCK_N=112（hdim=128），dKV 用 BLOCK_M=16, BLOCK_N=128），因为 dQ 沿 Q 切、dKV 沿 KV 切，逆向运动方向。
FA-Dao 也有类似理念但分成两个 kernel，FA-Bond-vLLM 更是直接沿用 Bond-pure 这个 kernel。

3.3 反向里的 dS 计算——Bond-pure 的精彩展开
Bond-pure dKV 阶段的 dS/P 合计算，用了一个按 uint4 打包的分批写：

// 8 个 fp16 一次打包成 1 个 uint4 via inline asm (mov.b32)
asm volatile(
    "{ mov.b32 %0, {%4,%5}; mov.b32 %1, {%6,%7}; "
    "  mov.b32 %2, {%8,%9}; mov.b32 %3, {%10,%11}; }\n"
    : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
    : "h"(__half_as_ushort(__float2half_rn(ds0))), ... );
buf[cnt++] = res;
比用内置的 __half2 打包少 2-3 条 SASS，这是 老式 SM70 上的延迟敏感优化，因为 SM70 还没有 fp16x2 的自动合并指令。Bond-vLLM 的 paged forward 里没用到这个技巧（不需要写 dS）。

3.4 反向性能
三者都没公开反向 benchmark 数据。定性判断：

FA-Dao 反向：特性最全，是唯一支持 dropout/alibi 反向的；但依赖 cutlass，编译慢，占 shmem 大；
FA-Bond-pure 反向：设计最工整（GemmType 枚举 + union 布局 + 单 kernel），适合训练 Unsloth 这类场景（见 README 的 Unsloth 日志）；
FA-Bond-vLLM 反向：复刻 Bond-pure，但不是 vLLM 的主用路径（推理不反向）。
4. SM70 的硬件资源利用
4.1 Tensor Core
指令	每 warp 一次	累加器	每线程 fp32 存储
FA-Dao	mma.m8n8k4	8×8×4 = 256 FMA	C = f32	2（每 mma）
FA-Bond-pure	wmma.m16n16k16	16×16×16 = 4096 FMA	C = f32	8（acc_frag.x[8]）
FA-Bond-vLLM	wmma.m16n16k16	同 Bond-pure	同	同
理论峰值一样（硬件是同一套 Volta Tensor Core，内部仍是 m8n8k4 四路并行）。差异在指令调度空间：

mma.m8n8k4 粒度细，发射频率高，编译器可以在 4 条 mma 之间插 softmax / shmem 预取来遮蔽延迟。FA-Dao 的 gemm_rs 就是这样做的。
wmma.m16n16k16 是一条"大指令"，编译器腾挪空间小，但一次发射抵得上 FA-Dao 8 条 mma，发射开销更低。在 SM70 上 wmma 路径的优势是：bank-conflict 少出现（因为 load.a/load.b 用硬件管理的 swizzle）、调度压力低。
从 README 给出的实测 q8k 数据看，Bond 系（wmma 路线）不比 FA-Dao 慢；FA-Dao 的理论优势在寄存器驻留 + 细粒度指令交错，只有在 shmem 带宽吃紧时才显现。

4.2 CUDA Core
三者的 softmax 都是 CUDA core 上做的，但规约方式完全不同：

FA-Dao：Allreduce<4>::run 在"持有同一行 8 lane"子集内归约，这是 m8n8k4 布局的自然结果；
FA-Bond-pure / vLLM：THREADS_PER_ROW = THREADS_PER_BLOCK / BLOCK_M（hdim=128、BLOCK_M=32 时 =16 lane 管一行），用 __shfl_down_sync(mask, x, offset, THREADS_PER_ROW) 做组内归约，THREADS_PER_ROW 可以是任意 2 的幂，不绑定 tensor core 布局。
Bond-pure 的 WMMA_GEMM_DOT_PRODUCT（反向用来算 row_dot）里甚至直接用 ld.global.v4.u16 + ld.shared.v4.u16 + fmaf 做 warp 内 8-way dot，完全没用 tensor core——对 reduce 操作这是对的选择。

4.3 Shared Memory（单 CTA 预算）
hdim=128 配置	FA-Dao	FA-Bond-pure	FA-Bond-vLLM
BLOCK_M / BLOCK_N	64 / 64	32 / 176	32 / 176
warp / CTA	4	16	16
Q smem	~16 KB	~10 KB	~10 KB
K or V smem (union)	2 × 16 KB = 32 KB（同时常驻）	~55 KB	~55 KB
S or P smem (union)	0（寄存器）	~27 KB	~27 KB
O smem	复用 Q 位置	~20 KB	~20 KB
总计	~48 KB（2 CTA/SM 可行）	~96 KB（1 CTA/SM，紧贴上限）	~96 KB（1 CTA/SM）
96KB 检查	不触发 cudaFuncAttributeMaxDynamicSharedMemorySize	显式 set max dynamic smem	同左
并发取舍：

FA-Dao 占一半 shmem → 2 CTA/SM → memory latency 能被另一 CTA 遮蔽；
Bond 系占满 shmem → 1 CTA/SM → 靠 16 warp 内部并行度遮蔽延迟。
V100 每 SM 64K 寄存器 / 96K shmem / 2048 线程。Bond 系 THREADS_PER_BLOCK = 16×32 = 512，每 SM 至多 4 个 CTA（线程数限制），但 shmem 卡死成 1 CTA。FA-Dao kNThreads = 4×32 = 128，每 SM 原则可 16 CTA，shmem 限到 2 CTA。两种都是 shmem 驱动的。

4.4 寄存器压力
FA-Dao：硬核路线。acc_s[ MMA_M × MMA_N ]、acc_o[ MMA_M × hdim ] 都是寄存器 tensor，单线程 60-80 个 fp32 寄存器，配合 acc_s->rP 转换的临时量容易冲 255 上限（已知 spill 历史）。
FA-Bond-pure / vLLM：acc_frag.x[8] 只在 QK、PV 临时 live；softmax 临时用 half_buffer[20]（20 个 fp16 = 10 个 32-bit），每线程常驻 40 左右个寄存器。
结论：Bond 系用 shmem 换寄存器，在 SM70 这个 64K RF 的老架构上更稳；FA-Dao 吃到寄存器 + swizzle 的双重红利，但代价是 spill 调优。

5. 三者分别在什么场景有明确优势
场景	首选	原因
vLLM 在 V100 上直接跑 Llama/Qwen 推理	FA-Dao 或 FA-Bond-vLLM	FA-Dao 改一行 SM 支持就能开 FLASH_ATTN backend；Bond-vLLM 性能更猛但要写适配层
vLLM paged KV + decode 为主（长上下文对话）	FA-Bond-vLLM	唯一有 decode split-partition + reduce 专用 kernel，q8k 实测快 540%
训练场景（需要反向 + 自定义特性）	FA-Dao	唯一支持 dropout/alibi/softcap/rotary 反向
在 V100 上用 Unsloth 做 LoRA 微调	FA-Bond-pure	它就是 README 演示的场景，单 kernel 两相反向简单高效
想阅读/学习 FA2 在 SM70 的实现	FA-Bond-pure	2316 行、分层清晰、GemmType 位字段枚举优雅
想阅读/学习 CuTe 如何在老架构上写 FA	FA-Dao	CuTe TiledMMA、Swizzle、convert_layout_C_to_A_v2 都是教科书级别实例
生产环境需要和 upstream Tri Dao 同步更新	FA-Dao	继承了上游目录结构，git rebase 可行
长 KV 长 prefill（q≈k≈8k+）	FA-Bond-vLLM > Bond-pure > FA-Dao	BLOCK_N=176 外循环次数少，softmax 归约摊薄；Bond-vLLM 还能用 paged 快速路径
6. 家族谱系与工程建议
/mnt/data/apps/flash-attention-v100（FA-Bond-pure）是"树根"：同作者 D.Skryabin 的最干净实现。
1Cat-vLLM/flash-attention-v100（FA-Bond-vLLM）是 Bond-pure 的 vLLM 侧分支：保留 forward/backward，替换 forward 抽象（展平 WMMA_GEMM_* 宏为 inline），新增 prefill_paged / decode_paged / MQA-GQA / block_table 支持。
flash-attention-v100（FA-Dao）和这俩无继承关系，是 vLLM Team 在 Tri Dao 主仓上的 SM70 移植。
如果你要在当前这个 workspace 里整合一个"最终版 V100 flash-attn"，我的建议：

以 FA-Bond-pure 的抽象（00_*/01_*/02_wmma.cuh）为骨架，代码最规整；
把 FA-Bond-vLLM 的 paged/decode 两个 kernel 移植过来，当作 forward 的变体；
把 FA-Dao 的 convert_layout_C_to_A_v2 + 寄存器驻留 softmax 作为"高寄存器预算"的可选路径（#ifdef FA_REGISTER_RESIDENT），在 short-seq 场景自动切换；
反向直接用 Bond-pure 的单 kernel 双相设计；
训练特性（dropout/alibi/softcap）从 FA-Dao 的 dropout.h / alibi.h 借过来即可，接口已经和 Bond 系对齐。
