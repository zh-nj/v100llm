/*
 * SM70 (Tesla V100) fused MoE forward kernel: linear1 -> SwiGLU -> linear2,
 * consuming dsv4f MXFP4 expert weights.
 *
 * This file hosts the hand-written CUDA fused MoE expert kernel for the
 * DeepGEMM Mega MoE concept port to SM70 (spec
 * `deepgemm-megamoe-sm70-port`). The fused path keeps the `gate/up`
 * ([M, 2*I]) and intermediate activation `h` ([M, I]) on chip (shared memory /
 * registers) and never materialises them to HBM, which is the core saving
 * relative to the three-kernel baseline (design §"Components and Interfaces").
 *
 * Constraints (V100 / SM70):
 *   - float16 compute only (no bfloat16).
 *   - No FP8/FP4 tensor cores, no TMA / WGMMA / cp.async / cluster primitives.
 *   - dsv4f expert weights are MXFP4 (E2M1 nibbles + per-32 E8M0 block scale);
 *     the kernel SOFTWARE-dequantises MXFP4 -> fp16 in-kernel and then feeds
 *     the first-generation `mma.sync` (HMMA m8n8k4) fp16 tensor core. No FP4
 *     hardware is used (R2.2).
 *
 * Arch gating is done at the CMake compile-target level
 * (set_gencode_flags_for_srcs ... CUDA_ARCHS "7.0"), mirroring
 * awq_sm70_gemm.cu; this source is only compiled for SM70.
 *
 * ===========================================================================
 * Task 4.1 scope (THIS CHANGE): op-binding SKELETON only.
 *   - Re-defines the host op `sm70_fused_moe_out(...)` to consume dsv4f MXFP4
 *     expert weights (packed FP4 weight tensors + per-32 block-scale tensors)
 *     instead of the prior AWQ int4 StridedPtr arrays.
 *   - Performs full TORCH_CHECK shape/dtype validation of the MXFP4 contract.
 *   - The body is a STUB (`out.zero_()`); the GEMM / SwiGLU / dequant device
 *     code is intentionally NOT implemented here.
 * Task 4.2 scope (THIS CHANGE): linear1 + SwiGLU epilogue DEVICE SEGMENT.
 *   Adds the composable `__device__` building blocks (namespace
 *   `vllm::sm70_fused_moe`) that task 4.3 drives from its `__global__` kernel:
 *     - `sm70_e2m1_decode_signed` / `sm70_e8m0_decode` / `sm70_mxfp4_decode_*`
 *       — the in-kernel MXFP4 (E2M1 + per-32 E8M0 block scale) -> fp16 software
 *       dequant. These numerically match `mxfp4_dequant_to_fp16` in
 *       `vllm/model_executor/layers/fused_moe/sm70_moe_reference.py` exactly
 *       (E2M1 magnitude LUT {0,0.5,1,1.5,2,3,4,6}, sign = bit 3, E8M0 scale =
 *       2^(raw-127)), so reference and kernel share one numeric source (R5.1).
 *     - `sm70_mma_m8n8k4_row_col` — a local copy of the first-generation Volta
 *       HMMA PTX (mirrors `turbomind::mma_m8n8k4_row_col`,
 *       `src/turbomind/kernels/core/mma.h`).
 *     - `sm70_stage_x_kchunk` / `sm70_stage_w13_kchunk_dequant` — synchronous,
 *       block-cooperative SMEM staging of the activation tile and the MXFP4
 *       gate/up weight tile (dequantised on the way into SMEM). No cp.async/TMA.
 *     - `sm70_accumulate_kchunk` — per-warp `mma.sync` accumulation of one
 *       16x16 gate tile and the matching up tile over a K_STEP contraction
 *       chunk; the A/B/C fragment->lane maps mirror the verified row.col path
 *       in `lmdeploy/.../attention/impl_884.h` (TransformQ / StateQK::Load /
 *       ForeachS).
 *     - `sm70_swiglu_epilogue` — on-chip `silu(gate)*up` writing the `h` block
 *       into shared memory.
 *     - `sm70_linear1_swiglu_epilogue` — the classic double-buffered
 *       "load-next / compute-current" pipeline over the K contraction
 *       (two SMEM buffers, `__syncthreads`-separated) that composes the above
 *       and leaves `h` ([M_b, I_b]) resident in shared memory for linear2.
 *   These are device-side only; the host op below remains the task-4.1 stub.
 *   End-to-end numerical equivalence is exercised on SM70 hardware by the
 *   property tests (task 5.5); there is no nvcc / V100 build in this env, so
 *   4.2 is verified by reading + getDiagnostics and decode-vs-reference parity.
 * Task 4.3 (THIS CHANGE): linear2 (down projection) device segment + on-chip
 *   hand-off and the host-side kernel launch. The `h` block is consumed
 *   DIRECTLY from shared memory (`sm70_linear2_downproj`), so neither `gate/up`
 *   ([M_b, 2*I]) nor `h` ([M_b, I]) is ever written back to HBM (R2.1 / R2.6).
 *   Adds:
 *     - `sm70_mma_accumulate_kchunk_a_lda` — like the 4.2 accumulate helper but
 *       reads the A operand (`h`) from the resident SMEM block at an arbitrary
 *       leading dim (== full intermediate I), so the contraction sub-tile is
 *       gathered straight out of `smem_h` with no HBM round-trip.
 *     - `sm70_store_acc_global` — writes a warp's 16x16 output tile to the
 *       global [M, K] output (same lane->element map as the SwiGLU epilogue).
 *     - `sm70_linear2_downproj` — the double-buffered "load-next / compute-
 *       current" pipeline over the I contraction; only the MXFP4 down weight is
 *       streamed+dequantised (the `h` activation operand is already on chip).
 *     - `sm70_fused_moe_kernel` — the `__global__` tying linear1+SwiGLU ->
 *       linear2 per (expert, m-tile): maps blocks via `expert_offsets`, loops
 *       i_block slabs to build the full h[16][I] in SMEM, then loops the down
 *       projection, writing the final [16, hidden_K] output tile.
 *   The host `sm70_fused_moe_out` below resolves per-expert MXFP4 weight/scale
 *   bases from the [E,...] tensors, sizes the dynamic SMEM blob within the V100
 *   96KB budget (opting into >48KB when needed), and launches with
 *   C10_CUDA_KERNEL_LAUNCH_CHECK. End-to-end numerical equivalence is exercised
 *   on SM70 hardware by the property tests (task 5.5); there is no nvcc / V100
 *   build in this env, so 4.3 is verified by reading + getDiagnostics.
 *
 * Reusable Volta primitives for 4.2/4.3 (not wired here): the first-gen HMMA
 * fragment maps live in `lmdeploy/.../attention/impl_884.h` (the row.col
 * Q*K^T path) and `turbomind::mma_m8n8k4_row_col`
 * (`src/turbomind/kernels/core/mma.h`); the MXFP4 nibble unpack + E8M0 scale
 * semantics this kernel must reproduce in-register are mirrored from
 * `awq_sm70_gemm.cu::unpack_mxfp4_to_u16` and
 * `turbomind::AdjustUe8m0ScaleForHalf` (the `sm70_mxfp4_moe_direct_prepare`
 * preparation path). 4.2/4.3 add the device kernels fresh against the MXFP4
 * contract documented below; end-to-end numerical equivalence is exercised by
 * the property tests (task 5.5) on SM70 hardware (no nvcc / V100 in this env).
 *
 * ===========================================================================
 * MXFP4 expert-weight + block-scale contract (mirrors dsv4f's
 * `DeepseekV4MegaMoEExperts` / `Mxfp4SM70MoEMethod.create_weights`, the exact
 * raw layout task 5.1's weight-prep will hand to this op):
 *
 *   E   = num_experts, K = hidden_K, I = inter_I, gs = group_size (= 32).
 *
 *   linear1 (gate/up) — fused gate|up output of width 2*I:
 *     w13_weight        : uint8  [E, 2*I, K/2]    packed MXFP4 (E2M1), two
 *                                                 4-bit nibbles per byte along
 *                                                 the K (in-feature) axis;
 *                                                 nibble layout matches
 *                                                 `unpack_mxfp4_to_u16`
 *                                                 (low nibble = even K, high
 *                                                 nibble = odd K).
 *     w13_weight_scale  : uint8  [E, 2*I, K/gs]   per-32 E8M0 block scale
 *                                                 (one uint8 exponent per group
 *                                                 of 32 contiguous K elements).
 *     The 2*I output rows are gate rows [0, I) followed by up rows [I, 2*I).
 *
 *   linear2 (down) — output of width K, contraction over I:
 *     w2_weight         : uint8  [E, K, I/2]      packed MXFP4 (E2M1), two
 *                                                 nibbles per byte along the I
 *                                                 (in-feature) axis.
 *     w2_weight_scale   : uint8  [E, K, I/gs]     per-32 E8M0 block scale along
 *                                                 the I axis.
 *
 *   Dequant semantics (per element, reproduced in-kernel by 4.2/4.3):
 *     val_fp16 = e2m1_decode(nibble) * e8m0_decode(block_scale_of_group)
 *   where e2m1_decode maps the 4-bit code to its E2M1 magnitude/sign and
 *   e8m0_decode interprets the uint8 as a power-of-two exponent (E8M0). This
 *   is the same numeric source the per-operator reference uses (R5.1), so the
 *   only differences vs. the baseline are fusion-order rounding effects.
 *
 *   Activations / routing tensors:
 *     out               : fp16   [M, K]           row-major contiguous output.
 *     permuted_input    : fp16   [M_padded, K]    expert-grouped (contiguous)
 *                                                 input; rows are m_block-
 *                                                 aligned per expert segment.
 *     expert_offsets    : int32  [E+1]            per-expert segment starts
 *                                                 into `permuted_input`.
 *     num_experts/hidden_K/inter_I/group_size     int64 dims (group_size==32).
 *     m_block/i_block                             int64 tiling granularities
 *                                                 (token-block / intermediate),
 *                                                 honoured by the 4.3 launcher.
 * ===========================================================================
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cstdint>

// VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize (opt into >48KB dynamic
// SMEM, which the V100 supports up to 96KB/block).
#include "../../cuda_compat.h"

// ===========================================================================
// Task 4.2: linear1 + SwiGLU epilogue DEVICE SEGMENT.
//
// Composable `__device__` building blocks for the SM70 fused MoE kernel that
// task 4.3 drives from its `__global__` launcher. Everything here is device
// side only and pulls in no TurboMind headers, so it is self-contained and
// compiles for the `CUDA_ARCHS "7.0"` target alongside the host op below.
//
// Pipeline shape (per warp): one warp computes a single 16x16 `gate`/`up`
// output tile (M_TILE x N_TILE) via the first-generation Volta HMMA
// `mma.sync.m8n8k4.row.col`, accumulating over the K contraction in K_STEP
// (== MXFP4 group_size == 32) chunks. A classic double-buffered SMEM pipeline
// ("load-next / compute-current", `__syncthreads`-separated, NO cp.async/TMA)
// stages the activation tile and the MXFP4-dequantised gate/up weight tiles.
// The on-chip `silu(gate)*up` epilogue writes the `h` block into shared memory
// for the linear2 segment (task 4.3) to consume; `gate`/`up` never touch HBM.
// ===========================================================================
namespace vllm {
namespace sm70_fused_moe {

// --- Tile / contraction constants ------------------------------------------
// First-gen Volta HMMA `mma.sync.m8n8k4` computes, per warp, a 16x16 output
// tile (OP_M=OP_N=16) accumulated over OP_K=4 at a time (see the impl_884
// fragment maps in lmdeploy/.../attention/impl_884.h). We contract the K axis
// in K_STEP=32 chunks so each chunk maps onto exactly one MXFP4 per-32 block
// scale (one E8M0 exponent covers the whole chunk for a given weight row).
constexpr int kWarpSize = 32;
constexpr int kMmaM = 16;     // tile rows  (tokens, M)
constexpr int kMmaN = 16;     // tile cols  (output channels, N)
constexpr int kKStep = 32;    // K contraction chunk == MXFP4 group_size
constexpr int kK4PerStep = kKStep / 4;  // m8n8k4 sub-steps per K_STEP chunk

// --- MXFP4 (E2M1 + per-32 E8M0) -> fp16 software dequant -------------------
// These reproduce, bit-for-bit, `mxfp4_dequant_to_fp16` in
// vllm/model_executor/layers/fused_moe/sm70_moe_reference.py (the single
// shared numeric source, R5.1):
//   * E2M1 magnitude LUT {0,0.5,1,1.5,2,3,4,6} indexed by the low 3 bits,
//     sign = bit 3 (1 -> negative).
//   * E8M0 block scale = 2^(raw - 127); computed with single-precision exp2f
//     so the raw==255 overflow yields +inf and `0 * inf -> NaN`, exactly
//     matching the reference (verified against all 256 nibble bytes x a spread
//     of E8M0 exponents, including the 0*inf=NaN boundary).
//   * val_fp16 = __float2half( e2m1_signed(nibble) * e8m0_scale(raw) ).

__device__ __forceinline__ float sm70_e2m1_signed(uint32_t nibble) {
  // Magnitude for the 3-bit code; sign from bit 3.
  const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float mag = kMag[nibble & 0x7u];
  return (nibble & 0x8u) ? -mag : mag;
}

__device__ __forceinline__ float sm70_e8m0_scale(uint32_t raw) {
  // 2^(raw-127); single-precision exp2f (raw==255 -> +inf -> 0*inf=NaN).
  return exp2f(static_cast<float>(raw) - 127.0f);
}

__device__ __forceinline__ half sm70_mxfp4_decode_nibble(uint32_t nibble,
                                                         float scale) {
  return __float2half(sm70_e2m1_signed(nibble) * scale);
}

// --- First-gen Volta HMMA: D = A * B^T + C (m8n8k4, row.col, fp16->fp32) ----
// Local copy of turbomind::mma_m8n8k4_row_col (src/turbomind/kernels/core/
// mma.h) using plain register arrays so this file needs no TurboMind headers.
// A: Array<half,4> (2x b32 regs); B: Array<half,4>; C/D: Array<float,8>.
__device__ __forceinline__ void sm70_mma_m8n8k4_row_col(
    float (&d)[8], const uint32_t (&a)[2], const uint32_t (&b)[2],
    const float (&c)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  // clang-format off
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11},"
      "{%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
        "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
      : "r"(a[0]), "r"(a[1]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
  // clang-format on
#else
  // Host / non-SM70 path: never executed on the V100 target, present so the
  // helper is well-formed under any compiler.
#pragma unroll
  for (int i = 0; i < 8; ++i) d[i] = c[i];
#endif
}

// --- Warp-tile fragment <-> lane maps (mirror impl_884 row.col path) --------
// A operand = activation `x` tile, row-major [kMmaM, kKStep] in SMEM.
// Mirrors impl_884 TransformQ: lane -> (row, k4*4) reading 4 contiguous halfs.
__device__ __forceinline__ void sm70_load_a_frag(uint32_t (&a)[2],
                                                 const half* smem_x, int ldx,
                                                 int lane, int k4) {
  const int row = (lane & 8) + (lane & 3) + (lane >> 4) * 4;  // 0..15
  const int col = k4 * 4;
  const half* p = smem_x + row * ldx + col;
  a[0] = *reinterpret_cast<const uint32_t*>(p);      // halfs col,   col+1
  a[1] = *reinterpret_cast<const uint32_t*>(p + 2);  // halfs col+2, col+3
}

// B operand = weight tile, row-major [kMmaN, kKStep] in SMEM (N = output
// channel). Mirrors impl_884 StateQK::Load: lane -> (n_row, k4*4).
__device__ __forceinline__ void sm70_load_b_frag(uint32_t (&b)[2],
                                                 const half* smem_w, int ldw,
                                                 int lane, int k4) {
  const int n_row = (lane >> 4) * 4 + (lane & 4) * 2 + (lane & 3);  // 0..15
  const int col = k4 * 4;
  const half* p = smem_w + n_row * ldw + col;
  b[0] = *reinterpret_cast<const uint32_t*>(p);
  b[1] = *reinterpret_cast<const uint32_t*>(p + 2);
}

// Accumulator element e (0..7) -> (row, col) within the warp's 16x16 tile.
// Mirrors impl_884 ForeachS: e == s1*4 + q*2 + s0.
__device__ __forceinline__ void sm70_acc_coord(int lane, int e, int& row,
                                               int& col) {
  const int s0 = e & 1;
  const int q = (e >> 1) & 1;
  const int s1 = (e >> 2) & 1;
  row = (lane & 8) + (lane & 1) + (lane >> 4) * 4 + q * 2;        // 0..15
  col = (lane & 4) * 2 + (lane & 2) + s1 * 4 + s0;               // 0..15
}

// --- Accumulate one K_STEP chunk for a warp's 16x16 tile --------------------
// acc[8] += sum over the staged K_STEP chunk of x_tile (16 x kKStep) times
// w_tile (16 x kKStep), both row-major in SMEM with leading dim kKStep.
__device__ __forceinline__ void sm70_mma_accumulate_kchunk(
    float (&acc)[8], const half* smem_x, const half* smem_w, int lane) {
#pragma unroll
  for (int k4 = 0; k4 < kK4PerStep; ++k4) {
    uint32_t a[2];
    uint32_t b[2];
    sm70_load_a_frag(a, smem_x, kKStep, lane, k4);
    sm70_load_b_frag(b, smem_w, kKStep, lane, k4);
    float d[8];
    sm70_mma_m8n8k4_row_col(d, a, b, acc);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = d[i];
  }
}

// --- SwiGLU epilogue: silu(gate) * up, on-chip into the `h` SMEM block ------
__device__ __forceinline__ float sm70_silu(float x) {
  // silu(x) = x * sigmoid(x) = x / (1 + e^-x); fp32 then stored fp16, matching
  // the reference's silu_and_mul data flow within the design fp16 tolerance.
  return x / (1.0f + expf(-x));
}

// Write this warp's 16x16 tile of h = silu(gate)*up into smem_h at column
// offset `h_col0` (row-major, leading dim ldh). Padded rows/cols (beyond the
// valid m_rows / n_cols of a partial tile) are skipped so `h` stays clean.
//
// `swiglu_limit` mirrors the reference / TurboMind ``sm70_fused_swiglu_limit``
// clamp: when > 0, gate is clamped to <= limit and up to [-limit, +limit]
// BEFORE silu/mul (DeepSeek-V4 style SwiGLU limit). limit <= 0 disables it
// (plain silu(gate)*up), preserving the original behaviour exactly.
__device__ __forceinline__ void sm70_swiglu_epilogue(
    const float (&gate_acc)[8], const float (&up_acc)[8], half* smem_h, int ldh,
    int h_col0, int m_rows, int n_cols, int lane, float swiglu_limit) {
  const bool has_limit = swiglu_limit > 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int row;
    int col;
    sm70_acc_coord(lane, e, row, col);
    if (row < m_rows && col < n_cols) {
      float gate = gate_acc[e];
      float up = up_acc[e];
      if (has_limit) {
        // gate = min(gate, limit); up = clamp(up, -limit, +limit).
        gate = fminf(gate, swiglu_limit);
        up = fmaxf(fminf(up, swiglu_limit), -swiglu_limit);
      }
      const float h = sm70_silu(gate) * up;
      smem_h[row * ldh + h_col0 + col] = __float2half(h);
    }
  }
}

// --- Synchronous, block-cooperative SMEM staging (no cp.async / TMA) --------
// Stage a [kMmaM, kKStep] activation tile from global `x` (row-major
// [M, K], leading dim x_row_stride) at (row0, k0) into smem_x. Out-of-range
// rows/cols are zero-filled so padded tiles contribute 0 to the mma.
__device__ __forceinline__ void sm70_stage_x_kchunk(half* smem_x, const half* x,
                                                    int x_row_stride, int row0,
                                                    int k0, int m_rows, int K,
                                                    int tid, int nthreads) {
  const int total = kMmaM * kKStep;
  for (int idx = tid; idx < total; idx += nthreads) {
    const int r = idx / kKStep;
    const int c = idx % kKStep;
    half v = __float2half(0.0f);
    if (r < m_rows && (k0 + c) < K) {
      v = x[static_cast<long long>(row0 + r) * x_row_stride + (k0 + c)];
    }
    smem_x[idx] = v;
  }
}

// Stage a [n_tile, kKStep] MXFP4 weight slab and dequantise it to fp16 in
// SMEM (row-major, leading dim kKStep). `w_packed` is the expert's [N, K/2]
// packed-E2M1 base (row stride w_row_stride bytes); `w_scale` is its
// [N, K/group_size] E8M0 base (row stride s_row_stride). One E8M0 exponent
// (group index k0/group_size) covers the whole kKStep chunk for a given output
// row. `n0` is the global output channel of slab row 0; `n_valid` is the number
// of slab rows that map to real output channels (rows [n_valid, n_tile) and
// in-feature columns >= K are zero-filled so padded lanes contribute 0 to the
// mma). Block-cooperative: every thread strides over (tid, nthreads).
__device__ __forceinline__ void sm70_stage_w_kchunk_dequant(
    half* smem_w, const uint8_t* w_packed, const uint8_t* w_scale,
    int w_row_stride, int s_row_stride, int n0, int k0, int n_tile, int n_valid,
    int K, int group_size, int tid, int nthreads) {
  const int bytes_per_chunk = kKStep / 2;            // 16 bytes -> 32 nibbles
  const int total = n_tile * bytes_per_chunk;
  const int kbyte0 = k0 / 2;
  const int sgroup = k0 / group_size;                // block-scale group index
  for (int idx = tid; idx < total; idx += nthreads) {
    const int r = idx / bytes_per_chunk;             // output channel in slab
    const int j = idx % bytes_per_chunk;             // byte within chunk
    half lo = __float2half(0.0f);
    half hi = __float2half(0.0f);
    if (r < n_valid && (k0 + 2 * j) < K) {
      const int n_global = n0 + r;
      const uint8_t byte =
          w_packed[static_cast<long long>(n_global) * w_row_stride + kbyte0 + j];
      const float scale = sm70_e8m0_scale(
          w_scale[static_cast<long long>(n_global) * s_row_stride + sgroup]);
      lo = sm70_mxfp4_decode_nibble(byte & 0x0Fu, scale);   // even K element
      hi = sm70_mxfp4_decode_nibble(byte >> 4, scale);      // odd  K element
    }
    smem_w[r * kKStep + 2 * j] = lo;
    smem_w[r * kKStep + 2 * j + 1] = hi;
  }
}

// --- Per-expert tile parameters fed by the task-4.3 launcher ----------------
struct Sm70Linear1Params {
  const half* x;            // permuted_input base (row-major [M, K])
  int x_row_stride;         // elements per row (== K)
  const uint8_t* w13_packed;  // expert slice base, packed MXFP4 [2*I, K/2]
  const uint8_t* w13_scale;   // expert slice base, E8M0       [2*I, K/group]
  int w13_row_stride;       // bytes per output row (== K/2)
  int w13_scale_stride;     // scales per output row (== K/group_size)
  int K;                    // hidden (linear1 contraction depth)
  int I;                    // intermediate (gate rows [0,I), up rows [I,2I))
  int group_size;           // MXFP4 block size (== 32 == kKStep)
};

// --- Composed linear1 + SwiGLU epilogue (block-cooperative, one m-tile) -----
// Classic double-buffered "load-next / compute-current" pipeline over the K
// contraction (two SMEM buffers, `__syncthreads()`-separated, NO cp.async/TMA)
// that leaves the resulting `h` (silu(gate)*up) block resident in `smem_h` for
// task 4.3's linear2 segment. The `gate`/`up` accumulators stay in registers
// and are never written to HBM (R2.1 / R2.6).
//
// Block decomposition (the launch contract task 4.3 honours):
//   * The block processes ONE m-tile: 16 tokens at rows [row0, row0+m_rows),
//     m_rows = min(16, segment_rows - row0).
//   * The block has `nwarps = nthreads/32` warps; together they compute one
//     i_block of I_b = nwarps*16 intermediate channels starting at `i0`.
//     Warp `warp_id` owns the 16-wide n-tile at local columns
//     [warp_id*16, warp_id*16+16) -> global gate channels [i0+warp_id*16, ...)
//     and the matching up channels [I + i0 + warp_id*16, ...).
//
// SMEM the caller (task 4.3) must provide (fp16):
//   smem_x  : 2 * (kMmaM * kKStep)        double-buffered shared activation tile
//   smem_wg : 2 * (I_b   * kKStep)        double-buffered dequantised gate slab
//   smem_wu : 2 * (I_b   * kKStep)        double-buffered dequantised up   slab
//   smem_h  : kMmaM * ldh   (ldh >= I_b)  persistent h block (consumed later)
// Every thread of the block must call this uniformly (it contains
// `__syncthreads()` and `p.K / kKStep` is identical for all threads). The
// caller MUST `__syncthreads()` after this returns before reading `smem_h`
// across warps (warp w writes only its own columns).
__device__ inline void sm70_linear1_swiglu_epilogue(
    const Sm70Linear1Params& p, int row0, int i0, int m_rows, int i_block,
    half* smem_x, half* smem_wg, half* smem_wu, half* smem_h, int ldh, int lane,
    int warp_id, int tid, int nthreads, float swiglu_limit) {
  const int x_slot = kMmaM * kKStep;        // elements per x double-buffer slot
  const int w_slot = i_block * kKStep;      // elements per weight slab slot
  const int num_chunks = p.K / kKStep;
  const int n_valid = max(0, min(i_block, p.I - i0));   // real channels in slab
  const int n_local0 = warp_id * kMmaN;                 // this warp's columns
  const int n_cols = max(0, min(kMmaN, n_valid - n_local0));

  float gate_acc[8];
  float up_acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    gate_acc[i] = 0.0f;
    up_acc[i] = 0.0f;
  }

  // Stage chunk 0 into buffer slot 0 (block-cooperative).
  sm70_stage_x_kchunk(smem_x, p.x, p.x_row_stride, row0, /*k0=*/0, m_rows, p.K,
                      tid, nthreads);
  sm70_stage_w_kchunk_dequant(smem_wg, p.w13_packed, p.w13_scale,
                              p.w13_row_stride, p.w13_scale_stride, /*n0=*/i0,
                              /*k0=*/0, i_block, n_valid, p.K, p.group_size, tid,
                              nthreads);
  sm70_stage_w_kchunk_dequant(smem_wu, p.w13_packed, p.w13_scale,
                              p.w13_row_stride, p.w13_scale_stride,
                              /*n0=*/p.I + i0, /*k0=*/0, i_block, n_valid, p.K,
                              p.group_size, tid, nthreads);
  __syncthreads();

  for (int kc = 0; kc < num_chunks; ++kc) {
    const int cur = kc & 1;
    const int nxt = cur ^ 1;

    // Load-next: stage chunk kc+1 into the *other* buffer (disjoint from the
    // buffer being computed this iteration -> no hazard before the sync).
    if (kc + 1 < num_chunks) {
      const int k0n = (kc + 1) * kKStep;
      sm70_stage_x_kchunk(smem_x + nxt * x_slot, p.x, p.x_row_stride, row0, k0n,
                          m_rows, p.K, tid, nthreads);
      sm70_stage_w_kchunk_dequant(smem_wg + nxt * w_slot, p.w13_packed,
                                  p.w13_scale, p.w13_row_stride,
                                  p.w13_scale_stride, i0, k0n, i_block, n_valid,
                                  p.K, p.group_size, tid, nthreads);
      sm70_stage_w_kchunk_dequant(smem_wu + nxt * w_slot, p.w13_packed,
                                  p.w13_scale, p.w13_row_stride,
                                  p.w13_scale_stride, p.I + i0, k0n, i_block,
                                  n_valid, p.K, p.group_size, tid, nthreads);
    }

    // Compute-current: accumulate this warp's 16x16 gate and up tiles from the
    // shared x tile and this warp's slice of the weight slabs.
    if (n_cols > 0) {
      const half* x_cur = smem_x + cur * x_slot;
      const half* wg_cur = smem_wg + cur * w_slot + n_local0 * kKStep;
      const half* wu_cur = smem_wu + cur * w_slot + n_local0 * kKStep;
      sm70_mma_accumulate_kchunk(gate_acc, x_cur, wg_cur, lane);
      sm70_mma_accumulate_kchunk(up_acc, x_cur, wu_cur, lane);
    }

    // Reuse-safety: the next buffer is fully staged and the current buffer is
    // done being read before it is overwritten two iterations later.
    __syncthreads();
  }

  // On-chip SwiGLU: silu(gate)*up -> smem_h (stays resident for linear2).
  if (n_cols > 0) {
    sm70_swiglu_epilogue(gate_acc, up_acc, smem_h, ldh, /*h_col0=*/n_local0,
                         m_rows, n_cols, lane, swiglu_limit);
  }
}

// ===========================================================================
// Task 4.3: linear2 (down projection) DEVICE SEGMENT + on-chip hand-off.
//
// linear2 contracts the intermediate `h` ([M_b, I]) against the MXFP4 down
// weight `w2` ([K_out, I]) to produce the [M_b, K_out] expert output. The key
// fusion property (R2.1 / R2.6): `h` is consumed DIRECTLY from `smem_h` (where
// task 4.2's `sm70_linear1_swiglu_epilogue` left it) — neither `gate/up`
// ([M_b, 2*I]) nor `h` ([M_b, I]) is ever written to / read from HBM. Only the
// MXFP4 down weight is streamed in (double-buffered + dequantised on the way
// into SMEM), exactly mirroring the linear1 pipeline; the activation operand
// is already on chip.
//
// Warp/tile contract (identical decomposition to linear1):
//   * The block processes ONE m-tile: 16 tokens at rows [row0, row0+m_rows).
//   * `nwarps = nthreads/32` warps together compute one k_block of
//     K_b = nwarps*16 OUTPUT channels starting at `k_out0`. Warp `warp_id`
//     owns the 16-wide n-tile at output channels [k_out0+warp_id*16, ...).
//   * The contraction over I is done in kKStep (== 32 == MXFP4 group_size)
//     chunks; one E8M0 exponent covers each chunk per output row.
// ===========================================================================

// Accumulate one kKStep contraction chunk for linear2. Same as
// `sm70_mma_accumulate_kchunk` except the A operand (`h`) lives in the resident
// `smem_h` block with an arbitrary leading dim `lda` (== the full intermediate
// I, NOT kKStep), so its per-chunk sub-tile is read with stride `lda`; the B
// operand (dequantised down-weight slab) is row-major [n_tile, kKStep] exactly
// like linear1. `smem_a` is the chunk base (== smem_h + kc*kKStep).
__device__ __forceinline__ void sm70_mma_accumulate_kchunk_a_lda(
    float (&acc)[8], const half* smem_a, int lda, const half* smem_w,
    int lane) {
#pragma unroll
  for (int k4 = 0; k4 < kK4PerStep; ++k4) {
    uint32_t a[2];
    uint32_t b[2];
    sm70_load_a_frag(a, smem_a, lda, lane, k4);
    sm70_load_b_frag(b, smem_w, kKStep, lane, k4);
    float d[8];
    sm70_mma_m8n8k4_row_col(d, a, b, acc);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = d[i];
  }
}

// Store this warp's accumulated 16x16 output tile to the global output.
// `acc[e]` maps to (row, col) within the tile via `sm70_acc_coord` (the same
// lane->element map the SwiGLU epilogue uses), and lands at global
// out[(row0+row) * out_ld + (col_base+col)]. Rows >= m_rows and cols >= n_cols
// (partial token / output tiles) are skipped so padded lanes never write.
__device__ __forceinline__ void sm70_store_acc_global(const float (&acc)[8],
                                                      half* out, int out_ld,
                                                      int row0, int m_rows,
                                                      int col_base, int n_cols,
                                                      int lane) {
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int row;
    int col;
    sm70_acc_coord(lane, e, row, col);
    if (row < m_rows && col < n_cols) {
      out[static_cast<long long>(row0 + row) * out_ld + (col_base + col)] =
          __float2half(acc[e]);
    }
  }
}

// --- Per-expert linear2 tile parameters fed by the kernel -------------------
struct Sm70Linear2Params {
  const uint8_t* w2_packed;  // expert slice base, packed MXFP4 [K_out, I/2]
  const uint8_t* w2_scale;   // expert slice base, E8M0       [K_out, I/group]
  int w2_row_stride;         // bytes per output row (== I/2)
  int w2_scale_stride;       // scales per output row (== I/group_size)
  int K_out;                 // hidden (linear2 output width == hidden_K)
  int I;                     // intermediate (linear2 contraction depth)
  int group_size;            // MXFP4 block size (== 32 == kKStep)
};

// --- linear2 down projection for one k_block output slab --------------------
// Classic double-buffered "load-next / compute-current" pipeline over the I
// contraction (two SMEM buffers, `__syncthreads()`-separated, NO cp.async/TMA),
// reading the A operand (`h`) STRAIGHT FROM `smem_h` (on-chip hand-off, no HBM
// round-trip) and streaming the MXFP4 down weight (dequantised into `smem_wd`).
// Writes this warp's [16, 16] slice of `out`. Every thread of the block must
// call this uniformly (it contains `__syncthreads()` and `p.I / kKStep` is
// identical for all threads).
//
// SMEM the caller provides (fp16):
//   smem_wd : 2 * (k_block * kKStep)   double-buffered dequantised down slab
// `smem_h` (ldh == I) is the resident intermediate; it is only read here.
__device__ inline void sm70_linear2_downproj(
    const Sm70Linear2Params& p, int row0, int k_out0, int m_rows, int k_block,
    const half* smem_h, int ldh, half* smem_wd, half* out, int out_ld, int lane,
    int warp_id, int tid, int nthreads) {
  const int wd_slot = k_block * kKStep;        // elements per down-slab slot
  const int num_chunks = p.I / kKStep;         // I is a multiple of kKStep
  const int n_valid = max(0, min(k_block, p.K_out - k_out0));  // real out chans
  const int n_local0 = warp_id * kMmaN;        // this warp's output columns
  const int n_cols = max(0, min(kMmaN, n_valid - n_local0));

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

  // Stage I-chunk 0 of the down-weight slab into buffer slot 0. n0 == k_out0
  // selects the down output-channel rows; k0 == 0 is the contraction origin
  // along I; the helper's `K` argument is the contraction extent (== I).
  sm70_stage_w_kchunk_dequant(smem_wd, p.w2_packed, p.w2_scale, p.w2_row_stride,
                              p.w2_scale_stride, /*n0=*/k_out0, /*k0=*/0,
                              k_block, n_valid, /*K=*/p.I, p.group_size, tid,
                              nthreads);
  __syncthreads();

  for (int kc = 0; kc < num_chunks; ++kc) {
    const int cur = kc & 1;
    const int nxt = cur ^ 1;

    // Load-next: stage I-chunk kc+1 into the *other* buffer.
    if (kc + 1 < num_chunks) {
      const int i0n = (kc + 1) * kKStep;
      sm70_stage_w_kchunk_dequant(smem_wd + nxt * wd_slot, p.w2_packed,
                                  p.w2_scale, p.w2_row_stride, p.w2_scale_stride,
                                  /*n0=*/k_out0, /*k0=*/i0n, k_block, n_valid,
                                  /*K=*/p.I, p.group_size, tid, nthreads);
    }

    // Compute-current: contract this warp's 16x16 output tile against the h
    // sub-tile [16, kKStep] (read directly from smem_h at chunk offset).
    if (n_cols > 0) {
      const half* h_cur = smem_h + kc * kKStep;  // h[:, kc*kKStep] @ ldh == I
      const half* wd_cur = smem_wd + cur * wd_slot + n_local0 * kKStep;
      sm70_mma_accumulate_kchunk_a_lda(acc, h_cur, ldh, wd_cur, lane);
    }

    __syncthreads();
  }

  // Write the [16, 16] output tile (col_base = k_out0 + this warp's columns).
  if (n_cols > 0) {
    sm70_store_acc_global(acc, out, out_ld, row0, m_rows,
                          /*col_base=*/k_out0 + n_local0, n_cols, lane);
  }
}

// ===========================================================================
// Fused MoE mega-kernel: linear1 -> SwiGLU -> linear2 for one (expert,
// m_block) tile, fully on chip. One block handles the kMmaM (== 16) rows
// starting at blockIdx.x * kMmaM of the contiguous permuted input; it resolves
// its expert from `expert_offsets`, builds the FULL intermediate h[16][I] in
// shared memory (gate/up never leave SMEM), then runs linear2 reading h
// STRAIGHT FROM SMEM and writes out[16, K_out] — neither gate/up ([M_b, 2*I])
// nor h ([M_b, I]) is ever written back to HBM (R2.1 / R2.6).
//
// On-chip hand-off / SMEM-residency choice (R2.1 / R2.6):
//   The design frames the intermediate as a tiled [M_b, I_b] block. Because
//   linear2 contracts over the *full* intermediate dim I, fusing without an
//   HBM round-trip requires either (A) the full h[16][I] resident while
//   linear2 tiles its K_out output, or (B) a tiled h[16][I_b] with the full
//   linear2 output accumulator [16][K_out] resident across the I-loop. (B)
//   needs an accumulator spanning all of K_out (infeasible for large hidden,
//   and equal SMEM cost if spilled), so we take (A): keep the full h on chip
//   and tile the linear2 output over K_b. Both keep gate/up ([M_b, 2*I]) and h
//   ([M_b, I]) entirely in SMEM/registers — the HBM round-trip the fusion
//   eliminates — so both satisfy R2.1 / R2.6; (A) simply trades the design's
//   `M_b*I_b(h)` budget term for `kMmaM*I`, which the host budget check below
//   accounts for exactly.
//
// Block / warp decomposition (the launch contract the host honours):
//   * blockDim.x == nwarps * 32; the slab width is I_b = K_b = nwarps * kMmaN
//     (linear1 intermediate slab == linear2 output slab). The kernel loops i0
//     over [0, I) and k0 over [0, K_out) in steps of that slab width; the last
//     (ragged) slab is handled by the staging/epilogue n_valid clamps.
//   * contiguous segments are m_block-aligned and kMmaM | m_block (host-
//     checked), so an m-tile block never straddles two experts.
//
// Dynamic shared memory layout (single blob, host-validated <= 96KB):
//   [ smem_h : kMmaM * I ]  [ scratch : max(linear1, linear2) ]
// The linear1 scratch (double-buffered x + gate/up weight slabs) and the
// linear2 scratch (double-buffered down-weight slab) are time-disjoint and
// therefore OVERLAID (`scratch` is reused); smem_h is resident across both.
__global__ void sm70_fused_moe_kernel(
    half* __restrict__ out, int out_ld,
    const half* __restrict__ permuted_input, int x_ld,
    const int* __restrict__ expert_offsets,
    const uint8_t* __restrict__ w13_weight,
    const uint8_t* __restrict__ w13_scale,
    const uint8_t* __restrict__ w2_weight, const uint8_t* __restrict__ w2_scale,
    int num_experts, int K, int I, int group_size, int total_rows,
    float swiglu_limit) {
  // Capture-safe (expert, tile) grid: block (blockIdx.x = tile, blockIdx.y =
  // expert) owns the kMmaM-row tile starting at expert_offsets[e] + tile*kMmaM,
  // confined to expert e's segment. This needs NO m_block alignment of the
  // segments (a tile never straddles two experts because the expert is fixed by
  // blockIdx.y and the row range is clamped to [seg_start, seg_end)), so it
  // consumes the dense, *unaligned* `moe_permute` output of the capture-safe
  // production buffer flow directly. The grid shape (max_tiles x num_experts)
  // depends only on the fixed buffer capacity, not the routing distribution, so
  // it stays constant across CUDA-graph replays.
  const int expert = blockIdx.y;
  if (expert >= num_experts) return;
  const int seg_start = expert_offsets[expert];
  const int seg_end = expert_offsets[expert + 1];
  const int row0 = seg_start + blockIdx.x * kMmaM;
  if (row0 >= seg_end || row0 >= total_rows) return;  // padding / empty tail
  const int m_rows = min(kMmaM, min(seg_end, total_rows) - row0);
  if (m_rows <= 0) return;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid >> 5;
  const int nwarps = nthreads >> 5;
  const int slab = nwarps * kMmaN;  // I_b (linear1) == K_b (linear2)

  // ---- Partition the dynamic SMEM blob ------------------------------------
  extern __shared__ half smem[];
  const int ldh = I;                 // h leading dim == full intermediate
  half* smem_h = smem;               // [kMmaM][I], resident across segments
  half* scratch = smem + kMmaM * ldh;
  const int x_slot = kMmaM * kKStep;    // elements per x double-buffer slot
  const int w_slot = slab * kKStep;     // elements per weight slab slot
  // linear1 sub-tiles inside `scratch`:
  half* smem_x = scratch;
  half* smem_wg = smem_x + 2 * x_slot;
  half* smem_wu = smem_wg + 2 * w_slot;
  // linear2 sub-tile overlays `scratch` (time-disjoint from linear1):
  half* smem_wd = scratch;

  // Zero the unused tail rows [m_rows, kMmaM) of h so a partial token tile can
  // never feed stale/NaN SMEM into the linear2 contraction. (Valid rows are
  // fully written by linear1 across the i0 loop; mma rows are independent so
  // tail garbage would only touch unstored output rows, but we zero anyway to
  // keep the on-chip h pristine — R5.4.)
  if (m_rows < kMmaM) {
    for (int idx = tid; idx < (kMmaM - m_rows) * I; idx += nthreads) {
      const int rr = m_rows + idx / I;
      const int cc = idx % I;
      smem_h[rr * ldh + cc] = __float2half(0.0f);
    }
    __syncthreads();
  }

  // ---- Resolve per-expert MXFP4 weight / scale bases (contiguous [E,...]) --
  Sm70Linear1Params l1;
  l1.x = permuted_input;
  l1.x_row_stride = x_ld;
  l1.w13_packed =
      w13_weight + static_cast<long long>(expert) * (2LL * I) * (K / 2);
  l1.w13_scale =
      w13_scale + static_cast<long long>(expert) * (2LL * I) * (K / group_size);
  l1.w13_row_stride = K / 2;
  l1.w13_scale_stride = K / group_size;
  l1.K = K;
  l1.I = I;
  l1.group_size = group_size;

  // ---- Segment 1: build the full h[kMmaM][I] on chip (gate/up resident) ----
  // Each i0 slab's SwiGLU output is written into the resident h buffer at its
  // column offset (pass smem_h + i0, ldh == I), so gate/up never leave SMEM.
  for (int i0 = 0; i0 < I; i0 += slab) {
    sm70_linear1_swiglu_epilogue(l1, row0, i0, m_rows, slab, smem_x, smem_wg,
                                 smem_wu, smem_h + i0, ldh, lane, warp_id, tid,
                                 nthreads, swiglu_limit);
    __syncthreads();  // h slab complete + scratch free before reuse / linear2
  }

  // ---- Segment 2: linear2 reads h straight from SMEM (on-chip hand-off) ----
  Sm70Linear2Params l2;
  l2.w2_packed =
      w2_weight + static_cast<long long>(expert) * static_cast<long long>(K) *
                      (I / 2);
  l2.w2_scale = w2_scale + static_cast<long long>(expert) *
                               static_cast<long long>(K) * (I / group_size);
  l2.w2_row_stride = I / 2;
  l2.w2_scale_stride = I / group_size;
  l2.K_out = K;
  l2.I = I;
  l2.group_size = group_size;

  for (int k0 = 0; k0 < K; k0 += slab) {
    sm70_linear2_downproj(l2, row0, k0, m_rows, slab, smem_h, ldh, smem_wd, out,
                          out_ld, lane, warp_id, tid, nthreads);
    __syncthreads();  // reuse the down-weight scratch for the next K slab
  }
}

}  // namespace sm70_fused_moe
}  // namespace vllm

// SM70 fused MoE forward (contiguous / grouped layout), dsv4f MXFP4 weights.
//
// See the file-level "MXFP4 expert-weight + block-scale contract" block for
// the full tensor layout / dtype specification. After validating the MXFP4
// contract via TORCH_CHECK, this op launches the fused linear1 -> SwiGLU ->
// linear2 mega-kernel (`vllm::sm70_fused_moe::sm70_fused_moe_kernel`): MXFP4 ->
// fp16 software dequant + first-gen `mma.sync`, with the `gate/up` and `h`
// intermediates kept entirely on chip (tasks 4.2 + 4.3).
//
//   out               : [M, hidden_K]               fp16, row-major contiguous
//   permuted_input    : [M_padded, hidden_K]        fp16 (expert-grouped)
//   expert_offsets    : [num_experts + 1]           int32 segment starts
//   w13_weight        : [E, 2*inter_I, hidden_K/2]  uint8 packed MXFP4 (gate/up)
//   w13_weight_scale  : [E, 2*inter_I, hidden_K/gs] uint8 E8M0 block scale
//   w2_weight         : [E, hidden_K, inter_I/2]    uint8 packed MXFP4 (down)
//   w2_weight_scale   : [E, hidden_K, inter_I/gs]   uint8 E8M0 block scale
//   num_experts       : number of experts (E)
//   hidden_K          : hidden dim K (linear1 K, linear2 N)
//   inter_I           : intermediate dim I (linear1 N/2 per gate|up, linear2 K)
//   group_size        : MXFP4 block size along the in-feature axis (== 32)
//   m_block           : token-block tiling granularity (M_b)
//   i_block           : intermediate tiling granularity (I_b)
void sm70_fused_moe_out(
    torch::Tensor out,
    torch::Tensor permuted_input,
    torch::Tensor expert_offsets,
    torch::Tensor w13_weight,
    torch::Tensor w13_weight_scale,
    torch::Tensor w2_weight,
    torch::Tensor w2_weight_scale,
    int64_t num_experts,
    int64_t hidden_K,
    int64_t inter_I,
    int64_t group_size,
    int64_t m_block,
    int64_t i_block,
    double swiglu_limit) {
  // ---- Tensor placement / dtype checks (fp16 activations, uint8 MXFP4) -----
  TORCH_CHECK(permuted_input.is_cuda() &&
                  permuted_input.scalar_type() == torch::kFloat16,
              "sm70_fused_moe: permuted_input must be CUDA float16.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "sm70_fused_moe: out must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32,
              "sm70_fused_moe: expert_offsets must be CUDA int32.");
  TORCH_CHECK(w13_weight.is_cuda() && w13_weight_scale.is_cuda() &&
                  w2_weight.is_cuda() && w2_weight_scale.is_cuda(),
              "sm70_fused_moe: MXFP4 weight / scale tensors must be CUDA.");
  TORCH_CHECK(w13_weight.scalar_type() == torch::kUInt8 &&
                  w2_weight.scalar_type() == torch::kUInt8,
              "sm70_fused_moe: packed MXFP4 weights must be uint8 (E2M1, 2 "
              "nibbles/byte).");
  TORCH_CHECK(w13_weight_scale.scalar_type() == torch::kUInt8 &&
                  w2_weight_scale.scalar_type() == torch::kUInt8,
              "sm70_fused_moe: MXFP4 block scales must be uint8 (E8M0).");

  // ---- Dimension sanity ----------------------------------------------------
  TORCH_CHECK(num_experts > 0 && hidden_K > 0 && inter_I > 0,
              "sm70_fused_moe: invalid dimensions (num_experts=", num_experts,
              ", hidden_K=", hidden_K, ", inter_I=", inter_I, ").");
  TORCH_CHECK(m_block > 0 && i_block > 0,
              "sm70_fused_moe: tiling params must be positive (m_block=",
              m_block, ", i_block=", i_block, ").");
  TORCH_CHECK(expert_offsets.numel() == num_experts + 1,
              "sm70_fused_moe: expert_offsets must have num_experts+1 entries.");

  // ---- MXFP4 block size: group_size must be the per-32 micro-scaling block -
  TORCH_CHECK(group_size == 32,
              "sm70_fused_moe: MXFP4 requires group_size == 32, got ",
              group_size, ".");

  // ---- HMMA alignment: K % 8 == 0, N = 2*inter_I % 8 == 0 ------------------
  // linear1: K = hidden_K, N = 2 * inter_I (gate/up).
  // linear2: K = inter_I,  N = hidden_K.
  TORCH_CHECK((hidden_K % 8) == 0,
              "sm70_fused_moe: hidden_K must be a multiple of 8, got ",
              hidden_K, ".");
  TORCH_CHECK((inter_I % 8) == 0,
              "sm70_fused_moe: inter_I must be a multiple of 8, got ", inter_I,
              ".");
  const int64_t gate_up_dim = 2 * inter_I;  // linear1 output width (N)
  TORCH_CHECK((gate_up_dim % 8) == 0,
              "sm70_fused_moe: gate/up dim (N = 2*inter_I) must be a multiple "
              "of 8, got ", gate_up_dim, ".");

  // ---- MXFP4 group alignment along the in-feature axes (K % gs, I % gs) ----
  TORCH_CHECK((hidden_K % group_size) == 0,
              "sm70_fused_moe: hidden_K must be divisible by group_size (",
              hidden_K, " % ", group_size, ").");
  TORCH_CHECK((inter_I % group_size) == 0,
              "sm70_fused_moe: inter_I must be divisible by group_size (",
              inter_I, " % ", group_size, ").");

  // ---- SwiGLU split: gate/up width must be even so it splits into halves ---
  TORCH_CHECK((gate_up_dim % 2) == 0,
              "sm70_fused_moe: SwiGLU requires an even gate/up dim so it can "
              "be split into equal gate and up halves, got ", gate_up_dim,
              ".");

  // ---- MXFP4 weight / scale shape consistency (see contract block) ---------
  // linear1 gate/up: w13_weight [E, 2*I, K/2], w13_weight_scale [E, 2*I, K/gs].
  TORCH_CHECK(w13_weight.dim() == 3 && w13_weight_scale.dim() == 3,
              "sm70_fused_moe: w13_weight / w13_weight_scale must be 3-D "
              "[E, 2*inter_I, ...].");
  TORCH_CHECK(w13_weight.size(0) == num_experts &&
                  w13_weight.size(1) == gate_up_dim &&
                  w13_weight.size(2) == hidden_K / 2,
              "sm70_fused_moe: w13_weight must be [E, 2*inter_I, hidden_K/2] "
              "(packed MXFP4, 2 nibbles/byte).");
  TORCH_CHECK(w13_weight_scale.size(0) == num_experts &&
                  w13_weight_scale.size(1) == gate_up_dim &&
                  w13_weight_scale.size(2) == hidden_K / group_size,
              "sm70_fused_moe: w13_weight_scale must be "
              "[E, 2*inter_I, hidden_K/group_size] (per-32 E8M0 block scale).");
  // linear2 down: w2_weight [E, K, I/2], w2_weight_scale [E, K, I/gs].
  TORCH_CHECK(w2_weight.dim() == 3 && w2_weight_scale.dim() == 3,
              "sm70_fused_moe: w2_weight / w2_weight_scale must be 3-D "
              "[E, hidden_K, ...].");
  TORCH_CHECK(w2_weight.size(0) == num_experts &&
                  w2_weight.size(1) == hidden_K &&
                  w2_weight.size(2) == inter_I / 2,
              "sm70_fused_moe: w2_weight must be [E, hidden_K, inter_I/2] "
              "(packed MXFP4, 2 nibbles/byte).");
  TORCH_CHECK(w2_weight_scale.size(0) == num_experts &&
                  w2_weight_scale.size(1) == hidden_K &&
                  w2_weight_scale.size(2) == inter_I / group_size,
              "sm70_fused_moe: w2_weight_scale must be "
              "[E, hidden_K, inter_I/group_size] (per-32 E8M0 block scale).");

  // ---- Output / input shape consistency ------------------------------------
  TORCH_CHECK(permuted_input.dim() == 2 && out.dim() == 2,
              "sm70_fused_moe: permuted_input and out must be 2-D.");
  TORCH_CHECK(permuted_input.size(1) == hidden_K,
              "sm70_fused_moe: permuted_input cols must equal hidden_K.");
  TORCH_CHECK(out.size(1) == hidden_K,
              "sm70_fused_moe: out cols must equal hidden_K.");
  TORCH_CHECK(out.size(0) == permuted_input.size(0),
              "sm70_fused_moe: out rows must match permuted_input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_fused_moe: out must be row-major contiguous.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(permuted_input));

  const int64_t total_tokens = permuted_input.size(0);
  if (total_tokens == 0) return;

  // =========================================================================
  // Task 4.3 launch: fused linear1 -> SwiGLU -> linear2, fully on chip.
  //
  // The device segments live in namespace `vllm::sm70_fused_moe`:
  //   * `sm70_linear1_swiglu_epilogue` (task 4.2) drives the double-buffered
  //     SMEM pipeline (synchronous loads + __syncthreads, NO cp.async/TMA),
  //     accumulates the gate/up GEMM with first-gen `mma.sync.m8n8k4`,
  //     dequantises MXFP4 (E2M1 + per-32 E8M0) -> fp16 on the way into SMEM,
  //     and computes `silu(gate)*up` on chip, leaving the `h` block resident
  //     in shared memory.
  //   * `sm70_linear2_downproj` (task 4.3) contracts `h` — read STRAIGHT FROM
  //     shared memory (on-chip hand-off, no HBM round-trip) — against the
  //     MXFP4-dequantised down weight and writes out[16, K_out].
  // `sm70_fused_moe_kernel` ties them per (expert, m-tile): it builds the full
  // h[16][I] in SMEM (gate/up never leave SMEM), then runs linear2 from SMEM.
  // Neither gate/up ([M_b, 2*I]) nor h ([M_b, I]) is ever written to HBM
  // (R2.1 / R2.6). End-to-end numerical equivalence is validated on SM70
  // hardware by the property tests (task 5.5); there is no nvcc / V100 build
  // in this env.
  // =========================================================================
  namespace fused = vllm::sm70_fused_moe;

  // ---- Tiling constants ----------------------------------------------------
  // The kernel processes ONE m-tile of kMmaM (== 16) tokens per block (one
  // Volta HMMA warp tile in M). `i_block` requests the per-block slab width
  // shared by linear1's intermediate slab (I_b) and linear2's output slab
  // (K_b); each warp owns a 16-wide n-tile, so the block runs `i_block/16`
  // warps. Ragged final slabs (and ragged 16-wide warp tiles when I or K_out
  // is a multiple of 8 but not 16) are handled by the staging `n_valid` /
  // epilogue `n_cols` clamps, so the only hard shape constraints are the HMMA
  // K%8 / N%8 and MXFP4 %32 checks already validated above.
  const int M_TILE = fused::kMmaM;          // 16 tokens per block
  const int kKStepC = fused::kKStep;        // 32 (== group_size)
  const int slab = static_cast<int>(i_block);
  TORCH_CHECK((slab % fused::kMmaN) == 0 && slab > 0,
              "sm70_fused_moe: i_block (", i_block,
              ") must be a positive multiple of ", fused::kMmaN, ".");
  // NOTE: the (expert, tile) grid confines each block to one expert's segment
  // (blockIdx.y == expert), so a tile never straddles two experts regardless of
  // segment alignment. `m_block` is therefore NOT required to align the
  // segments here (it is retained in the op signature for API compatibility and
  // as a tiling hint); the kernel consumes the dense, unaligned `moe_permute`
  // output of the capture-safe production buffer flow directly.
  (void)m_block;
  const int nwarps = slab / fused::kMmaN;
  const int kThreads = nwarps * fused::kWarpSize;
  TORCH_CHECK(kThreads > 0 && kThreads <= 1024,
              "sm70_fused_moe: derived block size ", kThreads,
              " (i_block/16*32) must be in (0, 1024]; pick a smaller i_block.");

  // ---- V100 96KB dynamic-SMEM budget (design §4 "SMEM 预算", R6.2) ---------
  // Single dynamic SMEM blob (all `half`):
  //   [ smem_h : M_TILE * I ]  [ scratch : max(linear1, linear2) ]
  // The design frames the budget as
  //   M_b*K_step (staged x) + MXFP4 weight tile + block scale + M_b*I_b (h)
  //   <= 96KB.
  // Two refinements this kernel makes, both accounted for exactly below:
  //   (1) MXFP4 weights are dequantised to fp16 *into* SMEM, so the "weight
  //       tile" term is the fp16 slab (slab*K_step halves), double-buffered for
  //       gate AND up in linear1; the E8M0 block scale is consumed inline
  //       during staging and never resides in this blob (its term is 0 here).
  //   (2) linear2 contracts the FULL intermediate I, so we keep the full
  //       h[M_TILE][I] resident (residency choice (A), see the kernel comment)
  //       instead of a tiled M_b*I_b block — i.e. the design's `M_b*I_b(h)`
  //       term becomes `M_TILE*I`.
  // linear1 scratch (halves) = smem_x(2*M_TILE*K_step)
  //                          + smem_wg(2*slab*K_step) + smem_wu(2*slab*K_step)
  // linear2 scratch (halves) = smem_wd(2*slab*K_step)
  // linear1 always dominates, so scratch = 2*M_TILE*K_step + 4*slab*K_step.
  //   total_halves = M_TILE*I + 2*M_TILE*K_step + 4*slab*K_step
  const int64_t I_rt = inter_I;
  const int64_t h_halves = static_cast<int64_t>(M_TILE) * I_rt;
  const int64_t scratch_halves =
      2LL * M_TILE * kKStepC + 4LL * slab * kKStepC;
  const int64_t total_halves = h_halves + scratch_halves;
  const int64_t smem_bytes_ll =
      total_halves * static_cast<int64_t>(sizeof(half));
  constexpr int kV100SmemBudget = 96 * 1024;  // 98304 B (V100 per-block max)
  TORCH_CHECK(smem_bytes_ll <= kV100SmemBudget,
              "sm70_fused_moe: dynamic SMEM ", smem_bytes_ll,
              " bytes exceeds the V100 96KB per-block budget for inter_I=",
              inter_I, ", i_block=", i_block, " (M_TILE=", M_TILE,
              "). The Python gate should reject this shape and fall back "
              "(R6.2).");
  const int smem_bytes = static_cast<int>(smem_bytes_ll);

  // ---- Launch --------------------------------------------------------------
  // 2D grid (max_tiles_per_expert, num_experts): block (tile, expert) handles
  // expert e's rows [expert_offsets[e] + tile*M_TILE, ...). The grid dims depend
  // ONLY on the fixed buffer capacity (total_tokens == permuted_input.size(0),
  // a persistent buffer in the capture-safe production flow) and num_experts —
  // never on the routing distribution — so the launch is constant across CUDA-
  // graph replays. Blocks whose tile lies past their expert's segment (or past
  // the dense total) early-return.
  const int max_tiles = static_cast<int>(
      (total_tokens + M_TILE - 1) / M_TILE);
  const dim3 grid(static_cast<unsigned int>(max_tiles),
                  static_cast<unsigned int>(num_experts));

  auto* kernel = fused::sm70_fused_moe_kernel;

  // Opt into >48KB dynamic shared memory (V100 supports up to 96KB/block).
  if (smem_bytes > (48 << 10)) {
    C10_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
        (void*)kernel, smem_bytes));
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<grid, kThreads, smem_bytes, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      static_cast<int>(out.stride(0)),
      reinterpret_cast<const half*>(permuted_input.data_ptr()),
      static_cast<int>(permuted_input.stride(0)),
      expert_offsets.data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(w13_weight.data_ptr()),
      reinterpret_cast<const uint8_t*>(w13_weight_scale.data_ptr()),
      reinterpret_cast<const uint8_t*>(w2_weight.data_ptr()),
      reinterpret_cast<const uint8_t*>(w2_weight_scale.data_ptr()),
      static_cast<int>(num_experts), static_cast<int>(hidden_K),
      static_cast<int>(inter_I), static_cast<int>(group_size),
      static_cast<int>(total_tokens), static_cast<float>(swiglu_limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
