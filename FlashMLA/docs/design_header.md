# Design Document

## Overview

This design document lays out the technical plan for the
`deepseek-v4-flash-prefill-throughput` feature spec. The program-level goal is
to reach **>= 1500 tok/s** measured prefill throughput on DeepSeek V4 Flash
under 8x V100 TP=8 from a currently-measured **~91 tok/s** post-K_TILE=16
baseline (a ~16.5x gap). The program is organized as a sequence of
independently-landable rounds; each round has a declared targeted phase, a
measured uplift threshold, and a preserved correctness invariant set as
specified in `requirements.md`.

Three concrete rounds are designed here:

- **Round 0** -- commit the uncommitted K_TILE=16 rebuild + the
  `_HAS_DEEP_GEMM_CACHED` cache fix, capture a clean post-K_TILE=16 profile
  trace, and identify the new top bottleneck.
- **Round 1** -- enable `cudagraph_mode=FULL_AND_PIECEWISE` with
  `cudagraph_capture_sizes=[1]` by shielding the specific SM70-unsafe FP8 op
  behind a custom op so inductor cannot fuse it into an `fp8e4nv` Triton
  kernel. No global inductor-fusion disablers.
- **Round 2** -- rewrite the FlashMLA SM70 sparse-prefill kernel to batch
  `BLOCK_M = 32` queries per CUDA block (FA-Bond-vLLM design), replacing the
  current 1-query-per-block grid launched as `dim3(s_q, h_q)` at
  `fwd.cuh:1136`.

Rounds 3+ are intentionally deferred: the spec's Requirement 8 allows new
rounds to be appended to this document without amending requirements, but
only once Rounds 0-2 have landed and a fresh trace reveals the next top
bottleneck. A sketch of likely Round-3+ directions is included at the end for
planning visibility, not as committed design.

The design is grounded in measured numbers from the existing
post-K_TILE=16 profile trace
(`.kiro/specs/deepseek-v4-flash-bottleneck-fixes-measured/measurements/prefill/profile_ktile16.jsonl`,
2048-token prompt x 688 decoder calls, TP=8):

| phase label                            | total_ms | n    | us/call |
|----------------------------------------|---------:|-----:|--------:|
| `wrapper.attention_total`              | 284,277  | 1376 | 206,596 |
| `impl.mla_attn_total`                  | 267,714  |  688 | 389,119 |
| `prefill.flashmla_sparse_fwd`          | 267,248  |  688 | 388,443 |
| `wrapper.wo_b`                         |  25,205  | 1376 |  18,317 |
| `impl.indexer_kv_compress_overlap`     |  14,914  |  672 |  22,193 |
| `decoder.moe.experts`                  |   4,459  | 1376 |   3,241 |
| `moe.runner.apply_quant_method`        |   4,242  | 1376 |   3,083 |
| `moe.runner.quant_apply`               |   3,649  | 1376 |   2,652 |
| `moe.experts.gemm_w13`                 |   1,654  | 1376 |   1,202 |
| `decoder.mhc_pre_attn`                 |   1,472  | 1376 |   1,070 |
| `moe.experts.gemm_w2`                  |   1,076  | 1376 |     782 |

`prefill.flashmla_sparse_fwd` is **94.1% of `wrapper.attention_total`** and
`wrapper.attention_total` is the overwhelming top label. On the 2048-token
prompt x 688 layer calls, prefill is dominated by the FlashMLA SM70 sparse
kernel to a degree that no other optimization can close the 22x gap to
1500 tok/s. Round 2 is therefore the load-bearing round; Rounds 0 and 1 are
prerequisites that preserve and extend the existing launch harness.
