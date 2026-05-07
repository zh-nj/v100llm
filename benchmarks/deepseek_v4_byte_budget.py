#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash per-operation byte/FLOP budget calculator.

Loads the model config (no weights) and emits an illustrative byte/FLOP
budget for every operation category instrumented by the in-tree profiler,
accounting for SM70 (V100) software-dequant overheads:

  * FP8 weights: stored as 1 byte/element but must be dequantized to FP16
    before SM70 tensor cores can consume them. Effective HBM read traffic
    therefore counts the FP8 byte (1 B) plus the FP16 staging buffer write
    (2 B) → ~3x the FP8-on-Hopper figure for the matmul side. We approximate
    this as a 3x read multiplier for FP8-quantized weight tensors.
  * MXFP4 / FP4 expert weights: 0.5 B/element storage with per-32 group
    scales; SM70 unpack to FP16 inflates effective read to ~2.5 B/element.
  * O-projection FP8 einsum: cached FP32 weights on SM70 → 4 B/element.

Output: JSON to stdout (or --output PATH) keyed by phase label, with
fields per the design doc:
  bytes_per_token_per_layer_hbm_read
  bytes_per_token_per_layer_hbm_write
  flops_per_token_per_layer
  arithmetic_intensity_flops_per_byte

Example
-------
    python deepseek_v4_byte_budget.py \
        --model /mnt/data6/models/DeepSeek-V4-Flash \
        --tp 8 --topk 512 --ctx 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# SM70 dequant overhead constants
# ---------------------------------------------------------------------------
FP8_READ_MULT = 3.0   # 1 B FP8 read + 2 B FP16 staging = 3 B effective
FP4_READ_MULT = 5.0   # 0.5 B FP4 + 2 B FP16 staging ≈ 2.5 B/elt → ratio vs 0.5
FP32_BYTES = 4
FP16_BYTES = 2
BF16_BYTES = 2
FP8_BYTES = 1


def _bytes_for_dtype(dtype: str) -> float:
    d = dtype.lower()
    if d in ("fp8", "e4m3", "e5m2"):
        return FP8_BYTES
    if d in ("fp4", "mxfp4", "nvfp4"):
        return 0.5
    if d in ("fp16", "f16", "half"):
        return FP16_BYTES
    if d in ("bf16",):
        return BF16_BYTES
    if d in ("fp32", "f32"):
        return FP32_BYTES
    return FP16_BYTES


def load_config(model_path: str) -> dict[str, Any]:
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-operation budget computation. All figures are *per token, per layer*,
# *per GPU* (i.e. divided by tp where applicable). They are illustrative —
# downstream consumers refine with the offline analyzer's measurements.
# ---------------------------------------------------------------------------
def compute_budgets(cfg: dict[str, Any], *,
                    tp: int = 8,
                    topk: int | None = None,
                    ctx: int = 1024) -> dict[str, dict[str, float]]:
    H = int(cfg["hidden_size"])                         # 4096
    n_heads = int(cfg["num_attention_heads"])           # 64
    head_dim = int(cfg["head_dim"])                     # 512
    rope_dim = int(cfg.get("qk_rope_head_dim", 64))
    q_lora = int(cfg.get("q_lora_rank", 1024))
    o_lora = int(cfg.get("o_lora_rank", 1024))
    idx_n_heads = int(cfg.get("index_n_heads", 64))
    idx_head_dim = int(cfg.get("index_head_dim", 128))
    topk = topk or int(cfg.get("index_topk", 512))
    moe_dim = int(cfg.get("moe_intermediate_size", 2048))
    n_experts = int(cfg.get("n_routed_experts", 256))
    n_shared = int(cfg.get("n_shared_experts", 1))
    experts_per_tok = int(cfg.get("num_experts_per_tok", 6))
    sliding = int(cfg.get("sliding_window", 128))
    swiglu = 2  # gate + up

    qcfg = cfg.get("quantization_config") or {}
    weight_dtype = qcfg.get("quant_method", "fp8")
    expert_dtype = cfg.get("expert_dtype", "fp4")

    # SM70 effective bytes/elt for weight tensors
    w_bytes = (FP8_BYTES * FP8_READ_MULT
               if weight_dtype.startswith("fp8")
               else _bytes_for_dtype(weight_dtype) * 2)
    e_bytes = (0.5 * FP4_READ_MULT
               if expert_dtype in ("fp4", "mxfp4", "nvfp4")
               else _bytes_for_dtype(expert_dtype) * 2)
    act_bytes = FP16_BYTES  # SM70 forces fp16 activations

    H_per_gpu = H / tp
    n_heads_per_gpu = max(1, n_heads // tp)

    out: dict[str, dict[str, float]] = {}

    def emit(label: str, *, r: float, w: float, f: float) -> None:
        ai = f / max(r + w, 1.0)
        out[label] = {
            "bytes_per_token_per_layer_hbm_read": float(r),
            "bytes_per_token_per_layer_hbm_write": float(w),
            "flops_per_token_per_layer": float(f),
            "arithmetic_intensity_flops_per_byte": float(ai),
        }

    # --- (a) wrapper.fused_wqa_wkv: hidden -> (Q_lora, KV_lora+rope)
    # weight: H * (q_lora + kv_lora + rope_dim) ≈ H * (q_lora + o_lora + rope_dim)
    fused_in = H
    fused_out = q_lora + o_lora + rope_dim
    w_elts = fused_in * fused_out
    emit("wrapper.fused_wqa_wkv",
         r=w_elts * w_bytes / tp + fused_in * act_bytes,
         w=fused_out * act_bytes,
         f=2 * fused_in * fused_out / tp)

    # --- (b) impl.q_kv_rmsnorm  (RMSNorm on q_lora and kv_lora)
    rn_elts = (q_lora + o_lora)
    emit("impl.q_kv_rmsnorm",
         r=rn_elts * act_bytes,
         w=rn_elts * act_bytes,
         f=4 * rn_elts)

    # --- (b') impl.q_proj  (q_lora -> n_heads * head_dim)
    qp_in, qp_out = q_lora, n_heads * (head_dim + rope_dim)
    emit("impl.q_proj",
         r=qp_in * qp_out * w_bytes / tp + qp_in * act_bytes,
         w=qp_out * act_bytes / tp,
         f=2 * qp_in * qp_out / tp)

    # --- (c) impl.kv_insert  (write KV cache slot, FP8 + scale)
    kv_elt = (head_dim + rope_dim)
    emit("impl.kv_insert",
         r=kv_elt * act_bytes,
         w=kv_elt * FP8_BYTES + 4,
         f=kv_elt)  # quantize + write

    # --- (d) impl.indexer_kv_compress_overlap
    # indexer projects q to idx_n_heads * idx_head_dim and reads compressed
    # KV (a fraction of ctx via per-layer compress ratio).
    idx_w = q_lora * idx_n_heads * idx_head_dim
    indexer_read = idx_w * w_bytes / tp + ctx * idx_head_dim * act_bytes
    indexer_flops = 2 * q_lora * idx_n_heads * idx_head_dim / tp \
                    + 2 * idx_n_heads * idx_head_dim * ctx / tp
    emit("impl.indexer_kv_compress_overlap",
         r=indexer_read,
         w=topk * 4,  # int32 indices
         f=indexer_flops)

    # --- (g) decode.attn.fallback (BF16 reference path, SM70 → fp16 workspace)
    # reads full ctx KV in fp16
    fallback_read = ctx * kv_elt * act_bytes / tp + n_heads_per_gpu * (head_dim + rope_dim) * act_bytes
    fallback_flops = 2 * n_heads_per_gpu * (head_dim + rope_dim) * ctx \
                     + 2 * n_heads_per_gpu * head_dim * ctx
    emit("decode.attn.fallback",
         r=fallback_read,
         w=n_heads_per_gpu * head_dim * act_bytes,
         f=fallback_flops)

    # --- decode.attn.direct_flashmla (FlashMLA sparse, topk gather)
    direct_read = topk * (kv_elt * FP8_BYTES + 4) / tp \
                  + n_heads_per_gpu * (head_dim + rope_dim) * act_bytes
    direct_flops = 2 * n_heads_per_gpu * (head_dim + rope_dim) * topk \
                   + 2 * n_heads_per_gpu * head_dim * topk
    emit("decode.attn.direct_flashmla",
         r=direct_read,
         w=n_heads_per_gpu * head_dim * act_bytes,
         f=direct_flops)

    # --- prefill chunk-loop ops (per-token amortized over chunk)
    emit("prefill.compressed_gather",
         r=topk * kv_elt * FP8_BYTES / tp,
         w=topk * kv_elt * act_bytes / tp,
         f=topk * kv_elt)  # dequant FLOPs

    emit("prefill.swa_gather",
         r=sliding * kv_elt * FP8_BYTES / tp,
         w=sliding * kv_elt * act_bytes / tp,
         f=sliding * kv_elt)

    emit("prefill.combine_indices",
         r=(topk + sliding) * 4,
         w=(topk + sliding) * 4,
         f=topk + sliding)

    emit("prefill.flashmla_sparse_fwd",
         r=(topk + sliding) * kv_elt * act_bytes / tp
            + n_heads_per_gpu * (head_dim + rope_dim) * act_bytes,
         w=n_heads_per_gpu * head_dim * act_bytes,
         f=2 * n_heads_per_gpu * (head_dim + rope_dim) * (topk + sliding)
            + 2 * n_heads_per_gpu * head_dim * (topk + sliding))

    # --- (h) O-projection pipeline: inv-rope + fp8 quant, fp8 einsum, wo_b
    emit("wrapper.o_inv_rope_fp8_quant",
         r=n_heads_per_gpu * head_dim * act_bytes,
         w=n_heads_per_gpu * head_dim * FP8_BYTES + n_heads_per_gpu * 4,
         f=8 * n_heads_per_gpu * head_dim)  # rope rotate + quant scale

    # SM70 cached FP32 einsum weight: (n_heads * head_dim) x o_lora
    o_einsum_w = n_heads * head_dim * o_lora
    emit("wrapper.o_fp8_einsum",
         r=o_einsum_w * FP32_BYTES / tp + n_heads_per_gpu * head_dim * act_bytes,
         w=o_lora * act_bytes / tp,
         f=2 * n_heads_per_gpu * head_dim * o_lora)

    # wo_b: o_lora -> H, FP8 weight
    wo_w = o_lora * H
    emit("wrapper.wo_b",
         r=wo_w * w_bytes / tp + o_lora * act_bytes,
         w=H * act_bytes / tp,
         f=2 * o_lora * H / tp)

    # --- mHC pre/post (LayerNorm + small linear)
    emit("decoder.mhc_pre_attn",
         r=H * act_bytes,
         w=H * act_bytes,
         f=4 * H)
    emit("decoder.mhc_post_attn",
         r=H * act_bytes,
         w=H * act_bytes,
         f=4 * H)

    # --- MoE routing: H -> n_experts (FP16 router)
    router_w = H * n_experts
    emit("decoder.moe.routing",
         r=router_w * FP16_BYTES + H * act_bytes,
         w=experts_per_tok * 8,  # ids+weights
         f=2 * H * n_experts)

    # --- MoE experts: per-token activates (experts_per_tok + n_shared) experts.
    # Each expert SwiGLU: H -> moe_dim (gate+up) -> H. FP4 weights on SM70.
    active = experts_per_tok + n_shared
    expert_w_elts = (swiglu * H * moe_dim + moe_dim * H)  # per expert
    moe_read = active * expert_w_elts * e_bytes / tp + H * act_bytes
    moe_write = H * act_bytes
    moe_flops = active * (2 * swiglu * H * moe_dim + 2 * moe_dim * H) / tp
    emit("decoder.moe.experts",
         r=moe_read, w=moe_write, f=moe_flops)

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to DeepSeek V4 Flash model dir (config.json read).")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--output", default="-",
                   help="Output JSON path (default '-' = stdout).")
    args = p.parse_args()

    cfg = load_config(args.model)
    budgets = compute_budgets(cfg, tp=args.tp, topk=args.topk, ctx=args.ctx)

    payload = {
        "model_path": args.model,
        "tp": args.tp,
        "ctx": args.ctx,
        "topk": args.topk or cfg.get("index_topk"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "notes": (
            "SM70 software-dequant accounted for: FP8 weights × 3x read "
            "multiplier (FP8 + FP16 staging), FP4 experts × 5x "
            "(0.5 B FP4 + 2 B FP16 staging ≈ 2.5 B/elt), O-einsum keeps "
            "cached FP32 weights (4 B/elt). All figures are per-token "
            "per-layer per-GPU (TP-divided where applicable)."
        ),
        "budgets": budgets,
    }

    blob = json.dumps(payload, indent=2, sort_keys=True)
    if args.output == "-":
        sys.stdout.write(blob + "\n")
    else:
        with open(args.output, "w") as f:
            f.write(blob + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
