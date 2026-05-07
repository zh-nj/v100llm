# DeepSeek V4 Flash Profiling Harness

This directory ships three companion tools for profiling DeepSeek V4 Flash
inference on 8x V100 (SM70):

- `deepseek_v4_byte_budget.py` — derives per-operation byte/FLOP budgets
  from the model config (no weight load). Input to the analyzer's roofline
  classifier.
- `deepseek_v4_profile_analyze.py` — stdlib-only offline analyzer that
  ingests the JSONL traces emitted by `_DeepseekV4PhaseProfiler` and
  produces `summary_{decode,prefill}.md`, `roofline.csv`,
  `theoretical_peak.md`, `diff_vs_baseline.md`, and the
  `optimization_plan.md` scaffold.
- Nsight Systems canonical capture + `nsys stats` extraction recipe
  (below). The analyzer accepts `--nsys-stats-csv` to cross-link
  kernel-level attribution into `summary_decode.md`.

---

## 1. Env Var Matrix

All profiling gates default **off**; runtime cost when disabled is zero.

| Env var | Values | Meaning |
|---|---|---|
| `VLLM_DEEPSEEK_V4_PROFILE` | `0` / `1` | Master gate for CUDA-event recording. |
| `VLLM_DEEPSEEK_V4_PROFILE_NVTX` | `0` / `1` | Emit NVTX ranges (`step` > `layer` > `operation`) for `nsys`. Cheap; keep on during nsys runs even when `PROFILE=0`. |
| `VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH` | path | JSONL sink path. Parent dir auto-created. A `<path>.meta.json` sidecar is written once at profiler init. |
| `VLLM_DEEPSEEK_V4_PROFILE_MODE` | `queue` (default) / `eager` | `queue`: enqueue start/end events and sync once at step boundary (<15% overhead budget). `eager`: per-event sync (debug only). |
| `VLLM_DEEPSEEK_V4_PROFILE_STEP_LIMIT` | int | Stop recording after N steps (protects long runs from unbounded JSONL growth). |
| `VLLM_DEEPSEEK_V4_PROFILE_PHASE_FILTER` | `decode` / `prefill` / `both` (default) | Restrict recording to one phase kind. |
| `VLLM_DEEPSEEK_V4_PROFILE_LOG_EVERY` | int | Periodic in-memory aggregator log frequency. |

Example (1K decode capture, queue mode, first 64 steps only):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export VLLM_DEEPSEEK_V4_PROFILE=1
export VLLM_DEEPSEEK_V4_PROFILE_NVTX=1
export VLLM_DEEPSEEK_V4_PROFILE_MODE=queue
export VLLM_DEEPSEEK_V4_PROFILE_PHASE_FILTER=decode
export VLLM_DEEPSEEK_V4_PROFILE_STEP_LIMIT=64
export VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH=/mnt/data6/profiles/decode_1k.jsonl
```

---

## 2. Raw JSONL Schema

One JSON object per line, fields per design §Data Models:

```json
{
  "pid": 18714,
  "cuda_device": 2,
  "tp_rank": 0,
  "step_id": 1423,
  "step_kind": "decode",
  "step_token_count": 1,
  "layer_idx": 17,
  "compress_ratio": 4,
  "phase": "decode.attn.direct_flashmla",
  "elapsed_us": 38.435,
  "chunk_idx": null,
  "num_chunk_tokens": null,
  "combined_lens_max": null,
  "topk_length_max": 512,
  "timestamp_ns": 1715010501234567890
}
```

Rules:

- `phase` and `elapsed_us` are the only strictly required fields.
- Structural fields (`step_id`, `layer_idx`, `tp_rank`, …) may be `null`
  for backward compatibility; the analyzer tolerates nulls.
- `elapsed_us` is always a positive float.
- Taking `max_rank` (slowest TP rank) is the convention for wall time —
  the analyzer does this via `pick_max_rank`.

Run metadata sidecar: `<raw_path>.meta.json` captures git commit, FlashMLA
HEAD, torch/CUDA/driver versions, conda env, GPU list, VLLM_* env vars,
model config, and compilation config.

---

## 3. Canonical `nsys` Command

For an 8x V100 decode capture (one nvtx `step` range per forward):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
VLLM_DEEPSEEK_V4_PROFILE_NVTX=1 \
VLLM_DEEPSEEK_V4_PROFILE=0 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --cuda-memory-usage=true \
  --capture-range=nvtx \
  --capture-range-end=stop \
  --nvtx-capture='step' \
  --output=/mnt/data6/profiles/deepseek_v4_decode_$(date +%s) \
  --force-overwrite=true \
  -- python benchmarks/deepseek_v4_flashmla_sm70.py --mode openai-stream \
     --endpoint http://127.0.0.1:18080/v1 --prompt-file prompts/1k.txt \
     --max-tokens 8
```

`VLLM_DEEPSEEK_V4_PROFILE=0` disables the CUDA-event sink during nsys to
avoid double-instrumentation cost; NVTX ranges are free.

---

## 4. `nsys stats` Extraction Recipe

Once a `.nsys-rep` is captured:

```bash
REP=/mnt/data6/profiles/deepseek_v4_decode_<ts>.nsys-rep

# Top kernels by cumulative GPU time
nsys stats --format csv -r cuda_gpu_kern_sum \
  -o /mnt/data6/profiles/kern_sum.csv "$REP"

# NVTX push/pop aggregate (step / layer / operation)
nsys stats --format csv -r nvtx_pushpop_sum \
  -o /mnt/data6/profiles/nvtx_sum.csv "$REP"

# Device memory throughput
nsys stats --format csv -r cuda_gpu_mem_time_sum \
  -o /mnt/data6/profiles/mem_sum.csv "$REP"
```

Feed the kernel-sum CSV into the analyzer:

```bash
python benchmarks/deepseek_v4_profile_analyze.py \
  --raw-trace /mnt/data6/profiles/decode_1k.jsonl \
  --byte-budget byte_budget.json \
  --nsys-stats-csv /mnt/data6/profiles/kern_sum.csv \
  --output-dir /mnt/data6/profiles/report_decode_1k \
  --context-lens 1024,3072 --phase decode
```

The top-5 kernels (cumulative GPU time) and their throughput / occupancy
columns are merged into `summary_decode.md` under a
**Nsight Systems Top Kernels** section.

---

## 5. `ncu --set basic` Spot-Check

For per-kernel occupancy / achieved DRAM throughput on a specific
kernel (e.g. `flash_mla_sparse_fwd_kernel`), skip the warmup window and
capture 5 launches:

```bash
ncu --set basic --target-processes all \
  --launch-skip 100 --launch-count 5 \
  --kernel-id ::flash_mla_sparse_fwd_kernel: \
  --export /mnt/data6/profiles/deepseek_v4_sparse_fwd.ncu-rep \
  -- python benchmarks/deepseek_v4_flashmla_sm70.py --mode openai-stream \
     --endpoint http://127.0.0.1:18080/v1 --prompt-file prompts/1k.txt \
     --max-tokens 8
```

Open `.ncu-rep` in Nsight Compute UI or export with
`ncu --import --csv` for scriptable analysis.

---

## 6. End-to-End Example

```bash
# 1) Derive the byte/FLOP budget
python benchmarks/deepseek_v4_byte_budget.py \
  --model /mnt/data6/models/DeepSeek-V4-Flash \
  --tp 8 --ctx 1024 --output byte_budget.json

# 2) Run the server with profiling on, drive a 1K decode workload
export VLLM_DEEPSEEK_V4_PROFILE=1
export VLLM_DEEPSEEK_V4_PROFILE_NVTX=1
export VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH=/mnt/data6/profiles/decode_1k.jsonl
# … launch vLLM and drive client …

# 3) Analyze
python benchmarks/deepseek_v4_profile_analyze.py \
  --raw-trace /mnt/data6/profiles/decode_1k.jsonl \
  --byte-budget byte_budget.json \
  --output-dir /mnt/data6/profiles/report_1k \
  --context-lens 1024,3072 --phase decode
```

The report directory will contain:

```
report_1k/
├── summary_decode.md      # top-5 table (+ nsys kernels if provided)
├── summary_prefill.md
├── roofline.csv
├── theoretical_peak.md
├── diff_vs_baseline.md    # only if --baseline
└── optimization_plan.md   # scaffold with per-bottleneck sections
```
