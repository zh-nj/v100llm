#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-tilelang}"
PORT="${PREFILL_V2_PORT:-8199}"
LENGTHS="${PREFILL_V2_LENGTHS:-1024,10000,32000,64000}"
MAX_TOKENS="${PREFILL_V2_MAX_TOKENS:-32}"
MAX_MODEL_LEN="${PREFILL_V2_MAX_MODEL_LEN:-524288}"
MAX_BATCHED_TOKENS="${PREFILL_V2_MAX_BATCHED_TOKENS:-4096}"
REQUEST_TIMEOUT="${PREFILL_V2_REQUEST_TIMEOUT:-14400}"
MODEL="${PREFILL_V2_MODEL:-/mnt/data6/models/DeepSeek-V4-Flash}"
OUT_DIR="${PREFILL_V2_OUT_DIR:-/tmp/deepseek_v4_sparse_prefill_v2_${MODE}}"
WORKTREE="/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split"
LENGTHS_HARNESS="${WORKTREE}/tests/benchmarks/deepseek_v4_sparse_prefill_v2_lengths.py"
REASONING_HARNESS="${WORKTREE}/tests/benchmarks/deepseek_v4_sparse_prefill_v2_reasoning.py"
SERVER_LOG="${OUT_DIR}/server.log"
RESULT_JSONL="${OUT_DIR}/prefill_decode.jsonl"
REASONING_JSON="${OUT_DIR}/reasoning.json"
PHASE_JSONL="${OUT_DIR}/phase_profile.jsonl"
ENABLE_PHASE_PROFILE="${PREFILL_V2_ENABLE_PHASE_PROFILE:-0}"
GPU_SAMPLE_INTERVAL="${PREFILL_V2_GPU_SAMPLE_INTERVAL:-1}"
GPU_UTIL_CSV="${OUT_DIR}/gpu_util.csv"
GPU_PROCESS_CSV="${OUT_DIR}/gpu_processes.csv"
GPU_SUMMARY_JSON="${OUT_DIR}/gpu_util_summary.json"
GPU_SUMMARY_HARNESS="${WORKTREE}/tests/benchmarks/deepseek_v4_sparse_prefill_v2_gpu_util_summary.py"

mkdir -p "$OUT_DIR"

source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate gptq
set -u

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,5,4,6,7,8,9,10}"
GPU_QUERY_IDS="$CUDA_VISIBLE_DEVICES"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/z/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/z/.cache/triton}"
export PYTHONPATH="${WORKTREE}:${PYTHONPATH:-}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Use checkpoint/config top-k for long-prefill validation instead of forcing
# the older short-context override.
unset VLLM_DEEPSEEK_V4_INDEXER_TOPK

export VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK="${VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK:-1}"
export VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK_DEBUG_COMPARE="${VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK_DEBUG_COMPARE:-0}"
export VLLM_SM70_USE_SPARSE_PREFILL_V2=0
export VLLM_SM70_USE_SPARSE_PREFILL_V2_MAPPED_FUSED=0
export VLLM_SM70_USE_SPARSE_PREFILL_V2_CUDA_MAINLOOP=0
export VLLM_SM70_USE_SPARSE_PREFILL_V2_MMA_MAINLOOP=0
export VLLM_SM70_SPARSE_PREFILL_V2_DEBUG_COMPARE=0
if [ "$ENABLE_PHASE_PROFILE" = "1" ]; then
    export VLLM_DEEPSEEK_V4_PROFILE="${VLLM_DEEPSEEK_V4_PROFILE:-1}"
    export VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH="$PHASE_JSONL"
    export VLLM_DEEPSEEK_V4_PROFILE_PHASE_FILTER="${VLLM_DEEPSEEK_V4_PROFILE_PHASE_FILTER:-both}"
    export VLLM_DEEPSEEK_V4_PROFILE_LOG_EVERY="${VLLM_DEEPSEEK_V4_PROFILE_LOG_EVERY:-0}"
else
    export VLLM_DEEPSEEK_V4_PROFILE=0
    unset VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH
    export VLLM_DEEPSEEK_V4_PROFILE_NVTX=0
fi

case "$MODE" in
    flashmla)
        export VLLM_SM70_USE_TILELANG_SPARSE_PREFILL=0
        export VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO=0
        export VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK=0
        ;;
    default)
        export VLLM_SM70_USE_TILELANG_SPARSE_PREFILL="${VLLM_SM70_USE_TILELANG_SPARSE_PREFILL:-1}"
        export VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO="${VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO:-1}"
        ;;
    tilelang)
        export VLLM_SM70_USE_TILELANG_SPARSE_PREFILL="${VLLM_SM70_USE_TILELANG_SPARSE_PREFILL:-1}"
        export VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO="${VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO:-1}"
        ;;
    selected|mma)
        echo "mode '$MODE' is retired: sparse prefill v2 is closed; use tilelang or flashmla" >&2
        exit 2
        ;;
    *)
        echo "unknown mode: $MODE (expected flashmla, default, tilelang)" >&2
        exit 2
        ;;
esac

SERVER_PID=""
SERVER_PGID=""
GPU_SAMPLER_PID=""
cleanup_server() {
    if [ -n "$GPU_SAMPLER_PID" ]; then
        kill "$GPU_SAMPLER_PID" 2>/dev/null || true
        wait "$GPU_SAMPLER_PID" 2>/dev/null || true
        GPU_SAMPLER_PID=""
    fi
    if [ -z "$SERVER_PGID" ]; then
        return
    fi
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    pkill -TERM -f "[v]llm serve ${MODEL} .*--port ${PORT}" 2>/dev/null || true
    pkill -TERM -f "[V]LLM::EngineCore|[V]LLM::Worker_TP" 2>/dev/null || true
    for _ in $(seq 1 30); do
        if ! kill -0 -- "-$SERVER_PGID" 2>/dev/null \
            && ! pgrep -f "[V]LLM::EngineCore|[V]LLM::Worker_TP" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    pkill -KILL -f "[v]llm serve ${MODEL} .*--port ${PORT}" 2>/dev/null || true
    pkill -KILL -f "[V]LLM::EngineCore|[V]LLM::Worker_TP" 2>/dev/null || true
    if [ -n "$SERVER_PID" ]; then
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    SERVER_PGID=""
}
trap cleanup_server EXIT

cd /tmp
echo "mode=$MODE"
echo "out_dir=$OUT_DIR"
echo "lengths=$LENGTHS"
echo "max_tokens=$MAX_TOKENS"
echo "request_timeout=$REQUEST_TIMEOUT"
echo "max_model_len=$MAX_MODEL_LEN"
echo "max_num_batched_tokens=$MAX_BATCHED_TOKENS"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "gpu_query_ids=$GPU_QUERY_IDS"
echo "streaming_topk=$VLLM_SPARSE_INDEXER_PREFILL_STREAMING_TOPK"
echo "tilelang_sparse_prefill=${VLLM_SM70_USE_TILELANG_SPARSE_PREFILL:-<env-default>}"
echo "tilelang_sparse_prefill_fast_io=${VLLM_SM70_TILELANG_SPARSE_PREFILL_FAST_IO:-<env-default>}"
echo "sparse_prefill_v2=retired"
echo "sparse_prefill_v2_mma=retired"
echo "phase_profile=$ENABLE_PHASE_PROFILE"

setsid vllm serve "$MODEL" \
    --tensor-parallel-size 8 \
    --dtype float16 \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8_ds_mla \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1, 2, 4, 8]}' \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID

echo "server_pid=$SERVER_PID"
HTTP=000
for i in $(seq 1 240); do
    sleep 5
    HTTP=$(curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" \
        -o /dev/null -w "%{http_code}" || echo "000")
    if [ "$HTTP" = "200" ]; then
        echo "server_ready_s=$((i * 5))"
        break
    fi
done

if [ "$HTTP" != "200" ]; then
    echo "server never became ready" >&2
    tail -160 "$SERVER_LOG" >&2 || true
    exit 1
fi

nvidia-smi > "${OUT_DIR}/nvidia_smi_ready.txt" || true
echo "sample_ms,timestamp,index,uuid,util_gpu_pct,util_mem_pct,mem_used_mib,power_w" > "$GPU_UTIL_CSV"
echo "sample_ms,timestamp,gpu_uuid,pid,process_name,used_memory_mib" > "$GPU_PROCESS_CSV"
(
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        sample_ms="$(date +%s%3N)"
        nvidia-smi -i "$GPU_QUERY_IDS" \
            --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw \
            --format=csv,noheader,nounits \
            | sed "s/^/${sample_ms},/" >> "$GPU_UTIL_CSV" || true
        nvidia-smi -i "$GPU_QUERY_IDS" \
            --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader,nounits \
            | sed "s/^/${sample_ms},/" >> "$GPU_PROCESS_CSV" || true
        sleep "$GPU_SAMPLE_INTERVAL"
    done
) &
GPU_SAMPLER_PID=$!
echo "gpu_util_csv=$GPU_UTIL_CSV"
echo "gpu_process_csv=$GPU_PROCESS_CSV"

CUDA_VISIBLE_DEVICES="" /home/z/anaconda3/envs/gptq/bin/python \
    "$LENGTHS_HARNESS" \
    "http://127.0.0.1:${PORT}" "$MODEL" "$LENGTHS" \
    --stream --max-tokens "$MAX_TOKENS" --timeout "$REQUEST_TIMEOUT" \
    | tee "$RESULT_JSONL"

CUDA_VISIBLE_DEVICES="" /home/z/anaconda3/envs/gptq/bin/python \
    "$REASONING_HARNESS" \
    "http://127.0.0.1:${PORT}/v1" "$MODEL" \
    | tee "$REASONING_JSON"

nvidia-smi > "${OUT_DIR}/nvidia_smi_done.txt" || true
CUDA_VISIBLE_DEVICES="" /home/z/anaconda3/envs/gptq/bin/python \
    "$GPU_SUMMARY_HARNESS" "$GPU_UTIL_CSV" \
    | tee "$GPU_SUMMARY_JSON" || true
echo "result_jsonl=$RESULT_JSONL"
echo "reasoning_json=$REASONING_JSON"
echo "gpu_util_csv=$GPU_UTIL_CSV"
echo "gpu_process_csv=$GPU_PROCESS_CSV"
echo "gpu_summary_json=$GPU_SUMMARY_JSON"
if [ "$ENABLE_PHASE_PROFILE" = "1" ]; then
    echo "phase_jsonl=$PHASE_JSONL"
fi
echo "server_log=$SERVER_LOG"
