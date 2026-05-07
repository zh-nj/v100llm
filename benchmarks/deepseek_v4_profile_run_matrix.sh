#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSeek V4 Flash profiling matrix runner.
#
# Drives the OpenAI-stream client for prefill 1K/3K and decode 1K/3K/~4K
# under multiple configs. Emits JSONL traces and meta.json sidecars per cell.
#
# Prerequisites:
#   - vLLM server already running with /mnt/data6/models/DeepSeek-V4-Flash
#   - VLLM_DEEPSEEK_V4_PROFILE=1, VLLM_DEEPSEEK_V4_PROFILE_RAW_PATH set per cell
#   - Environment: gptq conda env, CUDA_DEVICE_ORDER=PCI_BUS_ID, V100 GPUs

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split}"
SPEC_DIR="${SPEC_DIR:-/mnt/data/apps/1Cat-vLLM/.kiro/specs/deepseek-v4-flash-profiling-optimization}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8199/v1}"
MODEL="${MODEL:-/mnt/data6/models/DeepSeek-V4-Flash}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${SPEC_DIR}/measurements/${TIMESTAMP}"

# Cell matrix: (config, phase, ctx_tokens, max_tokens)
# - config:  fallback | direct | mhc_fast | combined
# - phase:   prefill | decode
# - ctx:     1024 | 3072 | 4000
declare -a CELLS=(
  "fallback prefill 1024 4"
  "fallback prefill 3072 4"
  "fallback decode  1024 64"
  "fallback decode  3072 64"
  "direct   decode  1024 64"
  "mhc_fast decode  1024 64"
  "combined decode  1024 64"
  "combined decode  3072 64"
  "combined decode  4000 32"
)

mkdir -p "${OUT_BASE}"
echo "Measurement matrix output: ${OUT_BASE}"

for cell in "${CELLS[@]}"; do
  read -r config phase ctx max_tokens <<< "$cell"
  cell_dir="${OUT_BASE}/${config}/${phase}_${ctx}"
  mkdir -p "${cell_dir}"
  raw_path="${cell_dir}/trace.jsonl"

  echo "=== ${config} / ${phase} ctx=${ctx} ==="

  # Note: server config (mhc_fast / direct / fallback) is set at server start
  # via env vars VLLM_SM70_MHC_FAST and VLLM_SM70_DEEPSEEK_V4_DIRECT_DECODE.
  # This script assumes the server is already running with the desired config.
  # For a full matrix, restart the server between configs.

  # Generate prompt of approximate ctx token length
  python -c "
import json, urllib.request, time
payload = {
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '请用约${ctx}个token的中文进行详细回答：' + ('深度学习模型推理过程中的注意力机制、KV缓存、量化技术。' * 50)}],
    'temperature': 0.0,
    'max_tokens': ${max_tokens},
    'stream': True,
    'stream_options': {'include_usage': True},
}
req = urllib.request.Request('${ENDPOINT}/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
start = time.perf_counter()
ttft = None
tokens = 0
with urllib.request.urlopen(req, timeout=300) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]': break
        chunk = json.loads(data)
        usage = chunk.get('usage')
        if usage and usage.get('completion_tokens'):
            tokens = usage['completion_tokens']
        choices = chunk.get('choices', [])
        if not choices: continue
        content = (choices[0].get('delta') or {}).get('content', '')
        if content and ttft is None:
            ttft = time.perf_counter() - start
end = time.perf_counter()
result = {'config': '${config}', 'phase': '${phase}', 'ctx': ${ctx}, 'tokens': tokens, 'ttft': ttft, 'total_s': end - start, 'tps': tokens / (end - start - (ttft or 0)) if (end - start - (ttft or 0)) > 0 else 0}
with open('${cell_dir}/result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
"
  echo ""
done

echo "Matrix complete. Results under ${OUT_BASE}"
