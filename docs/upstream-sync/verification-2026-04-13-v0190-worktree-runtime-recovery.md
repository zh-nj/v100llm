# 2026-04-13 v0.19.0 Worktree Runtime Recovery Verification

## Environment

- Worktree: `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split`
- Conda env: `gptq`
- Root `PYTHONPATH` disabled: `yes`

## Build And Import

- 2026-04-13 16:47:32 CST, external GPU environment:
  - `torch.cuda.is_available=True`
  - `torch.cuda.device_count=11`
  - `import vllm` resolved to `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/__init__.py`
  - `import vllm._C` succeeded
  - `current_platform=NvmlCudaPlatform cuda`
- 2026-04-15:
  - Command: `VLLM_FLASH_ATTN_SRC_DIR=/mnt/data/apps/fa2/flash-attention-v100 python -m pip install -e . --no-build-isolation`
  - Result: editable rebuild and reinstall succeeded in the `gptq` environment.
  - Worktree-local FA2 artifact:
    - `/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`
    - contains `flash_fwd_*_sm70` symbols after rebuild
  - Post-rebuild import checks:
    - `vllm_path=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/__init__.py`
    - `imported _C`
    - `device_type=cuda`
    - `fa2_varlen_fwd=True` after importing `vllm.vllm_flash_attn.flash_attn_interface`

## Targeted Pytest

- 2026-04-13 16:47 CST:
  - `pytest -q tests/cuda/test_platform_no_cuda_init.py -k "disable_nvml_plugin_uses_non_nvml_fallback or cuda_plugin_falls_back_to_device_files_on_nvml_error or disable_nvml_still_detects_cuda_platform"`
  - Result: `3 passed, 2 deselected`
- 2026-04-13 16:46 CST:
  - `pytest -q tests/model_executor/test_fused_moe_routing_method.py tests/model_executor/test_fused_moe_utils_compat.py tests/model_executor/test_fused_moe_modular_kernel_compat.py tests/engine/test_arg_utils.py tests/v1/engine/test_engine_args.py`
  - Result: `63 passed`

### Runtime contract fixes covered by the passing suite

- `vllm.utils.collection_utils.as_iter` restored for `entrypoints/llm.py` import path.
- Non-NVML CUDA platform fallback restored in `vllm.platforms`.
- `vllm.envs.VLLM_BATCH_INVARIANT` restored for parallel config validation.
- `fused_moe` compatibility restored for:
  - `get_routing_method_type`
  - `apply_moe_activation` / `activation_without_mul` via `fused_moe.utils`
  - `MoEPrepareAndFinalizeNoEP` / `make_moe_prepare_and_finalize_no_ep`
  - `FusedMoEPermuteExpertsUnpermute` / `FusedMoEModularKernel` legacy aliases
- `CacheConfig.verify_with_parallel_config()` compatibility hook restored.
- `CompilationConfig.compile_ranges_split_points` compatibility alias restored.
- 2026-04-14:
  - `pytest -q tests/cuda/test_platform_no_cuda_init.py`
  - Result: `5 passed`
- 2026-04-14:
  - `pytest -q tests/engine/test_arg_utils.py -k "runtime_config_groups or unrecognized_env"`
  - Result: `52 deselected` and exit code `5` because the selector no longer matches current test names.
- 2026-04-14:
  - `pytest -q tests/engine/test_arg_utils.py`
  - Result: `52 passed`
- 2026-04-14:
  - `pytest -q tests/v1/engine/test_engine_args.py -k "runtime_config_groups_round_trip_from_engine_args or defaults_with_usage_context"`
  - Result: `1 passed, 2 deselected`
- 2026-04-14:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 pytest -q tests/v1/executor/test_executor.py -k custom_executor_async`
  - Result: `2 passed, 7 deselected`
  - Note: the same test without fixed `CUDA_DEVICE_ORDER` on this mixed-GPU host first hit `cudaErrorNoKernelImageForDevice`; with the runtime device mask applied it exposed, and then validated, the real split-branch regression fix in `vllm/v1/metrics/loggers.py`.
- 2026-04-14:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 pytest -q tests/v1/attention/test_flash_attn_sm70.py`
  - Result: `3 passed`
- 2026-04-14:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 pytest -q tests/quantization/test_minimax_m2_awq_sm70.py`
  - Result: `6 passed`
- 2026-04-14:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 pytest -q tests/quantization/test_compressed_tensors.py -k test_compressed_tensors_moe_ignore_with_model`
  - Result: `1 passed, 17 deselected`

## Qwen3.5-27B-AWQ AsyncLLM Smoke

- 2026-04-14:
  - Command: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 python /tmp/qwen27_async_smoke.py`
  - `vllm_file=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/__init__.py`
  - `import vllm._C`: `ok`
  - Output text:
    - `<think>`
    - `Thinking Process:`
    - `1.  **Analyze the Request:**`
  - `finish_reason=length`
  - Notes:
    - Engine started from the worktree path with root `PYTHONPATH` disabled.
    - Runtime emitted the existing FLA format-mismatch warning during generation, but the request completed successfully.

## MiniMax-M2.5-AWQ AsyncLLM Smoke

- 2026-04-14:
  - Command parameters:
    - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    - `CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9`
    - `tensor_parallel_size=8`
    - `quantization=awq`
    - `dtype=half`
    - `gpu_memory_utilization=0.85`
    - `max_model_len=4096`
    - `enforce_eager=True`
    - `trust_remote_code=True`
  - `vllm_file=/mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split/vllm/__init__.py`
  - `import vllm._C`: `ok`
  - Output text: `回答是或不是。"`
  - Continuation: `我回答："是。"`
  - Continuation: `"好，你已经回答`
  - `finish_reason=length`
  - Notes:
    - The plan's original 2-GPU MiniMax example is not sufficient for this host/model combination; the passing runtime configuration used 8 V100 GPUs as above.
    - The engine completed full model load, KV-cache initialization, SM70 AWQ warmup, and one generation request without importing the root source tree.

## MiniMax-M2.7-AWQ-4bit AsyncLLM Smoke

- 2026-04-15:
  - Command parameters:
    - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    - `CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9`
    - `model=/mnt/data6/models/MiniMax-M2.7-AWQ-4bit`
    - `dtype=half`
    - `tensor_parallel_size=8`
    - `gpu_memory_utilization=0.85`
    - `max_model_len=4096`
    - `enforce_eager=True`
    - `trust_remote_code=True`
  - Effective quantization path:
    - `quantization=compressed-tensors` was auto-detected from the model `config.json`
    - Dense layers used `CompressedTensorsWNA16` with `ExllamaLinearKernel`
    - MoE layers selected `CompressedTensorsSM70WNA16MoEMethod (TurboMind SM70 kernels)`
    - The worktree completed load-time `SM70 CT→AWQ` conversion for the experts and then ran 8-GPU `SM70 AWQ` warmup
  - Result:
    - `import vllm._C`: `ok`
    - Output text: `",`
    - Continuation: `"score": 0.0,`
    - Continuation: `"metadata": {`
    - `finish_reason=length`
  - Notes:
    - Reusing the `M2.5` smoke with `quantization=awq` fails at `ModelConfig` validation because this model advertises `quant_method=compressed-tensors`; the passing command kept the quantization argument unset.
    - The engine completed full model load, KV-cache initialization, `compressed-tensors -> AWQ` conversion, SM70 warmup, and one generation request from the worktree path on 8 V100 GPUs.

## Qwen3.5-27B-AWQ Serve Benchmarks

- 2026-04-14:
  - Server command parameters:
    - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    - `CUDA_VISIBLE_DEVICES=2,3,4,5`
    - `model=/mnt/data6/models/Qwen3.5-27B-AWQ`
    - `quantization=awq`
    - `dtype=float16`
    - `tensor_parallel_size=2`
    - `attention_backend=FLASH_ATTN`
    - `compilation_config={"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}`
    - `host=127.0.0.1`
    - `port=28000`
  - Health and inference checks:
    - `curl -sf http://127.0.0.1:28000/health`: success
    - `curl -sS http://127.0.0.1:28000/v1/models`: returned `/mnt/data6/models/Qwen3.5-27B-AWQ`
    - `curl -sS http://127.0.0.1:28000/v1/completions ...`: success
    - Minimal completion result:
      - output prefix: `<think>`
      - `finish_reason=length`
  - 1k benchmark:
    - Command: `vllm bench serve --backend vllm --host 127.0.0.1 --port 28000 --endpoint /v1/completions --model /mnt/data6/models/Qwen3.5-27B-AWQ --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 8 --max-concurrency 1 --save-result --save-detailed --result-dir /tmp/v0190-qwen27-1k`
    - Result: `Successful requests: 8`, `Failed requests: 0`
    - `Mean TTFT (ms): 889.54`
    - `Output token throughput (tok/s): 42.09`
    - Result file: `/tmp/v0190-qwen27-1k/vllm-infqps-concurrency1-Qwen3.5-27B-AWQ-20260414-202235.json`
  - 32k benchmark:
    - Command: `vllm bench serve --backend vllm --host 127.0.0.1 --port 28000 --endpoint /v1/completions --model /mnt/data6/models/Qwen3.5-27B-AWQ --dataset-name random --random-input-len 31744 --random-output-len 128 --num-prompts 2 --max-concurrency 1 --save-result --save-detailed --result-dir /tmp/v0190-qwen27-32k`
    - Result: `Successful requests: 2`, `Failed requests: 0`
    - `Mean TTFT (ms): 22267.38`
    - `Output token throughput (tok/s): 5.17`
    - Result file: `/tmp/v0190-qwen27-32k/vllm-infqps-concurrency1-Qwen3.5-27B-AWQ-20260414-202503.json`
  - Notes:
    - The worktree-local server completed model load, AOT cache load, SM70 AWQ warmup, CUDA graph capture, and OpenAI route startup on V100.
    - The earlier `FlashAttention only supports Ampere GPUs or newer.` runtime blocker is no longer reproduced on this serving path.

## Qwen3.5-122B-A10B-AWQ-4bit Extended Validation

- 2026-04-15:
  - Command parameters:
    - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    - `CUDA_VISIBLE_DEVICES=2,3,4,5`
    - `model=/mnt/data6/models/Qwen3.5-122B-A10B-AWQ-4bit`
    - `dtype=half`
    - `tensor_parallel_size=4`
    - `gpu_memory_utilization=0.90`
    - `max_model_len=4096`
    - `enforce_eager=True`
    - `attention_backend=FLASH_ATTN`
  - Effective quantization path:
    - `quantization=compressed-tensors` was auto-detected from the model `config.json`
    - Both branches selected `CompressedTensorsSM70WNA16MoEMethod (TurboMind SM70 kernels)`
    - The worktree completed load-time `compressed-tensors -> AWQ` conversion for the MoE experts and then ran SM70 AWQ warmup
  - Result:
    - Output text:
      - `<think>`
      - `</think>`
      - `384`
    - `finish_reason=stop`
  - Notes:
    - The earlier failing smoke command was forcing `quantization=awq` for a model whose `config.json` advertises `quant_method=compressed-tensors`.
    - That explicit mismatch was reproduced on both `feature/vllm-0190-upstream-split` and `chore/ignore-project-worktrees`, so it is not a split-only runtime regression.
    - With the explicit `quantization` override removed, the split worktree loaded and generated successfully on `CUDA_VISIBLE_DEVICES=2,3,4,5`.
    - A source-path comparison run on `chore/ignore-project-worktrees` showed the same auto-detected `compressed-tensors` path and the same SM70 TurboMind MoE method selection before manual stop.

## Remaining Gaps

- Prefill/decode chart parity with the root branch is still not collected in this verification pass.
