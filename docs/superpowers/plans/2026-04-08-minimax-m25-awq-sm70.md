# MiniMax-M2.5-AWQ SM70 Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/mnt/data6/models/MiniMax-M2.5-AWQ` follow TurboMind's `moe-only quant + per-layer expert skip` semantics inside `1Cat-vLLM`, so `model.layers.0` experts stay unquantized while later compatible experts use `AWQSM70MoEMethod` on `SM70`, while keeping the patch small enough to be replayed when `vllm` is upgraded independently.

**Architecture:** Keep `MiniMaxM2ForCausalLM` and the existing vLLM execution path unchanged. Fix the decision point in `AWQConfig.get_quant_method()` for `FusedMoE` so it honors `modules_to_not_convert` before choosing the SM70 TurboMind MoE kernel, and lock that behavior with deterministic CPU-side unit tests plus one manual local-model smoke run on V100. The default implementation target is one core `vllm` file plus one test file; if a real-model mismatch later proves HF config names do not align with runtime prefixes, prefer a model-local mapper over widening global quantization behavior.

**Tech Stack:** Python, PyTorch, vLLM quantization stack, `FusedMoE`, TurboMind-backed `AWQSM70MoEMethod`, pytest

---

## File Structure

### Files to modify

- `vllm/model_executor/layers/quantization/awq.py`
  - Responsibility: choose the quantization method for `LinearBase` and `FusedMoE`.
  - Planned change: for `FusedMoE`, run the same skip check already used by `AWQMarlinConfig` before entering the SM70 branch.

### Files to create

- `tests/quantization/test_minimax_m2_awq_sm70.py`
  - Responsibility: pin the exact MiniMax-M2.5-AWQ decision semantics that matter for SM70:
    - `model.layers.0.mlp.experts` returns `UnquantizedFusedMoEMethod`
    - `model.layers.1.mlp.experts` returns `AWQSM70MoEMethod`
    - incompatible MoE shapes still fall back to `MoeWNA16Method`

### Reference-only files

- `vllm/model_executor/models/minimax_m2.py`
  - Confirm that MiniMax M2 uses `model.layers.N.mlp.experts` as the runtime prefix.
  - Only becomes a code-change target if the local model smoke proves HF config names must be mapped into vLLM runtime names.
- `vllm/model_executor/layers/quantization/awq_marlin.py`
  - Mirror the existing MoE skip behavior instead of inventing new logic.
- `/mnt/data/apps/lmdeploy/lmdeploy/turbomind/deploy/converter.py`
  - Reference for TurboMind's `moe-only quant` and per-layer expert skip semantics.

## Upgrade Constraint

- Keep the default patch surface to:
  - `vllm/model_executor/layers/quantization/awq.py`
  - `tests/quantization/test_minimax_m2_awq_sm70.py`
- Do not introduce a new `MiniMaxM2AWQConfig`, planner, or custom quantization subclass in the first landing.
- Do not touch `csrc/`, `CMakeLists.txt`, `awq_sm70_moe.py`, or `minimax_m2.py` unless the local target-model smoke proves the bug cannot be fixed from the existing AWQ decision point.
- If name translation is needed after real-model validation, prefer adding `hf_to_vllm_mapper` or an equivalent model-local mapping in `vllm/model_executor/models/minimax_m2.py` instead of broadening generic skip heuristics.

## Task 1: Fix SM70 AWQ MoE Selection for MiniMax-M2.5-AWQ

**Files:**
- Create: `tests/quantization/test_minimax_m2_awq_sm70.py`
- Modify: `vllm/model_executor/layers/quantization/awq.py`
- Reference: `vllm/model_executor/layers/quantization/awq_marlin.py`

- [ ] **Step 1: Write the failing test**

Create `tests/quantization/test_minimax_m2_awq_sm70.py` with this exact content:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_sm70_moe import AWQSM70MoEMethod
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method


def _make_moe_layer(
    *,
    hidden_size: int = 256,
    intermediate_size: int = 256,
) -> FusedMoE:
    return FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=torch.float16,
        prefix="model.layers.0.mlp.experts",
    )


def _make_awq_config() -> AWQConfig:
    return AWQConfig(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        modules_to_not_convert=["self_attn", "block_sparse_moe.gate", "model.layers.0."],
    )


def test_awq_fused_moe_skips_layer_zero_on_sm70(monkeypatch) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_layer()

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.0.mlp.experts",
    )

    assert isinstance(quant_method, UnquantizedFusedMoEMethod)


def test_awq_fused_moe_uses_sm70_method_for_non_skipped_layer(monkeypatch) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_layer()

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.1.mlp.experts",
    )

    assert isinstance(quant_method, AWQSM70MoEMethod)


def test_awq_fused_moe_falls_back_when_sm70_shape_is_incompatible(monkeypatch) -> None:
    monkeypatch.setattr(
        AWQConfig,
        "_is_sm70_available",
        staticmethod(lambda: True),
    )
    quant_config = _make_awq_config()
    layer = _make_moe_layer(intermediate_size=192)

    quant_method = quant_config.get_quant_method(
        layer,
        prefix="model.layers.1.mlp.experts",
    )

    assert isinstance(quant_method, MoeWNA16Method)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/quantization/test_minimax_m2_awq_sm70.py
```

Expected:

- `test_awq_fused_moe_skips_layer_zero_on_sm70` fails
- Failure shows the returned type is `AWQSM70MoEMethod` instead of `UnquantizedFusedMoEMethod`
- The other two tests may already pass; that is acceptable

- [ ] **Step 3: Write minimal implementation**

Modify `vllm/model_executor/layers/quantization/awq.py` in two places.

1. Update the imports at the top so `UnquantizedFusedMoEMethod` is available in this file.

```python
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
```

2. Add the MoE skip check at the start of the `elif isinstance(layer, FusedMoE):` branch.

```python
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix,
                getattr(self, "modules_to_not_convert", []),
                skip_with_substr=True,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)

            # SM70 (V100): use TurboMind GEMM kernels for MoE,
            # since Marlin requires SM75+.
            if self._is_sm70_available():
                # SM70 (V100): TurboMind s884h kernels require:
                #   K % 8 == 0, N % 8 == 0, K % group_size == 0
                # No requirement on (K/group_size) % 8.
                moe_cfg = layer.moe_config
                hidden = moe_cfg.hidden_dim
                inter = moe_cfg.intermediate_size_per_partition
                gs = self.group_size
                sm70_compatible = (
                    gs in (32, 64, 128)
                    and hidden % gs == 0
                    and inter % gs == 0
                    and hidden % 8 == 0
                    and inter % 8 == 0
                )
                if sm70_compatible:
                    from .awq_sm70_moe import AWQSM70MoEMethod

                    return AWQSM70MoEMethod(
                        weight_bits=self.weight_bits,
                        group_size=self.group_size,
                        zero_point=self.zero_point,
                        moe=moe_cfg,
                    )
                else:
                    logger.warning_once(
                        f"Layer '{prefix}' MoE dimensions incompatible "
                        "with SM70 TurboMind kernels "
                        f"(hidden={hidden}, inter={inter}, "
                        f"group_size={gs}). "
                        "Falling back to MoeWNA16 kernels."
                    )
                    from .moe_wna16 import MoeWNA16Config

                    config = {
                        "quant_method": "awq",
                        "bits": self.weight_bits,
                        "group_size": self.group_size,
                        "zero_point": self.zero_point,
                        "lm_head": False,
                        "modules_to_not_convert": self.modules_to_not_convert,
                    }
                    return MoeWNA16Config.from_config(
                        config
                    ).get_quant_method(layer, prefix)
```

Do not modify `vllm/model_executor/models/minimax_m2.py` in this task. The runtime prefix `model.layers.N.mlp.experts` already matches the `model.layers.0.` skip rule through substring matching. Also do not refactor unrelated parts of `AWQConfig`; keep the branch shape close to `AWQMarlinConfig` so future `vllm` upgrades can reapply the patch mechanically.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest -q tests/quantization/test_minimax_m2_awq_sm70.py
```

Expected:

```text
3 passed
```

Then run one additional regression to make sure the generic MoE ignore path still behaves as expected:

```bash
pytest -q tests/quantization/test_compressed_tensors.py -k test_compressed_tensors_moe_ignore_with_model
```

Expected:

- `1 passed` if the required test model is available locally
- or `1 skipped` if the environment skips that integration case
- no new failure caused by the AWQ change

- [ ] **Step 5: Audit patch surface before commit**

Run:

```bash
git diff --stat -- vllm/model_executor/layers/quantization/awq.py tests/quantization/test_minimax_m2_awq_sm70.py
```

Expected:

- Only `vllm/model_executor/layers/quantization/awq.py` and `tests/quantization/test_minimax_m2_awq_sm70.py` appear in the diff for the default landing
- No accidental edits to `csrc/`, `CMakeLists.txt`, `awq_sm70_moe.py`, or `minimax_m2.py`

- [ ] **Step 6: Commit**

```bash
git add vllm/model_executor/layers/quantization/awq.py tests/quantization/test_minimax_m2_awq_sm70.py
git commit -m "修正 SM70 上 MiniMax AWQ MoE 的按层跳过逻辑"
```

## Manual Verification

After Task 1 is green, run one local smoke check against the actual target model on V100. This is not a committed test; it is a developer verification step for this machine.

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python - <<'PY'
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.awq_sm70_moe import AWQSM70MoEMethod

model_path = "/mnt/data6/models/MiniMax-M2.5-AWQ"
llm = LLM(
    model=model_path,
    quantization="awq",
    tensor_parallel_size=2,
    dtype="half",
    enforce_eager=True,
    gpu_memory_utilization=0.80,
    max_model_len=4096,
)

def inspect(model):
    layer0 = model.model.layers[0].block_sparse_moe.experts
    layer1 = model.model.layers[1].block_sparse_moe.experts
    assert isinstance(layer0.quant_method, UnquantizedFusedMoEMethod), type(layer0.quant_method)
    assert isinstance(layer1.quant_method, AWQSM70MoEMethod), type(layer1.quant_method)
    return type(layer0.quant_method).__name__, type(layer1.quant_method).__name__

print(llm.apply_model(inspect))
outputs = llm.generate(
    "Hello",
    sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
    use_tqdm=False,
)
print(outputs[0].outputs[0].text)
PY
```

Expected:

- `apply_model` prints something equivalent to `('UnquantizedFusedMoEMethod', 'AWQSM70MoEMethod')`
- generation returns non-empty text
- no startup failure caused by forcing layer 0 into an unquantized MoE path

If this smoke test fails specifically because the HF config ignore names do not map cleanly to vLLM runtime prefixes, do not widen the generic AWQ skip logic further. Instead, revise the plan to add a model-local mapper in `vllm/model_executor/models/minimax_m2.py` and keep the generic `awq.py` patch minimal.

## Self-Review

### Spec coverage

- `moe-only quant` semantics: covered by Task 1 implementation and tests
- `model.layers.0` experts stay `fp16`: covered by `test_awq_fused_moe_skips_layer_zero_on_sm70`
- later compatible experts use `AWQSM70MoEMethod`: covered by `test_awq_fused_moe_uses_sm70_method_for_non_skipped_layer`
- incompatible shapes still fallback: covered by `test_awq_fused_moe_falls_back_when_sm70_shape_is_incompatible`
- local target-model validation: covered by Manual Verification
- independent `vllm` upgrade friendliness: covered by `Upgrade Constraint`, the restricted patch surface, and the pre-commit diff audit

### Placeholder scan

- No `TODO`
- No `TBD`
- No unresolved file paths
- No “similar to previous task” shortcuts

### Type consistency

- Quant method names used consistently:
  - `UnquantizedFusedMoEMethod`
  - `AWQSM70MoEMethod`
  - `MoeWNA16Method`
- Runtime prefix used consistently as `model.layers.N.mlp.experts`
