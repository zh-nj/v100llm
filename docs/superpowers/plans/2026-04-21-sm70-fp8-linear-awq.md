# SM70 FP8 Linear AWQ Repack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable serialized FP8 dense linear layers to run on `sm70` by repacking loaded FP8 weights into the existing SM70 AWQ runtime format.

**Architecture:** Keep FP8 checkpoints as the on-disk format, but on `sm70` convert dense FP8 linear weights into AWQ-style int4 group-quantized tensors during `process_weights_after_loading()`. Reuse the existing `awq_sm70_prepare` and `awq_gemm_sm70` runtime path, and explicitly reject serialized FP8 MoE on `sm70` until a matching compressed expert path exists.

**Tech Stack:** Python quantization methods, PyTorch tensor ops, existing vLLM AWQ/SM70 custom ops, pytest unit tests.

---

### Task 1: Add Failing Selection And Conversion Tests

**Files:**
- Create: `tests/quantization/test_fp8_sm70.py`
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8Config


def test_fp8_sm70_linear_uses_sm70_method(monkeypatch):
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = torch.nn.Linear(256, 64, bias=False)

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
    )

    method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

    assert method.__class__.__name__ == "Fp8SM70LinearMethod"


def test_fp8_sm70_linear_repacks_to_awq_and_prepares(monkeypatch):
    ...


def test_fp8_sm70_serialized_moe_raises_clear_error(monkeypatch):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/quantization/test_fp8_sm70.py -v`
Expected: FAIL because `Fp8SM70LinearMethod` does not exist and the selection logic still routes serialized FP8 to `Fp8LinearMethod`.

- [ ] **Step 3: Write minimal implementation**

```python
def _get_current_capability_int() -> int:
    capability = current_platform.get_device_capability()
    return capability.to_int() if capability is not None else 0


if _get_current_capability_int() == 70:
    return Fp8SM70LinearMethod(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/quantization/test_fp8_sm70.py -v`
Expected: PASS for selection, remaining tests still FAIL.

- [ ] **Step 5: Commit**

```bash
git add tests/quantization/test_fp8_sm70.py vllm/model_executor/layers/quantization/fp8.py
git commit -m "test: cover sm70 fp8 linear selection"
```

### Task 2: Repack FP8 Block Weights Into SM70 AWQ Runtime Format

**Files:**
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fp8_sm70_linear_repacks_to_awq_and_prepares(monkeypatch):
    layer = _make_block_fp8_layer()
    calls = {}

    def fake_prepare(qweight, scales, qzeros, group_size, **kwargs):
        calls["shapes"] = (tuple(qweight.shape), tuple(scales.shape), tuple(qzeros.shape))
        return (
            torch.ones(256, 8, dtype=torch.int32),
            torch.ones(2, 64, dtype=torch.int32),
            torch.tensor([256, 32], dtype=torch.int64),
        )

    monkeypatch.setattr("vllm.model_executor.layers.quantization.fp8.ops.awq_sm70_prepare", fake_prepare)

    method = Fp8SM70LinearMethod(_make_fp8_config())
    method.process_weights_after_loading(layer)

    assert calls["shapes"] == ((256, 8), (2, 64), (2, 8))
    assert layer._awq_sm70_prepared is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_linear_repacks_to_awq_and_prepares -v`
Expected: FAIL because the method does not repack FP8 weights into AWQ tensors.

- [ ] **Step 3: Write minimal implementation**

```python
weight_fp16 = scaled_dequantize(
    layer.weight,
    layer.weight_scale_inv,
    group_shape=GroupShape(*self.weight_block_size),
    out_dtype=torch.float16,
)
qweight, qzeros, scales = _quantize_weight_to_awq(weight_fp16, group_size=128)
tm_weight, tm_scales, meta = ops.awq_sm70_prepare(qweight, scales, qzeros, 128)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_linear_repacks_to_awq_and_prepares -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/quantization/test_fp8_sm70.py vllm/model_executor/layers/quantization/fp8.py
git commit -m "feat: repack sm70 fp8 linear weights to awq"
```

### Task 3: Route Linear Apply Through Existing SM70 AWQ GEMM

**Files:**
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fp8_sm70_linear_apply_uses_awq_gemm(monkeypatch):
    layer = torch.nn.Module()
    layer._awq_sm70_prepared = True
    layer._awq_sm70_weight = torch.zeros(256, 8, dtype=torch.int32)
    layer._awq_sm70_scales = torch.ones(2, 64, dtype=torch.int32)
    layer._awq_sm70_k_ld = 256
    layer._awq_sm70_q_ld = 32

    def fake_awq_gemm_sm70(x, qweight, scales, group_size, k_ld, q_ld):
        return torch.full((x.shape[0], 64), 5, dtype=x.dtype)

    monkeypatch.setattr("vllm.model_executor.layers.quantization.fp8.ops.awq_gemm_sm70", fake_awq_gemm_sm70)

    method = Fp8SM70LinearMethod(_make_fp8_config())
    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert tuple(out.shape) == (2, 64)
    assert torch.all(out == 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_linear_apply_uses_awq_gemm -v`
Expected: FAIL because `apply()` still routes through the standard FP8 kernel path.

- [ ] **Step 3: Write minimal implementation**

```python
if getattr(layer, "_awq_sm70_prepared", False):
    out = ops.awq_gemm_sm70(
        x.view(-1, x.shape[-1]),
        layer._awq_sm70_weight,
        layer._awq_sm70_scales,
        self.sm70_group_size,
        layer._awq_sm70_k_ld,
        layer._awq_sm70_q_ld,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_linear_apply_uses_awq_gemm -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/quantization/test_fp8_sm70.py vllm/model_executor/layers/quantization/fp8.py
git commit -m "feat: run sm70 fp8 linear via awq gemm"
```

### Task 4: Guard Unsupported SM70 FP8 MoE Clearly

**Files:**
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fp8_sm70_serialized_moe_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
    )
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = FusedMoE(num_experts=1, top_k=1, hidden_size=128, intermediate_size=128)

    with pytest.raises(ValueError, match="sm70.*MoE.*not supported"):
        config.get_quant_method(layer, "model.layers.0.moe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_serialized_moe_raises_clear_error -v`
Expected: FAIL because SM70 still returns the default FP8 MoE method.

- [ ] **Step 3: Write minimal implementation**

```python
if _get_current_capability_int() == 70 and self.is_checkpoint_fp8_serialized:
    raise ValueError("Serialized FP8 MoE is not supported on sm70 yet.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_serialized_moe_raises_clear_error -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/quantization/test_fp8_sm70.py vllm/model_executor/layers/quantization/fp8.py
git commit -m "feat: guard unsupported sm70 fp8 moe"
```

### Task 5: Verify Dense SM70 FP8 Unit Coverage

**Files:**
- Modify: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Run the focused unit suite**

Run: `pytest tests/quantization/test_fp8_sm70.py -v`
Expected: PASS.

- [ ] **Step 2: Run adjacent FP8/AWQ regression tests**

Run: `pytest tests/quantization/test_fp8.py tests/quantization/test_compressed_tensors_sm70.py tests/quantization/test_awq_sm70_compat.py -v`
Expected: PASS or explicit environment skips only.

- [ ] **Step 3: Capture remaining scope**

```text
Not in this plan:
- SM70 serialized FP8 MoE execution
- SM70 FP8 KV cache / attention backend routing
- End-to-end Qwen3.5-0.8B-FP8 V100 runtime verification
```

- [ ] **Step 4: Commit**

```bash
git add tests/quantization/test_fp8_sm70.py
git commit -m "test: verify sm70 fp8 linear awq path"
```
