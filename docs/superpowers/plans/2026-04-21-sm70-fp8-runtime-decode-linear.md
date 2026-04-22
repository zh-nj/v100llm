# SM70 FP8 Runtime Decode Dense Linear Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current experimental `SM70 FP8 -> AWQ` fallback with an `FP8-resident + runtime panel decode + SM70 f16 GEMM` path for dense and fused/merged linear layers.

**Architecture:** Keep serialized block-FP8 weights resident in GPU memory, normalize and pad them at load time without changing the quantized format, then add a new `sm70` runtime-decode custom op that decodes one `N` panel at a time to transient `f16`, converts that panel to TurboMind layout with the existing `prepare_sm70_f16_weight()` helper, and runs the existing `SM70` dense GEMM. Python-side quantization stays narrow: only `LinearBase` on `sm70`, only serialized block-FP8, and `MoE / KV cache` remain rejected.

**Tech Stack:** Python 3.13, Conda `gptq`, PyTorch custom ops, C++/CUDA, TurboMind SM70 GEMM helpers in `csrc/quantization/awq/awq_sm70_gemm.cu`, `pytest`, local model `/mnt/data6/models/Qwen3.5-0.8B-FP8`

---

## File Map

- Modify: `vllm/model_executor/layers/quantization/fp8.py`
  - Replace the current experimental `Fp8SM70LinearMethod` AWQ-repack logic with `Fp8SM70RuntimeDecodeLinearMethod`
  - Keep FP8 resident after loading and record runtime-decode metadata
  - Continue rejecting serialized FP8 `MoE` on `sm70`
- Modify: `vllm/config/vllm.py`
  - Keep the quantization capability gate narrow and only allow the runtime-decode `sm70` FP8 linear path
- Modify: `vllm/model_executor/layers/linear.py`
  - Update `WEIGHT_LOADER_V2_SUPPORTED` to the new method name
- Modify: `vllm/_custom_ops.py`
  - Add Python wrappers and fake registrations for `sm70_fp8_runtime_gemm{,_out}`
- Modify: `csrc/ops.h`
  - Declare the new `sm70_fp8_runtime_gemm{,_out}` entry points
- Modify: `csrc/torch_bindings.cpp`
  - Register the new custom op schema and CUDA implementations
- Modify: `csrc/quantization/awq/awq_sm70_gemm.cu`
  - Add runtime-decode validation
  - Reuse `prepare_sm70_f16_weight()` and the existing SM70 dense GEMM path for transient panels
  - Keep LUT cache and workspace behavior aligned with existing SM70 dense paths
- Modify: `vllm/model_executor/warmup/awq_sm70_warmup.py`
  - Extend the SM70 warmup routine to prime runtime-decode dense layers
- Modify: `tests/quantization/test_fp8_sm70.py`
  - Rewrite the current AWQ-repack expectations to the runtime-decode behavior
  - Keep the `modules_to_not_convert` alias regression and the `MoE` rejection test
- Create: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`
  - Add GPU kernel correctness coverage for single-panel and multi-panel runtime decode
- Create: `tests/model_executor/test_sm70_runtime_warmup.py`
  - Cover the new warmup branch without requiring a full engine startup

## Task 1: Switch Selection Tests From AWQ Repack To Runtime Decode

**Files:**
- Modify: `tests/quantization/test_fp8_sm70.py`
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `vllm/config/vllm.py`
- Modify: `vllm/model_executor/layers/linear.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Rewrite the red tests so they target the runtime-decode method name and merged-layer selection**

Edit `tests/quantization/test_fp8_sm70.py` so the existing selection test becomes:

```python
def test_fp8_sm70_linear_uses_runtime_decode_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = _make_linear(monkeypatch)

    assert layer.quant_method.__class__.__name__ == (
        "Fp8SM70RuntimeDecodeLinearMethod"
    )
```

Add a merged-layer regression next to it:

```python
def test_fp8_sm70_merged_linear_uses_runtime_decode_method(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    _patch_single_rank_params(monkeypatch)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = MergedColumnParallelLinear(
        input_size=256,
        output_sizes=[128, 128, 64],
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_fp8_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )

    assert layer.quant_method.__class__.__name__ == (
        "Fp8SM70RuntimeDecodeLinearMethod"
    )
```

- [ ] **Step 2: Run the rewritten tests and verify they fail for the current experimental AWQ path**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_linear_uses_runtime_decode_method \
tests/quantization/test_fp8_sm70.py::test_fp8_sm70_merged_linear_uses_runtime_decode_method -v'
```

Expected:

- both tests fail because the current class name is still `Fp8SM70LinearMethod`

- [ ] **Step 3: Rename the Python method and keep the capability gate narrow**

In `vllm/model_executor/layers/quantization/fp8.py`, rename the class and selection branch:

```python
class Fp8SM70RuntimeDecodeLinearMethod(LinearMethodBase):
    ...

if (
    _get_current_capability_int() == 70
    and self.supports_sm70_checkpoint_fallback()
):
    return Fp8SM70RuntimeDecodeLinearMethod(self)
```

In `vllm/model_executor/layers/linear.py`, replace the supported name:

```python
WEIGHT_LOADER_V2_SUPPORTED = [
    ...
    "Fp8SM70RuntimeDecodeLinearMethod",
    ...
]
```

Keep the capability gate in `vllm/config/vllm.py` narrow:

```python
allow_sm70_fp8_fallback = (
    capability == 70
    and hasattr(quant_config, "supports_sm70_checkpoint_fallback")
    and quant_config.supports_sm70_checkpoint_fallback()
)
```

- [ ] **Step 4: Re-run the selection tests and confirm they turn green**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_linear_uses_runtime_decode_method \
tests/quantization/test_fp8_sm70.py::test_fp8_sm70_merged_linear_uses_runtime_decode_method -v'
```

Expected:

- both tests pass

- [ ] **Step 5: Commit the selection/gate rename**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/layers/quantization/fp8.py \
  vllm/config/vllm.py \
  vllm/model_executor/layers/linear.py \
  tests/quantization/test_fp8_sm70.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "test: rename sm70 fp8 linear path to runtime decode"
```

## Task 2: Keep FP8 Resident And Dispatch Python Apply Through The New Runtime Op

**Files:**
- Modify: `tests/quantization/test_fp8_sm70.py`
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Replace the current AWQ-specific tests with resident-layout and runtime-op dispatch tests**

Replace `test_fp8_sm70_linear_repacks_to_awq_and_prepares()` with:

```python
def test_fp8_sm70_process_keeps_fp8_resident_and_records_runtime_meta(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )

    layer = _make_linear(monkeypatch, output_size=384)
    _populate_fp8_block_weights(layer)

    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.numel() > 0
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer._sm70_fp8_runtime_prepared is True
    assert layer._sm70_fp8_panel_n == 128
    assert layer._sm70_fp8_block_shape == (128, 128)
    assert layer._sm70_fp8_output_size == 384
    assert layer._sm70_fp8_logical_widths == (384,)
    assert not hasattr(layer, "_awq_sm70_prepared")
```

Replace `test_fp8_sm70_linear_apply_uses_awq_gemm()` with:

```python
def test_fp8_sm70_apply_calls_runtime_decode_custom_op(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.fp8"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_panel_n = 128
    layer._sm70_fp8_block_shape = (128, 128)

    called = {}

    def fake_runtime_gemm_out(out, x, weight, weight_scale, block_n, block_k, panel_n):
        called["shape"] = (tuple(out.shape), tuple(x.shape), tuple(weight.shape))
        called["scale_shape"] = tuple(weight_scale.shape)
        called["params"] = (block_n, block_k, panel_n)
        out.copy_(torch.full_like(out, 7))

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert called["shape"] == ((2, 384), (2, 256), (384, 256))
    assert called["scale_shape"] == (3, 2)
    assert called["params"] == (128, 128, 128)
    assert torch.all(out == 7)
```

Add a merged/tail regression next to it:

```python
def test_fp8_sm70_apply_slices_runtime_decode_output_to_logical_width(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fp8_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.fp8"
    )
    ops_module = importlib.import_module("vllm._custom_ops")

    method_cls = getattr(fp8_module, "Fp8SM70RuntimeDecodeLinearMethod")
    method = method_cls(_make_fp8_config())

    layer = torch.nn.Module()
    layer.weight = torch.zeros(384, 256, dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32)
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_panel_n = 128
    layer._sm70_fp8_block_shape = (128, 128)
    layer._sm70_fp8_output_size = 320
    layer._sm70_fp8_logical_widths = (128, 128, 64)

    def fake_runtime_gemm_out(out, x, weight, weight_scale, block_n, block_k, panel_n):
        out.copy_(
            torch.arange(
                out.numel(),
                dtype=out.dtype,
                device=out.device,
            ).reshape_as(out)
        )

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    out = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert tuple(out.shape) == (2, 320)
    assert torch.equal(
        out,
        torch.arange(2 * 384, dtype=torch.float16).reshape(2, 384)[:, :320],
    )
```

Leave these existing regressions in the file unchanged:

- `test_fp8_config_keeps_modules_to_not_convert_alias()`
- `test_fp8_sm70_serialized_moe_raises_clear_error()`

- [ ] **Step 2: Run the updated tests and confirm they fail because the current code still empties FP8 weights and calls AWQ**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_process_keeps_fp8_resident_and_records_runtime_meta \
tests/quantization/test_fp8_sm70.py::test_fp8_sm70_apply_calls_runtime_decode_custom_op \
tests/quantization/test_fp8_sm70.py::test_fp8_sm70_apply_slices_runtime_decode_output_to_logical_width -v'
```

Expected:

- the prepare test fails because the current code converts weights to AWQ and empties `layer.weight`
- the apply test fails because the current code still calls `awq_gemm_sm70`

- [ ] **Step 3: Replace the current AWQ-specific Python prepare/apply logic with a resident FP8 layout and runtime decode dispatch**

In `vllm/model_executor/layers/quantization/fp8.py`, delete the current AWQ helpers and replace the method body with resident-layout logic:

```python
class Fp8SM70RuntimeDecodeLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.panel_n = 128
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        if not self.block_quant or self.act_q_static:
            raise ValueError(
                "SM70 runtime decode only supports serialized block-FP8 "
                "weights with dynamic activation scaling."
            )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight, weight_scale_inv = process_fp8_weight_block_strategy(
            layer.weight, layer.weight_scale_inv
        )
        replace_parameter(layer, "weight", weight.data)
        replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)
        layer._sm70_fp8_runtime_prepared = True
        layer._sm70_fp8_panel_n = self.panel_n
        layer._sm70_fp8_block_shape = tuple(self.weight_block_size)
        layer._sm70_fp8_output_size = int(layer.output_size_per_partition)
        layer._sm70_fp8_logical_widths = tuple(layer.logical_widths)
        layer.input_scale = None
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        if not getattr(layer, "_sm70_fp8_runtime_prepared", False):
            raise RuntimeError("SM70 FP8 runtime decode weights were not prepared.")

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        padded_out_dim = int(layer.weight.size(0))
        logical_out_dim = int(
            getattr(layer, "_sm70_fp8_output_size", padded_out_dim)
        )
        out_padded = torch.empty(
            (x_2d.size(0), padded_out_dim),
            dtype=x_2d.dtype,
            device=x_2d.device,
        )
        block_n, block_k = layer._sm70_fp8_block_shape
        ops.sm70_fp8_runtime_gemm_out(
            out_padded,
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            block_n,
            block_k,
            layer._sm70_fp8_panel_n,
        )
        out = out_padded[:, :logical_out_dim]
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))
```

- [ ] **Step 4: Re-run the Python-only tests and verify the runtime-decode path turns green**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/quantization/test_fp8_sm70.py -v'
```

Expected:

- all `tests/quantization/test_fp8_sm70.py` tests pass
- no test references `_awq_sm70_*` anymore

- [ ] **Step 5: Commit the resident-layout Python path**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/layers/quantization/fp8.py \
  tests/quantization/test_fp8_sm70.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: keep sm70 fp8 linear weights resident for runtime decode"
```

## Task 3: Add The Runtime-Decode Custom Op And Prove Kernel Correctness

**Files:**
- Create: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`
- Modify: `vllm/_custom_ops.py`
- Modify: `csrc/ops.h`
- Modify: `csrc/torch_bindings.cpp`
- Modify: `csrc/quantization/awq/awq_sm70_gemm.cu`
- Test: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`

- [ ] **Step 1: Write a failing GPU kernel test for single-panel and multi-panel decode**

Create `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops


def _require_sm70():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 runtime decode test requires a V100")


def _to_block_fp8(weight_fp16: torch.Tensor):
    block_n = 128
    block_k = 128
    out_dim, in_dim = weight_fp16.shape
    scales = []
    q_rows = []
    for row_start in range(0, out_dim, block_n):
        row_end = min(row_start + block_n, out_dim)
        row_q = []
        row_scales = []
        for col_start in range(0, in_dim, block_k):
            col_end = min(col_start + block_k, in_dim)
            block = weight_fp16[row_start:row_end, col_start:col_end].float()
            scale = block.abs().amax().clamp(min=1e-6) / 448.0
            row_scales.append(scale)
            row_q.append((block / scale).to(torch.float8_e4m3fn).to(torch.float16))
        q_rows.append(torch.cat(row_q, dim=1))
        scales.append(torch.stack(row_scales))
    q = torch.cat(q_rows, dim=0).to(torch.float8_e4m3fn)
    s = torch.stack(scales).to(torch.float32).cuda()
    return q.cuda(), s


@pytest.mark.cuda
@pytest.mark.parametrize("out_dim", [128, 320])
def test_sm70_fp8_runtime_gemm_matches_reference(out_dim: int):
    _require_sm70()
    torch.manual_seed(0)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(out_dim, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)

    out = ops.sm70_fp8_runtime_gemm(
        x,
        w_fp8,
        w_scale,
        128,
        128,
        128,
    )
    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)
```

- [ ] **Step 2: Run the new kernel test and confirm it fails because the op is not bound**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -v -s'
```

Expected:

- the test fails with `AttributeError` / missing `_C::sm70_fp8_runtime_gemm`

- [ ] **Step 3: Bind and implement the runtime-decode op inside the existing SM70 GEMM source**

In `csrc/ops.h`, add:

```cpp
torch::Tensor sm70_fp8_runtime_gemm(torch::Tensor _in_feats,
                                    torch::Tensor _weight,
                                    torch::Tensor _weight_scale,
                                    int64_t block_n,
                                    int64_t block_k,
                                    int64_t panel_n);

void sm70_fp8_runtime_gemm_out(torch::Tensor out,
                               torch::Tensor _in_feats,
                               torch::Tensor _weight,
                               torch::Tensor _weight_scale,
                               int64_t block_n,
                               int64_t block_k,
                               int64_t panel_n);
```

In `csrc/torch_bindings.cpp`, register:

```cpp
ops.def(
    "sm70_fp8_runtime_gemm(Tensor _in_feats, Tensor _weight, Tensor _weight_scale, "
    "int block_n, int block_k, int panel_n) -> Tensor");
ops.impl("sm70_fp8_runtime_gemm", torch::kCUDA, &sm70_fp8_runtime_gemm);

ops.def(
    "sm70_fp8_runtime_gemm_out(Tensor(a!) out, Tensor _in_feats, Tensor _weight, "
    "Tensor _weight_scale, int block_n, int block_k, int panel_n) -> ()");
ops.impl("sm70_fp8_runtime_gemm_out", torch::kCUDA, &sm70_fp8_runtime_gemm_out);
```

In `vllm/_custom_ops.py`, add:

```python
def sm70_fp8_runtime_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
    panel_n: int,
) -> torch.Tensor:
    return torch.ops._C.sm70_fp8_runtime_gemm(
        input, weight, weight_scale, block_n, block_k, panel_n
    )


def sm70_fp8_runtime_gemm_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
    panel_n: int,
) -> None:
    torch.ops._C.sm70_fp8_runtime_gemm_out(
        out, input, weight, weight_scale, block_n, block_k, panel_n
    )
```

Add fake registrations mirroring the output shape of `weight.size(0)`.

In `csrc/quantization/awq/awq_sm70_gemm.cu`, implement the op by reusing the existing SM70 helpers:

```cpp
void sm70_fp8_runtime_gemm_out(torch::Tensor out,
                               torch::Tensor in_feats,
                               torch::Tensor weight,
                               torch::Tensor weight_scale,
                               int64_t block_n,
                               int64_t block_k,
                               int64_t panel_n) {
  validate_fp8_runtime_inputs(in_feats, weight, weight_scale, out,
                              block_n, block_k, panel_n);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);

  for (int64_t n0 = 0; n0 < n; n0 += panel_n) {
    const int64_t panel_cols = std::min(panel_n, n - n0);
    auto w_panel_q = weight.narrow(0, n0, panel_cols).contiguous();
    auto s_panel = weight_scale.narrow(0, n0 / block_n, (panel_cols + block_n - 1) / block_n);
    auto s_exp = s_panel.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1);
    s_exp = s_exp.slice(0, 0, panel_cols).slice(1, 0, k);
    auto w_panel_f16 = (w_panel_q.to(torch::kFloat32) * s_exp).to(torch::kFloat16);

    auto prepared = prepare_sm70_f16_weight(w_panel_f16, stream);
    auto out_panel = out.narrow(1, n0, panel_cols);
    sm70_f16_gemm_out(out_panel, in_feats, prepared.tm_weight, prepared.k_ld, false);
  }
}

torch::Tensor sm70_fp8_runtime_gemm(torch::Tensor in_feats,
                                    torch::Tensor weight,
                                    torch::Tensor weight_scale,
                                    int64_t block_n,
                                    int64_t block_k,
                                    int64_t panel_n) {
  auto out = torch::empty({in_feats.size(0), weight.size(0)},
                          in_feats.options().dtype(torch::kFloat16));
  sm70_fp8_runtime_gemm_out(
      out, in_feats, weight, weight_scale, block_n, block_k, panel_n);
  return out;
}
```

- [ ] **Step 4: Build the extension and re-run the kernel test**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
MAX_JOBS=8 python setup.py build_ext --inplace'
```

Expected:

- `_C` builds successfully
- no unresolved symbol for `sm70_fp8_runtime_gemm`

Then run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -v -s'
```

Expected:

- both `out_dim=128` and `out_dim=320` pass

- [ ] **Step 5: Commit the new runtime-decode custom op**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/_custom_ops.py \
  csrc/ops.h \
  csrc/torch_bindings.cpp \
  csrc/quantization/awq/awq_sm70_gemm.cu \
  tests/kernels/quantization/test_sm70_fp8_runtime_decode.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: add sm70 fp8 runtime decode gemm"
```

## Task 4: Cover Warmup And Multi-Panel Dense Integration

**Files:**
- Create: `tests/model_executor/test_sm70_runtime_warmup.py`
- Modify: `vllm/model_executor/warmup/awq_sm70_warmup.py`
- Modify: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/model_executor/test_sm70_runtime_warmup.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Add a failing warmup regression for runtime-decode dense layers**

Create `tests/model_executor/test_sm70_runtime_warmup.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.warmup.awq_sm70_warmup import sm70_awq_warmup


class _DummyWorker:
    def __init__(self, layer):
        self.device = torch.device("cuda:0")
        self.scheduler_config = type("Scheduler", (), {"max_num_batched_tokens": 8})()
        self.vllm_config = type(
            "Cfg",
            (),
            {"compilation_config": type("CC", (), {"cudagraph_capture_sizes": [1, 2, 4]})()},
        )()
        self._model = torch.nn.Module()
        self._model.layer = layer

    def get_model(self):
        return self._model


def test_sm70_awq_warmup_handles_runtime_decode_dense(monkeypatch):
    layer = torch.nn.Module()
    layer._sm70_fp8_runtime_prepared = True
    layer._sm70_fp8_panel_n = 128
    layer._sm70_fp8_block_shape = (128, 128)
    layer._sm70_fp8_output_size = 320
    layer.weight = torch.zeros(320, 256, dtype=torch.float8_e4m3fn, device="cuda")
    layer.weight_scale_inv = torch.ones(3, 2, dtype=torch.float32, device="cuda")

    calls = []

    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda *_args, **_kwargs: (7, 0),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.awq_sm70_warmup.ops.sm70_fp8_runtime_gemm_out",
        lambda out, x, w, s, bn, bk, pn: calls.append((tuple(out.shape), pn)),
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda *_args, **_kwargs: None)

    sm70_awq_warmup(_DummyWorker(layer))

    assert calls
    assert calls[0][1] == 128
    assert calls[0][0][1] == 320
```

- [ ] **Step 2: Run the warmup regression and verify it fails because the warmup code only knows about AWQ dense layers**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/model_executor/test_sm70_runtime_warmup.py -v -s'
```

Expected:

- the test fails because `sm70_awq_warmup()` ignores `_sm70_fp8_runtime_prepared`

- [ ] **Step 3: Extend the existing SM70 warmup module to prime runtime-decode dense layers**

In `vllm/model_executor/warmup/awq_sm70_warmup.py`, add a runtime-decode dense iterator and warmup branch:

```python
def _iter_unique_runtime_decode_dense_layers(model: torch.nn.Module):
    seen: set[tuple[int, int, int, int, int]] = set()
    for layer in model.modules():
        if not getattr(layer, "_sm70_fp8_runtime_prepared", False):
            continue
        block_n, block_k = layer._sm70_fp8_block_shape
        key = (
            int(layer.weight.shape[1]),
            int(layer.weight.shape[0]),
            block_n,
            block_k,
            int(layer._sm70_fp8_panel_n),
        )
        if key in seen:
            continue
        seen.add(key)
        yield layer


def _warmup_runtime_decode_dense_layers(dense_layers, m_values):
    calls = 0
    for layer in dense_layers:
        device = layer.weight.device
        k_dim = int(layer.weight.shape[1])
        n_dim = int(layer.weight.shape[0])
        block_n, block_k = layer._sm70_fp8_block_shape
        for m_dim in m_values:
            x = torch.empty((m_dim, k_dim), dtype=torch.float16, device=device)
            out = torch.empty((m_dim, n_dim), dtype=torch.float16, device=device)
            ops.sm70_fp8_runtime_gemm_out(
                out,
                x,
                layer.weight,
                layer.weight_scale_inv,
                block_n,
                block_k,
                layer._sm70_fp8_panel_n,
            )
            calls += 1
    return calls
```

Fold that branch into `sm70_awq_warmup()` and update the log string to mention both AWQ and runtime-decode dense warmup.

- [ ] **Step 4: Re-run the warmup and quantization suites**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/model_executor/test_sm70_runtime_warmup.py \
tests/quantization/test_fp8_sm70.py -v -s'
```

Expected:

- both suites pass

- [ ] **Step 5: Commit the warmup integration**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/warmup/awq_sm70_warmup.py \
  tests/model_executor/test_sm70_runtime_warmup.py \
  tests/quantization/test_fp8_sm70.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: warm up sm70 fp8 runtime decode dense shapes"
```

## Task 5: Full Verification And Real-Model Smoke On Qwen3.5-0.8B-FP8

**Files:**
- No new code by default; verify the files changed in Tasks 1-4
- Test: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`
- Test: `tests/model_executor/test_sm70_runtime_warmup.py`

- [ ] **Step 1: Run the focused automated verification suite on an SM70 GPU**

First inspect GPU availability:

```bash
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits
```

Expected:

- identify a free `Tesla V100-SXM2-32GB` device (for example `6` or `8`)

Then run the focused suite on that SM70 device:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
pytest tests/quantization/test_fp8_sm70.py \
tests/kernels/quantization/test_sm70_fp8_runtime_decode.py \
tests/model_executor/test_sm70_runtime_warmup.py -v -s'
```

Expected:

- all three test files pass on `sm70`

- [ ] **Step 2: Run a real load-and-generate smoke test with the provided FP8 checkpoint**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python - <<'"'"'PY'"'"'
from vllm import LLM, SamplingParams

llm = LLM(
    model="/mnt/data6/models/Qwen3.5-0.8B-FP8",
    trust_remote_code=True,
    max_model_len=256,
    gpu_memory_utilization=0.35,
    enforce_eager=True,
)
outputs = llm.generate(
    ["你好"],
    SamplingParams(max_tokens=8, temperature=0.0),
)
print(outputs[0].outputs[0].text)
PY'
```

Expected:

- engine starts without `Minimum capability: 75`
- engine does not fail with the old `partially quantized fused layers` error
- a non-empty completion is printed

- [ ] **Step 3: Record the final diff and create the implementation summary commit**

Run:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split status --short
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split diff --stat
```

Expected:

- only the intended runtime-decode files are modified

Then commit:

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/layers/quantization/fp8.py \
  vllm/config/vllm.py \
  vllm/model_executor/layers/linear.py \
  vllm/_custom_ops.py \
  csrc/ops.h \
  csrc/torch_bindings.cpp \
  csrc/quantization/awq/awq_sm70_gemm.cu \
  vllm/model_executor/warmup/awq_sm70_warmup.py \
  tests/quantization/test_fp8_sm70.py \
  tests/kernels/quantization/test_sm70_fp8_runtime_decode.py \
  tests/model_executor/test_sm70_runtime_warmup.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: add sm70 fp8 runtime decode dense linear path"
```
