# SM70 FP8 Runtime-Decode GEMM Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Python/Torch SM70 FP8 runtime-decode hot path with a reusable prepared-representation + explicit-workspace + custom CUDA op foundation for dense FP8 Linear, and validate it on exact-retrieval `1k/32k` benchmarks.

**Architecture:** Keep FP8 weights resident in GPU memory, run a one-time `sm70_fp8_prepare()` after loading, attach prepared tensors and reusable workspace metadata to dense FP8 layers, then execute via `sm70_fp8_runtime_gemm_out()` that decodes one panel at a time into caller-owned workspace and reuses the existing TurboMind-backed `sm70_f16_gemm_out()` path. The first adopter stays `dense Linear`, but the prepare metadata, layout abstraction, and workspace protocol are shaped so tensor-wise, channel-wise, and block-wise FP8 layouts can share one foundation and future `MoE / KV cache` work does not need a data-model rewrite.

**Tech Stack:** Python 3.13, Conda `gptq`, PyTorch custom ops, C++/CUDA, TurboMind SM70 GEMM helpers, `pytest`, local benchmark model `/mnt/data6/models/Qwen3.5-0.8B-FP8`

---

## File Map

- Create: `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py`
  - Own layout constants, prepared-meta indices, workspace dataclass, and reusable workspace allocation helpers for the new foundation
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
  - Replace raw weight/scale runtime usage with prepared tensors and explicit workspace handling inside `Fp8SM70RuntimeDecodeLinearMethod`
  - Keep dense-only `sm70` selection narrow and keep serialized FP8 `MoE` rejected
- Modify: `vllm/_custom_ops.py`
  - Remove Python panel decode implementation from the hot path
  - Add wrappers and fake registrations for `sm70_fp8_prepare`, `sm70_fp8_runtime_gemm`, and `sm70_fp8_runtime_gemm_out` with explicit workspace tensors
- Modify: `csrc/ops.h`
  - Declare the new SM70 FP8 prepare/runtime op signatures
- Modify: `csrc/torch_bindings.cpp`
  - Register the new custom op schemas and CUDA implementations
- Modify: `csrc/quantization/awq/awq_sm70_gemm.cu`
  - Implement the SM70 FP8 prepared representation, layout validation, runtime decode, panel pack, deterministic workspace checks, and reuse of existing TurboMind GEMM helpers
- Modify: `vllm/model_executor/warmup/awq_sm70_warmup.py`
  - Warm up prepared FP8 dense layers through the new runtime op and reusable workspace helpers
- Modify: `tests/quantization/test_fp8_sm70.py`
  - Lock prepared-representation attr names, merged/fused dense behavior, workspace reuse, `modules_to_not_convert` alias behavior, `MoE` rejection, and the existing `Qwen3.5` tuple-shard regression
- Modify: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`
  - Cover prepare/runtime numerics, fake op behavior, tensor/channel/block scale layouts, multi-panel correctness, and deterministic workspace capacity errors
- Modify: `tests/model_executor/test_sm70_runtime_warmup.py`
  - Verify warmup uses the new prepared representation and explicit workspace contract
- Create: `benchmarks/benchmark_sm70_fp8_exact_retrieval.py`
  - Provide a reusable offline benchmark harness that prints `prefill tokens/s`, `decode tokens/s`, `TTFT`, `finish_reason`, and a semantic exact-match conclusion for `1k/32k`
- Create: `tests/benchmarks/test_sm70_fp8_exact_retrieval.py`
  - Unit-test prompt/report helpers so the benchmark output contract does not regress

## Task 1: Lock The Python-Side Prepared Representation And Workspace Contract

**Files:**
- Create: `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py`
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/quantization/test_fp8_sm70.py`

- [ ] **Step 1: Add red tests for prepare-op dispatch and workspace reuse**

Append these tests to `tests/quantization/test_fp8_sm70.py` next to the current SM70 runtime-decode tests:

```python
def test_fp8_sm70_process_calls_prepare_op_and_stores_prepared_tensors(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8._get_current_capability_int",
        lambda: 70,
        raising=False,
    )
    ops_module = importlib.import_module("vllm._custom_ops")
    layer = _make_linear(monkeypatch, output_size=384)
    _populate_fp8_block_weights(layer)
    called = {}

    def fake_prepare(weight, weight_scale, layout_kind, scale_axis, block_n, block_k, panel_n):
        called["args"] = (
            tuple(weight.shape),
            tuple(weight_scale.shape),
            layout_kind,
            scale_axis,
            block_n,
            block_k,
            panel_n,
        )
        prepared_meta = torch.tensor(
            [384, 256, 384, 256, 128, layout_kind, scale_axis, block_n, block_k],
            dtype=torch.int64,
        )
        workspace_meta = torch.tensor([128, 256, 384, 256, 4], dtype=torch.int64)
        return [weight, weight_scale, prepared_meta, workspace_meta]

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        fake_prepare,
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert called["args"] == ((384, 256), (3, 2), 2, -1, 128, 128, 128)
    assert layer._sm70_fp8_prepared_weight is layer.weight
    assert layer._sm70_fp8_prepared_scale is layer.weight_scale_inv
    assert tuple(layer._sm70_fp8_prepared_meta.tolist()) == (
        384, 256, 384, 256, 128, 2, -1, 128, 128
    )
    assert tuple(layer._sm70_fp8_workspace_meta.tolist()) == (128, 256, 384, 256, 4)
```

```python
def test_fp8_sm70_apply_reuses_workspace_and_slices_logical_width(
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
    layer._sm70_fp8_prepared_weight = layer.weight
    layer._sm70_fp8_prepared_scale = layer.weight_scale_inv
    layer._sm70_fp8_prepared_meta = torch.tensor(
        [384, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64
    )
    layer._sm70_fp8_workspace_meta = torch.tensor([128, 256, 384, 256, 4], dtype=torch.int64)
    layer._sm70_fp8_output_size = 320
    layer._sm70_fp8_logical_widths = (128, 128, 64)

    seen_workspace_ptrs = []

    def fake_runtime_gemm_out(
        out,
        x,
        prepared_weight,
        prepared_scale,
        prepared_meta,
        decoded_panel,
        packed_panel,
        meta_buffer,
    ):
        seen_workspace_ptrs.append(
            (decoded_panel.data_ptr(), packed_panel.data_ptr(), meta_buffer.data_ptr())
        )
        out.copy_(
            torch.arange(out.numel(), dtype=out.dtype, device=out.device).reshape_as(out)
        )

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_runtime_gemm_out",
        fake_runtime_gemm_out,
        raising=False,
    )

    out1 = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)
    out2 = method.apply(layer, torch.ones(2, 256, dtype=torch.float16), None)

    assert tuple(out1.shape) == (2, 320)
    assert tuple(out2.shape) == (2, 320)
    assert seen_workspace_ptrs[0] == seen_workspace_ptrs[1]
```

- [ ] **Step 2: Run the new tests and confirm the current code fails before the foundation work**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_process_calls_prepare_op_and_stores_prepared_tensors \
tests/quantization/test_fp8_sm70.py::test_fp8_sm70_apply_reuses_workspace_and_slices_logical_width -v'
```

Expected:

- `process_weights_after_loading()` still never calls `sm70_fp8_prepare`
- `apply()` still calls the old runtime op signature without explicit workspace tensors

- [ ] **Step 3: Add the workspace helper module and switch `fp8.py` to prepared tensors**

Create `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


SM70_FP8_LAYOUT_TENSOR = 0
SM70_FP8_LAYOUT_CHANNEL = 1
SM70_FP8_LAYOUT_BLOCK = 2


@dataclass
class Sm70Fp8RuntimeWorkspace:
    decoded_panel: torch.Tensor
    packed_panel: torch.Tensor
    meta_buffer: torch.Tensor
    capacity_m: int


def alloc_sm70_fp8_workspace(
    workspace_meta: torch.Tensor,
    *,
    device: torch.device,
    m_capacity: int,
) -> Sm70Fp8RuntimeWorkspace:
    decoded_rows, decoded_cols, packed_rows, packed_cols, meta_ints = (
        int(v) for v in workspace_meta.tolist()
    )
    return Sm70Fp8RuntimeWorkspace(
        decoded_panel=torch.empty(
            (decoded_rows, decoded_cols), dtype=torch.float16, device=device
        ),
        packed_panel=torch.empty(
            (packed_rows, packed_cols), dtype=torch.float16, device=device
        ),
        meta_buffer=torch.empty((meta_ints,), dtype=torch.int64, device=device),
        capacity_m=m_capacity,
    )


def get_or_create_sm70_fp8_workspace(
    layer: torch.nn.Module,
    x_2d: torch.Tensor,
) -> Sm70Fp8RuntimeWorkspace:
    cache = getattr(layer, "_sm70_fp8_workspace_cache", None)
    if cache is None:
        cache = {}
        layer._sm70_fp8_workspace_cache = cache
    device_idx = -1 if x_2d.device.index is None else int(x_2d.device.index)
    workspace = cache.get(device_idx)
    if workspace is None or workspace.capacity_m < int(x_2d.shape[0]):
        workspace = alloc_sm70_fp8_workspace(
            layer._sm70_fp8_workspace_meta,
            device=x_2d.device,
            m_capacity=int(x_2d.shape[0]),
        )
        cache[device_idx] = workspace
    return workspace
```

Update `Fp8SM70RuntimeDecodeLinearMethod.process_weights_after_loading()` and `.apply()` in `vllm/model_executor/layers/quantization/fp8.py`:

```python
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode import (
    SM70_FP8_LAYOUT_BLOCK,
    get_or_create_sm70_fp8_workspace,
)
```

```python
prepared_weight, prepared_scale, prepared_meta, workspace_meta = ops.sm70_fp8_prepare(
    layer.weight,
    layer.weight_scale_inv,
    SM70_FP8_LAYOUT_BLOCK,
    -1,
    self.weight_block_size[0],
    self.weight_block_size[1],
    self.panel_n,
)
layer._sm70_fp8_prepared_weight = prepared_weight
layer._sm70_fp8_prepared_scale = prepared_scale
layer._sm70_fp8_prepared_meta = prepared_meta
layer._sm70_fp8_workspace_meta = workspace_meta
layer._sm70_fp8_workspace_cache = {}
```

```python
workspace = get_or_create_sm70_fp8_workspace(layer, x_2d)
ops.sm70_fp8_runtime_gemm_out(
    out_padded,
    x_2d,
    layer._sm70_fp8_prepared_weight,
    layer._sm70_fp8_prepared_scale,
    layer._sm70_fp8_prepared_meta,
    workspace.decoded_panel,
    workspace.packed_panel,
    workspace.meta_buffer,
)
```

- [ ] **Step 4: Re-run the Python contract tests and confirm they pass**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/quantization/test_fp8_sm70.py::test_fp8_sm70_process_calls_prepare_op_and_stores_prepared_tensors \
tests/quantization/test_fp8_sm70.py::test_fp8_sm70_apply_reuses_workspace_and_slices_logical_width -v'
```

Expected:

- both tests pass

- [ ] **Step 5: Commit the Python prepared-representation contract**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py \
  vllm/model_executor/layers/quantization/fp8.py \
  tests/quantization/test_fp8_sm70.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: add sm70 fp8 prepared representation contract"
```

## Task 2: Add The Explicit-Workspace SM70 FP8 Custom Op For Block Layout

**Files:**
- Modify: `vllm/_custom_ops.py`
- Modify: `csrc/ops.h`
- Modify: `csrc/torch_bindings.cpp`
- Modify: `csrc/quantization/awq/awq_sm70_gemm.cu`
- Modify: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`
- Test: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`

- [ ] **Step 1: Replace the old kernel test with prepare/runtime-op tests for the new signature**

Rewrite `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py` so it contains:

```python
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode import (
    alloc_sm70_fp8_workspace,
)


def test_sm70_fp8_prepare_and_runtime_gemm_block_layout_matches_reference():
    _require_sm70()
    torch.manual_seed(0)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(320, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)

    prepared_w, prepared_s, prepared_meta, workspace_meta = ops.sm70_fp8_prepare(
        w_fp8,
        w_scale,
        2,
        -1,
        128,
        128,
        128,
    )
    workspace = alloc_sm70_fp8_workspace(
        workspace_meta,
        device=x.device,
        m_capacity=x.shape[0],
    )
    out = torch.empty((x.shape[0], w_ref.shape[0]), dtype=torch.float16, device="cuda")
    ops.sm70_fp8_runtime_gemm_out(
        out,
        x,
        prepared_w,
        prepared_s,
        prepared_meta,
        workspace.decoded_panel,
        workspace.packed_panel,
        workspace.meta_buffer,
    )

    ref = x @ w_ref.t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)
```

```python
def test_sm70_fp8_runtime_gemm_fake_uses_prepared_meta_output_dim():
    import vllm._custom_ops as ops_module

    input = torch.empty((4, 256), device="meta", dtype=torch.float16)
    prepared_weight = torch.empty((384, 256), device="meta", dtype=torch.float8_e4m3fn)
    prepared_scale = torch.empty((3, 2), device="meta", dtype=torch.float32)
    prepared_meta = torch.tensor(
        [320, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64
    )
    decoded = torch.empty((128, 256), device="meta", dtype=torch.float16)
    packed = torch.empty((384, 256), device="meta", dtype=torch.float16)
    meta_buffer = torch.empty((4,), device="meta", dtype=torch.int64)

    out = ops_module._sm70_fp8_runtime_gemm_fake(
        input,
        prepared_weight,
        prepared_scale,
        prepared_meta,
        decoded,
        packed,
        meta_buffer,
    )

    assert tuple(out.shape) == (4, 320)
```

- [ ] **Step 2: Run the kernel tests and confirm the current branch fails on the missing op surface**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -v -s'
```

Expected:

- `sm70_fp8_prepare` is missing
- the fake op signature still matches the old raw-weight API

- [ ] **Step 3: Add the new wrappers, fake ops, and CUDA implementation for block-wise prepared runtime decode**

In `vllm/_custom_ops.py`, replace the old Python decode helpers with:

```python
def sm70_fp8_prepare(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    layout_kind: int,
    scale_axis: int,
    block_n: int,
    block_k: int,
    panel_n: int,
) -> list[torch.Tensor]:
    return torch.ops._C.sm70_fp8_prepare(
        weight, weight_scale, layout_kind, scale_axis, block_n, block_k, panel_n
    )
```

```python
def sm70_fp8_runtime_gemm_out(
    out: torch.Tensor,
    input: torch.Tensor,
    prepared_weight: torch.Tensor,
    prepared_scale: torch.Tensor,
    prepared_meta: torch.Tensor,
    decoded_panel: torch.Tensor,
    packed_panel: torch.Tensor,
    meta_buffer: torch.Tensor,
) -> None:
    torch.ops._C.sm70_fp8_runtime_gemm_out(
        out,
        input,
        prepared_weight,
        prepared_scale,
        prepared_meta,
        decoded_panel,
        packed_panel,
        meta_buffer,
    )
```

```python
def sm70_fp8_runtime_gemm(
    input: torch.Tensor,
    prepared_weight: torch.Tensor,
    prepared_scale: torch.Tensor,
    prepared_meta: torch.Tensor,
    decoded_panel: torch.Tensor,
    packed_panel: torch.Tensor,
    meta_buffer: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.sm70_fp8_runtime_gemm(
        input,
        prepared_weight,
        prepared_scale,
        prepared_meta,
        decoded_panel,
        packed_panel,
        meta_buffer,
    )
```

```python
@register_fake("_C::sm70_fp8_runtime_gemm")
def _sm70_fp8_runtime_gemm_fake(
    input: torch.Tensor,
    prepared_weight: torch.Tensor,
    prepared_scale: torch.Tensor,
    prepared_meta: torch.Tensor,
    decoded_panel: torch.Tensor,
    packed_panel: torch.Tensor,
    meta_buffer: torch.Tensor,
) -> torch.Tensor:
    out_dim = int(prepared_meta[0].item())
    return torch.empty((input.size(0), out_dim), dtype=input.dtype, device=input.device)
```

In `csrc/ops.h`, declare:

```cpp
std::vector<torch::Tensor> sm70_fp8_prepare(torch::Tensor weight,
                                            torch::Tensor weight_scale,
                                            int64_t layout_kind,
                                            int64_t scale_axis,
                                            int64_t block_n,
                                            int64_t block_k,
                                            int64_t panel_n);

torch::Tensor sm70_fp8_runtime_gemm(torch::Tensor input,
                                    torch::Tensor prepared_weight,
                                    torch::Tensor prepared_scale,
                                    torch::Tensor prepared_meta,
                                    torch::Tensor decoded_panel,
                                    torch::Tensor packed_panel,
                                    torch::Tensor meta_buffer);

void sm70_fp8_runtime_gemm_out(torch::Tensor out,
                               torch::Tensor input,
                               torch::Tensor prepared_weight,
                               torch::Tensor prepared_scale,
                               torch::Tensor prepared_meta,
                               torch::Tensor decoded_panel,
                               torch::Tensor packed_panel,
                               torch::Tensor meta_buffer);
```

In `csrc/torch_bindings.cpp`, register:

```cpp
ops.def(
    "sm70_fp8_prepare(Tensor weight, Tensor weight_scale, int layout_kind, "
    "int scale_axis, int block_n, int block_k, int panel_n) -> Tensor[]");
ops.impl("sm70_fp8_prepare", torch::kCUDA, &sm70_fp8_prepare);

ops.def(
    "sm70_fp8_runtime_gemm(Tensor input, Tensor prepared_weight, "
    "Tensor prepared_scale, Tensor prepared_meta, Tensor decoded_panel, "
    "Tensor packed_panel, Tensor meta_buffer) -> Tensor");
ops.impl("sm70_fp8_runtime_gemm", torch::kCUDA, &sm70_fp8_runtime_gemm);

ops.def(
    "sm70_fp8_runtime_gemm_out(Tensor(a!) out, Tensor input, "
    "Tensor prepared_weight, Tensor prepared_scale, Tensor prepared_meta, "
    "Tensor(b!) decoded_panel, Tensor(c!) packed_panel, "
    "Tensor(d!) meta_buffer) -> ()");
ops.impl("sm70_fp8_runtime_gemm_out", torch::kCUDA, &sm70_fp8_runtime_gemm_out);
```

In `csrc/quantization/awq/awq_sm70_gemm.cu`, add a block-layout implementation that prepares resident tensors and uses caller-owned workspace:

```cpp
std::vector<torch::Tensor> sm70_fp8_prepare(torch::Tensor weight,
                                            torch::Tensor weight_scale,
                                            int64_t layout_kind,
                                            int64_t scale_axis,
                                            int64_t block_n,
                                            int64_t block_k,
                                            int64_t panel_n) {
  TORCH_CHECK(layout_kind == 2, "sm70_fp8_prepare: block layout is required here.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn,
              "sm70_fp8_prepare: weight must be float8_e4m3fn.");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32,
              "sm70_fp8_prepare: weight_scale must be float32.");
  auto prepared_meta = torch::tensor(
      {weight.size(0), weight.size(1), weight.size(0), weight.size(1),
       panel_n, layout_kind, scale_axis, block_n, block_k},
      torch::TensorOptions().dtype(torch::kInt64));
  auto workspace_meta = torch::tensor(
      {panel_n, weight.size(1), panel_n, weight.size(1), 4},
      torch::TensorOptions().dtype(torch::kInt64));
  return {weight.contiguous(), weight_scale.contiguous(), prepared_meta, workspace_meta};
}
```

```cpp
void decode_block_fp8_panel_to_fp16(torch::Tensor decoded_panel,
                                    torch::Tensor weight_panel,
                                    torch::Tensor scale_panel,
                                    int64_t block_n,
                                    int64_t block_k,
                                    int64_t panel_cols,
                                    int64_t k) {
  auto decoded_view = decoded_panel.narrow(0, 0, panel_cols).narrow(1, 0, k);
  for (int64_t row = 0; row < panel_cols; ++row) {
    const int64_t scale_row = row / block_n;
    auto scale_row_view = scale_panel[scale_row];
    for (int64_t col_block = 0; col_block < scale_row_view.numel(); ++col_block) {
      const int64_t col0 = col_block * block_k;
      const int64_t col1 = std::min(col0 + block_k, k);
      decoded_view[row].slice(0, col0, col1).copy_(
          weight_panel[row].slice(0, col0, col1).to(torch::kFloat32) *
          scale_row_view[col_block].item<float>());
    }
  }
}
```

```cpp
torch::Tensor sm70_fp8_runtime_gemm(torch::Tensor input,
                                    torch::Tensor prepared_weight,
                                    torch::Tensor prepared_scale,
                                    torch::Tensor prepared_meta,
                                    torch::Tensor decoded_panel,
                                    torch::Tensor packed_panel,
                                    torch::Tensor meta_buffer) {
  auto out = torch::empty(
      {input.size(0), prepared_meta[0].item<int64_t>()},
      torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  sm70_fp8_runtime_gemm_out(out,
                            input,
                            prepared_weight,
                            prepared_scale,
                            prepared_meta,
                            decoded_panel,
                            packed_panel,
                            meta_buffer);
  return out;
}
```

```cpp
void sm70_fp8_runtime_gemm_out(torch::Tensor out,
                               torch::Tensor input,
                               torch::Tensor prepared_weight,
                               torch::Tensor prepared_scale,
                               torch::Tensor prepared_meta,
                               torch::Tensor decoded_panel,
                               torch::Tensor packed_panel,
                               torch::Tensor meta_buffer) {
  const int64_t logical_n = prepared_meta[0].item<int64_t>();
  const int64_t k = prepared_meta[1].item<int64_t>();
  const int64_t panel_n = prepared_meta[4].item<int64_t>();
  const int64_t block_n = prepared_meta[7].item<int64_t>();
  const int64_t block_k = prepared_meta[8].item<int64_t>();
  TORCH_CHECK(decoded_panel.size(0) >= panel_n && decoded_panel.size(1) >= k,
              "sm70_fp8_runtime_gemm: decoded workspace too small.");
  for (int64_t n0 = 0; n0 < logical_n; n0 += panel_n) {
    const int64_t panel_cols = std::min(panel_n, logical_n - n0);
    auto weight_panel = prepared_weight.narrow(0, n0, panel_cols);
    auto scale_panel = prepared_scale.narrow(0, n0 / block_n,
                                             (panel_cols + block_n - 1) / block_n);
    decode_block_fp8_panel_to_fp16(decoded_panel, weight_panel, scale_panel,
                                   block_n, block_k, panel_cols, k);
    auto panel_view = decoded_panel.narrow(0, 0, panel_cols);
    auto prepared = prepare_sm70_f16_weight(panel_view, at::cuda::getCurrentCUDAStream());
    sm70_f16_gemm_out(out.narrow(1, n0, panel_cols),
                      input,
                      prepared.tm_weight,
                      prepared.k_ld,
                      false);
  }
}
```

- [ ] **Step 4: Re-run the kernel tests on SM70 and confirm the block path is green**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -v -s'
```

Expected:

- both tests pass
- no runtime call goes through `_sm70_decode_fp8_weight_panel`

- [ ] **Step 5: Commit the block-layout runtime-decode op**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/_custom_ops.py \
  csrc/ops.h \
  csrc/torch_bindings.cpp \
  csrc/quantization/awq/awq_sm70_gemm.cu \
  tests/kernels/quantization/test_sm70_fp8_runtime_decode.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: add sm70 fp8 block runtime decode op"
```

## Task 3: Generalize The Foundation To Tensor/Channel/Block Layout Metadata

**Files:**
- Modify: `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py`
- Modify: `csrc/quantization/awq/awq_sm70_gemm.cu`
- Modify: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`
- Test: `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py`

- [ ] **Step 1: Add failing tests for tensor-wise, channel-wise, and workspace-capacity errors**

Extend `tests/kernels/quantization/test_sm70_fp8_runtime_decode.py` with:

```python
def _sm70_decode_reference_block(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    rows = []
    for row0 in range(0, weight_fp8.shape[0], block_n):
        row_chunks = []
        for col0 in range(0, weight_fp8.shape[1], block_k):
            block = weight_fp8[row0 : row0 + block_n, col0 : col0 + block_k].float()
            scale = weight_scale[row0 // block_n, col0 // block_k].float()
            row_chunks.append((block * scale).to(torch.float16))
        rows.append(torch.cat(row_chunks, dim=1))
    return torch.cat(rows, dim=0)
```

```python
def _to_tensor_fp8(weight_fp16: torch.Tensor):
    scale = weight_fp16.float().abs().amax().clamp(min=1e-6) / 448.0
    return weight_fp16.div(scale).to(torch.float8_e4m3fn), scale.reshape(1).cuda()


def _to_channel_fp8(weight_fp16: torch.Tensor, axis: int):
    reduce_dim = 1 if axis == 0 else 0
    scale = weight_fp16.float().abs().amax(dim=reduce_dim).values.clamp(min=1e-6) / 448.0
    if axis == 0:
        q = weight_fp16.float() / scale[:, None]
    else:
        q = weight_fp16.float() / scale[None, :]
    return q.to(torch.float8_e4m3fn).cuda(), scale.cuda()


def _dequantize_reference(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    layout_kind: int,
    scale_axis: int,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    if layout_kind == 0:
        return (weight_fp8.float() * float(weight_scale.item())).to(torch.float16)
    if layout_kind == 1 and scale_axis == 0:
        return (weight_fp8.float() * weight_scale[:, None].float()).to(torch.float16)
    if layout_kind == 1 and scale_axis == 1:
        return (weight_fp8.float() * weight_scale[None, :].float()).to(torch.float16)
    return _sm70_decode_reference_block(weight_fp8, weight_scale, block_n, block_k)
```

```python
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("layout_kind", "scale_axis"),
    [(0, -1), (1, 0), (1, 1), (2, -1)],
)
def test_sm70_fp8_runtime_gemm_supports_multiple_scale_layouts(
    layout_kind: int,
    scale_axis: int,
):
    _require_sm70()
    torch.manual_seed(1)
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(256, 256, device="cuda", dtype=torch.float16) / 4
    if layout_kind == 0:
        w_fp8, w_scale = _to_tensor_fp8(w_ref)
        block_n = block_k = 0
    elif layout_kind == 1:
        w_fp8, w_scale = _to_channel_fp8(w_ref, axis=scale_axis)
        block_n = block_k = 0
    else:
        w_fp8, w_scale = _to_block_fp8(w_ref)
        block_n = block_k = 128

    prepared_w, prepared_s, prepared_meta, workspace_meta = ops.sm70_fp8_prepare(
        w_fp8, w_scale, layout_kind, scale_axis, block_n, block_k, 128
    )
    workspace = alloc_sm70_fp8_workspace(
        workspace_meta, device=x.device, m_capacity=x.shape[0]
    )
    out = ops.sm70_fp8_runtime_gemm(
        x,
        prepared_w,
        prepared_s,
        prepared_meta,
        workspace.decoded_panel,
        workspace.packed_panel,
        workspace.meta_buffer,
    )
    ref = x @ _dequantize_reference(w_fp8, w_scale, layout_kind, scale_axis, block_n, block_k).t()
    torch.testing.assert_close(out, ref, atol=6e-1, rtol=8e-2)
```

```python
def test_sm70_fp8_runtime_gemm_raises_capacity_error_for_small_workspace():
    _require_sm70()
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16) / 4
    w_ref = torch.randn(256, 256, device="cuda", dtype=torch.float16) / 4
    w_fp8, w_scale = _to_block_fp8(w_ref)
    prepared_w, prepared_s, prepared_meta, _ = ops.sm70_fp8_prepare(
        w_fp8, w_scale, 2, -1, 128, 128, 128
    )
    out = torch.empty((2, 256), device="cuda", dtype=torch.float16)
    decoded = torch.empty((64, 256), device="cuda", dtype=torch.float16)
    packed = torch.empty((64, 256), device="cuda", dtype=torch.float16)
    meta_buffer = torch.empty((4,), device="cuda", dtype=torch.int64)

    with pytest.raises(RuntimeError, match="workspace.*too small"):
        ops.sm70_fp8_runtime_gemm_out(
            out,
            x,
            prepared_w,
            prepared_s,
            prepared_meta,
            decoded,
            packed,
            meta_buffer,
        )
```

- [ ] **Step 2: Run the expanded layout tests and confirm only block layout works so far**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_runtime_gemm_supports_multiple_scale_layouts \
tests/kernels/quantization/test_sm70_fp8_runtime_decode.py::test_sm70_fp8_runtime_gemm_raises_capacity_error_for_small_workspace -v -s'
```

Expected:

- tensor-wise and channel-wise layouts fail at prepare-time
- workspace error path is not deterministic yet

- [ ] **Step 3: Extend layout metadata parsing and runtime decode branches**

In `vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py`, add a layout inference helper:

```python
def infer_sm70_fp8_layout(
    weight_scale: torch.Tensor,
    weight_block_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    if weight_block_size is not None:
        return (SM70_FP8_LAYOUT_BLOCK, -1, weight_block_size[0], weight_block_size[1])
    if weight_scale.ndim == 0 or tuple(weight_scale.shape) == (1,):
        return (SM70_FP8_LAYOUT_TENSOR, -1, 0, 0)
    if weight_scale.ndim == 1:
        return (SM70_FP8_LAYOUT_CHANNEL, 0, 0, 0)
    return (SM70_FP8_LAYOUT_CHANNEL, 1, 0, 0)
```

In `csrc/quantization/awq/awq_sm70_gemm.cu`, branch on `layout_kind` in `sm70_fp8_prepare()` and `sm70_fp8_runtime_gemm_out()`:

```cpp
switch (layout_kind) {
  case 0:
    TORCH_CHECK(weight_scale.numel() == 1,
                "sm70_fp8_prepare: tensor-wise layout requires one scale.");
    break;
  case 1:
    TORCH_CHECK(scale_axis == 0 || scale_axis == 1,
                "sm70_fp8_prepare: channel-wise layout requires scale_axis 0 or 1.");
    break;
  case 2:
    TORCH_CHECK(block_n > 0 && block_k > 0,
                "sm70_fp8_prepare: block-wise layout requires positive block sizes.");
    break;
  default:
    TORCH_CHECK(false, "sm70_fp8_prepare: unsupported layout kind.");
}
```

```cpp
if (layout_kind == 0) {
  decode_tensorwise_fp8_panel_to_fp16(decoded_panel, weight_panel, prepared_scale, panel_cols, k);
} else if (layout_kind == 1) {
  decode_channelwise_fp8_panel_to_fp16(decoded_panel, weight_panel, prepared_scale,
                                       scale_axis, n0, panel_cols, k);
} else {
  decode_block_fp8_panel_to_fp16(decoded_panel, weight_panel, scale_panel,
                                 block_n, block_k, panel_cols, k);
}
```

Also validate workspace sizes before the loop:

```cpp
TORCH_CHECK(decoded_panel.size(0) >= panel_n && decoded_panel.size(1) >= k,
            "sm70_fp8_runtime_gemm: decoded workspace too small.");
TORCH_CHECK(packed_panel.size(0) >= panel_n && packed_panel.size(1) >= k,
            "sm70_fp8_runtime_gemm: packed workspace too small.");
TORCH_CHECK(meta_buffer.numel() >= 4,
            "sm70_fp8_runtime_gemm: meta workspace too small.");
```

Add the missing helpers above `sm70_fp8_runtime_gemm_out()`:

```cpp
void decode_tensorwise_fp8_panel_to_fp16(torch::Tensor decoded_panel,
                                         torch::Tensor weight_panel,
                                         torch::Tensor prepared_scale,
                                         int64_t panel_cols,
                                         int64_t k) {
  decoded_panel.narrow(0, 0, panel_cols)
      .narrow(1, 0, k)
      .copy_(weight_panel.to(torch::kFloat32) * prepared_scale[0].item<float>());
}

void decode_channelwise_fp8_panel_to_fp16(torch::Tensor decoded_panel,
                                          torch::Tensor weight_panel,
                                          torch::Tensor prepared_scale,
                                          int64_t scale_axis,
                                          int64_t n0,
                                          int64_t panel_cols,
                                          int64_t k) {
  auto decoded_view = decoded_panel.narrow(0, 0, panel_cols).narrow(1, 0, k);
  if (scale_axis == 0) {
    auto panel_scale = prepared_scale.narrow(0, n0, panel_cols).view({panel_cols, 1});
    decoded_view.copy_(weight_panel.to(torch::kFloat32) * panel_scale);
    return;
  }
  auto panel_scale = prepared_scale.view({1, k});
  decoded_view.copy_(weight_panel.to(torch::kFloat32) * panel_scale);
}
```

- [ ] **Step 4: Run the full kernel suite and confirm all layouts and capacity checks pass**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pytest tests/kernels/quantization/test_sm70_fp8_runtime_decode.py -v -s'
```

Expected:

- tensor-wise, channel-wise, and block-wise tests pass
- the small-workspace test fails with the explicit `workspace ... too small` message

- [ ] **Step 5: Commit the generic layout support**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/layers/quantization/utils/sm70_fp8_runtime_decode.py \
  csrc/quantization/awq/awq_sm70_gemm.cu \
  tests/kernels/quantization/test_sm70_fp8_runtime_decode.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: generalize sm70 fp8 runtime decode layouts"
```

## Task 4: Wire Warmup And Dense/Fused Layer Regressions Onto The Foundation

**Files:**
- Modify: `vllm/model_executor/layers/quantization/fp8.py`
- Modify: `vllm/model_executor/warmup/awq_sm70_warmup.py`
- Modify: `tests/quantization/test_fp8_sm70.py`
- Modify: `tests/model_executor/test_sm70_runtime_warmup.py`
- Test: `tests/quantization/test_fp8_sm70.py`
- Test: `tests/model_executor/test_sm70_runtime_warmup.py`

- [ ] **Step 1: Add red tests for merged dense metadata and warmup’s new op signature**

Add to `tests/quantization/test_fp8_sm70.py`:

```python
def test_fp8_sm70_merged_linear_process_records_logical_widths(
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
    ops_module = importlib.import_module("vllm._custom_ops")

    layer = MergedColumnParallelLinear(
        input_size=256,
        output_sizes=[128, 128, 64],
        bias=False,
        params_dtype=torch.float16,
        quant_config=_make_fp8_config(),
        prefix="model.layers.0.self_attn.qkv_proj",
        disable_tp=True,
    )
    _populate_fp8_block_weights(layer)

    monkeypatch.setattr(
        ops_module,
        "sm70_fp8_prepare",
        lambda weight, scale, *_args: [
            weight,
            scale,
            torch.tensor([384, 256, 384, 256, 128, 2, -1, 128, 128], dtype=torch.int64),
            torch.tensor([128, 256, 384, 256, 4], dtype=torch.int64),
        ],
        raising=False,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer._sm70_fp8_output_size == 320
    assert layer._sm70_fp8_logical_widths == (128, 128, 64)
```

Update `tests/model_executor/test_sm70_runtime_warmup.py` so the monkeypatched runtime op matches the new signature:

```python
monkeypatch.setattr(
    "vllm.model_executor.warmup.awq_sm70_warmup.ops.sm70_fp8_runtime_gemm_out",
    lambda out, x, pw, ps, pm, decoded, packed, meta: calls.append(
        (tuple(out.shape), tuple(decoded.shape), tuple(packed.shape))
    ),
    raising=False,
)
```

- [ ] **Step 2: Run the dense/warmup regressions and confirm warmup still calls the old path**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pytest tests/quantization/test_fp8_sm70.py \
tests/model_executor/test_sm70_runtime_warmup.py -v -s'
```

Expected:

- merged-layer prepared attrs are incomplete
- warmup still calls the old raw weight/scale runtime op signature

- [ ] **Step 3: Update warmup and the dense adopter to use prepared tensors everywhere**

In `vllm/model_executor/warmup/awq_sm70_warmup.py`, import the workspace helper:

```python
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode import (
    get_or_create_sm70_fp8_workspace,
)
```

Update `_iter_unique_runtime_decode_dense_layers()` to key on prepared tensors:

```python
key = (
    int(layer._sm70_fp8_prepared_meta[1].item()),
    int(layer._sm70_fp8_output_size),
    tuple(int(v) for v in layer._sm70_fp8_prepared_meta.tolist()[5:9]),
)
```

Update `_warmup_runtime_decode_dense_layers()`:

```python
workspace = get_or_create_sm70_fp8_workspace(layer, x)
ops.sm70_fp8_runtime_gemm_out(
    out,
    x,
    layer._sm70_fp8_prepared_weight,
    layer._sm70_fp8_prepared_scale,
    layer._sm70_fp8_prepared_meta,
    workspace.decoded_panel,
    workspace.packed_panel,
    workspace.meta_buffer,
)
```

In `vllm/model_executor/layers/quantization/fp8.py`, keep the dense-only selection narrow but derive layout from the scale metadata before calling `sm70_fp8_prepare()`:

```python
from vllm.model_executor.layers.quantization.utils.sm70_fp8_runtime_decode import (
    infer_sm70_fp8_layout,
)
```

```python
layout_kind, scale_axis, block_n, block_k = infer_sm70_fp8_layout(
    layer.weight_scale_inv,
    tuple(self.weight_block_size) if self.weight_block_size is not None else None,
)
prepared_weight, prepared_scale, prepared_meta, workspace_meta = ops.sm70_fp8_prepare(
    layer.weight,
    layer.weight_scale_inv,
    layout_kind,
    scale_axis,
    block_n,
    block_k,
    self.panel_n,
)
```

- [ ] **Step 4: Re-run the dense SM70 regression suite and confirm merged/fused paths stay on one foundation**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
pytest tests/quantization/test_fp8_sm70.py \
tests/model_executor/test_sm70_runtime_warmup.py -v -s'
```

Expected:

- the whole SM70 FP8 quantization test file passes, including the existing `modules_to_not_convert`, `MoE`, and `Qwen3.5` tuple-shard regressions
- warmup test passes with prepared tensors and workspace-backed runtime decode

- [ ] **Step 5: Commit the dense/warmup integration**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  vllm/model_executor/layers/quantization/fp8.py \
  vllm/model_executor/warmup/awq_sm70_warmup.py \
  tests/quantization/test_fp8_sm70.py \
  tests/model_executor/test_sm70_runtime_warmup.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "feat: wire sm70 fp8 foundation into dense warmup"
```

## Task 5: Add Exact-Retrieval Benchmarking And Run `1k/32k` Validation

**Files:**
- Create: `benchmarks/benchmark_sm70_fp8_exact_retrieval.py`
- Create: `tests/benchmarks/test_sm70_fp8_exact_retrieval.py`
- Test: `tests/benchmarks/test_sm70_fp8_exact_retrieval.py`

- [ ] **Step 1: Add failing unit tests for the benchmark output contract**

Create `tests/benchmarks/test_sm70_fp8_exact_retrieval.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from benchmarks.benchmark_sm70_fp8_exact_retrieval import (
    build_result_row,
    semantic_quality_conclusion,
)


def test_semantic_quality_requires_exact_match():
    assert semantic_quality_conclusion("4317", "4317") == "pass"
    assert semantic_quality_conclusion(" 4317 ", "4317") == "pass"
    assert semantic_quality_conclusion("4318", "4317") == "fail"


def test_build_result_row_reports_required_fields():
    metrics = SimpleNamespace(
        first_token_latency=0.5,
        first_token_ts=10.0,
        last_token_ts=12.0,
    )
    row = build_result_row(
        input_len=1024,
        prompt_tokens=1024,
        generated_tokens=5,
        finish_reason="length",
        output_text="4317",
        expected_answer="4317",
        metrics=metrics,
    )

    assert row["input_len"] == 1024
    assert set(row) >= {
        "prefill tokens/s",
        "decode tokens/s",
        "TTFT",
        "finish_reason",
        "semantic_quality",
    }
    assert row["semantic_quality"] == "pass"
```

- [ ] **Step 2: Run the benchmark unit tests and confirm the script does not exist yet**

Run:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/benchmarks/test_sm70_fp8_exact_retrieval.py -v'
```

Expected:

- import fails because `benchmarks/benchmark_sm70_fp8_exact_retrieval.py` does not exist yet

- [ ] **Step 3: Implement the exact-retrieval benchmark harness**

Create `benchmarks/benchmark_sm70_fp8_exact_retrieval.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from typing import Any

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def semantic_quality_conclusion(output_text: str, expected_answer: str) -> str:
    return "pass" if output_text.strip() == expected_answer.strip() else "fail"


def build_result_row(
    *,
    input_len: int,
    prompt_tokens: int,
    generated_tokens: int,
    finish_reason: str | None,
    output_text: str,
    expected_answer: str,
    metrics: Any,
) -> dict[str, Any]:
    ttft = float(metrics.first_token_latency)
    decode_time = max(float(metrics.last_token_ts - metrics.first_token_ts), 1e-6)
    decode_tokens = max(generated_tokens - 1, 0)
    return {
        "input_len": input_len,
        "prefill tokens/s": prompt_tokens / max(ttft, 1e-6),
        "decode tokens/s": decode_tokens / decode_time if decode_tokens else 0.0,
        "TTFT": ttft * 1000.0,
        "finish_reason": finish_reason or "unknown",
        "semantic_quality": semantic_quality_conclusion(output_text, expected_answer),
        "output_text": output_text,
    }
```

```python
def build_exact_retrieval_prompt(tokenizer, target_len: int, expected_answer: str) -> str:
    prefix = "You are doing exact retrieval. Return only the code.\n"
    needle = f"The verification code is {expected_answer}.\n"
    suffix = "Question: what is the verification code?\nAnswer:"
    filler = " filler"
    prompt = prefix + needle + suffix
    while len(tokenizer.encode(prompt)) < target_len:
        prompt = prefix + filler + prompt
    tokens = tokenizer.encode(prompt)[:target_len]
    return tokenizer.decode(tokens)
```

```python
def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_tokens)
    for input_len in args.input_lens:
        prompt = build_exact_retrieval_prompt(tokenizer, input_len, args.expected_answer)
        output = llm.generate([prompt], sampling_params=sampling_params)[0]
        row = build_result_row(
            input_len=input_len,
            prompt_tokens=len(output.prompt_token_ids or []),
            generated_tokens=len(output.outputs[0].token_ids),
            finish_reason=output.outputs[0].finish_reason,
            output_text=output.outputs[0].text,
            expected_answer=args.expected_answer,
            metrics=output.metrics,
        )
        print(json.dumps(row, ensure_ascii=False))
```

```python
def create_argument_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="Benchmark SM70 FP8 exact retrieval quality and latency."
    )
    parser.add_argument("--input-lens", type=int, nargs="+", required=True)
    parser.add_argument("--expected-answer", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    return EngineArgs.add_cli_args(parser)


if __name__ == "__main__":
    parser = create_argument_parser()
    main(parser.parse_args())
```

- [ ] **Step 4: Run the benchmark unit tests, then execute the real `1k/32k` validation on SM70**

Run the unit tests:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
pytest tests/benchmarks/test_sm70_fp8_exact_retrieval.py -v'
```

Then run the real benchmark:

```bash
bash -lc 'source /home/z/anaconda3/etc/profile.d/conda.sh && \
conda activate gptq && \
cd /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python benchmarks/benchmark_sm70_fp8_exact_retrieval.py \
  --model /mnt/data6/models/Qwen3.5-0.8B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --input-lens 1024 32768 \
  --expected-answer 4317 \
  --max-tokens 8'
```

Expected:

- the test file passes
- the benchmark prints two JSON rows
- each JSON row contains `prefill tokens/s`, `decode tokens/s`, `TTFT`, `finish_reason`, and `semantic_quality`
- `semantic_quality` is `pass` for both `1024` and `32768`

- [ ] **Step 5: Commit the benchmark harness**

```bash
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split add \
  benchmarks/benchmark_sm70_fp8_exact_retrieval.py \
  tests/benchmarks/test_sm70_fp8_exact_retrieval.py
git -C /mnt/data/apps/1Cat-vLLM/.worktrees/vllm-0190-upstream-split commit -m \
  "bench: add sm70 fp8 exact retrieval validation"
```
