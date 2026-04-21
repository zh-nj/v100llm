# SM70 FP8 Runtime-Decode GEMM Foundation Design

## 1. Summary

This spec defines a reusable `SM70 FP8 runtime-decode GEMM` foundation for
vLLM. The foundation targets `SM70` GPUs, keeps FP8 weights resident in GPU
memory, decodes only the currently needed panel into a fixed workspace, and
reuses the existing `SM70` TurboMind-backed `fp16` GEMM path for execution.

This is a dense-first sub-project. The implementation scope for this spec is
`dense Linear` only, but the data structures, custom op boundaries, and
workspace protocol must be designed so they can later support:

- all dense FP8 linear variants across models
- multiple FP8 scale layouts
- serialized and runtime-quantized FP8 dense weights
- future `MoE`
- future `KV cache / attention`

This spec intentionally does not implement `MoE` or `KV cache / attention`.
Instead, it reserves interface and metadata constraints so those systems can be
added without redesigning the foundation.

## 2. Current Problem

The current `SM70` FP8 runtime-decode path is implemented in Python/Torch:

1. slice a panel from resident FP8 weight
2. expand scales with `repeat_interleave`
3. cast FP8 weight to `float32`
4. multiply by expanded scales
5. cast to `float16`
6. call `sm70_f16_prepare`
7. call `sm70_f16_gemm_out`

This approach violates the intended "FP8 resident + temporary decode only"
model in two important ways:

- it creates multiple transient tensors per panel in Python/Torch
- those allocations are implicit and not controlled by a reusable workspace

The result is an unstable runtime memory profile. During long-context or
multi-step generation, the model can reach a state where only a few MiB remain
free, and a small panel decode allocation fails with CUDA OOM even though the
weight itself remains compressed.

This behavior is fundamentally different from the existing `AWQ SM70` path,
where:

- prepare happens once into an `SM70`-friendly representation
- runtime consumes that prepared representation
- GEMM uses stable C++/CUDA-side workspace management

The current Python decode path is good enough as a proof of concept, but it is
not an acceptable long-term foundation for "all FP8 models on SM70".

## 3. Goals

### 3.1 Required goals

- Provide a reusable `SM70 FP8 runtime-decode GEMM` foundation.
- Keep FP8 weights resident in GPU memory in compressed form.
- Eliminate Python/Torch intermediate decode allocations from the runtime hot
  path.
- Reuse the existing `SM70` TurboMind-backed `fp16` GEMM execution path.
- Use an explicit, reusable workspace contract similar in spirit to the
  existing `AWQ SM70` flow.
- Support dense Linear as the first adopter.
- Design metadata and APIs so future dense FP8 models can plug in without
  redesigning the foundation.
- Reserve compatibility constraints for future `MoE` and `KV cache/attention`.

### 3.2 Format goals

The foundation must be designed to support the following FP8 dense weight
families:

- serialized FP8 dense weights
- runtime-quantized FP8 dense weights
- tensor-wise scale layouts
- channel-wise scale layouts
- block-wise scale layouts

The first adopter may still migrate in phases, but the public foundation
interface must be defined around this broader set.

### 3.3 Benchmark and semantic goals

The final dense adopter must support benchmark validation that reports:

- `prefill tokens/s`
- `decode tokens/s`
- `TTFT`
- `finish_reason`
- semantic quality conclusion

The benchmark prompts must use exact retrieval style validation, where semantic
success requires exact match of the expected short answer.

## 4. Non-Goals

The following are out of scope for this spec's implementation phase:

- native `SM70` FP8 fused GEMM that bypasses the existing `fp16` GEMM path
- full `MoE` execution support
- `KV cache` FP8 storage and attention kernel changes
- changing global platform capability checks to pretend `SM70` fully supports
  every FP8 subsystem
- preserving the current Python decode path as a production fallback

## 5. Chosen Approach

The selected approach is:

`New reusable SM70 FP8 runtime-decode custom CUDA op + explicit workspace protocol`

This approach is preferred over a pure Python workspace refactor because:

- it removes panel decode and scale expansion from Python/Torch runtime
- it gives explicit control over temporary memory
- it aligns with the existing `AWQ SM70` mental model
- it creates a reusable foundation for future dense FP8 adopters
- it leaves `sm70_f16_gemm_out` in place, reducing risk versus writing a new
  fused GEMM stack

## 6. Architecture Overview

The foundation is split into three layers:

1. `prepare`
2. `workspace`
3. `execute`

### 6.1 Prepare

Prepare runs once after weight loading. Its job is to transform a model-facing
FP8 weight and associated quantization metadata into an `SM70`-friendly,
runtime-ready prepared representation.

Prepare does **not** produce a persistent full `fp16` copy of the weight.

Prepare does produce:

- resident compressed weight storage in an `SM70`-friendly layout
- resident scale storage in an `SM70`-friendly layout
- shape, stride, tile, and decode metadata
- layout identifiers required by the runtime op

### 6.2 Workspace

Workspace is explicit, reusable, and caller-managed. It is analogous to the
existing `AWQ SM70` runtime ownership model:

- the caller owns the tensors
- the custom op only reads/writes within the provided capacity
- the custom op must not allocate panel-sized temporary tensors internally

### 6.3 Execute

Execute performs runtime panel decode and computation:

1. read one prepared FP8 panel
2. decode into a fixed `fp16` panel buffer
3. convert/pack that panel into a TurboMind-compatible panel buffer
4. call existing `sm70_f16_gemm_out`

The execution path must not allocate temporary decode tensors whose size scales
with request length or decode step count.

## 7. Prepared Representation

The prepared representation is the core persistent output of `prepare`.

It must contain:

- `prepared_weight`
  - compressed FP8 weight in an `SM70`-friendly runtime layout
- `prepared_scale`
  - scale storage aligned with runtime panel decode
- `prepared_meta`
  - small metadata tensor or metadata struct encoded as tensor fields
- `logical_shape`
  - original `[N, K]`
- `storage_shape`
  - prepared storage shape
- `scale_shape`
  - prepared scale shape
- `panel_n`
  - runtime panel width
- `block_shape`
  - optional; only meaningful for block layouts
- `layout_kind`
  - encoded scale/weight layout identifier

It must not contain:

- a full persistent `fp16` decoded weight
- a full persistent TurboMind-packed `fp16` weight

### 7.1 Dense-first invariant

The first adopter is dense Linear, so the initial prepared representation only
needs to describe dense `[N, K]` matrices.

However, the representation must not bake in assumptions that would make future
expert or attention storage impossible. In particular:

- metadata should identify logical matrix shape cleanly
- panel decode should not assume there is only one permanent matrix owner
- scale layout should remain separable from model-specific module names

## 8. Generic FP8 Layout Abstraction

The runtime foundation must not branch on model classes. It must branch only on
prepared FP8 layout metadata.

To do that, the foundation defines a logical FP8 layout abstraction with at
least the following fields:

- `quant_kind`
  - `tensor`
  - `channel`
  - `block`
- `weight_dtype`
  - FP8 dtype identifier
- `logical_shape`
  - original matrix shape
- `storage_shape`
  - prepared storage shape
- `scale_shape`
  - prepared scale storage shape
- `block_shape`
  - optional
- `scale_broadcast_rule`
  - how scales map onto a decoded panel

### 8.1 Runtime meaning

At runtime, the decode op only needs to know:

- where the panel weight bytes are
- where the relevant scale data is
- how to map scales onto the current panel

That gives one runtime foundation with small layout-specific branches instead of
separate pipelines per model or per dense layer type.

### 8.2 Layout behavior

- `tensor-wise`
  - one scale or small scalar set applies to the entire panel
- `channel-wise`
  - scale broadcasts along one logical matrix dimension
- `block-wise`
  - scale is indexed in block space and expanded only within the current panel

The foundation must handle these via layout metadata, not by hardcoding model
names or Qwen-specific branches.

## 9. Workspace Protocol

The workspace protocol is explicit and reusable.

The caller-managed workspace must include enough capacity for the maximum panel
shape used by a given prepared weight and input batch shape.

At minimum it must provide:

- `decoded_fp16_panel`
  - fixed `fp16` buffer for one decoded panel
- `tm_packed_panel`
  - fixed buffer holding TurboMind-compatible packed data for one panel
- `meta_buffer`
  - small integer metadata buffer for ld/stride or similar

It may also include:

- temporary scale expansion scratch, if required by a layout implementation
- additional metadata fields for future multi-panel or expert-local use

### 9.1 Ownership

- Python or the model runner creates and retains workspace tensors.
- The custom op receives those tensors explicitly.
- The custom op writes in place.
- The custom op must not allocate hidden panel-sized temporaries.

### 9.2 Capacity errors

If the provided workspace cannot hold the requested panel or runtime batch
shape, the op must fail with a deterministic validation error, not with an
accidental CUDA allocation failure.

## 10. Custom Op Surface

The foundation adds two new categories of custom ops:

- `sm70_fp8_prepare(...)`
- `sm70_fp8_runtime_gemm_out(...)`

### 10.1 sm70_fp8_prepare

Responsibilities:

- validate FP8 input shape and quantization metadata
- convert raw model-facing FP8 storage into prepared resident layout
- convert raw scale storage into prepared resident layout
- return prepared tensors plus compact metadata

This is conceptually similar to `awq_sm70_prepare`, but for generic FP8 decode
and without producing a persistent full decoded weight.

### 10.2 sm70_fp8_runtime_gemm_out

Responsibilities:

- validate input tensor, prepared tensors, metadata, and workspace capacity
- decode one panel at a time into the provided workspace
- pack that panel into TurboMind-compatible form
- invoke existing `sm70_f16_gemm_out`
- write directly into the caller-provided output tensor

This op is the runtime execution heart of the foundation.

### 10.3 No hidden allocation rule

The runtime op must not allocate panel-proportional temporary tensors during the
hot path. If a tiny unavoidable control tensor is still required internally, it
must be constant-sized and independent of `[M, N, K]`.

## 11. Python Integration Boundary

Python remains the orchestration layer only.

### 11.1 Python responsibilities

- quantization method selection
- deciding which layers opt into the foundation
- invoking `prepare`
- attaching prepared tensors and metadata to the layer
- creating and reusing workspace objects
- invoking `sm70_fp8_runtime_gemm_out`
- handling output slicing to logical width
- bias application

### 11.2 Python non-responsibilities

Python must no longer do:

- scale expansion for runtime decode
- panel dequantization in Torch
- panel packing in Torch
- repeated `sm70_f16_prepare` on newly allocated panel tensors

## 12. File Boundary Plan

### 12.1 Python files

- `vllm/model_executor/layers/quantization/fp8.py`
  - route dense FP8 adopters onto the new foundation
  - attach prepared representation and workspace contract
- `vllm/_custom_ops.py`
  - add wrappers and fake registrations for the new ops
- `vllm/model_executor/warmup/awq_sm70_warmup.py`
  - warm up the new foundation rather than the Python fallback

### 12.2 C++ / CUDA files

- `csrc/ops.h`
  - declare new FP8 prepare/runtime op signatures
- `csrc/torch_bindings.cpp`
  - register the new ops
- one of:
  - extend `csrc/quantization/awq/awq_sm70_gemm.cu` if the code remains
    reasonably cohesive
  - or split into a dedicated `sm70_fp8_runtime_decode.cu` if the new logic
    becomes large enough to deserve an independent home

### 12.3 Placement rule

The final location should be chosen based on code size and coherence, not on
avoiding a new file at all costs.

## 13. Dense Linear Adopter Scope

This sub-project implements the foundation for dense Linear first.

The dense adopter must cover:

- replicated dense linear
- merged dense linear
- fused dense forms such as:
  - `QKV`
  - `gate_up`
  - `out_proj`

The adopter must reuse one common runtime foundation rather than forking a
special execution path per fused dense module.

## 14. Future Compatibility Constraints

### 14.1 Future MoE compatibility

This sub-project will not implement `MoE`, but the following constraints must
hold:

- prepared representation must be sliceable per expert or expert group
- workspace protocol must be reusable across repeated expert-local calls
- layout metadata must not assume a single globally owned dense matrix

### 14.2 Future KV cache / attention compatibility

This sub-project will not implement `KV cache / attention`, but the following
constraints must hold:

- the scale/layout abstraction must remain independent from GEMM-only naming
- metadata encoding should not assume the only runtime consumer is a dense GEMM
- workspace semantics must allow future non-GEMM decode buffers to coexist

This means the dense foundation becomes the first consumer of a generic FP8
runtime-decode protocol, not the only one forever.

## 15. Error Handling

Validation errors must be surfaced as early and as deterministically as
possible.

### 15.1 Prepare-time errors

Prepare must reject:

- unsupported FP8 dtype/layout combinations
- unsupported scale metadata
- inconsistent weight/scale shapes
- unsupported tile or panel settings
- invalid block metadata

These should fail during prepare, not later in the hot path.

### 15.2 Runtime errors

Runtime must reject:

- input dtype mismatch
- output dtype mismatch
- prepared tensor/layout mismatch
- workspace capacity mismatch
- illegal panel or stride metadata

Runtime failures should identify the violated contract directly. They should not
defer to incidental Torch allocation failures if validation could have caught
the problem earlier.

## 16. Testing Strategy

### 16.1 Unit tests

Tests must cover:

- tensor-wise scale layout
- channel-wise scale layout
- block-wise scale layout
- prepared metadata correctness
- workspace reuse across repeated calls
- deterministic workspace capacity errors
- replicated linear
- merged linear
- fused dense variants

### 16.2 Kernel-level correctness tests

Tests must verify:

- new runtime op numerics against a trusted reference
- fake op shape behavior
- multi-panel correctness
- logical width slicing behavior after padded execution

### 16.3 Real-model validation

Real-model validation must include:

- `/mnt/data6/models/Qwen3.5-0.8B-FP8`
- `1k` exact retrieval benchmark
- `32k` exact retrieval benchmark

Each benchmark result must report:

- `prefill tokens/s`
- `decode tokens/s`
- `TTFT`
- `finish_reason`
- semantic quality conclusion

Semantic quality must be judged by exact match against a known short answer.

### 16.4 Regression target

The new foundation is not acceptable if it reproduces the current failure mode
where a tiny panel decode allocation inside the runtime path OOMs after the
model has otherwise loaded successfully.

## 17. Rollout Strategy

Implementation should proceed in phases:

1. introduce the new custom op surface and workspace contract
2. migrate the current `SM70` serialized block-FP8 dense adopter onto the new
   path
3. add broader dense FP8 layout coverage on top of the same foundation
4. only after the dense foundation is stable, design `MoE` and
   `KV cache/attention` adopters separately

This keeps the foundation generic without forcing all future consumers into the
same implementation cycle.

## 18. Acceptance Criteria

This sub-project is complete when:

- the dense FP8 adopter no longer performs runtime decode through Python/Torch
  intermediate panel tensors
- the runtime path uses an explicit reusable workspace
- the runtime op does not allocate panel-sized temporary tensors internally
- existing `SM70 fp16 GEMM` remains the execution backend
- dense fused/merged linear variants share the same foundation
- exact retrieval validation reports the required five benchmark fields
- runtime memory behavior is stable enough to avoid the current decode-path OOM
  failure mode on supported benchmark cases

## 19. Design Decision

Adopt a new reusable `SM70 FP8 runtime-decode` foundation implemented as:

- `prepare` into a generic prepared FP8 representation
- explicit caller-managed reusable workspace
- custom CUDA runtime decode + pack op
- reuse of existing `sm70_f16_gemm_out`
- dense Linear as the first adopter
- future `MoE` and `KV cache / attention` compatibility enforced through data
  model and interface constraints, not through same-phase implementation
