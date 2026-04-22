# SM70 FP8 Linear AWQ Repack Design

## Goal

Enable serialized block-FP8 dense linear layers to execute on `sm70` by
repacking loaded FP8 weights into the existing SM70 AWQ runtime format.

## Scope

- In scope: dense `Linear` layers backed by serialized FP8 checkpoints with
  `weight_block_size` set and dynamic activation scaling.
- Out of scope: serialized FP8 `MoE`, FP8 `KV cache`, and native SM70 FP8 GEMM.

## Constraints

- Keep checkpoint FP8 as the storage format presented to the loader.
- Reuse existing SM70 CUDA kernels instead of introducing a new FP8 kernel
  stack.
- Allow lossy repacking into an SM70-friendly compressed runtime layout.

## Design

`Fp8Config.get_quant_method()` gains an SM70-only dense fallback that selects a
new `Fp8SM70LinearMethod` when the checkpoint is serialized FP8 block-quant and
the current capability is `70`.

`Fp8SM70LinearMethod` mirrors FP8 serialized weight registration so existing
weight loading still works. After loading, it dequantizes the block-FP8 weights
to `fp16`, requantizes them into symmetric AWQ-style int4 groups using the FP8
block `K` size as `group_size`, calls `ops.awq_sm70_prepare()`, stores the
prepared TurboMind tensors on the layer, and frees the original FP8 tensors.

At runtime, `apply()` calls `ops.awq_gemm_sm70()` directly with the prepared
SM70 tensors. Serialized FP8 `MoE` on `sm70` is rejected with a clear error so
the new path stays limited to dense linear layers.

## Validation

- Unit tests cover method selection, AWQ repack/prepare, AWQ GEMM dispatch, and
  the explicit serialized FP8 `MoE` rejection on `sm70`.
- The global quantization capability gate only allows this narrow serialized
  FP8 SM70 fallback instead of lowering FP8 support for all paths.
