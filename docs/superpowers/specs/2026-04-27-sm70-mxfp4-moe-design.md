# SM70 MXFP4 MoE Design

## Goal

Enable `mxfp4` GPT-OSS MoE inference on SM70/V100 by adding a TurboMind-backed MXFP4 MoE path that mirrors the existing SM70 AWQ and FP8 direct MoE integrations.

## Scope

The first implementation targets routed GPT-OSS MXFP4 MoE weights with BF16/FP16 activations converted to the existing SM70 half GEMM path. It does not add MXFP4 linear or attention kernels, and it does not add MXFP4 activation quantization.

## Architecture

Add a new MXFP4 backend enum value, `SM70_TURBOMIND`, selected only on CUDA SM70 when the compiled C++ ops are present. The backend uses the existing modular MXFP4 loading flow but returns a direct `FusedMoEMethodBase` implementation for SM70, following `Fp8SM70DirectMoEMethod`.

Weights load in the normal GPT-OSS layout:

- `w13_weight`: `[experts, 2 * intermediate, hidden / 2]`, two E2M1 values per byte.
- `w13_weight_scale`: `[experts, 2 * intermediate, hidden / 32]`, E8M0 scale bytes.
- `w2_weight`: `[experts, hidden, intermediate / 2]`.
- `w2_weight_scale`: `[experts, hidden, intermediate / 32]`.

The C++ prepare op unpacks MXFP4 bytes into 4-bit logical values, applies the gated-SiLU row interleave for `w13`, transposes scales into `[K / 32, N]`, converts both tensors through TurboMind `GetConverters(kHalf, kFloat4_e2m1, kHalf, grouped=true, sm=70)`, and returns prepared tensors plus `{logical_n, logical_k, 32, k_ld, q_ld}` metadata.

Runtime uses one batched TurboMind GEMM for `w13` with `Epilogue::kGatedSilu` and one for `w2` without fused activation. Token routing, buffers, strided pointer construction, CUDA graph friendliness, and unpermute match the existing SM70 FP8 direct MoE path.

## Error Handling

The backend is selected only when the current platform is CUDA SM70 and `torch.ops._C.sm70_mxfp4_moe_direct_prepare` exists. Shape checks require hidden and intermediate dimensions to be divisible by 32. Unsupported monolithic/shared-expert paths raise clear errors through the direct method.

## Testing

Add Python unit tests that fail before implementation:

- SM70 MXFP4 backend selection prefers `SM70_TURBOMIND`.
- MXFP4 config accepts min capability 70.
- `process_weights_after_loading` calls the new prepare op, stores prepared tensors, builds strided pointers, and records SM70 metadata.
- `apply` calls the new batched GEMM op twice and follows the same routed MoE buffer path as FP8.

After implementation, run the targeted MXFP4 tests, existing SM70 FP8/AWQ tests touched by shared ops, and a build/import check. If a compiled extension is available, smoke-test `/mnt/data6/models/gpt-oss-120b` on `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5`.
