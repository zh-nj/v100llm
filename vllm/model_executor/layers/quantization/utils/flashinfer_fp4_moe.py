# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for NVFP4 + FlashInfer fused-MoE path"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_fp4_moe_weights_for_fi,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutlass_fused_moe,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
    )

logger = init_logger(__name__)


__all__ = [
    "flashinfer_trtllm_fp4_moe",
    "flashinfer_trtllm_fp4_routed_moe",
    "reorder_w1w3_to_w3w1",
]


def is_flashinfer_fp4_cutlass_moe_available() -> bool:
    """Return `True` when FlashInfer CUTLASS NV-FP4 kernels can be used."""
    return (
        envs.VLLM_USE_FLASHINFER_MOE_FP4
        and has_flashinfer_cutlass_fused_moe()
        and current_platform.is_cuda()
        and current_platform.has_device_capability(100)
    )


def reorder_w1w3_to_w3w1(
    weight: torch.Tensor, scale: torch.Tensor, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order the concatenated `[w1, w3]` tensors to `[w3, w1]`"""
    size = weight.size(dim)
    assert size % 2 == 0, f"Expected even size in dim {dim}, got {size}"
    half = size // 2

    w1, w3 = weight.split(half, dim=dim)
    s1, s3 = scale.split(half, dim=dim)

    return (
        torch.cat([w3, w1], dim=dim).contiguous(),
        torch.cat([s3, s1], dim=dim).contiguous(),
    )


def prepare_static_weights_for_trtllm_fp4_moe(
    # args_dequant,
    # args,
    gemm1_weights,
    gemm2_weights,
    gemm1_scales_linear_fp4_bytes,
    gemm2_scales_linear_fp4_bytes,
    hidden_size,
    intermediate_size,
    num_experts,
    is_gated_activation: bool,
):
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    """Prepare quantized weights for kernel (done offline with weights)."""
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
    gemm1_intermediate_size = (
        2 * intermediate_size if is_gated_activation else intermediate_size
    )

    # Convert quantized weights to proper formats
    gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate_size, hidden_size // 2
    )  # packed fp4
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(
        num_experts, gemm1_intermediate_size, hidden_size // 16
    )  # fp8 scaling factors

    gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )  # packed fp4
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn
    ).reshape(num_experts, hidden_size, intermediate_size // 16)  # fp8 scaling factors

    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(num_experts):
        # Calculate the permute indices for the following:
        # 1. Reorder rows of W1 and scales for fused gated activation
        # 2. Shuffle weights and scaling factors for transposed mma output
        # for both w3_w1 and w2 weights and scale factors
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=is_gated_activation,
        )
        gemm1_weights_fp4_shuffled.append(
            gemm1_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            _cache_permute_indices,
            gemm1_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=is_gated_activation,
        )
        gemm1_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm1_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

        permute_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        gemm2_weights_fp4_shuffled.append(
            gemm2_weights_fp4[i]
            .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
            .contiguous()
        )

        permute_sf_indices = get_w2_permute_indices_with_cache(
            _cache_permute_indices,
            gemm2_scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        gemm2_scales_fp4_shuffled.append(
            nvfp4_block_scale_interleave(
                gemm2_scales_linear_fp4[i]
                .view(torch.uint8)[
                    permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                ]
                .contiguous()
            )
        )

    # Stack weights for all experts
    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = (
        torch.stack(gemm1_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_intermediate_size, hidden_size // 16)
    )

    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = (
        torch.stack(gemm2_scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    )


def prepare_nvfp4_moe_layer_for_fi_or_cutlass(
    backend: "NvFp4MoeBackend",
    layer: "FusedMoE",
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor,
    is_act_and_mul: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Delayed import for circular dependency avoidance.
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        is_global_sf_supported_for_nvfp4_backend,
    )

    assert backend in [
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
    ]

    # Reorder [w1, w3] to [w3, w1] for FI NVFP4 MoE kernels.
    is_gated = layer.activation.is_gated
    if (
        is_gated
        and is_act_and_mul
        and backend
        in [
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
        ]
    ):
        w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale)

    # For some FI kernels, the input scales are shared by all experts.
    if is_global_sf_supported_for_nvfp4_backend(backend):
        num_experts = w13.shape[0]
        a13_scale = a13_scale.max().to(torch.float32).expand(num_experts)
        a2_scale = a2_scale.max().to(torch.float32).expand(num_experts)
    else:
        a13_scale = a13_scale.max(dim=1).values.to(torch.float32)

    # Shuffle weights and scales for FI TRTLLM NVFP4 MoE kernels.
    if backend == NvFp4MoeBackend.FLASHINFER_TRTLLM:
        # Align weights for FI NVFP4 MoE kernels.
        min_alignment = 16 if is_gated else 128
        w13, w13_scale, w2, w2_scale, padded_intermediate = (
            align_fp4_moe_weights_for_fi(
                w13, w13_scale, w2, w2_scale, is_act_and_mul, min_alignment
            )
        )
        layer.moe_config.intermediate_size_per_partition = padded_intermediate

        w13, w13_scale, w2, w2_scale = prepare_static_weights_for_trtllm_fp4_moe(
            w13,
            w2,
            w13_scale,
            w2_scale,
            hidden_size=w2.size(-2),
            intermediate_size=w13.size(-2) // 2 if is_gated else w13.size(-2),
            num_experts=w13.size(0),
            is_gated_activation=is_gated,
        )
    else:
        # Swizzle the block scales for other FI NVFP4 MoE kernels.
        w13_scale = swizzle_blockscale(w13_scale)

        # Apply padding if needed.
        pad_size = w13_scale.size(1) - w13.size(1)
        if pad_size > 0:
            if is_act_and_mul:
                raise NotImplementedError(
                    "Intermediate size padding for w1 and w3, for %s "
                    "NvFp4 backend, but this is not currently supported",
                    backend.value,
                )
            w13 = torch.nn.functional.pad(w13, (0, 0, 0, pad_size))
            w2 = torch.nn.functional.pad(w2, (0, pad_size // 2, 0, 0))
            w2_scale = torch.nn.functional.pad(w2_scale, (0, pad_size // 16))

        w2_scale = swizzle_blockscale(w2_scale)

    return w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale


def flashinfer_trtllm_fp4_moe(
    layer: torch.nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    router_logits: torch.Tensor,
    top_k: int,
    activation: str,
    global_num_experts: int,
    num_expert_group: int | None,
    topk_group: int | None,
    custom_routing_function: object | None,
    e_score_correction_bias: torch.Tensor | None,
) -> torch.Tensor:
    import flashinfer

    from vllm.model_executor.models.llama4 import Llama4MoE

    assert activation == "silu", (
        "Only SiLU activation is supported for FlashInfer TRTLLM FP4 MoE. "
        f"{activation} found instead."
    )

    if isinstance(x, tuple):
        hidden_states_fp4, hidden_states_scale_linear_fp4 = x
    else:
        hidden_states_fp4, hidden_states_scale_linear_fp4 = ops.scaled_fp4_quant(
            x, layer.a1_gscale, is_sf_swizzled_layout=False
        )

    use_llama4_routing = custom_routing_function is Llama4MoE.custom_routing_function
    routing_method_type = layer.routing_method_type
    if use_llama4_routing:
        routing_method_type = flashinfer.RoutingMethodType.Llama4

    routing_bias = e_score_correction_bias
    if routing_bias is not None:
        routing_bias = routing_bias.to(torch.bfloat16)

    router_logits = (
        router_logits.to(torch.float32)
        if routing_method_type == RoutingMethodType.DeepSeekV3
        else router_logits
    )

    out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale_linear_fp4.view(
            torch.float8_e4m3fn
        ).flatten(),
        gemm1_weights=layer.w13_weight.data,
        gemm1_weights_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer.w2_weight.data,
        gemm2_weights_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=layer.g1_scale_c.data,
        output1_scale_gate_scalar=layer.g1_alphas.data,
        output2_scale_scalar=layer.g2_alphas.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=num_expert_group if num_expert_group is not None else 0,
        topk_group=topk_group if topk_group is not None else 0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type,
        do_finalize=True,
    )[0]

    return out


def flashinfer_trtllm_fp4_routed_moe(
    layer: torch.nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    activation: str,
    global_num_experts: int,
) -> torch.Tensor:
    import flashinfer

    assert activation == "silu", (
        "Only SiLU activation is supported for FlashInfer TRTLLM FP4 Routed MoE. "
        f"{activation} found instead."
    )

    packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    if isinstance(x, tuple):
        hidden_states_fp4, hidden_states_scale_linear_fp4 = x
    else:
        hidden_states_fp4, hidden_states_scale_linear_fp4 = ops.scaled_fp4_quant(
            x, layer.a1_gscale, is_sf_swizzled_layout=False
        )

    out = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed_tensor,
        routing_bias=None,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale_linear_fp4.view(
            torch.float8_e4m3fn
        ).flatten(),
        gemm1_weights=layer.w13_weight.data,
        gemm1_weights_scale=layer.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=layer.w2_weight.data,
        gemm2_weights_scale=layer.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=layer.g1_scale_c.data,
        output1_scale_gate_scalar=layer.g1_alphas.data,
        output2_scale_scalar=layer.g2_alphas.data,
        num_experts=global_num_experts,
        top_k=top_k,
        n_group=0,
        topk_group=0,
        intermediate_size=layer.intermediate_size_per_partition,
        local_expert_offset=layer.ep_rank * layer.local_num_experts,
        local_num_experts=layer.local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,
        do_finalize=True,
    )[0]

    return out
