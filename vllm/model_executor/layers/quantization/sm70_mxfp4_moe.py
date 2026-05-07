# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.model_executor.layers.deepseek_v4_attention import _profile_or_null
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from vllm.model_executor.layers.fused_moe.swiglu_limit_triton import (
    sm70_fused_swiglu_limit,
)
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _DEFAULT_PERSISTENT_MAX_TOKENS,
    _moe_permute_accepts_scale_and_m_indices,
)
from vllm.model_executor.utils import set_weight_attrs


class Mxfp4SM70MoEMethod(FusedMoEMethodBase):
    """SM70 direct MXFP4 MoE method using TurboMind grouped GEMM."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.group_size = 32

    @property
    def supports_eplb(self) -> bool:
        return True

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )
        hidden_size = (
            (hidden_size + self.group_size - 1) // self.group_size
        ) * self.group_size
        intermediate_size_per_partition = (
            (intermediate_size_per_partition + self.group_size - 1)
            // self.group_size
        ) * self.group_size
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if hidden_size % self.group_size != 0:
            raise ValueError(
                "SM70 MXFP4 MoE requires hidden_size divisible by 32, "
                f"got {hidden_size}."
            )
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                "SM70 MXFP4 MoE requires intermediate_size_per_partition "
                f"divisible by 32, got {intermediate_size_per_partition}."
            )

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts

        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        w13_scale.quant_method = "block"

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        w2_scale.quant_method = "block"

        bias_dtype = getattr(layer, "orig_dtype", params_dtype)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=bias_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=bias_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _prepare_matrix(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        interleave_gated_silu: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ops.sm70_mxfp4_moe_direct_prepare(
            weight,
            scale,
            interleave_gated_silu,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        w13, w13_scale, w13_meta = self._prepare_matrix(
            layer.w13_weight,
            layer.w13_weight_scale,
            False,
        )
        w2, w2_scale, w2_meta = self._prepare_matrix(
            layer.w2_weight,
            layer.w2_weight_scale,
            False,
        )
        layer.w13_tm_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w13_tm_scales = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_tm_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w2_tm_scales = torch.nn.Parameter(w2_scale, requires_grad=False)
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale, layer.w2_weight_scale

        num_experts = int(w13.shape[0])
        w13_k_ld = int(w13_meta[3].item())
        w13_q_ld = int(w13_meta[4].item())
        w2_k_ld = int(w2_meta[3].item())
        w2_q_ld = int(w2_meta[4].item())
        w13_ptrs = ops.awq_moe_build_strided_ptrs(
            w13, w13_scale, w13_k_ld, w13_q_ld, num_experts
        )
        w2_ptrs = ops.awq_moe_build_strided_ptrs(
            w2, w2_scale, w2_k_ld, w2_q_ld, num_experts
        )
        layer.w13_strided_ptrs_w = torch.nn.Parameter(w13_ptrs[0], requires_grad=False)
        layer.w13_strided_ptrs_s = torch.nn.Parameter(w13_ptrs[1], requires_grad=False)
        layer.w2_strided_ptrs_w = torch.nn.Parameter(w2_ptrs[0], requires_grad=False)
        layer.w2_strided_ptrs_s = torch.nn.Parameter(w2_ptrs[1], requires_grad=False)
        layer.w13_strided_ptrs_w_rows = layer.w13_strided_ptrs_w.view(num_experts, -1)
        layer.w13_strided_ptrs_s_rows = layer.w13_strided_ptrs_s.view(num_experts, -1)
        layer.w2_strided_ptrs_w_rows = layer.w2_strided_ptrs_w.view(num_experts, -1)
        layer.w2_strided_ptrs_s_rows = layer.w2_strided_ptrs_s.view(num_experts, -1)

        layer.sm70_num_experts = num_experts
        layer.sm70_w13_n_dim = int(w13_meta[0].item())
        layer.sm70_w13_k_dim = int(w13_meta[1].item())
        layer.sm70_w2_n_dim = int(w2_meta[0].item())
        layer.sm70_w2_k_dim = int(w2_meta[1].item())
        layer.sm70_hidden_logical_size = layer.sm70_w2_n_dim
        layer.sm70_intermediate_size = layer.sm70_w2_k_dim
        layer.sm70_batched_ready = True
        layer._sm70_mxfp4_moe_direct_prepared = True
        self._allocate_buffers(layer, w13.device)

    def _allocate_buffers(self, layer: Module, device: torch.device) -> None:
        top_k = self.moe.experts_per_token
        persistent_tokens = _DEFAULT_PERSISTENT_MAX_TOKENS
        max_slots = persistent_tokens * top_k
        hidden_size = layer.sm70_hidden_logical_size
        layer._buf_max_tokens = persistent_tokens
        layer._buf_max_slots = max_slots
        layer._buf_top_k = top_k
        layer._buf_expert_offsets = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int32, device=device
        )
        layer._buf_expert_offsets64 = torch.empty(
            layer.sm70_num_experts + 1, dtype=torch.int64, device=device
        )
        layer._buf_intermediate = torch.empty(
            max_slots, layer.sm70_intermediate_size, dtype=torch.float16, device=device
        )
        layer._buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device
        )
        layer._buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim, dtype=torch.float16, device=device
        )
        layer._buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._buf_topk_ids_i32 = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device
        )
        layer._buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device
        ).view(persistent_tokens, top_k)
        layer._buf_permuted_idx = torch.empty(max_slots, dtype=torch.int32, device=device)
        layer._buf_m_indices = torch.empty(max_slots, dtype=torch.int32, device=device)
        layer._buf_output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16, device=device
        )

    def _get_buffers(self, layer: Module, total_slots: int, num_tokens: int):
        if total_slots <= layer._buf_max_slots and num_tokens <= layer._buf_max_tokens:
            return {
                "output": layer._buf_output[:num_tokens],
                "permuted_input": layer._buf_permuted_input[:total_slots],
                "sorted_output": layer._buf_sorted_output[:total_slots],
                "gate_up": layer._buf_gate_up[:total_slots],
                "intermediate": layer._buf_intermediate[:total_slots],
                "expert_offsets": layer._buf_expert_offsets,
                "expert_offsets64": layer._buf_expert_offsets64,
                "inv_permuted_idx": layer._buf_inv_permuted_idx[:num_tokens],
                "topk_ids_i32": layer._buf_topk_ids_i32[:num_tokens],
                "token_expert_indices": layer._buf_token_expert_indices[:num_tokens],
                "permuted_idx": layer._buf_permuted_idx[:total_slots],
                "m_indices": layer._buf_m_indices[:total_slots],
            }
        device = layer._buf_output.device
        top_k = layer._buf_top_k
        hidden_size = layer.sm70_hidden_logical_size
        return {
            "output": torch.empty(
                num_tokens, hidden_size, dtype=torch.float16, device=device
            ),
            "permuted_input": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "sorted_output": torch.empty(
                total_slots, hidden_size, dtype=torch.float16, device=device
            ),
            "gate_up": torch.empty(
                total_slots,
                layer.sm70_w13_n_dim,
                dtype=torch.float16,
                device=device,
            ),
            "intermediate": torch.empty(
                total_slots,
                layer.sm70_intermediate_size,
                dtype=torch.float16,
                device=device,
            ),
            "expert_offsets": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int32, device=device
            ),
            "expert_offsets64": torch.empty(
                layer.sm70_num_experts + 1, dtype=torch.int64, device=device
            ),
            "inv_permuted_idx": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "topk_ids_i32": torch.empty(
                num_tokens, top_k, dtype=torch.int32, device=device
            ),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device
            ).view(num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32, device=device),
            "m_indices": torch.empty(total_slots, dtype=torch.int32, device=device),
        }

    def _activation(self) -> MoEActivation:
        activation = self.moe.activation
        if isinstance(activation, MoEActivation):
            return activation
        return MoEActivation.from_str(activation)

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts_input
        if not getattr(layer, "sm70_batched_ready", False):
            raise RuntimeError("SM70 MXFP4 MoE batched runtime is not prepared.")
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        topk_ids_i32 = buffers["topk_ids_i32"]
        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        with _profile_or_null("moe.experts.permute", x):
            if _moe_permute_accepts_scale_and_m_indices():
                torch.ops._moe_C.moe_permute(
                    x,
                    topk_ids_i32,
                    buffers["token_expert_indices"],
                    None,
                    layer.sm70_num_experts,
                    layer.sm70_num_experts,
                    top_k,
                    None,
                    buffers["permuted_input"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["permuted_idx"],
                    buffers["m_indices"],
                )
            else:
                torch.ops._moe_C.moe_permute(
                    x,
                    topk_ids_i32,
                    buffers["token_expert_indices"],
                    None,
                    layer.sm70_num_experts,
                    layer.sm70_num_experts,
                    top_k,
                    buffers["permuted_input"],
                    buffers["expert_offsets64"],
                    buffers["inv_permuted_idx"],
                    buffers["permuted_idx"],
                )
            buffers["expert_offsets"].copy_(buffers["expert_offsets64"], non_blocking=True)

        activation = self._activation()
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        swiglu_limit = getattr(layer, "swiglu_limit", None)
        has_swiglu_limit = swiglu_limit is not None and swiglu_limit > 0
        use_unfused_activation = (
            activation != MoEActivation.SILU
            or w13_bias is not None
            or has_swiglu_limit
        )
        if use_unfused_activation:
            with _profile_or_null("moe.experts.gemm_w13", x):
                ops.sm70_mxfp4_moe_gemm_out(
                    buffers["gate_up"],
                    buffers["permuted_input"],
                    buffers["expert_offsets"],
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    layer.sm70_num_experts,
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    False,
                )
            if w13_bias is not None:
                with _profile_or_null("moe.experts.add_bias_w13", x):
                    ops.sm70_moe_add_bias_out(
                        buffers["gate_up"],
                        buffers["expert_offsets"],
                        w13_bias,
                        layer.sm70_num_experts,
                    )
            with _profile_or_null("moe.experts.activation", x):
                if activation == MoEActivation.SILU and has_swiglu_limit:
                    sm70_fused_swiglu_limit(
                        buffers["intermediate"],
                        buffers["gate_up"],
                        float(swiglu_limit),
                    )
                elif activation == MoEActivation.SILU:
                    sm70_fused_swiglu_limit(
                        buffers["intermediate"],
                        buffers["gate_up"],
                        0.0,
                    )
                else:
                    apply_moe_activation(
                        activation,
                        buffers["intermediate"],
                        buffers["gate_up"],
                    )
        else:
            with _profile_or_null("moe.experts.gemm_w13_fused_silu", x):
                ops.sm70_mxfp4_moe_gemm_out(
                    buffers["intermediate"],
                    buffers["permuted_input"],
                    buffers["expert_offsets"],
                    layer.w13_strided_ptrs_w,
                    layer.w13_strided_ptrs_s,
                    layer.sm70_num_experts,
                    layer.sm70_w13_k_dim,
                    layer.sm70_w13_n_dim,
                    self.group_size,
                    True,
                )
        with _profile_or_null("moe.experts.gemm_w2", x):
            ops.sm70_mxfp4_moe_gemm_out(
                buffers["sorted_output"],
                buffers["intermediate"],
                buffers["expert_offsets"],
                layer.w2_strided_ptrs_w,
                layer.w2_strided_ptrs_s,
                layer.sm70_num_experts,
                layer.sm70_w2_k_dim,
                layer.sm70_w2_n_dim,
                self.group_size,
                False,
            )
        if w2_bias is not None:
            with _profile_or_null("moe.experts.add_bias_w2", x):
                ops.sm70_moe_add_bias_out(
                    buffers["sorted_output"],
                    buffers["expert_offsets"],
                    w2_bias,
                    layer.sm70_num_experts,
                )
        with _profile_or_null("moe.experts.unpermute", x):
            torch.ops._moe_C.moe_unpermute(
                buffers["sorted_output"][:, : layer.sm70_hidden_logical_size],
                topk_weights,
                buffers["inv_permuted_idx"],
                buffers["expert_offsets64"],
                top_k,
                output,
            )
        return output

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Mxfp4SM70MoEMethod only supports the routed FusedMoE path."
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None
