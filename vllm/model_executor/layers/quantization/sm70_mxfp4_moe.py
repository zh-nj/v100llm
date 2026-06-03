# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
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
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_experts import (
    SM70FusedMoEExperts,
    SM70MXFP4QuantParams,
)
from vllm.model_executor.layers.fused_moe.sm70_fused_moe_gate import (
    SM70FusedConfig,
    SM70FusionLevel,
    sm70_fused_support,
)
from vllm.model_executor.layers.quantization.awq_sm70_moe import (
    _DEFAULT_PERSISTENT_MAX_TOKENS,
    _moe_permute_accepts_scale_and_m_indices,
    _sm70_fused_kernel_shape_supported,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class Mxfp4SM70MoEMethod(FusedMoEMethodBase):
    """SM70 direct MXFP4 MoE method using TurboMind grouped GEMM."""

    compile_boundary_output_dtype = torch.float16

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

    def _init_dims_from_raw_mxfp4(self, layer: Module) -> None:
        """Set the SM70 runtime dims from the raw MXFP4 pack (fused path only).

        Mirrors the dims ``process_weights_after_loading`` derives from the
        TurboMind meta, but reads them from the stashed
        :class:`SM70MXFP4QuantParams` so the TurboMind prep can be skipped when
        the fused path owns the layer (saving the redundant weight copy). The
        fused kernel writes a ``[M, hidden_K]`` output, so the logical hidden
        size equals ``hidden_K`` (the MXFP4 method already rounds hidden up to a
        group_size multiple in ``maybe_roundup_sizes``).
        """
        quant: SM70MXFP4QuantParams = layer.sm70_fused_quant_params
        layer.sm70_num_experts = quant.num_experts
        layer.sm70_w13_n_dim = 2 * quant.inter_I  # gate|up output width
        layer.sm70_w13_k_dim = quant.hidden_K
        layer.sm70_w2_n_dim = quant.hidden_K  # down output width
        layer.sm70_w2_k_dim = quant.inter_I
        layer.sm70_hidden_logical_size = quant.hidden_logical_size
        layer.sm70_intermediate_size = quant.inter_I
        layer.sm70_batched_ready = True
        layer._sm70_mxfp4_moe_direct_prepared = True

    def process_weights_after_loading(self, layer: Module) -> None:
        # Build the fused-path MXFP4 pack first (when the switch is on). It
        # references the *raw* MXFP4 weights (``w13_weight`` / ``w2_weight`` +
        # scales) which the fused kernel consumes directly. Doing this before the
        # TurboMind prep lets us SKIP that prep entirely when fusion is active —
        # the TurboMind-reformatted weights + strided pointers are only needed by
        # the per-operator fallback, and keeping both copies would roughly double
        # the per-GPU expert-weight footprint (OOM on large MoEs like the 256-
        # expert dsv4f at gpu-memory-utilization 0.95).
        self._maybe_stash_sm70_mxfp4_weights(layer)
        fused_active = getattr(layer, "sm70_fused_experts", None) is not None

        if fused_active:
            # Fused path owns this layer: keep only the raw MXFP4 weights (the
            # kernel's input) and the dims it needs; do NOT build the redundant
            # TurboMind weights / strided pointers. This frees the second full
            # expert-weight copy. The per-operator runtime fallback is therefore
            # unavailable for fused layers — acceptable since the gate approved
            # fusion and the kernel is numerically validated; a kernel failure
            # surfaces loudly rather than silently degrading.
            self._init_dims_from_raw_mxfp4(layer)
            logger.info_once(
                "SM70 fused MoE: skipping TurboMind weight prep for layer=%s "
                "(fused path active; raw MXFP4 weights retained, TurboMind copy "
                "freed to save HBM).",
                getattr(layer, "layer_name", "<unknown>"),
            )
            self._allocate_buffers(layer, layer.sm70_mxfp4_w13_weight.device)
            return

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

    def _activation_name(self) -> str:
        """Return the MoE activation as the gate's expected string name."""
        try:
            return self._activation().value
        except Exception:
            activation = self.moe.activation
            return activation if isinstance(activation, str) else str(activation)

    def _maybe_stash_sm70_mxfp4_weights(self, layer: Module) -> None:
        """Stash the raw MXFP4 expert weights + build the fused pack (gated).

        Only runs when ``VLLM_SM70_FUSED_MOE`` is enabled, so the default
        release path stashes/allocates nothing new and is byte-identical to the
        current behavior (R6.3). MUST be called while the *original* MXFP4
        parameters (``w13_weight`` / ``w13_weight_scale`` / ``w2_weight`` /
        ``w2_weight_scale``) are still present on ``layer`` — i.e. before the
        TurboMind prep deletes them — because the fused path consumes exactly
        those packed FP4 tensors and per-32 block scales.

        The raw tensors are stashed under the attribute names
        :meth:`SM70MXFP4QuantParams.from_layer` reads
        (``layer.sm70_mxfp4_w13_weight`` etc.) so the pack can be rebuilt after
        the originals are gone. On any failure the fused attrs are left ``None``
        so :meth:`apply` falls back to the existing TurboMind grouped-GEMM path.
        """
        layer.sm70_fused_quant_params = None
        layer.sm70_fused_experts = None
        if not envs.VLLM_SM70_FUSED_MOE:
            return

        layer_name = getattr(layer, "layer_name", "<unknown>")

        # Feature-parity guard (R6.2): the fused CUDA mega-kernel implements the
        # ``silu(gate) * up`` SwiGLU epilogue WITH the DeepSeek-V4 ``swiglu_limit``
        # clamp (applied on chip), but has no expert-bias add. If this layer
        # needs an expert bias (handled by the TurboMind per-operator path's bias
        # kernel) the fused path would be numerically wrong, so do not build (or
        # stash) the pack at all — the existing path is used. Skipping here also
        # avoids holding an extra full copy of the MXFP4 expert weights alive for
        # a pack that can never run. ``swiglu_limit`` is supported and forwarded
        # to the kernel, so it does NOT disable the fused path.
        has_bias = (
            getattr(layer, "w13_bias", None) is not None
            or getattr(layer, "w2_bias", None) is not None
            or bool(getattr(self.moe, "has_bias", False))
        )
        if has_bias:
            logger.info_once(
                "SM70 fused MoE: disabled for layer=%s (the fused kernel does "
                "not support expert bias; using the TurboMind per-operator "
                "path).",
                layer_name,
            )
            return

        # Stash the raw MXFP4 tensors before the TurboMind prep deletes the
        # originals. Reference the underlying tensors (storage stays alive while
        # referenced); these are the exact packed FP4 weights + per-32 block
        # scales the fused kernel / dense decode consume.
        layer.sm70_mxfp4_w13_weight = layer.w13_weight.data
        layer.sm70_mxfp4_w13_weight_scale = layer.w13_weight_scale.data
        layer.sm70_mxfp4_w2_weight = layer.w2_weight.data
        layer.sm70_mxfp4_w2_weight_scale = layer.w2_weight_scale.data
        layer.group_size = self.group_size

        try:
            quant = SM70MXFP4QuantParams.from_layer(
                layer, group_size=self.group_size
            )
        except Exception as e:  # pragma: no cover - depends on built ext / GPU
            logger.warning_once(
                "SM70 fused MoE: MXFP4 weight pack build failed (%s); the fused "
                "path is disabled for layer=%s and the existing TurboMind path "
                "is used.",
                e,
                layer_name,
            )
            layer.sm70_fused_quant_params = None
            layer.sm70_fused_experts = None
            return
        cfg = SM70FusedConfig.from_env()
        layer.sm70_fused_quant_params = quant
        layer.sm70_fused_config = cfg
        layer.sm70_fused_experts = SM70FusedMoEExperts(config=cfg)
        logger.info_once(
            "SM70 fused MoE: MXFP4 weight pack ready (experts=%d, K=%d, I=%d, "
            "group_size=%d, fusion_level=%s) for layer=%s",
            quant.num_experts,
            quant.hidden_K,
            quant.inter_I,
            quant.group_size,
            cfg.fusion_level.value,
            layer_name,
        )

    def _maybe_apply_sm70_fused(
        self,
        layer: Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Try the experimental SM70 MXFP4 fused path, else signal fallback.

        Layered gating + fallback (R2.4 / R6.1 / R6.2):

        1. **Pure-function gate** on the live device capability + the
           ``VLLM_SM70_FUSED_MOE`` switch + dtype/activation/format/shape
           (:func:`sm70_fused_support`, ``weight_format="mxfp4"``). On rejection,
           log ``reason`` once and return ``None`` (fall back).
        2. **Shape pre-check** mirroring the fused kernel tile / SMEM bounds for
           the fused-kernel levels (L2/L3); on rejection log once and fall back.
        3. **Capture-safety pre-check.** The fused CUDA mega-kernel consumes the
           contiguous layout, which is safe for both prefill and *eager decode*;
           only an active CUDA-graph capture is rejected (the contiguous layout
           is not replay-safe), logging once and falling back to the shape-stable
           existing path (R2.7).
        4. **Runtime guard.** Wrap the fused experts call in ``try/except`` so
           *any* runtime error (kernel/build/shape) falls back to the existing
           TurboMind grouped-GEMM path, logging once.

        Returns the fused output on success, or ``None`` to signal the caller to
        run the existing path.
        """
        quant: SM70MXFP4QuantParams | None = getattr(
            layer, "sm70_fused_quant_params", None
        )
        if quant is None:
            return None
        experts: SM70FusedMoEExperts = layer.sm70_fused_experts
        cfg: SM70FusedConfig = getattr(
            layer, "sm70_fused_config", SM70FusedConfig.from_env()
        )
        layer_name = getattr(layer, "layer_name", "<unknown>")

        # 1. Gate on the live device capability + switch (pure function, R6.1).
        try:
            device_capability = tuple(torch.cuda.get_device_capability(x.device))
        except Exception:
            device_capability = (0, 0)
        support = sm70_fused_support(
            device_capability=device_capability,
            flag_enabled=cfg.enabled,
            weight_format="mxfp4",
            group_size=self.group_size,
            activation=self._activation_name(),
            hidden=quant.hidden_K,
            intermediate=quant.inter_I,
            dtype=x.dtype,
        )
        if not support.enabled:
            logger.warning_once(
                "SM70 fused MoE: falling back (%s) for layer=%s",
                support.reason,
                layer_name,
            )
            return None

        # Feature-parity runtime guard (defense-in-depth): the fused kernel now
        # implements the DeepSeek-V4 ``swiglu_limit`` clamp, but still has no
        # expert-bias add, so fall back when the layer carries an expert bias
        # (the stash guard normally prevents a pack from being built then, but a
        # layer mutated after loading must still be handled correctly).
        if (
            getattr(layer, "w13_bias", None) is not None
            or getattr(layer, "w2_bias", None) is not None
        ):
            logger.warning_once(
                "SM70 fused MoE: falling back (expert bias not supported by the "
                "fused kernel) for layer=%s",
                layer_name,
            )
            return None
        swiglu_limit_val = getattr(layer, "swiglu_limit", None)
        swiglu_limit = (
            float(swiglu_limit_val)
            if swiglu_limit_val is not None and swiglu_limit_val > 0
            else 0.0
        )

        m_block = cfg.m_block
        i_block = cfg.i_block

        # 2. Shape pre-check mirroring the fused kernel constraints (R6.2): the
        # tile / SMEM bounds the mega-kernel needs. The buffer-based fused path
        # below always runs the mega-kernel, so this gates every call.
        ok, reason = _sm70_fused_kernel_shape_supported(
            quant.hidden_K, quant.inter_I, m_block
        )
        if not ok:
            logger.warning_once(
                "SM70 fused MoE: falling back (unsupported shape: %s) for "
                "layer=%s",
                reason,
                layer_name,
            )
            return None

        # 3. Run the fused mega-kernel inside the *capture-safe* persistent-buffer
        # flow (same buffers / moe_permute / moe_unpermute as the production
        # path, all CUDA-graph-replayable). The (expert, tile) kernel grid
        # consumes the dense, unaligned moe_permute output directly and decodes
        # MXFP4 -> fp16 + applies the swiglu_limit clamp on chip, so prefill AND
        # decode (incl. under CUDA-graph capture) take the fused path. A runtime
        # guard falls back on any error.
        try:
            return self._apply_sm70_fused_buffered(
                layer, x, topk_weights, topk_ids, quant, m_block, i_block,
                swiglu_limit,
            )
        except Exception as e:
            logger.warning_once(
                "SM70 fused MoE: runtime error (%s); falling back to the "
                "existing path for layer=%s",
                e,
                layer_name,
            )
            return None

    def _apply_sm70_fused_buffered(
        self,
        layer: Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant: "SM70MXFP4QuantParams",
        m_block: int,
        i_block: int,
        swiglu_limit: float,
    ) -> torch.Tensor:
        """Capture-safe fused MoE forward over the persistent production buffers.

        Mirrors the production ``apply`` buffer flow exactly — ``moe_permute``
        into the persistent ``permuted_input`` buffer with device-side
        ``expert_offsets``, then ``moe_unpermute`` weighted combine — but
        replaces the three-kernel ``gemm_w13 -> swiglu -> gemm_w2`` sequence with
        a single ``ops.sm70_fused_moe_out`` mega-kernel call. Every tensor is a
        pre-allocated persistent buffer and ``moe_permute`` / the fused kernel /
        ``moe_unpermute`` all read the device-side ``expert_offsets`` / shapes,
        so there is **no host sync and no data-dependent allocation** — the whole
        path is CUDA-graph capturable/replayable (unlike
        ``SM70FusedMoEExperts.forward``, which host-syncs to pad segments).

        The fused mega-kernel's (expert, tile) grid confines each block to one
        expert's segment, so it consumes the dense (unaligned) ``moe_permute``
        output directly — no Python-side ``m_block`` padding (the host-sync that
        broke capture) is needed. The ``swiglu_limit`` clamp is applied on chip.
        """
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        if total_slots == 0:
            output.zero_()
            return output

        if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
            topk_ids_i32 = topk_ids
        else:
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
            buffers["expert_offsets"].copy_(
                buffers["expert_offsets64"], non_blocking=True
            )

        # Fused linear1 -> SwiGLU(limit) -> linear2 over the grouped layout. The
        # mega-kernel writes one [M, hidden_K] output row per permuted row;
        # padding rows past each expert segment are skipped by its per-tile
        # m_rows clamp. ``sorted_output`` is a persistent buffer; combine only
        # gathers valid rows via ``inv_permuted_idx``.
        sorted_output = buffers["sorted_output"]
        with _profile_or_null("moe.experts.fused", x):
            ops.sm70_fused_moe_out(
                sorted_output,
                buffers["permuted_input"],
                buffers["expert_offsets"],
                quant.w13_weight,
                quant.w13_weight_scale,
                quant.w2_weight,
                quant.w2_weight_scale,
                quant.num_experts,
                quant.hidden_K,
                quant.inter_I,
                quant.group_size,
                m_block,
                i_block,
                swiglu_limit,
            )

        # moe_unpermute requires fp32 router weights; the SM70 router emits fp32
        # already, but cast defensively for any caller-supplied dtype.
        topk_weights_f32 = (
            topk_weights
            if topk_weights.dtype == torch.float32
            else topk_weights.to(torch.float32)
        )
        with _profile_or_null("moe.experts.unpermute", x):
            torch.ops._moe_C.moe_unpermute(
                sorted_output[:, : layer.sm70_hidden_logical_size],
                topk_weights_f32,
                buffers["inv_permuted_idx"],
                buffers["expert_offsets64"],
                top_k,
                output,
            )
        return output

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
        # --- Experimental SM70 MXFP4 fused MoE path (R2.4 / R6.1 / R6.2) -----
        # Only taken when the ``VLLM_SM70_FUSED_MOE`` switch stashed the MXFP4
        # weights + built a fused pack during process_weights_after_loading;
        # otherwise (the default) this is a single attribute check returning
        # None and the existing TurboMind grouped-GEMM path runs unchanged
        # (R6.3). The gate + layered try/except in the helper fall back on any
        # unsupported shape, layout, or runtime error.
        if getattr(layer, "sm70_fused_experts", None) is not None:
            fused_out = self._maybe_apply_sm70_fused(
                layer, x, topk_weights, topk_ids
            )
            if fused_out is not None:
                return fused_out
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        total_slots = num_tokens * top_k
        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        if total_slots == 0:
            # Nothing to dispatch; zero the output explicitly since
            # moe_unpermute won't be invoked this step.
            output.zero_()
            return output

        # H74-A: drop redundant `topk_ids_i32.copy_(topk_ids)` when the
        # router already returns int32 (the SM70 path), and feed the
        # result of select_experts directly to moe_permute.
        if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
            topk_ids_i32 = topk_ids
        else:
            topk_ids_i32 = buffers["topk_ids_i32"]
            topk_ids_i32.copy_(topk_ids, non_blocking=True)
        # H74-A: skip output.zero_(); finalizeMoeRoutingKernel writes every
        # original_row in the [:num_tokens] output slot.
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
