# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP proposer for speculative decoding."""

from collections import defaultdict
from copy import copy

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class Gemma4Proposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        # Gemma4 assistant layers share target KV and predict from the same
        # target position for each serial draft step.
        self.constant_draft_positions = True
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

        self._centroids_sizes: list[int] = []
        self._centroids_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._centroids_inputs: dict[int, torch.Tensor] = {}
        self._centroids_outputs: dict[int, torch.Tensor] = {}

    def set_per_group_block_table(self, gid: int, block_table: torch.Tensor) -> None:
        self._per_group_block_tables[gid] = block_table

    def model_returns_tuple(self) -> bool:
        return True

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[gid]
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._centroids_sizes:
            num_tokens = hidden_states.shape[0]
            for size in self._centroids_sizes:
                if size >= num_tokens:
                    self._centroids_inputs[size][:num_tokens].copy_(hidden_states)
                    self._centroids_graphs[size].replay()
                    return self._centroids_outputs[size][:num_tokens].clone()
            return self.model.get_top_tokens(hidden_states)
        return super()._greedy_sample(hidden_states)

    def _setup_centroids_cuda_graphs(self) -> None:
        masked_emb = self.model.masked_embedding
        lm_head_weight = self.model._get_full_lm_head_weight()

        for size in [1, 2, 4, 8, 16, 32, 64]:
            static_input = torch.zeros(
                size,
                masked_emb.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(3):
                masked_emb.get_top_tokens(static_input, lm_head_weight)
            torch.accelerator.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = masked_emb.get_top_tokens(
                    static_input,
                    lm_head_weight,
                )
            self._centroids_graphs[size] = graph
            self._centroids_inputs[size] = static_input
            self._centroids_outputs[size] = static_output

        self._centroids_sizes = sorted(self._centroids_graphs)
        logger.info(
            "Gemma4 MTP: captured centroids CUDA graphs for sizes %s.",
            self._centroids_sizes,
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        new_compilation = replace(
            base.compilation_config,
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        # `replace()` creates a fresh static_forward_context dict on the
        # cloned compilation_config; that hides draft attention layers from
        # the target vllm_config and breaks `_draft_attn_layer_names`
        # discovery in the eagle base class. Re-link the dict so draft
        # attentions register into the same registry as the target's.
        new_compilation.static_forward_context = (
            base.compilation_config.static_forward_context
        )
        base = replace(base, compilation_config=new_compilation)
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is not None:
            base = replace(
                base,
                attention_config=replace(
                    base.attention_config,
                    backend=target_backend,
                ),
            )
        return base

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        logger.info("Gemma4 MTP: keeping draft model's own lm_head.")

    def dummy_run(self, *args, **kwargs) -> None:
        logger.info_once(
            "Gemma4 MTP: skipping draft dummy_run during memory profiling."
        )
        return None

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        super().load_model(target_model)
        self._setup_gemma4_kv_sharing(target_attn_layer_names)

        if getattr(self.model, "masked_embedding", None) is not None:
            self._setup_centroids_cuda_graphs()

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Gemma4 draft layers intentionally span multiple KV cache groups."""

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                            layer_name
                        ]
                    else:
                        target = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if target and target in group_spec.kv_cache_specs:
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[tuple[tuple[str, str], KVCacheSpec], AttentionGroup] = {}
        logger.info(
            "Gemma4 MTP initialize_attn_backend: %d draft layers, "
            "%d kv groups, %d layer_to_spec entries",
            len(self._draft_attn_layer_names),
            len(kv_cache_config.kv_cache_groups),
            len(layer_to_spec),
        )
        for layer_name in self._draft_attn_layer_names:
            if layer_name not in layer_to_spec:
                # Draft layers are kv-shared with target layers, so they are
                # not registered as their own kv_cache_group. Resolve the spec
                # and gid via the target layer name instead.
                attn_layer = all_attn_layers.get(layer_name)
                target_name = getattr(
                    attn_layer, "kv_sharing_target_layer_name", None
                )
                if (
                    target_name is not None
                    and target_name in layer_to_spec
                    and target_name in layer_to_gid
                ):
                    layer_to_spec[layer_name] = layer_to_spec[target_name]
                    layer_to_gid[layer_name] = layer_to_gid[target_name]
                    logger.info(
                        "Gemma4 MTP draft %s -> target %s gid=%d",
                        layer_name,
                        target_name,
                        layer_to_gid[layer_name],
                    )
                else:
                    logger.warning(
                        "Gemma4 MTP: skipping draft layer %s (no kv_cache "
                        "group assignment; target=%r in_spec=%s in_gid=%s)",
                        layer_name,
                        target_name,
                        target_name in layer_to_spec
                        if target_name
                        else "n/a",
                        target_name in layer_to_gid
                        if target_name
                        else "n/a",
                    )
                    continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), spec)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )
        else:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _setup_gemma4_kv_sharing(
        self,
        target_attn_layer_names: set[str],
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        draft_config = self.speculative_config.draft_model_config.hf_config
        draft_text_config = draft_config.get_text_config()
        target_config = self.vllm_config.model_config.hf_config
        target_text_config = target_config.get_text_config()
        target_layer_types = getattr(target_text_config, "layer_types", [])

        if not (hasattr(self.model, "model") and hasattr(self.model.model, "layers")):
            return

        target_num_kv_shared = getattr(target_text_config, "num_kv_shared_layers", 0)
        num_non_shared = len(target_layer_types) - target_num_kv_shared
        type_to_target_indices: dict[str, list[int]] = defaultdict(list)
        for idx, layer_type in enumerate(target_layer_types[:num_non_shared]):
            type_to_target_indices[layer_type].append(idx)

        target_prefix = "model.layers"
        for name in target_attn_layer_names:
            if ".layers." in name:
                target_prefix = name.split(".layers.")[0] + ".layers"
                break

        draft_layer_types = getattr(draft_text_config, "layer_types", [])
        for draft_idx, layer in enumerate(self.model.model.layers):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is None:
                continue

            draft_layer_type = (
                draft_layer_types[draft_idx]
                if draft_idx < len(draft_layer_types)
                else "full_attention"
            )
            candidates = type_to_target_indices.get(draft_layer_type, [])
            if not candidates:
                logger.warning(
                    "No target layer of type '%s' for draft layer %d",
                    draft_layer_type,
                    draft_idx,
                )
                continue

            target_idx = candidates[-1]
            target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"
            attn.kv_sharing_target_layer_name = target_layer_name
            target_attn = all_attn_layers.get(target_layer_name)
            if target_attn is not None:
                self._inherit_target_kv_cache_state(attn, target_attn)
            logger.info(
                "Gemma4 MTP: draft layer %d (%s) -> %s",
                draft_idx,
                draft_layer_type,
                target_layer_name,
            )

    @staticmethod
    def _inherit_target_kv_cache_state(
        draft_attn: AttentionLayerBase,
        target_attn: AttentionLayerBase,
    ) -> None:
        for attr in (
            "kv_cache_dtype",
            "kv_cache_torch_dtype",
            "_k_scale",
            "_v_scale",
            "_k_scale_float",
            "_v_scale_float",
        ):
            if hasattr(target_attn, attr):
                setattr(draft_attn, attr, getattr(target_attn, attr))

        draft_impl = getattr(draft_attn, "impl", None)
        target_impl = getattr(target_attn, "impl", None)
        if draft_impl is not None and target_impl is not None and hasattr(
            target_impl,
            "kv_cache_dtype",
        ):
            draft_impl.kv_cache_dtype = target_impl.kv_cache_dtype
