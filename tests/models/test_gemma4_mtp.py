# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from safetensors import safe_open

from vllm.config import ModelConfig, ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.gemma4_mtp import (
    _get_gemma4_mtp_num_kv_heads,
    _select_gemma4_mtp_attention_backend,
)
from vllm.platforms.interface import DeviceCapability
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import _get_image_token_index_for_draft

GEMMA4_ASSISTANT_PATH = "/mnt/data6/models/gemma-4-31B-it-assistant"


def test_gemma4_assistant_config_loads_with_mtp_override() -> None:
    cfg = get_config(
        GEMMA4_ASSISTANT_PATH,
        trust_remote_code=False,
        hf_overrides_fn=SpeculativeConfig.hf_config_override,
    )

    assert cfg.model_type == "gemma4_mtp"
    assert cfg.architectures == ["Gemma4MTPModel"]
    assert cfg.n_predict == 1
    assert cfg.backbone_hidden_size == 5376
    assert cfg.text_config.hidden_size == 1024
    assert cfg.text_config.num_hidden_layers == 4


def test_gemma4_mtp_arch_convertor_uses_backbone_hidden_size() -> None:
    cfg = get_config(
        GEMMA4_ASSISTANT_PATH,
        trust_remote_code=False,
        hf_overrides_fn=SpeculativeConfig.hf_config_override,
    )
    text_cfg = get_hf_text_config(cfg)
    convertor = MODEL_ARCH_CONFIG_CONVERTORS["gemma4_mtp"](cfg, text_cfg)

    assert convertor.get_hidden_size() == 5376
    assert convertor.get_num_hidden_layers() == 4


def test_speculative_config_detects_gemma4_mtp_draft_model() -> None:
    target = ModelConfig(
        model="/mnt/data6/models/gemma-4-31B-it-AWQ-4bit",
        runner="generate",
        max_model_len=512,
        trust_remote_code=False,
    )
    spec = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=ParallelConfig(tensor_parallel_size=4),
        model=GEMMA4_ASSISTANT_PATH,
        method="mtp",
        num_speculative_tokens=1,
    )

    assert spec.use_gemma4_mtp()
    assert spec.draft_model_config.hf_config.model_type == "gemma4_mtp"
    assert spec.draft_model_config.architecture == "Gemma4MTPModel"


def test_gemma4_assistant_checkpoint_has_unquantized_attention() -> None:
    with safe_open(
        f"{GEMMA4_ASSISTANT_PATH}/model.safetensors",
        framework="pt",
    ) as checkpoint:
        keys = set(checkpoint.keys())

    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert not any(".self_attn.attn." in key for key in keys)


def test_gemma4_mtp_uses_image_token_id_for_multimodal_target() -> None:
    target_model = SimpleNamespace(config=SimpleNamespace(image_token_id=258880))

    assert (
        _get_image_token_index_for_draft(
            target_model,
            "Gemma4ForConditionalGeneration",
        )
        == 258880
    )


def test_gemma4_mtp_draft_full_attention_uses_flash_attn() -> None:
    sliding_backend = _select_gemma4_mtp_attention_backend(
        layer_type="sliding_attention",
        head_dim=256,
        capability=DeviceCapability(7, 0),
        user_backend=None,
        kv_transfer_enabled=False,
    )
    full_backend = _select_gemma4_mtp_attention_backend(
        layer_type="full_attention",
        head_dim=512,
        capability=DeviceCapability(7, 0),
        user_backend=None,
        kv_transfer_enabled=False,
    )

    assert sliding_backend is FlashAttentionBackend
    assert full_backend is FlashAttentionBackend


def test_gemma4_mtp_respects_flash_attn_v100_override() -> None:
    sliding_backend = _select_gemma4_mtp_attention_backend(
        layer_type="sliding_attention",
        head_dim=256,
        capability=DeviceCapability(7, 0),
        user_backend=AttentionBackendEnum.FLASH_ATTN_V100,
        kv_transfer_enabled=False,
    )
    full_backend = _select_gemma4_mtp_attention_backend(
        layer_type="full_attention",
        head_dim=512,
        capability=DeviceCapability(7, 0),
        user_backend=AttentionBackendEnum.FLASH_ATTN_V100,
        kv_transfer_enabled=False,
    )

    assert sliding_backend is FlashAttnV100Backend
    assert full_backend is FlashAttnV100Backend


def test_gemma4_mtp_full_attention_uses_global_kv_heads_for_k_eq_v() -> None:
    cfg = SimpleNamespace(
        attention_k_eq_v=True,
        num_global_key_value_heads=4,
        num_key_value_heads=16,
    )

    assert _get_gemma4_mtp_num_kv_heads(cfg, "full_attention") == 4
    assert _get_gemma4_mtp_num_kv_heads(cfg, "sliding_attention") == 16
