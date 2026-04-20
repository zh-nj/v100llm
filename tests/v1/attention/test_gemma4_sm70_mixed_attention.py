# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.config import Gemma4Config
from vllm.model_executor.models.gemma4 import (
    _select_gemma4_text_attention_backend,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_gemma4_sm70_sliding_layers_prefer_flash_attn() -> None:
    backend = _select_gemma4_text_attention_backend(
        layer_type="sliding_attention",
        head_dim=256,
        capability=DeviceCapability(7, 0),
        user_backend=None,
        kv_transfer_enabled=False,
    )

    assert backend is FlashAttentionBackend


def test_gemma4_sm70_full_hdim512_layers_prefer_flash_attn() -> None:
    backend = _select_gemma4_text_attention_backend(
        layer_type="full_attention",
        head_dim=512,
        capability=DeviceCapability(7, 0),
        user_backend=None,
        kv_transfer_enabled=False,
    )

    assert backend is FlashAttentionBackend


def test_gemma4_mixed_attention_respects_user_backend_override() -> None:
    backend = _select_gemma4_text_attention_backend(
        layer_type="sliding_attention",
        head_dim=256,
        capability=DeviceCapability(7, 0),
        user_backend=AttentionBackendEnum.TRITON_ATTN,
        kv_transfer_enabled=False,
    )

    assert backend is None


def test_gemma4_mixed_attention_disables_itself_for_kv_transfer() -> None:
    backend = _select_gemma4_text_attention_backend(
        layer_type="full_attention",
        head_dim=512,
        capability=DeviceCapability(7, 0),
        user_backend=None,
        kv_transfer_enabled=True,
    )

    assert backend is None


def test_gemma4_mixed_attention_is_sm70_only() -> None:
    backend = _select_gemma4_text_attention_backend(
        layer_type="full_attention",
        head_dim=512,
        capability=DeviceCapability(8, 0),
        user_backend=None,
        kv_transfer_enabled=False,
    )

    assert backend is None


def test_gemma4_config_keeps_global_backend_unset() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(head_dim=256, global_head_dim=512)
        ),
        attention_config=SimpleNamespace(backend=None),
    )

    Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend is None
