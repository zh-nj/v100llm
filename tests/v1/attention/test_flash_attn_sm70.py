# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.vllm_flash_attn import flash_attn_interface


def test_flash_attention_backend_supports_sm70():
    assert FlashAttentionBackend.supports_compute_capability(
        DeviceCapability(7, 0)
    )


def test_sm70_prioritizes_standard_flash_attention():
    assert _get_backend_priorities(False, DeviceCapability(7, 0)) == [
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.FLASH_ATTN_V100,
        AttentionBackendEnum.TRITON_ATTN,
    ]


def test_fa2_version_check_accepts_sm70_when_kernel_is_available(monkeypatch):
    monkeypatch.setattr(flash_attn_interface, "FA2_AVAILABLE", True)
    monkeypatch.setattr(
        flash_attn_interface.torch.cuda,
        "get_device_capability",
        lambda device=None: (7, 0),
    )

    assert flash_attn_interface.is_fa_version_supported(2)
