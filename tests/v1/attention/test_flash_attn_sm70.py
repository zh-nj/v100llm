# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.platforms.cuda import _get_backend_priorities
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends import flash_attn
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.flash_attn_v100 import FlashAttnV100Backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.vllm_flash_attn import flash_attn_interface


def _validate_flash_attn(
    head_size: int,
    capability: DeviceCapability,
    has_sink: bool = False,
) -> list[str]:
    return FlashAttentionBackend.validate_configuration(
        head_size=head_size,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=None,
        use_mla=False,
        has_sink=has_sink,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type=AttentionType.DECODER,
    )


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


def test_flash_attn_v100_keeps_triton_kv_layout():
    assert FlashAttnV100Backend.get_required_kv_cache_layout() is None
    assert FlashAttnV100Backend.get_kv_cache_shape(8, 16, 4, 256) == (
        8,
        2,
        16,
        4,
        256,
    )
    assert FlashAttnV100Backend.supports_head_size(256)
    assert FlashAttnV100Backend.supports_head_size(512)


def test_fa2_version_check_accepts_sm70_when_kernel_is_available(monkeypatch):
    monkeypatch.setattr(flash_attn_interface, "FA2_AVAILABLE", True)
    monkeypatch.setattr(
        flash_attn_interface.torch.cuda,
        "get_device_capability",
        lambda device=None: (7, 0),
    )

    assert flash_attn_interface.is_fa_version_supported(2)


def test_flash_attention_sm70_accepts_hdim512_via_fa2_v100():
    reasons = _validate_flash_attn(512, DeviceCapability(7, 0))

    assert reasons == []


def test_flash_attention_sm70_accepts_sinks_when_fa2_supports_them(monkeypatch):
    monkeypatch.setattr(
        FlashAttentionBackend,
        "supports_sink",
        classmethod(lambda cls: True),
    )

    assert _validate_flash_attn(
        128,
        DeviceCapability(7, 0),
        has_sink=True,
    ) == []


def test_flash_attention_reports_fa2_sink_support(monkeypatch):
    monkeypatch.setattr(fa_utils, "get_flash_attn_version", lambda: 2)
    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(fa_utils, "fa2_varlen_supports_s_aux", lambda: True)

    assert fa_utils.flash_attn_supports_sinks()


def test_flash_attention_hides_sinks_for_legacy_fa2_abi(monkeypatch):
    monkeypatch.setattr(fa_utils, "get_flash_attn_version", lambda: 2)
    monkeypatch.setattr(fa_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(fa_utils, "fa2_varlen_supports_s_aux", lambda: False)

    assert not fa_utils.flash_attn_supports_sinks()


def test_flash_attention_hdim512_is_not_available_on_sm80():
    reasons = _validate_flash_attn(512, DeviceCapability(8, 0))

    assert "head_size > 256 is only supported by SM70 FlashAttention kernels" in reasons


def test_flash_attention_paged_verify_skips_hdim512(monkeypatch):
    monkeypatch.setattr(flash_attn.current_platform, "is_cuda", lambda: True)
    metadata = SimpleNamespace(
        causal=True,
        max_query_len=2,
        spec_decode_seq_lens_minus_one=object(),
        spec_decode_paged_verify_output=object(),
    )

    impl = object.__new__(FlashAttentionImpl)
    impl.vllm_flash_attn_version = 2
    impl.dcp_world_size = 1
    impl.alibi_slopes = None
    impl.sinks = None
    impl.logits_soft_cap = 0
    impl.sliding_window = (-1, -1)
    impl.kv_cache_dtype = "auto"

    impl.head_size = 256
    assert impl._can_use_paged_verify(metadata, 2, 1)

    impl.head_size = 512
    assert not impl._can_use_paged_verify(metadata, 2, 1)
