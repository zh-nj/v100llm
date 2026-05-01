# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
)
from vllm.v1.attention.ops import flashmla


def test_flashmla_sparse_backend_supports_sm70() -> None:
    assert FlashMLASparseBackend.supports_compute_capability(DeviceCapability(7, 0))
    assert DeepseekV4FlashMLASparseBackend.supports_compute_capability(
        DeviceCapability(7, 0)
    )


def test_flashmla_sparse_runtime_probe_accepts_sm70_core_extension(
    monkeypatch,
) -> None:
    monkeypatch.setattr(flashmla, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla, "_flashmla_extension_C_AVAILABLE", False)
    monkeypatch.setattr(
        flashmla.current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 70,
    )

    assert flashmla.is_flashmla_sparse_supported() == (True, None)


def test_flashmla_dense_runtime_probe_stays_rejected_on_sm70(
    monkeypatch,
) -> None:
    monkeypatch.setattr(flashmla, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla, "_flashmla_extension_C_AVAILABLE", True)
    monkeypatch.setattr(
        flashmla.current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 70,
    )

    ok, reason = flashmla.is_flashmla_dense_supported()

    assert not ok
    assert reason == "FlashMLA Dense is only supported on Hopper devices."
