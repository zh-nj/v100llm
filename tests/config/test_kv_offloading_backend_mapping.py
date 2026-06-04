# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``--kv-offloading-backend`` -> KV connector mapping.

Covers the wiring added so DeepSeek-V4 (and any hybrid / heterogeneous-KV
model) can offload KV to host CPU memory: a new ``native_hma`` backend value
maps to the HMA-aware ``SimpleCPUOffloadConnector`` instead of the plain
``OffloadingConnector`` (which rejects the hybrid KV-cache manager).

These tests drive ``VllmConfig._post_init_kv_transfer_config`` directly with a
lightweight stand-in config so they run without constructing a full engine /
loading a model.
"""

from __future__ import annotations

import typing
from types import SimpleNamespace

import pytest

from vllm.config.cache import KVOffloadingBackend
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.vllm import VllmConfig


def _make_cfg(backend: str, size_gib: float | None = 16.0) -> SimpleNamespace:
    """A minimal stand-in exposing only what the method under test reads."""
    cfg = SimpleNamespace()
    cfg.cache_config = SimpleNamespace(
        kv_offloading_size=size_gib,
        kv_offloading_backend=backend,
    )
    cfg.parallel_config = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
    )
    cfg.kv_transfer_config = None
    return cfg


def _run(cfg: SimpleNamespace) -> None:
    # Call the real method, bound to our stand-in (it only touches the
    # attributes set above).
    VllmConfig._post_init_kv_transfer_config(cfg)


def test_literal_includes_native_hma() -> None:
    assert "native_hma" in typing.get_args(KVOffloadingBackend)


def test_native_hma_maps_to_simple_cpu_offload_connector() -> None:
    cfg = _make_cfg("native_hma", size_gib=16.0)
    _run(cfg)

    assert isinstance(cfg.kv_transfer_config, KVTransferConfig)
    assert cfg.kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"
    # cpu_bytes_to_use is server-wide and in bytes.
    extra = cfg.kv_transfer_config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == 16 * (1 << 30)
    assert cfg.kv_transfer_config.kv_role == "kv_both"


def test_native_maps_to_plain_offloading_connector() -> None:
    cfg = _make_cfg("native", size_gib=8.0)
    _run(cfg)

    assert cfg.kv_transfer_config.kv_connector == "OffloadingConnector"
    extra = cfg.kv_transfer_config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == 8 * (1 << 30)


def test_no_offloading_when_size_unset() -> None:
    cfg = _make_cfg("native_hma", size_gib=None)
    _run(cfg)
    # Method returns early; no connector configured.
    assert cfg.kv_transfer_config is None


def test_lmcache_unaffected() -> None:
    cfg = _make_cfg("lmcache", size_gib=32.0)
    _run(cfg)
    assert cfg.kv_transfer_config.kv_connector == "LMCacheConnectorV1"


def test_simple_cpu_offload_connector_is_hma_aware() -> None:
    """The target connector must actually subclass SupportsHMA, otherwise the
    factory would reject it under the (still-enabled) hybrid KV-cache manager.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )

    assert supports_hma(SimpleCPUOffloadConnector)
