# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``--kv-offloading-backend`` -> KV connector mapping.

After backporting the upstream v0.22.0 HMA-aware offloading, both connectors
reachable via ``native`` support the hybrid KV-cache manager:

* ``native`` (default)                       -> ``OffloadingConnector``
* ``native`` + ``VLLM_USE_SIMPLE_KV_OFFLOAD`` -> ``SimpleCPUOffloadConnector``

Both are HMA-aware, so DeepSeek-V4 (SWA ring + compressed 4/128 + indexer) can
offload KV to host CPU memory without ``--disable-hybrid-kv-cache-manager``.

These tests drive ``VllmConfig._post_init_kv_transfer_config`` directly with a
lightweight stand-in config so they run without constructing a full engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import vllm.envs as envs
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.vllm import VllmConfig


def _make_cfg(backend: str, size_gib: float | None = 16.0) -> SimpleNamespace:
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
    VllmConfig._post_init_kv_transfer_config(cfg)


def test_native_default_maps_to_offloading_connector() -> None:
    cfg = _make_cfg("native", size_gib=16.0)
    with mock.patch.object(envs, "VLLM_USE_SIMPLE_KV_OFFLOAD", False):
        _run(cfg)
    assert isinstance(cfg.kv_transfer_config, KVTransferConfig)
    assert cfg.kv_transfer_config.kv_connector == "OffloadingConnector"
    extra = cfg.kv_transfer_config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == 16 * (1 << 30)
    assert cfg.kv_transfer_config.kv_role == "kv_both"


def test_native_with_simple_env_maps_to_simple_cpu_offload() -> None:
    cfg = _make_cfg("native", size_gib=8.0)
    with mock.patch.object(envs, "VLLM_USE_SIMPLE_KV_OFFLOAD", True):
        _run(cfg)
    assert cfg.kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector"
    extra = cfg.kv_transfer_config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == 8 * (1 << 30)


def test_no_offloading_when_size_unset() -> None:
    cfg = _make_cfg("native", size_gib=None)
    _run(cfg)
    assert cfg.kv_transfer_config is None


def test_lmcache_unaffected() -> None:
    cfg = _make_cfg("lmcache", size_gib=32.0)
    _run(cfg)
    assert cfg.kv_transfer_config.kv_connector == "LMCacheConnectorV1"


def test_both_native_connectors_are_hma_aware() -> None:
    """Both connectors reachable via 'native' must subclass SupportsHMA,
    otherwise the factory rejects them under the hybrid KV-cache manager.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )

    assert supports_hma(OffloadingConnector)
    assert supports_hma(SimpleCPUOffloadConnector)
