# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import BatchDescriptor
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend


def _make_vllm_config_for_full_decode_only():
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        mode=CompilationMode.NONE,
        cudagraph_capture_sizes=[1],
    )
    compilation_config.max_cudagraph_capture_size = 1
    compilation_config.post_init_cudagraph_sizes()

    config = MagicMock(spec=VllmConfig)
    config.compilation_config = compilation_config
    config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=1)
    config.parallel_config = ParallelConfig()
    config.speculative_config = None
    config.lora_config = None
    return config


def test_deepseek_v4_flashmla_sparse_backends_allow_uniform_decode_cudagraph():
    assert (
        FlashMLASparseBackend.get_builder_cls().get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        DeepseekV4FlashMLASparseBackend.get_builder_cls().get_cudagraph_support(
            None, None
        )
        == AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        DeepseekSparseSWABackend.get_builder_cls().get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_full_decode_only_dispatches_single_token_uniform_decode(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda *args, **kwargs: DeviceCapability(7, 0),
    )
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    dispatcher = CudagraphDispatcher(_make_vllm_config_for_full_decode_only())
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_DECODE_ONLY, 1)

    mode, desc = dispatcher.dispatch(num_tokens=1, uniform_decode=True)

    assert mode == CUDAGraphMode.FULL
    assert desc == BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True)
