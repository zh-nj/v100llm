# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.config import DeepseekV4ForCausalLMConfig


def _vllm_config(cache_dtype: str) -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))


def test_deepseek_v4_defaults_auto_kv_cache_to_fp8_ds_mla() -> None:
    vllm_config = _vllm_config("auto")

    DeepseekV4ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"


def test_deepseek_v4_normalizes_fp8_kv_cache_to_fp8_ds_mla() -> None:
    vllm_config = _vllm_config("fp8")

    DeepseekV4ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
