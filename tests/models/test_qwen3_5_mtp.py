# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP


def test_mtp_quant_disable_keeps_partial_fp8_skip_list_quantized() -> None:
    cfg = SimpleNamespace(
        quantization_config={
            "modules_to_not_convert": [
                "mtp.fc",
                "mtp.norm",
                "mtp.pre_fc_norm_hidden",
                "mtp.layers.0.input_layernorm",
                "mtp.layers.0.self_attn.q_norm",
            ],
        }
    )

    assert Qwen3_5MTP._mtp_quant_disabled_in_hf_config(cfg) is False


def test_mtp_quant_disable_still_recognizes_whole_branch_skip_marker() -> None:
    cfg = SimpleNamespace(
        quantization_config={
            "modules_to_not_convert": [
                "mtp",
            ],
        }
    )

    assert Qwen3_5MTP._mtp_quant_disabled_in_hf_config(cfg) is True
