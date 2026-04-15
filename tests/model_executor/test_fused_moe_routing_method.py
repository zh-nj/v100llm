# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm"
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "config.py"
)
spec = spec_from_file_location("test_fused_moe_config_module", CONFIG_PATH)
assert spec is not None and spec.loader is not None
config_module = module_from_spec(spec)
spec.loader.exec_module(config_module)

RoutingMethodType = config_module.RoutingMethodType
get_routing_method_type = config_module.get_routing_method_type

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("scoring_func", "renormalize", "expected"),
    [
        ("softmax", False, RoutingMethodType.Renormalize),
        ("softmax", True, RoutingMethodType.RenormalizeNaive),
        ("sigmoid", False, RoutingMethodType.Renormalize),
        ("sigmoid", True, RoutingMethodType.RenormalizeNaive),
    ],
)
def test_get_routing_method_type_matches_legacy_topk_router(
    scoring_func: str,
    renormalize: bool,
    expected: RoutingMethodType,
) -> None:
    assert get_routing_method_type(
        scoring_func=scoring_func,
        top_k=2,
        renormalize=renormalize,
        num_expert_group=None,
        has_e_score_bias=False,
    ) == expected
    assert get_routing_method_type(
        scoring_func=scoring_func,
        top_k=2,
        renormalize=renormalize,
        num_expert_group=None,
        has_e_score_bias=True,
    ) == expected


def test_get_routing_method_type_maps_grouped_sigmoid_to_deepseek_v3() -> None:
    assert get_routing_method_type(
        scoring_func="sigmoid",
        top_k=8,
        renormalize=True,
        num_expert_group=8,
        has_e_score_bias=True,
    ) == RoutingMethodType.DeepSeekV3


def test_get_routing_method_type_leaves_grouped_softmax_unspecified() -> None:
    assert get_routing_method_type(
        scoring_func="softmax",
        top_k=8,
        renormalize=True,
        num_expert_group=8,
        has_e_score_bias=False,
    ) == RoutingMethodType.Unspecified
