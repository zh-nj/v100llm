# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig, PoolerConfig


def make_model_config(
    *,
    task: str | None = None,
    architectures: list[str] | None = None,
    convert_type: str = "none",
    score_type: str = "bi-encoder",
):
    model_config = object.__new__(ModelConfig)
    model_config.pooler_config = PoolerConfig(task=task)
    model_config.model_arch_config = SimpleNamespace(
        architectures=architectures or ["DummyModel"]
    )
    model_config.convert_type = convert_type
    model_config._model_info = SimpleNamespace(score_type=score_type)
    return model_config


def test_get_pooling_task_uses_explicit_pooler_task():
    model_config = make_model_config(task="token_embed")

    assert model_config.get_pooling_task(("embed", "token_embed")) == "token_embed"


def test_get_pooling_task_rejects_unsupported_explicit_pooler_task():
    model_config = make_model_config(task="token_embed")

    with pytest.raises(RuntimeError, match="Unsupported task: 'token_embed'"):
        model_config.get_pooling_task(("embed",))


def test_get_pooling_task_prefers_embedding_for_multitask_embedding_model():
    model_config = make_model_config()

    assert model_config.get_pooling_task(("embed", "token_embed")) == "embed"


def test_get_pooling_task_detects_token_classification_architecture():
    model_config = make_model_config(
        architectures=["BertForTokenClassification"],
    )

    assert model_config.get_pooling_task(("classify", "token_classify")) == (
        "token_classify"
    )


def test_score_type_reports_cross_encoder_for_classify_conversion():
    model_config = make_model_config(convert_type="classify", score_type="bi-encoder")

    assert model_config.score_type == "cross-encoder"


def test_score_type_uses_model_info_without_classify_conversion():
    model_config = make_model_config(score_type="late-interaction")

    assert model_config.score_type == "late-interaction"
