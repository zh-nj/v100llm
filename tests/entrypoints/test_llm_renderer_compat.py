# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.llm import LLM
from vllm.renderers.inputs.preprocess import parse_model_prompt


class _DummyTokParams:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def with_kwargs(self, **kwargs):
        self.calls.append(kwargs)
        return ("tok-params", kwargs)


class _DummyRenderer:
    def __init__(self) -> None:
        self.default_cmpl_tok_params = _DummyTokParams()
        self.calls: list[tuple[list[object], object]] = []

    def render_cmpl(self, prompts, tok_params):
        self.calls.append((list(prompts), tok_params))
        return ["engine-input"]


def test_preprocess_completion_uses_renderer_render_cmpl():
    renderer = _DummyRenderer()
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        encoder_config=None,
        max_model_len=4096,
        is_multimodal_model=False,
    )

    llm = object.__new__(LLM)
    llm.model_config = model_config
    llm.llm_engine = SimpleNamespace(renderer=renderer)

    output = llm._preprocess_completion(
        "hello",
        tokenization_kwargs={"truncate_prompt_tokens": 32},
    )

    assert output == ["engine-input"]
    assert renderer.default_cmpl_tok_params.calls == [
        {"truncate_prompt_tokens": 32},
    ]
    assert renderer.calls == [
        (
            [parse_model_prompt(model_config, "hello")],
            ("tok-params", {"truncate_prompt_tokens": 32}),
        )
    ]
