# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.deepseek_v4_flashmla_sm70 import (
    choice_contains_generated_delta,
)


def test_generated_delta_detection_counts_empty_special_token_chunks() -> None:
    assert not choice_contains_generated_delta({
        "delta": {
            "role": "assistant",
            "content": "",
        },
        "token_ids": None,
    })
    assert choice_contains_generated_delta({
        "delta": {
            "content": "",
        },
        "token_ids": None,
    })
    assert choice_contains_generated_delta({
        "delta": {
            "content": "<|special|>",
        },
        "token_ids": [0],
    })
