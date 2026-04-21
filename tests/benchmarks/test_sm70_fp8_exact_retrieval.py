# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from benchmarks.benchmark_sm70_fp8_exact_retrieval import (
    build_result_row,
    semantic_quality_conclusion,
)


def test_semantic_quality_requires_exact_match():
    assert semantic_quality_conclusion("4317", "4317") == "pass"
    assert semantic_quality_conclusion(" 4317 ", "4317") == "pass"
    assert (
        semantic_quality_conclusion("\n<think>\n\n</think>\n\n4317", "4317")
        == "pass"
    )
    assert semantic_quality_conclusion("4318", "4317") == "fail"


def test_build_result_row_reports_required_fields():
    metrics = SimpleNamespace(
        first_token_latency=0.5,
        first_token_ts=10.0,
        last_token_ts=12.0,
    )
    row = build_result_row(
        input_len=1024,
        prompt_tokens=1024,
        generated_tokens=5,
        finish_reason="length",
        output_text="4317",
        expected_answer="4317",
        metrics=metrics,
    )

    assert row["input_len"] == 1024
    assert set(row) >= {
        "prefill tokens/s",
        "decode tokens/s",
        "TTFT",
        "finish_reason",
        "semantic_quality",
    }
    assert row["semantic_quality"] == "pass"
