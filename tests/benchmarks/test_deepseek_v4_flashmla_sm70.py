# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from benchmarks import deepseek_v4_flashmla_sm70 as bench
from benchmarks.deepseek_v4_flashmla_sm70 import choice_contains_generated_delta


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


def test_inspect_static_reports_flashmla_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        bench,
        "inspect_flashmla_source",
        lambda: {
            "path": "/mnt/data/apps/FlashMLA",
            "branch": "feature/sm70-volta-flashmla",
            "head": "a507081",
            "has_sm70_sparse_decode_sources": True,
            "has_sm70_sparse_prefill_sources": True,
        },
    )
    monkeypatch.setattr(
        bench,
        "inspect_flashmla_runtime",
        lambda: {
            "vllm_file": "/tmp/worktree/vllm/__init__.py",
            "flashmla_core_importable": False,
            "flashmla_core_import_error": "ModuleNotFoundError",
            "sparse_supported": [False, "not built"],
        },
    )

    report = bench.inspect_static()

    assert report["flashmla_source"]["path"] == "/mnt/data/apps/FlashMLA"
    assert "flashmla_runtime" in report
    assert "flashmla_core_importable" in report["flashmla_runtime"]


def test_flashmla_source_inspect_checks_sm70_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    for relpath in (
        "csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu",
        "csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu",
        "csrc/sm70/prefill/sparse/instantiations/bf16.cu",
    ):
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// test\n", encoding="utf-8")

    def fake_git_output(_repo: Path, *args: str) -> str:
        if args == ("branch", "--show-current"):
            return "feature/sm70-volta-flashmla"
        if args == ("rev-parse", "--short", "HEAD"):
            return "a507081"
        raise AssertionError(args)

    monkeypatch.setattr(bench, "FLASHMLA_SRC_PATH", tmp_path)
    monkeypatch.setattr(bench, "_git_output", fake_git_output)

    report = bench.inspect_flashmla_source()

    assert report == {
        "path": str(tmp_path),
        "branch": "feature/sm70-volta-flashmla",
        "head": "a507081",
        "has_sm70_sparse_decode_sources": True,
        "has_sm70_sparse_prefill_sources": True,
    }
