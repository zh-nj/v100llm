# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

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


def test_phase_profile_summary_reports_slowest_rank(tmp_path: Path) -> None:
    profile = tmp_path / "phase.jsonl"
    profile.write_text(
        "\n".join(
            [
                '{"pid":11,"cuda_device":0,"phase":"prefill.swa_gather","elapsed_us":1000}',
                '{"pid":11,"cuda_device":0,"phase":"prefill.swa_gather","elapsed_us":2000}',
                '{"pid":22,"cuda_device":1,"phase":"prefill.swa_gather","elapsed_us":7000}',
                '{"pid":22,"cuda_device":1,"phase":"prefill.flashmla_sparse_fwd","elapsed_us":5000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = bench.summarize_phase_profile(profile)

    assert summary["record_count"] == 4
    assert summary["rank_count"] == 2
    assert summary["phases"][0] == {
        "phase": "prefill.swa_gather",
        "records": 3,
        "rank_count": 2,
        "total_ms_all_ranks": 10.0,
        "max_rank_ms": 7.0,
        "avg_us": 10000.0 / 3.0,
    }
    assert summary["phases"][1]["phase"] == "prefill.flashmla_sparse_fwd"


def test_phase_profile_summary_can_skip_existing_records(tmp_path: Path) -> None:
    profile = tmp_path / "phase.jsonl"
    profile.write_text(
        "\n".join(
            [
                '{"pid":11,"cuda_device":0,"phase":"warmup","elapsed_us":1000}',
                '{"pid":11,"cuda_device":0,"phase":"prefill","elapsed_us":2000}',
                '{"pid":22,"cuda_device":1,"phase":"prefill","elapsed_us":5000}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = bench.summarize_phase_profile(profile, skip_records=1)

    assert summary["record_count"] == 2
    assert [row["phase"] for row in summary["phases"]] == ["prefill"]
    assert summary["phases"][0]["max_rank_ms"] == 5.0


def test_openai_logprobs_reports_first_token(monkeypatch, capsys) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"""{
              "choices": [{
                "text": "Z",
                "finish_reason": "length",
                "logprobs": {
                  "tokens": ["Z"],
                  "token_logprobs": [-0.125],
                  "top_logprobs": [{"Z": -0.125, "X": -2.0}]
                }
              }],
              "usage": {"prompt_tokens": 19, "completion_tokens": 1}
            }"""

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = request.data
        return FakeResponse()

    monkeypatch.setattr(bench.urllib.request, "urlopen", fake_urlopen)

    rc = bench.run_openai_logprobs(
        SimpleNamespace(
            endpoint="http://127.0.0.1:18080/v1",
            api_key="EMPTY",
            model="/models/dsv4",
            prompt="prompt",
            prompt_name="fallback",
            bench_mode="fallback",
            max_tokens=1,
            logprobs=5,
            timeout=30.0,
            preview_chars=32,
        )
    )

    assert rc == 0
    assert captured["url"] == "http://127.0.0.1:18080/v1/completions"
    payload = bench.json.loads(captured["payload"])
    assert payload["prompt"] == "prompt"
    assert payload["logprobs"] == 5
    assert payload["max_tokens"] == 1
    assert payload["stream"] is False

    result = bench.json.loads(capsys.readouterr().out)
    assert result["mode"] == "fallback"
    assert result["prompt_name"] == "fallback"
    assert result["first_token"] == "Z"
    assert result["first_token_logprob"] == -0.125
    assert result["first_token_top_logprobs"] == {"Z": -0.125, "X": -2.0}
    assert result["prompt_tokens"] == 19
    assert result["completion_tokens"] == 1


def test_openai_stream_reports_usage_prompt_tokens(monkeypatch, capsys) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(
                [
                    b'data: {"choices":[{"delta":{"content":"A"},"token_ids":[123]}]}\n',
                    b'data: {"choices":[{"finish_reason":"stop","delta":{}}],'
                    b'"usage":{"prompt_tokens":7,"completion_tokens":1,'
                    b'"total_tokens":8}}\n',
                    b"data: [DONE]\n",
                ]
            )

    monkeypatch.setattr(
        bench.urllib.request,
        "urlopen",
        lambda _request, timeout: FakeResponse(),
    )

    rc = bench.run_openai_stream(
        SimpleNamespace(
            endpoint="http://127.0.0.1:18080/v1",
            api_key="EMPTY",
            model="/models/dsv4",
            prompt="prompt",
            prompt_file=None,
            prompt_name=None,
            bench_mode=None,
            max_tokens=1,
            timeout=30.0,
            preview_chars=32,
        )
    )

    assert rc == 0
    result = bench.json.loads(capsys.readouterr().out)
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 1
    assert result["total_tokens"] == 8


def test_prompt_file_overrides_inline_prompt(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")

    prompt = bench.resolve_prompt(
        SimpleNamespace(prompt="inline prompt", prompt_file=prompt_file)
    )

    assert prompt == "file prompt"
