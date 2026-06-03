# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SMOKE test for the SM70 fused MoE benchmark script (R7.1 / R7.3).

Feature: deepgemm-megamoe-sm70-port

This is a SMOKE test (NOT a property test). It validates the two benchmark
deliverables from the design's Requirements Mapping (R7 -> "SMOKE + EXAMPLE"):

* **R7.1 — the benchmark runs and produces a comparison table.** The script
  ``benchmarks/kernels/benchmark_sm70_fused_moe.py`` (task 7.1 / 7.2) must be
  runnable and emit a non-empty fused-vs-baseline comparison table reporting the
  per-variant latency / throughput at identical shapes. We exercise this three
  ways, none of which needs a real V100 or the built CUDA kernel:

    1. ``SM70MoEBenchmark(device="cpu").run_shape(...)`` on a tiny shape — the
       masked per-operator path runs on CPU and every variant degrades to an
       *available=False* row rather than aborting, so the rendered table always
       has the header columns + at least one variant row.
    2. ``format_comparison_table(_demo_results(), ...)`` on the synthetic demo
       rows — deterministic header / variant-row assertions in both ``plain``
       and ``markdown`` formats.
    3. the ``main(["--self-test"])`` CLI — prints the synthetic table and exits
       ``0`` (validates the CLI wiring + table formatting end to end).

* **R7.3 — the artifact lands in the repo's ``测试结果/`` convention.**
  ``archive_results(...)`` must write a markdown document whose path follows the
  existing ``<device>_x<count>,<repo>-<version>,<subject>,<YYYYMMDD_HHMMSS>``
  naming scheme (see e.g.
  ``测试结果/Qwen3.5-27B-AWQ/tp4/Tesla_V100-16G_x4,1Cat-vLLM-0.0.2,...,20260321_104628.png``)
  and whose body contains the comparison table. To avoid polluting the
  committed ``测试结果/`` tree the archive is written into pytest's ``tmp_path``
  (``out_dir=tmp_path``); the timestamp / device-count are pinned so the
  filename pattern is asserted deterministically. ``tmp_path`` is cleaned up by
  pytest automatically.

The benchmark lives under ``benchmarks/`` which is not an importable package, so
(as in ``tests/kernels/moe/test_sm70_fused_moe_logging.py``) the module is loaded
directly from its file path via ``importlib.util``.

Validates: Requirements 7.1, 7.3
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Path to the benchmark script (tasks 7.1 / 7.2). ``benchmarks/`` is not an
# importable package, so the module is loaded directly from its file path.
_BENCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "kernels"
    / "benchmark_sm70_fused_moe.py"
)

# Columns produced by ``format_comparison_table`` (its ``headers`` list). The
# table is meaningless without these, so the smoke test asserts they are all
# present.
_EXPECTED_HEADER_COLUMNS = (
    "shape",
    "layout",
    "variant",
    "latency_ms",
    "tokens/s",
    "GFLOP/s",
    "speedup_vs_reference",
    "note",
)

# The archive filename encodes four comma-delimited fields, the last of which is
# a ``YYYYMMDD_HHMMSS`` timestamp, and the device field ends with ``_x<count>``:
#   <device>_x<count>,<repo>-<version>,<subject>,<YYYYMMDD_HHMMSS>.md
_ARCHIVE_NAME_RE = re.compile(
    r"^(?P<device>.+_x\d+),(?P<version>[^,]+),(?P<subject>[^,]+),"
    r"(?P<ts>\d{8}_\d{6})\.md$"
)


# --------------------------------------------------------------------------- #
# Module loading
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bench_mod() -> ModuleType:
    """Import the benchmark script from its file path (it is not a package)."""
    assert _BENCH_PATH.is_file(), f"benchmark script missing: {_BENCH_PATH}"
    mod_name = "benchmark_sm70_fused_moe"
    spec = importlib.util.spec_from_file_location(mod_name, _BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing so dataclass field resolution can find this
    # module while its ``@dataclass`` definitions are being built.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _assert_table_has_headers_and_a_row(table: str, *, fmt: str) -> None:
    """Common assertions: non-empty table, every header column, >=1 variant row."""
    assert isinstance(table, str)
    assert table.strip() != "", "comparison table is empty"
    for col in _EXPECTED_HEADER_COLUMNS:
        assert col in table, f"header column {col!r} missing from {fmt} table"
    # At least one variant row — every BenchmarkResult carries a ``reference``
    # baseline variant, so its name must appear in the rendered body.
    assert "reference" in table, f"no variant row found in {fmt} table"
    # The body must have more than just the 2 header/separator lines.
    assert len(table.splitlines()) >= 3, f"{fmt} table has no data rows"


# --------------------------------------------------------------------------- #
# R7.1: the benchmark runs and produces a non-empty comparison table
# --------------------------------------------------------------------------- #


def test_benchmark_runs_on_cpu_and_produces_table(bench_mod) -> None:
    """R7.1: ``SM70MoEBenchmark`` runs a tiny shape on CPU -> non-empty table.

    No V100 / built kernel needed: each variant degrades to an unavailable row
    with a note instead of aborting, so the rendered table still carries the
    header columns and at least one variant row.
    """
    bench = bench_mod.SM70MoEBenchmark(
        device="cpu",
        warmup=1,
        iters=1,
        fusion_levels=(bench_mod.SM70FusionLevel.L0, bench_mod.SM70FusionLevel.L1),
    )
    shape = bench_mod.MoEShape(M=8, K=64, I=64, E=4, topk=2, group_size=32)

    results = bench.run([shape])

    # The run produced exactly one result with variant rows (reference + levels).
    assert len(results) == 1
    res = results[0]
    assert res.shape == shape
    assert res.layout in ("contiguous", "masked")
    variant_names = {v.name for v in res.variants}
    assert "reference" in variant_names
    assert {"L0", "L1"} <= variant_names

    table = bench_mod.format_comparison_table(results, fmt="plain")
    _assert_table_has_headers_and_a_row(table, fmt="plain")
    # The benchmarked shape label is rendered into the table body.
    assert shape.label() in table


@pytest.mark.parametrize("fmt", ["plain", "markdown"])
def test_format_comparison_table_on_demo_results(bench_mod, fmt) -> None:
    """R7.1: ``format_comparison_table`` renders the synthetic demo rows.

    Deterministic, GPU-free assertions in both output formats: header columns,
    a variant row, and the speedup column (``Nx``) computed from the demo data.
    """
    demo = bench_mod._demo_results()
    assert demo, "demo results should be non-empty"

    table = bench_mod.format_comparison_table(demo, fmt=fmt)
    _assert_table_has_headers_and_a_row(table, fmt=fmt)

    # Every fused variant from the demo data shows up as a row.
    for name in ("L0", "L1", "L2"):
        assert name in table, f"variant {name!r} missing from {fmt} table"
    # A speedup over the baseline is rendered (e.g. ``1.61x``) for the demo data.
    assert re.search(r"\d+\.\d{2}x", table), f"no speedup value in {fmt} table"
    if fmt == "markdown":
        # Markdown tables use pipe-delimited rows.
        assert table.lstrip().startswith("|")


def test_self_test_cli_prints_table_and_exits_zero(bench_mod, capsys) -> None:
    """R7.1: the ``--self-test`` CLI prints the synthetic table and returns 0."""
    rc = bench_mod.main(["--self-test"])
    assert rc == 0

    out = capsys.readouterr().out
    _assert_table_has_headers_and_a_row(out, fmt="plain")


# --------------------------------------------------------------------------- #
# R7.3: archive_results writes to the 测试结果/ convention path + format
# --------------------------------------------------------------------------- #


def test_archive_results_writes_convention_path_with_table(bench_mod, tmp_path) -> None:
    """R7.3: ``archive_results`` writes a ``测试结果/``-style markdown artifact.

    Written into ``tmp_path`` (not the committed ``测试结果/``) with a pinned
    timestamp + device count so the filename pattern and content are asserted
    deterministically. Validates the returned path exists, matches the
    ``<device>_x<count>,<repo>-<version>,<subject>,<ts>.md`` scheme, lives under
    a ``<subject>/`` sub-directory, and contains the comparison table.
    """
    demo = bench_mod._demo_results()
    subject = "sm70-fused-moe"
    ts = "20990101_000000"

    out_path = bench_mod.archive_results(
        demo,
        out_dir=tmp_path,
        subject=subject,
        device_count=4,
        timestamp=ts,
    )

    # --- the returned path exists and lives under <tmp>/<subject>/ ---------- #
    assert isinstance(out_path, Path)
    assert out_path.is_file(), f"archive file not written: {out_path}"
    assert out_path.parent == tmp_path / subject
    # The artifact was written to the temp dir, NOT the committed 测试结果/ tree.
    assert tmp_path in out_path.parents

    # --- filename matches the device,version,subject,timestamp.md scheme ---- #
    m = _ARCHIVE_NAME_RE.match(out_path.name)
    assert m is not None, f"filename does not match convention: {out_path.name!r}"
    assert m.group("device").endswith("_x4"), out_path.name
    assert m.group("subject") == subject, out_path.name
    assert m.group("ts") == ts, out_path.name

    # --- the document body contains the comparison table -------------------- #
    content = out_path.read_text(encoding="utf-8")
    assert content.strip() != "", "archive document is empty"
    # Metadata header (device / repo / timestamp).
    assert "# SM70 Fused MoE Benchmark" in content
    assert ts in content
    assert "## Comparison" in content
    # The embedded comparison table (markdown) with its columns + a variant row.
    _assert_table_has_headers_and_a_row(content, fmt="markdown")
    for name in ("reference", "L0", "L1", "L2"):
        assert name in content, f"variant {name!r} missing from archived table"


def test_archive_via_self_test_cli(bench_mod, tmp_path, capsys) -> None:
    """R7.1 + R7.3: ``--self-test --archive`` archives the synthetic table.

    Exercises the CLI's archive wiring end to end without a GPU, writing into
    ``tmp_path`` so the committed ``测试结果/`` tree is untouched.
    """
    rc = bench_mod.main(
        ["--self-test", "--archive", "--archive-dir", str(tmp_path)]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Archived comparison table to:" in out

    # Exactly one .md artifact landed under <tmp>/<subject>/.
    archived = list(tmp_path.rglob("*.md"))
    assert len(archived) == 1, f"expected one archived file, got {archived}"
    written = archived[0]
    assert _ARCHIVE_NAME_RE.match(written.name) is not None, written.name
    content = written.read_text(encoding="utf-8")
    _assert_table_has_headers_and_a_row(content, fmt="markdown")
