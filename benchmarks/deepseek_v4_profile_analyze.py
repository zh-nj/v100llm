#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline analyzer for DeepSeek V4 Flash profiler JSONL traces.

Stdlib-only. Consumes one or more JSONL traces produced by
`_DeepseekV4PhaseProfiler` (see vllm/model_executor/layers/deepseek_v4_attention.py)
and emits:

  * summary_decode.md / summary_prefill.md  — top-5 bottleneck tables
  * roofline.csv                            — operation roofline classification
  * theoretical_peak.md                     — V100 TP=8 peak derivation
  * diff_vs_baseline.md                     — flagged regressions (if --baseline)
  * optimization_plan.md                    — per-bottleneck scaffold

Usage
-----
    python deepseek_v4_profile_analyze.py \
        --raw-trace /path/to/trace1.jsonl \
        --raw-trace /path/to/trace2.jsonl \
        --byte-budget /path/to/byte_budget.json \
        --baseline /path/to/baseline.jsonl \
        --output-dir /tmp/report \
        --context-lens 1024,3072 \
        --phase decode
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Hardware constants (V100 SXM2 32GB, TP=8)
# ---------------------------------------------------------------------------
V100_FP16_TFLOPS = 125.0
V100_HBM_GBPS = 900.0
V100_RIDGE_FLOPS_PER_BYTE = (V100_FP16_TFLOPS * 1e12) / (V100_HBM_GBPS * 1e9)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_traces(paths: Iterable[str]) -> list[dict[str, Any]]:
    """Load JSONL trace files into a flat list of events.

    Tolerates malformed lines (skipped with stderr warning) and missing
    optional fields.
    """
    events: list[dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"WARNING: trace file not found: {p}", file=sys.stderr)
            continue
        with open(p) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: {p}:{ln} bad JSON: {e}", file=sys.stderr)
                    continue
                if "phase" not in ev or "elapsed_us" not in ev:
                    continue
                events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
class AggStats:
    __slots__ = ("count", "total_us", "_samples")

    def __init__(self) -> None:
        self.count = 0
        self.total_us = 0.0
        self._samples: list[float] = []

    def add(self, us: float) -> None:
        self.count += 1
        self.total_us += float(us)
        self._samples.append(float(us))

    @property
    def mean_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0

    def quantile(self, q: float) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[idx]

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "total_us": self.total_us,
            "mean_us": self.mean_us,
            "p50_us": self.quantile(0.5),
            "p90_us": self.quantile(0.9),
            "p99_us": self.quantile(0.99),
        }


def aggregate_by_rank(
    events: list[dict[str, Any]],
    *, phase_filter: str | None = None,
) -> dict[tuple[int | None, str], AggStats]:
    """Group events by `(tp_rank, phase)` and accumulate stats.

    `phase_filter` selects events whose `step_kind` matches ('decode',
    'prefill'). When None, both kinds are included.
    """
    out: dict[tuple[int | None, str], AggStats] = defaultdict(AggStats)
    for ev in events:
        if phase_filter is not None and ev.get("step_kind") not in (None, phase_filter):
            continue
        rank = ev.get("tp_rank")
        out[(rank, ev["phase"])].add(ev["elapsed_us"])
    return out


def pick_max_rank(
    agg: dict[tuple[int | None, str], AggStats],
) -> dict[str, AggStats]:
    """Reduce per-rank stats to the slowest rank per label (wall time)."""
    by_label: dict[str, AggStats] = {}
    for (_rank, label), stats in agg.items():
        cur = by_label.get(label)
        if cur is None or stats.total_us > cur.total_us:
            by_label[label] = stats
    return by_label


def top_n(agg: dict[str, AggStats], n: int = 5) -> list[tuple[str, AggStats]]:
    items = sorted(agg.items(), key=lambda kv: kv[1].total_us, reverse=True)
    return items[: min(n, len(items))]


# ---------------------------------------------------------------------------
# Roofline classification
# ---------------------------------------------------------------------------
def classify_roofline(
    agg: dict[str, AggStats],
    byte_budget: dict[str, dict[str, float]],
    *,
    bw_gbps: float = V100_HBM_GBPS,
    tflops_peak: float = V100_FP16_TFLOPS,
    num_layers: int = 43,
    num_steps: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ridge = (tflops_peak * 1e12) / (bw_gbps * 1e9)
    for label, stats in agg.items():
        bb = byte_budget.get(label)
        if not bb:
            rows.append({
                "operation": label,
                "bytes_per_step": 0,
                "flops_per_step": 0,
                "flops_per_byte": 0.0,
                "achieved_gbps": 0.0,
                "achieved_tflops": 0.0,
                "peak_util_pct": 0.0,
                "bound": "unknown",
            })
            continue
        bytes_total = (bb["bytes_per_token_per_layer_hbm_read"]
                       + bb["bytes_per_token_per_layer_hbm_write"]) * num_layers * num_steps
        flops_total = bb["flops_per_token_per_layer"] * num_layers * num_steps
        ai = bb["arithmetic_intensity_flops_per_byte"]
        time_s = stats.total_us / 1e6 if stats.total_us else 1e-9
        achieved_gbps = (bytes_total / time_s) / 1e9
        achieved_tflops = (flops_total / time_s) / 1e12
        if ai > ridge:
            bound = "compute"
            util = 100.0 * achieved_tflops / tflops_peak
        else:
            bound = "memory"
            util = 100.0 * achieved_gbps / bw_gbps
        rows.append({
            "operation": label,
            "bytes_per_step": bytes_total,
            "flops_per_step": flops_total,
            "flops_per_byte": ai,
            "achieved_gbps": achieved_gbps,
            "achieved_tflops": achieved_tflops,
            "peak_util_pct": util,
            "bound": bound,
        })
    rows.sort(key=lambda r: r["operation"])
    return rows


def write_roofline_csv(rows: list[dict[str, Any]], path: str) -> None:
    cols = ["operation", "bytes_per_step", "flops_per_step", "flops_per_byte",
            "achieved_gbps", "achieved_tflops", "peak_util_pct", "bound"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Theoretical peak
# ---------------------------------------------------------------------------
def theoretical_peak(
    model_config: dict[str, Any],
    *,
    bw_gbps: float = V100_HBM_GBPS,
    tflops_peak: float = V100_FP16_TFLOPS,
    tp: int = 8,
    measured_tok_s: float | None = None,
) -> dict[str, float]:
    """Estimate the per-token decode floor on V100 TP=8.

    Uses the byte/FLOP totals derived directly from model config.
    """
    H = int(model_config.get("hidden_size", 4096))
    n_layers = int(model_config.get("num_hidden_layers", 43))
    moe_dim = int(model_config.get("moe_intermediate_size", 2048))
    experts_per_tok = int(model_config.get("num_experts_per_tok", 6))
    n_shared = int(model_config.get("n_shared_experts", 1))
    o_lora = int(model_config.get("o_lora_rank", 1024))
    q_lora = int(model_config.get("q_lora_rank", 1024))

    # Bytes per token per layer per GPU (FP8 weights, 3x SM70 dequant overhead)
    fp8_w = 1.0 * 3.0
    fp4_w = 0.5 * 5.0
    attn_proj_elts = (H * (q_lora + o_lora)) + (q_lora * H) + (o_lora * H)
    attn_bytes = attn_proj_elts * fp8_w / tp
    moe_active = experts_per_tok + n_shared
    moe_elts = moe_active * (3 * H * moe_dim)
    moe_bytes = moe_elts * fp4_w / tp

    bytes_per_token = (attn_bytes + moe_bytes) * n_layers
    flops_per_token = (2 * attn_proj_elts / tp
                       + moe_active * 6 * H * moe_dim / tp) * n_layers

    t_mem = bytes_per_token / (bw_gbps * 1e9)
    t_cmp = flops_per_token / (tflops_peak * 1e12)
    t_floor = max(t_mem, t_cmp)
    peak_tok_s = 1.0 / t_floor if t_floor else float("inf")

    out = {
        "bytes_per_token": bytes_per_token,
        "flops_per_token": flops_per_token,
        "t_mem_s": t_mem,
        "t_compute_s": t_cmp,
        "t_floor_s": t_floor,
        "peak_tok_s": peak_tok_s,
        "bound": "memory" if t_mem >= t_cmp else "compute",
    }
    if measured_tok_s is not None:
        out["measured_tok_s"] = measured_tok_s
        out["util_pct"] = 100.0 * measured_tok_s / peak_tok_s
    return out


# ---------------------------------------------------------------------------
# Baseline diff
# ---------------------------------------------------------------------------
def diff_vs_baseline(
    current: dict[str, AggStats],
    baseline: dict[str, AggStats],
    *, threshold: float = 0.10,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = set(current) | set(baseline)
    for k in sorted(keys):
        c = current.get(k).total_us if k in current else 0.0
        b = baseline.get(k).total_us if k in baseline else 0.0
        if b == 0:
            delta_pct = float("inf") if c else 0.0
        else:
            delta_pct = (c - b) / b
        flagged = abs(delta_pct) > threshold if b else (c > 0)
        out.append({
            "operation": k,
            "baseline_us": b,
            "current_us": c,
            "delta_pct": delta_pct * 100.0,
            "flagged": flagged,
            "regressed": c > b,
        })
    return out


def classify_growth(t_ref: float, t_cur: float,
                    n_ref: float, n_cur: float,
                    eps: float = 0.10) -> str:
    """Classify latency growth vs token-count growth."""
    if t_ref <= 0 or n_ref <= 0:
        return "unknown"
    lat_ratio = t_cur / t_ref
    tok_ratio = n_cur / n_ref
    if lat_ratio < tok_ratio * (1 - eps):
        return "sub"
    if lat_ratio > tok_ratio * (1 + eps):
        return "super"
    return "linear"


# ---------------------------------------------------------------------------
# Task-DAG utilities
# ---------------------------------------------------------------------------
def detect_cycle(graph: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def dfs(n: str) -> bool:
        color[n] = GRAY
        for m in graph.get(n, []):
            if color.get(m, WHITE) == GRAY:
                return True
            if color.get(m, WHITE) == WHITE and dfs(m):
                return True
        color[n] = BLACK
        return False
    return any(color[n] == WHITE and dfs(n) for n in graph)


def topological_groups(graph: dict[str, list[str]]) -> list[list[str]]:
    """Kahn's algorithm with parallel-group output."""
    indeg: dict[str, int] = {n: 0 for n in graph}
    for n, deps in graph.items():
        for d in deps:
            indeg[n] = indeg.get(n, 0) + 1
            indeg.setdefault(d, 0)
    # nodes with indeg==0 form the first group; iterate.
    rev: dict[str, list[str]] = {n: [] for n in indeg}
    for n, deps in graph.items():
        for d in deps:
            rev.setdefault(d, []).append(n)
    groups: list[list[str]] = []
    remaining = dict(indeg)
    while remaining:
        layer = sorted([n for n, d in remaining.items() if d == 0])
        if not layer:
            raise ValueError("cycle detected during topological sort")
        groups.append(layer)
        for n in layer:
            for m in rev.get(n, []):
                if m in remaining:
                    remaining[m] -= 1
            remaining.pop(n)
    return groups


def critical_path(graph: dict[str, list[str]],
                  effort: dict[str, float]) -> tuple[list[str], float]:
    """Longest path by effort over a DAG. `graph[n]` is n's prerequisites."""
    memo: dict[str, tuple[float, list[str]]] = {}

    def best(n: str) -> tuple[float, list[str]]:
        if n in memo:
            return memo[n]
        deps = graph.get(n, [])
        if not deps:
            memo[n] = (effort.get(n, 0.0), [n])
            return memo[n]
        best_dep = max((best(d) for d in deps), key=lambda x: x[0])
        memo[n] = (best_dep[0] + effort.get(n, 0.0), best_dep[1] + [n])
        return memo[n]

    if not graph:
        return [], 0.0
    overall = max((best(n) for n in graph), key=lambda x: x[0])
    return overall[1], overall[0]


# ---------------------------------------------------------------------------
# nsys stats CSV merge (Task 7.2)
# ---------------------------------------------------------------------------
def parse_nsys_stats_csv(path: str) -> list[dict[str, str]]:
    """Parse an `nsys stats --format csv` output file (cuda_gpu_kern_sum
    or similar). Returns a list of row dicts. Tolerates header offsets
    by skipping leading blank/comment lines.
    """
    rows: list[dict[str, str]] = []
    if not os.path.exists(path):
        print(f"WARNING: nsys stats CSV not found: {path}", file=sys.stderr)
        return rows
    with open(path, newline="") as f:
        # Skip header banner lines until we find a comma-bearing header.
        lines = [ln for ln in f if ln.strip()]
    start = 0
    for i, ln in enumerate(lines):
        if "," in ln and any(tok.strip().lower() in ln.lower()
                              for tok in ("Time", "Name", "Kernel")):
            start = i
            break
    reader = csv.DictReader(lines[start:])
    for r in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in r.items() if k})
    return rows


def render_nsys_section(rows: list[dict[str, str]], n: int = 5) -> str:
    if not rows:
        return ""
    # Try common column names from `nsys stats -r cuda_gpu_kern_sum`.
    name_keys = ("Name", "Kernel Name", "Kernel")
    time_keys = ("Total Time (ns)", "Total Time", "Time (ns)", "Total Time (s)")
    pct_keys = ("Time (%)", "Time(%)", "Percent")
    name_k = next((k for k in name_keys if rows[0].get(k) is not None), None)
    time_k = next((k for k in time_keys if rows[0].get(k) is not None), None)
    pct_k = next((k for k in pct_keys if rows[0].get(k) is not None), None)
    if not name_k:
        return ""

    def _f(x: str) -> float:
        try:
            return float(str(x).replace(",", ""))
        except ValueError:
            return 0.0

    if time_k:
        ranked = sorted(rows, key=lambda r: _f(r.get(time_k, "0")), reverse=True)
    else:
        ranked = rows
    top = ranked[:n]

    lines = [
        "## Nsight Systems Top Kernels",
        "",
        f"| rank | kernel | {time_k or 'time'} | {pct_k or '%'} |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(top, 1):
        lines.append(f"| {i} | `{r.get(name_k, '')[:80]}` | "
                     f"{r.get(time_k, '')} | {r.get(pct_k, '')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _fmt_us(us: float) -> str:
    return f"{us / 1000.0:.3f}" if us else "0.000"


def render_report(
    *,
    phase_kind: str,
    by_label: dict[str, AggStats],
    context_lens: list[int],
    meta: dict[str, Any] | None,
    nsys_section: str = "",
) -> str:
    total_us = sum(s.total_us for s in by_label.values()) or 1.0
    top = top_n(by_label, 5)

    out: list[str] = []
    out.append(f"# DeepSeek V4 Flash {phase_kind.title()} Profile Summary")
    out.append("")
    if meta:
        out.append("## Run Metadata")
        out.append("")
        for k in ("git_commit", "git_branch", "torch_version", "cuda_version",
                  "model_path", "timestamp_start_utc"):
            if meta.get(k):
                out.append(f"- **{k}**: `{meta[k]}`")
        out.append("")

    out.append("## Top-5 Bottlenecks (max-rank wall time)")
    out.append("")
    ctx_cols = " | ".join(f"ctx {c}" for c in context_lens)
    out.append("| rank | operation | ms/step (max_rank) | % of step | bound | "
               + ctx_cols + " | Δ 1K→3K |")
    out.append("|---|---|---|---|---|" + "---|" * (len(context_lens) + 1))
    for i, (label, stats) in enumerate(top, 1):
        pct = 100.0 * stats.total_us / total_us
        critical = " ⚠️ critical" if pct > 50 else ""
        ctx_vals = " | ".join("—" for _ in context_lens)
        out.append(f"| {i} | `{label}` | {_fmt_us(stats.total_us)} | "
                   f"{pct:.1f}%{critical} | — | {ctx_vals} | — |")
    out.append("")
    out.append(f"_Total measured: {_fmt_us(total_us)} ms across "
               f"{len(by_label)} labels._")
    out.append("")
    if nsys_section:
        out.append(nsys_section)
    return "\n".join(out)


def render_optimization_plan(top_decode: list[tuple[str, AggStats]],
                             top_prefill: list[tuple[str, AggStats]]) -> str:
    out: list[str] = []
    out.append("# DeepSeek V4 Flash Optimization Plan (Scaffold)")
    out.append("")
    out.append("Generated by `deepseek_v4_profile_analyze.py`. Each section "
               "below corresponds to one top-N bottleneck. Populate the "
               "placeholders manually with concrete approaches, complexity, "
               "dependencies, verification, regression risks, and estimated "
               "speedup derived from the measured percentages.")
    out.append("")

    def _section(header: str, items: list[tuple[str, AggStats]]) -> None:
        out.append(f"## {header}")
        out.append("")
        if not items:
            out.append("_(no events recorded)_\n")
            return
        total = sum(s.total_us for _, s in items) or 1.0
        for i, (label, stats) in enumerate(items, 1):
            pct = 100.0 * stats.total_us / total
            out.append(f"### {i}. `{label}` ({pct:.1f}% of phase)")
            out.append("")
            out.append("- **Approaches**: TODO (list concrete optimizations).")
            out.append("- **Complexity**: TODO (Low / Medium / High).")
            out.append("- **Dependencies**: TODO (other optimizations or "
                       "prereqs).")
            out.append("- **Verification**: TODO "
                       "(`semantic_canary_zx42` / `numeric_ab` / "
                       "`throughput_benchmark` / `phase_delta_check`).")
            out.append("- **Target source files**: TODO.")
            out.append("- **Regression risks**: TODO.")
            out.append("- **Estimated speedup range**: TODO (use "
                       f"{pct:.1f}% × (1 - new/old) formula).")
            out.append("")
            out.append("#### Tasks")
            out.append("")
            out.append("- [ ] description: TODO; effort: Xh; "
                       "prerequisites: []; verification_criteria: TODO; "
                       "target_files: [].")
            out.append("- [ ] phase_delta_check after change "
                       "(20% tolerance vs predicted Δ).")
            out.append("")

    _section("Decode top-5", top_decode)
    _section("Prefill top-5", top_prefill)
    return "\n".join(out)


def render_theoretical_peak_md(peak: dict[str, float]) -> str:
    return (
        "# Theoretical Peak (V100 TP=8)\n\n"
        "Derivation per design §Theoretical peak decode throughput.\n\n"
        f"- **bytes_per_token**: {peak['bytes_per_token']:.3e} B\n"
        f"- **flops_per_token**: {peak['flops_per_token']:.3e}\n"
        f"- **t_mem (HBM-bound floor)**: {peak['t_mem_s']*1e3:.3f} ms\n"
        f"- **t_compute (FLOP-bound floor)**: {peak['t_compute_s']*1e3:.3f} ms\n"
        f"- **t_floor**: {peak['t_floor_s']*1e3:.3f} ms\n"
        f"- **peak_tok_s**: {peak['peak_tok_s']:.1f} tok/s\n"
        f"- **bound**: {peak['bound']}\n"
        + (f"- **measured_tok_s**: {peak['measured_tok_s']:.2f}\n"
           f"- **util_pct**: {peak['util_pct']:.2f}%\n"
           if "measured_tok_s" in peak else "")
    )


def render_diff_md(diffs: list[dict[str, Any]]) -> str:
    lines = ["# Diff vs Baseline\n",
             "| operation | baseline ms | current ms | Δ% | flag | reg |",
             "|---|---|---|---|---|---|"]
    for d in diffs:
        flag = "⚠️" if d["flagged"] else ""
        reg = "↑" if d["regressed"] else "↓"
        lines.append(f"| `{d['operation']}` | {d['baseline_us']/1000:.3f} | "
                     f"{d['current_us']/1000:.3f} | "
                     f"{d['delta_pct']:+.1f}% | {flag} | {reg} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_meta(raw_paths: list[str]) -> dict[str, Any] | None:
    for p in raw_paths:
        meta = p + ".meta.json"
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-trace", action="append", default=[],
                    help="Path to a JSONL trace (may be repeated).")
    ap.add_argument("--byte-budget", default=None,
                    help="Path to byte_budget.json from "
                         "deepseek_v4_byte_budget.py.")
    ap.add_argument("--baseline", default=None,
                    help="Path to baseline JSONL trace for diff.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--context-lens", default="1024,3072",
                    help="Comma-separated context lengths for table cols.")
    ap.add_argument("--phase", default="both",
                    choices=("decode", "prefill", "both"))
    ap.add_argument("--nsys-stats-csv", default=None,
                    help="Optional nsys stats CSV "
                         "(cuda_gpu_kern_sum) for kernel attribution.")
    args = ap.parse_args(argv)

    if not args.raw_trace:
        print("ERROR: at least one --raw-trace is required", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    ctx_lens = [int(x) for x in args.context_lens.split(",") if x.strip()]
    events = load_traces(args.raw_trace)
    if not events:
        print("WARNING: no events recorded", file=sys.stderr)

    byte_budget: dict[str, dict[str, float]] = {}
    model_config: dict[str, Any] = {}
    if args.byte_budget and os.path.exists(args.byte_budget):
        with open(args.byte_budget) as f:
            bb_payload = json.load(f)
        byte_budget = bb_payload.get("budgets", {})
        model_config = {
            "num_hidden_layers": bb_payload.get("num_hidden_layers", 43),
        }

    meta = _load_meta(args.raw_trace)
    if meta and meta.get("model_config"):
        model_config.update(meta["model_config"])

    nsys_rows = (parse_nsys_stats_csv(args.nsys_stats_csv)
                 if args.nsys_stats_csv else [])
    nsys_section = render_nsys_section(nsys_rows) if nsys_rows else ""

    phases = (["decode", "prefill"] if args.phase == "both"
              else [args.phase])
    for phase in phases:
        agg = aggregate_by_rank(events, phase_filter=phase)
        by_label = pick_max_rank(agg)
        report = render_report(
            phase_kind=phase, by_label=by_label,
            context_lens=ctx_lens, meta=meta,
            nsys_section=(nsys_section if phase == "decode" else ""),
        )
        with open(os.path.join(args.output_dir,
                               f"summary_{phase}.md"), "w") as f:
            f.write(report)

    # Roofline (derived from "both")
    agg_all = pick_max_rank(aggregate_by_rank(events))
    if byte_budget:
        rl = classify_roofline(
            agg_all, byte_budget,
            num_layers=int(model_config.get("num_hidden_layers", 43)))
        write_roofline_csv(rl, os.path.join(args.output_dir, "roofline.csv"))

    # Theoretical peak
    if model_config:
        peak = theoretical_peak(model_config)
        with open(os.path.join(args.output_dir,
                               "theoretical_peak.md"), "w") as f:
            f.write(render_theoretical_peak_md(peak))

    # Baseline diff
    if args.baseline:
        base_events = load_traces([args.baseline])
        base_agg = pick_max_rank(aggregate_by_rank(base_events))
        diffs = diff_vs_baseline(agg_all, base_agg)
        with open(os.path.join(args.output_dir,
                               "diff_vs_baseline.md"), "w") as f:
            f.write(render_diff_md(diffs))

    # Optimization plan scaffold
    decode_top = top_n(pick_max_rank(
        aggregate_by_rank(events, phase_filter="decode")), 5)
    prefill_top = top_n(pick_max_rank(
        aggregate_by_rank(events, phase_filter="prefill")), 5)
    with open(os.path.join(args.output_dir,
                           "optimization_plan.md"), "w") as f:
        f.write(render_optimization_plan(decode_top, prefill_top))

    print(f"OK: wrote reports under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
