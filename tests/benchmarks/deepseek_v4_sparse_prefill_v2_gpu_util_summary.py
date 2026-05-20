#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_util_csv")
    parser.add_argument("--active-threshold", type=float, default=10.0)
    args = parser.parse_args()

    rows_by_gpu: dict[str, list[float]] = defaultdict(list)
    mem_by_gpu: dict[str, list[float]] = defaultdict(list)
    power_by_gpu: dict[str, list[float]] = defaultdict(list)
    with open(args.gpu_util_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gpu = row["index"].strip()
            util = _to_float(row["util_gpu_pct"])
            mem = _to_float(row["mem_used_mib"])
            power = _to_float(row["power_w"])
            if util is not None:
                rows_by_gpu[gpu].append(util)
            if mem is not None:
                mem_by_gpu[gpu].append(mem)
            if power is not None:
                power_by_gpu[gpu].append(power)

    summary = []
    for gpu in sorted(rows_by_gpu, key=lambda item: int(item) if item.isdigit() else item):
        utils = rows_by_gpu[gpu]
        if not utils:
            continue
        active = [value for value in utils if value >= args.active_threshold]
        mems = mem_by_gpu.get(gpu, [])
        powers = power_by_gpu.get(gpu, [])
        summary.append(
            {
                "gpu": gpu,
                "samples": len(utils),
                "avg_util_pct": sum(utils) / len(utils),
                "max_util_pct": max(utils),
                "active_sample_ratio": len(active) / len(utils),
                "avg_mem_used_mib": sum(mems) / len(mems) if mems else None,
                "max_mem_used_mib": max(mems) if mems else None,
                "avg_power_w": sum(powers) / len(powers) if powers else None,
                "max_power_w": max(powers) if powers else None,
            }
        )

    all_utils = [value for values in rows_by_gpu.values() for value in values]
    payload = {
        "gpu_count": len(summary),
        "active_threshold": args.active_threshold,
        "overall_avg_util_pct": (
            sum(all_utils) / len(all_utils) if all_utils else None
        ),
        "per_gpu": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
