#!/usr/bin/env python3
"""Summarize 010 foot end-load reward curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def _read_curve(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {key: (float(value) if key not in {"step"} else int(value)) for key, value in row.items()}
            for row in reader
        ]
        return list(reader.fieldnames or []), rows


def _motion_name(path: Path) -> str:
    stem = path.name.removesuffix("_reward_curve.csv")
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_curve(path: Path, foot_weight: float) -> dict[str, float | str | int]:
    headers, rows = _read_curve(path)
    if "foot_end_load" not in headers:
        raise ValueError(f"{path} has no foot_end_load column")
    if len(rows) >= 2:
        dts = [rows[i]["time_s"] - rows[i - 1]["time_s"] for i in range(1, len(rows))]
        dt = median(dts)
    else:
        dt = rows[0]["time_s"] if rows else 0.02

    weighted = [float(row["foot_end_load"]) for row in rows]
    raw = [value / (foot_weight * dt) if dt > 0.0 else 0.0 for value in weighted]
    active_raw = [value for value in raw if value > 1.0e-9]
    total = [float(row["total_reward"]) for row in rows]

    return {
        "motion": _motion_name(path),
        "num_steps": len(rows),
        "dt": dt,
        "trigger_fraction": len(active_raw) / len(rows) if rows else 0.0,
        "mean_raw_penalty": _mean(raw),
        "mean_raw_penalty_when_active": _mean(active_raw),
        "max_raw_penalty": max(raw) if raw else 0.0,
        "mean_weighted_foot_end_load": _mean(weighted),
        "mean_abs_weighted_foot_end_load": _mean([abs(value) for value in weighted]),
        "max_abs_weighted_foot_end_load": max((abs(value) for value in weighted), default=0.0),
        "mean_total_reward": _mean(total),
    }


def _term_aggregate(curves: list[Path]) -> list[dict[str, float | str]]:
    sums: dict[str, float] = {}
    signed_sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for path in curves:
        headers, rows = _read_curve(path)
        for term in headers:
            if term in {"step", "time_s"}:
                continue
            for row in rows:
                value = float(row[term])
                sums[term] = sums.get(term, 0.0) + abs(value)
                signed_sums[term] = signed_sums.get(term, 0.0) + value
                counts[term] = counts.get(term, 0) + 1
    return [
        {
            "term": term,
            "mean_abs": sums[term] / counts[term],
            "mean_signed": signed_sums[term] / counts[term],
        }
        for term in sorted(sums, key=lambda name: sums[name] / counts[name], reverse=True)
    ]


def _write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-curve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foot-weight", type=float, default=-0.5)
    args = parser.parse_args()

    curves = sorted(args.reward_curve_dir.glob("*_reward_curve.csv"))
    if not curves:
        raise SystemExit(f"no reward curves found in {args.reward_curve_dir}")

    per_clip = [_summarize_curve(path, args.foot_weight) for path in curves]
    term_summary = _term_aggregate(curves)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "foot_end_load_per_clip_summary.csv", per_clip)
    _write_csv(args.output_dir / "reward_term_magnitude_summary.csv", term_summary)
    with (args.output_dir / "foot_end_load_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_clip": per_clip, "term_summary": term_summary}, f, indent=2)

    print(f"curves={len(curves)}")
    print(f"per_clip_csv={args.output_dir / 'foot_end_load_per_clip_summary.csv'}")
    print(f"term_csv={args.output_dir / 'reward_term_magnitude_summary.csv'}")


if __name__ == "__main__":
    main()
