#!/usr/bin/env python3
"""Summarize A3 motor T-N reward curves from IsaacLab evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gear_sonic.utils.a3_motor_model import get_tn_curve

TN_DIAGNOSTIC_COLUMNS = {
    "serial_soft": "motor_tn_serial_soft_diag",
    "serial_hard": "motor_tn_serial_hard_diag",
    "parallel_soft": "motor_tn_parallel_soft_diag",
    "parallel_hard": "motor_tn_parallel_hard_diag",
}
CONSTRAINT_DIAGNOSTIC_COLUMNS = {
    "ankle_motor_speed": "ankle_motor_speed_diag",
    "ankle_motor_power": "ankle_motor_power_diag",
    "ankle_motor_fixed_torque": "ankle_motor_fixed_torque_diag",
    "parallel_ankle_cross": "parallel_ankle_cross_diag",
    "parallel_waist_cross": "parallel_waist_cross_diag",
    "ankle_position_85": "ankle_position_85_diag",
}
DIAGNOSTIC_COLUMNS = TN_DIAGNOSTIC_COLUMNS | CONSTRAINT_DIAGNOSTIC_COLUMNS
CANDIDATE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
CURRENT_REFERENCE_CURVES = {
    "PFP41": (
        (0.0, 20.943951, 26.179939),
        (6.0, 6.0, 0.0),
    ),
    "PFP59": (
        (0.0, 13.613568, 14.660766, 15.707963, 16.755161, 20.943951),
        (34.8, 34.8, 24.053, 20.918, 17.073, 0.0),
    ),
    "PFP78": (
        (0.0, 16.755161, 17.802358, 18.849556, 23.561945),
        (60.0, 60.0, 50.104, 44.310, 0.0),
    ),
    "PFP93": (
        (
            0.0,
            10.471976,
            11.519173,
            12.566371,
            13.613568,
            14.660766,
            15.707963,
            16.755161,
            20.943951,
        ),
        (220.0, 220.0, 193.135, 160.961, 133.026, 104.350, 75.010, 54.401, 0.0),
    ),
    "PFP110": (
        (
            0.0,
            9.424778,
            10.471976,
            11.519173,
            12.566371,
            13.613568,
            14.660766,
            15.707963,
            19.634954,
        ),
        (305.0, 305.0, 264.297, 224.922, 182.896, 143.348, 105.184, 62.183, 0.0),
    ),
}


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("reward curves contain non-finite values")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _read_reward_curve(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        columns = list(reader.fieldnames)
        rows = [[float(row[name]) for name in columns] for row in reader]
    if not rows:
        raise ValueError(f"{path}: reward curve is empty")
    return columns, np.asarray(rows, dtype=np.float64)


def _motion_key_map(metrics: dict[str, Any]) -> dict[str, str]:
    mapping = {}
    for item in metrics.get("reward_curves", []):
        csv_path = item.get("csv")
        motion_key = item.get("motion_key")
        if csv_path and motion_key:
            mapping[Path(csv_path).name] = str(motion_key)
    return mapping


def _tracking_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    all_metrics = metrics["eval/all_metrics_dict"]
    terminated = np.asarray(all_metrics["terminated"], dtype=bool)
    return {
        "attempted": int(terminated.size),
        "terminated": int(terminated.sum()),
        "success_rate": float(metrics["eval/success/success_rate"]),
        "progress_rate": float(metrics["eval/success/progress_rate"]),
        "mpjpe_l_mm": float(metrics["eval/success/mpjpe_l"]),
        "mpjpe_g_mm": float(metrics["eval/success/mpjpe_g"]),
        "mpjpe_pa_mm": float(metrics["eval/success/mpjpe_pa"]),
    }


def summarize_dataset(dataset_dir: Path, dataset_name: str) -> dict[str, Any]:
    """Aggregate per-motion reward curves for one Lab Eval dataset."""

    dataset_dir = Path(dataset_dir)
    metrics_path = dataset_dir / "metrics" / "metrics_eval.json"
    curve_dir = dataset_dir / "metrics" / "reward_curves"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    motion_names = _motion_key_map(metrics)
    termination_by_key = dict(
        zip(
            map(str, metrics["eval/all_metrics_dict"]["motion_keys"]),
            map(bool, metrics["eval/all_metrics_dict"]["terminated"]),
            strict=True,
        )
    )
    paths = sorted(curve_dir.glob("*_reward_curve.csv"))
    if not paths:
        raise FileNotFoundError(f"no reward curves found under {curve_dir}")

    aggregate = {
        "serial_soft": [],
        "serial_hard": [],
        "serial": [],
        "parallel_soft": [],
        "parallel_hard": [],
        "parallel": [],
        "baseline_positive": [],
        "baseline_negative": [],
    }
    aggregate.update({key: [] for key in CONSTRAINT_DIAGNOSTIC_COLUMNS})
    motion_rows = []
    expected_columns: list[str] | None = None
    available_constraint_keys: list[str] | None = None
    step_dt: float | None = None
    for path in paths:
        columns, values = _read_reward_curve(path)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"{path}: reward columns differ from the first curve")
        index = {name: idx for idx, name in enumerate(columns)}
        missing = set(TN_DIAGNOSTIC_COLUMNS.values()).difference(index)
        if missing:
            raise ValueError(f"{path}: missing diagnostic columns {sorted(missing)}")
        curve_constraint_keys = [
            key for key, column in CONSTRAINT_DIAGNOSTIC_COLUMNS.items() if column in index
        ]
        if available_constraint_keys is None:
            available_constraint_keys = curve_constraint_keys
        elif curve_constraint_keys != available_constraint_keys:
            raise ValueError(f"{path}: optional constraint columns differ from the first curve")

        time_values = values[:, index["time_s"]]
        curve_dt = float(time_values[0])
        if curve_dt <= 0.0:
            raise ValueError(f"{path}: invalid first time value {curve_dt}")
        if step_dt is None:
            step_dt = curve_dt
        elif not np.isclose(step_dt, curve_dt, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{path}: step_dt={curve_dt} differs from {step_dt}")

        per_step = {
            key: -values[:, index[column]] / curve_dt
            for key, column in TN_DIAGNOSTIC_COLUMNS.items()
        }
        for key in available_constraint_keys:
            column = CONSTRAINT_DIAGNOSTIC_COLUMNS[key]
            per_step[key] = -values[:, index[column]] / curve_dt
        per_step["serial"] = per_step["serial_soft"] + per_step["serial_hard"]
        per_step["parallel"] = per_step["parallel_soft"] + per_step["parallel_hard"]

        ordinary_columns = [
            name
            for name in columns
            if name not in {"step", "time_s", "total_reward"} and name not in DIAGNOSTIC_COLUMNS.values()
        ]
        ordinary = values[:, [index[name] for name in ordinary_columns]] / curve_dt
        baseline_positive = np.clip(ordinary, 0.0, None).sum(axis=1)
        baseline_negative = -np.clip(ordinary, None, 0.0).sum(axis=1)

        for key in (
            "serial_soft",
            "serial_hard",
            "serial",
            "parallel_soft",
            "parallel_hard",
            "parallel",
        ):
            aggregate[key].append(per_step[key])
        for key in available_constraint_keys:
            aggregate[key].append(per_step[key])
        aggregate["baseline_positive"].append(baseline_positive)
        aggregate["baseline_negative"].append(baseline_negative)

        motion_key = motion_names.get(path.name)
        if motion_key is None:
            stem = path.name.removesuffix("_reward_curve.csv")
            motion_key = stem.split("_", 1)[1] if "_" in stem else stem
        motor_tn = per_step["serial"] + per_step["parallel"]
        constraint_total = sum(
            (per_step[key] for key in available_constraint_keys),
            start=np.zeros_like(motor_tn),
        )
        combined = motor_tn + constraint_total
        motion_rows.append(
            {
                "motion_key": motion_key,
                "terminated": bool(termination_by_key.get(motion_key, False)),
                "frames": int(combined.size),
                "serial_mean_per_step": float(np.mean(per_step["serial"])),
                "parallel_mean_per_step": float(np.mean(per_step["parallel"])),
                "constraint_mean_per_step": float(np.mean(constraint_total)),
                "combined_mean_per_step": float(np.mean(combined)),
                "combined_integral": float(np.sum(combined) * curve_dt),
            }
        )

    arrays = {key: np.concatenate(parts) for key, parts in aggregate.items() if parts}
    serial = arrays["serial"]
    parallel = arrays["parallel"]
    motor_tn = serial + parallel
    constraint_total = sum(
        (arrays[key] for key in available_constraint_keys),
        start=np.zeros_like(motor_tn),
    )
    combined = motor_tn + constraint_total
    baseline_positive_mean = float(np.mean(arrays["baseline_positive"]))
    baseline_negative_mean = float(np.mean(arrays["baseline_negative"]))
    combined_mean = float(np.mean(combined))
    candidate_impacts = []
    for multiplier in CANDIDATE_MULTIPLIERS:
        magnitude = multiplier * combined_mean
        candidate_impacts.append(
            {
                "multiplier": multiplier,
                "penalty_mean_per_step": magnitude,
                "ratio_to_baseline_positive": (
                    magnitude / baseline_positive_mean if baseline_positive_mean > 0.0 else None
                ),
                "ratio_to_baseline_negative": (
                    magnitude / baseline_negative_mean if baseline_negative_mean > 0.0 else None
                ),
            }
        )

    motion_rows.sort(key=lambda row: row["combined_mean_per_step"], reverse=True)
    constraint_terms = {
        key: {
            "penalty_per_step": _stats(arrays[key]),
            "nonzero_rate": float(np.mean(arrays[key] > 0.0)),
        }
        for key in available_constraint_keys
    }
    return {
        "dataset": dataset_name,
        "motion_count": len(paths),
        "frame_count": int(combined.size),
        "step_dt": float(step_dt),
        "tracking": _tracking_metrics(metrics),
        "serial": {
            "penalty_per_step": _stats(serial),
            "soft_nonzero_rate": float(np.mean(arrays["serial_soft"] > 0.0)),
            "hard_trigger_rate": float(np.mean(arrays["serial_hard"] > 0.0)),
        },
        "parallel": {
            "penalty_per_step": _stats(parallel),
            "soft_nonzero_rate": float(np.mean(arrays["parallel_soft"] > 0.0)),
            "hard_trigger_rate": float(np.mean(arrays["parallel_hard"] > 0.0)),
        },
        "motor_tn_combined": {"penalty_per_step": _stats(motor_tn)},
        "constraint_terms": constraint_terms,
        "constraint_combined": {"penalty_per_step": _stats(constraint_total)},
        "combined": {"penalty_per_step": _stats(combined)},
        "baseline_reward": {
            "positive_mean_per_step": baseline_positive_mean,
            "negative_magnitude_mean_per_step": baseline_negative_mean,
        },
        "candidate_multiplier_impact": candidate_impacts,
        "top_motions": motion_rows[:10],
    }


def _load_baseline_metrics(path: Path) -> dict[str, Any]:
    path = Path(path)
    metrics_path = path if path.is_file() else path / "metrics" / "metrics_eval.json"
    return _tracking_metrics(json.loads(metrics_path.read_text(encoding="utf-8")))


def _tracking_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        key: current[key] - baseline[key]
        for key in (
            "terminated",
            "success_rate",
            "progress_rate",
            "mpjpe_l_mm",
            "mpjpe_g_mm",
            "mpjpe_pa_mm",
        )
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _plot_penalty_quantiles(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = []
    values = []
    for summary in summaries:
        for group in ("serial", "parallel"):
            labels.append(f"{summary['dataset']}\n{group}")
            stats = summary[group]["penalty_per_step"]
            values.append([stats[key] for key in ("mean", "p95", "p99", "max")])
    values_array = np.asarray(values)
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for idx, key in enumerate(("mean", "P95", "P99", "max")):
        ax.bar(x + (idx - 1.5) * width, values_array[:, idx], width, label=key)
    ax.set_xticks(x, labels)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylabel("最终惩罚 / control step（当前配置，倍率 1）")
    ax.set_title("A3 motor T-N reward 最终量级")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_trigger_rates(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = [summary["dataset"] for summary in summaries]
    x = np.arange(len(labels))
    width = 0.2
    series = [
        ("serial soft 非零", [s["serial"]["soft_nonzero_rate"] for s in summaries]),
        ("serial >100%", [s["serial"]["hard_trigger_rate"] for s in summaries]),
        ("parallel soft 非零", [s["parallel"]["soft_nonzero_rate"] for s in summaries]),
        ("parallel >100%", [s["parallel"]["hard_trigger_rate"] for s in summaries]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for idx, (name, rates) in enumerate(series):
        ax.bar(x + (idx - 1.5) * width, np.asarray(rates) * 100.0, width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("触发帧比例 (%)")
    ax.set_title("T-N 稠密 soft 非零率与越界率")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_constraint_quantiles(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = []
    values = []
    for summary in summaries:
        for term, term_summary in summary["constraint_terms"].items():
            labels.append(f"{summary['dataset']}\n{term}")
            stats = term_summary["penalty_per_step"]
            values.append([stats[key] for key in ("mean", "p95", "p99", "max")])
    if not values:
        return

    values_array = np.asarray(values)
    x = np.arange(len(labels))
    width = 0.19
    fig_width = max(12.0, 1.25 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2), constrained_layout=True)
    for idx, key in enumerate(("mean", "P95", "P99", "max")):
        ax.bar(x + (idx - 1.5) * width, values_array[:, idx], width, label=key)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_yscale("symlog", linthresh=1.0e-5)
    ax.set_ylabel("最终惩罚 / control step")
    ax.set_title("A3 额外电机约束 reward 量级")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_candidate_multipliers(summaries: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for summary in summaries:
        impacts = summary["candidate_multiplier_impact"]
        ax.plot(
            [item["multiplier"] for item in impacts],
            [100.0 * item["ratio_to_baseline_positive"] for item in impacts],
            marker="o",
            label=f"{summary['dataset']} / positive reward",
        )
        ax.plot(
            [item["multiplier"] for item in impacts],
            [100.0 * item["ratio_to_baseline_negative"] for item in impacts],
            marker="s",
            linestyle="--",
            label=f"{summary['dataset']} / existing penalty",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("相对当前完整配置的倍率")
    ax.set_ylabel("平均 motor penalty 相对量级 (%)")
    ax.set_title("完整 motor constraint reward 倍率影响")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _load_measured_envelopes(path: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (row["family"], row["serial"], row["direction"])
            grouped.setdefault(key, []).append(
                (
                    float(row["measured_speed_rad_s"]),
                    float(row["max_measured_torque_nm"]),
                )
            )
    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for (family, _, _), points in grouped.items():
        points.sort()
        result.setdefault(family, []).append(
            (
                np.asarray([point[0] for point in points]),
                np.asarray([point[1] for point in points]),
            )
        )
    return result


def _plot_tn_comparison(measured_path: Path, output: Path) -> None:
    measured = _load_measured_envelopes(measured_path)
    families = ("PFP41", "PFP59", "PFP78", "PFP93", "PFP110")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    for ax, family in zip(axes.flat, families, strict=False):
        for idx, (speed, torque) in enumerate(measured[family]):
            ax.plot(
                speed,
                torque,
                color=("#f0a84b", "#75b6ef", "#e8c174", "#8bb4bf")[idx % 4],
                alpha=0.48,
                linewidth=1.25,
                label="原始实测上包络（各样机/方向）" if idx == 0 else None,
            )
        old_speed, old_torque = CURRENT_REFERENCE_CURVES[family]
        curve = get_tn_curve(family, profile="reward_024")
        new_speed = curve.speed_rad_s.cpu().numpy()
        new_torque = curve.torque_nm.cpu().numpy()
        ax.plot(
            old_speed,
            old_torque,
            "--",
            color="#606060",
            linewidth=2.0,
            label="当前多折点保守近似",
        )
        ax.plot(
            new_speed,
            new_torque,
            "-o",
            color="#d62728",
            linewidth=3.2,
            markersize=5.5,
            label="新两段近似",
        )
        ax.annotate(
            f"平台终点\n({new_speed[1]:.2f}, {new_torque[1]:g})",
            (new_speed[1], new_torque[1]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
        )
        ax.annotate(
            f"({new_speed[2]:.2f}, 0)",
            (new_speed[2], 0),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
        )
        ax.set_title(family)
        ax.set_xlabel("|电机转速| (rad/s)")
        ax.set_ylabel("电机力矩上限 (Nm)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.03,
        0.95,
        "两段式定义\n\n1. 恒力矩平台\n2. 单一直线下降到 0 Nm\n\n红线保持在当前保守近似之下。",
        va="top",
        fontsize=13,
        linespacing=1.6,
    )
    fig.suptitle("A3 电机 T-N：原始实测、当前近似与新两段式保守近似", fontsize=20)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A3 024 完整电机约束 reward 校准",
        "",
        "惩罚使用 implicit PD 限幅前 `computed_torque`；本次评估不改变仿真动力学。",
        "",
        "## Lab Eval 与惩罚量级",
        "",
        "| 数据集 | attempted | terminated | MPJPE-L/G/PA (mm) | 组 | "
        "soft 非零帧 | >100% 帧 | mean | P95 | P99 | max |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report["datasets"]:
        tracking = summary["tracking"]
        for idx, group in enumerate(("serial", "parallel")):
            stats = summary[group]["penalty_per_step"]
            lines.append(
                "| {dataset} | {attempted} | {terminated} | {metrics} | {group} | "
                "{soft:.3%} | {hard:.3%} | {mean:.6g} | {p95:.6g} | "
                "{p99:.6g} | {max_value:.6g} |".format(
                    dataset=summary["dataset"] if idx == 0 else "",
                    attempted=tracking["attempted"] if idx == 0 else "",
                    terminated=tracking["terminated"] if idx == 0 else "",
                    metrics=(
                        f"{tracking['mpjpe_l_mm']:.3f}/{tracking['mpjpe_g_mm']:.3f}/{tracking['mpjpe_pa_mm']:.3f}"
                        if idx == 0
                        else ""
                    ),
                    group=group,
                    soft=summary[group]["soft_nonzero_rate"],
                    hard=summary[group]["hard_trigger_rate"],
                    mean=stats["mean"],
                    p95=stats["p95"],
                    p99=stats["p99"],
                    max_value=stats["max"],
                )
            )

    if any(summary["constraint_terms"] for summary in report["datasets"]):
        lines.extend(
            [
                "",
                "## 额外约束项",
                "",
                "| 数据集 | reward 项 | 非零帧 | mean | P95 | P99 | max |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in report["datasets"]:
            for term, term_summary in summary["constraint_terms"].items():
                stats = term_summary["penalty_per_step"]
                lines.append(
                    f"| {summary['dataset']} | `{term}` | "
                    f"{term_summary['nonzero_rate']:.3%} | {stats['mean']:.6g} | "
                    f"{stats['p95']:.6g} | {stats['p99']:.6g} | {stats['max']:.6g} |"
                )

    lines.extend(
        [
            "",
            "## 相对当前完整配置的候选倍率",
            "",
            "| 数据集 | 倍率 | motor penalty / 正 reward | motor penalty / 原有负 reward |",
            "|---|---:|---:|---:|",
        ]
    )
    for summary in report["datasets"]:
        for item in summary["candidate_multiplier_impact"]:
            lines.append(
                f"| {summary['dataset']} | {item['multiplier']:.2g} | "
                f"{item['ratio_to_baseline_positive']:.3%} | "
                f"{item['ratio_to_baseline_negative']:.3%} |"
            )

    lines.extend(["", "## 惩罚最高的 motions", ""])
    for summary in report["datasets"]:
        lines.extend(
            [
                f"### {summary['dataset']}",
                "",
                "| motion | terminated | serial mean | parallel mean | constraints mean | combined mean |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["top_motions"]:
            lines.append(
                f"| `{row['motion_key']}` | {str(row['terminated']).lower()} | "
                f"{row['serial_mean_per_step']:.6g} | "
                f"{row['parallel_mean_per_step']:.6g} | "
                f"{row['constraint_mean_per_step']:.6g} | "
                f"{row['combined_mean_per_step']:.6g} |"
            )
        lines.append("")

    if report.get("baseline_comparison"):
        lines.extend(
            [
                "## 与 reward 接入前 step34k 基线比较",
                "",
                "| 数据集 | Δterminated | ΔMPJPE-L | ΔMPJPE-G | ΔMPJPE-PA |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, comparison in report["baseline_comparison"].items():
            delta = comparison["delta"]
            lines.append(
                f"| {name} | {delta['terminated']:+d} | "
                f"{delta['mpjpe_l_mm']:+.6f} | {delta['mpjpe_g_mm']:+.6f} | "
                f"{delta['mpjpe_pa_mm']:+.6f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME:PATH") from exc
    return name, Path(path).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=_parse_named_path)
    parser.add_argument("--baseline", action="append", type=_parse_named_path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--measured-envelope", type=Path)
    parser.add_argument("--tn-outer-weight", type=float, default=-0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    summaries = [summarize_dataset(path, name) for name, path in args.dataset]
    tn_outer_magnitude = abs(float(args.tn_outer_weight))
    report: dict[str, Any] = {
        "schema_version": 2,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "penalty_definition": {
            "outer_weight": args.tn_outer_weight,
            "lambda_soft": 0.05,
            "lambda_violation": 5.0,
            "soft_exponent": 8.0,
            "violation_exponent": 2.0,
            "utilization_cap": 2.0,
            "effective_soft_coefficient": tn_outer_magnitude * 0.05,
            "effective_violation_coefficient": tn_outer_magnitude * 5.0,
            "serial": "direct physical motors",
            "parallel": "left/right ankle and waist pitch/roll mapped through HAL Jacobian",
            "ankle_speed_power_fixed_torque": "physical PFP78 motor coordinates after HAL Jacobian mapping",
            "cross": "low-weight pitch/roll requested-torque shaping in serial joint coordinates",
            "ankle_position": "85% of physical hard joint range",
        },
        "datasets": summaries,
    }

    if args.baseline:
        baselines = dict(args.baseline)
        comparisons = {}
        for summary in summaries:
            name = summary["dataset"]
            if name not in baselines:
                raise ValueError(f"baseline missing dataset {name!r}")
            baseline = _load_baseline_metrics(baselines[name])
            comparisons[name] = {
                "baseline": baseline,
                "current": summary["tracking"],
                "delta": _tracking_delta(summary["tracking"], baseline),
            }
        report["baseline_comparison"] = comparisons

    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _format_markdown(report),
        encoding="utf-8",
    )
    _plot_penalty_quantiles(summaries, output_dir / "motor_penalty_quantiles.png")
    _plot_trigger_rates(summaries, output_dir / "motor_penalty_trigger_rates.png")
    _plot_constraint_quantiles(
        summaries,
        output_dir / "motor_constraint_quantiles.png",
    )
    _plot_candidate_multipliers(
        summaries,
        output_dir / "candidate_multiplier_impact.png",
    )
    if args.measured_envelope is not None:
        _plot_tn_comparison(
            args.measured_envelope.expanduser().resolve(),
            output_dir / "tn_curve_two_segment_comparison.png",
        )
    print(output_dir)


if __name__ == "__main__":
    main()
