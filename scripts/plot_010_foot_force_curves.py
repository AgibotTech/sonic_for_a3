#!/usr/bin/env python3
"""Plot A3 010 foot segment force curves and CoP summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


DEFAULT_MESH_DIR = Path(
    "gear_sonic/data/assets/robot_description/urdf/a3/meshes_collision_optimized"
)
SEGMENTS = ("rear", "mid", "front")
FORCE_COLUMNS = (
    "Fz_left_front",
    "Fz_left_mid",
    "Fz_left_rear",
    "Fz_right_front",
    "Fz_right_mid",
    "Fz_right_rear",
)


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe.strip("._") or "motion"


def load_manifest(force_dir: Path) -> list[dict]:
    manifest_path = force_dir / "foot_force_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            return json.load(f)
    entries = []
    for csv_path in sorted(force_dir.glob("*_foot_forces.csv")):
        stem = csv_path.name.removesuffix("_foot_forces.csv")
        motion_idx = int(stem.split("_", 1)[0])
        motion_key = stem.split("_", 1)[1] if "_" in stem else stem
        entries.append(
            {
                "env_idx": motion_idx,
                "motion_idx": motion_idx,
                "motion_key": motion_key,
                "csv": str(csv_path),
                "npz": str(csv_path.with_suffix(".npz")),
            }
        )
    return entries


def load_segment_centers(mesh_dir: Path) -> dict[str, dict[str, float]]:
    centers: dict[str, dict[str, float]] = {"left": {}, "right": {}}
    bounds: dict[str, dict[str, list[list[float]]]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for segment in SEGMENTS:
            mesh_path = mesh_dir / f"{side}_foot_{segment}_collision_3seg.STL"
            mesh = trimesh.load(mesh_path, force="mesh")
            x_center = float(mesh.bounds[:, 0].mean())
            centers[side][segment] = x_center
            bounds[side][segment] = (mesh.bounds * 1000.0).round(3).tolist()
    return {"centers_m": centers, "bounds_mm": bounds}


def read_force_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        time_s = []
        steps = []
        forces = []
        for row in reader:
            steps.append(int(row["step"]))
            time_s.append(float(row["time_s"]))
            forces.append([float(row[column]) for column in FORCE_COLUMNS])
    return np.asarray(steps), np.asarray(time_s), np.asarray(forces, dtype=np.float64)


def compute_curves(
    steps: np.ndarray,
    time_s: np.ndarray,
    forces: np.ndarray,
    segment_centers: dict,
    load_threshold_n: float,
) -> dict[str, np.ndarray]:
    left = {
        "front": forces[:, 0],
        "mid": forces[:, 1],
        "rear": forces[:, 2],
    }
    right = {
        "front": forces[:, 3],
        "mid": forces[:, 4],
        "rear": forces[:, 5],
    }
    out: dict[str, np.ndarray] = {"step": steps, "time_s": time_s, "forces": forces}
    for side, data in (("left", left), ("right", right)):
        total = data["front"] + data["mid"] + data["rear"]
        loaded = total > load_threshold_n
        out[f"{side}_total"] = total
        out[f"{side}_loaded"] = loaded
        for segment in ("front", "mid", "rear"):
            ratio = np.divide(
                data[segment],
                total,
                out=np.full_like(total, np.nan, dtype=np.float64),
                where=loaded,
            )
            out[f"{side}_{segment}_ratio"] = ratio
        numerator = sum(
            data[segment] * segment_centers["centers_m"][side][segment]
            for segment in ("front", "mid", "rear")
        )
        cop = np.divide(
            numerator,
            total,
            out=np.full_like(total, np.nan, dtype=np.float64),
            where=loaded,
        )
        out[f"{side}_cop_x"] = cop

    combined_total = out["left_total"] + out["right_total"]
    out["combined_total"] = combined_total
    for segment in ("front", "mid", "rear"):
        left_idx = {"front": 0, "mid": 1, "rear": 2}[segment]
        right_idx = {"front": 3, "mid": 4, "rear": 5}[segment]
        out[f"combined_{segment}"] = forces[:, left_idx] + forces[:, right_idx]
    return out


def write_enriched_csv(path: Path, curves: dict[str, np.ndarray]) -> None:
    headers = [
        "step",
        "time_s",
        *FORCE_COLUMNS,
        "left_total",
        "right_total",
        "left_cop_x",
        "right_cop_x",
        "left_front_ratio",
        "left_mid_ratio",
        "left_rear_ratio",
        "right_front_ratio",
        "right_mid_ratio",
        "right_rear_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx in range(len(curves["time_s"])):
            row = [
                int(curves["step"][idx]),
                float(curves["time_s"][idx]),
                *curves["forces"][idx].tolist(),
                float(curves["left_total"][idx]),
                float(curves["right_total"][idx]),
                float(curves["left_cop_x"][idx]) if np.isfinite(curves["left_cop_x"][idx]) else "",
                float(curves["right_cop_x"][idx]) if np.isfinite(curves["right_cop_x"][idx]) else "",
                *[
                    float(curves[key][idx]) if np.isfinite(curves[key][idx]) else ""
                    for key in (
                        "left_front_ratio",
                        "left_mid_ratio",
                        "left_rear_ratio",
                        "right_front_ratio",
                        "right_mid_ratio",
                        "right_rear_ratio",
                    )
                ],
            ]
            writer.writerow(row)


def plot_motion(path: Path, motion_key: str, curves: dict[str, np.ndarray]) -> None:
    time_s = curves["time_s"]
    forces = curves["forces"]
    colors = {"front": "tab:red", "mid": "tab:green", "rear": "tab:blue"}
    fig, axes = plt.subplots(5, 1, figsize=(14, 12.5), sharex=True)
    fig.suptitle(f"010 foot segment Fz / CoP: {motion_key}", fontsize=13, y=0.995)

    for ax, side, offset in ((axes[0], "left", 0), (axes[1], "right", 3)):
        ax.plot(time_s, forces[:, offset + 0], color=colors["front"], linewidth=1.0, label="front")
        ax.plot(time_s, forces[:, offset + 1], color=colors["mid"], linewidth=1.0, label="mid")
        ax.plot(time_s, forces[:, offset + 2], color=colors["rear"], linewidth=1.0, label="rear")
        ax.set_ylabel(f"{side} Fz (N)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    axes[2].plot(time_s, curves["left_cop_x"], color="tab:purple", linewidth=1.0, label="left CoP_x")
    axes[2].plot(time_s, curves["right_cop_x"], color="tab:orange", linewidth=1.0, label="right CoP_x")
    axes[2].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("CoP_x (m)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8)

    for ax, side in ((axes[3], "left"), (axes[4], "right")):
        ratios = [
            np.nan_to_num(curves[f"{side}_{segment}_ratio"], nan=0.0)
            for segment in ("rear", "mid", "front")
        ]
        ax.stackplot(
            time_s,
            ratios,
            colors=[colors["rear"], colors["mid"], colors["front"]],
            labels=["rear", "mid", "front"],
            alpha=0.78,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"{side} share")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_motion(
    motion_key: str,
    motion_idx: int,
    curves: dict[str, np.ndarray],
    load_threshold_n: float,
    active_ratio_threshold: float,
) -> dict:
    combined_total = curves["combined_total"]
    loaded = combined_total > load_threshold_n
    total_force_integral = float(combined_total[loaded].sum())
    summary = {
        "motion_idx": int(motion_idx),
        "motion_key": str(motion_key),
        "num_steps": int(len(combined_total)),
        "duration_s": float(curves["time_s"][-1]) if len(combined_total) else 0.0,
        "loaded_fraction_combined": float(loaded.mean()) if len(combined_total) else 0.0,
        "left_loaded_fraction": float(curves["left_loaded"].mean()) if len(combined_total) else 0.0,
        "right_loaded_fraction": float(curves["right_loaded"].mean()) if len(combined_total) else 0.0,
        "max_left_total_fz": float(np.nanmax(curves["left_total"])) if len(combined_total) else 0.0,
        "max_right_total_fz": float(np.nanmax(curves["right_total"])) if len(combined_total) else 0.0,
    }
    for segment in ("rear", "mid", "front"):
        value = (
            float(curves[f"combined_{segment}"][loaded].sum() / total_force_integral)
            if total_force_integral > 0.0
            else math.nan
        )
        summary[f"combined_{segment}_force_share"] = value

    for side in ("left", "right"):
        side_loaded = curves[f"{side}_loaded"]
        side_cop = curves[f"{side}_cop_x"][side_loaded]
        summary[f"{side}_cop_mean_x"] = float(np.nanmean(side_cop)) if side_cop.size else math.nan
        summary[f"{side}_cop_min_x"] = float(np.nanmin(side_cop)) if side_cop.size else math.nan
        summary[f"{side}_cop_max_x"] = float(np.nanmax(side_cop)) if side_cop.size else math.nan
        for segment in ("front", "mid", "rear"):
            ratio = curves[f"{side}_{segment}_ratio"]
            active = side_loaded & (ratio > active_ratio_threshold)
            summary[f"{side}_{segment}_active_fraction"] = (
                float(active.sum() / max(side_loaded.sum(), 1))
                if len(side_loaded)
                else 0.0
            )

    cop_values = np.concatenate(
        [
            curves["left_cop_x"][curves["left_loaded"]],
            curves["right_cop_x"][curves["right_loaded"]],
        ]
    )
    summary["cop_mean_x"] = float(np.nanmean(cop_values)) if cop_values.size else math.nan
    summary["cop_min_x"] = float(np.nanmin(cop_values)) if cop_values.size else math.nan
    summary["cop_max_x"] = float(np.nanmax(cop_values)) if cop_values.size else math.nan
    summary["cop_range_x"] = (
        float(summary["cop_max_x"] - summary["cop_min_x"])
        if np.isfinite(summary["cop_min_x"]) and np.isfinite(summary["cop_max_x"])
        else math.nan
    )

    if loaded.any():
        rear = curves["combined_rear"][loaded]
        mid = curves["combined_mid"][loaded]
        front = curves["combined_front"][loaded]
        summary["mid_dominant_fraction"] = float(((mid >= rear) & (mid >= front)).mean())
    else:
        summary["mid_dominant_fraction"] = math.nan
    return summary


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: row["motion_idx"])
    names = [safe_name(row["motion_key"]) for row in rows]
    y = np.arange(len(rows))
    rear = np.asarray([row["combined_rear_force_share"] for row in rows])
    mid = np.asarray([row["combined_mid_force_share"] for row in rows])
    front = np.asarray([row["combined_front_force_share"] for row in rows])
    cop_mean = np.asarray([row["cop_mean_x"] for row in rows])
    cop_min = np.asarray([row["cop_min_x"] for row in rows])
    cop_max = np.asarray([row["cop_max_x"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(7.5, 0.42 * len(rows))), sharey=True)
    axes[0].barh(y, rear, color="tab:blue", label="rear")
    axes[0].barh(y, mid, left=rear, color="tab:green", label="mid")
    axes[0].barh(y, front, left=rear + mid, color="tab:red", label="front")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("force-integral share over loaded frames")
    axes[0].set_yticks(y, names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)

    xerr = np.vstack([cop_mean - cop_min, cop_max - cop_mean])
    axes[1].errorbar(cop_mean, y, xerr=xerr, fmt="o", color="black", ecolor="gray", capsize=2)
    axes[1].axvline(-0.077, color="tab:blue", linestyle="--", linewidth=1.0, label="rear/mid split")
    axes[1].axvline(0.161, color="tab:red", linestyle="--", linewidth=1.0, label="mid/front split")
    axes[1].set_xlabel("CoP_x mean and range (m)")
    axes[1].grid(True, axis="x", alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)

    fig.suptitle("010 foot segment force share and CoP summary", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def maybe_copy_videos(entries: list[dict], video_dir: Path | None, output_dir: Path) -> list[dict]:
    if video_dir is None:
        return []
    copied = []
    named_dir = output_dir / "videos_named"
    named_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        env_idx = int(entry.get("env_idx", entry.get("motion_idx", 0)))
        source = video_dir / f"{env_idx:06d}.mp4"
        if not source.exists():
            continue
        dest = named_dir / f"010_eval_{safe_name(entry['motion_key'])}.mp4"
        shutil.copy2(source, dest)
        copied.append(
            {
                "motion_idx": int(entry.get("motion_idx", env_idx)),
                "motion_key": str(entry["motion_key"]),
                "source": str(source),
                "video": str(dest),
            }
        )
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--load-threshold-n", type=float, default=20.0)
    parser.add_argument("--active-ratio-threshold", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    curves_dir = args.output_dir / "enriched_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    segment_info = load_segment_centers(args.mesh_dir)
    with (args.output_dir / "segment_centers.json").open("w", encoding="utf-8") as f:
        json.dump(segment_info, f, indent=2)

    entries = load_manifest(args.force_dir)
    summaries = []
    analysis_entries = []
    for entry in entries:
        csv_path = Path(entry["csv"])
        if not csv_path.exists():
            continue
        steps, time_s, forces = read_force_csv(csv_path)
        curves = compute_curves(
            steps,
            time_s,
            forces,
            segment_info,
            load_threshold_n=args.load_threshold_n,
        )
        motion_key = str(entry["motion_key"])
        motion_idx = int(entry.get("motion_idx", len(summaries)))
        safe = f"{motion_idx:06d}_{safe_name(motion_key)}"
        enriched_csv = curves_dir / f"{safe}_foot_force_cop.csv"
        plot_path = plots_dir / f"{safe}_foot_force_cop.png"
        write_enriched_csv(enriched_csv, curves)
        plot_motion(plot_path, motion_key, curves)
        summary = summarize_motion(
            motion_key,
            motion_idx,
            curves,
            load_threshold_n=args.load_threshold_n,
            active_ratio_threshold=args.active_ratio_threshold,
        )
        summary["force_csv"] = str(csv_path)
        summary["enriched_csv"] = str(enriched_csv)
        summary["plot"] = str(plot_path)
        summaries.append(summary)
        analysis_entries.append({**entry, "enriched_csv": str(enriched_csv), "plot": str(plot_path)})

    summaries = sorted(summaries, key=lambda row: row["motion_idx"])
    write_summary_csv(args.output_dir / "foot_force_summary.csv", summaries)
    with (args.output_dir / "foot_force_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    if summaries:
        plot_summary(args.output_dir / "foot_force_summary.png", summaries)

    videos = maybe_copy_videos(entries, args.video_dir, args.output_dir)
    with (args.output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segment_info": segment_info,
                "force_entries": analysis_entries,
                "videos": videos,
                "load_threshold_n": args.load_threshold_n,
                "active_ratio_threshold": args.active_ratio_threshold,
            },
            f,
            indent=2,
        )

    print(f"motions={len(summaries)}")
    print(f"plots={plots_dir}")
    print(f"summary={args.output_dir / 'foot_force_summary.csv'}")
    if videos:
        print(f"videos_named={args.output_dir / 'videos_named'}")


if __name__ == "__main__":
    main()
