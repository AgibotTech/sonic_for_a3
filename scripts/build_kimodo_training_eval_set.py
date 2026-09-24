#!/usr/bin/env python3
"""Build a deterministic native-30fps Kimodo training-data eval set.

The output contains only a manifest, README, and CSV symlinks.  CSV and PKL
contents remain in their training-data locations.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path


NATIVE_CSV_FPS = 30.0


def default_training_pkl_dir(source_csv_dir: Path) -> Path:
    return source_csv_dir.parent / f"{source_csv_dir.name}_pkl" / source_csv_dir.name


def csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        return sum(1 for _ in reader)


def _source_and_pkl_by_stem(source_csv_dir: Path, training_pkl_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    source_by_stem = {path.stem: path.resolve() for path in sorted(source_csv_dir.glob("*.csv"))}
    if not source_by_stem:
        raise ValueError(f"no .csv files found in {source_csv_dir}")
    pkl_by_stem = {path.stem: path.resolve() for path in sorted(training_pkl_dir.rglob("*.pkl"))}
    if not pkl_by_stem:
        raise ValueError(f"no .pkl files found in {training_pkl_dir}")
    source_stems = set(source_by_stem)
    pkl_stems = set(pkl_by_stem)
    if source_stems != pkl_stems:
        missing_pkl = sorted(source_stems - pkl_stems)
        missing_csv = sorted(pkl_stems - source_stems)
        raise ValueError(
            "source CSV stems and training PKL stems must match exactly; "
            f"missing PKL={missing_pkl[:10]} ({len(missing_pkl)} total), "
            f"missing CSV={missing_csv[:10]} ({len(missing_csv)} total)"
        )
    return source_by_stem, pkl_by_stem


def _clear_output_csv_symlinks(out_dir: Path) -> None:
    for path in out_dir.glob("*.csv"):
        if path.name == "manifest.csv":
            continue
        if not path.is_symlink():
            raise ValueError(f"refusing to replace non-symlink output CSV: {path}")
        path.unlink()


def build_eval_set(
    source_csv_dir: Path,
    out_dir: Path,
    *,
    target_seconds: float = 1800.0,
    seed: int = 20260714,
    training_pkl_dir: Path | None = None,
) -> dict[str, object]:
    if target_seconds <= 0.0:
        raise ValueError("target_seconds must be positive")
    source_csv_dir = source_csv_dir.resolve()
    training_pkl_dir = (default_training_pkl_dir(source_csv_dir) if training_pkl_dir is None else training_pkl_dir).resolve()
    source_by_stem, pkl_by_stem = _source_and_pkl_by_stem(source_csv_dir, training_pkl_dir)

    stems = sorted(source_by_stem)
    random.Random(seed).shuffle(stems)
    selected: list[dict[str, object]] = []
    total_seconds = 0.0
    for selection_rank, stem in enumerate(stems, start=1):
        source_csv = source_by_stem[stem]
        row_count = csv_row_count(source_csv)
        native_seconds = row_count / NATIVE_CSV_FPS
        selected.append(
            {
                "selection_rank": selection_rank,
                "stem": stem,
                "source_csv": source_csv,
                "training_pkl": pkl_by_stem[stem],
                "row_count": row_count,
                "native_seconds": native_seconds,
            }
        )
        total_seconds += native_seconds
        if total_seconds >= target_seconds:
            break
    if total_seconds < target_seconds:
        raise ValueError(
            f"source set totals only {total_seconds:.6f}s, below requested {target_seconds:.6f}s"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_output_csv_symlinks(out_dir)
    manifest_path = out_dir / "manifest.csv"
    for item in selected:
        source_csv = Path(item["source_csv"])
        link = out_dir / source_csv.name
        link.symlink_to(os.path.relpath(source_csv, start=out_dir))
        item["symlink_csv"] = link.name

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "selection_rank",
                "stem",
                "symlink_csv",
                "source_csv",
                "training_pkl",
                "row_count",
                "native_seconds",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)

    max_clip = max(float(item["native_seconds"]) for item in selected)
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "# Kimodo boxing native-30fps fixed eval set v1\n\n"
        f"- Source CSV: `{source_csv_dir}`\n"
        f"- Training PKL root (validated stem-for-stem): `{training_pkl_dir}`\n"
        f"- Validated source/training clip count: `{len(source_by_stem)}` / `{len(pkl_by_stem)}`\n"
        f"- Seed: `{seed}`\n"
        f"- Target duration: `{target_seconds:.6f}` seconds\n"
        f"- Actual clips: `{len(selected)}`\n"
        f"- Actual native duration: `{total_seconds:.6f}` seconds\n"
        f"- Maximum single-clip native duration: `{max_clip:.6f}` seconds\n"
        f"- Time semantics: source CSV is native `{NATIVE_CSV_FPS:.0f}`fps; later sim2sim must use "
        "`--csv-source-fps 30 --csv-frame-stride 1`.\n\n"
        "Each `*.csv` in this directory is a symlink to its original training CSV; no raw motion data is copied.\n",
        encoding="utf-8",
    )
    return {
        "manifest_path": manifest_path,
        "readme_path": readme_path,
        "clip_count": len(selected),
        "total_seconds": total_seconds,
        "max_clip_seconds": max_clip,
        "source_clip_count": len(source_by_stem),
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--training-pkl-dir",
        type=Path,
        default=None,
        help="Optional PKL root. Default derives <source-parent>/<source-name>_pkl/<source-name>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_eval_set(
        args.source_csv_dir,
        args.out_dir,
        target_seconds=args.target_seconds,
        seed=args.seed,
        training_pkl_dir=args.training_pkl_dir,
    )
    print(
        "[kimodo-eval-set] "
        f"clips={result['clip_count']} total_seconds={result['total_seconds']:.6f} "
        f"max_clip_seconds={result['max_clip_seconds']:.6f} seed={result['seed']} "
        f"manifest={result['manifest_path']}"
    )


if __name__ == "__main__":
    main()
