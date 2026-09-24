#!/usr/bin/env python3
"""Build a symlinked failure-top50 (adaptive sampler proxy) eval manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path

from build_kimodo_training_eval_set import (
    NATIVE_CSV_FPS,
    csv_row_count,
    default_training_pkl_dir,
)


def _is_active(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"unrecognized is_active value: {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_step_from_path(adaptive_csv: Path) -> int | None:
    match = re.search(r"step_(\d+)", adaptive_csv.stem)
    return None if match is None else int(match.group(1))


def _pkl_by_stem(training_pkl_dir: Path) -> dict[str, Path]:
    result = {path.stem: path.resolve() for path in sorted(training_pkl_dir.rglob("*.pkl"))}
    if not result:
        raise ValueError(f"no .pkl files found in {training_pkl_dir}")
    return result


def build_top50_manifest(
    adaptive_csv: Path,
    source_csv_dir: Path,
    out_dir: Path,
    *,
    top: int = 50,
    training_pkl_dir: Path | None = None,
    checkpoint_step: int | None = None,
) -> dict[str, object]:
    if top <= 0:
        raise ValueError("top must be positive")
    adaptive_csv = adaptive_csv.resolve()
    source_csv_dir = source_csv_dir.resolve()
    training_pkl_dir = (default_training_pkl_dir(source_csv_dir) if training_pkl_dir is None else training_pkl_dir).resolve()
    source_by_stem = {path.stem: path.resolve() for path in sorted(source_csv_dir.glob("*.csv"))}
    if not source_by_stem:
        raise ValueError(f"no .csv files found in {source_csv_dir}")
    pkl_by_stem = _pkl_by_stem(training_pkl_dir)

    with adaptive_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty adaptive CSV: {adaptive_csv}")
        required = {"motion_key", "motion_sampling_prob", "is_active"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{adaptive_csv} missing required columns: {missing}")
        active_rows = []
        for row in reader:
            if not _is_active(str(row["is_active"])):
                continue
            motion_key = str(row["motion_key"]).strip()
            if not motion_key:
                raise ValueError(f"{adaptive_csv} has an active row with an empty motion_key")
            try:
                probability = float(row["motion_sampling_prob"])
            except ValueError as exc:
                raise ValueError(
                    f"{adaptive_csv} has invalid motion_sampling_prob for {motion_key!r}: "
                    f"{row['motion_sampling_prob']!r}"
                ) from exc
            active_rows.append((motion_key, probability, row))
    active_rows.sort(key=lambda item: (-item[1], item[0]))
    if len(active_rows) < top:
        raise ValueError(f"only {len(active_rows)} active motions, cannot build top {top}")
    selected = active_rows[:top]
    keys = [item[0] for item in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("adaptive CSV has duplicate motion_key values within selected active motions")

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    for path in clips_dir.glob("*.csv"):
        if not path.is_symlink():
            raise ValueError(f"refusing to replace non-symlink output CSV: {path}")
        path.unlink()

    adaptive_sha256 = _sha256(adaptive_csv)
    if checkpoint_step is None:
        checkpoint_step = _checkpoint_step_from_path(adaptive_csv)
    manifest_rows: list[dict[str, object]] = []
    for rank, (motion_key, probability, row) in enumerate(selected, start=1):
        stem = Path(motion_key).stem
        source_csv = source_by_stem.get(stem)
        training_pkl = pkl_by_stem.get(stem)
        if source_csv is None or training_pkl is None:
            raise ValueError(
                f"active motion_key {motion_key!r} cannot resolve to both training CSV and PKL "
                f"(csv={source_csv is not None}, pkl={training_pkl is not None})"
            )
        link = clips_dir / source_csv.name
        link.symlink_to(os.path.relpath(source_csv, start=clips_dir))
        row_count = csv_row_count(source_csv)
        manifest_rows.append(
            {
                "rank": rank,
                "motion_key": motion_key,
                "is_active": "true",
                "motion_sampling_prob": probability,
                "max_bin_sampling_prob": row.get("max_bin_sampling_prob", ""),
                "training_pkl": training_pkl,
                "source_csv": source_csv,
                "symlink_csv": Path("clips") / link.name,
                "row_count": row_count,
                "native_seconds": row_count / NATIVE_CSV_FPS,
                "adaptive_csv_sha256": adaptive_sha256,
                "checkpoint_step": "" if checkpoint_step is None else checkpoint_step,
            }
        )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "README.md").write_text(
        "# failure-top50（adaptive sampler proxy）\n\n"
        "该集合按 active motion 的 `motion_sampling_prob` 降序、`motion_key` 升序选取。"
        "它是训练 adaptive sampler 的困难/失败代理，不是直接测得的真实 fall rate。\n\n"
        f"- Adaptive CSV: `{adaptive_csv}`\n"
        f"- Adaptive CSV SHA-256: `{adaptive_sha256}`\n"
        f"- Checkpoint/adaptive step: `{checkpoint_step}`\n"
        f"- Clips: `{len(manifest_rows)}`\n"
        f"- Native CSV FPS: `{NATIVE_CSV_FPS:.0f}`\n",
        encoding="utf-8",
    )
    return {
        "manifest_path": manifest_path,
        "clip_dir": clips_dir,
        "rows": manifest_rows,
        "adaptive_csv_sha256": adaptive_sha256,
        "checkpoint_step": checkpoint_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-csv", type=Path, required=True)
    parser.add_argument("--source-csv-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--training-pkl-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_top50_manifest(
        args.adaptive_csv,
        args.source_csv_dir,
        args.out_dir,
        top=args.top,
        training_pkl_dir=args.training_pkl_dir,
        checkpoint_step=args.checkpoint_step,
    )
    print(
        "[adaptive-top50] "
        f"clips={len(result['rows'])} step={result['checkpoint_step']} "
        f"manifest={result['manifest_path']}"
    )


if __name__ == "__main__":
    main()
