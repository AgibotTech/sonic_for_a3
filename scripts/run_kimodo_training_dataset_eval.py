#!/usr/bin/env python3
"""Run A3 MuJoCo sim2sim metrics and optional videos for a Kimodo CSV symlink set.

Only direct ``*.csv`` symlinks in ``--clips-dir`` are treated as motions.  In
particular, a regular ``manifest.csv`` is never passed to sim2sim.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIM2SIM_SCRIPT = REPO_ROOT / "gear_sonic" / "scripts" / "sim2sim_a3_mujoco.py"
RESERVED_SIM2SIM_FLAGS = {
    "--checkpoint",
    "--motion",
    "--metrics-out",
    "--batch-once",
    "--csv-source-fps",
    "--csv-frame-stride",
    "--encoder-mode",
    "--action-delay-ms",
    "--output-video",
    "--preview-video",
}


def select_clip_symlinks(clips_dir: Path) -> list[Path]:
    """Return sorted direct CSV clip symlinks, excluding manifests and regular CSVs."""
    clips_dir = clips_dir.expanduser()
    if not clips_dir.is_dir():
        raise ValueError(f"clips directory does not exist or is not a directory: {clips_dir}")

    clips: list[Path] = []
    for path in clips_dir.iterdir():
        if path.name == "manifest.csv" or path.suffix != ".csv" or not path.is_symlink():
            continue
        if not path.exists() or not path.is_file():
            raise ValueError(f"clip symlink is dangling or not a file: {path}")
        clips.append(path)
    return sorted(clips, key=lambda path: path.name)


def reject_reserved_sim2sim_flags(extra_args: Sequence[str]) -> None:
    for extra_arg in extra_args:
        for reserved in RESERVED_SIM2SIM_FLAGS:
            if extra_arg == reserved or extra_arg.startswith(f"{reserved}="):
                raise ValueError(
                    f"{reserved} is controlled by this runner; use its corresponding runner option instead"
                )


def build_sim2sim_command(args: argparse.Namespace, clip: Path, metrics_out: Path) -> list[str]:
    command = [
        args.python,
        str(args.sim2sim_script),
        "--checkpoint",
        str(args.checkpoint),
        "--motion",
        str(clip),
        "--batch-once",
        "--metrics-out",
        str(metrics_out),
        "--csv-source-fps",
        str(args.csv_source_fps),
        "--csv-frame-stride",
        str(args.csv_frame_stride),
        "--encoder-mode",
        args.encoder_mode,
        "--action-delay-ms",
        str(args.action_delay_ms),
    ]
    if args.videos_dir is not None:
        command.extend(("--output-video", str(args.videos_dir / f"{clip.stem}.mp4")))
    if args.preview_video:
        command.append("--preview-video")
    return [*command, *args.sim2sim_args]


@dataclass(frozen=True)
class ClipResult:
    clip: Path
    ok: bool
    message: str


def run_one_clip(args: argparse.Namespace, clip: Path) -> ClipResult:
    metrics_out = args.metrics_dir / f"{clip.stem}.json"
    temporary_metrics = args.metrics_dir / f".{clip.stem}.{os.getpid()}.{uuid.uuid4().hex}.json.tmp"
    video_out = args.videos_dir / f"{clip.stem}.mp4" if args.videos_dir is not None else None
    command = build_sim2sim_command(args, clip, temporary_metrics)
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            output = completed.stdout[-2000:].strip()
            return ClipResult(clip, False, f"sim2sim exit {completed.returncode}: {output}")
        try:
            with temporary_metrics.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            return ClipResult(clip, False, f"metrics JSON is missing or not parseable: {exc}")
        if not isinstance(payload, dict):
            return ClipResult(clip, False, "metrics JSON must contain an object")
        if video_out is not None and (not video_out.is_file() or video_out.stat().st_size == 0):
            return ClipResult(clip, False, f"preview video is missing or empty: {video_out}")
        temporary_metrics.replace(metrics_out)
        return ClipResult(clip, True, f"{metrics_out}")
    finally:
        if temporary_metrics.exists():
            temporary_metrics.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A3 .pt checkpoint to evaluate.")
    parser.add_argument("--clips-dir", type=Path, required=True, help="Directory containing direct CSV clip symlinks.")
    parser.add_argument("--metrics-dir", type=Path, required=True, help="Directory for one metrics JSON per clip.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent sim2sim processes (default: 8).")
    parser.add_argument("--sim2sim-script", type=Path, default=DEFAULT_SIM2SIM_SCRIPT)
    parser.add_argument("--python", default=sys.executable, help="Python executable used to invoke sim2sim.")
    parser.add_argument("--csv-source-fps", type=float, default=30.0, help="CSV source FPS (default: 30).")
    parser.add_argument("--csv-frame-stride", type=int, default=1, help="CSV frame stride (default: 1).")
    parser.add_argument("--encoder-mode", default="a3_fast", help="sim2sim encoder mode (default: a3_fast).")
    parser.add_argument("--action-delay-ms", type=float, default=10.0, help="Action delay in ms (default: 10).")
    parser.add_argument("--videos-dir", type=Path, default=None, help="Optional directory for one MP4 per clip.")
    parser.add_argument("--preview-video", action="store_true", help="Use sim2sim preview-video settings; requires --videos-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected clips and commands without running sim2sim.")
    parser.add_argument(
        "--sim2sim-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra sim2sim arguments; place last, e.g. --sim2sim-args --urdf /path/model.urdf",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.csv_source_fps <= 0.0:
        parser.error("--csv-source-fps must be positive")
    if args.csv_frame_stride < 1:
        parser.error("--csv-frame-stride must be >= 1")
    if args.action_delay_ms < 0.0:
        parser.error("--action-delay-ms must be non-negative")
    if args.preview_video and args.videos_dir is None:
        parser.error("--preview-video requires --videos-dir")
    try:
        reject_reserved_sim2sim_flags(args.sim2sim_args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        clips = select_clip_symlinks(args.clips_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not clips:
        print(f"error: no direct CSV symlink clips found in {args.clips_dir}", file=sys.stderr)
        return 2

    args.checkpoint = args.checkpoint.expanduser()
    args.metrics_dir = args.metrics_dir.expanduser()
    args.sim2sim_script = args.sim2sim_script.expanduser()
    if args.videos_dir is not None:
        args.videos_dir = args.videos_dir.expanduser()
    print(
        "Kimodo eval: "
        f"clips={len(clips)} workers={args.workers} source_fps={args.csv_source_fps:g} "
        f"frame_stride={args.csv_frame_stride} encoder_mode={args.encoder_mode} "
        f"action_delay_ms={args.action_delay_ms:g} "
        f"videos_dir={args.videos_dir if args.videos_dir is not None else 'disabled'} "
        f"preview_video={args.preview_video}"
    )

    if args.dry_run:
        for clip in clips:
            print(" ".join(build_sim2sim_command(args, clip, args.metrics_dir / f"{clip.stem}.json")))
        return 0

    if not args.checkpoint.is_file():
        print(f"error: checkpoint is not a file: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.sim2sim_script.is_file():
        print(f"error: sim2sim script is not a file: {args.sim2sim_script}", file=sys.stderr)
        return 2
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.videos_dir is not None:
        args.videos_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda clip: run_one_clip(args, clip), clips))

    failures = [result for result in results if not result.ok]
    for result in results:
        prefix = "OK" if result.ok else "FAIL"
        print(f"[{prefix}] {result.clip.name}: {result.message}")
    if failures:
        print(f"error: {len(failures)}/{len(results)} clips did not produce parseable metrics JSON", file=sys.stderr)
        return 1
    print(f"PASS: all {len(results)} selected clips produced parseable metrics JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
