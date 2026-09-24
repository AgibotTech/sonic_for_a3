#!/usr/bin/env python3
"""Materialize deterministic speed variants of one-motion joblib datasets.

The spatial samples are preserved exactly.  Speed is represented by multiplying
the source ``fps`` value, which lets the canonical MotionLib loader resample the
motion to its configured target FPS.  Outputs are regular files with globally
unique filename stems and inner motion keys, so multiple speed directories can
be passed to ``motion_file`` as a ListConfig without collisions.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import time
from typing import Any

import joblib
import numpy as np

SOURCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SEQUENCE_FIELDS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints")


@dataclass(frozen=True)
class SpeedSpec:
    factor: str
    tag: str

    @property
    def decimal(self) -> Decimal:
        return Decimal(self.factor)

    @property
    def directory(self) -> str:
        return f"speed_{self.tag}"


def _speed_spec(value: float | str | Decimal) -> SpeedSpec:
    try:
        factor = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid speed factor: {value}") from exc
    if not factor.is_finite() or factor <= 0:
        raise ValueError(f"speed factor must be finite and positive, got {value}")
    quantized = factor.quantize(Decimal("0.01"))
    if quantized != factor:
        raise ValueError(f"speed factor supports at most two decimal places, got {value}")
    display = f"{quantized:.2f}"
    return SpeedSpec(factor=display, tag=display.replace(".", "p"))


def _normalise_speeds(speeds: Sequence[float | str | Decimal]) -> tuple[SpeedSpec, ...]:
    specs = tuple(_speed_spec(speed) for speed in speeds)
    if not specs:
        raise ValueError("at least one speed factor is required")
    tags = [spec.tag for spec in specs]
    if len(tags) != len(set(tags)):
        raise ValueError(f"duplicate speed factors after normalisation: {tags}")
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp_path = Path(stream.name)
    os.replace(temp_path, path)


def _atomic_joblib_dump(payload: Mapping[str, Any], path: Path, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        joblib.dump(payload, temp_path, compress=compression)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _load_single_motion(path: Path) -> tuple[str, dict[str, Any], int, float]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or len(payload) != 1:
        length = len(payload) if hasattr(payload, "__len__") else "unknown"
        raise ValueError(f"expected one-motion dict in {path}, got {type(payload)} len={length}")
    motion_key, motion = next(iter(payload.items()))
    if not isinstance(motion, dict):
        raise TypeError(f"expected dict motion payload in {path}, got {type(motion)}")
    if "root_trans_offset" not in motion or "fps" not in motion:
        raise KeyError(f"root_trans_offset/fps missing from {path}")

    num_frames = int(motion["root_trans_offset"].shape[0])
    fps = float(motion["fps"])
    if num_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid motion contract in {path}: frames={num_frames}, fps={fps}")
    for field in SEQUENCE_FIELDS:
        value = motion.get(field)
        if value is not None and hasattr(value, "shape") and int(value.shape[0]) != num_frames:
            raise ValueError(
                f"sequence field length mismatch in {path}: {field}={value.shape[0]} "
                f"root_trans_offset={num_frames}"
            )
    return str(motion_key), motion, num_frames, fps


def _output_stem(source_stem: str, source_label: str, speed: SpeedSpec) -> str:
    return f"{source_stem}__src_{source_label}__speed_{speed.tag}"


def _scaled_fps(original_fps: float, speed: SpeedSpec) -> float:
    return float(Decimal(str(original_fps)) * speed.decimal)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if hasattr(left, "detach") and hasattr(left, "cpu"):
        if not (hasattr(right, "detach") and hasattr(right, "cpu")):
            return False
        return np.array_equal(left.detach().cpu().numpy(), right.detach().cpu().numpy())
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _values_equal(left_value, right_value) for left_value, right_value in zip(left, right, strict=True)
        )
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, np.ndarray):
        return bool(result.all())
    return bool(result)


def _validate_augmented_motion(
    *,
    output_path: Path,
    expected_key: str,
    expected_fps: float,
    source_motion: Mapping[str, Any],
) -> None:
    if output_path.is_symlink() or not output_path.is_file():
        raise ValueError(f"output must be a regular file: {output_path}")
    payload = joblib.load(output_path)
    if not isinstance(payload, dict) or list(payload) != [expected_key]:
        raise ValueError(f"unexpected output wrapper/key in {output_path}")
    output_motion = payload[expected_key]
    if set(output_motion) != set(source_motion):
        raise ValueError(f"field mismatch in {output_path}")
    if float(output_motion["fps"]) != expected_fps:
        raise ValueError(f"fps mismatch in {output_path}: {output_motion['fps']} != {expected_fps}")
    for field, source_value in source_motion.items():
        if field == "fps":
            continue
        if not _values_equal(source_value, output_motion[field]):
            raise ValueError(f"field changed during speed augmentation: {output_path} field={field}")


def _process_source_file(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        source_path_text,
        source_relative_text,
        source_label,
        output_root_text,
        speed_rows,
        compression,
        resume,
    ) = args
    source_path = Path(source_path_text)
    source_relative = Path(source_relative_text)
    output_root = Path(output_root_text)
    speeds = tuple(SpeedSpec(**row) for row in speed_rows)
    source_key, source_motion, num_frames, original_fps = _load_single_motion(source_path)
    source_sha256 = _sha256(source_path)

    variants = []
    for speed in speeds:
        output_key = _output_stem(source_path.stem, source_label, speed)
        output_relative = Path(source_label) / speed.directory / source_relative.parent / f"{output_key}.pkl"
        output_path = output_root / output_relative
        new_fps = _scaled_fps(original_fps, speed)
        if output_path.exists() or output_path.is_symlink():
            if not resume:
                raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
            _validate_augmented_motion(
                output_path=output_path,
                expected_key=output_key,
                expected_fps=new_fps,
                source_motion=source_motion,
            )
        else:
            augmented_motion = dict(source_motion)
            augmented_motion["fps"] = new_fps
            _atomic_joblib_dump({output_key: augmented_motion}, output_path, compression)

        variants.append(
            {
                "speed": speed.factor,
                "speed_tag": speed.tag,
                "output_file": output_relative.as_posix(),
                "output_key": output_key,
                "fps": new_fps,
                "frames": num_frames,
                "duration_seconds": num_frames / new_fps,
            }
        )

    return {
        "source": source_label,
        "source_file": source_relative.as_posix(),
        "source_key": source_key,
        "source_sha256": source_sha256,
        "source_frames": num_frames,
        "source_fps": original_fps,
        "source_duration_seconds": num_frames / original_fps,
        "variants": variants,
    }


def _validate_manifest_row(args: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    source_root_text, output_root_text, row = args
    source_path = Path(source_root_text) / row["source_file"]
    output_root = Path(output_root_text)
    _, source_motion, num_frames, original_fps = _load_single_motion(source_path)
    source_sha256 = _sha256(source_path)
    if source_sha256 != row["source_sha256"]:
        raise ValueError(f"source changed since generation: {source_path}")
    if num_frames != row["source_frames"] or original_fps != row["source_fps"]:
        raise ValueError(f"source metadata changed since generation: {source_path}")

    output_frames = 0
    for variant in row["variants"]:
        output_path = output_root / variant["output_file"]
        _validate_augmented_motion(
            output_path=output_path,
            expected_key=variant["output_key"],
            expected_fps=float(variant["fps"]),
            source_motion=source_motion,
        )
        output_frames += int(variant["frames"])
    return {
        "source": row["source"],
        "source_frames": num_frames,
        "output_motions": len(row["variants"]),
        "output_frames": output_frames,
    }


def _normalise_sources(sources: Mapping[str, Path]) -> dict[str, Path]:
    if not sources:
        raise ValueError("at least one source is required")
    normalised = {}
    for label, path in sorted(sources.items()):
        if not SOURCE_LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"invalid source label {label!r}")
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        normalised[label] = resolved
    return normalised


def _source_paths(source_root: Path) -> list[Path]:
    paths = sorted(path for path in source_root.rglob("*.pkl") if path.name != "metadata.pkl")
    if not paths:
        raise FileNotFoundError(f"no PKLs found in {source_root}")
    stems: dict[str, Path] = {}
    for path in paths:
        if path.stem in stems:
            raise ValueError(
                f"duplicate filename stem {path.stem!r} in {source_root}: {stems[path.stem]} and {path}"
            )
        stems[path.stem] = path
    return paths


def _ensure_separate_output(sources: Mapping[str, Path], output_root: Path) -> Path:
    output_root = output_root.absolute()
    output_resolved = output_root.resolve(strict=False)
    for source in sources.values():
        if output_resolved == source or output_resolved.is_relative_to(source):
            raise ValueError(f"output root must not contain or be inside source: {output_root}")
        if source.is_relative_to(output_resolved):
            raise ValueError(f"output root must not contain source: {output_root}")
    return output_root


def _write_manifest(rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> str:
    descriptor, temp_name = tempfile.mkstemp(dir=manifest_path.parent, prefix=f".{manifest_path.name}.")
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        manifest_sha256 = _sha256(temp_path)
        os.replace(temp_path, manifest_path)
        return manifest_sha256
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return [json.loads(line) for line in manifest_path.read_text().splitlines() if line]


def _full_validation(
    *,
    sources: Mapping[str, Path],
    output_root: Path,
    rows: Sequence[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    tasks = []
    for row in rows:
        label = row["source"]
        if label not in sources:
            raise ValueError(f"manifest references unknown source {label!r}")
        tasks.append((str(sources[label]), str(output_root), row))

    source_frames = 0
    output_frames = 0
    output_motions = 0
    source_counts = {label: 0 for label in sources}
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_validate_manifest_row, tasks, chunksize=1), start=1):
            source_counts[result["source"]] += 1
            source_frames += result["source_frames"]
            output_motions += result["output_motions"]
            output_frames += result["output_frames"]
            if index % 1000 == 0 or index == len(tasks):
                print(
                    f"validated={index}/{len(tasks)} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    actual_count = 0
    for path in output_root.glob("**/*.pkl"):
        if path.is_symlink():
            raise ValueError(f"augmented output contains symlink: {path}")
        actual_count += 1
    if actual_count != output_motions:
        raise ValueError(f"output PKL count mismatch: filesystem={actual_count} manifest={output_motions}")
    return {
        "source_counts": source_counts,
        "source_motions": len(rows),
        "output_motions": output_motions,
        "source_frames": source_frames,
        "output_frames": output_frames,
        "symlink_count": 0,
        "validation_seconds": time.time() - started,
    }


def validate_dataset(*, sources: Mapping[str, Path], output_root: Path, workers: int) -> dict[str, Any]:
    sources = _normalise_sources(sources)
    output_root = _ensure_separate_output(sources, Path(output_root))
    manifest_path = output_root / "semantic_manifest.jsonl"
    rows = _read_manifest(manifest_path)
    result = _full_validation(sources=sources, output_root=output_root, rows=rows, workers=workers)
    result.update(
        {
            "status": "complete",
            "output_root": str(output_root),
            "manifest_sha256": _sha256(manifest_path),
        }
    )
    return result


def build_dataset(
    *,
    sources: Mapping[str, Path],
    expected_counts: Mapping[str, int],
    output_root: Path,
    speeds: Sequence[float | str | Decimal],
    workers: int,
    compression: int,
    resume: bool,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if compression < 0 or compression > 9:
        raise ValueError(f"compression must be between 0 and 9, got {compression}")
    sources = _normalise_sources(sources)
    if set(expected_counts) != set(sources):
        raise ValueError("expected_counts labels must exactly match source labels")
    specs = _normalise_speeds(speeds)
    output_root = _ensure_separate_output(sources, Path(output_root))

    if output_root.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    complete_path = output_root / "COMPLETE.json"
    if complete_path.exists() and resume:
        return validate_dataset(sources=sources, output_root=output_root, workers=workers)
    _atomic_json_dump(
        {
            "status": "incomplete",
            "host": socket.gethostname(),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
        output_root / "INCOMPLETE.json",
    )

    speed_rows = tuple({"factor": spec.factor, "tag": spec.tag} for spec in specs)
    tasks = []
    for label, source_root in sources.items():
        paths = _source_paths(source_root)
        expected = int(expected_counts[label])
        if len(paths) != expected:
            raise ValueError(f"expected {expected} PKLs in {source_root}, got {len(paths)}")
        for path in paths:
            tasks.append(
                (
                    str(path),
                    path.relative_to(source_root).as_posix(),
                    label,
                    str(output_root),
                    speed_rows,
                    compression,
                    resume,
                )
            )

    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(_process_source_file, tasks, chunksize=1), start=1):
            rows.append(row)
            if index % 500 == 0 or index == len(tasks):
                print(
                    f"generated={index}/{len(tasks)} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    rows.sort(key=lambda row: (row["source"], row["source_file"]))
    manifest_path = output_root / "semantic_manifest.jsonl"
    manifest_sha256 = _write_manifest(rows, manifest_path)
    validation = _full_validation(sources=sources, output_root=output_root, rows=rows, workers=workers)
    source_duration = sum(float(row["source_duration_seconds"]) for row in rows)
    output_duration = sum(float(variant["duration_seconds"]) for row in rows for variant in row["variants"])
    summary = {
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "host": socket.gethostname(),
        "sources": {label: str(path) for label, path in sources.items()},
        "output_root": str(output_root),
        "speeds": [spec.factor for spec in specs],
        "compression": compression,
        "workers": workers,
        "manifest_sha256": manifest_sha256,
        "source_motions": validation["source_motions"],
        "output_motions": validation["output_motions"],
        "source_counts": validation["source_counts"],
        "source_frames": validation["source_frames"],
        "output_frames": validation["output_frames"],
        "source_duration_hours": source_duration / 3600.0,
        "output_duration_hours": output_duration / 3600.0,
        "symlink_count": validation["symlink_count"],
        "generation_seconds": time.time() - started,
        "validation_seconds": validation["validation_seconds"],
    }
    _atomic_json_dump(summary, output_root / "dataset_summary.json")
    _atomic_json_dump(summary, complete_path)
    (output_root / "INCOMPLETE.json").unlink(missing_ok=True)
    return summary


def _parse_mapping(values: Sequence[str], *, option: str, value_type=str) -> dict[str, Any]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects LABEL=VALUE, got {value!r}")
        label, raw = value.split("=", 1)
        if label in result:
            raise ValueError(f"duplicate {option} label {label!r}")
        result[label] = value_type(raw)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=PATH", help="repeatable")
    parser.add_argument(
        "--expected-count",
        action="append",
        required=True,
        metavar="LABEL=COUNT",
        help="repeatable",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--speeds", nargs="+", default=("0.9", "0.95", "1.0", "1.05", "1.1"))
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--compression", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    sources = _parse_mapping(args.source, option="--source", value_type=Path)
    expected_counts = _parse_mapping(args.expected_count, option="--expected-count", value_type=int)
    if args.validate_only:
        result = validate_dataset(sources=sources, output_root=args.output_root, workers=args.workers)
    else:
        result = build_dataset(
            sources=sources,
            expected_counts=expected_counts,
            output_root=args.output_root,
            speeds=args.speeds,
            workers=args.workers,
            compression=args.compression,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
