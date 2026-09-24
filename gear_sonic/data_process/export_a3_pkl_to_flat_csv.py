#!/usr/bin/env python3
# ruff: noqa: T201
"""Export A3 PKLs to flat A3 CSV files.

The output CSV schema matches the flat A3 CSVs consumed by
``convert_soma_csv_to_motion_lib.py``:

    Frame,
    root_translateX, root_translateY, root_translateZ,      # centimeters
    root_rotateX, root_rotateY, root_rotateZ,               # xyz Euler, degrees
    <31 A3 joint columns in MJCF order>                      # degrees

The exporter supports both motion-lib-like A3 PKLs and the original A3 PKL
layout documented in ``a3_data/base_model_data_original/original_pkl_description.md``.
By default it preserves the source FPS/frame count. Pass ``--fps 30`` to
stride-downsample 120fps sources to 30fps.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import multiprocessing as mp
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

_CONVERTER_PATH = Path(__file__).with_name("convert_a3_pkl_to_motion_lib_29d.py")
_CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "convert_a3_pkl_to_motion_lib_29d", _CONVERTER_PATH
)
if _CONVERTER_SPEC is None or _CONVERTER_SPEC.loader is None:
    raise ImportError(f"Could not load {_CONVERTER_PATH}")
_converter = importlib.util.module_from_spec(_CONVERTER_SPEC)
sys.modules[_CONVERTER_SPEC.name] = _converter
_CONVERTER_SPEC.loader.exec_module(_converter)

DEFAULT_INPUT = _converter.DEFAULT_INPUT
DEFAULT_MJCF = _converter.DEFAULT_MJCF
REPO_ROOT = _converter.REPO_ROOT
A3Spec = _converter.A3Spec
_derive_dof_from_pose = _converter._derive_dof_from_pose
iter_pkl_files = _converter.iter_pkl_files
load_a3_mjcf_spec = _converter.load_a3_mjcf_spec
is_motion_lib_entry = _converter.is_motion_entry


DEFAULT_OUTPUT = REPO_ROOT / "a3_data/agibot_a3_from_pkl_csv"
WORKER_SPEC: A3Spec | None = None
ORIGINAL_REQUIRED_ENTRY_KEYS = {"root_trans_offset", "dof", "root_rot", "fps"}
ORIGINAL_DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def is_original_pkl_entry(value: object) -> bool:
    return isinstance(value, dict) and ORIGINAL_REQUIRED_ENTRY_KEYS.issubset(value.keys())


def _period_sort_key(period_idx: object) -> tuple[int, int | str]:
    text = str(period_idx)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def extract_motion_lib_entries_from_data(data: object, pkl_path: Path) -> list[tuple[str, str, dict]]:
    if is_motion_lib_entry(data):
        return [(pkl_path.stem, "", data)]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported PKL top-level type: {type(data).__name__}")

    entries: list[tuple[str, str, dict]] = []
    for name, value in data.items():
        if is_motion_lib_entry(value):
            entries.append((str(name), "", value))

    if not entries:
        raise ValueError("No motion-lib entries with root_trans_offset and pose_aa found")
    return entries


def extract_original_entries_from_data(data: object, pkl_path: Path) -> list[tuple[str, str, dict]]:
    if is_original_pkl_entry(data):
        return [(pkl_path.stem, "", data)]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported PKL top-level type: {type(data).__name__}")

    period_entries = [
        (str(period_idx), value)
        for period_idx, value in sorted(data.items(), key=lambda item: _period_sort_key(item[0]))
        if is_original_pkl_entry(value)
    ]
    if not period_entries:
        raise ValueError("No original A3 period entries with root_trans_offset/dof/root_rot/fps found")

    if len(period_entries) == 1:
        period_idx, entry = period_entries[0]
        return [(pkl_path.stem, period_idx, entry)]

    return [
        (f"{pkl_path.stem}__period_{period_idx}", period_idx, entry)
        for period_idx, entry in period_entries
    ]


def extract_entries_for_format(
    pkl_path: Path, source_format: str
) -> tuple[list[tuple[str, str, dict]], str]:
    data = joblib.load(pkl_path)

    if source_format == "motion-lib":
        return extract_motion_lib_entries_from_data(data, pkl_path), "motion-lib"
    if source_format == "original-pkl":
        return extract_original_entries_from_data(data, pkl_path), "original-pkl"

    try:
        return extract_motion_lib_entries_from_data(data, pkl_path), "motion-lib"
    except ValueError:
        return extract_original_entries_from_data(data, pkl_path), "original-pkl"


def _full_dof_from_raw_or_pose(entry: dict, pose_aa: np.ndarray, spec: A3Spec) -> np.ndarray:
    raw_dof = entry.get("dof")
    if raw_dof is None:
        return _derive_dof_from_pose(pose_aa, spec)

    raw_dof = np.asarray(raw_dof, dtype=np.float32)
    if raw_dof.ndim != 2 or raw_dof.shape[0] != pose_aa.shape[0]:
        raise ValueError(f"dof must have shape (T, N), got {raw_dof.shape}")

    if raw_dof.shape[1] == spec.num_dof:
        return raw_dof
    if raw_dof.shape[1] == len(spec.dof_output_indices):
        full_dof = np.zeros((raw_dof.shape[0], spec.num_dof), dtype=np.float32)
        full_dof[:, spec.dof_output_indices] = raw_dof
        return full_dof

    raise ValueError(
        f"dof must be {spec.num_dof}D or {len(spec.dof_output_indices)}D, "
        f"got {raw_dof.shape[1]}D"
    )


def entry_to_flat_csv_arrays(
    name: str,
    entry: dict,
    spec: A3Spec,
    fps_target: float | None,
    fps_source_override: float | None,
    dof_source: str,
) -> tuple[np.ndarray, float, list[str]]:
    root_trans_m = np.asarray(entry["root_trans_offset"], dtype=np.float32)
    pose_aa = np.asarray(entry["pose_aa"], dtype=np.float32)

    if root_trans_m.ndim != 2 or root_trans_m.shape[1] != 3:
        raise ValueError(f"{name}: root_trans_offset must have shape (T, 3)")
    if pose_aa.ndim != 3 or pose_aa.shape[1:] != (spec.num_bodies, 3):
        raise ValueError(
            f"{name}: pose_aa must have shape (T, {spec.num_bodies}, 3), "
            f"got {pose_aa.shape}"
        )
    if root_trans_m.shape[0] != pose_aa.shape[0]:
        raise ValueError(
            f"{name}: root_trans_offset length {root_trans_m.shape[0]} does not match "
            f"pose_aa length {pose_aa.shape[0]}"
        )

    if dof_source == "raw":
        dof_rad = _full_dof_from_raw_or_pose(entry, pose_aa, spec)
    else:
        dof_rad = _derive_dof_from_pose(pose_aa, spec)

    source_fps = float(fps_source_override or entry.get("fps", fps_target or 0.0) or 0.0)
    effective_fps = source_fps
    stride = 1
    notes: list[str] = []

    if fps_target is not None:
        if source_fps <= 0:
            raise ValueError(f"{name}: source fps is required when --fps is set")
        if source_fps < fps_target:
            raise ValueError(
                f"{name}: source fps {source_fps:g} is lower than target fps "
                f"{fps_target:g}; upsampling is not supported"
            )
        stride_float = source_fps / fps_target
        stride = int(stride_float)
        if stride <= 0:
            stride = 1
        if abs(stride_float - stride) > 1e-6:
            notes.append(
                f"{name}: source fps {source_fps:g} is not an exact multiple of "
                f"{fps_target:g}; using stride {stride}"
            )
        effective_fps = fps_target

    if stride > 1:
        root_trans_m = root_trans_m[::stride]
        pose_aa = pose_aa[::stride]
        dof_rad = dof_rad[::stride]

    frames = np.arange(root_trans_m.shape[0], dtype=np.float64)[:, None]
    root_trans_cm = root_trans_m.astype(np.float64) * 100.0
    root_euler_deg = Rotation.from_rotvec(pose_aa[:, 0, :].astype(np.float64)).as_euler(
        "xyz", degrees=True
    )
    dof_deg = np.rad2deg(dof_rad.astype(np.float64))

    table = np.concatenate([frames, root_trans_cm, root_euler_deg, dof_deg], axis=1)
    return table, effective_fps, notes


def _apply_optional_stride(
    name: str,
    source_fps: float,
    fps_target: float | None,
    arrays: tuple[np.ndarray, ...],
) -> tuple[float, tuple[np.ndarray, ...], list[str]]:
    effective_fps = source_fps
    notes: list[str] = []

    if fps_target is None:
        return effective_fps, arrays, notes

    if source_fps <= 0:
        raise ValueError(f"{name}: source fps is required when --fps is set")
    if source_fps < fps_target:
        raise ValueError(
            f"{name}: source fps {source_fps:g} is lower than target fps "
            f"{fps_target:g}; upsampling is not supported"
        )

    stride_float = source_fps / fps_target
    stride = int(stride_float)
    if stride <= 0:
        stride = 1
    if abs(stride_float - stride) > 1e-6:
        notes.append(
            f"{name}: source fps {source_fps:g} is not an exact multiple of "
            f"{fps_target:g}; using stride {stride}"
        )

    if stride > 1:
        arrays = tuple(array[::stride] for array in arrays)
    effective_fps = fps_target
    return effective_fps, arrays, notes


def original_dof_to_a3_csv_dof(
    name: str,
    dof_rad: np.ndarray,
    spec: A3Spec,
    head_mapping: str,
) -> np.ndarray:
    dof_rad = np.asarray(dof_rad, dtype=np.float32)
    if dof_rad.ndim != 2 or dof_rad.shape[1] != len(ORIGINAL_DOF_NAMES):
        raise ValueError(
            f"{name}: original dof must have shape (T, {len(ORIGINAL_DOF_NAMES)}), "
            f"got {dof_rad.shape}"
        )

    source_index = {joint_name: idx for idx, joint_name in enumerate(ORIGINAL_DOF_NAMES)}
    csv_dof = np.zeros((dof_rad.shape[0], spec.num_dof), dtype=np.float32)

    for target_idx, joint_name in enumerate(spec.joint_names):
        if joint_name in source_index:
            csv_dof[:, target_idx] = dof_rad[:, source_index[joint_name]]
        elif joint_name == "head_yaw_joint" and head_mapping == "head_yaw":
            csv_dof[:, target_idx] = dof_rad[:, source_index["head_joint"]]
        elif joint_name == "head_pitch_joint" and head_mapping == "head_pitch":
            csv_dof[:, target_idx] = dof_rad[:, source_index["head_joint"]]

    return csv_dof


def original_entry_to_flat_csv_arrays(
    name: str,
    entry: dict,
    spec: A3Spec,
    fps_target: float | None,
    fps_source_override: float | None,
    head_mapping: str,
) -> tuple[np.ndarray, float, list[str]]:
    root_trans_m = np.asarray(entry["root_trans_offset"], dtype=np.float32)
    root_rot_xyzw = np.asarray(entry["root_rot"], dtype=np.float32)
    dof_rad = original_dof_to_a3_csv_dof(name, entry["dof"], spec, head_mapping)

    if root_trans_m.ndim != 2 or root_trans_m.shape[1] != 3:
        raise ValueError(f"{name}: root_trans_offset must have shape (T, 3)")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"{name}: root_rot must have shape (T, 4)")
    if root_trans_m.shape[0] != root_rot_xyzw.shape[0] or root_trans_m.shape[0] != dof_rad.shape[0]:
        raise ValueError(
            f"{name}: root_trans_offset/root_rot/dof frame counts differ: "
            f"{root_trans_m.shape[0]}, {root_rot_xyzw.shape[0]}, {dof_rad.shape[0]}"
        )

    source_fps = float(fps_source_override or entry.get("fps", fps_target or 0.0) or 0.0)
    effective_fps, arrays, notes = _apply_optional_stride(
        name, source_fps, fps_target, (root_trans_m, root_rot_xyzw, dof_rad)
    )
    root_trans_m, root_rot_xyzw, dof_rad = arrays

    frames = np.arange(root_trans_m.shape[0], dtype=np.float64)[:, None]
    root_trans_cm = root_trans_m.astype(np.float64) * 100.0
    root_euler_deg = Rotation.from_quat(root_rot_xyzw.astype(np.float64)).as_euler(
        "xyz", degrees=True
    )
    dof_deg = np.rad2deg(dof_rad.astype(np.float64))

    table = np.concatenate([frames, root_trans_cm, root_euler_deg, dof_deg], axis=1)
    return table, effective_fps, notes


def write_flat_csv(path: Path, table: np.ndarray, spec: A3Spec) -> None:
    header = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *spec.joint_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in table:
            writer.writerow([str(int(row[0])), *[f"{v:.9g}" for v in row[1:]]])


def write_metadata_csv(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "name",
        "source_format",
        "source",
        "source_pkl",
        "period_idx",
        "csv",
        "frames",
        "fps",
        "head_mapping",
    ]
    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def init_worker(mjcf_path: Path) -> None:
    global WORKER_SPEC
    WORKER_SPEC = load_a3_mjcf_spec(mjcf_path)


def process_pkl_file(task: tuple) -> dict:
    (
        pkl_path,
        output_dir,
        source_format,
        fps_target,
        fps_source,
        dof_source,
        original_head_joint_to,
        dry_run,
        overwrite,
    ) = task
    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)
    spec = WORKER_SPEC
    if spec is None:
        raise RuntimeError("worker A3 spec was not initialized")

    result = {
        "exported": 0,
        "skipped": 0,
        "failed": 0,
        "total_entries": 0,
        "notes_seen": 0,
        "metadata_rows": [],
        "messages": [],
    }

    try:
        entries, detected_source_format = extract_entries_for_format(pkl_path, source_format)
    except Exception as exc:
        result["failed"] += 1
        result["messages"].append(f"  FAIL {pkl_path.name}: {exc}")
        return result

    for name, period_idx, entry in entries:
        result["total_entries"] += 1
        out_path = output_dir / f"{name}.csv"
        if not dry_run and out_path.exists() and not overwrite:
            result["skipped"] += 1
            result["messages"].append(f"  SKIP {name}: output exists")
            continue

        try:
            if detected_source_format == "original-pkl":
                table, fps, notes = original_entry_to_flat_csv_arrays(
                    name=name,
                    entry=entry,
                    spec=spec,
                    fps_target=fps_target,
                    fps_source_override=fps_source,
                    head_mapping=original_head_joint_to,
                )
            else:
                table, fps, notes = entry_to_flat_csv_arrays(
                    name=name,
                    entry=entry,
                    spec=spec,
                    fps_target=fps_target,
                    fps_source_override=fps_source,
                    dof_source=dof_source,
                )
        except Exception as exc:
            result["failed"] += 1
            result["messages"].append(f"  FAIL {name}: {exc}")
            continue

        result["notes_seen"] += len(notes)
        result["messages"].extend(f"  NOTE {note}" for note in notes[:3])

        if not dry_run:
            write_flat_csv(out_path, table, spec)
            result["metadata_rows"].append(
                {
                    "name": name,
                    "source_format": detected_source_format,
                    "source": str(pkl_path),
                    "source_pkl": str(pkl_path),
                    "period_idx": period_idx,
                    "csv": str(out_path),
                    "frames": int(table.shape[0]),
                    "fps": f"{fps:g}" if fps else "",
                    "head_mapping": (
                        original_head_joint_to
                        if detected_source_format == "original-pkl"
                        else ""
                    ),
                }
            )
        result["exported"] += 1

    return result


def export_all(args: argparse.Namespace) -> int:
    spec = load_a3_mjcf_spec(args.mjcf)
    pkl_files = iter_pkl_files(args.input)
    if args.limit is not None:
        pkl_files = pkl_files[: args.limit]
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"A3 CSV columns: 7 root/frame columns + {spec.num_dof} joints")
    print(f"Found {len(pkl_files)} PKL files (metadata.pkl skipped)")
    print(f"Workers: {args.num_workers}")

    if not pkl_files:
        print("No input PKL files found")
        return 1

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            pkl_path,
            args.output,
            args.source_format,
            args.fps,
            args.fps_source,
            args.dof_source,
            args.original_head_joint_to,
            args.dry_run,
            args.overwrite,
        )
        for pkl_path in pkl_files
    ]

    metadata_rows: list[dict] = []
    exported = 0
    skipped = 0
    failed = 0
    total_entries = 0
    notes_seen = 0

    worker_count = max(1, args.num_workers)
    pool = None
    if worker_count == 1:
        init_worker(args.mjcf)
        result_iter = (process_pkl_file(task) for task in tasks)
    else:
        pool = mp.Pool(processes=worker_count, initializer=init_worker, initargs=(args.mjcf,))
        result_iter = pool.imap_unordered(process_pkl_file, tasks)

    try:
        for done, result in enumerate(result_iter, start=1):
            exported += result["exported"]
            skipped += result["skipped"]
            failed += result["failed"]
            total_entries += result["total_entries"]
            notes_seen += result["notes_seen"]
            metadata_rows.extend(result["metadata_rows"])
            for message in result["messages"]:
                print(message)
            if done == len(tasks) or done % 100 == 0:
                print(
                    f"  progress {done}/{len(tasks)} files: "
                    f"exported={exported}, skipped={skipped}, failed={failed}"
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if not args.dry_run:
        metadata_rows.sort(key=lambda row: row["csv"])
        write_metadata_csv(args.output, metadata_rows)

    action = "Validated" if args.dry_run else "Exported"
    print(
        f"\n{action}: {exported}/{total_entries} entries, "
        f"skipped={skipped}, failed={failed}, notes={notes_seen}"
    )
    if args.dry_run:
        print("Dry run only; no files were written")
    else:
        print(f"Saved CSVs under: {args.output}")
        print(f"Saved metadata: {args.output / 'metadata.csv'}")

    return 0 if failed == 0 else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export A3 PKLs to flat A3 CSV files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input PKL file or directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional target FPS. Defaults to preserving source frame count.",
    )
    parser.add_argument(
        "--fps-source",
        type=float,
        default=None,
        help="Override source FPS. Defaults to each entry's fps field.",
    )
    parser.add_argument(
        "--dof-source",
        choices=("pose", "raw"),
        default="pose",
        help=(
            "Use pose_aa-derived 31D joint angles or raw entry['dof'] when available. "
            "Default: pose."
        ),
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "motion-lib", "original-pkl"),
        default="auto",
        help=(
            "Input PKL layout. 'original-pkl' expects period entries with "
            "root_trans_offset/dof/root_rot/fps. Default: auto."
        ),
    )
    parser.add_argument(
        "--original-head-joint-to",
        choices=("head_yaw", "head_pitch", "zero"),
        default="zero",
        help=(
            "How to map the original 30D head_joint into the 31D A3 CSV head columns. "
            "Only used for --source-format original-pkl or auto-detected original PKLs."
        ),
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=DEFAULT_MJCF,
        help=f"A3 MJCF path (default: {DEFAULT_MJCF})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without writing CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N input PKL files after sorting.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for PKL-to-CSV export. Default: 1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return export_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
