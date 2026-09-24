#!/usr/bin/env python3
# ruff: noqa: T201
"""Convert A3 motion-lib-like PKLs to the 29D motion_lib format.

The source files under ``a3_data/output_pkl_v2v3`` already contain A3
``root_trans_offset`` and ``pose_aa`` data in MuJoCo body DFS order. Some of
them also contain a 31D ``dof`` array. This script normalizes those entries to
the 29D format used by the current A3 training path:

    root_trans_offset, pose_aa, dof, root_rot, smpl_joints, fps

Head joints are present in the A3 MJCF and pose data, but the training policy
uses the 29 non-head DOFs. The output keeps the 32-body pose_aa shape for FK
compatibility and zeros the head joint bodies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "a3_data/output_pkl_v2v3"
DEFAULT_OUTPUT = REPO_ROOT / "data/agibot_a3_motion_lib_29d/from_pkl/agibot_a3"
DEFAULT_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5.xml"
HEAD_JOINT_NAMES = {"head_yaw_joint", "head_pitch_joint"}
REQUIRED_ENTRY_KEYS = {"root_trans_offset", "pose_aa"}


@dataclass(frozen=True)
class A3Spec:
    joint_names: list[str]
    body_names: list[str]
    dof_axis: np.ndarray
    head_dof_indices: list[int]
    dof_output_indices: list[int]

    @property
    def num_dof(self) -> int:
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def head_body_indices(self) -> list[int]:
        return [idx + 1 for idx in self.head_dof_indices]


def load_a3_mjcf_spec(mjcf_path: Path = DEFAULT_MJCF) -> A3Spec:
    """Load A3 body, joint, and hinge-axis order from the MJCF body DFS order."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in {mjcf_path}")
    root_body = worldbody.find("body")
    if root_body is None:
        raise ValueError(f"Missing root <body> in {mjcf_path}")

    body_names: list[str] = []
    joint_names: list[str] = []
    axes: list[list[float]] = []

    def visit(body: ET.Element) -> None:
        body_names.append(body.attrib["name"])
        for joint in body.findall("joint"):
            if joint.attrib.get("type") == "free":
                continue
            joint_names.append(joint.attrib["name"])
            axes.append([float(v) for v in joint.attrib["axis"].split()])
        for child in body.findall("body"):
            visit(child)

    visit(root_body)
    if len(body_names) != len(joint_names) + 1:
        raise ValueError(
            f"Expected one actuated joint per non-root body, got "
            f"{len(body_names)} bodies and {len(joint_names)} joints"
        )

    head_dof_indices = [i for i, name in enumerate(joint_names) if name in HEAD_JOINT_NAMES]
    dof_output_indices = [
        i for i, name in enumerate(joint_names) if name not in HEAD_JOINT_NAMES
    ]
    if len(head_dof_indices) != len(HEAD_JOINT_NAMES):
        raise ValueError(
            f"Expected head joints {sorted(HEAD_JOINT_NAMES)}, found indices {head_dof_indices}"
        )

    return A3Spec(
        joint_names=joint_names,
        body_names=body_names,
        dof_axis=np.asarray(axes, dtype=np.float32),
        head_dof_indices=head_dof_indices,
        dof_output_indices=dof_output_indices,
    )


def is_motion_entry(value: object) -> bool:
    return isinstance(value, dict) and REQUIRED_ENTRY_KEYS.issubset(value.keys())


def iter_pkl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".pkl":
            raise ValueError(f"Input file must be a .pkl file: {input_path}")
        if input_path.name == "metadata.pkl":
            return []
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.glob("*.pkl") if p.name != "metadata.pkl"
        )

    raise FileNotFoundError(input_path)


def extract_entries(pkl_path: Path) -> list[tuple[str, dict]]:
    data = joblib.load(pkl_path)
    if is_motion_entry(data):
        return [(pkl_path.stem, data)]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported PKL top-level type: {type(data).__name__}")

    entries: list[tuple[str, dict]] = []
    for name, value in data.items():
        if is_motion_entry(value):
            entries.append((str(name), value))

    if not entries:
        raise ValueError("No motion entries with root_trans_offset and pose_aa found")
    return entries


def _as_float32_array(entry: dict, key: str) -> np.ndarray:
    return np.asarray(entry[key], dtype=np.float32)


def _derive_dof_from_pose(pose_aa: np.ndarray, spec: A3Spec) -> np.ndarray:
    joint_pose_aa = pose_aa[:, 1 : spec.num_bodies, :]
    return np.einsum("tij,ij->ti", joint_pose_aa, spec.dof_axis).astype(np.float32)


def _validate_raw_dof_against_pose(
    name: str,
    raw_dof: np.ndarray,
    pose_aa: np.ndarray,
    spec: A3Spec,
    tolerance: float = 1e-3,
) -> str | None:
    if raw_dof.shape[1] != spec.num_dof:
        return None

    pose_dof = _derive_dof_from_pose(pose_aa, spec)
    keep = spec.dof_output_indices
    max_diff = float(np.max(np.abs(raw_dof[:, keep] - pose_dof[:, keep])))
    if max_diff > tolerance:
        return (
            f"{name}: raw non-head dof differs from pose-derived dof by "
            f"max {max_diff:.6g}; using raw dof"
        )
    return None


def normalize_entry(
    name: str,
    entry: dict,
    spec: A3Spec,
    target_fps: int,
    fps_source_override: float | None = None,
) -> tuple[dict, dict, list[str]]:
    root_trans = _as_float32_array(entry, "root_trans_offset")
    pose_aa = _as_float32_array(entry, "pose_aa").copy()

    if root_trans.ndim != 2 or root_trans.shape[1] != 3:
        raise ValueError(f"{name}: root_trans_offset must have shape (T, 3)")
    if pose_aa.ndim != 3 or pose_aa.shape[1:] != (spec.num_bodies, 3):
        raise ValueError(
            f"{name}: pose_aa must have shape (T, {spec.num_bodies}, 3), "
            f"got {pose_aa.shape}"
        )
    if pose_aa.shape[0] != root_trans.shape[0]:
        raise ValueError(
            f"{name}: root_trans_offset length {root_trans.shape[0]} does not match "
            f"pose_aa length {pose_aa.shape[0]}"
        )

    notes: list[str] = []
    for body_idx in spec.head_body_indices:
        pose_aa[:, body_idx, :] = 0.0

    raw_dof = entry.get("dof")
    if raw_dof is not None:
        raw_dof = np.asarray(raw_dof, dtype=np.float32)
        if raw_dof.ndim != 2 or raw_dof.shape[0] != pose_aa.shape[0]:
            raise ValueError(f"{name}: dof must have shape (T, N)")
        if raw_dof.shape[1] == spec.num_dof:
            warning = _validate_raw_dof_against_pose(name, raw_dof, pose_aa, spec)
            if warning:
                notes.append(warning)
            dof = raw_dof[:, spec.dof_output_indices]
        elif raw_dof.shape[1] == len(spec.dof_output_indices):
            dof = raw_dof
        else:
            raise ValueError(
                f"{name}: dof must be {spec.num_dof}D or "
                f"{len(spec.dof_output_indices)}D, got {raw_dof.shape[1]}D"
            )
    else:
        dof_full = _derive_dof_from_pose(pose_aa, spec)
        dof = dof_full[:, spec.dof_output_indices]

    source_fps = float(fps_source_override or entry.get("fps", target_fps))
    motion = {
        "root_trans_offset": root_trans.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": Rotation.from_rotvec(pose_aa[:, 0, :].astype(np.float64))
        .as_quat()
        .astype(np.float32),
        "smpl_joints": np.zeros((pose_aa.shape[0], 24, 3), dtype=np.float32),
        "fps": int(round(source_fps)),
    }

    if abs(source_fps - float(target_fps)) > 1e-6:
        motion = downsample_entry(name, motion, source_fps, target_fps, notes)
    else:
        motion["fps"] = target_fps

    metadata = {
        "length": int(motion["root_trans_offset"].shape[0]),
        "fps": float(motion["fps"]),
        "root_range_xyz": (
            motion["root_trans_offset"].max(axis=0)
            - motion["root_trans_offset"].min(axis=0)
        ).astype(float).tolist(),
    }
    return motion, metadata, notes


def downsample_entry(
    name: str,
    motion: dict,
    source_fps: float,
    target_fps: int,
    notes: list[str],
) -> dict:
    if source_fps < target_fps:
        raise ValueError(
            f"{name}: source fps {source_fps:g} is lower than target fps {target_fps}; "
            "upsampling is not supported"
        )

    stride_float = source_fps / float(target_fps)
    stride = int(stride_float)
    if stride <= 1:
        motion["fps"] = target_fps
        return motion

    if abs(stride_float - stride) > 1e-6:
        notes.append(
            f"{name}: source fps {source_fps:g} is not an exact multiple of "
            f"{target_fps}; using stride {stride}"
        )

    return {
        "root_trans_offset": motion["root_trans_offset"][::stride],
        "pose_aa": motion["pose_aa"][::stride],
        "dof": motion["dof"][::stride],
        "root_rot": motion["root_rot"][::stride],
        "smpl_joints": motion["smpl_joints"][::stride],
        "fps": target_fps,
    }


def load_existing_metadata(output_dir: Path, overwrite: bool) -> dict:
    metadata_path = output_dir / "metadata.pkl"
    if overwrite or not metadata_path.exists():
        return {}
    try:
        metadata = joblib.load(metadata_path)
    except Exception:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def convert_all(args: argparse.Namespace) -> int:
    spec = load_a3_mjcf_spec(args.mjcf)
    pkl_files = iter_pkl_files(args.input)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(
        f"A3 spec: {spec.num_dof} DOFs/{spec.num_bodies} bodies, "
        f"head dof indices={spec.head_dof_indices}"
    )
    print(f"Found {len(pkl_files)} PKL files (metadata.pkl skipped)")

    if not pkl_files:
        print("No input PKL files found")
        return 1

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    metadata = load_existing_metadata(args.output, args.overwrite)
    seen_names: set[str] = set()
    converted = 0
    skipped = 0
    failed = 0
    total_entries = 0
    notes_seen = 0

    for pkl_path in pkl_files:
        try:
            entries = extract_entries(pkl_path)
        except Exception as exc:
            failed += 1
            print(f"  FAIL {pkl_path.name}: {exc}")
            continue

        for name, entry in entries:
            total_entries += 1
            if name in seen_names:
                skipped += 1
                print(f"  SKIP {name}: duplicate motion name in input set")
                continue
            seen_names.add(name)

            out_path = args.output / f"{name}.pkl"
            if not args.dry_run and out_path.exists() and not args.overwrite:
                skipped += 1
                print(f"  SKIP {name}: output exists")
                continue

            try:
                motion, motion_metadata, notes = normalize_entry(
                    name=name,
                    entry=entry,
                    spec=spec,
                    target_fps=args.fps,
                    fps_source_override=args.fps_source,
                )
            except Exception as exc:
                failed += 1
                print(f"  FAIL {name}: {exc}")
                continue

            notes_seen += len(notes)
            for note in notes[:3]:
                print(f"  NOTE {note}")
            if len(notes) > 3:
                print(f"  NOTE {name}: {len(notes) - 3} additional notes suppressed")

            if args.dry_run:
                converted += 1
                continue

            joblib.dump({name: motion}, out_path, compress=True)
            metadata[name] = motion_metadata
            converted += 1

    if not args.dry_run:
        joblib.dump(metadata, args.output / "metadata.pkl", compress=True)

    action = "Validated" if args.dry_run else "Converted"
    print(
        f"\n{action}: {converted}/{total_entries} entries, "
        f"skipped={skipped}, failed={failed}, notes={notes_seen}"
    )
    if args.dry_run:
        print("Dry run only; no files were written")
    else:
        print(f"Saved metadata: {args.output / 'metadata.pkl'}")

    return 0 if failed == 0 else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert A3 motion-lib-like PKLs to 29D motion_lib PKLs."
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
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target output FPS")
    parser.add_argument(
        "--fps-source",
        type=float,
        default=None,
        help="Override source FPS. Defaults to each entry's fps field.",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=DEFAULT_MJCF,
        help=f"A3 MJCF path (default: {DEFAULT_MJCF})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PKLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without writing output files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return convert_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
