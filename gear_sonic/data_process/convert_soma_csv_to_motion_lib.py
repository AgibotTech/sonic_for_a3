#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert flat A3 retargeted CSVs to motion-lib PKLs for SONIC 024.

The public release intentionally accepts only the A3 flat-CSV schema: one
motion per CSV, with root translation/rotation in centimeters/degrees and A3
joint columns in MJCF order.  It writes individual motion-lib PKLs suitable
for the A3 training, IsaacLab evaluation, and MuJoCo sim2sim paths.

Usage:
    python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
        --input a3_data/agibot_a3 \
        --output a3_data/motionlib_024_smoke \
        --individual --robot a3_29 --fps 30 --fps_source 120
"""

import argparse
import os
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial import transform

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A3_MJCF_PATH = os.path.join(
    REPO_ROOT, "gear_sonic", "data", "assets", "robot_description", "mjcf", "a3_t2d5.xml"
)
def load_a3_mjcf_spec(mjcf_path: str = A3_MJCF_PATH, robot: str = "a3") -> dict:
    """Load A3 body/joint order and joint axes from the MJCF body DFS order."""
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

    def visit(body):
        body_names.append(body.attrib["name"])
        for joint in body.findall("joint"):
            if joint.attrib.get("type") == "free":
                continue
            joint_names.append(joint.attrib["name"])
            axis = [float(v) for v in joint.attrib["axis"].split()]
            axes.append(axis)
        for child in body.findall("body"):
            visit(child)

    visit(root_body)
    if len(body_names) != len(joint_names) + 1:
        raise ValueError(
            f"Expected A3 to have one actuated joint per non-root body, "
            f"got {len(body_names)} bodies and {len(joint_names)} joints"
        )
    return {
        "robot": robot,
        "joint_names": joint_names,
        "body_names": body_names,
        "dof_axis": np.asarray(axes, dtype=np.float32),
        "num_dof": len(joint_names),
        "num_bodies": len(body_names),
    }


A3_SPEC = load_a3_mjcf_spec()
A3_HEAD_JOINTS = {"head_yaw_joint", "head_pitch_joint"}
A3_29D_SPEC = {
    **A3_SPEC,
    "robot": "a3_29",
    "input_joint_names": A3_SPEC["joint_names"],
    "head_dof_indices": [
        i for i, name in enumerate(A3_SPEC["joint_names"]) if name in A3_HEAD_JOINTS
    ],
    "dof_output_indices": [
        i for i, name in enumerate(A3_SPEC["joint_names"]) if name not in A3_HEAD_JOINTS
    ],
}


def detect_flat_csv_robot(data, robot: str) -> str:
    """Detect the supported A3 flat-CSV variant."""
    if robot != "auto":
        if robot not in {"a3", "a3_29"}:
            raise ValueError(f"Unsupported robot format '{robot}'; expected A3 or A3_29")
        return robot
    columns = set(data.columns)
    if _has_a3_joint_columns(columns, A3_29D_SPEC["input_joint_names"]):
        return "a3_29"
    if _has_a3_joint_columns(columns, A3_SPEC["joint_names"]):
        return "a3"
    raise ValueError(
        "Could not detect the A3 flat-CSV schema: expected all A3 joint columns "
        "(with an optional '_dof' suffix)."
    )


def _has_a3_joint_columns(columns: set[str], joint_names: list[str]) -> bool:
    """Return true when every A3 joint exists as name or name_dof."""
    return all(name in columns or f"{name}_dof" in columns for name in joint_names)


def _resolve_a3_joint_columns(data, joint_names: list[str]) -> list[str]:
    """Resolve A3 joint columns, accepting both name and name_dof headers."""
    columns = set(data.columns)
    joint_cols = []
    missing = []
    for name in joint_names:
        if name in columns:
            joint_cols.append(name)
        elif f"{name}_dof" in columns:
            joint_cols.append(f"{name}_dof")
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"A3 CSV is missing joint columns: {missing}")
    return joint_cols


def load_a3_flat_csv(csv_path: str, robot: str = "auto") -> dict:
    """Load one flat A3 CSV motion.

    Flat CSV format: Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, joint DOFs.
    All angles in degrees, positions in centimeters. A3 joint headers may use
    either joint names directly or the same names with a "_dof" suffix.
    """
    import pandas as pd

    data = pd.read_csv(csv_path)
    robot = detect_flat_csv_robot(data, robot)
    T = len(data)

    # Root position: cm → meters
    root_pos = (
        np.stack(
            [
                data["root_translateX"].values,  # noqa: PD011
                data["root_translateY"].values,  # noqa: PD011
                data["root_translateZ"].values,  # noqa: PD011
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )  # cm → m

    # Root rotation: Euler xyz (intrinsic) degrees → quaternion (xyzw scipy convention).
    euler_deg = np.stack(
        [
            data["root_rotateX"].values,  # noqa: PD011
            data["root_rotateY"].values,  # noqa: PD011
            data["root_rotateZ"].values,  # noqa: PD011
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
    )
    # Convert xyzw → wxyz for body_quat_w format
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    # Joint DOFs: degrees -> radians, already in A3 MJCF body-DFS order.
    spec = A3_SPEC if robot == "a3" else A3_29D_SPEC
    input_joint_names = spec.get("input_joint_names", spec["joint_names"])
    joint_cols = _resolve_a3_joint_columns(data, input_joint_names)
    joint_pos_mj = np.deg2rad(data[joint_cols].values).astype(np.float32)
    for head_idx in spec.get("head_dof_indices", []):
        joint_pos_mj[:, head_idx] = 0.0

    # Create dummy body_pos_w and body_quat_w (only root body populated, rest zeros)
    # The converter only uses body_pos_w[:,0] for root_trans and body_quat_w[:,0] for root_rot
    body_pos_w = np.zeros((T, 14, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w = np.zeros((T, 14, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0  # identity quaternion wxyz
    body_quat_w[:, 0, :] = root_quat_wxyz

    return {
        "joint_pos": joint_pos_mj,  # (T, 31) MuJoCo body-DFS order, radians
        "body_pos_w": body_pos_w,  # (T, 14, 3)
        "body_quat_w": body_quat_w,  # (T, 14, 4) wxyz
        "robot": spec["robot"],
        "joint_names": spec["joint_names"],
        "dof_axis": spec["dof_axis"],
        "num_dof": spec["num_dof"],
        "num_bodies": spec["num_bodies"],
        "dof_output_indices": spec.get("dof_output_indices"),
    }


def convert_sequence(seq_data: dict, fps: int) -> dict:
    """Convert one flat A3 sequence to the motion-lib format.

    Args:
        seq_data: dict with joint_pos (T, 31), body_pos_w (T, 14, 3),
                  body_quat_w (T, 14, 4 wxyz)
        fps: frame rate of the input data

    Returns:
        motion_lib entry dict with root_trans_offset, pose_aa, dof, root_rot, fps
    """
    joint_pos = seq_data["joint_pos"]  # (T, num_dof)
    body_pos_w = seq_data["body_pos_w"]  # (T, 14, 3)
    body_quat_w = seq_data["body_quat_w"]  # (T, 14, 4) wxyz
    dof_axis = seq_data["dof_axis"]
    num_dof = int(seq_data["num_dof"])
    num_bodies = int(seq_data["num_bodies"])
    dof_output_indices = seq_data.get("dof_output_indices")

    T = joint_pos.shape[0]

    # 1. Root position: body_0 (pelvis) position
    root_trans_offset = body_pos_w[:, 0, :].copy()  # (T, 3)

    # 2. Root quaternion: body_0 quaternion, convert wxyz → xyzw (scipy convention)
    root_quat_wxyz = body_quat_w[:, 0, :]  # (T, 4) [w, x, y, z]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]  # (T, 4) [x, y, z, w]

    # 3. Convert the MJCF-ordered DOFs to pose_aa using the bundled A3 axes.
    dof = joint_pos[:, :num_dof]
    dof_out = dof[:, dof_output_indices] if dof_output_indices is not None else dof

    # pose_aa[body_idx] = dof_axis * dof_value (axis-angle representation)
    # Body 0 = pelvis (root), bodies 1..N = actuated joints
    pose_aa = np.zeros((T, num_bodies, 3), dtype=np.float32)
    # Actuated joints: body idx = dof idx + 1
    pose_aa[:, 1:num_bodies, :] = dof_axis[None, :, :] * dof[:, :, None]

    # Set root rotation as axis-angle
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof_out.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),  # xyzw (scipy convention)
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),  # placeholder
        "fps": fps,
    }


def downsample_sequence(entry: dict, fps_source: int, fps_target: int) -> dict:
    """Downsample a motion_lib entry using stride-based frame skipping.

    Matches process_bones_to_motionlib.py: jump = int(fps_source / fps_target).
    Best used when fps_source is an exact multiple of fps_target (e.g. 120→30).
    The resulting PKL is stored at fps_target; fk_batch handles the final
    resampling to target_fps at load time using the canonical interploate_pose formula.
    """
    if fps_source == fps_target:
        return entry
    jump = int(fps_source / fps_target)
    if jump <= 1:
        return entry
    return {
        "root_trans_offset": entry["root_trans_offset"][::jump],
        "pose_aa": entry["pose_aa"][::jump],
        "dof": entry["dof"][::jump],
        "root_rot": entry["root_rot"][::jump],
        "smpl_joints": entry["smpl_joints"][::jump],
        "fps": fps_target,
    }


def process_session_csvs(args_tuple):
    """Process all CSVs in a single session directory. Used by multiprocessing."""
    session_dir, session_name, out_dir, fps, fps_source, robot = args_tuple
    import warnings

    warnings.filterwarnings("ignore")

    csv_files = sorted([f for f in os.listdir(session_dir) if f.endswith(".csv")])

    session_out = os.path.join(out_dir, session_name)
    os.makedirs(session_out, exist_ok=True)

    converted = 0
    failed = 0
    for csv_f in csv_files:
        name = os.path.splitext(csv_f)[0]
        out_path = os.path.join(session_out, name + ".pkl")
        if os.path.exists(out_path):
            converted += 1  # skip existing
            continue
        try:
            seq = load_a3_flat_csv(os.path.join(session_dir, csv_f), robot=robot)
            fps_for_convert = fps_source if fps_source else fps
            entry = convert_sequence(seq, fps_for_convert)
            if fps_source and fps_source != fps:
                entry = downsample_sequence(entry, fps_source, fps)
            joblib.dump({name: entry}, out_path, compress=True)
            converted += 1
        except Exception as err:  # noqa: BLE001
            print(f"WARNING: failed to convert {session_name}/{csv_f}: {err}")
            failed += 1
    return session_name, converted, failed, len(csv_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert flat A3 CSVs to individual SONIC 024 motion-lib PKLs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory of flat A3 CSVs, or a parent directory containing such session directories.",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for per-motion PKLs."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target output FPS (default: 30).",
    )
    parser.add_argument(
        "--fps_source",
        type=int,
        default=None,
        help="Source data FPS. If set and != --fps, data is downsampled. "
        "The bundled A3 CSV examples are typically 120fps.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Required public-release mode: write one PKL per A3 CSV.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--robot",
        choices=("auto", "a3", "a3_29"),
        default="auto",
        help=(
            "Flat CSV robot format. 'a3' keeps head joints as 31D; 'a3_29' reads A3 CSVs "
            "and omits head_yaw/head_pitch from the 29D policy view."
        ),
    )
    args = parser.parse_args()

    print(
        f"Supported flat CSV profiles: A3 {A3_SPEC['num_dof']} DOFs/{A3_SPEC['num_bodies']} bodies, "
        f"A3_29 29 DOFs/{A3_29D_SPEC['num_bodies']} bodies"
    )

    if not args.individual:
        parser.error("This A3-only release supports only --individual flat-CSV conversion.")
    if not os.path.isdir(args.input):
        parser.error("--input must be a directory containing flat A3 CSVs.")

    # Accept either one CSV session directory or one parent directory of sessions.
    input_entries = os.listdir(args.input)
    if any(entry.endswith(".csv") for entry in input_entries):
        session_dirs = [
            (
                args.input,
                os.path.basename(args.input.rstrip("/")),
                args.output,
                args.fps,
                args.fps_source,
                args.robot,
            )
        ]
    else:
        session_dirs = [
            (path, name, args.output, args.fps, args.fps_source, args.robot)
            for name in sorted(input_entries)
            if os.path.isdir(path := os.path.join(args.input, name))
            and any(entry.endswith(".csv") for entry in os.listdir(path))
        ]
    if not session_dirs:
        parser.error("No flat A3 CSV files were found under --input.")

    print(f"\nBatch converting {len(session_dirs)} sessions with {args.num_workers} workers")
    print(f"Output: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    import multiprocessing

    total_converted = 0
    total_failed = 0
    total_csvs = 0
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for session_name, converted, failed, n_csvs in pool.imap_unordered(
            process_session_csvs, session_dirs
        ):
            total_converted += converted
            total_failed += failed
            total_csvs += n_csvs
            print(
                f"  {session_name}: {converted}/{n_csvs} converted"
                + (f" ({failed} failed)" if failed else "")
            )

    print(
        f"\nDone: {total_converted} motions converted, {total_failed} failed, {total_csvs} total CSVs"
    )


if __name__ == "__main__":
    main()
