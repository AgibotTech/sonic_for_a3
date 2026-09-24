#!/usr/bin/env python3
"""Minimal PICO body-pose ZMQ sender for A3 SMPL teleop.

This sender intentionally avoids the full desktop teleop stack.  It does not
import torch, pinocchio, robot models, hand IK, or visualization code.  It reads
XRoboToolkit body joint poses and publishes the packed "pose" topic consumed by
A3ZmqSmplSource.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import zmq

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None


HEADER_SIZE = 1280
DEFAULT_HUMAN_JOINTS_NPZ = (
    Path(__file__).resolve().parents[1] / "data" / "human" / "human_joints_info.npz"
)
SMPL_OUTPUT_JOINT_INDEX = np.concatenate([np.arange(22), np.array([39, 54])])
Y_TO_Z_UP_QUAT_WXYZ = R.from_rotvec([np.pi / 2.0, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
SMPL_BASE_ROT_CONJ_WXYZ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
PARENT_INDICES = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
]


def _build_header(fields: list[dict], version: int = 3, count: int = 1) -> bytes:
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def pack_pose_message(pose_data: dict[str, np.ndarray], topic: str = "pose") -> bytes:
    fields = []
    chunks = []
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            value = value.astype(np.float32)
            dtype_str = "f32"
        value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        chunks.append(value.tobytes())
    return topic.encode("utf-8") + _build_header(fields) + b"".join(chunks)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.split(a, 4, axis=1)
    bw, bx, by, bz = np.split(b, 4, axis=1)
    return np.concatenate(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(norm > 1e-12, q / np.maximum(norm, 1e-12), np.array([1.0, 0.0, 0.0, 0.0]))


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    rot = R.from_quat(_quat_normalize_wxyz(q)[[1, 2, 3, 0]])
    return rot.apply(v)


def _decompose_rotation_aa(rotation_aa: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    rot = R.from_rotvec(rotation_aa.reshape(-1, 3))
    q_xyzw = rot.as_quat()
    q = q_xyzw[:, [3, 0, 1, 2]]
    w = q[:, :1]
    v = q[:, 1:]
    v_twist = np.dot(v, axis)[:, None] * axis[None, :]
    q_twist = np.concatenate([w, v_twist], axis=1)
    norms = np.linalg.norm(q_twist, axis=1, keepdims=True)
    identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    q_twist = np.where(norms > 1e-12, q_twist / np.maximum(norms, 1e-12), identity)
    q_twist_inv = q_twist * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    q_swing = _quat_multiply_wxyz(q_twist_inv, q)
    return q_twist, q_swing


class HumanJointsFK:
    """Tiny NumPy FK for the SMPL joint inputs expected by the A3 SMPL encoder."""

    def __init__(self, human_joints_npz: str | Path = DEFAULT_HUMAN_JOINTS_NPZ):
        path = Path(human_joints_npz)
        if path.exists():
            data = np.load(path)
            self.joints_rest = np.asarray(data["J"], dtype=np.float64)
            self.parents = np.asarray(data["parents_list"], dtype=np.int64)
        else:
            self.joints_rest, self.parents = self._load_from_torch_pickle(path)
        if self.joints_rest.shape != (55, 3) or self.parents.shape != (55,):
            raise ValueError(
                f"invalid human joints FK data: J={self.joints_rest.shape}, "
                f"parents={self.parents.shape}"
            )

        self.rel_joints = self.joints_rest.copy()
        self.rel_joints[1:] -= self.joints_rest[self.parents[1:]]
        print(f"[minimal] Loaded SMPL FK data from {path}")

    @staticmethod
    def _load_from_torch_pickle(path: Path) -> tuple[np.ndarray, np.ndarray]:
        pkl_path = path.with_suffix(".pkl")
        try:
            import torch
        except ImportError as exc:
            raise FileNotFoundError(
                f"{path} is missing and torch is unavailable for fallback loading. "
                "Copy gear_sonic/data/human/human_joints_info.npz to the target."
            ) from exc
        if not pkl_path.exists():
            raise FileNotFoundError(f"missing SMPL FK data: {path} or {pkl_path}")
        data = torch.load(pkl_path, map_location="cpu")
        joints_rest = data["J"].detach().cpu().numpy().astype(np.float64)
        parents = np.asarray(data["parents_list"], dtype=np.int64)
        print(f"[minimal] Loaded SMPL FK data from torch pickle fallback: {pkl_path}")
        return joints_rest, parents

    def compute(self, body_pose_21x3: np.ndarray, global_orient_3: np.ndarray) -> np.ndarray:
        full_pose = np.zeros((55, 3), dtype=np.float64)
        full_pose[0] = np.asarray(global_orient_3, dtype=np.float64)
        full_pose[1:22] = np.asarray(body_pose_21x3, dtype=np.float64).reshape(21, 3)

        rot_mats = R.from_rotvec(full_pose.reshape(-1, 3)).as_matrix()
        transforms = np.zeros((55, 4, 4), dtype=np.float64)
        transforms[:, :3, :3] = rot_mats
        transforms[:, :3, 3] = self.rel_joints
        transforms[:, 3, 3] = 1.0

        chain = [None] * 55
        for idx, parent in enumerate(self.parents):
            if parent < 0:
                chain[idx] = transforms[idx]
            else:
                chain[idx] = chain[int(parent)] @ transforms[idx]

        joints = np.stack(chain, axis=0)[:, :3, 3]
        return joints[SMPL_OUTPUT_JOINT_INDEX].astype(np.float32)


def _compute_local_smpl_from_xrt(
    body_poses_np: np.ndarray,
    human_fk: HumanJointsFK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if body_poses_np.shape[0] < 24 or body_poses_np.shape[1] < 7:
        raise ValueError(f"expected at least [24,7] body poses, got {body_poses_np.shape}")

    body_poses_np = body_poses_np[:24].astype(np.float64, copy=False)
    positions = body_poses_np[:, :3]
    global_quats_wxyz = body_poses_np[:, [6, 3, 4, 5]]

    # Match the lightweight conversion used by the full sender before it calls
    # the heavier SMPL joint utility.
    global_rots = R.from_quat(global_quats_wxyz[:, [1, 2, 3, 0]])
    global_rots = global_rots * R.from_euler("y", 180.0, degrees=True)

    local_rots = []
    for idx, parent in enumerate(PARENT_INDICES):
        if parent < 0:
            local_rots.append(global_rots[idx])
        else:
            local_rots.append(global_rots[parent].inv() * global_rots[idx])
    pose_aa = np.asarray([rot.as_rotvec() for rot in local_rots], dtype=np.float32)

    smpl_pose = pose_aa[1:22].astype(np.float32)
    # Rebuild the root quaternion from the root rotvec just like
    # torch_transform.angle_axis_to_quaternion() in the full sender.  This also
    # gives the same quaternion sign convention, which keeps interpolation calm.
    root_quat_wxyz = R.from_rotvec(pose_aa[0]).as_quat()[[3, 0, 1, 2]]

    # Match the deploy runtime's canonical SMPL-frame conversion:
    # root Y-up -> Z-up, SMPL FK, remove the SMPL base rotation, then express
    # joints in the adjusted root-local frame.
    root_quat_zup = _quat_multiply_wxyz(
        Y_TO_Z_UP_QUAT_WXYZ.reshape(1, 4),
        root_quat_wxyz.reshape(1, 4),
    )[0]
    root_rotvec_zup = R.from_quat(root_quat_zup[[1, 2, 3, 0]]).as_rotvec()
    joints = human_fk.compute(smpl_pose, root_rotvec_zup)

    body_quat_w = _quat_multiply_wxyz(
        root_quat_zup.reshape(1, 4),
        SMPL_BASE_ROT_CONJ_WXYZ.reshape(1, 4),
    )[0]
    body_quat_w = _quat_normalize_wxyz(body_quat_w).astype(np.float32)
    smpl_joints_local = _quat_apply_wxyz(_quat_conjugate_wxyz(body_quat_w), joints).astype(
        np.float32
    )
    return smpl_pose, smpl_joints_local, body_quat_w


def _is_safe_pose_sample(
    smpl_pose: np.ndarray,
    smpl_joints: np.ndarray,
    body_quat_w: np.ndarray,
    joint_pos: np.ndarray,
    prev_smpl_joints: np.ndarray | None,
    max_abs_smpl_joint: float,
    max_frame_jump: float,
) -> tuple[bool, str]:
    arrays = {
        "smpl_pose": smpl_pose,
        "smpl_joints": smpl_joints,
        "body_quat_w": body_quat_w,
        "joint_pos": joint_pos,
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            return False, f"{name} contains non-finite values"

    quat_norm = float(np.linalg.norm(body_quat_w))
    if not 0.75 <= quat_norm <= 1.25:
        return False, f"root quaternion norm out of range: {quat_norm:.3f}"

    max_abs = float(np.max(np.abs(smpl_joints)))
    if max_abs > max_abs_smpl_joint:
        return False, f"smpl_joints max abs {max_abs:.3f} > {max_abs_smpl_joint:.3f}"

    if prev_smpl_joints is not None:
        max_jump = float(np.max(np.linalg.norm(smpl_joints - prev_smpl_joints, axis=1)))
        if max_jump > max_frame_jump:
            return False, f"smpl_joints frame jump {max_jump:.3f} > {max_frame_jump:.3f}"

    return True, ""


def _compute_wrist_joint_pos(smpl_pose: np.ndarray) -> np.ndarray:
    joint_pos = np.zeros(29, dtype=np.float32)
    body_pose = smpl_pose.reshape(1, 21, 3)

    smpl_l_elbow_idx = 17
    smpl_l_wrist_idx = 19
    smpl_r_elbow_idx = 18
    smpl_r_wrist_idx = 20
    g1_l_wrist_roll_idx = 23
    g1_l_wrist_pitch_idx = 25
    g1_l_wrist_yaw_idx = 27
    g1_r_wrist_roll_idx = 24
    g1_r_wrist_pitch_idx = 26
    g1_r_wrist_yaw_idx = 28

    smpl_l_elbow_aa = body_pose[:, smpl_l_elbow_idx]
    smpl_l_wrist_aa = body_pose[:, smpl_l_wrist_idx]
    smpl_r_elbow_aa = body_pose[:, smpl_r_elbow_idx]
    smpl_r_wrist_aa = body_pose[:, smpl_r_wrist_idx]

    _, g1_l_elbow_q_swing = _decompose_rotation_aa(smpl_l_elbow_aa, np.array([0.0, 1.0, 0.0]))
    _, g1_r_elbow_q_swing = _decompose_rotation_aa(smpl_r_elbow_aa, np.array([0.0, 1.0, 0.0]))

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]
    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    joint_pos[g1_l_wrist_roll_idx] = g1_l_wrist_roll[0]
    joint_pos[g1_l_wrist_pitch_idx] = -g1_l_wrist_pitch[0]
    joint_pos[g1_l_wrist_yaw_idx] = g1_l_wrist_yaw[0]
    joint_pos[g1_r_wrist_roll_idx] = g1_r_wrist_roll[0]
    joint_pos[g1_r_wrist_pitch_idx] = g1_r_wrist_pitch[0]
    joint_pos[g1_r_wrist_yaw_idx] = g1_r_wrist_yaw[0]
    return joint_pos


def _start_robotics_service(no_start_service: bool) -> None:
    if no_start_service:
        return
    service = "/opt/apps/roboticsservice/runService.sh"
    if os.path.exists(service):
        subprocess.Popen(["bash", service])
        time.sleep(0.5)
    else:
        print(f"[minimal] robotics service script not found: {service}")


def _get_a_button_pressed() -> bool:
    if xrt is None:
        return False
    try:
        return bool(xrt.get_A_button())
    except Exception:
        return False


def run_sender(
    port: int,
    num_frames_to_send: int,
    target_fps: int,
    no_start_service: bool,
    report_interval: float,
    human_joints_npz: str,
    max_abs_smpl_joint: float,
    max_frame_jump: float,
    start_paused: bool,
) -> None:
    if xrt is None:
        raise ImportError("xrobotoolkit_sdk is not installed or cannot be loaded")
    human_fk = HumanJointsFK(human_joints_npz)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[minimal] ZMQ pose socket bound to tcp://*:{port}")

    _start_robotics_service(no_start_service)
    xrt.init()
    print("[minimal] Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("[minimal] waiting for body data...")
        time.sleep(1.0)

    frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))
    last_stamp_ns = None
    last_report = time.time()
    sent = 0
    skipped = 0
    step = 0
    prev_smpl_joints = None
    last_skip_report = 0.0
    frame_time = 1.0 / max(1, target_fps)
    paused = bool(start_paused)
    prev_a_pressed = _get_a_button_pressed()
    print(f"[minimal] Stream state: {'PAUSED' if paused else 'RUNNING'} (press A to toggle)")

    try:
        while True:
            loop_start = time.time()
            if not xrt.is_body_data_available():
                time.sleep(0.005)
                continue

            stamp_ns = int(xrt.get_time_stamp_ns())
            if last_stamp_ns is not None and stamp_ns == last_stamp_ns:
                time.sleep(0.001)
                continue
            pico_dt = ((stamp_ns - last_stamp_ns) * 1e-9) if last_stamp_ns is not None else 0.0
            last_stamp_ns = stamp_ns

            a_pressed = _get_a_button_pressed()
            if a_pressed and not prev_a_pressed:
                paused = not paused
                frame_buffer.clear()
                prev_smpl_joints = None
                state = "PAUSED" if paused else "RUNNING"
                print(f"[minimal] A pressed: Stream state -> {state}")
            prev_a_pressed = a_pressed

            body_poses_np = np.asarray(xrt.get_body_joints_pose(), dtype=np.float32)
            smpl_pose, smpl_joints, body_quat_w = _compute_local_smpl_from_xrt(
                body_poses_np, human_fk
            )
            joint_pos = _compute_wrist_joint_pos(smpl_pose)
            ok, reason = _is_safe_pose_sample(
                smpl_pose,
                smpl_joints,
                body_quat_w,
                joint_pos,
                prev_smpl_joints,
                max_abs_smpl_joint,
                max_frame_jump,
            )
            if not ok:
                frame_buffer.clear()
                prev_smpl_joints = None
                skipped += 1
                now = time.time()
                if now - last_skip_report >= 1.0:
                    print(f"[minimal] skipping unsafe pose sample: {reason}")
                    last_skip_report = now
                continue
            prev_smpl_joints = smpl_joints.copy()

            frame_buffer["smpl_pose"].append(smpl_pose)
            frame_buffer["smpl_joints"].append(smpl_joints)
            frame_buffer["body_quat_w"].append(body_quat_w)
            frame_buffer["joint_pos"].append(joint_pos)
            frame_buffer["joint_vel"].append(np.zeros(29, dtype=np.float32))
            frame_buffer["frame_index"].append(np.int64(step))

            if (not paused) and len(frame_buffer["frame_index"]) >= num_frames_to_send:
                data = {
                    "smpl_pose": np.stack(frame_buffer["smpl_pose"], axis=0).astype(np.float32),
                    "smpl_joints": np.stack(frame_buffer["smpl_joints"], axis=0).astype(np.float32),
                    "body_quat_w": np.stack(frame_buffer["body_quat_w"], axis=0).astype(
                        np.float32
                    ),
                    "joint_pos": np.stack(frame_buffer["joint_pos"], axis=0).astype(np.float32),
                    "joint_vel": np.stack(frame_buffer["joint_vel"], axis=0).astype(np.float32),
                    "frame_index": np.asarray(frame_buffer["frame_index"], dtype=np.int64),
                    "pico_dt": np.asarray([pico_dt], dtype=np.float32),
                    "timestamp_realtime": np.asarray([time.time()], dtype=np.float64),
                    "timestamp_monotonic": np.asarray([time.monotonic()], dtype=np.float64),
                }
                socket.send(pack_pose_message(data, topic="pose"))
                sent += 1

            step += 1
            now = time.time()
            if now - last_report >= report_interval:
                fps = sent / max(now - last_report, 1e-6)
                print(
                    f"[minimal] sent_fps={fps:.2f}, step={step}, window={num_frames_to_send}, "
                    f"skipped={skipped}, state={'PAUSED' if paused else 'RUNNING'}"
                )
                sent = 0
                skipped = 0
                last_report = now

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    finally:
        socket.close()
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--num_frames_to_send", type=int, default=10)
    parser.add_argument("--target_fps", type=int, default=50)
    parser.add_argument("--no_start_service", action="store_true")
    parser.add_argument("--report_interval", type=float, default=5.0)
    parser.add_argument(
        "--human_joints_npz",
        type=str,
        default=str(DEFAULT_HUMAN_JOINTS_NPZ),
        help="NumPy SMPL FK data path (default: gear_sonic/data/human/human_joints_info.npz)",
    )
    parser.add_argument(
        "--max_abs_smpl_joint",
        type=float,
        default=5.0,
        help="Safety guard: skip samples whose local SMPL joint abs value exceeds this",
    )
    parser.add_argument(
        "--max_frame_jump",
        type=float,
        default=1.25,
        help="Safety guard: skip samples with too-large per-joint frame jump in meters",
    )
    parser.add_argument(
        "--start_unpaused",
        action="store_true",
        help="Start streaming immediately instead of the default paused state",
    )
    parser.add_argument("--manager", action="store_true", help="Accepted for command compatibility")
    parser.add_argument("--vis_vr3pt", action="store_true", help="Ignored by the minimal sender")
    parser.add_argument("--vis_smpl", action="store_true", help="Ignored by the minimal sender")
    parser.add_argument("--waist_tracking", action="store_true", help="Ignored by the minimal sender")
    args = parser.parse_args()

    if args.manager:
        print("[minimal] --manager accepted; this sender streams pose continuously.")
    ignored = []
    if args.vis_vr3pt:
        ignored.append("--vis_vr3pt")
    if args.vis_smpl:
        ignored.append("--vis_smpl")
    if args.waist_tracking:
        ignored.append("--waist_tracking")
    if ignored:
        print(f"[minimal] Ignoring visualization options: {' '.join(ignored)}")

    run_sender(
        port=args.port,
        num_frames_to_send=args.num_frames_to_send,
        target_fps=args.target_fps,
        no_start_service=args.no_start_service,
        report_interval=args.report_interval,
        human_joints_npz=args.human_joints_npz,
        max_abs_smpl_joint=args.max_abs_smpl_joint,
        max_frame_jump=args.max_frame_jump,
        start_paused=not args.start_unpaused,
    )


if __name__ == "__main__":
    main()
