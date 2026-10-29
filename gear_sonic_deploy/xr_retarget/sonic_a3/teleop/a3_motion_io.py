"""Flat A3 CSV motion I/O shared by the teleop tools.

CSV layout (same as ``a3_data/agibot_a3``): ``Frame``, root translation in cm, root
extrinsic-XYZ Euler angles in degrees, then joint angles in degrees. Files written
here are sampled at 120 Hz so the sim2sim / deploy defaults (stride 4, 30 fps) apply.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sonic_a3.contract.joint_order import IL_JOINT_NAMES as A3_IL_JOINT_NAMES

CSV_RAW_FPS = 120.0
CSV_SOURCE_FPS = 30.0
CSV_FRAME_STRIDE = 4
POLICY_FPS = 50.0
CSV_JOINT_COLUMNS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
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
)


@dataclass
class A3Motion:
    fps: float
    root_pos: np.ndarray
    """(T, 3) metres."""
    root_quat_wxyz: np.ndarray
    """(T, 4)."""
    dof_il: np.ndarray
    """(T, 29) radians, policy (IsaacLab) joint order."""

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    def resample(self, fps: float) -> A3Motion:
        if self.num_frames < 2 or np.isclose(fps, self.fps):
            return A3Motion(fps, self.root_pos.copy(), self.root_quat_wxyz.copy(), self.dof_il.copy())
        src_t = np.arange(self.num_frames) / self.fps
        dst_t = np.arange(0.0, src_t[-1] + 1e-9, 1.0 / fps)
        root_pos = np.stack([np.interp(dst_t, src_t, self.root_pos[:, i]) for i in range(3)], axis=1)
        dof = np.stack([np.interp(dst_t, src_t, self.dof_il[:, i]) for i in range(self.dof_il.shape[1])], axis=1)
        rots = Rotation.from_quat(self.root_quat_wxyz[:, [1, 2, 3, 0]])
        quat = Slerp(src_t, rots)(dst_t).as_quat()[:, [3, 0, 1, 2]]
        return A3Motion(fps, root_pos, quat, dof)


def load_a3_csv(
    path: Path,
    *,
    source_fps: float = CSV_SOURCE_FPS,
    frame_stride: int = CSV_FRAME_STRIDE,
    target_fps: float = POLICY_FPS,
) -> A3Motion:
    """Load a flat A3 CSV with the same row selection as sim2sim, resampled to ``target_fps``."""
    with Path(path).open(newline="") as f:
        rows = [row for idx, row in enumerate(csv.DictReader(f)) if idx % frame_stride == 0]
    if not rows:
        raise ValueError(f"no rows in {path}")

    def column(name: str) -> np.ndarray:
        key = name if name in rows[0] else f"{name}_dof"
        return np.array([float(row[key]) for row in rows])

    root_pos = np.stack([column(f"root_translate{a}") for a in "XYZ"], axis=1) * 0.01
    euler = np.stack([column(f"root_rotate{a}") for a in "XYZ"], axis=1)
    root_quat = Rotation.from_euler("xyz", euler, degrees=True).as_quat()[:, [3, 0, 1, 2]]
    dof_il = np.deg2rad(np.stack([column(name) for name in A3_IL_JOINT_NAMES], axis=1))
    return A3Motion(source_fps, root_pos, root_quat, dof_il).resample(target_fps)
