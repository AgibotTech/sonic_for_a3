#!/usr/bin/env python3
# ruff: noqa: T201
"""Quality-check A3 flat-ground training PKLs and copy suspect clips for review."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import joblib
import mujoco
import numpy as np


DEFAULT_INPUT = Path("a3_data/base_model_data_flat_ground_training_pkl/base_model_data_flat_ground_csv")
DEFAULT_OUTPUT = Path("a3_data/base_model_data_flat_ground_quality_review")
DEFAULT_MJCF = Path("gear_sonic/data/assets/robot_description/mjcf/a3_t2d5.xml")
FATAL_RULES = (
    "very_high_root",
    "wrist_flicker",
    "sitting_posture_keyword",
    "hard_motion_keyword",
    "command_velocity_pattern",
    "non_ankle_body_below_ground",
    "non_ankle_body_touch_ground",
    "mjcf_joint_limit_violation",
    "terrain_or_forbidden_keyword",
    "contact_ground_height_drift",
    "upper_body_late_jitter",
)

SITTING_POSTURE_KEYWORDS = (
    "zuozi",
    "zuo_zi",
    "zuo-zi",
    "zuodaozhan",
    "zuo_dao_zhan",
    "zuo-dao-zhan",
    "zuoxia",
    "zuo_xia",
    "zuo-xia",
    "sit_down",
    "sit_to_stand",
    "sitdown",
    "sitting",
    "sit_high",
    "8ziyou-man_long",
    "youcezhuan-kuai_long",
    "youcezhuan_kuai_long",
    "chair",
    "坐姿",
    "坐下",
    "坐到站",
    "座椅",
    "坐椅",
)

HARD_MOTION_KEYWORDS = (
    "bmd_0120_a3_filtered_20260120_193312__2_ceti_skeleton0",
    "bmd_0124",
    "bmd_0122_a3_filtered_20260122_133745__poes5tongzibaifo_wushu",
    "bmd_0122_a3_filtered_20260122_133745_poes5tongzibaifo_wushu",
    "bmd_0122_a3_mirror_filtered_20260226_143727__poes5tongzibaifo_wushu",
    "bmd_0122_a3_mirror_filtered_20260226_143727_poes5tongzibaifo_wushu",
    "bmd_0122_a3_filtered_20260226_152730__poes5tongzibaifo_wushu",
    "bmd_0122_a3_filtered_20260226_152730_poes5tongzibaifo_wushu",
    "bmd_0311_a3_filtered_20260312_102626__jump_004_skeleton0",
    "bmd_0311_a3_mirror_filtered_20260312_113047__jump_004_skeleton0",
    "bmd_1213_a3_filtered_20260128_162239__quanji",
    "bmd_1213_a3_mirror_filtered_20260128_191629__quanji",
    "feiti",
    "gongfu",
    "gongfutaolu",
    "j3xiao_xiao",
    "kongfan",
    "shibatongren",
    "wushudongzuo",
    "xuanzhaunfeiti",
    "xuanzhuanfeiti",
)
COMMAND_VELOCITY_NAME_RE = re.compile(
    r"^(walk|run)_fwd[+-]\d+(?:\.\d+)?_yaw[+-]\d+(?:\.\d+)?_strafe[+-]\d+(?:\.\d+)?$"
)

TERRAIN_KEYWORDS = (
    "shangpo",
    "shang_po",
    "shang-po",
    "shanpo",
    "shan_po",
    "shan-po",
    "xiapo",
    "xiaopo",
    "xia_po",
    "xiao_po",
    "xia-po",
    "slope",
    "uphill",
    "downhill",
    "ramp",
    "shanglouti",
    "xialouti",
    "xialou",
    "shanglou",
    "louti",
    "lou_ti",
    "lou-ti",
    "stair",
    "stairs",
    "upstairs",
    "downstairs",
    "ruxiang",
    "chuxiang",
    "paxiangzi",
    "tiaoxiangzi",
    "fanxiangzi",
    "xiangzi",
    "paoku",
    "parkour",
    "zuoxia",
    "sit_down",
    "sit_to_stand",
    "sitdown",
    "sitting",
    "sit_high",
    "上坡",
    "下坡",
    "楼梯",
    "入箱",
    "出箱",
    "箱子",
    "跑酷",
    "坐下",
)

FOOT_BODY_NAMES = ("left_ankle_roll_Link", "right_ankle_roll_Link")
ALLOWED_GROUND_BODY_NAMES = FOOT_BODY_NAMES
NON_ANKLE_TOUCH_Z = 0.005
NON_ANKLE_BELOW_Z = -0.02
INITIAL_FOOT_FLOAT_Z = 0.12
INITIAL_FOOT_LOW_Z = 0.015
INITIAL_FOOT_HEIGHT_DIFF_Z = 0.05
FOOT_GROUND_CONTACT_Z = 0.12
VERY_HIGH_ROOT_Z = 1.60
UPPER_BODY_LATE_WINDOW_SEC = 3.0
UPPER_BODY_LATE_WRIST_ACC = 2000.0
UPPER_BODY_LATE_ARM_ACC = 2400.0
UPPER_BODY_LATE_VEL = 8.0
UPPER_BODY_REVERSAL_VEL = 6.0
UPPER_BODY_TAIL_WINDOW_SEC = 5.0
UPPER_BODY_TAIL_WRIST_VEL = 9.0
UPPER_BODY_TAIL_WRIST_ACC = 1200.0
UPPER_BODY_TAIL_WRIST_RANGE = 1.4
WRIST_FLICKER_VEL = 14.0
WRIST_FLICKER_ACC = 3000.0
WRIST_FLICKER_MIN_FRAMES = 30
KEY_BODY_NAMES = (
    "pelvis_link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
)

METRIC_FIELDS = (
    "name",
    "source_pkl",
    "frames",
    "fps",
    "status",
    "severity",
    "rules",
    "effective_rules",
    "error",
    "root_z_min",
    "root_z_max",
    "root_z_range",
    "root_speed_max",
    "root_vz_max",
    "root_ang_vel_max",
    "dof_abs_max",
    "dof_vel_fd_max",
    "joint_limit_violation_max",
    "root_rot_norm_min",
    "root_rot_norm_max",
    "upper_body_vel_max",
    "wrist_vel_max",
    "upper_body_acc_max",
    "wrist_acc_max",
    "late_upper_body_vel_max",
    "late_wrist_vel_max",
    "late_upper_body_acc_max",
    "late_wrist_acc_max",
    "late_upper_body_reversal_frames",
    "late_wrist_reversal_frames",
    "late_upper_body_spike_frame",
    "late_upper_body_spike_joint",
    "tail_upper_body_vel_max",
    "tail_wrist_vel_max",
    "tail_upper_body_acc_max",
    "tail_wrist_acc_max",
    "tail_upper_body_range_max",
    "tail_wrist_range_max",
    "wrist_flicker_frame_count",
    "wrist_flicker_event_count",
    "wrist_flicker_joint_count",
    "wrist_flicker_spike_frame",
    "wrist_flicker_spike_joint",
    "foot_z_min",
    "foot_z_max",
    "foot_z_range",
    "foot_contact_height_range",
    "key_body_z_min",
    "first_root_z",
    "first_left_foot_z",
    "first_right_foot_z",
    "first_foot_z_min",
    "first_foot_z_max",
    "first_foot_height_diff",
    "first_torso_z",
    "first_knee_z_min",
    "first_non_ankle_geom_z_min",
    "first_non_ankle_geom_touch_bodies",
    "non_ankle_geom_z_min",
    "non_ankle_geom_touch_frames",
    "non_ankle_geom_below_frames",
    "non_ankle_geom_touch_ratio",
    "non_ankle_geom_touch_bodies",
    "foot_ground_contact_frames",
    "foot_ground_contact_ratio",
    "longest_air_duration",
    "sampled_frames",
)

SUMMARY_FIELDS = ("rule", "count")

A3_29_DOF_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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
A3_31_DOF_NAMES = (
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

MODEL = None
DATA = None
BODY_AXES = None
DOF_BODY_INDICES = None
DOF_AXES = None
HINGE_RANGES = None
HINGE_LIMITED = None
BODY_IDS = None
NON_ANKLE_COLLISION_GEOM_IDS = None
GEOM_BODY_NAMES = None
MESH_VERTICES_BY_GEOM = None


def parse_mjcf_dof_axes(mjcf_path: Path) -> list[np.ndarray | None]:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    root_body = root.find("worldbody/body")
    if root_body is None:
        raise ValueError(f"Missing root body in {mjcf_path}")

    body_axes: list[np.ndarray | None] = []

    def walk(body: ET.Element) -> None:
        hinge_joints = [
            joint for joint in body.findall("joint") if joint.attrib.get("type", "hinge") != "free"
        ]
        if hinge_joints:
            axis = np.fromstring(hinge_joints[0].attrib.get("axis", "0 0 1"), sep=" ")
        else:
            axis = None
        body_axes.append(axis)
        for child in body.findall("body"):
            walk(child)

    walk(root_body)
    return body_axes


def init_worker(mjcf_path: str) -> None:
    global MODEL, DATA, BODY_AXES, DOF_BODY_INDICES, DOF_AXES
    global HINGE_RANGES, HINGE_LIMITED, BODY_IDS
    global NON_ANKLE_COLLISION_GEOM_IDS, GEOM_BODY_NAMES, MESH_VERTICES_BY_GEOM

    MODEL = mujoco.MjModel.from_xml_path(mjcf_path)
    DATA = mujoco.MjData(MODEL)
    BODY_AXES = parse_mjcf_dof_axes(Path(mjcf_path))
    DOF_BODY_INDICES = [idx for idx in range(1, len(BODY_AXES)) if BODY_AXES[idx] is not None]
    DOF_AXES = np.asarray([BODY_AXES[idx] for idx in DOF_BODY_INDICES], dtype=np.float64)

    hinge_joint_ids = [
        j
        for j in range(MODEL.njnt)
        if MODEL.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    hinge_joint_ids.sort(key=lambda j: int(MODEL.jnt_qposadr[j]))
    HINGE_RANGES = np.asarray([MODEL.jnt_range[j] for j in hinge_joint_ids], dtype=np.float64)
    HINGE_LIMITED = np.asarray([bool(MODEL.jnt_limited[j]) for j in hinge_joint_ids], dtype=bool)
    BODY_IDS = {
        name: mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in set(FOOT_BODY_NAMES + KEY_BODY_NAMES)
    }
    GEOM_BODY_NAMES = {}
    MESH_VERTICES_BY_GEOM = {}
    NON_ANKLE_COLLISION_GEOM_IDS = []
    for geom_id in range(MODEL.ngeom):
        body_id = int(MODEL.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        body_name = mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        GEOM_BODY_NAMES[geom_id] = body_name
        if body_name in ALLOWED_GROUND_BODY_NAMES:
            continue
        if int(MODEL.geom_contype[geom_id]) == 0 and int(MODEL.geom_conaffinity[geom_id]) == 0:
            continue
        NON_ANKLE_COLLISION_GEOM_IDS.append(geom_id)
        if MODEL.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(MODEL.geom_dataid[geom_id])
            if mesh_id >= 0:
                start = int(MODEL.mesh_vertadr[mesh_id])
                end = start + int(MODEL.mesh_vertnum[mesh_id])
                MESH_VERTICES_BY_GEOM[geom_id] = MODEL.mesh_vert[start:end].copy()


def list_pkl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.pkl") if path.name != "metadata.pkl")
    raise FileNotFoundError(input_path)


def load_motion_entry(path: Path) -> tuple[str, dict]:
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError("PKL is not a non-empty dict")
    if "pose_aa" in data:
        return path.stem, data
    first_key = next(iter(data))
    first_value = data[first_key]
    if isinstance(first_value, dict) and "pose_aa" in first_value:
        return str(first_key), first_value
    raise ValueError("No motion entry with pose_aa found")


def finite_or_raise(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def quat_norm_stats(root_rot: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(root_rot, axis=1)
    return float(norms.min()), float(norms.max())


def quat_ang_vel_max(root_rot: np.ndarray, fps: float) -> float:
    if len(root_rot) < 2:
        return 0.0
    q = root_rot.astype(np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    q = q / norms
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return float(np.max(angles * fps))


def dof_joint_names(dof_dim: int) -> tuple[str, ...]:
    if dof_dim == len(A3_29_DOF_NAMES):
        return A3_29_DOF_NAMES
    if dof_dim == len(A3_31_DOF_NAMES):
        return A3_31_DOF_NAMES
    return tuple(f"dof_{idx}" for idx in range(dof_dim))


def named_dof_indices(dof_dim: int, patterns: tuple[str, ...]) -> np.ndarray:
    names = dof_joint_names(dof_dim)
    indices = [
        idx
        for idx, name in enumerate(names)
        if idx < dof_dim and any(pattern in name for pattern in patterns)
    ]
    return np.asarray(indices, dtype=np.int64)


def reversal_frame_count(vel: np.ndarray, threshold: float) -> int:
    if vel.shape[0] < 2 or vel.shape[1] == 0:
        return 0
    reversals = (
        (vel[:-1] * vel[1:] < 0.0)
        & (np.abs(vel[:-1]) > threshold)
        & (np.abs(vel[1:]) > threshold)
    )
    return int(np.count_nonzero(np.any(reversals, axis=1)))


def upper_body_motion_metrics(dof: np.ndarray, fps: float) -> dict[str, object]:
    metrics: dict[str, object] = {
        "upper_body_vel_max": 0.0,
        "wrist_vel_max": 0.0,
        "upper_body_acc_max": 0.0,
        "wrist_acc_max": 0.0,
        "late_upper_body_vel_max": 0.0,
        "late_wrist_vel_max": 0.0,
        "late_upper_body_acc_max": 0.0,
        "late_wrist_acc_max": 0.0,
        "late_upper_body_reversal_frames": 0,
        "late_wrist_reversal_frames": 0,
        "late_upper_body_spike_frame": "",
        "late_upper_body_spike_joint": "",
        "tail_upper_body_vel_max": 0.0,
        "tail_wrist_vel_max": 0.0,
        "tail_upper_body_acc_max": 0.0,
        "tail_wrist_acc_max": 0.0,
        "tail_upper_body_range_max": 0.0,
        "tail_wrist_range_max": 0.0,
        "wrist_flicker_frame_count": 0,
        "wrist_flicker_event_count": 0,
        "wrist_flicker_joint_count": 0,
        "wrist_flicker_spike_frame": "",
        "wrist_flicker_spike_joint": "",
    }
    if dof.ndim != 2 or dof.shape[0] < 2:
        return metrics

    arm_idx = named_dof_indices(dof.shape[1], ("shoulder", "elbow", "wrist"))
    wrist_idx = named_dof_indices(dof.shape[1], ("wrist",))
    if arm_idx.size == 0:
        return metrics

    vel = np.diff(dof, axis=0) * fps
    acc = np.diff(vel, axis=0) * fps if vel.shape[0] > 1 else np.zeros((0, dof.shape[1]))

    def max_abs(array: np.ndarray, cols: np.ndarray) -> float:
        if array.size == 0 or cols.size == 0:
            return 0.0
        return float(np.max(np.abs(array[:, cols])))

    late_start = max(0, dof.shape[0] - int(round(UPPER_BODY_LATE_WINDOW_SEC * fps)))
    late_vel_start = max(0, late_start - 1)
    late_acc_start = max(0, late_start - 2)
    late_vel = vel[late_vel_start:]
    late_acc = acc[late_acc_start:]
    tail_start = max(0, dof.shape[0] - int(round(UPPER_BODY_TAIL_WINDOW_SEC * fps)))
    tail_vel_start = max(0, tail_start - 1)
    tail_acc_start = max(0, tail_start - 2)
    tail_dof = dof[tail_start:]
    tail_vel = vel[tail_vel_start:]
    tail_acc = acc[tail_acc_start:]

    metrics["upper_body_vel_max"] = max_abs(vel, arm_idx)
    metrics["wrist_vel_max"] = max_abs(vel, wrist_idx)
    metrics["upper_body_acc_max"] = max_abs(acc, arm_idx)
    metrics["wrist_acc_max"] = max_abs(acc, wrist_idx)
    metrics["late_upper_body_vel_max"] = max_abs(late_vel, arm_idx)
    metrics["late_wrist_vel_max"] = max_abs(late_vel, wrist_idx)
    metrics["late_upper_body_acc_max"] = max_abs(late_acc, arm_idx)
    metrics["late_wrist_acc_max"] = max_abs(late_acc, wrist_idx)
    metrics["tail_upper_body_vel_max"] = max_abs(tail_vel, arm_idx)
    metrics["tail_wrist_vel_max"] = max_abs(tail_vel, wrist_idx)
    metrics["tail_upper_body_acc_max"] = max_abs(tail_acc, arm_idx)
    metrics["tail_wrist_acc_max"] = max_abs(tail_acc, wrist_idx)
    if tail_dof.size:
        metrics["tail_upper_body_range_max"] = max_abs(np.ptp(tail_dof, axis=0)[None, :], arm_idx)
        metrics["tail_wrist_range_max"] = max_abs(np.ptp(tail_dof, axis=0)[None, :], wrist_idx)
    if vel.shape[0] > 1 and acc.size and wrist_idx.size:
        wrist_vel_pair_max = np.maximum(
            np.abs(vel[:-1, wrist_idx]),
            np.abs(vel[1:, wrist_idx]),
        )
        wrist_acc_abs = np.abs(acc[:, wrist_idx])
        flicker_mask = (
            (wrist_vel_pair_max >= WRIST_FLICKER_VEL)
            & (wrist_acc_abs >= WRIST_FLICKER_ACC)
        )
        if np.any(flicker_mask):
            frame_hits = np.any(flicker_mask, axis=1)
            joint_hits = np.any(flicker_mask, axis=0)
            metrics["wrist_flicker_frame_count"] = int(np.count_nonzero(frame_hits))
            metrics["wrist_flicker_event_count"] = int(np.count_nonzero(flicker_mask))
            metrics["wrist_flicker_joint_count"] = int(np.count_nonzero(joint_hits))
            masked_acc = np.where(flicker_mask, wrist_acc_abs, -1.0)
            local_t, local_col = np.unravel_index(int(np.argmax(masked_acc)), masked_acc.shape)
            joint_idx = int(wrist_idx[local_col])
            metrics["wrist_flicker_spike_frame"] = int(local_t + 1)
            metrics["wrist_flicker_spike_joint"] = dof_joint_names(dof.shape[1])[joint_idx]
    metrics["late_upper_body_reversal_frames"] = reversal_frame_count(
        late_vel[:, arm_idx] if late_vel.size else np.zeros((0, 0)),
        UPPER_BODY_REVERSAL_VEL,
    )
    metrics["late_wrist_reversal_frames"] = reversal_frame_count(
        late_vel[:, wrist_idx] if late_vel.size and wrist_idx.size else np.zeros((0, 0)),
        UPPER_BODY_REVERSAL_VEL,
    )

    if late_acc.size and arm_idx.size:
        arm_acc = np.abs(late_acc[:, arm_idx])
        local_flat_idx = int(np.argmax(arm_acc))
        local_t, local_col = np.unravel_index(local_flat_idx, arm_acc.shape)
        global_frame = late_acc_start + local_t
        joint_idx = int(arm_idx[local_col])
        metrics["late_upper_body_spike_frame"] = global_frame
        metrics["late_upper_body_spike_joint"] = dof_joint_names(dof.shape[1])[joint_idx]

    return metrics


def sample_indices(frame_count: int, max_samples: int, important: list[int]) -> np.ndarray:
    if frame_count <= max_samples:
        base = np.arange(frame_count, dtype=np.int64)
    else:
        base = np.linspace(0, frame_count - 1, max_samples, dtype=np.int64)
    extra = np.asarray([idx for idx in important if 0 <= idx < frame_count], dtype=np.int64)
    return np.unique(np.concatenate([base, extra]))


def qpos_from_pose(root_trans: np.ndarray, pose_aa: np.ndarray) -> np.ndarray:
    root_quat_xyzw = axis_angle_to_quat_xyzw(pose_aa[:, 0, :].astype(np.float64))
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_pose = pose_aa[:, DOF_BODY_INDICES, :].astype(np.float64)
    dof_angles = np.einsum("tdi,di->td", joint_pose, DOF_AXES)

    qpos = np.zeros((pose_aa.shape[0], 7 + dof_angles.shape[1]), dtype=np.float64)
    qpos[:, :3] = root_trans.astype(np.float64)
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = dof_angles
    return qpos


def axis_angle_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=1)
    quat = np.zeros((rotvec.shape[0], 4), dtype=np.float64)
    small = angle < 1e-12
    if np.any(~small):
        half = angle[~small] * 0.5
        axis = rotvec[~small] / angle[~small, None]
        quat[~small, :3] = axis * np.sin(half)[:, None]
        quat[~small, 3] = np.cos(half)
    if np.any(small):
        quat[small, :3] = 0.5 * rotvec[small]
        quat[small, 3] = 1.0
    return quat


def longest_sampled_duration(mask: np.ndarray, sample_idx: np.ndarray, fps: float) -> float:
    best = 0.0
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        if (not active or i == len(mask) - 1) and start is not None:
            end = i if active and i == len(mask) - 1 else i - 1
            duration = float(sample_idx[end] - sample_idx[start] + 1) / fps
            best = max(best, duration)
            start = None
    return best


def geom_bottom_z(geom_id: int) -> float:
    vertices = MESH_VERTICES_BY_GEOM.get(geom_id) if MESH_VERTICES_BY_GEOM is not None else None
    if vertices is not None:
        xmat = DATA.geom_xmat[geom_id].reshape(3, 3)
        return float(DATA.geom_xpos[geom_id, 2] + np.min(vertices @ xmat[2, :]))
    return float(DATA.geom_xpos[geom_id, 2] - MODEL.geom_rbound[geom_id])


def fk_metrics(
    root_trans: np.ndarray,
    pose_aa: np.ndarray,
    fps: float,
    sample_idx: np.ndarray,
) -> dict[str, float]:
    foot_z = []
    key_z = []
    first_metrics: dict[str, float | str] = {}
    non_ankle_min_z = math.inf
    non_ankle_touch_frames = 0
    non_ankle_below_frames = 0
    non_ankle_touch_bodies: dict[str, float] = {}
    qpos = qpos_from_pose(root_trans[sample_idx], pose_aa[sample_idx])

    foot_ids = [BODY_IDS[name] for name in FOOT_BODY_NAMES if BODY_IDS[name] >= 0]
    key_ids = [BODY_IDS[name] for name in KEY_BODY_NAMES if BODY_IDS[name] >= 0]

    for frame_idx, q in zip(sample_idx, qpos):
        DATA.qpos[: len(q)] = q
        mujoco.mj_forward(MODEL, DATA)
        if foot_ids:
            foot_z.append([float(DATA.xpos[body_id, 2]) for body_id in foot_ids])
        if key_ids:
            key_z.append([float(DATA.xpos[body_id, 2]) for body_id in key_ids])
        frame_touch = False
        frame_below = False
        frame_min_non_ankle_z = math.inf
        frame_touch_bodies: dict[str, float] = {}
        for geom_id in NON_ANKLE_COLLISION_GEOM_IDS:
            rough_bottom = float(DATA.geom_xpos[geom_id, 2] - MODEL.geom_rbound[geom_id])
            if rough_bottom > NON_ANKLE_TOUCH_Z + 0.02:
                continue
            bottom_z = geom_bottom_z(geom_id)
            non_ankle_min_z = min(non_ankle_min_z, bottom_z)
            frame_min_non_ankle_z = min(frame_min_non_ankle_z, bottom_z)
            if bottom_z <= NON_ANKLE_TOUCH_Z:
                frame_touch = True
                body_name = GEOM_BODY_NAMES.get(geom_id, f"geom_{geom_id}")
                previous = non_ankle_touch_bodies.get(body_name, math.inf)
                non_ankle_touch_bodies[body_name] = min(previous, bottom_z)
                frame_previous = frame_touch_bodies.get(body_name, math.inf)
                frame_touch_bodies[body_name] = min(frame_previous, bottom_z)
            if bottom_z < NON_ANKLE_BELOW_Z:
                frame_below = True
        if frame_touch:
            non_ankle_touch_frames += 1
        if frame_below:
            non_ankle_below_frames += 1
        if frame_idx == 0:
            first_foot = {
                name: float(DATA.xpos[body_id, 2])
                for name, body_id in BODY_IDS.items()
                if name in FOOT_BODY_NAMES and body_id >= 0
            }
            first_knees = [
                float(DATA.xpos[BODY_IDS[name], 2])
                for name in ("left_knee_Link", "right_knee_Link")
                if BODY_IDS.get(name, -1) >= 0
            ]
            first_touch_body_text = ";".join(
                f"{name}:{value:.3f}"
                for name, value in sorted(frame_touch_bodies.items(), key=lambda item: item[1])[:12]
            )
            first_values = list(first_foot.values())
            first_metrics = {
                "first_root_z": float(root_trans[0, 2]),
                "first_left_foot_z": first_foot.get("left_ankle_roll_Link", math.nan),
                "first_right_foot_z": first_foot.get("right_ankle_roll_Link", math.nan),
                "first_foot_z_min": float(min(first_values)) if first_values else math.nan,
                "first_foot_z_max": float(max(first_values)) if first_values else math.nan,
                "first_foot_height_diff": float(max(first_values) - min(first_values))
                if first_values
                else math.nan,
                "first_torso_z": float(DATA.xpos[BODY_IDS["torso_Link"], 2])
                if BODY_IDS.get("torso_Link", -1) >= 0
                else math.nan,
                "first_knee_z_min": float(min(first_knees)) if first_knees else math.nan,
                "first_non_ankle_geom_z_min": frame_min_non_ankle_z
                if not math.isinf(frame_min_non_ankle_z)
                else math.nan,
                "first_non_ankle_geom_touch_bodies": first_touch_body_text,
            }

    if foot_z:
        foot_z_arr = np.asarray(foot_z, dtype=np.float64)
        min_foot_each_frame = foot_z_arr.min(axis=1)
        ground_est = float(np.quantile(min_foot_each_frame, 0.05))
        low_mask = min_foot_each_frame <= ground_est + 0.04
        if np.any(low_mask):
            contact_height_range = float(np.ptp(min_foot_each_frame[low_mask]))
        else:
            contact_height_range = 0.0
        air_mask = min_foot_each_frame > ground_est + 0.12
        longest_air = longest_sampled_duration(air_mask, sample_idx, fps)
        world_contact_mask = min_foot_each_frame <= FOOT_GROUND_CONTACT_Z
        foot_ground_contact_frames = int(np.count_nonzero(world_contact_mask))
        foot_ground_contact_ratio = float(foot_ground_contact_frames) / float(len(sample_idx))
        foot_z_min = float(foot_z_arr.min())
        foot_z_max = float(foot_z_arr.max())
        foot_z_range = foot_z_max - foot_z_min
    else:
        foot_z_min = math.nan
        foot_z_max = math.nan
        foot_z_range = math.nan
        contact_height_range = math.nan
        longest_air = math.nan
        foot_ground_contact_frames = 0
        foot_ground_contact_ratio = math.nan

    if key_z:
        key_body_z_min = float(np.asarray(key_z, dtype=np.float64).min())
    else:
        key_body_z_min = math.nan
    if math.isinf(non_ankle_min_z):
        non_ankle_min_z = math.nan
    touch_body_text = ";".join(
        f"{name}:{value:.3f}"
        for name, value in sorted(non_ankle_touch_bodies.items(), key=lambda item: item[1])[:12]
    )

    return {
        "foot_z_min": foot_z_min,
        "foot_z_max": foot_z_max,
        "foot_z_range": foot_z_range,
        "foot_contact_height_range": contact_height_range,
        "key_body_z_min": key_body_z_min,
        **first_metrics,
        "non_ankle_geom_z_min": non_ankle_min_z,
        "non_ankle_geom_touch_frames": non_ankle_touch_frames,
        "non_ankle_geom_below_frames": non_ankle_below_frames,
        "non_ankle_geom_touch_ratio": float(non_ankle_touch_frames) / float(len(sample_idx)),
        "non_ankle_geom_touch_bodies": touch_body_text,
        "foot_ground_contact_frames": foot_ground_contact_frames,
        "foot_ground_contact_ratio": foot_ground_contact_ratio,
        "longest_air_duration": longest_air,
        "sampled_frames": int(len(sample_idx)),
    }


def joint_limit_violation_max(qpos_sample: np.ndarray) -> float:
    if qpos_sample.shape[1] < 8 or HINGE_RANGES is None:
        return 0.0
    dof = qpos_sample[:, 7 : 7 + len(HINGE_RANGES)]
    if dof.shape[1] != len(HINGE_RANGES):
        return math.nan
    limited = HINGE_LIMITED
    if not np.any(limited):
        return 0.0
    low = HINGE_RANGES[:, 0]
    high = HINGE_RANGES[:, 1]
    below = np.maximum(low[None, :] - dof, 0.0)
    above = np.maximum(dof - high[None, :], 0.0)
    violation = np.maximum(below, above)[:, limited]
    return float(violation.max()) if violation.size else 0.0


def process_file(task: tuple[str, int]) -> dict[str, object]:
    path_text, max_fk_samples = task
    path = Path(path_text)
    row: dict[str, object] = {
        "name": path.stem,
        "source_pkl": path.as_posix(),
        "status": "ok",
        "severity": "",
        "rules": "",
        "error": "",
    }

    try:
        name, entry = load_motion_entry(path)
        row["name"] = name

        required = ("root_trans_offset", "pose_aa", "dof", "root_rot", "fps")
        missing = [key for key in required if key not in entry]
        if missing:
            raise ValueError(f"missing required keys: {missing}")

        root_trans = np.asarray(entry["root_trans_offset"], dtype=np.float64)
        pose_aa = np.asarray(entry["pose_aa"], dtype=np.float64)
        dof = np.asarray(entry["dof"], dtype=np.float64)
        root_rot = np.asarray(entry["root_rot"], dtype=np.float64)
        fps = float(entry["fps"])

        finite_or_raise("root_trans_offset", root_trans)
        finite_or_raise("pose_aa", pose_aa)
        finite_or_raise("dof", dof)
        finite_or_raise("root_rot", root_rot)

        if root_trans.ndim != 2 or root_trans.shape[1] != 3:
            raise ValueError(f"root_trans_offset shape must be (T,3), got {root_trans.shape}")
        if pose_aa.ndim != 3 or pose_aa.shape[0] != root_trans.shape[0]:
            raise ValueError(f"pose_aa shape/frame mismatch: {pose_aa.shape}")
        if dof.ndim != 2 or dof.shape[0] != root_trans.shape[0]:
            raise ValueError(f"dof shape/frame mismatch: {dof.shape}")
        if root_rot.ndim != 2 or root_rot.shape != (root_trans.shape[0], 4):
            raise ValueError(f"root_rot shape/frame mismatch: {root_rot.shape}")
        if fps <= 0:
            raise ValueError(f"invalid fps: {fps}")

        frames = int(root_trans.shape[0])
        row["frames"] = frames
        row["fps"] = fps
        row["root_z_min"] = float(root_trans[:, 2].min())
        row["root_z_max"] = float(root_trans[:, 2].max())
        row["root_z_range"] = float(root_trans[:, 2].max() - root_trans[:, 2].min())

        if frames > 1:
            root_diff = np.diff(root_trans, axis=0)
            root_speed = np.linalg.norm(root_diff, axis=1) * fps
            root_vz = np.abs(root_diff[:, 2]) * fps
            dof_vel = np.abs(np.diff(dof, axis=0)) * fps
            row["root_speed_max"] = float(root_speed.max())
            row["root_vz_max"] = float(root_vz.max())
            row["dof_vel_fd_max"] = float(dof_vel.max())
            important = [
                int(np.argmax(root_speed)),
                int(np.argmax(root_speed)) + 1,
                int(np.argmax(root_vz)),
                int(np.argmax(root_vz)) + 1,
                int(np.argmin(root_trans[:, 2])),
                int(np.argmax(root_trans[:, 2])),
            ]
        else:
            row["root_speed_max"] = 0.0
            row["root_vz_max"] = 0.0
            row["dof_vel_fd_max"] = 0.0
            important = [0]

        row["dof_abs_max"] = float(np.abs(dof).max()) if dof.size else 0.0
        row["root_ang_vel_max"] = quat_ang_vel_max(root_rot, fps)
        qn_min, qn_max = quat_norm_stats(root_rot)
        row["root_rot_norm_min"] = qn_min
        row["root_rot_norm_max"] = qn_max
        row.update(upper_body_motion_metrics(dof, fps))

        sample_idx = sample_indices(frames, max_fk_samples, important)
        fk = fk_metrics(root_trans, pose_aa, fps, sample_idx)
        row.update(fk)
        qpos_sample = qpos_from_pose(root_trans[sample_idx], pose_aa[sample_idx])
        row["joint_limit_violation_max"] = joint_limit_violation_max(qpos_sample)

    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["severity"] = "critical"
        row["rules"] = "load_or_structure_error"
        row["error"] = str(exc)

    return row


def as_float(row: dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = [as_float(row, key) for row in rows if row.get("status") == "ok"]
    values = [value for value in values if math.isfinite(value)]
    return np.asarray(values, dtype=np.float64)


def quantile_or(values: np.ndarray, q: float, default: float) -> float:
    if values.size == 0:
        return default
    return float(np.quantile(values, q))


def add_rule(rules: list[tuple[str, str]], rule: str, severity: str) -> None:
    rules.append((rule, severity))


def classify_rows(rows: list[dict[str, object]]) -> dict[str, int]:
    adaptive = {
        "root_speed_max": min(quantile_or(finite_values(rows, "root_speed_max"), 0.99, 8.0), 8.0),
        "root_vz_max": min(quantile_or(finite_values(rows, "root_vz_max"), 0.99, 6.0), 6.0),
        "root_ang_vel_max": min(quantile_or(finite_values(rows, "root_ang_vel_max"), 0.99, 25.0), 25.0),
        "dof_vel_fd_max": min(quantile_or(finite_values(rows, "dof_vel_fd_max"), 0.99, 18.0), 18.0),
    }

    summary: dict[str, int] = {}
    for row in rows:
        rules: list[tuple[str, str]] = []
        name = str(row.get("name") or Path(str(row.get("source_pkl", ""))).stem).lower()

        if row.get("status") == "error":
            rules = [("load_or_structure_error", "critical")]
        else:
            frames = as_float(row, "frames", 0.0)
            fps = as_float(row, "fps", 0.0)
            if frames < 240:
                add_rule(rules, "too_short", "medium")
            if abs(fps - 120.0) > 1e-3:
                add_rule(rules, "unexpected_fps", "medium")

            qn_min = as_float(row, "root_rot_norm_min")
            qn_max = as_float(row, "root_rot_norm_max")
            if qn_min < 0.95 or qn_max > 1.05:
                add_rule(rules, "root_quaternion_norm", "high")

            if as_float(row, "root_z_min") < 0.35:
                add_rule(rules, "root_near_or_below_ground", "high")
            if as_float(row, "root_z_range") > 0.80:
                add_rule(rules, "large_root_height_range", "medium")
            if as_float(row, "root_z_max") > VERY_HIGH_ROOT_Z:
                add_rule(rules, "very_high_root", "high")
            if as_float(row, "root_speed_max") > adaptive["root_speed_max"]:
                add_rule(rules, "root_translation_jump", "high")
            if as_float(row, "root_ang_vel_max") > adaptive["root_ang_vel_max"]:
                add_rule(rules, "root_rotation_jump", "medium")
            if (
                as_float(row, "late_wrist_acc_max") > UPPER_BODY_LATE_WRIST_ACC
                and as_float(row, "late_wrist_vel_max") > UPPER_BODY_LATE_VEL
            ) or (
                as_float(row, "late_upper_body_acc_max") > UPPER_BODY_LATE_ARM_ACC
                and as_float(row, "late_upper_body_vel_max") > UPPER_BODY_LATE_VEL
                and as_float(row, "late_upper_body_reversal_frames", 0.0) > 0.0
            ) or (
                as_float(row, "tail_wrist_vel_max") > UPPER_BODY_TAIL_WRIST_VEL
                and as_float(row, "tail_wrist_acc_max") > UPPER_BODY_TAIL_WRIST_ACC
                and as_float(row, "tail_wrist_range_max") > UPPER_BODY_TAIL_WRIST_RANGE
            ):
                add_rule(rules, "upper_body_late_jitter", "medium")
            if as_float(row, "wrist_flicker_frame_count", 0.0) >= WRIST_FLICKER_MIN_FRAMES:
                add_rule(rules, "wrist_flicker", "medium")
            if as_float(row, "dof_abs_max") > 3.1:
                add_rule(rules, "large_dof_angle", "medium")
            if as_float(row, "joint_limit_violation_max") > 0.05:
                add_rule(rules, "mjcf_joint_limit_violation", "high")

            if as_float(row, "first_foot_z_min") > INITIAL_FOOT_FLOAT_Z:
                add_rule(rules, "initial_feet_floating", "high")
            if as_float(row, "first_foot_z_min") < INITIAL_FOOT_LOW_Z:
                add_rule(rules, "initial_feet_too_low_or_penetrating", "high")
            if as_float(row, "first_foot_height_diff") > INITIAL_FOOT_HEIGHT_DIFF_Z:
                add_rule(rules, "initial_feet_height_mismatch", "medium")
            if frames > 0 and as_float(row, "foot_ground_contact_frames", 0.0) == 0.0:
                add_rule(rules, "all_frames_feet_floating", "high")
            if as_float(row, "first_non_ankle_geom_z_min") < NON_ANKLE_BELOW_Z:
                add_rule(rules, "initial_non_ankle_body_below_ground", "high")
            if str(row.get("first_non_ankle_geom_touch_bodies", "")).strip():
                add_rule(rules, "initial_non_ankle_body_touch_ground", "high")

            if as_float(row, "foot_z_min") < -0.03:
                add_rule(rules, "foot_below_world_ground", "high")
            if as_float(row, "key_body_z_min") < -0.03:
                add_rule(rules, "key_body_below_world_ground", "high")
            if as_float(row, "non_ankle_geom_z_min") < NON_ANKLE_BELOW_Z:
                add_rule(rules, "non_ankle_body_below_ground", "high")
            if as_float(row, "non_ankle_geom_touch_frames", 0.0) > 0.0:
                add_rule(rules, "non_ankle_body_touch_ground", "high")
            if as_float(row, "foot_contact_height_range") > 0.10:
                add_rule(rules, "contact_ground_height_drift", "medium")

            if any(keyword in name for keyword in SITTING_POSTURE_KEYWORDS):
                add_rule(rules, "sitting_posture_keyword", "medium")
            if any(keyword in name for keyword in HARD_MOTION_KEYWORDS):
                add_rule(rules, "hard_motion_keyword", "medium")
            if COMMAND_VELOCITY_NAME_RE.match(name):
                add_rule(rules, "command_velocity_pattern", "medium")
            if any(keyword in name for keyword in TERRAIN_KEYWORDS):
                add_rule(rules, "terrain_or_forbidden_keyword", "medium")

        if rules:
            severity_order = {"medium": 1, "high": 2, "critical": 3}
            severity = max((sev for _, sev in rules), key=lambda sev: severity_order[sev])
            row["status"] = "suspect"
            row["severity"] = severity
            row["rules"] = ";".join(rule for rule, _ in rules)
            for rule, _ in rules:
                summary[rule] = summary.get(rule, 0) + 1
        else:
            row["status"] = "ok"
            row["severity"] = ""
            row["rules"] = ""

    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rule_names(row: dict[str, object]) -> list[str]:
    rules = str(row.get("rules", "")).strip()
    if not rules:
        return []
    return [rule for rule in rules.split(";") if rule]


def effective_rule_names(row: dict[str, object]) -> list[str]:
    rules = rule_names(row)
    for fatal_rule in FATAL_RULES:
        if fatal_rule in rules:
            return [fatal_rule]
    return rules


def summarize_rules(rows: list[dict[str, object]], *, effective: bool) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        names = effective_rule_names(row) if effective else rule_names(row)
        for rule in names:
            summary[rule] = summary.get(rule, 0) + 1
    return summary


def annotate_effective_rules(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row.get("status") == "suspect":
            row["effective_rules"] = ";".join(effective_rule_names(row))
        else:
            row["effective_rules"] = ""


def copy_suspects(rows: list[dict[str, object]], output_dir: Path) -> None:
    suspect_dir = output_dir / "suspect_pkl"
    by_rule_dir = output_dir / "suspect_by_rule"

    suspect_dir.mkdir(parents=True, exist_ok=True)
    for stale in suspect_dir.glob("*.pkl"):
        stale.unlink()

    if by_rule_dir.exists():
        shutil.rmtree(by_rule_dir)
    by_rule_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if row.get("status") != "suspect":
            continue
        src = Path(str(row["source_pkl"]))
        shutil.copy2(src, suspect_dir / src.name)
        for rule in effective_rule_names(row):
            rule_dir = by_rule_dir / rule
            rule_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, rule_dir / src.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-check A3 flat-ground training PKLs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-fk-samples", type=int, default=240)
    parser.add_argument("--copy-suspects", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = list_pkl_files(args.input)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No PKL files found under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Files: {len(files)}")
    print(f"Workers: {args.num_workers}")

    tasks = [(path.as_posix(), args.max_fk_samples) for path in files]
    worker_count = max(1, int(args.num_workers))
    rows: list[dict[str, object]] = []
    if worker_count == 1:
        init_worker(args.mjcf.as_posix())
        iterator = (process_file(task) for task in tasks)
        for idx, row in enumerate(iterator, start=1):
            rows.append(row)
            if idx == len(tasks) or idx % 100 == 0:
                print(f"  metrics {idx}/{len(tasks)}")
    else:
        with mp.Pool(
            processes=worker_count,
            initializer=init_worker,
            initargs=(args.mjcf.as_posix(),),
        ) as pool:
            for idx, row in enumerate(pool.imap_unordered(process_file, tasks), start=1):
                rows.append(row)
                if idx == len(tasks) or idx % 100 == 0:
                    print(f"  metrics {idx}/{len(tasks)}")

    rows.sort(key=lambda row: str(row.get("source_pkl", "")))
    raw_summary = classify_rows(rows)
    suspects = [row for row in rows if row.get("status") == "suspect"]
    annotate_effective_rules(rows)
    annotate_effective_rules(suspects)
    summary = summarize_rules(suspects, effective=True)
    fatal_summary = {
        rule: summary.get(rule, 0)
        for rule in FATAL_RULES
        if summary.get(rule, 0) > 0
    }

    write_csv(args.output / "all_metrics.csv", rows, METRIC_FIELDS)
    write_csv(args.output / "quality_manifest.csv", suspects, METRIC_FIELDS)

    summary_rows = [{"rule": rule, "count": count} for rule, count in sorted(summary.items())]
    summary_rows.insert(0, {"rule": "__total_files__", "count": len(rows)})
    summary_rows.insert(1, {"rule": "__suspect_files__", "count": len(suspects)})
    summary_rows.insert(2, {"rule": "__fatal_reject_files__", "count": sum(fatal_summary.values())})
    write_csv(args.output / "summary.csv", summary_rows, SUMMARY_FIELDS)

    raw_summary_rows = [{"rule": rule, "count": count} for rule, count in sorted(raw_summary.items())]
    raw_summary_rows.insert(0, {"rule": "__total_files__", "count": len(rows)})
    raw_summary_rows.insert(1, {"rule": "__suspect_files__", "count": len(suspects)})
    write_csv(args.output / "raw_rule_summary.csv", raw_summary_rows, SUMMARY_FIELDS)

    fatal_summary_rows = [{"rule": rule, "count": count} for rule, count in sorted(fatal_summary.items())]
    fatal_summary_rows.insert(0, {"rule": "__fatal_reject_files__", "count": sum(fatal_summary.values())})
    write_csv(args.output / "fatal_rules.csv", fatal_summary_rows, SUMMARY_FIELDS)

    if args.copy_suspects:
        copy_suspects(suspects, args.output)

    print(f"Checked: {len(rows)}")
    print(f"Suspects: {len(suspects)}")
    print(f"Saved: {args.output / 'all_metrics.csv'}")
    print(f"Saved: {args.output / 'quality_manifest.csv'}")
    print(f"Saved: {args.output / 'summary.csv'}")
    print(f"Saved: {args.output / 'raw_rule_summary.csv'}")
    print(f"Saved: {args.output / 'fatal_rules.csv'}")
    if args.copy_suspects:
        print(f"Copied suspects to: {args.output / 'suspect_pkl'}")
        print(f"Copied suspects by rule to: {args.output / 'suspect_by_rule'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
