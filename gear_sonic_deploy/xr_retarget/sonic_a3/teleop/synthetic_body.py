"""Synthesise Pico-format body tracking from an A3 joint-space motion.

Used to exercise the full Pico -> retarget -> stream -> sim chain without hardware,
and as a round-trip check of the retargeter (retarget(synthesise(q)) ~= q). The
generator places SMPL-24 joints on the robot kinematics, scales them to a different
body size, gives every joint an arbitrary local axis convention, and moves the whole
recording into an arbitrary device world frame; calibration has to undo all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sonic_a3.teleop.a3_motion_io import A3Motion
from sonic_a3.teleop.a3_retargeter import A3_IL_JOINT_NAMES, A3Retargeter, RetargetConfig, RetargetFrame
from sonic_a3.teleop.pico_body_source import (
    ANALOG_NAMES,
    BUTTON_NAMES,
    NUM_BODY_JOINTS,
    SMPL_JOINT_NAMES,
    RecordedBody,
    robot_to_unity_poses,
)

# SMPL joint -> (robot body for the position, local offset in that body, robot body for the rotation)
_JOINT_PLACEMENT: dict[str, tuple[str, tuple[float, float, float], str]] = {
    "pelvis": ("pelvis_link", (0.0, 0.0, 0.0), "pelvis_link"),
    "spine1": ("torso_Link", (0.0, 0.0, 0.10), "torso_Link"),
    "spine2": ("torso_Link", (0.0, 0.0, 0.20), "torso_Link"),
    "spine3": ("torso_Link", (0.0, 0.0, 0.28), "torso_Link"),
    "neck": ("torso_Link", (0.03, 0.0, 0.36), "torso_Link"),
    "head": ("torso_Link", (0.04, 0.0, 0.50), "torso_Link"),
}
for _side in ("left", "right"):
    _sign = 1.0 if _side == "left" else -1.0
    _JOINT_PLACEMENT.update(
        {
            f"{_side}_hip": (f"{_side}_hip_roll_Link", (0.0, 0.0, 0.0), f"{_side}_hip_yaw_Link"),
            f"{_side}_knee": (f"{_side}_knee_Link", (0.0, 0.0, 0.0), f"{_side}_knee_Link"),
            f"{_side}_ankle": (f"{_side}_ankle_roll_Link", (0.0, 0.0, 0.0), f"{_side}_ankle_roll_Link"),
            f"{_side}_foot": (f"{_side}_ankle_roll_Link", (0.15, 0.0, -0.06), f"{_side}_ankle_roll_Link"),
            f"{_side}_collar": ("torso_Link", (0.03, _sign * 0.07, 0.30), "torso_Link"),
            f"{_side}_shoulder": (f"{_side}_shoulder_roll_Link", (0.0, 0.0, 0.0), f"{_side}_shoulder_yaw_Link"),
            f"{_side}_elbow": (f"{_side}_elbow_Link", (0.0, 0.0, 0.0), f"{_side}_elbow_Link"),
            f"{_side}_wrist": (f"{_side}_wrist_yaw_Link", (0.0, 0.0, 0.0), f"{_side}_wrist_yaw_Link"),
            f"{_side}_hand": (f"{_side}_wrist_yaw_Link", (0.10, 0.0, 0.0), f"{_side}_wrist_yaw_Link"),
        }
    )


@dataclass
class SyntheticOptions:
    human_scale: float = 1.08
    calib_seconds: float = 1.5
    blend_seconds: float = 1.0
    world_yaw_deg: float = 37.0
    world_offset: tuple[float, float, float] = (1.2, -0.7, 0.0)
    random_axes: bool = True
    position_noise_m: float = 0.0
    rotation_noise_deg: float = 0.0
    seed: int = 0


def _calibration_motion(
    motion: A3Motion, calib: RetargetFrame, seconds: float, blend_seconds: float
) -> A3Motion:
    """Prepend a calibration hold (robot calibration pose, feet on the ground) and a blend into the motion."""
    n_hold = max(1, int(round(seconds * motion.fps)))
    n_blend = max(1, int(round(blend_seconds * motion.fps)))
    yaw = Rotation.from_quat(motion.root_quat_wxyz[0, [1, 2, 3, 0]]).as_euler("ZYX")[0]
    calib_quat = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
    calib_q = calib.dof_il
    calib_pos = np.array([motion.root_pos[0, 0], motion.root_pos[0, 1], calib.root_pos[2]])
    w = np.linspace(0.0, 1.0, n_blend + 2)[1:-1, None]
    dof = np.concatenate([np.tile(calib_q, (n_hold, 1)), (1 - w) * calib_q + w * motion.dof_il[0], motion.dof_il])
    pos = np.concatenate(
        [np.tile(calib_pos, (n_hold, 1)), (1 - w) * calib_pos + w * motion.root_pos[0], motion.root_pos]
    )
    rots = Rotation.from_quat(np.stack([calib_quat[[1, 2, 3, 0]], motion.root_quat_wxyz[0, [1, 2, 3, 0]]]))
    blend_rot = Rotation.concatenate([rots[0] * Rotation.from_rotvec(float(a) * (rots[0].inv() * rots[1]).as_rotvec()) for a in w[:, 0]])
    quat = np.concatenate(
        [np.tile(calib_quat, (n_hold, 1)), blend_rot.as_quat()[:, [3, 0, 1, 2]], motion.root_quat_wxyz]
    )
    return A3Motion(motion.fps, pos, quat, dof)


def synthesize_body_recording(
    motion: A3Motion,
    options: SyntheticOptions | None = None,
    *,
    mjcf: Path | None = None,
) -> RecordedBody:
    options = options or SyntheticOptions()
    rng = np.random.default_rng(options.seed)
    config = RetargetConfig.load()
    if mjcf is not None:
        config.mjcf = mjcf
    calib = A3Retargeter(config).calibration_reference()
    model = mujoco.MjModel.from_xml_path(str(config.mjcf))
    data = mujoco.MjData(model)
    qpos_addr = np.array([model.jnt_qposadr[model.joint(name).id] for name in A3_IL_JOINT_NAMES])

    full = _calibration_motion(motion, calib, options.calib_seconds, options.blend_seconds)
    calib_index = max(0, int(round(options.calib_seconds * motion.fps)) // 2)

    placements = [_JOINT_PLACEMENT[name] for name in SMPL_JOINT_NAMES]
    pos_body = np.array([model.body(p[0]).id for p in placements])
    rot_body = np.array([model.body(p[2]).id for p in placements])
    offsets = np.array([p[1] for p in placements])
    axes = Rotation.random(NUM_BODY_JOINTS, random_state=options.seed) if options.random_axes else Rotation.identity(NUM_BODY_JOINTS)
    world = Rotation.from_euler("z", options.world_yaw_deg, degrees=True)
    world_offset = np.asarray(options.world_offset, dtype=np.float64)

    frames = np.zeros((full.num_frames, NUM_BODY_JOINTS, 7))
    for t in range(full.num_frames):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = full.root_pos[t]
        data.qpos[3:7] = full.root_quat_wxyz[t]
        data.qpos[qpos_addr] = full.dof_il[t]
        mujoco.mj_kinematics(model, data)
        body_rot = data.xmat[pos_body].reshape(-1, 3, 3)
        positions = data.xpos[pos_body] + np.einsum("nij,nj->ni", body_rot, offsets)
        rotations = Rotation.from_matrix(data.xmat[rot_body].reshape(-1, 3, 3)) * axes
        positions = options.human_scale * positions
        if options.position_noise_m > 0.0:
            positions = positions + rng.normal(0.0, options.position_noise_m, positions.shape)
        if options.rotation_noise_deg > 0.0:
            noise = rng.normal(0.0, np.deg2rad(options.rotation_noise_deg), (NUM_BODY_JOINTS, 3))
            rotations = rotations * Rotation.from_rotvec(noise)
        frames[t] = robot_to_unity_poses(world.apply(positions) + world_offset, world * rotations)

    count = full.num_frames
    return RecordedBody(
        body_poses_unity=frames,
        timestamps=np.arange(count) / full.fps,
        controller_analog=np.zeros((count, len(ANALOG_NAMES)), dtype=np.float32),
        controller_buttons=np.zeros((count, len(BUTTON_NAMES)), dtype=bool),
        calib_index=calib_index,
    )
