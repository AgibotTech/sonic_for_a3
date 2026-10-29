"""Pico SMPL-24 body tracking -> A3 Ultra T3.0 29-DoF reference via mink differential IK.

Robot keypoint targets are built as a chain: robot bone lengths along the operator's
bone directions, starting from the scaled pelvis. Human proportions therefore only
enter through the root translation (scaled by the leg-length ratio). Link orientation
targets reuse the operator's joint rotations relative to a calibration pose, so the
device's per-joint axis conventions never need to be known.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from sonic_a3.contract.joint_order import IL_JOINT_NAMES as A3_IL_JOINT_NAMES
from sonic_a3.paths import RETARGET_CONFIG, STAND_CSV, resolve
from sonic_a3.teleop.a3_motion_io import load_a3_csv
from sonic_a3.teleop.pico_body_source import SMPL_INDEX, BodyFrame

DEFAULT_CONFIG = RETARGET_CONFIG

ROOT_BODY = "pelvis_link"
TORSO_BODY = "torso_Link"
SIDES = ("left", "right")


@dataclass(frozen=True)
class Bone:
    human_from: str
    human_to: str
    robot_from: str
    robot_to: str


def _limb_bones(side: str) -> dict[str, Bone]:
    return {
        "thigh": Bone(f"{side}_hip", f"{side}_knee", f"{side}_hip_roll_Link", f"{side}_knee_Link"),
        "shank": Bone(f"{side}_knee", f"{side}_ankle", f"{side}_knee_Link", f"{side}_ankle_roll_Link"),
        "upper_arm": Bone(f"{side}_shoulder", f"{side}_elbow", f"{side}_shoulder_roll_Link", f"{side}_elbow_Link"),
        "forearm": Bone(f"{side}_elbow", f"{side}_wrist", f"{side}_elbow_Link", f"{side}_wrist_yaw_Link"),
    }


BONES = {f"{side}_{name}": bone for side in SIDES for name, bone in _limb_bones(side).items()}


@dataclass(frozen=True)
class TaskSpec:
    body: str
    orientation_from: str | None
    position_cost: float
    orientation_cost: float


@dataclass
class RetargetConfig:
    mjcf: Path
    tasks: list[TaskSpec]
    calibration_q: dict[str, float]
    calibration_frames: int
    solver: str
    ik_dt: float
    ik_iterations: int
    calibration_iterations: int
    damping: float
    posture_cost: float
    limit_gain: float
    bone_direction_correction: bool
    ground_feet: bool
    ground_alpha: float
    output_alpha: float
    max_joint_step: float

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> RetargetConfig:
        raw = yaml.safe_load(Path(path).read_text())
        mjcf = resolve(raw["mjcf"])
        tasks = [
            TaskSpec(
                body=body,
                orientation_from=spec.get("orientation_from"),
                position_cost=float(spec.get("position_cost", 0.0)),
                orientation_cost=float(spec.get("orientation_cost", 0.0)),
            )
            for body, spec in raw["tasks"].items()
        ]
        for task in tasks:
            if task.orientation_from is not None and task.orientation_from not in SMPL_INDEX:
                raise ValueError(f"task {task.body}: unknown SMPL joint {task.orientation_from!r}")
        ik = raw.get("ik", {})
        root = raw.get("root", {})
        smoothing = raw.get("smoothing", {})
        return cls(
            mjcf=mjcf,
            tasks=tasks,
            calibration_q={str(k): float(v) for k, v in (raw.get("calibration_q") or {}).items()},
            calibration_frames=int(raw.get("calibration_frames", 10)),
            solver=str(ik.get("solver", "daqp")),
            ik_dt=float(ik.get("dt", 0.02)),
            ik_iterations=int(ik.get("iterations", 3)),
            calibration_iterations=int(ik.get("calibration_iterations", 100)),
            damping=float(ik.get("damping", 1e-3)),
            posture_cost=float(ik.get("posture_cost", 0.02)),
            limit_gain=float(ik.get("limit_gain", 0.95)),
            bone_direction_correction=bool(raw.get("bone_direction_correction", True)),
            ground_feet=bool(root.get("ground_feet", True)),
            ground_alpha=float(root.get("ground_alpha", 0.2)),
            output_alpha=float(smoothing.get("output_alpha", 1.0)),
            max_joint_step=float(smoothing.get("max_joint_step", math.inf)),
        )


@dataclass
class RetargetFrame:
    timestamp: float
    dof_il: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _rotation_between(a: np.ndarray, b: np.ndarray) -> Rotation:
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0.0:
            return Rotation.identity()
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(_unit(perp) * math.pi)
    return Rotation.from_rotvec(axis / s * math.atan2(s, c))


def _quat_wxyz(rot: Rotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


@dataclass
class _HumanPose:
    positions: np.ndarray
    rotations: Rotation


class A3Retargeter:
    def __init__(self, config: RetargetConfig | None = None):
        self.config = config or RetargetConfig.load()
        cfg = self.config
        self.model = mujoco.MjModel.from_xml_path(str(cfg.mjcf))
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)

        task_bodies = {task.body for task in cfg.tasks}
        missing = {ROOT_BODY, TORSO_BODY} - {t.body for t in cfg.tasks if t.orientation_from}
        if missing:
            raise ValueError(f"retarget config needs orientation tasks for {sorted(missing)}")
        for body in task_bodies | {b.robot_from for b in BONES.values()} | {b.robot_to for b in BONES.values()}:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
                raise ValueError(f"body {body!r} not found in {cfg.mjcf}")

        self.frame_tasks = {
            task.body: mink.FrameTask(
                frame_name=task.body,
                frame_type="body",
                position_cost=task.position_cost,
                orientation_cost=task.orientation_cost,
                lm_damping=1.0,
            )
            for task in cfg.tasks
        }
        self.posture_task = mink.PostureTask(self.model, cost=cfg.posture_cost)
        self.limits = [mink.ConfigurationLimit(self.model, gain=cfg.limit_gain)]

        self.qpos_addr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in A3_IL_JOINT_NAMES]
        )
        self._calib_qpos = self._robot_calibration_qpos()
        self._robot = self._robot_calibration_geometry()

        self._human_calib: _HumanPose | None = None
        self._align_yaw = Rotation.identity()
        self._align_origin = np.zeros(3)
        self._root_scale = 1.0
        self._orientation_offsets: dict[str, Rotation] = {}
        self._bone_corrections: dict[str, Rotation] = {}
        self._ground_dz = 0.0
        self._last_output: RetargetFrame | None = None

    @property
    def calibrated(self) -> bool:
        return self._human_calib is not None

    @property
    def root_scale(self) -> float:
        """Robot / operator leg-length ratio applied to root translation."""
        return self._root_scale

    # ------------------------------------------------------------------ robot side

    def _robot_calibration_qpos(self) -> np.ndarray:
        """Standing CSV frame 0, then optional ``calibration_q`` overrides.

        Isaac already holds this pose before streaming. Using it as the robot side
        of calibration keeps an operator who matches that stance from being mapped
        onto the MJCF zero pose (forearms forward).
        """
        qpos = self.model.qpos0.copy()
        qpos[:3] = 0.0
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        qpos[self.qpos_addr] = load_a3_csv(STAND_CSV).dof_il[0]
        for name, value in self.config.calibration_q.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"calibration_q joint {name!r} not found")
            qpos[self.model.jnt_qposadr[jid]] = value
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        sole_z = math.inf
        for side in SIDES:
            body_id = self.model.body(f"{side}_ankle_roll_Link").id
            for gid in np.flatnonzero(self.model.geom_bodyid == body_id):
                if self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]:
                    sole_z = min(sole_z, float(self.data.geom_xpos[gid, 2] - self.model.geom_rbound[gid]))
        if not math.isfinite(sole_z):
            raise ValueError("no foot collision geoms found to place the calibration pose on the ground")
        qpos[2] = -sole_z
        return qpos

    def _robot_calibration_geometry(self) -> dict:
        self.data.qpos[:] = self._calib_qpos
        mujoco.mj_forward(self.model, self.data)

        def pos(body: str) -> np.ndarray:
            return self.data.xpos[self.model.body(body).id].copy()

        def rot(body: str) -> Rotation:
            return Rotation.from_matrix(self.data.xmat[self.model.body(body).id].reshape(3, 3))

        root_p, root_r = pos(ROOT_BODY), rot(ROOT_BODY)
        torso_p, torso_r = pos(TORSO_BODY), rot(TORSO_BODY)
        geometry = {
            "root_pos": root_p,
            "torso_offset": torso_r.inv().apply(torso_p - root_p),
            "hip_offset": {s: root_r.inv().apply(pos(f"{s}_hip_roll_Link") - root_p) for s in SIDES},
            "shoulder_offset": {s: torso_r.inv().apply(pos(f"{s}_shoulder_roll_Link") - torso_p) for s in SIDES},
            "bone_length": {},
            "bone_dir": {},
            "link_rot": {task.body: rot(task.body) for task in self.config.tasks},
            "ankle_height": min(pos(f"{s}_ankle_roll_Link")[2] for s in SIDES),
        }
        for name, bone in BONES.items():
            vec = pos(bone.robot_to) - pos(bone.robot_from)
            geometry["bone_length"][name] = float(np.linalg.norm(vec))
            geometry["bone_dir"][name] = _unit(vec)
        geometry["leg_length"] = float(
            np.mean([geometry["bone_length"][f"{s}_thigh"] + geometry["bone_length"][f"{s}_shank"] for s in SIDES])
        )
        return geometry

    # ------------------------------------------------------------------ human side

    def _aligned(self, frame: BodyFrame) -> _HumanPose:
        positions = self._align_yaw.apply(np.asarray(frame.positions) - self._align_origin)
        return _HumanPose(positions=positions, rotations=self._align_yaw * frame.rotations)

    @staticmethod
    def _mean_pose(frames: Sequence[BodyFrame]) -> BodyFrame:
        positions = np.mean([np.asarray(f.positions) for f in frames], axis=0)
        quats = np.stack([f.rotations.as_quat() for f in frames], axis=1)
        rotations = Rotation.concatenate([Rotation.from_quat(quats[j]).mean() for j in range(quats.shape[0])])
        return BodyFrame(timestamp=frames[-1].timestamp, positions=positions, rotations=rotations)

    def calibrate(self, frames: BodyFrame | Sequence[BodyFrame]) -> None:
        """Calibrate with the operator matching the standing pose (see config header)."""
        if isinstance(frames, BodyFrame):
            frames = [frames]
        if not frames:
            raise ValueError("calibration needs at least one body frame")
        mean = self._mean_pose(frames)
        p = mean.positions
        left = (p[SMPL_INDEX["left_hip"]] - p[SMPL_INDEX["right_hip"]]) + (
            p[SMPL_INDEX["left_shoulder"]] - p[SMPL_INDEX["right_shoulder"]]
        )
        forward = np.cross(left, [0.0, 0.0, 1.0])
        yaw = math.atan2(forward[1], forward[0])
        self._align_yaw = Rotation.from_euler("z", -yaw)
        pelvis = p[SMPL_INDEX["pelvis"]]
        self._align_origin = np.array([pelvis[0], pelvis[1], 0.0])

        human = self._aligned(mean)
        self._human_calib = human
        robot = self._robot

        human_leg = np.mean(
            [
                np.linalg.norm(human.positions[SMPL_INDEX[f"{s}_hip"]] - human.positions[SMPL_INDEX[f"{s}_knee"]])
                + np.linalg.norm(human.positions[SMPL_INDEX[f"{s}_knee"]] - human.positions[SMPL_INDEX[f"{s}_ankle"]])
                for s in SIDES
            ]
        )
        self._root_scale = robot["leg_length"] / max(float(human_leg), 1e-3)

        self._orientation_offsets = {
            task.body: human.rotations[SMPL_INDEX[task.orientation_from]].inv() * robot["link_rot"][task.body]
            for task in self.config.tasks
            if task.orientation_from is not None
        }
        self._bone_corrections = {}
        for name, bone in BONES.items():
            joint_rot = human.rotations[SMPL_INDEX[bone.human_from]]
            u = human.positions[SMPL_INDEX[bone.human_to]] - human.positions[SMPL_INDEX[bone.human_from]]
            world_fix = _rotation_between(u, robot["bone_dir"][name])
            self._bone_corrections[name] = joint_rot.inv() * world_fix * joint_rot

        self._ground_dz = 0.0
        self._last_output = None
        self.configuration.update(self._calib_qpos.copy())
        self.posture_task.set_target(self._calib_qpos.copy())
        self._set_targets(self._targets(human))
        self._solve(self.config.calibration_iterations)

    # ------------------------------------------------------------------ targets + IK

    def _bone_dir(self, human: _HumanPose, name: str) -> np.ndarray:
        bone = BONES[name]
        u = _unit(human.positions[SMPL_INDEX[bone.human_to]] - human.positions[SMPL_INDEX[bone.human_from]])
        if not self.config.bone_direction_correction:
            return u
        joint_rot = human.rotations[SMPL_INDEX[bone.human_from]]
        return (joint_rot * self._bone_corrections[name] * joint_rot.inv()).apply(u)

    def _targets(self, human: _HumanPose) -> dict[str, tuple[np.ndarray | None, Rotation | None]]:
        robot = self._robot
        calib = self._human_calib
        orient = {
            body: human.rotations[SMPL_INDEX[task_src]] * self._orientation_offsets[body]
            for body, task_src in ((t.body, t.orientation_from) for t in self.config.tasks)
            if task_src is not None
        }
        pelvis_h = human.positions[SMPL_INDEX["pelvis"]]
        pelvis_c = calib.positions[SMPL_INDEX["pelvis"]]
        s = self._root_scale
        root = np.array(
            [s * pelvis_h[0], s * pelvis_h[1], robot["root_pos"][2] + s * (pelvis_h[2] - pelvis_c[2])]
        )
        positions = {ROOT_BODY: root}
        torso = root + orient[TORSO_BODY].apply(robot["torso_offset"])
        for side in SIDES:
            hip = root + orient[ROOT_BODY].apply(robot["hip_offset"][side])
            knee = hip + robot["bone_length"][f"{side}_thigh"] * self._bone_dir(human, f"{side}_thigh")
            ankle = knee + robot["bone_length"][f"{side}_shank"] * self._bone_dir(human, f"{side}_shank")
            shoulder = torso + orient[TORSO_BODY].apply(robot["shoulder_offset"][side])
            elbow = shoulder + robot["bone_length"][f"{side}_upper_arm"] * self._bone_dir(human, f"{side}_upper_arm")
            wrist = elbow + robot["bone_length"][f"{side}_forearm"] * self._bone_dir(human, f"{side}_forearm")
            positions[f"{side}_knee_Link"] = knee
            positions[f"{side}_ankle_roll_Link"] = ankle
            positions[f"{side}_elbow_Link"] = elbow
            positions[f"{side}_wrist_yaw_Link"] = wrist

        if self.config.ground_feet:
            lowest = min(positions[f"{s}_ankle_roll_Link"][2] for s in SIDES)
            dz = robot["ankle_height"] - lowest
            alpha = self.config.ground_alpha if self._last_output is not None else 1.0
            self._ground_dz += alpha * (dz - self._ground_dz)
            for body in positions:
                positions[body] = positions[body] + np.array([0.0, 0.0, self._ground_dz])

        return {
            task.body: (positions.get(task.body), orient.get(task.body)) for task in self.config.tasks
        }

    def _set_targets(self, targets: dict[str, tuple[np.ndarray | None, Rotation | None]]) -> None:
        for body, (position, rotation) in targets.items():
            current = self.configuration.get_transform_frame_to_world(body, "body")
            translation = current.translation() if position is None else position
            so3 = current.rotation() if rotation is None else mink.SO3(_quat_wxyz(rotation))
            self.frame_tasks[body].set_target(mink.SE3.from_rotation_and_translation(so3, translation))

    def _solve(self, iterations: int) -> None:
        tasks = [*self.frame_tasks.values(), self.posture_task]
        for _ in range(iterations):
            vel = mink.solve_ik(
                self.configuration,
                tasks,
                self.config.ik_dt,
                self.config.solver,
                damping=self.config.damping,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, self.config.ik_dt)

    def retarget(self, frame: BodyFrame) -> RetargetFrame:
        if not self.calibrated:
            raise RuntimeError("call calibrate() before retarget()")
        human = self._aligned(frame)
        self._set_targets(self._targets(human))
        self._solve(self.config.ik_iterations)

        q = self.configuration.q
        output = RetargetFrame(
            timestamp=float(frame.timestamp),
            dof_il=q[self.qpos_addr].copy(),
            root_pos=q[:3].copy(),
            root_quat_wxyz=q[3:7] / np.linalg.norm(q[3:7]),
        )
        return self._smooth(output)

    def _smooth(self, output: RetargetFrame) -> RetargetFrame:
        prev = self._last_output
        if prev is not None:
            alpha = self.config.output_alpha
            step = np.clip(alpha * (output.dof_il - prev.dof_il), -self.config.max_joint_step, self.config.max_joint_step)
            output.dof_il = prev.dof_il + step
            output.root_pos = prev.root_pos + alpha * (output.root_pos - prev.root_pos)
            q_prev = Rotation.from_quat(np.roll(prev.root_quat_wxyz, -1))
            q_new = Rotation.from_quat(np.roll(output.root_quat_wxyz, -1))
            blended = q_prev * Rotation.from_rotvec(alpha * (q_prev.inv() * q_new).as_rotvec())
            output.root_quat_wxyz = _quat_wxyz(blended)
        self._last_output = output
        return output

    def calibration_reference(self) -> RetargetFrame:
        """The robot calibration pose as a reference frame (used before streaming starts)."""
        q = self._calib_qpos
        return RetargetFrame(
            timestamp=0.0,
            dof_il=q[self.qpos_addr].copy(),
            root_pos=q[:3].copy(),
            root_quat_wxyz=q[3:7].copy(),
        )
