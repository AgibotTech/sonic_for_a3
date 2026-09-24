#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Prefer this checkout over another editable installation, and avoid shadowing
# the external Hugging Face trl package with gear_sonic/trl.
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
for _path in (_script_dir, _repo_root):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, _repo_root)

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for evaluation but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    import sys
    sys.exit(1)

import filelock  # noqa: I001
import json
import os
import shutil
import subprocess
import sys

import logging
from pathlib import Path

import easydict
import hydra
from hydra import utils
from hydra.core import hydra_config
from loguru import logger
import numpy as np
import omegaconf
import torch
import yaml

from gear_sonic import train_agent_trl
from gear_sonic.trl.utils import common as trl_utils_common, scheduler
from gear_sonic.utils import common as rl_utils_common, config_utils, obs_utils

config_utils.register_rl_resolvers()


class EvalDebugDumper:
    """Collect per-step policy/action/robot debug tensors for a single eval environment."""

    def __init__(
        self,
        output_path: str | os.PathLike | None = None,
        env_idx: int = 0,
        max_steps: int = 0,
        plot_dir: str | os.PathLike | None = None,
        make_plots: bool = True,
    ):
        self.output_path = Path(output_path) if output_path is not None else None
        self.env_idx = int(env_idx)
        self.max_steps = int(max_steps)
        if plot_dir is not None:
            self.plot_dir = Path(plot_dir)
        elif self.output_path is not None:
            self.plot_dir = self.output_path.with_suffix("")
        else:
            self.plot_dir = Path("eval_debug_plots")
        self.make_plots = bool(make_plots)
        self.records: dict[str, list[np.ndarray | int]] = {
            "step": [],
            "policy_action_mean": [],
            "action_raw": [],
            "action_processed": [],
            "joint_pos_ref": [],
            "joint_pos_target": [],
            "joint_pos": [],
            "joint_vel": [],
            "computed_torque": [],
            "applied_torque": [],
            "motor_velocity": [],
            "requested_motor_torque": [],
            "applied_motor_torque": [],
            "motor_torque_limit": [],
            "motor_utilization": [],
            "motor_saturated": [],
            "contact_force_norm": [],
            "contact_force_w": [],
            "anchor_pos_ref": [],
            "anchor_pos_robot": [],
            "anchor_quat_ref": [],
            "anchor_quat_robot": [],
            "anchor_pos_error_w": [],
            "anchor_pos_error_norm": [],
            "anchor_rot_error": [],
            "anchor_rot_error_squared": [],
            "motion_id": [],
            "motion_time_step": [],
            "dones": [],
            "terminated": [],
            "time_outs": [],
            "termination_terms": [],
            # Native simulator signals needed to build the same 50 Hz
            # kinematic/control schema as gear_sonic_deploy EvalFrameV7.
            # Keep these in IsaacLab order here; the offline exporter performs
            # the name-based permutation into A3's 31-DOF SDK order.
            "sim_time_s": [],
            "sim_joint_pos": [],
            "sim_joint_vel": [],
            "sim_computed_torque": [],
            "sim_applied_torque": [],
            "sim_joint_pos_target": [],
            "sim_joint_vel_target": [],
            "sim_joint_effort_target": [],
            "sim_joint_stiffness": [],
            "sim_joint_damping": [],
            "sim_pelvis_pos_w": [],
            "sim_pelvis_quat_wxyz": [],
            "sim_pelvis_gyro_b": [],
            "sim_torso_quat_wxyz": [],
            "sim_torso_gyro_b": [],
            "sim_reference_joint_pos": [],
            "sim_reference_joint_vel": [],
            "sim_reference_pelvis_pos_w": [],
            "sim_reference_pelvis_quat_wxyz": [],
        }
        self.joint_names: list[str] | None = None
        self.full_joint_names: list[str] | None = None
        self.reference_joint_names: list[str] | None = None
        self.body_names: list[str] | None = None
        self.motion_keys: list[str] | None = None
        self.contact_body_names: list[str] | None = None
        self.motor_names: list[str] | None = None
        self.termination_term_names: list[str] | None = None
        self._joint_ids = None
        self._motor_actuators = None
        self._torso_body_index: int | None = None
        self._control_dt_s: float | None = None

    @staticmethod
    def _to_numpy(value, env_idx: int | None = None) -> np.ndarray:
        if value is None:
            return np.asarray([])
        if hasattr(value, "detach"):
            value = value.detach()
        if env_idx is not None and hasattr(value, "dim") and value.dim() > 0:
            value = value[env_idx]
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        quat = quat / np.clip(norm, 1e-8, None)
        w, x, y, z = np.moveaxis(quat, -1, 0)
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.unwrap(np.stack((roll, pitch, yaw), axis=-1), axis=0)

    def enabled_for_step(self) -> bool:
        return self.max_steps <= 0 or len(self.records["step"]) < self.max_steps

    def capture(self, step_count: int, env, actor_state: dict, dones=None) -> None:
        if not self.enabled_for_step():
            return
        base_env = env.env
        robot = base_env.scene["robot"]
        action_term = base_env.action_manager.get_term("joint_pos")
        command = base_env.command_manager.get_term("motion")

        if self.joint_names is None:
            joint_ids = action_term._joint_ids  # noqa: SLF001
            self._joint_ids = joint_ids
            self.joint_names = list(action_term._joint_names)  # noqa: SLF001
            self.full_joint_names = list(robot.joint_names)
            self.body_names = list(robot.body_names)
            self.motion_keys = list(getattr(command.motion_lib, "curr_motion_keys", []))
            self._control_dt_s = float(getattr(base_env, "step_dt", 0.02))
            for candidate in ("torso_Link", "torso_link", "torso"):
                if candidate in self.body_names:
                    self._torso_body_index = self.body_names.index(candidate)
                    break
            if self._torso_body_index is None:
                self._torso_body_index = next(
                    (idx for idx, name in enumerate(self.body_names) if "torso" in name.lower()),
                    None,
                )

        env_idx = None if self.env_idx < 0 else self.env_idx
        joint_ids = self._joint_ids
        self.records["step"].append(int(step_count))
        self.records["policy_action_mean"].append(self._to_numpy(actor_state.get("actions"), env_idx))
        self.records["action_raw"].append(self._to_numpy(action_term.raw_actions, env_idx))
        self.records["action_processed"].append(self._to_numpy(action_term.processed_actions, env_idx))
        ref_joint_pos = command.joint_pos
        if getattr(command, "has_dof_mismatch", False):
            ref_joint_pos_full = robot.data.joint_pos.new_full(robot.data.joint_pos.shape, np.nan)
            ref_joint_pos_full[:, command.body_joint_indices] = ref_joint_pos
            if hasattr(command, "extra_joint_indices") and hasattr(command, "extra_default_positions"):
                ref_joint_pos_full[:, command.extra_joint_indices] = command.extra_default_positions
            ref_joint_pos = ref_joint_pos_full
        self.records["joint_pos_ref"].append(self._to_numpy(ref_joint_pos[:, joint_ids], env_idx))
        self.records["joint_pos_target"].append(
            self._to_numpy(robot.data.joint_pos_target[:, joint_ids], env_idx)
        )
        self.records["joint_pos"].append(self._to_numpy(robot.data.joint_pos[:, joint_ids], env_idx))
        self.records["joint_vel"].append(self._to_numpy(robot.data.joint_vel[:, joint_ids], env_idx))
        self.records["computed_torque"].append(
            self._to_numpy(robot.data.computed_torque[:, joint_ids], env_idx)
        )
        self.records["applied_torque"].append(
            self._to_numpy(robot.data.applied_torque[:, joint_ids], env_idx)
        )
        if self._motor_actuators is None:
            self._motor_actuators = [
                actuator
                for actuator in robot.actuators.values()
                if hasattr(actuator, "motor_names")
            ]
            self.motor_names = [
                name
                for actuator in self._motor_actuators
                for name in actuator.motor_names
            ]
        for field in (
            "motor_velocity",
            "requested_motor_torque",
            "applied_motor_torque",
            "motor_torque_limit",
            "motor_utilization",
            "motor_saturated",
        ):
            if self._motor_actuators:
                value = torch.cat(
                    [getattr(actuator, field) for actuator in self._motor_actuators],
                    dim=-1,
                )
                self.records[field].append(self._to_numpy(value, env_idx))
            else:
                self.records[field].append(np.asarray([]))
        contact_force_w = np.asarray([])
        if hasattr(base_env.scene, "sensors") and "contact_forces" in base_env.scene.sensors:
            contact_sensor = base_env.scene["contact_forces"]
            if self.contact_body_names is None:
                self.contact_body_names = list(contact_sensor.body_names)
            contact_force_w = self._to_numpy(contact_sensor.data.net_forces_w, env_idx)
        self.records["contact_force_w"].append(contact_force_w)
        if contact_force_w.size:
            self.records["contact_force_norm"].append(np.linalg.norm(contact_force_w, axis=-1))
        else:
            self.records["contact_force_norm"].append(np.asarray([]))
        anchor_pos_ref = command.anchor_pos_w
        anchor_pos_robot = command.robot_anchor_pos_w
        anchor_quat_ref = command.anchor_quat_w
        anchor_quat_robot = command.robot_anchor_quat_w
        anchor_pos_error = anchor_pos_ref - anchor_pos_robot
        self.records["anchor_pos_ref"].append(self._to_numpy(anchor_pos_ref, env_idx))
        self.records["anchor_pos_robot"].append(self._to_numpy(anchor_pos_robot, env_idx))
        self.records["anchor_quat_ref"].append(self._to_numpy(anchor_quat_ref, env_idx))
        self.records["anchor_quat_robot"].append(self._to_numpy(anchor_quat_robot, env_idx))
        self.records["anchor_pos_error_w"].append(self._to_numpy(anchor_pos_error, env_idx))
        self.records["anchor_pos_error_norm"].append(
            self._to_numpy(anchor_pos_error.norm(dim=-1), env_idx)
        )
        try:
            from isaaclab.utils.math import quat_error_magnitude

            anchor_rot_error = quat_error_magnitude(
                command.anchor_quat_w, command.robot_anchor_quat_w
            )
            self.records["anchor_rot_error"].append(self._to_numpy(anchor_rot_error, env_idx))
            self.records["anchor_rot_error_squared"].append(
                self._to_numpy(anchor_rot_error.square(), env_idx)
            )
        except Exception:  # noqa: BLE001
            self.records["anchor_rot_error"].append(np.asarray([]))
            self.records["anchor_rot_error_squared"].append(np.asarray([]))
        self.records["motion_id"].append(self._to_numpy(command.motion_ids, env_idx))
        self.records["motion_time_step"].append(self._to_numpy(command.time_steps, env_idx))
        self.records["dones"].append(self._to_numpy(dones, env_idx))

        self.records["sim_time_s"].append(float(step_count) * float(self._control_dt_s))
        for field, value in (
            ("sim_joint_pos", robot.data.joint_pos),
            ("sim_joint_vel", robot.data.joint_vel),
            ("sim_computed_torque", robot.data.computed_torque),
            ("sim_applied_torque", robot.data.applied_torque),
            ("sim_joint_pos_target", robot.data.joint_pos_target),
            ("sim_joint_vel_target", robot.data.joint_vel_target),
            ("sim_joint_effort_target", robot.data.joint_effort_target),
            ("sim_joint_stiffness", robot.data.joint_stiffness),
            ("sim_joint_damping", robot.data.joint_damping),
            ("sim_pelvis_pos_w", robot.data.root_pos_w),
            ("sim_pelvis_quat_wxyz", robot.data.root_quat_w),
            ("sim_pelvis_gyro_b", robot.data.root_ang_vel_b),
        ):
            self.records[field].append(self._to_numpy(value, env_idx))
        if self._torso_body_index is not None:
            from isaaclab.utils.math import quat_apply_inverse

            torso_quat_w = robot.data.body_quat_w[:, self._torso_body_index]
            torso_ang_vel_w = robot.data.body_ang_vel_w[:, self._torso_body_index]
            torso_gyro_b = quat_apply_inverse(torso_quat_w, torso_ang_vel_w)
            self.records["sim_torso_quat_wxyz"].append(
                self._to_numpy(torso_quat_w, env_idx)
            )
            self.records["sim_torso_gyro_b"].append(
                self._to_numpy(torso_gyro_b, env_idx)
            )
        else:
            self.records["sim_torso_quat_wxyz"].append(np.asarray([]))
            self.records["sim_torso_gyro_b"].append(np.asarray([]))

        reference_joint_pos = command.joint_pos
        reference_joint_vel = command.joint_vel
        if self.reference_joint_names is None:
            if getattr(command, "has_dof_mismatch", False):
                self.reference_joint_names = [
                    robot.joint_names[int(index)] for index in command.body_joint_indices
                ]
            else:
                self.reference_joint_names = list(self.joint_names)
        self.records["sim_reference_joint_pos"].append(
            self._to_numpy(reference_joint_pos, env_idx)
        )
        self.records["sim_reference_joint_vel"].append(
            self._to_numpy(reference_joint_vel, env_idx)
        )
        self.records["sim_reference_pelvis_pos_w"].append(
            self._to_numpy(command.anchor_pos_w, env_idx)
        )
        self.records["sim_reference_pelvis_quat_wxyz"].append(
            self._to_numpy(command.anchor_quat_w, env_idx)
        )

        termination_manager = getattr(base_env, "termination_manager", None)
        if termination_manager is not None:
            if self.termination_term_names is None:
                self.termination_term_names = list(termination_manager.active_terms)
            self.records["terminated"].append(self._to_numpy(termination_manager.terminated, env_idx))
            self.records["time_outs"].append(self._to_numpy(termination_manager.time_outs, env_idx))
            term_values = [
                self._to_numpy(termination_manager.get_term(name), env_idx)
                for name in self.termination_term_names
            ]
            self.records["termination_terms"].append(np.asarray(term_values, dtype=np.bool_))
        else:
            self.records["terminated"].append(np.asarray([]))
            self.records["time_outs"].append(np.asarray([]))
            self.records["termination_terms"].append(np.asarray([]))

    def save(self) -> None:
        if not self.records["step"]:
            return
        arrays = {}
        for key, values in self.records.items():
            arrays[key] = np.asarray(values)
        arrays["joint_names"] = np.asarray(self.joint_names or [], dtype=object)
        arrays["full_joint_names"] = np.asarray(self.full_joint_names or [], dtype=object)
        arrays["reference_joint_names"] = np.asarray(
            self.reference_joint_names or [], dtype=object
        )
        arrays["body_names"] = np.asarray(self.body_names or [], dtype=object)
        arrays["motion_keys"] = np.asarray(self.motion_keys or [], dtype=object)
        arrays["telemetry_schema"] = np.asarray("sonic_isaaclab_eval_telemetry_v1")
        arrays["control_dt_s"] = np.asarray(self._control_dt_s or 0.02, dtype=np.float64)
        arrays["motor_names"] = np.asarray(self.motor_names or [], dtype=object)
        arrays["contact_body_names"] = np.asarray(self.contact_body_names or [], dtype=object)
        arrays["termination_term_names"] = np.asarray(self.termination_term_names or [], dtype=object)
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.output_path, **arrays)
            logger.info(f"Wrote eval debug dump to {self.output_path}")
        if self.make_plots:
            self._plot(arrays)

    def _plot(self, arrays: dict[str, np.ndarray]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        for legacy_name in (
            "anchor_error.png",
            "joint_error.png",
            "joint_state_blue_vs_action_target_orange.png",
            "torque_applied_red_vs_computed_purple.png",
        ):
            (self.plot_dir / legacy_name).unlink(missing_ok=True)
        time_axis = arrays["step"]
        joint_names = [str(name) for name in arrays["joint_names"]]
        self._plot_state_vs_target(plt, time_axis, joint_names, arrays)
        self._plot_joint_state_vs_ref(plt, time_axis, joint_names, arrays)
        self._plot_anchor_state_vs_ref(plt, time_axis, arrays)
        self._plot_effort(plt, time_axis, joint_names, arrays)
        self._plot_contact_forces(plt, time_axis, arrays)
        logger.info(f"Wrote eval debug plots to {self.plot_dir}")

    def _subplot_grid(self, plt, num_plots: int, title: str, max_cols: int = 3):
        cols = min(max_cols, max(1, num_plots))
        rows = int(np.ceil(num_plots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 2.35 * rows), sharex=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for ax in axes[num_plots:]:
            ax.set_visible(False)
        fig.suptitle(title, y=0.995, fontsize=14)
        return fig, axes[:num_plots]

    def _joint_groups(self, joint_names: list[str]) -> list[tuple[str, list[int]]]:
        group_specs = [
            ("left_leg", ("left_hip", "left_knee", "left_ankle")),
            ("right_leg", ("right_hip", "right_knee", "right_ankle")),
            ("waist", ("waist",)),
            ("left_arm", ("left_shoulder", "left_elbow", "left_wrist")),
            ("right_arm", ("right_shoulder", "right_elbow", "right_wrist")),
        ]
        groups = []
        used: set[int] = set()
        for group_name, keys in group_specs:
            indices = [idx for idx, name in enumerate(joint_names) if any(key in name for key in keys)]
            if indices:
                groups.append((group_name, indices))
                used.update(indices)
        rest = [idx for idx in range(len(joint_names)) if idx not in used]
        if rest:
            groups.append(("other", rest))
        return groups

    def _joint_group_grid(self, plt, joint_names: list[str], title: str):
        groups = self._joint_groups(joint_names)
        cols = max(len(indices) for _, indices in groups)
        rows = len(groups)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(3.25 * cols, 2.1 * rows),
            sharex=True,
            squeeze=False,
        )
        fig.suptitle(title, y=0.997, fontsize=14)
        for row, (group_name, indices) in enumerate(groups):
            for col in range(cols):
                if col >= len(indices):
                    axes[row, col].set_visible(False)
        return fig, axes, groups

    def _body_groups(self, body_names: list[str]) -> list[tuple[str, list[int]]]:
        group_specs = [
            ("left_leg", ("left_hip", "left_knee", "left_ankle")),
            ("right_leg", ("right_hip", "right_knee", "right_ankle")),
            ("torso_waist", ("pelvis", "waist", "torso", "head", "neck", "base")),
            ("left_arm", ("left_shoulder", "left_elbow", "left_wrist", "left_hand")),
            ("right_arm", ("right_shoulder", "right_elbow", "right_wrist", "right_hand")),
        ]
        groups = []
        used: set[int] = set()
        for group_name, keys in group_specs:
            indices = [idx for idx, name in enumerate(body_names) if any(key in name for key in keys)]
            if indices:
                groups.append((group_name, indices))
                used.update(indices)
        rest = [idx for idx in range(len(body_names)) if idx not in used]
        if rest:
            groups.append(("other", rest))
        return groups

    def _body_group_grid(self, plt, body_names: list[str], title: str):
        groups = self._body_groups(body_names)
        cols = max(len(indices) for _, indices in groups)
        rows = len(groups)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(3.15 * cols, 2.1 * rows),
            sharex=True,
            squeeze=False,
        )
        fig.suptitle(title, y=0.997, fontsize=14)
        for row, (_, indices) in enumerate(groups):
            for col in range(cols):
                if col >= len(indices):
                    axes[row, col].set_visible(False)
        return fig, axes, groups

    def _plot_state_vs_target(self, plt, time_axis: np.ndarray, joint_names: list[str], arrays: dict[str, np.ndarray]):
        state = arrays["joint_pos"]
        target = arrays["joint_pos_target"]
        fig, axes, groups = self._joint_group_grid(
            plt,
            joint_names,
            "Joint State vs Action Target (action * scale + default)",
        )
        for row, (group_name, indices) in enumerate(groups):
            for col, idx in enumerate(indices):
                ax = axes[row, col]
                ax.plot(
                    time_axis,
                    state[:, idx],
                    linewidth=1.25,
                    color="tab:blue",
                    label="STATE actual joint_pos",
                )
                ax.plot(
                    time_axis,
                    target[:, idx],
                    linewidth=1.15,
                    linestyle="--",
                    color="tab:orange",
                    label="ACTION target",
                )
                ax.set_title(joint_names[idx], fontsize=8.5, pad=2)
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=7)
                if row == len(groups) - 1:
                    ax.set_xlabel("eval step", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{group_name.replace('_', ' ')}\nrad", fontsize=8)
        axes[0, 0].legend(loc="upper left", fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.45, w_pad=0.25)
        fig.savefig(self.plot_dir / "state_vs_target.png", dpi=160)
        plt.close(fig)

    def _plot_joint_state_vs_ref(self, plt, time_axis: np.ndarray, joint_names: list[str], arrays: dict[str, np.ndarray]):
        state = arrays["joint_pos"]
        reference = arrays["joint_pos_ref"]
        fig, axes, groups = self._joint_group_grid(plt, joint_names, "Joint State vs Reference Motion")
        for row, (group_name, indices) in enumerate(groups):
            for col, idx in enumerate(indices):
                ax = axes[row, col]
                ax.plot(time_axis, state[:, idx], linewidth=1.2, color="tab:blue", label="STATE actual joint_pos")
                ax.plot(
                    time_axis,
                    reference[:, idx],
                    linewidth=1.1,
                    linestyle="--",
                    color="tab:orange",
                    label="REF motion joint_pos",
                )
                ax.set_title(joint_names[idx], fontsize=8.5, pad=2)
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=7)
                if row == len(groups) - 1:
                    ax.set_xlabel("eval step", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{group_name.replace('_', ' ')}\nrad", fontsize=8)
        axes[0, 0].legend(loc="upper left", fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.45, w_pad=0.25)
        fig.savefig(self.plot_dir / "joint_state_vs_ref.png", dpi=160)
        plt.close(fig)

    def _plot_anchor_state_vs_ref(self, plt, time_axis: np.ndarray, arrays: dict[str, np.ndarray]):
        pos_ref = arrays.get("anchor_pos_ref")
        pos_robot = arrays.get("anchor_pos_robot")
        quat_ref = arrays.get("anchor_quat_ref")
        quat_robot = arrays.get("anchor_quat_robot")
        if pos_ref is None or pos_robot is None or pos_ref.size == 0 or pos_robot.size == 0:
            return
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 6.4), sharex=True, squeeze=False)
        axes = axes.reshape(-1)
        for dim, axis_name in enumerate(("X", "Y", "Z")):
            ax = axes[dim]
            ax.plot(time_axis, pos_robot[:, dim], linewidth=1.2, color="tab:blue", label="STATE robot")
            ax.plot(time_axis, pos_ref[:, dim], linewidth=1.1, linestyle="--", color="tab:orange", label="REF motion")
            ax.set_title(f"Anchor Position {axis_name}", fontsize=10)
            ax.set_ylabel("m")
        if quat_ref is not None and quat_robot is not None and quat_ref.size and quat_robot.size:
            ref_rpy = self._quat_wxyz_to_rpy(quat_ref)
            robot_rpy = self._quat_wxyz_to_rpy(quat_robot)
            for dim, axis_name in enumerate(("Roll", "Pitch", "Yaw")):
                ax = axes[dim + 3]
                ax.plot(time_axis, robot_rpy[:, dim], linewidth=1.2, color="tab:blue", label="STATE robot")
                ax.plot(
                    time_axis,
                    ref_rpy[:, dim],
                    linewidth=1.1,
                    linestyle="--",
                    color="tab:orange",
                    label="REF motion",
                )
                ax.set_title(f"Anchor Orientation {axis_name}", fontsize=10)
                ax.set_ylabel("rad")

        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=8)
        axes[0].legend(loc="upper left", fontsize=8)
        for ax in axes[3:]:
            ax.set_xlabel("eval step", fontsize=9)
        fig.suptitle("Anchor State vs Reference Motion", y=0.995, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=0.55, w_pad=0.35)
        fig.savefig(self.plot_dir / "anchor_state_vs_ref.png", dpi=160)
        plt.close(fig)

    def _plot_effort(self, plt, time_axis: np.ndarray, joint_names: list[str], arrays: dict[str, np.ndarray]):
        applied = arrays["applied_torque"]
        computed = arrays["computed_torque"]
        fig, axes, groups = self._joint_group_grid(plt, joint_names, "Joint Effort / Torque")
        for row, (group_name, indices) in enumerate(groups):
            for col, idx in enumerate(indices):
                ax = axes[row, col]
                ax.plot(time_axis, applied[:, idx], linewidth=1.2, color="tab:red", label="APPLIED torque")
                ax.plot(
                    time_axis,
                    computed[:, idx],
                    linewidth=1.0,
                    linestyle="--",
                    color="tab:purple",
                    alpha=0.75,
                    label="COMPUTED torque",
                )
                ax.set_title(joint_names[idx], fontsize=8.5, pad=2)
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=7)
                if row == len(groups) - 1:
                    ax.set_xlabel("eval step", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{group_name.replace('_', ' ')}\nNm", fontsize=8)
        axes[0, 0].legend(loc="upper left", fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.45, w_pad=0.25)
        fig.savefig(self.plot_dir / "effort.png", dpi=160)
        plt.close(fig)

    def _plot_contact_forces(self, plt, time_axis: np.ndarray, arrays: dict[str, np.ndarray]):
        contact = arrays.get("contact_force_norm")
        contact_w = arrays.get("contact_force_w")
        body_names = [str(name) for name in arrays.get("contact_body_names", [])]
        if contact is None or contact.size == 0 or not body_names:
            return
        peak = np.nanmax(contact, axis=0)
        fig, axes, groups = self._body_group_grid(plt, body_names, "Contact Force Per Body")
        has_xyz = contact_w is not None and contact_w.ndim == 3 and contact_w.shape[-1] == 3
        for row, (group_name, indices) in enumerate(groups):
            for col, idx in enumerate(indices):
                ax = axes[row, col]
                ax.plot(time_axis, contact[:, idx], linewidth=1.1, color="black", label="norm")
                if has_xyz:
                    ax.plot(time_axis, contact_w[:, idx, 0], linewidth=0.75, alpha=0.7, label="Fx")
                    ax.plot(time_axis, contact_w[:, idx, 1], linewidth=0.75, alpha=0.7, label="Fy")
                    ax.plot(time_axis, contact_w[:, idx, 2], linewidth=0.75, alpha=0.7, label="Fz")
                ax.set_title(f"{body_names[idx]} peak={peak[idx]:.1f}N", fontsize=8.5, pad=2)
                ax.grid(True, alpha=0.25)
                ax.tick_params(axis="both", labelsize=7)
                if row == len(groups) - 1:
                    ax.set_xlabel("eval step", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{group_name.replace('_', ' ')}\nN", fontsize=8)
        axes[0, 0].legend(loc="upper left", fontsize=6.5, ncol=2)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=0.45, w_pad=0.25)
        fig.savefig(self.plot_dir / "contact_forces.png", dpi=160)
        plt.close(fig)


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: omegaconf.OmegaConf):

    hydra_log_path = os.path.join(hydra_config.HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gear_sonic.utils import logging as utils_logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(utils_logging.HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                raw = file.read()
            # Backward compatibility: rewrite internal repo module paths to release repo paths
            raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
            raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
            raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
            raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
            raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
            raw = raw.replace("groot/rl/data/", "gear_sonic/data/")
            raw = raw.replace("assets/bm/unitree_description/", "assets/robot_description/")
            raw = raw.replace("1215_bones_seed_filtered", "bones_seed_smpl")
            import io
            train_config = omegaconf.OmegaConf.load(io.StringIO(raw))

            if train_config.eval_overrides is not None:
                train_config = omegaconf.OmegaConf.merge(train_config, train_config.eval_overrides)

            config = omegaconf.OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    elif override_config.eval_overrides is not None:
        config = override_config.copy()
        eval_overrides = omegaconf.OmegaConf.to_container(config.eval_overrides, resolve=True)
        for arg in sys.argv[1:]:
            if not arg.startswith("+"):
                key = arg.split("=")[0]
                if key in eval_overrides:
                    del eval_overrides[key]
        config.eval_overrides = omegaconf.OmegaConf.create(eval_overrides)
        config = omegaconf.OmegaConf.merge(config, eval_overrides)
    else:
        config = override_config

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))  # noqa: SIM115
        if config.get("wandb", None) is not None and meta.get("wandb_run"):
            config.wandb.wandb_id = meta["wandb_run"]
            print(f"resume wandb from run: {config.wandb.wandb_id}")  # noqa: T201

    with omegaconf.open_dict(config):
        for event in config.manager_env.config.get("train_only_events", []):
            if event in config.manager_env.events:
                config.manager_env.events.pop(event)
            remove_schedule_keys = []
            for key in config.trainer.get("schedule_dict", {}):
                if event in key:
                    remove_schedule_keys.append(key)
            for key in remove_schedule_keys:
                config.trainer.schedule_dict.pop(key)

        for termination in config.manager_env.config.get("train_only_terminations", []):
            if termination in config.manager_env.terminations:
                config.manager_env.terminations.pop(termination)

    use_encoder = config.get("use_encoder", None)
    if use_encoder is not None:
        encoder_sample_probs = config.manager_env.commands.motion.encoder_sample_probs
        if encoder_sample_probs is not None:
            if use_encoder not in encoder_sample_probs:
                raise ValueError(
                    f"use_encoder={use_encoder} is not in encoder_sample_probs: "
                    f"{list(encoder_sample_probs.keys())}"
                )
            for encoder in encoder_sample_probs:
                encoder_sample_probs[encoder] = 1.0 if encoder == use_encoder else 0.0
            print(f"Using encoder: {use_encoder}")  # noqa: T201
            print(f"Encoder sample probs: {encoder_sample_probs}")  # noqa: T201

    simulator_type = "IsaacSim"
    env_config = config.manager_env

    import datetime as dt

    import accelerate
    import torch  # noqa: E402, RUF100

    kwargs = accelerate.InitProcessGroupKwargs(timeout=dt.timedelta(seconds=6000))
    accelerator = accelerate.Accelerator(kwargs_handlers=[kwargs])

    device = str(accelerator.device)
    if accelerator.device.type == "cuda":
        try:
            torch.cuda.set_device(accelerator.local_process_index)
        except Exception:  # noqa: S110, BLE001
            pass

    device = str(accelerator.device)
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    rl_utils_common.seeding(config.seed)

    def _pick_display_gpu_index(default_idx: int = 0) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,display_active,name", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    idx, active = int(parts[0]), parts[1].lower()
                    if active.startswith("enabled") or active.startswith("on"):
                        return idx
        except Exception:  # noqa: S110, BLE001
            pass
        return default_idx

    render_gpu_idx = _pick_display_gpu_index(default_idx=0)

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2
        import argparse

        parser = argparse.ArgumentParser(description="Evaluate an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args  # noqa: RUF005
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = env_config.config.get(
            "render_results", False
        ) or env_config.config.get("enable_cameras", False)

        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        base_kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )
        if args_cli.headless:
            args_cli.kit_args = base_kit_args + " --no-window"
        else:
            args_cli.kit_args = base_kit_args + f" --/renderer/activeGpu={render_gpu_idx}"

        _lock_path = "/tmp/isaaclab_app_launcher.lock"  # noqa: S108
        with filelock.FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    unresolved_conf = omegaconf.OmegaConf.to_container(config, resolve=False)  # noqa: F841
    os.chdir(hydra.utils.get_original_cwd())

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]

    if env_config.config.get("save_rendering_dir", None) is None:
        env_config.config.save_rendering_dir = str(
            checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
        )

    metrics_file = config.get("metrics_file", None)
    if metrics_file is not None:
        metrics_file = Path(metrics_file)
        assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist"
        if metrics_file.exists():
            metrics = json.load(open(metrics_file))  # noqa: SIM115
            all_dict = metrics["eval/all_metrics_dict"]

            # Check if this is grab evaluation (has success_lift)
            has_obj_metrics = "obj_pos_error" in all_dict
            if "success_lift" in all_dict:
                # Grab evaluation: prioritize failed grasps (not lifted) and terminated trajectories
                motion_keys = all_dict["motion_keys"]
                terminated = all_dict["terminated"]
                success_lift = all_dict["success_lift"]
                progress = all_dict.get("progress", [1.0] * len(motion_keys))
                obj_pos_errors = all_dict.get("obj_pos_error", [0.0] * len(motion_keys))

                pairs = []
                for i in range(len(motion_keys)):
                    term = bool(terminated[i]) if i < len(terminated) else False
                    lifted = bool(success_lift[i]) if i < len(success_lift) else False
                    prog = progress[i] if i < len(progress) else 1.0
                    obj_err = obj_pos_errors[i] if i < len(obj_pos_errors) else 0.0
                    priority = 0 if not lifted else (1 if term else 2)
                    pairs.append((motion_keys[i], term, lifted, prog, obj_err, priority))

                pairs_sorted = sorted(pairs, key=lambda x: (x[5], x[3]))
                if len(pairs_sorted) > config.num_envs:
                    pairs_sorted = pairs_sorted[: config.num_envs]

                render_info = []
                for pair in pairs_sorted:
                    motion_key, term, lifted, prog, obj_err, _ = pair
                    status = "FAILED" if not lifted else ("TERMINATED" if term else "SUCCESS")
                    info = [
                        f"{motion_key}",
                        f"lifted: {lifted}",
                        f"progress: {prog:.3f}",
                        f"status: {status}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {obj_err:.4f}m")
                    render_info.append(tuple(info))

                filter_keys = [pair[0] for pair in pairs_sorted]

                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(render_info)
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys
            else:
                # Imitation evaluation: use MPJPE-based sorting
                obj_pos_errors = all_dict.get("obj_pos_error", None)
                success_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        True,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if not all_dict["terminated"][i]
                ]
                render_sort_by = config.get("render_sort_by", "mpjpe_l")
                sort_idx = 4 if render_sort_by == "obj_pos_error" else 1
                success_pair_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
                failed_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        False,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if all_dict["terminated"][i]
                ]
                failed_pair_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)
                all_pair = failed_pair_sorted + success_pair_sorted
                if len(all_pair) > config.num_envs:
                    all_pair = all_pair[: config.num_envs]
                render_info = []
                for pair in all_pair:
                    info = [
                        f"{pair[0]}",
                        f"mpjpe_l: {pair[1]:.2f}",
                        f"mpjpe_g: {pair[2]:.2f}",
                        f"success: {pair[3]}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {pair[4]:.4f}m")
                    render_info.append(tuple(info))
                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(all_pair)
                filter_keys = [pair[0] for pair in all_pair]
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys

    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy_backbone_kwargs = {}
    critic_backbone_kwargs = {}
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = (
                obs_utils.get_group_term_obs_shape(example_obs, key)
            )
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)

    if not getattr(config.algo.config, "distill_only", False):
        value_model = trl_utils_common.custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=critic_backbone_kwargs,
            _resolve=False,
        ).to(device)

    accelerator.wait_for_everyone()

    args = easydict.EasyDict()
    args.is_main_process = accelerator.is_main_process
    args.global_rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    state = easydict.EasyDict()

    from gear_sonic.trl.trainer import ppo_trainer

    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)

    checkpoint_path = str(config.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)

    # Load policy state dict with backward compatibility for std/log_std
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = None

    if state_dict is not None:
        model_uses_std = "std" in model.policy.state_dict()
        checkpoint_has_std = "std" in state_dict
        checkpoint_has_log_std = "log_std" in state_dict

        logger.info(f"Model parameterization: {'std' if model_uses_std else 'log_std'}")
        logger.info(
            f"Checkpoint parameterization: {'std' if checkpoint_has_std else 'log_std' if checkpoint_has_log_std else 'unknown'}"  # noqa: E501
        )

        if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
            logger.info("Transforming 'log_std' -> 'std' (applying exp) for backward compatibility")
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
            logger.info("Transforming 'std' -> 'log_std' (applying log) for backward compatibility")
            state_dict["log_std"] = torch.log(state_dict.pop("std"))

        model.policy.load_state_dict(state_dict)
        logger.info("Successfully loaded policy state dict")

    if "state" in checkpoint and hasattr(checkpoint["state"], "global_step"):
        state.global_step = checkpoint["state"].global_step
    elif "dagger_state" in checkpoint and isinstance(checkpoint["dagger_state"], dict):
        # DAgger checkpoints carry a minimal state payload {"dagger_state": {"step": N}}
        # instead of the PPO trainer's full state object. Reuse that step for
        # export naming and any downstream schedule bookkeeping.
        state.global_step = int(checkpoint["dagger_state"].get("step", 0))
        logger.info(
            f"Checkpoint has no trainer state; using dagger_state.step={state.global_step} as global_step"
        )
    else:
        raise KeyError(
            "checkpoint missing both state.global_step and dagger_state.step; cannot derive export step"
        )

    schedule_wrapper = easydict.EasyDict(env=env, model=model)
    if "schedule_dict" in config.trainer:
        scheduled_params_dict = scheduler.update_scheduled_params(  # noqa: F841
            schedule_wrapper, config.trainer.schedule_dict, state.global_step
        )
    env.reinit_dr()

    global_step = state.global_step
    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_step_{global_step:06d}.onnx"
    new_cp_path = f"{os.path.dirname(config.checkpoint)}/model_step_{global_step:06d}.pt"
    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if config.get("export_onnx_only", False):

        def get_example_obs():
            obs_dict = env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].cpu()
            return obs_dict

        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gear_sonic.utils import inference_helpers

        example_obs_dict = get_example_obs()

        # Check if actor has universal-token encoder structure
        has_actor_module = hasattr(model.policy, "actor_module")
        has_encoders = has_actor_module and hasattr(
            model.policy.actor_module, "encoders_to_iterate"
        )

        if "tokenizer" in example_obs_dict and has_encoders:
            actor_module = model.policy.actor_module
            encoder_input_features = getattr(actor_module, "encoder_input_features", {})
            decoder_input_features = getattr(actor_module, "decoder_input_features", {})
            encoder_names = [
                name
                for name in getattr(actor_module, "encoders_to_iterate", [])
                if name in encoder_input_features
            ]
            # Preserve the previous export order for full models, then append any custom encoders.
            preferred_encoder_order = ["smpl", "g1", "teleop"]
            ordered_encoder_names = [
                name for name in preferred_encoder_order if name in encoder_names
            ] + [name for name in encoder_names if name not in preferred_encoder_order]

            decoder_name = "g1_dyn"
            if decoder_name not in decoder_input_features:
                logger.warning(
                    f"Skipping universal-token ONNX export: decoder '{decoder_name}' is absent. "
                    f"Available decoders: {list(decoder_input_features)}"
                )
            for encoder_name in ordered_encoder_names:
                if decoder_name not in decoder_input_features:
                    break
                inference_helpers.export_universal_token_module_as_onnx(
                    actor_module,
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    path=exported_policy_path,
                    exported_model_name=exported_onnx_name.replace(
                        ".onnx", f"_{encoder_name}.onnx"
                    ),
                    batch_size=1,
                )

            inference_helpers.export_universal_token_encoders_as_onnx(
                actor_module,
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_encoder.onnx"),
                batch_size=1,
            )
            if decoder_name in decoder_input_features:
                inference_helpers.export_universal_token_decoder_as_onnx(
                    actor_module,
                    decoder_name=decoder_name,
                    path=exported_policy_path,
                    exported_model_name=exported_onnx_name.replace(".onnx", "_decoder.onnx"),
                    batch_size=1,
                )
            print(  # noqa: T201
                f'Exported encoders ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_encoder.onnx"))}'  # noqa: E501
            )
            print(  # noqa: T201
                f'Exported decoder ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_decoder.onnx"))}'  # noqa: E501
            )

        else:
            inference_helpers.export_policy_as_onnx(
                {"actor": model.policy}, exported_policy_path, exported_onnx_name, example_obs_dict
            )

        logger.info(f"Exported policy as onnx to: {os.path.join(exported_policy_path)}")

        # Export configs to YAML
        export_config = {
            "env_config": omegaconf.OmegaConf.to_container(env.config, resolve=True),
            "algo_config": omegaconf.OmegaConf.to_container(config.algo.config, resolve=True),
        }
        config_yaml_path = os.path.join(os.path.dirname(config.checkpoint), "model_config.yaml")
        with open(config_yaml_path, "w") as f:
            yaml.dump(export_config, f, default_flow_style=False)
        logger.info(f"Exported config to: {config_yaml_path}")
        exit()  # noqa: PLR1722

    eval_callbacks = config.get("eval_callbacks", [])
    if isinstance(eval_callbacks, str):
        eval_callbacks = [eval_callbacks]

    callbacks = {}
    for callback_name in eval_callbacks:
        if callback_name == "im_eval":
            with omegaconf.open_dict(config.callbacks.im_eval):
                config.callbacks.im_eval.eval_only = True
                config.callbacks.im_eval.eval_frequency = 1
                config.callbacks.im_eval.output_dir = config.get("eval_output_dir", None)
                config.callbacks.im_eval.log_keys = config.get("log_keys", None)
                config.callbacks.im_eval.max_render_steps = config.get("max_render_steps", 0)
                config.callbacks.im_eval.reward_curve_enabled = config.get(
                    "reward_curve_enabled", False
                )
                config.callbacks.im_eval.reward_curve_output_dir = config.get(
                    "reward_curve_output_dir", None
                )
                config.callbacks.im_eval.foot_force_dump_dir = config.get(
                    "foot_force_dump_dir", None
                )
        if callback_name not in config.callbacks:
            raise ValueError(f"Callback {callback_name} not found")
        callbacks[callback_name] = utils.instantiate(config.callbacks[callback_name])

    for callback_name, callback in callbacks.items():  # noqa: B007
        if hasattr(callback, "model") and callback.model is None:
            callback.model = model

    for callback_name, callback in callbacks.items():  # noqa: B007
        callback.on_step_end(args, state, None, env=env, model=model, accelerator=accelerator)

    if config.get("run_eval_loop", True):
        env.set_is_evaluating(True)
        obs_dict = env.reset_all()
        model.eval()
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        eval_step_callbacks = {
            name: cb
            for name, cb in callbacks.items()
            if hasattr(cb, "eval_step") and callable(getattr(cb, "eval_step"))  # noqa: B009
        }
        if eval_step_callbacks:
            logger.info(f"Eval step callbacks enabled: {list(eval_step_callbacks.keys())}")

        step_count = 0
        max_render_steps = config.get("max_render_steps", 0)
        run_once = config.get("run_once", False)
        envs_completed = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
        ckpt_path = Path(config.checkpoint)
        debug_dump_path = config.get("eval_debug_dump_path", None)
        write_debug_dump = debug_dump_path is not None or config.get("eval_debug_dump", False)
        make_debug_plots = config.get("eval_debug_plot", True)
        if write_debug_dump and debug_dump_path is None:
            debug_dump_path = ckpt_path.parent / f"{ckpt_path.stem}_eval_debug.npz"
        debug_dumper = None
        if write_debug_dump or make_debug_plots:
            debug_plot_dir = config.get("eval_debug_plot_dir", None)
            if debug_plot_dir is None:
                debug_plot_dir = ckpt_path.parent / f"{ckpt_path.stem}_eval_debug_plots"
            debug_dumper = EvalDebugDumper(
                debug_dump_path,
                env_idx=config.get("eval_debug_dump_env_idx", 0),
                max_steps=config.get("eval_debug_dump_max_steps", 0),
                plot_dir=debug_plot_dir,
                make_plots=make_debug_plots,
            )
            logger.info(
                "Eval debug collection enabled: "
                f"dump_path={debug_dumper.output_path}, env_idx={debug_dumper.env_idx}, "
                f"max_steps={debug_dumper.max_steps}, plot_dir={debug_dumper.plot_dir}, "
                f"make_plots={debug_dumper.make_plots}"
            )

        try:
            with torch.no_grad():
                while True:
                    policy_model = model.policy
                    value_model = model.value_model
                    policy_model.init_rollout()

                    actor_state = {}
                    actions = policy_model.rollout(obs_dict=obs_dict)
                    actor_state["actions"] = policy_model.action_mean.detach()
                    actor_state["obs_dict"] = actions["obs_dict"]

                    step_count += 1

                    if max_render_steps > 0 and step_count >= max_render_steps:
                        logger.info(f"Reached max_render_steps={max_render_steps}. Exiting.")
                        if hasattr(env, "end_render_results"):
                            env.end_render_results()
                        break

                    results = env.step(actor_state)
                    obs_dict, rewards, dones, infos = (
                        results[0],
                        results[1],
                        results[2],
                        results[3],
                    )  # noqa: F841
                    if debug_dumper is not None:
                        debug_dumper.capture(step_count, env, actor_state, dones)

                    if eval_step_callbacks:
                        all_want_exit = all(
                            cb.eval_step(env, results) for cb in eval_step_callbacks.values()
                        )
                        if all_want_exit:
                            logger.info("All eval step callbacks signaled exit. Exiting evaluation loop.")
                            break

                    if run_once:
                        envs_completed = (
                            envs_completed | dones.squeeze(-1)
                            if dones.dim() > 1
                            else envs_completed | dones
                        )
                        if envs_completed.all():
                            logger.info("All environments completed one episode. Exiting (run_once=True).")
                            if hasattr(env, "end_render_results"):
                                env.end_render_results()
                            break

                    for obs_key in obs_dict.keys():  # noqa: SIM118
                        obs_dict[obs_key] = obs_dict[obs_key].to(device)
        finally:
            if debug_dumper is not None:
                debug_dumper.save()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":
    main()
