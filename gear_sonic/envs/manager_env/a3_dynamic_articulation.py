"""Opt-in, per-physics-step dynamic ankle inertia for existing A3 drives."""

from __future__ import annotations

import copy
import dataclasses
import re

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass
import torch

from gear_sonic.utils.a3_dynamic_armature import A3_ANKLE_MOTOR_ARMATURE, A3DynamicArmature


class A3DynamicArticulation(Articulation):
    """Keeps the selected PD/motor model, adds fictitious inertia forces afterwards.

    Joint effort telemetry remains motor/drive effort, not fictitious inertia.
    PhysX drive limits remain unchanged. No clipping is applied to compensation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        names = [f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll")]
        self.dynamic_joint_ids, found = self.find_joints(names, preserve_order=True)
        if found != names:
            raise ValueError(f"Dynamic ankles require {names}, got {found}")
        self.dynamic_model = A3DynamicArmature(
            device=self.device,
            dtype=self.data.joint_pos.dtype,
            table_dir=self.cfg.dynamic_ankle_table_dir,
            motor_armature=self.cfg.dynamic_ankle_motor_armature,
        )
        self.dynamic_compensation = torch.zeros((self.num_instances, 4), device=self.device)
        self.dynamic_diagonal = self.dynamic_compensation.clone()
        self._dynamic_valid = torch.zeros((self.num_instances, 1), dtype=torch.bool, device=self.device)
        self._dynamic_write_seen = False
        self.dynamic_mode = self.cfg.dynamic_ankle_mode
        zero = torch.zeros((1, 4), device=self.device, dtype=self.data.joint_pos.dtype)
        self.motor_inertia_scales = torch.ones((self.num_instances, 4), device=self.device)
        self.nominal_diagonal, _, self.nominal_offdiag = self.dynamic_model.terms(
            zero.expand(self.num_instances, -1), zero.expand(self.num_instances, -1), self.motor_inertia_scales
        )
        self._install_nominal_armature()

    def _install_nominal_armature(self, env_ids=None):
        if self.dynamic_mode == "nominal":
            self.write_joint_armature_to_sim(
                self.nominal_diagonal if env_ids is None else self.nominal_diagonal[env_ids],
                joint_ids=self.dynamic_joint_ids,
                env_ids=env_ids,
            )

    def randomize_motor_inertia(self, env_ids, scale_range):
        """Independent output-shaft rotor inertia factors, ordered L1,L2,R1,R2.

        Fixed D0/O0 and live h use the SAME factors; factors persist between
        event invocations. Never randomize projected pitch/roll independently.
        """
        if self.dynamic_mode != "nominal":
            raise ValueError("Motor inertia DR currently requires nominal mode")
        lo, hi = scale_range
        if not 0 < lo <= hi:
            raise ValueError("Invalid motor inertia scale range")
        if env_ids is None:
            env_ids = torch.arange(self.num_instances, device=self.device)
        count = len(env_ids)
        scales = torch.empty((count, 4), device=self.device).uniform_(lo, hi)
        self.motor_inertia_scales[env_ids] = scales
        zero = torch.zeros_like(scales)
        diagonal, _, offdiag = self.dynamic_model.terms(zero, zero, scales)
        self.nominal_diagonal[env_ids] = diagonal
        self.nominal_offdiag[env_ids] = offdiag
        self._install_nominal_armature(env_ids)
        self._dynamic_valid[env_ids] = False
        if count and not getattr(self, "_motor_dr_logged", False):
            print(
                f"[ANKLE_MOTOR_DR] motors=4 envs={count} range={tuple(scale_range)} "
                f"sampled=[{float(scales.min()):.4f}, {float(scales.max()):.4f}] "
                "D0/O0/h share rotor factors; reset cadence controlled by event"
            )
            self._motor_dr_logged = True

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.cfg.shared_action_delay:
            ids = slice(None) if env_ids is None else env_ids
            count = self.num_instances if isinstance(ids, slice) else len(ids)
            actuators = list(self.actuators.values())
            lo, hi = actuators[0].cfg.min_delay, actuators[0].cfg.max_delay
            lags = torch.randint(lo, hi + 1, (count,), device=self.device, dtype=torch.int)
            for actuator in actuators:
                assert (actuator.cfg.min_delay, actuator.cfg.max_delay) == (lo, hi)
                for field in ('positions_delay_buffer', 'velocities_delay_buffer', 'efforts_delay_buffer'):
                    getattr(actuator, field).set_time_lag(lags, env_ids)
        if hasattr(self, "_dynamic_valid"):
            ids = slice(None) if env_ids is None else env_ids
            self._dynamic_valid[ids] = False
            self.dynamic_compensation[ids] = 0

    def update(self, dt):
        super().update(dt)
        # Only mark acceleration valid after a completed physical step, not after
        # repeated scene.write_data_to_sim() calls during an environment reset.
        if getattr(self, "_dynamic_write_seen", False):
            self._dynamic_valid[:] = True
            self._dynamic_write_seen = False

    def _apply_actuator_model(self):
        super()._apply_actuator_model()
        # Component-only modes are timing ablations.
        if self.dynamic_mode == "baseline":
            return
        ids = self.dynamic_joint_ids
        q, v = self.data.joint_pos[:, ids], self.data.joint_vel[:, ids]
        diagonal, h, offdiag = self.dynamic_model.terms(q, v, self.motor_inertia_scales)
        if self.dynamic_mode == "nominal":
            diagonal, offdiag = self.nominal_diagonal, self.nominal_offdiag
        self.dynamic_diagonal.copy_(diagonal)
        acc = torch.where(self._dynamic_valid, self.data.joint_acc[:, ids], 0.0)
        self.dynamic_compensation.copy_(self.dynamic_model.correction(h, offdiag, acc))
        if self.dynamic_mode in ("dynamic", "armature_only"):
            # IsaacLab currently requires a CPU transfer inside this setter.
            self.write_joint_armature_to_sim(diagonal, joint_ids=ids)
        if self.dynamic_mode in ("dynamic", "nominal", "correction_only"):
            self._joint_effort_target_sim[:, ids] += self.dynamic_compensation
        self._dynamic_write_seen = True


@configclass
class A3DynamicArticulationCfg(ArticulationCfg):
    class_type: type = A3DynamicArticulation
    dynamic_ankle_mode: str = "dynamic"
    dynamic_ankle_table_dir: str | None = None
    dynamic_ankle_motor_armature: float = A3_ANKLE_MOTOR_ARMATURE
    shared_action_delay: bool = False


def with_dynamic_ankles(robot_cfg: ArticulationCfg, options: dict) -> A3DynamicArticulationCfg:
    """Copy the asset config, retaining all actuator gains, delays, and limits."""
    mode = options.get("mode", "dynamic")
    if mode == "simplest":
        # Set the actuator's default armature so existing joint-side DR scales
        # the new nominal value. Keep the ordinary Articulation and hot path.
        cfg = copy.deepcopy(robot_cfg)
        model = A3DynamicArmature(
            device="cpu",
            table_dir=options.get("jacobian_table_dir"),
            motor_armature=float(options.get("motor_armature", A3_ANKLE_MOTOR_ARMATURE)),
        )
        zero = torch.zeros((1, 4))
        diagonal, _, _ = model.terms(zero, zero)
        names = [f"{side}_ankle_{axis}_joint" for side in ("left", "right") for axis in ("pitch", "roll")]
        changed = set()
        for actuator in cfg.actuators.values():
            if not isinstance(actuator.armature, dict):
                continue
            for pattern in list(actuator.armature):
                matches = [i for i, name in enumerate(names) if re.fullmatch(pattern, name)]
                if matches:
                    values = diagonal[0, matches]
                    if not torch.allclose(values, values[0].expand_as(values)):
                        raise ValueError("Armature pattern mixes unequal nominal axes")
                    actuator.armature[pattern] = float(values[0])
                    changed.update(matches)
        if len(changed) != 4:
            raise ValueError("Expected all four ankle armatures in actuator config")
        print(
            f"[ANKLE_INERTIA] mode=simplest zero_pose_diagonal={diagonal[0].tolist()} "
            "ordinary Articulation; existing joint-side DR retained"
        )
        return cfg
    if mode not in ("dynamic", "nominal"):
        raise ValueError(f"Unknown dynamic_ankle mode: {mode}")
    values = {
        f.name: copy.deepcopy(getattr(robot_cfg, f.name))
        for f in dataclasses.fields(robot_cfg)
        if f.init and f.name != "class_type"
    }
    return A3DynamicArticulationCfg(
        **values,
        dynamic_ankle_mode=mode,
        dynamic_ankle_table_dir=options.get("jacobian_table_dir"),
        dynamic_ankle_motor_armature=float(options.get("motor_armature", A3_ANKLE_MOTOR_ARMATURE)),
        shared_action_delay=bool(options.get("shared_action_delay", False)),
    )


def randomize_a3_ankle_motor_inertia(env, env_ids, scale_range=(0.7, 2.0)):
    """Reset event; IsaacLab controls resampling cadence via EventTermCfg."""
    env.scene["robot"].randomize_motor_inertia(env_ids, scale_range)
