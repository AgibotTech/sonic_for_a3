"""Reward functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)
import torch

from gear_sonic.envs.manager_env.mdp.commands import (
    ForceTrackingCommand,
    TrackingCommand,
    _get_body_indexes,
)
from gear_sonic.utils.a3_motor_model import (
    A3MotorConstraintCache,
    compute_cross_penalty,
    compute_motor_penalty,
    compute_position_threshold_penalty,
    compute_threshold_penalty,
    get_tn_curve,
    peak_mechanical_power,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    # Optional focused leg-pose term.  Course configs can enable this in
    # addition to the whole-body term without changing its behaviour.
    tracking_relative_body_pos_legs = None
    tracking_relative_body_ori = None
    tracking_relative_body_ori_weighted = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    undesired_contacts_no_hands = None
    undesired_contacts_no_ankle_hand = None
    tracking_body_pos = None
    tracking_body_ori = None
    tracking_vr_3point_global = None
    tracking_vr_3point_local = None
    tracking_vr_3point_force = None
    tracking_vr_2wrists_ori_tight = None
    tracking_vr_2wrists_local_ori = None
    tracking_head_local_ori = None
    anti_shake_ang_vel = None
    tracking_vr_5point_local = None
    motion_5point_local_pos = None
    feet_acc = None
    foot_end_load = None
    is_terminated = None
    upright_penalty = None
    motor_tn_serial = None
    motor_tn_parallel = None
    motor_tn_serial_soft_diag = None
    motor_tn_serial_hard_diag = None
    motor_tn_parallel_soft_diag = None
    motor_tn_parallel_hard_diag = None
    ankle_motor_speed = None
    ankle_motor_power = None
    ankle_motor_fixed_torque = None
    parallel_ankle_cross = None
    parallel_waist_cross = None
    ankle_position_85 = None
    ankle_motor_speed_diag = None
    ankle_motor_power_diag = None
    ankle_motor_fixed_torque_diag = None
    parallel_ankle_cross_diag = None
    parallel_waist_cross_diag = None
    ankle_position_85_diag = None


_A3_PARALLEL_PAIRS = (
    (
        "left_ankle",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    (
        "right_ankle",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
    (
        "waist",
        "waist_pitch_joint",
        "waist_roll_joint",
    ),
)


def _get_a3_motor_constraint_cache(
    env: ManagerBasedRLEnv,
    robot,
    *,
    table_dir: str | None,
) -> A3MotorConstraintCache:
    caches = getattr(env, "_a3_motor_constraint_caches", None)
    if caches is None:
        caches = {}
        setattr(env, "_a3_motor_constraint_caches", caches)
    key = (id(robot), str(table_dir or ""))
    cache = caches.get(key)
    if cache is None:
        cache = A3MotorConstraintCache(
            robot.joint_names,
            device=env.device,
            dtype=robot.data.computed_torque.dtype,
            table_dir=table_dir,
        )
        caches[key] = cache
    return cache


def _prepare_a3_motor_constraint_cache(
    cache: A3MotorConstraintCache,
    env: ManagerBasedRLEnv,
    robot,
) -> A3MotorConstraintCache:
    return cache.begin_step(
        env.common_step_counter,
        joint_position=robot.data.joint_pos,
        joint_velocity=robot.data.joint_vel,
        joint_torque=robot.data.computed_torque,
    )


class A3MotorTNPenalty(ManagerTermBase):
    """Penalize requested implicit-PD torque outside the physical motor T-N envelope.

    The simulator remains unchanged: this term reads ``computed_torque`` before
    the implicit actuator's effort clipping and returns only a reward penalty.
    Direct-drive joints are grouped by their physical motor family.  The A3
    ankle and waist pitch/roll pairs are first converted through the same
    position-dependent Jacobian convention as the HAL loop solver.
    """

    _COMPONENTS = {"soft", "hard", "combined"}
    _GROUPS = {"serial", "parallel"}

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._group = str(params["group"])
        self._component = str(params.get("component", "combined"))
        if self._group not in self._GROUPS:
            raise ValueError(f"A3 motor T-N group must be one of {sorted(self._GROUPS)}, got {self._group!r}")
        if self._component not in self._COMPONENTS:
            raise ValueError(
                f"A3 motor T-N component must be one of {sorted(self._COMPONENTS)}, got {self._component!r}"
            )

        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self._robot = env.scene[asset_cfg.name]
        table_dir = params.get("jacobian_table_dir")
        self._motor_cache = _get_a3_motor_constraint_cache(
            env,
            self._robot,
            table_dir=table_dir,
        )

    def _serial_utilization(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        cache = _prepare_a3_motor_constraint_cache(
            self._motor_cache,
            env,
            self._robot,
        )
        return cache.serial_utilization()

    def _parallel_utilization(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        cache = _prepare_a3_motor_constraint_cache(
            self._motor_cache,
            env,
            self._robot,
        )
        return cache.parallel_utilization()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        group: str = "serial",
        component: str = "combined",
        lambda_soft: float = 0.05,
        lambda_violation: float = 5.0,
        soft_exponent: float = 8.0,
        violation_exponent: float = 2.0,
        utilization_cap: float = 2.0,
        jacobian_table_dir: str | None = None,
    ) -> torch.Tensor:
        del asset_cfg, jacobian_table_dir
        if group != self._group or component != self._component:
            raise ValueError(
                "A3MotorTNPenalty runtime parameters differ from initialized "
                f"parameters: {(group, component)} != "
                f"{(self._group, self._component)}"
            )
        utilization = self._serial_utilization(env) if group == "serial" else self._parallel_utilization(env)
        penalty = compute_motor_penalty(
            utilization,
            lambda_soft=lambda_soft,
            lambda_violation=lambda_violation,
            soft_exponent=soft_exponent,
            violation_exponent=violation_exponent,
            utilization_cap=utilization_cap,
        )
        return getattr(penalty, component).sum(dim=-1)


class A3AnkleMotorConstraintPenalty(ManagerTermBase):
    """Apply speed, positive-power, or fixed-torque limits in ankle motor coordinates."""

    _CONSTRAINTS = {"speed", "power", "fixed_torque"}

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._constraint = str(params["constraint"])
        if self._constraint not in self._CONSTRAINTS:
            raise ValueError(
                f"A3 ankle motor constraint must be one of {sorted(self._CONSTRAINTS)}, got {self._constraint!r}"
            )

        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self._robot = env.scene[asset_cfg.name]
        dtype = self._robot.data.computed_torque.dtype
        table_dir = params.get("jacobian_table_dir")
        self._motor_cache = _get_a3_motor_constraint_cache(
            env,
            self._robot,
            table_dir=table_dir,
        )
        self._curve = get_tn_curve(
            "PFP78",
            device=env.device,
            dtype=dtype,
            profile="reward_024",
        )

    def _motor_ratio(
        self,
        env: ManagerBasedRLEnv,
        *,
        motor_speed_limit_rad_s: float | None,
        motor_power_limit_w: float | None,
        motor_torque_limit_nm: float | None,
    ) -> torch.Tensor:
        cache = _prepare_a3_motor_constraint_cache(
            self._motor_cache,
            env,
            self._robot,
        )
        ratios = []
        for table_name in cache.ankle_names:
            if self._constraint == "speed":
                limit = (
                    float(motor_speed_limit_rad_s)
                    if motor_speed_limit_rad_s is not None
                    else float(self._curve.speed_rad_s[-1])
                )
                value = torch.abs(cache.parallel_motor_velocity(table_name))
            elif self._constraint == "power":
                limit = (
                    float(motor_power_limit_w)
                    if motor_power_limit_w is not None
                    else float(peak_mechanical_power(self._curve))
                )
                value = torch.relu(
                    cache.parallel_motor_torque(table_name) * cache.parallel_motor_velocity(table_name)
                )
            else:
                limit = (
                    float(motor_torque_limit_nm)
                    if motor_torque_limit_nm is not None
                    else float(self._curve.torque_nm[0])
                )
                value = torch.abs(cache.parallel_motor_torque(table_name))
            if limit <= 0.0:
                raise ValueError(f"A3 ankle motor {self._constraint} limit must be positive")
            ratios.append(value / limit)
        return torch.cat(ratios, dim=-1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        constraint: str = "speed",
        threshold_ratio: float = 0.8,
        lambda_soft: float = 1.0,
        lambda_violation: float = 5.0,
        soft_exponent: float = 2.0,
        violation_exponent: float = 2.0,
        utilization_cap: float = 2.0,
        motor_speed_limit_rad_s: float | None = None,
        motor_power_limit_w: float | None = None,
        motor_torque_limit_nm: float | None = None,
        jacobian_table_dir: str | None = None,
    ) -> torch.Tensor:
        del asset_cfg, jacobian_table_dir
        if constraint != self._constraint:
            raise ValueError(
                "A3AnkleMotorConstraintPenalty runtime constraint differs from initialized "
                f"constraint: {constraint!r} != {self._constraint!r}"
            )
        ratio = self._motor_ratio(
            env,
            motor_speed_limit_rad_s=motor_speed_limit_rad_s,
            motor_power_limit_w=motor_power_limit_w,
            motor_torque_limit_nm=motor_torque_limit_nm,
        )
        penalty = compute_threshold_penalty(
            ratio,
            threshold_ratio=threshold_ratio,
            lambda_soft=lambda_soft,
            lambda_violation=lambda_violation,
            soft_exponent=soft_exponent,
            violation_exponent=violation_exponent,
            utilization_cap=utilization_cap,
        )
        return penalty.combined.sum(dim=-1)


class A3ParallelJointCrossPenalty(ManagerTermBase):
    """Penalize simultaneous high pitch/roll requested torque as a shaping term."""

    _GROUP_PAIRS = {
        "ankle": _A3_PARALLEL_PAIRS[:2],
        "waist": _A3_PARALLEL_PAIRS[2:],
    }

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._group = str(params["group"])
        if self._group not in self._GROUP_PAIRS:
            raise ValueError(
                f"A3 parallel cross group must be one of {sorted(self._GROUP_PAIRS)}, got {self._group!r}"
            )
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self._robot = env.scene[asset_cfg.name]
        joint_index = {name: index for index, name in enumerate(self._robot.joint_names)}
        effort_limits = params["joint_effort_limits"]
        pair_indices = []
        pair_limits = []
        for _, pitch_name, roll_name in self._GROUP_PAIRS[self._group]:
            try:
                pair_indices.append((joint_index[pitch_name], joint_index[roll_name]))
                pair_limits.append(
                    (
                        float(effort_limits[pitch_name]),
                        float(effort_limits[roll_name]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"A3 parallel cross penalty is missing joint or effort limit {exc.args[0]!r}"
                ) from exc
        self._pair_indices = torch.tensor(pair_indices, device=env.device, dtype=torch.long)
        self._pair_limits = torch.tensor(
            pair_limits,
            device=env.device,
            dtype=self._robot.data.computed_torque.dtype,
        )
        if torch.any(self._pair_limits <= 0.0):
            raise ValueError("A3 parallel cross effort limits must be positive")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        group: str = "ankle",
        threshold_ratio: float = 0.75,
        joint_effort_limits: dict[str, float] | None = None,
    ) -> torch.Tensor:
        del env, asset_cfg, joint_effort_limits
        if group != self._group:
            raise ValueError(
                "A3ParallelJointCrossPenalty runtime group differs from initialized "
                f"group: {group!r} != {self._group!r}"
            )
        requested = torch.abs(self._robot.data.computed_torque[:, self._pair_indices])
        normalized = requested / self._pair_limits
        return compute_cross_penalty(
            normalized,
            threshold_ratio=threshold_ratio,
        ).sum(dim=-1)


class A3AnklePositionPenalty(ManagerTermBase):
    """Penalize ankle pitch/roll beyond a fraction of the physical hard range."""

    _JOINT_NAMES = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._robot = env.scene[asset_cfg.name]
        joint_index = {name: index for index, name in enumerate(self._robot.joint_names)}
        try:
            indices = [joint_index[name] for name in self._JOINT_NAMES]
        except KeyError as exc:
            raise ValueError(f"A3 ankle position penalty could not find joint {exc.args[0]!r}") from exc
        self._joint_indices = torch.tensor(indices, device=env.device, dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold_ratio: float = 0.85,
        exponent: float = 2.0,
    ) -> torch.Tensor:
        del env, asset_cfg
        return compute_position_threshold_penalty(
            self._robot.data.joint_pos[:, self._joint_indices],
            self._robot.data.joint_pos_limits[:, self._joint_indices, :],
            threshold_ratio=threshold_ratio,
            exponent=exponent,
        )


def tracking_anchor_pos_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Compute anchor position tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) position to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel. Smaller values produce
            sharper falloff and stricter tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    sq_dist = (diff * diff).sum(dim=-1)
    return torch.exp(-sq_dist / (std * std))


def tracking_anchor_ori_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Compute anchor orientation tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) orientation to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-angular_err.square() / (std * std))


def upright_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_name: str | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize tilt of bodies away from upright.

    Compute the squared magnitude of the x/y components of the gravity vector
    in each body's local frame, summed across all specified bodies. When a body
    is perfectly upright the local gravity is [0, 0, -1] and the penalty is 0.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        body_name: Single body name (for backwards compatibility).
        body_names: List of body names. If both are None, defaults to ["pelvis"].

    Returns:
        Penalty tensor of shape (num_envs,). Zero when upright, positive when tilted.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    if body_names is None:
        body_names = [body_name] if body_name else ["pelvis"]

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for name in body_names:
        body_idx = robot.body_names.index(name)
        body_quat = robot.data.body_quat_w[:, body_idx]
        g_local = quat_apply(quat_inv(body_quat), command.down_dir)
        total_penalty += g_local[:, 0] ** 2 + g_local[:, 1] ** 2

    return total_penalty


def tracking_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body positions to match the reference motion. The reward is
    the mean squared distance across all tracked bodies, passed through an exponential.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_vr_3point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 3-point tracking reward in world frame using a Gaussian kernel.

    Encourages the robot's 3 VR tracking points (typically left wrist, right wrist,
    head) to match their reference positions in world frame.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.robot_vr_3point_pos_w - command.vr_3point_body_pos_w
    per_point_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_point_err.mean(dim=-1) / (std * std))


def tracking_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in world frame.

    Measure the orientation error of 2 wrist bodies against the reference motion,
    similar to tracking_relative_body_ori_error but restricted to wrist links.

    NOTE: The rigid extension defined in vr_3point_body_offset can be skipped for
    orientation error since it does not affect rotations.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked])
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_local_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in the anchor's local frame.

    Transform both reference and robot wrist orientations into the anchor (root)
    frame before computing the angular error. This makes the reward invariant to
    global root orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    body_indexes = _get_body_indexes(command, body_names)
    num_bodies = len(body_indexes)

    # reference motion
    ref_wrist_quat_w = command.body_quat_w[:, body_indexes]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    ref_wrist_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_wrist_quat_w)

    # robot
    robot_wrist_quat_w = command.robot_body_quat_w[:, body_indexes]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    robot_wrist_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_wrist_quat_w)

    error = quat_error_magnitude(ref_wrist_quat_local, robot_wrist_quat_local) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def tracking_local_head_ori_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Compute head orientation tracking reward in the anchor's local frame.

    Transform the head (torso_link) orientation into the anchor's local frame for
    both the reference motion and the robot, then compute the angular error. This
    encourages the robot to match the head-to-root relative orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get the head body index (torso_link)
    head_body_names = ["torso_link"]
    body_indexes = _get_body_indexes(command, head_body_names)

    # reference motion: head orientation in world frame, transformed to anchor's local frame
    ref_head_quat_w = command.body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4)
    ref_head_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_head_quat_w)

    # robot: head orientation in world frame, transformed to anchor's local frame
    robot_head_quat_w = command.robot_body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4)
    robot_head_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_head_quat_w)

    error = quat_error_magnitude(ref_head_quat_local, robot_head_quat_local) ** 2
    return torch.exp(-error.squeeze(-1) / std**2)


def tracking_local_vr_3point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    point_weights: list[float] | None = None,
):
    """Compute VR 3-point tracking reward in the anchor's local frame.

    Transform tracking points into the anchor (root) local frame before computing
    position error, making the reward invariant to global root position/orientation.
    Supports optional per-point weighting.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        point_weights: Optional weights for each tracking point. Order matches
            vr_3point_body config (typically [left_wrist, right_wrist, head]).
            If None, all points are weighted equally.
            Example: [2, 2, 1] gives wrists 2x importance vs head.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).expand(
        -1, len(command.cfg.vr_3point_body), -1
    )
    ref_3point_pos = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).expand(
        -1, len(command.cfg.vr_3point_body), -1
    )
    robot_3point_diff = command.robot_vr_3point_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_3point_pos = quat_apply(quat_inv(robot_root_quat), robot_3point_diff)
    diff = robot_3point_pos - ref_3point_pos
    error = torch.sum(torch.square(diff), dim=-1)  # [num_envs, num_points]

    if point_weights is not None:
        # Weighted mean: sum(w_i * e_i) / sum(w_i)
        weights = torch.tensor(point_weights, dtype=error.dtype, device=error.device)
        weighted_error = (error * weights).sum(dim=-1) / weights.sum()
    else:
        # Simple mean (equal weights)
        weighted_error = error.mean(dim=-1)

    return torch.exp(-weighted_error / std**2)


def tracking_local_vr_5point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 5-point tracking reward in the anchor's local frame.

    Same approach as tracking_local_vr_3point_error but with 5 tracking points
    (e.g., 2 wrists + head + 2 feet).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_5point_diff = command.reward_point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).expand(
        -1, len(command.cfg.reward_point_body), -1
    )
    ref_5point_pos = quat_apply(quat_inv(ref_root_quat), ref_5point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).expand(
        -1, len(command.cfg.reward_point_body), -1
    )
    robot_5point_diff = command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_5point_pos = quat_apply(quat_inv(robot_root_quat), robot_5point_diff)
    diff = robot_5point_pos - ref_5point_pos
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def _fz_history(sensor, body_ids) -> torch.Tensor:
    """Return clamped world-Z contact force history for selected contact bodies."""
    forces = getattr(sensor.data, "net_forces_w_history", None)
    if forces is None or forces.numel() == 0:
        forces = sensor.data.net_forces_w.unsqueeze(1)
    return forces[:, :, body_ids, 2].clamp_min(0.0)


def persistent_foot_end_load_penalty(
    env: ManagerBasedRLEnv,
    front_cfg: SceneEntityCfg,
    mid_cfg: SceneEntityCfg,
    rear_cfg: SceneEntityCfg,
    front_ratio_limit: float = 0.30,
    front_force_limit: float = 0.22,
    min_foot_load_ratio: float = 0.07,
    persistence_ratio: float = 0.45,
    abs_force_weight: float = 0.25,
    robot_mass: float = 57.98,
) -> torch.Tensor:
    """Penalize persistent toe/front overload on segmented hard-sole feet.

    The configured body order must be [left, right] for every segment. This
    function returns a positive penalty; reward configs should use a negative
    weight.
    """
    front_sensor = env.scene.sensors[front_cfg.name]
    mid_sensor = env.scene.sensors[mid_cfg.name]
    rear_sensor = env.scene.sensors[rear_cfg.name]

    front_fz = _fz_history(front_sensor, front_cfg.body_ids)
    mid_fz = _fz_history(mid_sensor, mid_cfg.body_ids)
    rear_fz = _fz_history(rear_sensor, rear_cfg.body_ids)

    foot_load = front_fz + mid_fz + rear_fz
    body_weight = float(robot_mass) * 9.81
    loaded = foot_load > (float(min_foot_load_ratio) * body_weight)
    foot_load_safe = foot_load.clamp_min(1.0e-6)

    front_ratio = torch.where(loaded, front_fz / foot_load_safe, torch.zeros_like(front_fz))

    front_ratio_excess = torch.relu(front_ratio - float(front_ratio_limit))
    front_force_excess = torch.relu(front_fz / body_weight - float(front_force_limit))

    instant_penalty = front_ratio_excess + float(abs_force_weight) * front_force_excess
    instant_penalty = torch.where(loaded, instant_penalty, torch.zeros_like(instant_penalty))

    overloaded = loaded & ((front_ratio_excess > 0.0) | (front_force_excess > 0.0))
    persistent = overloaded.float().mean(dim=1) >= float(persistence_ratio)
    penalty_per_foot = instant_penalty.mean(dim=1) * persistent.float()
    return penalty_per_foot.sum(dim=-1)


def tracking_vr_3point_error_pos_force(
    env: ManagerBasedRLEnv, motion_command_name: str, force_command_name: str, std: float
):
    """Compute VR 3-point tracking reward with force-based compliance correction.

    Add a force-proportional offset to the wrist tracking error so that applied
    external forces shift the tracking target, enabling compliant behavior under
    force perturbations.

    Args:
        env: The environment.
        motion_command_name: Name of the motion tracking command term.
        force_command_name: Name of the force tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    motion_command: TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    diff = motion_command.robot_vr_3point_pos_w - motion_command.vr_3point_body_pos_w
    force_error_wrists = force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    diff[:, :2] += force_error_wrists
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body orientations to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked])
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward using anchor-relative reference positions.

    Use reference body positions that have been shifted to share the robot's anchor
    (root) position, so only the relative pose matters rather than absolute position.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward using anchor-relative reference orientations.

    Use reference body orientations that have been transformed to share the robot's
    anchor (root) orientation, making the reward invariant to global heading.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_relative_w[:, tracked],
        command.robot_body_quat_w[:, tracked],
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_ori_weighted_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute anchor-relative body orientation tracking reward with per-body weights.

    Same as tracking_relative_body_ori_error but allows different bodies to contribute
    differently to the mean error. Useful for relaxing tracking on certain joints
    (e.g., wrists during manipulation).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.
        body_weights: Dict mapping body name to weight multiplier. Bodies not listed
            default to 1.0. E.g. {"left_wrist_yaw_link": 0.1} to relax wrist tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    if body_weights is not None:
        tracked_names = [command.cfg.body_names[i] for i in body_indexes]
        weights = torch.tensor(
            [body_weights.get(name, 1.0) for name in tracked_names],
            device=error.device,
            dtype=error.dtype,
        )
        weighted_error = (error * weights).sum(-1) / weights.sum()
    else:
        weighted_error = error.mean(-1)
    return torch.exp(-weighted_error / std**2)


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body linear velocity tracking reward using a Gaussian kernel.

    Encourages tracked body linear velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body angular velocity tracking reward using a Gaussian kernel.

    Encourages tracked body angular velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize excessive angular velocity on selected bodies with a deadzone.

    Discourage high-frequency jitter on small links (wrists, head) while allowing
    normal intentional motion within the threshold. Speeds below the threshold
    incur zero penalty.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        threshold: Angular velocity deadzone (rad/s). No penalty below this.
        body_names: Bodies to penalize. If None, uses all tracked bodies.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # [E, B, 3]
    ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
    # magnitude per body: [E, B]
    speed = torch.linalg.norm(ang_vel, dim=-1)
    # deadzone then square: [E, B]
    excess = torch.relu(speed - threshold)
    penalty = (excess * excess).mean(dim=-1)
    return penalty
