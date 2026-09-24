"""Pure-Torch A3 motor torque-speed and parallel-joint utilities.

The policy and simulator remain in serial joint coordinates.  For the A3
parallel ankle and waist pitch/roll pairs, the physical motor coordinates use

    velocity_joint = J @ velocity_motor
    torque_motor = J.T @ torque_joint

The T-N curves below use output-shaft rad/s and Nm.  Each approximation is
deliberately limited to two line segments: one constant-torque plateau followed
by one linear decline to zero torque.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

_TN_CURVE_DATA: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "PFP110": (
        (
            0.0,
            9.42477796076938,
            10.471975511965976,
            11.519173063162574,
            12.566370614359172,
            13.613568165555769,
            14.660765716752367,
            15.707963267948966,
            19.634954084936208,
        ),
        (305.0, 305.0, 264.297, 224.922, 182.896, 143.348, 105.184, 62.183, 0.0),
    ),
    "PFP93": (
        (
            0.0,
            10.471975511965976,
            11.519173063162574,
            12.566370614359172,
            13.613568165555769,
            14.660765716752367,
            15.707963267948966,
            16.755160819145562,
            20.94395102393195,
        ),
        (220.0, 220.0, 193.135, 160.961, 133.026, 104.350, 75.010, 54.401, 0.0),
    ),
    "PFP78": (
        (
            0.0,
            16.755160819145562,
            17.80235837034216,
            18.84955592153876,
            23.561944901923447,
        ),
        (60.0, 60.0, 50.104, 44.310, 0.0),
    ),
    "PFP59": (
        (
            0.0,
            13.613568165555769,
            14.660765716752367,
            15.707963267948966,
            16.755160819145562,
            20.94395102393195,
        ),
        (34.8, 34.8, 24.053, 20.918, 17.073, 0.0),
    ),
    "PFP41": (
        (0.0, 20.94395102393195, 26.17993877991494),
        (6.0, 6.0, 0.0),
    ),
}

# Experiment 024 was trained with the conservative two-segment reward curves
# below.  Keep them separate from the actuator envelopes above so integrating
# the experiment does not change the existing A3MotorSpaceActuator behavior.
_MOTOR_REWARD_TN_CURVE_DATA: dict[
    str, tuple[tuple[float, ...], tuple[float, ...]]
] = {
    "PFP41": ((0.0, 20.943951, 26.179939), (5.8, 5.8, 0.0)),
    "PFP59": ((0.0, 13.613568, 17.0), (34.0, 34.0, 0.0)),
    "PFP78": ((0.0, 16.755161, 23.561945), (58.0, 58.0, 0.0)),
    "PFP93": ((0.0, 10.471976, 18.5), (215.0, 215.0, 0.0)),
    "PFP110": ((0.0, 9.424778, 17.3), (300.0, 300.0, 0.0)),
}


@dataclass(frozen=True)
class TNCurve:
    """One symmetric output-shaft torque-speed capability curve."""

    family: str
    speed_rad_s: torch.Tensor
    torque_nm: torch.Tensor


@dataclass(frozen=True)
class MotorPenalty:
    """Per-motor soft, hard, and combined utilization penalties."""

    soft: torch.Tensor
    hard: torch.Tensor
    combined: torch.Tensor


@dataclass(frozen=True)
class ParallelMotorState:
    """Physical motor values mapped from one or more parallel joint pairs."""

    motor_velocity: torch.Tensor
    motor_torque: torch.Tensor
    torque_limit: torch.Tensor
    utilization: torch.Tensor


@dataclass(frozen=True)
class ParallelActuationResult:
    """Motor- and joint-space values produced by one parallel saturation step."""

    motor_velocity: torch.Tensor
    requested_motor_torque: torch.Tensor
    applied_motor_torque: torch.Tensor
    motor_torque_limit: torch.Tensor
    utilization: torch.Tensor
    saturated: torch.Tensor
    applied_joint_torque: torch.Tensor


def get_tn_curve(
    family: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    profile: str = "actuator",
) -> TNCurve:
    """Return an A3 motor T-N curve on the requested device.

    ``actuator`` preserves the pre-existing mainline actuator envelopes.
    ``reward_024`` reproduces the exact curves used by experiment 024.
    """

    if profile == "actuator":
        curve_data = _TN_CURVE_DATA
    elif profile == "reward_024":
        curve_data = _MOTOR_REWARD_TN_CURVE_DATA
    else:
        raise ValueError(f"unknown A3 T-N curve profile {profile!r}")

    try:
        speeds, torques = curve_data[family]
    except KeyError as exc:
        raise KeyError(f"unknown A3 motor family {family!r}") from exc
    return TNCurve(
        family=family,
        speed_rad_s=torch.tensor(speeds, device=device, dtype=dtype),
        torque_nm=torch.tensor(torques, device=device, dtype=dtype),
    )


def interpolate_tn_limit(speed_rad_s: torch.Tensor, curve: TNCurve) -> torch.Tensor:
    """Linearly interpolate the symmetric torque limit at ``abs(speed)``."""

    speed = torch.abs(speed_rad_s)
    grid = curve.speed_rad_s
    torque = curve.torque_nm
    if grid.ndim != 1 or torque.ndim != 1 or grid.numel() != torque.numel():
        raise ValueError("T-N curve grids must be same-length one-dimensional tensors")
    if grid.numel() < 2:
        raise ValueError("T-N curve requires at least two points")

    upper = torch.searchsorted(grid, speed.contiguous(), right=False)
    upper = torch.clamp(upper, 1, grid.numel() - 1)
    lower = upper - 1
    x0, x1 = grid[lower], grid[upper]
    y0, y1 = torque[lower], torque[upper]
    weight = (speed - x0) / torch.clamp_min(x1 - x0, torch.finfo(grid.dtype).eps)
    result = y0 + torch.clamp(weight, 0.0, 1.0) * (y1 - y0)
    result = torch.where(speed <= grid[0], torque[0], result)
    result = torch.where(speed >= grid[-1], torque[-1], result)
    return torch.clamp_min(result, 0.0)


def peak_mechanical_power(curve: TNCurve) -> torch.Tensor:
    """Return the maximum ``speed * torque`` on a two-segment T-N curve."""

    speed = curve.speed_rad_s
    torque = curve.torque_nm
    if speed.ndim != 1 or torque.ndim != 1 or speed.numel() != 3 or torque.numel() != 3:
        raise ValueError("A3 peak-power calculation requires a three-vertex T-N curve")

    slope = (torque[2] - torque[1]) / torch.clamp_min(
        speed[2] - speed[1],
        torch.finfo(speed.dtype).eps,
    )
    intercept = torque[1] - slope * speed[1]
    decline_vertex = torch.clamp(
        -intercept / torch.clamp_max(2.0 * slope, -torch.finfo(speed.dtype).eps),
        min=speed[1],
        max=speed[2],
    )
    decline_torque = slope * decline_vertex + intercept
    candidates = torch.stack(
        (
            speed[0] * torque[0],
            speed[1] * torque[1],
            speed[2] * torque[2],
            decline_vertex * decline_torque,
        )
    )
    return torch.max(candidates)


def compute_motor_penalty(
    utilization: torch.Tensor,
    *,
    lambda_soft: float = 0.05,
    lambda_violation: float = 5.0,
    soft_exponent: float = 8.0,
    violation_exponent: float = 2.0,
    utilization_cap: float = 2.0,
) -> MotorPenalty:
    """Compute dense T-N shaping plus a bounded over-limit penalty.

    The soft term is ``min(utilization, 1) ** soft_exponent`` and therefore has
    no deadzone.  Above one, an additional
    ``relu(utilization - 1) ** violation_exponent`` term is applied.  The
    utilization cap keeps the penalty finite when the interpolated T-N limit
    reaches zero.
    """

    if lambda_soft < 0.0 or lambda_violation < 0.0:
        raise ValueError("motor penalty coefficients must be non-negative")
    if soft_exponent <= 0.0 or violation_exponent <= 0.0:
        raise ValueError("motor penalty exponents must be positive")
    if utilization_cap < 1.0:
        raise ValueError("utilization_cap must be at least 1")

    capped = torch.clamp(utilization, min=0.0, max=float(utilization_cap))
    soft = torch.pow(
        torch.minimum(capped, torch.ones_like(capped)),
        float(soft_exponent),
    )
    hard = torch.pow(
        torch.relu(capped - 1.0),
        float(violation_exponent),
    )
    return MotorPenalty(
        soft=soft,
        hard=hard,
        combined=float(lambda_soft) * soft + float(lambda_violation) * hard,
    )


def compute_threshold_penalty(
    utilization: torch.Tensor,
    *,
    threshold_ratio: float,
    lambda_soft: float = 1.0,
    lambda_violation: float = 5.0,
    soft_exponent: float = 2.0,
    violation_exponent: float = 2.0,
    utilization_cap: float = 2.0,
) -> MotorPenalty:
    """Compute an early threshold penalty plus an over-limit penalty."""

    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in [0, 1]")
    if lambda_soft < 0.0 or lambda_violation < 0.0:
        raise ValueError("threshold penalty coefficients must be non-negative")
    if soft_exponent <= 0.0 or violation_exponent <= 0.0:
        raise ValueError("threshold penalty exponents must be positive")
    if utilization_cap < 1.0:
        raise ValueError("utilization_cap must be at least 1")

    capped = torch.clamp(utilization, min=0.0, max=float(utilization_cap))
    soft = torch.pow(
        torch.relu(capped - float(threshold_ratio)),
        float(soft_exponent),
    )
    hard = torch.pow(
        torch.relu(capped - 1.0),
        float(violation_exponent),
    )
    return MotorPenalty(
        soft=soft,
        hard=hard,
        combined=float(lambda_soft) * soft + float(lambda_violation) * hard,
    )


def compute_cross_penalty(
    normalized_pair_load: torch.Tensor,
    *,
    threshold_ratio: float = 0.75,
) -> torch.Tensor:
    """Penalize a two-axis pair only when both normalized loads are high."""

    if normalized_pair_load.shape[-1] != 2:
        raise ValueError("parallel cross penalty expects pitch/roll pairs on the last axis")
    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in [0, 1]")
    excess = torch.relu(normalized_pair_load - float(threshold_ratio))
    return excess[..., 0] * excess[..., 1]


def compute_position_threshold_penalty(
    joint_position: torch.Tensor,
    hard_joint_limits: torch.Tensor,
    *,
    threshold_ratio: float = 0.85,
    exponent: float = 2.0,
) -> torch.Tensor:
    """Sum joint-position excess relative to the physical hard range."""

    if hard_joint_limits.shape[-1] != 2:
        raise ValueError("joint limits must contain lower/upper bounds on the last axis")
    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in [0, 1]")
    if exponent <= 0.0:
        raise ValueError("position penalty exponent must be positive")

    lower = hard_joint_limits[..., 0]
    upper = hard_joint_limits[..., 1]
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    valid = half_range > torch.finfo(joint_position.dtype).eps
    ratio = torch.where(
        valid,
        torch.abs(joint_position - center) / torch.clamp_min(half_range, torch.finfo(joint_position.dtype).eps),
        torch.zeros_like(joint_position),
    )
    excess = torch.pow(torch.relu(ratio - float(threshold_ratio)), float(exponent))
    return torch.where(valid, excess, torch.zeros_like(excess)).sum(dim=-1)


def motor_family_for_joint(joint_name: str) -> str | None:
    """Map one direct-drive A3 joint to its physical motor family.

    Parallel ankle and waist pitch/roll joints intentionally return ``None``;
    they are evaluated only after mapping to their two physical PFP78 motors.
    """

    if "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
        return None
    if joint_name in {"waist_pitch_joint", "waist_roll_joint"}:
        return None
    if joint_name.endswith("_knee_joint"):
        return "PFP110"
    if (
        "_hip_pitch_joint" in joint_name
        or "_hip_roll_joint" in joint_name
        or "_hip_yaw_joint" in joint_name
        or joint_name == "waist_yaw_joint"
    ):
        return "PFP93"
    if "_shoulder_pitch_joint" in joint_name or "_shoulder_roll_joint" in joint_name:
        return "PFP78"
    if "_shoulder_yaw_joint" in joint_name or "_elbow_joint" in joint_name or "_wrist_roll_joint" in joint_name:
        return "PFP59"
    if "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
        return "PFP41"
    return None


@dataclass(frozen=True)
class ParallelJacobianTable:
    """GPU-resident bilinear table for ``J = d(q_joint) / d(q_motor)``."""

    name: str
    pitch_axis: torch.Tensor
    roll_axis: torch.Tensor
    jacobian: torch.Tensor
    reachable: torch.Tensor

    _CACHE: ClassVar[dict[tuple[str, str, torch.dtype, str], "ParallelJacobianTable"]] = {}

    @staticmethod
    def default_table_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "a3_motor_model"

    @classmethod
    def load(
        cls,
        name: str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        table_dir: str | Path | None = None,
    ) -> "ParallelJacobianTable":
        """Load and cache one A3 HAL Jacobian table."""

        resolved_dir = Path(table_dir) if table_dir is not None else cls.default_table_dir()
        resolved_dir = resolved_dir.expanduser().resolve()
        device_key = str(torch.device(device or "cpu"))
        key = (name, device_key, dtype, str(resolved_dir))
        if key in cls._CACHE:
            return cls._CACHE[key]

        path = resolved_dir / f"{name}_jacobi.npz"
        if not path.is_file():
            raise FileNotFoundError(f"A3 parallel Jacobian table not found: {path}")
        with np.load(path) as data:
            table = cls(
                name=name,
                pitch_axis=torch.as_tensor(
                    np.ascontiguousarray(data["pitch_axis"]),
                    device=device,
                    dtype=dtype,
                ),
                roll_axis=torch.as_tensor(
                    np.ascontiguousarray(data["roll_axis"]),
                    device=device,
                    dtype=dtype,
                ),
                jacobian=torch.as_tensor(
                    np.ascontiguousarray(data["J"]),
                    device=device,
                    dtype=dtype,
                ),
                reachable=torch.as_tensor(
                    np.ascontiguousarray(data["reachable"]).astype(np.bool_),
                    device=device,
                    dtype=torch.bool,
                ),
            )
        cls._CACHE[key] = table
        return table

    def interpolate(self, position_pr: torch.Tensor) -> torch.Tensor:
        """Bilinearly interpolate J for ``position_pr[..., (pitch, roll)]``."""

        if position_pr.shape[-1] != 2:
            raise ValueError(f"expected position_pr[..., 2], got {tuple(position_pr.shape)}")
        pitch = position_pr[..., 0]
        roll = position_pr[..., 1]
        pitch_high = torch.searchsorted(
            self.pitch_axis,
            pitch.contiguous(),
            right=False,
        )
        roll_high = torch.searchsorted(
            self.roll_axis,
            roll.contiguous(),
            right=False,
        )
        pitch_high = torch.clamp(pitch_high, 1, self.pitch_axis.numel() - 1)
        roll_high = torch.clamp(roll_high, 1, self.roll_axis.numel() - 1)
        pitch_low = pitch_high - 1
        roll_low = roll_high - 1

        pitch0 = self.pitch_axis[pitch_low]
        pitch1 = self.pitch_axis[pitch_high]
        roll0 = self.roll_axis[roll_low]
        roll1 = self.roll_axis[roll_high]
        pitch_weight = torch.clamp(
            (pitch - pitch0) / (pitch1 - pitch0),
            0.0,
            1.0,
        )[..., None, None]
        roll_weight = torch.clamp(
            (roll - roll0) / (roll1 - roll0),
            0.0,
            1.0,
        )[..., None, None]

        c00 = self.jacobian[pitch_low, roll_low]
        c01 = self.jacobian[pitch_low, roll_high]
        c10 = self.jacobian[pitch_high, roll_low]
        c11 = self.jacobian[pitch_high, roll_high]
        low = c00 * (1.0 - roll_weight) + c01 * roll_weight
        high = c10 * (1.0 - roll_weight) + c11 * roll_weight
        return low * (1.0 - pitch_weight) + high * pitch_weight


def solve_parallel_motor_velocity_2x2(
    jacobian: torch.Tensor,
    joint_velocity: torch.Tensor,
) -> torch.Tensor:
    """Solve ``J @ motor_velocity = joint_velocity`` for batched 2x2 Jacobians."""

    if jacobian.shape[-2:] != (2, 2):
        raise ValueError(f"expected jacobian[..., 2, 2], got {tuple(jacobian.shape)}")
    if joint_velocity.shape[-1] != 2:
        raise ValueError("parallel joint velocity must end in dimension 2")

    a = jacobian[..., 0, 0]
    b = jacobian[..., 0, 1]
    c = jacobian[..., 1, 0]
    d = jacobian[..., 1, 1]
    v0 = joint_velocity[..., 0]
    v1 = joint_velocity[..., 1]
    determinant = a * d - b * c
    return torch.stack(
        (
            (d * v0 - b * v1) / determinant,
            (a * v1 - c * v0) / determinant,
        ),
        dim=-1,
    )


def map_parallel_joint_torque(
    jacobian: torch.Tensor,
    joint_torque: torch.Tensor,
) -> torch.Tensor:
    """Map joint torque to motor torque without computing motor velocity."""

    if jacobian.shape[-2:] != (2, 2):
        raise ValueError(f"expected jacobian[..., 2, 2], got {tuple(jacobian.shape)}")
    if joint_torque.shape[-1] != 2:
        raise ValueError("parallel joint torque must end in dimension 2")
    return torch.einsum(
        "...ji,...j->...i",
        jacobian,
        joint_torque,
    )


def map_parallel_joint_to_motor(
    *,
    jacobian: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_torque: torch.Tensor,
    motor_family: str = "PFP78",
    curve: TNCurve | None = None,
) -> ParallelMotorState:
    """Map parallel pitch/roll joint demand into the two physical motors."""

    if jacobian.shape[-2:] != (2, 2):
        raise ValueError(f"expected jacobian[..., 2, 2], got {tuple(jacobian.shape)}")
    if joint_velocity.shape[-1] != 2 or joint_torque.shape[-1] != 2:
        raise ValueError("parallel joint velocity and torque must end in dimension 2")

    motor_velocity = solve_parallel_motor_velocity_2x2(jacobian, joint_velocity)
    motor_torque = map_parallel_joint_torque(jacobian, joint_torque)
    if curve is None:
        curve = get_tn_curve(
            motor_family,
            device=motor_velocity.device,
            dtype=motor_velocity.dtype,
        )
    torque_limit = interpolate_tn_limit(motor_velocity, curve)
    utilization = torch.abs(motor_torque) / torch.clamp_min(
        torque_limit,
        torch.finfo(torque_limit.dtype).eps,
    )
    return ParallelMotorState(
        motor_velocity=motor_velocity,
        motor_torque=motor_torque,
        torque_limit=torque_limit,
        utilization=utilization,
    )


def apply_parallel_motor_limits(
    *,
    jacobian: torch.Tensor,
    joint_velocity: torch.Tensor,
    requested_joint_torque: torch.Tensor,
    motor_family: str = "PFP78",
    curve: TNCurve | None = None,
    torque_scale: float | torch.Tensor = 1.0,
) -> ParallelActuationResult:
    """Apply independent motor T-N limits to one or more 2-DoF pairs."""

    if jacobian.shape[-2:] != (2, 2):
        raise ValueError(f"expected jacobian[..., 2, 2], got {tuple(jacobian.shape)}")
    if joint_velocity.shape[-1] != 2 or requested_joint_torque.shape[-1] != 2:
        raise ValueError("parallel joint velocity and torque must have a final dimension of 2")

    motor_velocity = solve_parallel_motor_velocity_2x2(jacobian, joint_velocity)
    requested_motor_torque = map_parallel_joint_torque(jacobian, requested_joint_torque)
    if curve is None:
        curve = get_tn_curve(
            motor_family,
            device=motor_velocity.device,
            dtype=motor_velocity.dtype,
        )
    motor_torque_limit = torch.clamp_min(
        interpolate_tn_limit(motor_velocity, curve) * torque_scale,
        0.0,
    )
    applied_motor_torque = torch.clamp(
        requested_motor_torque,
        min=-motor_torque_limit,
        max=motor_torque_limit,
    )
    applied_joint_torque = torch.linalg.solve(
        jacobian.transpose(-1, -2),
        applied_motor_torque.unsqueeze(-1),
    ).squeeze(-1)
    utilization = torch.abs(requested_motor_torque) / torch.clamp_min(
        motor_torque_limit,
        torch.finfo(motor_torque_limit.dtype).eps,
    )
    saturated = torch.abs(requested_motor_torque) > motor_torque_limit
    return ParallelActuationResult(
        motor_velocity=motor_velocity,
        requested_motor_torque=requested_motor_torque,
        applied_motor_torque=applied_motor_torque,
        motor_torque_limit=motor_torque_limit,
        utilization=utilization,
        saturated=saturated,
        applied_joint_torque=applied_joint_torque,
    )


@dataclass
class _CachedParallelMotorState:
    jacobian: torch.Tensor
    motor_torque: torch.Tensor
    motor_velocity: torch.Tensor | None = None
    torque_limit: torch.Tensor | None = None
    utilization: torch.Tensor | None = None


class A3MotorConstraintCache:
    """Lazily compute and reuse A3 motor coordinates within one environment step."""

    _PARALLEL_PAIRS = (
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

    def __init__(
        self,
        joint_names: tuple[str, ...] | list[str],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        table_dir: str | Path | None = None,
    ):
        self._num_joints = len(joint_names)
        joint_index = {name: index for index, name in enumerate(joint_names)}
        family_indices: dict[str, list[int]] = {}
        for index, name in enumerate(joint_names):
            family = motor_family_for_joint(name)
            if family is not None:
                family_indices.setdefault(family, []).append(index)
        self.serial_family_indices = {
            family: torch.tensor(indices, device=device, dtype=torch.long)
            for family, indices in family_indices.items()
        }
        self._serial_curves = {
            family: get_tn_curve(
                family,
                device=device,
                dtype=dtype,
                profile="reward_024",
            )
            for family in self.serial_family_indices
        }

        self.parallel_pair_indices = {}
        self._parallel_tables = {}
        for table_name, pitch_name, roll_name in self._PARALLEL_PAIRS:
            try:
                indices = (joint_index[pitch_name], joint_index[roll_name])
            except KeyError as exc:
                raise ValueError(
                    f"A3 motor constraint cache could not find parallel joint {exc.args[0]!r}"
                ) from exc
            self.parallel_pair_indices[table_name] = torch.tensor(
                indices,
                device=device,
                dtype=torch.long,
            )
            self._parallel_tables[table_name] = ParallelJacobianTable.load(
                table_name,
                device=device,
                dtype=dtype,
                table_dir=table_dir,
            )
        self._parallel_curve = get_tn_curve(
            "PFP78",
            device=device,
            dtype=dtype,
            profile="reward_024",
        )

        self._step_token: int | None = None
        self._joint_position: torch.Tensor | None = None
        self._joint_velocity: torch.Tensor | None = None
        self._joint_torque: torch.Tensor | None = None
        self._serial_utilization: torch.Tensor | None = None
        self._parallel_states: dict[str, _CachedParallelMotorState] = {}

    @property
    def parallel_names(self) -> tuple[str, ...]:
        return tuple(self.parallel_pair_indices)

    @property
    def ankle_names(self) -> tuple[str, str]:
        return ("left_ankle", "right_ankle")

    def begin_step(
        self,
        step_token: int,
        *,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        joint_torque: torch.Tensor,
    ) -> A3MotorConstraintCache:
        """Bind the current simulator tensors and invalidate values on a new step."""

        if step_token == self._step_token:
            return self
        if joint_position.shape != joint_velocity.shape or joint_position.shape != joint_torque.shape:
            raise ValueError("joint position, velocity, and torque tensors must have identical shapes")
        if joint_position.ndim != 2:
            raise ValueError("joint state tensors must have shape (num_envs, num_joints)")
        if joint_position.shape[1] != self._num_joints:
            raise ValueError(
                "joint state tensor width does not match the configured joint names: "
                f"{joint_position.shape[1]} != {self._num_joints}"
            )

        self._step_token = step_token
        self._joint_position = joint_position
        self._joint_velocity = joint_velocity
        self._joint_torque = joint_torque
        self._serial_utilization = None
        self._parallel_states.clear()
        return self

    def _require_step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._joint_position is None or self._joint_velocity is None or self._joint_torque is None:
            raise RuntimeError("begin_step() must be called before reading cached motor values")
        return self._joint_position, self._joint_velocity, self._joint_torque

    def _parallel_state(self, table_name: str) -> _CachedParallelMotorState:
        joint_position, _, joint_torque = self._require_step()
        if table_name not in self.parallel_pair_indices:
            raise KeyError(f"unknown A3 parallel pair {table_name!r}")
        state = self._parallel_states.get(table_name)
        if state is None:
            indices = self.parallel_pair_indices[table_name]
            jacobian = self._parallel_tables[table_name].interpolate(joint_position[:, indices])
            state = _CachedParallelMotorState(
                jacobian=jacobian,
                motor_torque=map_parallel_joint_torque(
                    jacobian,
                    joint_torque[:, indices],
                ),
            )
            self._parallel_states[table_name] = state
        return state

    def serial_utilization(self) -> torch.Tensor:
        """Return concatenated serial-motor T-N utilization for the current step."""

        _, joint_velocity, joint_torque = self._require_step()
        if self._serial_utilization is None:
            utilization = []
            for family, indices in self.serial_family_indices.items():
                limit = interpolate_tn_limit(
                    joint_velocity[:, indices],
                    self._serial_curves[family],
                )
                utilization.append(
                    torch.abs(joint_torque[:, indices]) / torch.clamp_min(limit, torch.finfo(limit.dtype).eps)
                )
            if utilization:
                self._serial_utilization = torch.cat(utilization, dim=-1)
            else:
                self._serial_utilization = torch.zeros(
                    (joint_torque.shape[0], 0),
                    device=joint_torque.device,
                    dtype=joint_torque.dtype,
                )
        return self._serial_utilization

    def parallel_motor_torque(self, table_name: str) -> torch.Tensor:
        """Return motor torque for one parallel pair without forcing a velocity solve."""

        return self._parallel_state(table_name).motor_torque

    def parallel_motor_velocity(self, table_name: str) -> torch.Tensor:
        """Return motor velocity for one parallel pair, solving it at most once per step."""

        _, joint_velocity, _ = self._require_step()
        state = self._parallel_state(table_name)
        if state.motor_velocity is None:
            indices = self.parallel_pair_indices[table_name]
            state.motor_velocity = solve_parallel_motor_velocity_2x2(
                state.jacobian,
                joint_velocity[:, indices],
            )
        return state.motor_velocity

    def parallel_pair_utilization(self, table_name: str) -> torch.Tensor:
        """Return T-N utilization for one parallel pair."""

        state = self._parallel_state(table_name)
        if state.utilization is None:
            motor_velocity = self.parallel_motor_velocity(table_name)
            state.torque_limit = interpolate_tn_limit(
                motor_velocity,
                self._parallel_curve,
            )
            state.utilization = torch.abs(state.motor_torque) / torch.clamp_min(
                state.torque_limit,
                torch.finfo(state.torque_limit.dtype).eps,
            )
        return state.utilization

    def parallel_utilization(self) -> torch.Tensor:
        """Return concatenated T-N utilization for ankles and waist."""

        return torch.cat(
            [self.parallel_pair_utilization(name) for name in self.parallel_names],
            dim=-1,
        )
