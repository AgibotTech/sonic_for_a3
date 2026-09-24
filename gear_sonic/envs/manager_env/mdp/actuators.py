from __future__ import annotations

from collections.abc import Sequence

from isaaclab.actuators import (
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitActuator,
    ImplicitActuatorCfg,
)
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions
import torch

from gear_sonic.utils.a3_motor_model import (
    ParallelJacobianTable,
    apply_parallel_motor_limits,
    get_tn_curve,
    interpolate_tn_limit,
)


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.velocities_delay_buffer = DelayBuffer(
            cfg.max_delay, self._num_envs, device=self._device
        )
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts
        )
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


class A3MotorSpaceActuator(IdealPDActuator):
    """Explicit A3 PD actuator with measured T-N and parallel motor saturation.

    Policy commands remain in the serial joint coordinates.  For ankle and
    waist pitch/roll pairs, requested torque and velocity are mapped through
    the pose-dependent HAL Jacobian, independently limited in motor space,
    then mapped back to the serial coordinates before being sent to PhysX.
    """

    cfg: "A3MotorSpaceActuatorCfg"

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

    def __init__(self, cfg: "A3MotorSpaceActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._all_indices = torch.arange(
            self._num_envs,
            dtype=torch.long,
            device=self._device,
        )
        self._positions_delay_buffer = DelayBuffer(
            cfg.max_delay,
            self._num_envs,
            device=self._device,
        )
        self._velocities_delay_buffer = DelayBuffer(
            cfg.max_delay,
            self._num_envs,
            device=self._device,
        )
        self._efforts_delay_buffer = DelayBuffer(
            cfg.max_delay,
            self._num_envs,
            device=self._device,
        )

        name_to_index = {name: index for index, name in enumerate(self.joint_names)}
        self._parallel_pairs: list[
            tuple[str, int, int, torch.Tensor, ParallelJacobianTable]
        ] = []
        parallel_indices: set[int] = set()
        dtype = self.computed_effort.dtype
        for table_name, pitch_name, roll_name in self._PARALLEL_PAIRS:
            if pitch_name not in name_to_index or roll_name not in name_to_index:
                continue
            pitch_index = name_to_index[pitch_name]
            roll_index = name_to_index[roll_name]
            table = ParallelJacobianTable.load(
                table_name,
                device=self._device,
                dtype=dtype,
                table_dir=cfg.jacobian_table_dir,
            )
            pair_indices = torch.tensor(
                (pitch_index, roll_index),
                device=self._device,
                dtype=torch.long,
            )
            self._parallel_pairs.append(
                (table_name, pitch_index, roll_index, pair_indices, table)
            )
            parallel_indices.update((pitch_index, roll_index))

        family_indices: dict[str, list[int]] = {}
        for index, name in enumerate(self.joint_names):
            if index in parallel_indices:
                continue
            family = self._motor_family_for_joint(name)
            if family is not None:
                family_indices.setdefault(family, []).append(index)
        self._serial_family_indices = {
            family: torch.tensor(indices, device=self._device, dtype=torch.long)
            for family, indices in family_indices.items()
        }
        self._tn_curves = {
            family: get_tn_curve(family, device=self._device, dtype=dtype)
            for family in family_indices
        }
        self._parallel_tn_curve = get_tn_curve(
            "PFP78",
            device=self._device,
            dtype=dtype,
        )

        self.motor_names = list(self.joint_names)
        for table_name, pitch_index, roll_index, _, _ in self._parallel_pairs:
            self.motor_names[pitch_index] = f"{table_name}_motor_0"
            self.motor_names[roll_index] = f"{table_name}_motor_1"
        self.motor_velocity = torch.zeros_like(self.computed_effort)
        self.requested_motor_torque = torch.zeros_like(self.computed_effort)
        self.applied_motor_torque = torch.zeros_like(self.computed_effort)
        self.motor_torque_limit = torch.full_like(self.computed_effort, torch.inf)
        self.motor_utilization = torch.zeros_like(self.computed_effort)
        self.motor_saturated = torch.zeros_like(self.computed_effort, dtype=torch.bool)
        self.joint_effort_loss = torch.zeros_like(self.computed_effort)

    @staticmethod
    def _motor_family_for_joint(joint_name: str) -> str | None:
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
        if (
            "_shoulder_yaw_joint" in joint_name
            or "_elbow_joint" in joint_name
            or "_wrist_roll_joint" in joint_name
        ):
            return "PFP59"
        if "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
            return "PFP41"
        return None

    def reset(self, env_ids: Sequence[int] | slice | None):
        if env_ids is None or env_ids == slice(None):
            env_ids = self._all_indices
        num_envs = len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self._positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self._velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self._efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        self._positions_delay_buffer.reset(env_ids)
        self._velocities_delay_buffer.reset(env_ids)
        self._efforts_delay_buffer.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        desired_pos = (
            control_action.joint_positions
            if control_action.joint_positions is not None
            else joint_pos
        )
        desired_vel = (
            control_action.joint_velocities
            if control_action.joint_velocities is not None
            else torch.zeros_like(joint_vel)
        )
        feedforward = (
            control_action.joint_efforts
            if control_action.joint_efforts is not None
            else torch.zeros_like(joint_pos)
        )
        desired_pos = self._positions_delay_buffer.compute(desired_pos)
        desired_vel = self._velocities_delay_buffer.compute(desired_vel)
        feedforward = self._efforts_delay_buffer.compute(feedforward)

        self.computed_effort = (
            self.stiffness * (desired_pos - joint_pos)
            + self.damping * (desired_vel - joint_vel)
            + feedforward
        )
        self.applied_effort.copy_(self.computed_effort)
        self.motor_velocity.copy_(joint_vel)
        self.requested_motor_torque.copy_(self.computed_effort)
        self.applied_motor_torque.copy_(self.computed_effort)
        self.motor_torque_limit.fill_(torch.inf)
        self.motor_utilization.zero_()
        self.motor_saturated.zero_()

        for family, indices in self._serial_family_indices.items():
            requested = self.computed_effort[:, indices]
            speed = joint_vel[:, indices]
            tn_limit = interpolate_tn_limit(speed, self._tn_curves[family])
            limit = torch.minimum(
                tn_limit * self.cfg.tn_torque_scale,
                self.effort_limit[:, indices],
            )
            applied = torch.clamp(requested, min=-limit, max=limit)
            self.applied_effort[:, indices] = applied
            self.applied_motor_torque[:, indices] = applied
            self.motor_torque_limit[:, indices] = limit
            self.motor_utilization[:, indices] = torch.abs(requested) / torch.clamp_min(
                limit,
                torch.finfo(limit.dtype).eps,
            )
            self.motor_saturated[:, indices] = torch.abs(requested) > limit

        for _, pitch_index, roll_index, pair_indices, table in self._parallel_pairs:
            position_pr = joint_pos[:, pair_indices]
            velocity_pr = joint_vel[:, pair_indices]
            requested_pr = self.computed_effort[:, pair_indices]
            jacobian = table.interpolate(position_pr)
            result = apply_parallel_motor_limits(
                jacobian=jacobian,
                joint_velocity=velocity_pr,
                requested_joint_torque=requested_pr,
                motor_family="PFP78",
                curve=self._parallel_tn_curve,
                torque_scale=self.cfg.parallel_torque_scale,
            )
            self.applied_effort[:, pair_indices] = result.applied_joint_torque
            self.motor_velocity[:, pair_indices] = result.motor_velocity
            self.requested_motor_torque[:, pair_indices] = result.requested_motor_torque
            self.applied_motor_torque[:, pair_indices] = result.applied_motor_torque
            self.motor_torque_limit[:, pair_indices] = result.motor_torque_limit
            self.motor_utilization[:, pair_indices] = result.utilization
            self.motor_saturated[:, pair_indices] = result.saturated

        self.joint_effort_loss.copy_(self.computed_effort - self.applied_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class A3MotorSpaceActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`A3MotorSpaceActuator`."""

    class_type: type = A3MotorSpaceActuator
    tn_torque_scale: float = 1.0
    parallel_torque_scale: float = 1.0
    jacobian_table_dir: str | None = None
    min_delay: int = 0
    max_delay: int = 0
