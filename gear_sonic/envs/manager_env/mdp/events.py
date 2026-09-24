"""Event functions for domain randomization and environment resets in RL training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import (
    _randomize_prop_by_op,
    randomize_rigid_body_mass as _IsaacLabRandomizeRigidBodyMass,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = None
    physics_mass_other_links = None
    physics_mass_torso = None
    hand_payload_mass = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None
    arm_com = None
    waist_com = None
    leg_com = None
    robot_inertia_random_torso = None
    robot_inertia_random_other_links = None
    joint_actuator_gains = None
    passive_foot_actuator_gains = None
    joint_armature = None
    joint_soft_pos_limits = None

    # interval - balance training
    push_robot = None
    random_gravity = None

    randomize_rigid_body_mass = None
    robot_joint_friction_armature = None
    robot_joint_friction_armature_regular = None
    robot_joint_friction_armature_ankle_waist = None


class randomize_rigid_body_mass(_IsaacLabRandomizeRigidBodyMass):
    """Randomize rigid body mass with an optional minimum mass clamp."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float | None = None,
    ):
        if min_mass is None:
            return super().__call__(
                env,
                env_ids,
                asset_cfg,
                mass_distribution_params,
                operation,
                distribution,
                recompute_inertia,
            )

        if min_mass <= 0.0:
            raise ValueError(f"min_mass must be positive, got {min_mass}.")

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        masses = self.asset.root_physx_view.get_masses()
        masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[
            env_ids[:, None], body_ids
        ].clone()

        masses = _randomize_prop_by_op(
            masses,
            mass_distribution_params,
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )
        masses[env_ids[:, None], body_ids] = masses[env_ids[:, None], body_ids].clamp_min(
            min_mass
        )

        self.asset.root_physx_view.set_masses(masses, env_ids)

        if recompute_inertia:
            ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[
                env_ids[:, None], body_ids
            ]
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                inertias[env_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
            self.asset.root_physx_view.set_inertias(inertias, env_ids)


def _scale_joint_pos_limits(
    hard_limits: torch.Tensor, factor: float
) -> torch.Tensor:
    """Scale joint ranges around their midpoint without changing hard limits."""
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"factor must be within (0, 1], got {factor}.")
    midpoint = hard_limits.mean(dim=-1, keepdim=True)
    return midpoint + (hard_limits - midpoint) * factor


def set_joint_soft_pos_limit_factor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    factor: float,
):
    """Apply a soft-limit factor to selected joints only.

    IsaacLab's articulation setting exposes one global factor.  This event
    narrows only the selected joints in ``asset.data.soft_joint_pos_limits``
    while leaving URDF/PhysX hard limits and all unselected joints unchanged.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    if env_ids is None:
        asset.data.soft_joint_pos_limits[:, joint_ids, :] = _scale_joint_pos_limits(
            asset.data.joint_pos_limits[:, joint_ids, :], factor
        )
        return

    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    if joint_ids == slice(None):
        asset.data.soft_joint_pos_limits[env_ids, :, :] = _scale_joint_pos_limits(
            asset.data.joint_pos_limits[env_ids, :, :], factor
        )
        return

    joint_ids_tensor = torch.as_tensor(joint_ids, device=asset.device, dtype=torch.long)
    env_index = env_ids[:, None]
    joint_index = joint_ids_tensor[None, :]
    asset.data.soft_joint_pos_limits[env_index, joint_index, :] = _scale_joint_pos_limits(
        asset.data.joint_pos_limits[env_index, joint_index, :], factor
    )


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint default positions to simulate calibration errors.

    Applies random offsets to the default joint positions of the robot, modeling
    real-world joint encoder calibration inaccuracies. Also updates the action
    manager offset to keep action space aligned with the new defaults.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        asset_cfg: Scene entity config with joint IDs to randomize.
        pos_distribution_params: Min/max range for the position offset distribution.
        operation: How to combine the random value with the original ("add", "scale", "abs").
        distribution: Sampling distribution type.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if not isinstance(env_ids, slice) and not isinstance(joint_ids, slice):
            target_env_ids = env_ids[:, None]
        else:
            target_env_ids = env_ids
        asset.data.default_joint_pos[target_env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically

        action_joint_names = env.action_manager.get_term("joint_pos")._joint_names
        asset_joint_names = asset.joint_names
        shared_joint_names = list(set(action_joint_names).intersection(set(asset_joint_names)))
        shared_joint_indices_action = [
            action_joint_names.index(name) for name in shared_joint_names
        ]
        shared_joint_indices_asset = [asset_joint_names.index(name) for name in shared_joint_names]

        if shared_joint_names:
            shared_joint_indices_action = torch.tensor(
                shared_joint_indices_action, dtype=torch.long, device=asset.device
            )
            shared_joint_indices_asset = torch.tensor(
                shared_joint_indices_asset, dtype=torch.long, device=asset.device
            )
            if isinstance(env_ids, slice):
                shared_offset = asset.data.default_joint_pos[:, shared_joint_indices_asset]
                env.action_manager.get_term("joint_pos")._offset[
                    :, shared_joint_indices_action
                ] = shared_offset
            else:
                shared_offset = asset.data.default_joint_pos[
                    env_ids[:, None], shared_joint_indices_asset
                ]
                env.action_manager.get_term("joint_pos")._offset[
                    env_ids[:, None], shared_joint_indices_action
                ] = shared_offset


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values independently for each selected body
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    num_bodies = asset.num_bodies if asset_cfg.body_ids == slice(None) else len(body_ids)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), num_bodies, 3), device="cpu"
    )

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    inertia_distribution_params: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize rigid-body inertia tensors.

    The random scalar is sampled per env/body and applied to all nine inertia tensor
    entries. This preserves the effect of any previous mass randomization in the
    same startup pass.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for inertia randomization."
        )

    inertias = asset.root_physx_view.get_inertias()
    inertia_values = inertias[env_ids[:, None], body_ids, :]
    samples = dist_fn(
        inertia_distribution_params[0],
        inertia_distribution_params[1],
        (len(env_ids), len(body_ids), 1),
        device=inertias.device,
    )

    if operation == "add":
        inertia_values += samples
    elif operation == "scale":
        inertia_values *= samples
    elif operation == "abs":
        inertia_values[:] = samples
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for inertia randomization."
        )

    inertias[env_ids[:, None], body_ids, :] = inertia_values
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_actuator_stiffness_with_damping_ratio(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float],
    damping_ratio_distribution_params: tuple[float, float] = (1.0, 1.0),
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize actuator stiffness while keeping damping ratio bounded.

    The damping is computed as ``d0 * sqrt(k / k0) * ratio_scale``. Assuming
    constant effective inertia, this keeps the damping ratio near the nominal
    value while allowing a wider stiffness range.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    def sample_scale(params: tuple[float, float], shape: tuple[int, int]) -> torch.Tensor:
        scales = torch.ones(shape, device=asset.device)
        return _randomize_prop_by_op(
            scales,
            params,
            dim_0_ids=None,
            dim_1_ids=slice(None),
            operation="scale",
            distribution=distribution,
        )

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            elif isinstance(actuator.joint_indices, torch.Tensor):
                global_indices = actuator.joint_indices.to(asset.device)
            else:
                raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
        elif isinstance(actuator.joint_indices, slice):
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            actuator_joint_indices = actuator.joint_indices.to(asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        nominal_stiffness = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
        nominal_damping = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
        stiffness_scale = sample_scale(stiffness_distribution_params, nominal_stiffness.shape)
        damping_ratio_scale = sample_scale(damping_ratio_distribution_params, nominal_damping.shape)

        stiffness = actuator.stiffness[env_ids].clone()
        damping = actuator.damping[env_ids].clone()
        stiffness[:, actuator_indices] = nominal_stiffness * stiffness_scale
        damping[:, actuator_indices] = nominal_damping * torch.sqrt(stiffness_scale) * damping_ratio_scale

        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping
        asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def push_by_setting_velocity_if_in_contact(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    force_threshold: float = 1.0,
):
    """Apply root velocity pushes only to envs with selected bodies in contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    body_ids = sensor_cfg.body_ids
    if body_ids == slice(None):
        body_ids = slice(None)
    else:
        body_ids = torch.tensor(body_ids, dtype=torch.long, device=asset.device)

    contact_forces = contact_sensor.data.net_forces_w[env_ids][:, body_ids, :]
    has_contact = torch.norm(contact_forces, dim=-1).amax(dim=1) > force_threshold
    push_env_ids = env_ids[has_contact]
    if len(push_env_ids) == 0:
        return

    vel_w = asset.data.root_vel_w[push_env_ids].clone()
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device
    )
    asset.write_root_velocity_to_sim(vel_w, env_ids=push_env_ids)
