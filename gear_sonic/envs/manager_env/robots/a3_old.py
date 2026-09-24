import logging
import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
import torch

from gear_sonic.utils.a3_motor_params import (
    ARMATURE_PFP_110_75,
    ARMATURE_PFP_41_48,
    ARMATURE_PFP_59_60,
    ARMATURE_PFP_78_58,
    ARMATURE_PFP_93_65,
)

ASSET_DIR = "gear_sonic/data/assets"
logger = logging.getLogger(__name__)


A3_ACTIVE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

A3_FOOT_CONTACT_LINK_NAMES = [
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_foot_forefoot_Link",
    "right_foot_forefoot_Link",
    "left_foot_toe_Link",
    "right_foot_toe_Link",
    "left_foot_rear_Link",
    "right_foot_rear_Link",
    "left_foot_mid_Link",
    "right_foot_mid_Link",
    "left_foot_front_Link",
    "right_foot_front_Link",
]


def _sample_uniform_pair(value: tuple[float, float] | list[float]) -> float:
    return float(torch.empty((), device="cpu").uniform_(float(value[0]), float(value[1])))


def _find_descendant_prims_by_name(root_prim, names: list[str]) -> list:
    name_set = set(names)
    matches = []
    pending = [root_prim]
    while pending:
        prim = pending.pop(0)
        if prim.GetName() in name_set:
            matches.append(prim)
        pending.extend(prim.GetChildren())
    return matches


def _make_foot_contact_material_cfg(compliance_cfg: dict) -> tuple:
    base_stiffness = float(compliance_cfg.get("base_stiffness", 69050.0))
    base_damping = float(compliance_cfg.get("base_damping", 650.0))
    stiffness_scale = _sample_uniform_pair(compliance_cfg.get("stiffness_scale_range", (1.0, 1.0)))
    damping_ratio_scale = _sample_uniform_pair(
        compliance_cfg.get("damping_ratio_scale_range", (1.0, 1.0))
    )
    stiffness = base_stiffness * stiffness_scale
    damping = base_damping * math.sqrt(stiffness_scale) * damping_ratio_scale

    material_cfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=compliance_cfg.get("friction_combine_mode", "multiply"),
        restitution_combine_mode=compliance_cfg.get("restitution_combine_mode", "multiply"),
        static_friction=float(compliance_cfg.get("static_friction", 1.0)),
        dynamic_friction=float(compliance_cfg.get("dynamic_friction", 1.0)),
        restitution=float(compliance_cfg.get("restitution", 0.0)),
        compliant_contact_stiffness=float(stiffness),
        compliant_contact_damping=float(damping),
    )
    return material_cfg, stiffness, stiffness_scale, damping, damping_ratio_scale


def _bind_foot_contact_material(
    root_prim,
    material_path: str,
    material_cfg: sim_utils.RigidBodyMaterialCfg,
    link_names: list[str],
):
    stage = sim_utils.get_current_stage()
    material_cfg.func(material_path, material_cfg)
    link_prims = _find_descendant_prims_by_name(root_prim, link_names)
    if not link_prims:
        raise RuntimeError(
            "Could not find A3 foot contact link prims under"
            f" '{root_prim.GetPath()}'. Expected one of: {link_names}"
        )

    for link_prim in link_prims:
        sim_utils.make_uninstanceable(link_prim.GetPath(), stage=stage)
        sim_utils.bind_physics_material(link_prim.GetPath(), material_path, stage=stage)
    return len(link_prims)


@sim_utils.clone
def spawn_a3_urdf_with_foot_contact_compliance(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn A3 from URDF and bind one compliant-contact material before cloning."""
    spawn_from_urdf = getattr(sim_utils.spawn_from_urdf, "__wrapped__", sim_utils.spawn_from_urdf)
    prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)

    compliance_cfg = getattr(cfg, "foot_contact_compliance", None)
    if not compliance_cfg or not compliance_cfg.get("enabled", False):
        return prim

    material_cfg, stiffness, stiffness_scale, damping, damping_ratio_scale = (
        _make_foot_contact_material_cfg(compliance_cfg)
    )
    material_path = f"{prim_path}/footContactComplianceMaterial"

    stage = sim_utils.get_current_stage()
    link_names = list(compliance_cfg.get("link_names", A3_FOOT_CONTACT_LINK_NAMES))
    num_links = _bind_foot_contact_material(
        stage.GetPrimAtPath(prim_path), material_path, material_cfg, link_names
    )

    logger.info(
        "Bound A3 foot compliant-contact material to %d links: stiffness=%.3f"
        " (scale=%.4f), damping=%.3f (ratio_scale=%.4f)",
        num_links,
        stiffness,
        stiffness_scale,
        damping,
        damping_ratio_scale,
    )
    return prim


def spawn_a3_urdf_with_per_env_foot_contact_compliance(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn A3 from URDF and bind a separately sampled foot material to each robot."""
    prim = sim_utils.spawn_from_urdf(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )

    compliance_cfg = getattr(cfg, "foot_contact_compliance", None)
    if not compliance_cfg or not compliance_cfg.get("enabled", False):
        return prim

    stage = sim_utils.get_current_stage()
    robot_paths = sorted(sim_utils.find_matching_prim_paths(prim_path))
    if not robot_paths:
        robot_paths = [prim.GetPath().pathString]

    link_names = list(compliance_cfg.get("link_names", A3_FOOT_CONTACT_LINK_NAMES))
    stiffness_values = []
    damping_values = []
    total_links = 0
    for robot_path in robot_paths:
        material_cfg, stiffness, _, damping, _ = _make_foot_contact_material_cfg(compliance_cfg)
        material_path = f"{robot_path}/footContactComplianceMaterial"
        total_links += _bind_foot_contact_material(
            stage.GetPrimAtPath(robot_path), material_path, material_cfg, link_names
        )
        stiffness_values.append(stiffness)
        damping_values.append(damping)

    logger.info(
        "Bound A3 per-env foot compliant-contact materials to %d links across %d robots:"
        " stiffness=[%.3f, %.3f], damping=[%.3f, %.3f]",
        total_links,
        len(robot_paths),
        min(stiffness_values),
        max(stiffness_values),
        min(damping_values),
        max(damping_values),
    )
    return prim


A3_CYLINDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/a3/model.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.09),
        joint_pos={
            "left_hip_pitch_joint": -0.1311,
            "right_hip_pitch_joint": -0.1311,
            "left_hip_roll_joint": 0.0056,
            "right_hip_roll_joint": -0.0056,
            "left_hip_yaw_joint": -0.0348,
            "right_hip_yaw_joint": 0.0348,
            "left_knee_joint": 0.2468,
            "right_knee_joint": 0.2468,
            "left_ankle_pitch_joint": -0.1204,
            "right_ankle_pitch_joint": -0.1204,
            "left_ankle_roll_joint": -0.0078,
            "right_ankle_roll_joint": 0.0078,
            # Waist.
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # Arms.
            "left_shoulder_pitch_joint": 0.3,
            "right_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.12,
            "right_shoulder_roll_joint": -0.12,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.8,
            "right_elbow_joint": 0.8,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # Fixed wrist joints (not in URDF, excluded from robot)
            "left_wrist_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 220.0,
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 320.0,
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 12.042771838760874,
                ".*_hip_yaw_joint": 12.042771838760874,
                ".*_hip_pitch_joint": 12.042771838760874,
                ".*_knee_joint": 14.660765716752367,
            },
            stiffness={
                ".*_hip_roll_joint": 120,
                ".*_hip_yaw_joint": 80,
                ".*_hip_pitch_joint": 80,
                ".*_knee_joint": 250.0,
            },
            damping={
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_hip_pitch_joint": 3.0,
                ".*_knee_joint": 8.0,
            },
            armature={
                ".*_hip_roll_joint": ARMATURE_PFP_93_65,
                ".*_hip_yaw_joint": ARMATURE_PFP_93_65,
                ".*_hip_pitch_joint": ARMATURE_PFP_93_65,
                ".*_knee_joint": ARMATURE_PFP_110_75,
            },
            friction={
                ".*_hip_roll_joint": 0.15,
                ".*_hip_yaw_joint": 0.15,
                ".*_hip_pitch_joint": 0.15,
                ".*_knee_joint": 0.15,
            },
            dynamic_friction={
                ".*_hip_roll_joint": 0.1,
                ".*_hip_yaw_joint": 0.1,
                ".*_hip_pitch_joint": 0.1,
                ".*_knee_joint": 0.1,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 118.2,  # 115,
                ".*_ankle_roll_joint": 54.75,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 10.8,
                ".*_ankle_roll_joint": 19.37,
            },
            stiffness={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature={
                ".*_ankle_pitch_joint": ARMATURE_PFP_78_58 * 5.333,
                ".*_ankle_roll_joint": ARMATURE_PFP_78_58 * 1.66562,
            },
            friction={
                ".*_ankle_pitch_joint": 0.25,
                ".*_ankle_roll_joint": 0.25,
            },
            dynamic_friction={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.2,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 220.0,
                "waist_roll_joint": 46.0,
                "waist_pitch_joint": 115,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 12.042771838760874,
                "waist_roll_joint": 22.7,
                "waist_pitch_joint": 9.24785,
            },
            stiffness={
                "waist_yaw_joint": 85,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            damping={
                "waist_yaw_joint": 3.0,
                "waist_roll_joint": 2.0,
                "waist_pitch_joint": 2.0,
            },
            armature={
                "waist_yaw_joint": ARMATURE_PFP_93_65,
                "waist_roll_joint": ARMATURE_PFP_78_58 * 1.21,
                "waist_pitch_joint": ARMATURE_PFP_78_58 * 7.3,
            },
            friction={
                "waist_yaw_joint": 0.15,
                "waist_roll_joint": 0.15,
                "waist_pitch_joint": 0.15,
            },
            dynamic_friction={
                "waist_yaw_joint": 0.1,
                "waist_roll_joint": 0.1,
                "waist_pitch_joint": 0.1,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_roll_joint": 24.0,
                ".*_wrist_pitch_joint": 6.0,
                ".*_wrist_yaw_joint": 6.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.613568165555769,
                ".*_shoulder_roll_joint": 13.613568165555769,
                ".*_shoulder_yaw_joint": 15.707963267948966,
                ".*_elbow_joint": 15.707963267948966,
                ".*_wrist_roll_joint": 15.707963267948966,
                ".*_wrist_pitch_joint": 12.775810124598491,
                ".*_wrist_yaw_joint": 12.775810124598491,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 30.0,
                ".*_elbow_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_yaw_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_PFP_78_58,
                ".*_shoulder_roll_joint": ARMATURE_PFP_78_58,
                ".*_shoulder_yaw_joint": ARMATURE_PFP_59_60,
                ".*_elbow_joint": ARMATURE_PFP_59_60,
                ".*_wrist_roll_joint": ARMATURE_PFP_59_60,
                ".*_wrist_pitch_joint": ARMATURE_PFP_41_48,
                ".*_wrist_yaw_joint": ARMATURE_PFP_41_48,
            },
            friction={
                ".*_shoulder_pitch_joint": 0.1,
                ".*_shoulder_roll_joint": 0.1,
                ".*_shoulder_yaw_joint": 0.1,
                ".*_elbow_joint": 0.1,
                ".*_wrist_roll_joint": 0.1,
                ".*_wrist_pitch_joint": 0.1,
                ".*_wrist_yaw_joint": 0.1,
            },
            dynamic_friction={
                ".*_shoulder_pitch_joint": 0.1,
                ".*_shoulder_roll_joint": 0.1,
                ".*_shoulder_yaw_joint": 0.1,
                ".*_elbow_joint": 0.1,
                ".*_wrist_roll_joint": 0.1,
                ".*_wrist_pitch_joint": 0.1,
                ".*_wrist_yaw_joint": 0.1,
            },
        ),
    },
)

A3_ISAACLAB_JOINTS = [
    "pelvis_link",
    "left_hip_pitch_Link",
    "right_hip_pitch_Link",
    "waist_yaw_Link",
    "left_hip_roll_Link",
    "right_hip_roll_Link",
    "waist_roll_Link",
    "left_hip_yaw_Link",
    "right_hip_yaw_Link",
    "torso_Link",
    "left_knee_Link",
    "right_knee_Link",
    "left_shoulder_pitch_Link",
    "right_shoulder_pitch_Link",
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
    "left_shoulder_yaw_Link",
    "right_shoulder_yaw_Link",
    "left_elbow_Link",
    "right_elbow_Link",
    "left_wrist_roll_Link",
    "right_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "right_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
]

# Gather-index convention (matching G1/H2):
#   X_TO_Y[output_pos_in_Y] = input_idx_in_X
#   Usage: data_in_X[..., X_TO_Y] -> data_in_Y

A3_ISAACLAB_TO_MUJOCO_DOF = [
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
]

A3_MUJOCO_TO_ISAACLAB_DOF = [
    19,
    25,
    0,
    20,
    26,
    1,
    21,
    27,
    2,
    22,
    28,
    5,
    12,
    23,
    29,
    6,
    13,
    24,
    30,
    7,
    14,
    8,
    15,
    9,
    16,
    10,
    17,
    11,
    18,
]

A3_ISAACLAB_TO_MUJOCO_BODY = [
    0,
    3,
    6,
    9,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    13,
    17,
    21,
    23,
    25,
    27,
    29,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
]

A3_MUJOCO_TO_ISAACLAB_BODY = [
    0,
    20,
    26,
    1,
    21,
    27,
    2,
    22,
    28,
    3,
    23,
    29,
    6,
    13,
    24,
    30,
    7,
    14,
    25,
    31,
    8,
    15,
    9,
    16,
    10,
    17,
    11,
    18,
    12,
    19,
]

A3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": A3_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": A3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": A3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": A3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": A3_MUJOCO_TO_ISAACLAB_BODY,
}

A3_ACTION_SCALE = {}
for a in A3_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            A3_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
