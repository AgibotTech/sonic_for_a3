import copy

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"


ARMATURE_PFP_78_58 = 30.118e-6 * 20.03 * 20.03
ARMATURE_PFP_93_65 = 138.5069e-6 * 21.906 * 21.906
ARMATURE_PFP_41_48 = 1.359e-6 * 24.415 * 24.415
ARMATURE_PFP_59_60 = 10.1374e-6 * 22.136 * 22.136
ARMATURE_PFP_110_75 = 300.851e-6 * 20 * 20

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

A3_PASSIVE_FOOT_JOINT_NAMES = [
    "left_foot_forefoot_joint",
    "left_foot_toe_joint",
    "right_foot_forefoot_joint",
    "right_foot_toe_joint",
]


def _build_action_scale(robot_cfg) -> dict[str, float]:
    """Derive the position-action scale from each actuator's effort and stiffness."""
    action_scale: dict[str, float] = {}
    for actuator in robot_cfg.actuators.values():
        effort = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        names = actuator.joint_names_expr
        if not isinstance(effort, dict):
            effort = dict.fromkeys(names, effort)
        if not isinstance(stiffness, dict):
            stiffness = dict.fromkeys(names, stiffness)
        for name in names:
            if "foot_forefoot_joint" in name or "foot_toe_joint" in name:
                continue
            if name in effort and name in stiffness and stiffness[name]:
                action_scale[name] = 0.25 * effort[name] / stiffness[name]
    return action_scale

A3_CYLINDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf",
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
            # Passive compliant-foot joints. These are physics-only and not policy outputs.
            "left_foot_forefoot_joint": 0.0,
            "left_foot_toe_joint": 0.0,
            "right_foot_forefoot_joint": 0.0,
            "right_foot_toe_joint": 0.0,
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
        "passive_foot_springs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_foot_forefoot_joint",
                ".*_foot_toe_joint",
            ],
            effort_limit_sim={
                ".*_foot_forefoot_joint": 400.0,
                ".*_foot_toe_joint": 100.0,
            },
            velocity_limit_sim={
                ".*_foot_forefoot_joint": 20.0,
                ".*_foot_toe_joint": 20.0,
            },
            stiffness={
                ".*_foot_forefoot_joint": 2396.666,
                ".*_foot_toe_joint": 27.58407,
            },
            damping={
                ".*_foot_forefoot_joint": 1.51683722,
                ".*_foot_toe_joint": 0.0686125693,
            },
            armature={
                ".*_foot_forefoot_joint": 1.0e-4,
                ".*_foot_toe_joint": 3.0e-5,
            },
            friction=0.0,
            dynamic_friction=0.0,
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

# The 024 contract keeps the fitted passive-foot asset and applies its policy
# gains/effort limits explicitly.  It is kept separate from the base A3
# configuration so an exported 024 checkpoint reconstructs the same action
# scale and simulation contract.
A3_024_CYLINDER_CFG = copy.deepcopy(A3_CYLINDER_CFG)

_a3_024_legs = A3_024_CYLINDER_CFG.actuators["legs"]
_a3_024_legs.effort_limit_sim = dict(_a3_024_legs.effort_limit_sim)
_a3_024_legs.effort_limit_sim[".*_knee_joint"] = 300.0

_a3_024_feet = A3_024_CYLINDER_CFG.actuators["feet"]
_a3_024_feet.stiffness = dict(_a3_024_feet.stiffness)
_a3_024_feet.damping = dict(_a3_024_feet.damping)
_a3_024_feet.stiffness.update(
    {".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 60.0}
)

_a3_024_waist = A3_024_CYLINDER_CFG.actuators["waist"]
_a3_024_waist.effort_limit_sim = dict(_a3_024_waist.effort_limit_sim)
_a3_024_waist.effort_limit_sim.update(
    {
        "waist_yaw_joint": 200.0,
        "waist_roll_joint": 42.0,
        "waist_pitch_joint": 105.0,
    }
)

# The lower ankle damping is part of the released 024 contract.
_a3_024_feet.damping.update(
    {
        ".*_ankle_pitch_joint": 1.2,
        ".*_ankle_roll_joint": 1.2,
    }
)

# Backward compatibility for a released 024 PT that carries its historical
# serialized robot type.  New configuration and documentation use `a3_024`.
A3_EXP020_CYLINDER_CFG = copy.deepcopy(A3_024_CYLINDER_CFG)
_exp020_feet = A3_EXP020_CYLINDER_CFG.actuators["feet"]
_exp020_feet.damping = dict(_exp020_feet.damping)
_exp020_feet.damping.update(
    {
        ".*_ankle_pitch_joint": 1.5,
        ".*_ankle_roll_joint": 1.5,
    }
)
A3_EXP023_CYLINDER_CFG = A3_024_CYLINDER_CFG

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

A3_ACTION_SCALE = _build_action_scale(A3_CYLINDER_CFG)
A3_024_ACTION_SCALE = _build_action_scale(A3_024_CYLINDER_CFG)
A3_EXP020_ACTION_SCALE = _build_action_scale(A3_EXP020_CYLINDER_CFG)
A3_EXP023_ACTION_SCALE = A3_024_ACTION_SCALE
