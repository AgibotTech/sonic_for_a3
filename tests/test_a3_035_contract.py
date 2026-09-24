from __future__ import annotations

import ast
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
A3_ROBOT = REPO_ROOT / "gear_sonic/envs/manager_env/robots/a3.py"
CONFIG_DIR = REPO_ROOT / "gear_sonic/config"
LAUNCHER = (
    REPO_ROOT
    / "train_a3_035_fromscratch.sh"
)
EXP_CONFIG = (
    "manager/universal_token/all_modes/"
    "sonic_a3_035"
)


def _find_update_mapping(
    tree: ast.Module, object_name: str, attribute_name: str
) -> dict[str, float]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 1:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "update"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == attribute_name
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == object_name
        ):
            continue
        return ast.literal_eval(node.args[0])
    raise AssertionError(f"missing {object_name}.{attribute_name}.update")


def test_035_robot_contract_sets_release_ankle_damping():
    tree = ast.parse(A3_ROBOT.read_text())
    damping = _find_update_mapping(tree, "_a3_024_feet", "damping")
    assert damping == {
        ".*_ankle_pitch_joint": pytest.approx(1.2),
        ".*_ankle_roll_joint": pytest.approx(1.2),
    }

    source = A3_ROBOT.read_text()
    assert "A3_024_CYLINDER_CFG = copy.deepcopy(A3_CYLINDER_CFG)" in source
    assert "A3_024_ACTION_SCALE = _build_action_scale(A3_024_CYLINDER_CFG)" in source


def test_035_composes_as_dual_encoder_with_ten_real_frames():
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base", overrides=[f"+exp={EXP_CONFIG}"])

    backbone = cfg.algo.config.actor.backbone
    motion = cfg.manager_env.commands.motion
    vr_reward = cfg.manager_env.rewards.tracking_vr_5point_local
    foot_orientation = cfg.manager_env.rewards.tracking_relative_body_ori_weighted

    assert cfg.manager_env.config.robot.type == "a3_024"
    assert cfg.algo.config.compute_aux_loss is True
    assert cfg.algo.config.actor.has_aux_loss is True
    assert OmegaConf.to_container(backbone.active_encoders) == ["g1", "a3_fast"]
    assert OmegaConf.to_container(backbone.active_decoders) == ["g1_dyn", "g1_kin"]
    assert list(backbone.aux_loss_func) == [
        "g1_recon",
        "a3_fast_g1_latent",
        "reencoded_a3_fast_g1_latent",
    ]
    assert backbone.aux_loss_coef.g1_recon == pytest.approx(0.01)
    assert backbone.aux_loss_coef.a3_fast_g1_latent == pytest.approx(1.0)
    assert backbone.aux_loss_coef.reencoded_a3_fast_g1_latent == pytest.approx(0.0)
    assert backbone.reencode_a3_fast_g1_recon is False
    assert backbone.encoders.a3_fast.init_from_encoder.source == "g1"
    assert backbone.encoders.a3_fast.init_from_encoder.policy == "missing_only"

    assert motion.a3_fast_num_future_frames == 10
    assert motion.a3_fast_dt_future_ref_frames == pytest.approx(0.02)
    assert motion.a3_fast_valid_future_frames is None
    assert motion.a3_fast_zero_pad_invalid_frames is False
    assert motion.coactivate_g1_when_a3_fast is True
    assert OmegaConf.to_container(motion.encoder_sample_probs) == {
        "g1": 1.0,
        "teleop": 0.0,
        "smpl": 0.0,
        "a3_fast": 1.0,
    }
    assert motion.motion_lib_cfg.adaptive_sampling.report_full_csv is False

    assert cfg.manager_env.events.joint_soft_pos_limits.params.factor == pytest.approx(0.8)
    assert vr_reward.weight == pytest.approx(3.0)
    assert OmegaConf.to_container(motion.reward_point_body) == [
        "torso_Link",
        "left_wrist_yaw_Link",
        "right_wrist_yaw_Link",
    ]
    assert foot_orientation is None


def test_035_public_launcher_supports_from_scratch_and_warm_start():
    script = LAUNCHER.read_text()
    assert "model_step_200000.pt" not in script
    assert "EXPECTED_CHECKPOINT" not in script
    assert 'CHECKPOINT="${CHECKPOINT:-}"' in script
    assert "--checkpoint PATH" in script
    assert 'MOTION_FILE="${MOTION_FILE:-}"' in script
    assert "MOTION_FILE inputs" in script
    # Build the markers so public-source privacy scans do not mistake this
    # regression assertion for a machine-local path.
    assert "/" + "home/" not in script
    assert "/" + "mnt/" not in script
    assert 'checkpoint="${CHECKPOINT:-null}"' in script
    assert "+resume=false" in script
    assert "+resume=true" not in script
    assert 'export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"' in script
    assert 'export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"' in script
