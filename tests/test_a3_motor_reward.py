from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

import gear_sonic.utils.a3_motor_model as motor_model
from gear_sonic.utils.a3_motor_model import (
    A3MotorConstraintCache,
    ParallelJacobianTable,
    compute_cross_penalty,
    compute_motor_penalty,
    compute_position_threshold_penalty,
    compute_threshold_penalty,
    get_tn_curve,
    interpolate_tn_limit,
    map_parallel_joint_to_motor,
    motor_family_for_joint,
    peak_mechanical_power,
    solve_parallel_motor_velocity_2x2,
)

TWO_SEGMENT_CURVES = {
    "PFP41": ((0.0, 20.943951, 26.179939), (5.8, 5.8, 0.0)),
    "PFP59": ((0.0, 13.613568, 17.0), (34.0, 34.0, 0.0)),
    "PFP78": ((0.0, 16.755161, 23.561945), (58.0, 58.0, 0.0)),
    "PFP93": ((0.0, 10.471976, 18.5), (215.0, 215.0, 0.0)),
    "PFP110": ((0.0, 9.424778, 17.3), (300.0, 300.0, 0.0)),
}
CONFIG_DIR = Path(__file__).resolve().parents[1] / "gear_sonic" / "config"
TRAIN_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "train_a3_035_fromscratch.sh"
)


@pytest.mark.parametrize(("family", "expected"), TWO_SEGMENT_CURVES.items())
def test_tn_curves_have_one_plateau_and_one_linear_decline(family, expected):
    expected_speed, expected_torque = expected
    curve = get_tn_curve(family, dtype=torch.float64, profile="reward_024")

    torch.testing.assert_close(
        curve.speed_rad_s,
        torch.tensor(expected_speed, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        curve.torque_nm,
        torch.tensor(expected_torque, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )

    midpoint = 0.5 * (curve.speed_rad_s[1] + curve.speed_rad_s[2])
    limits = interpolate_tn_limit(
        torch.stack((curve.speed_rad_s[1], midpoint, curve.speed_rad_s[2])),
        curve,
    )
    torch.testing.assert_close(
        limits,
        torch.tensor(
            [expected_torque[1], 0.5 * expected_torque[1], 0.0],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_motor_penalty_matches_dense_eighth_power_tn_shaping():
    utilization = torch.tensor(
        [0.0, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 3.0],
        dtype=torch.float64,
    )

    penalty = compute_motor_penalty(
        utilization,
        lambda_soft=0.05,
        lambda_violation=5.0,
        soft_exponent=8.0,
        violation_exponent=2.0,
        utilization_cap=2.0,
    )

    torch.testing.assert_close(
        penalty.soft,
        torch.tensor(
            [0.0, 0.5**8, 0.7**8, 0.9**8, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        penalty.hard,
        torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.25, 1.0],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        penalty.combined,
        0.05 * penalty.soft + 5.0 * penalty.hard,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_threshold_penalty_starts_at_80_percent_and_caps_at_two():
    utilization = torch.tensor(
        [0.7, 0.8, 0.9, 1.0, 1.1, 3.0],
        dtype=torch.float64,
    )

    penalty = compute_threshold_penalty(
        utilization,
        threshold_ratio=0.8,
        lambda_soft=1.0,
        lambda_violation=5.0,
        soft_exponent=2.0,
        violation_exponent=2.0,
        utilization_cap=2.0,
    )

    torch.testing.assert_close(
        penalty.soft,
        torch.tensor([0.0, 0.0, 0.01, 0.04, 0.09, 1.44], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        penalty.hard,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.01, 1.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        penalty.combined,
        penalty.soft + 5.0 * penalty.hard,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_parallel_cross_penalty_requires_both_axes_above_threshold():
    normalized_pair_load = torch.tensor(
        [[0.9, 0.9], [0.8, 0.7], [1.2, 1.0]],
        dtype=torch.float64,
    )

    penalty = compute_cross_penalty(normalized_pair_load, threshold_ratio=0.75)

    torch.testing.assert_close(
        penalty,
        torch.tensor([0.0225, 0.0, 0.1125], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_position_penalty_uses_85_percent_of_hard_joint_range():
    joint_pos = torch.tensor([[0.0, 0.9], [-1.0, 0.8]], dtype=torch.float64)
    hard_limits = torch.tensor(
        [[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]],
        dtype=torch.float64,
    )

    penalty = compute_position_threshold_penalty(
        joint_pos,
        hard_limits,
        threshold_ratio=0.85,
        exponent=2.0,
    )

    torch.testing.assert_close(
        penalty,
        torch.tensor([0.05**2, 0.15**2], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_pfp78_peak_power_comes_from_plateau_endpoint():
    curve = get_tn_curve("PFP78", dtype=torch.float64, profile="reward_024")
    torch.testing.assert_close(
        peak_mechanical_power(curve),
        torch.tensor(16.755161 * 58.0, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_eval_diagnostics_match_notes_final_reward_weights():
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base_eval_motor_reward")

    rewards = cfg.manager_env.rewards
    for group in ("serial", "parallel"):
        soft = rewards[f"motor_tn_{group}_soft_diag"]
        hard = rewards[f"motor_tn_{group}_hard_diag"]
        assert soft.weight == pytest.approx(-0.05 * 0.05)
        assert hard.weight == pytest.approx(-0.05 * 5.0)
        assert soft.params.lambda_soft == hard.params.lambda_soft == 0.05
        assert soft.params.lambda_violation == hard.params.lambda_violation == 5.0
        assert soft.params.soft_exponent == hard.params.soft_exponent == 8.0
        assert soft.params.violation_exponent == hard.params.violation_exponent == 2.0
        assert soft.params.utilization_cap == hard.params.utilization_cap == 2.0


def test_full_motor_constraint_eval_config_matches_selected_weights():
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base_eval_motor_constraints")

    rewards = cfg.manager_env.rewards
    assert rewards.motor_tn_serial_soft_diag.weight == pytest.approx(-0.05)
    assert rewards.motor_tn_serial_hard_diag.weight == pytest.approx(-5.0)
    assert rewards.motor_tn_parallel_soft_diag.weight == pytest.approx(-0.05)
    assert rewards.motor_tn_parallel_hard_diag.weight == pytest.approx(-5.0)
    assert rewards.ankle_motor_speed_diag.weight == pytest.approx(-2.0)
    assert rewards.ankle_motor_power_diag.weight == pytest.approx(-2.0)
    assert rewards.ankle_motor_fixed_torque_diag.weight == pytest.approx(-1.0)
    assert rewards.parallel_ankle_cross_diag.weight == pytest.approx(-0.1)
    assert rewards.parallel_waist_cross_diag.weight == pytest.approx(-0.05)
    assert rewards.ankle_position_85_diag.weight == pytest.approx(-10.0)


def test_035_training_config_adds_motor_constraints():
    exp_config = "manager/universal_token/all_modes/sonic_a3_035"
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="base", overrides=[f"+exp={exp_config}"])

    assert cfg.exp_var == "a3_035_vr3_no_ankleori_g1_a3fast_f10"
    assert cfg.manager_env.config.robot.type == "a3_024"
    assert cfg.manager_env.commands.motion.a3_fast_num_future_frames == 10
    assert list(cfg.algo.config.actor.backbone.active_encoders) == ["g1", "a3_fast"]

    rewards = cfg.manager_env.rewards
    assert rewards.motor_tn_serial.weight == pytest.approx(-1.0)
    assert rewards.motor_tn_parallel.weight == pytest.approx(-1.0)
    assert rewards.ankle_motor_speed.weight == pytest.approx(-2.0)
    assert rewards.ankle_motor_power.weight == pytest.approx(-2.0)
    assert rewards.ankle_motor_fixed_torque.weight == pytest.approx(-1.0)
    assert rewards.parallel_ankle_cross.weight == pytest.approx(-0.1)
    assert rewards.parallel_waist_cross.weight == pytest.approx(-0.05)
    assert rewards.ankle_position_85.weight == pytest.approx(-10.0)


def test_035_launcher_uses_weight_warm_start_without_full_state_resume():
    source = TRAIN_LAUNCHER.read_text()

    assert "manager/universal_token/all_modes/sonic_a3_035" in source
    assert 'CHECKPOINT="${CHECKPOINT:-}"' in source
    assert 'checkpoint="${CHECKPOINT:-null}"' in source
    assert "+resume=false" in source
    assert "+resume=true" not in source


@pytest.mark.parametrize(
    ("joint_name", "family"),
    [
        ("left_knee_joint", "PFP110"),
        ("right_hip_pitch_joint", "PFP93"),
        ("waist_yaw_joint", "PFP93"),
        ("left_shoulder_roll_joint", "PFP78"),
        ("right_elbow_joint", "PFP59"),
        ("left_wrist_roll_joint", "PFP59"),
        ("right_wrist_pitch_joint", "PFP41"),
        ("left_wrist_yaw_joint", "PFP41"),
        ("left_ankle_pitch_joint", None),
        ("waist_roll_joint", None),
        ("left_foot_toe_joint", None),
    ],
)
def test_serial_joint_motor_family_mapping(joint_name, family):
    assert motor_family_for_joint(joint_name) == family


def test_parallel_mapping_preserves_virtual_power():
    jacobian = torch.tensor(
        [[[0.55, 0.31], [0.42, -0.64]]],
        dtype=torch.float64,
    )
    joint_velocity = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    joint_torque = torch.tensor([[160.0, 70.0]], dtype=torch.float64)

    result = map_parallel_joint_to_motor(
        jacobian=jacobian,
        joint_velocity=joint_velocity,
        joint_torque=joint_torque,
        motor_family="PFP78",
    )

    joint_power = torch.sum(joint_torque * joint_velocity, dim=-1)
    motor_power = torch.sum(result.motor_torque * result.motor_velocity, dim=-1)
    torch.testing.assert_close(joint_power, motor_power, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        result.utilization,
        torch.abs(result.motor_torque) / result.torque_limit,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1.0e-5, 1.0e-5),
        (torch.float64, 1.0e-12, 1.0e-12),
    ],
)
@pytest.mark.parametrize("table_name", ["left_ankle", "right_ankle", "waist"])
def test_analytic_2x2_velocity_solve_matches_torch_linalg(
    table_name,
    dtype,
    rtol,
    atol,
):
    table = ParallelJacobianTable.load(table_name, dtype=dtype)
    positions = torch.stack(
        (
            torch.linspace(table.pitch_axis[1], table.pitch_axis[-2], 17),
            torch.linspace(table.roll_axis[-2], table.roll_axis[1], 17),
        ),
        dim=-1,
    )
    jacobian = table.interpolate(positions)
    generator = torch.Generator().manual_seed(20260730)
    joint_velocity = torch.randn((17, 2), generator=generator, dtype=dtype)

    expected = torch.linalg.solve(
        jacobian,
        joint_velocity.unsqueeze(-1),
    ).squeeze(-1)
    actual = solve_parallel_motor_velocity_2x2(jacobian, joint_velocity)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("table_name", ["left_ankle", "right_ankle", "waist"])
def test_parallel_jacobian_table_returns_exact_grid_values(table_name):
    table = ParallelJacobianTable.load(table_name, dtype=torch.float64)
    pitch_idx = table.pitch_axis.numel() // 2
    roll_idx = table.roll_axis.numel() // 2
    position = torch.stack((table.pitch_axis[pitch_idx], table.roll_axis[roll_idx])).unsqueeze(0)

    interpolated = table.interpolate(position)

    torch.testing.assert_close(
        interpolated[0],
        table.jacobian[pitch_idx, roll_idx],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert bool(table.reachable[pitch_idx, roll_idx])


def test_parallel_model_matches_packaged_x86_a3_loop_solver():
    from gear_sonic.scripts.sim2sim_a3_mujoco import (
        DEFAULT_BRIDGE_CACHE,
        DEFAULT_SOLVER_INCLUDE,
        DEFAULT_SOLVER_LIB,
        A3LoopSolverBridge,
    )

    torque = np.array([37.0, -19.0], dtype=np.float64)
    velocity = np.array([2.5, -1.25], dtype=np.float64)
    cases = [("left_ankle", 0), ("right_ankle", 1), ("waist", None)]
    solver = A3LoopSolverBridge(
        DEFAULT_SOLVER_LIB,
        DEFAULT_SOLVER_INCLUDE,
        DEFAULT_BRIDGE_CACHE,
    )
    try:
        for table_name, leg in cases:
            table = ParallelJacobianTable.load(table_name, dtype=torch.float64)
            pitch_idx = table.pitch_axis.numel() // 2
            roll_idx = table.roll_axis.numel() // 2
            position = np.array(
                [
                    table.pitch_axis[pitch_idx].item(),
                    table.roll_axis[roll_idx].item(),
                ],
                dtype=np.float64,
            )
            jacobian = table.jacobian[pitch_idx, roll_idx].numpy()

            if leg is None:
                solver.waist_fk(solver.waist_ik(position))
                expected_torque = solver.waist_idyn(torque)
                expected_velocity = solver.waist_dik(velocity)
            else:
                solver.ankle_fk(leg, solver.ankle_ik(leg, position))
                expected_torque = solver.ankle_idyn(leg, torque)
                expected_velocity = solver.ankle_dik(leg, velocity)

            np.testing.assert_allclose(jacobian.T @ torque, expected_torque, atol=1.0e-8)
            np.testing.assert_allclose(
                np.linalg.solve(jacobian, velocity),
                expected_velocity,
                atol=1.0e-8,
            )
    finally:
        solver.close()


_CACHE_TEST_JOINT_NAMES = (
    "left_knee_joint",
    "left_hip_pitch_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)


def _cache_test_inputs(num_envs: int = 8):
    generator = torch.Generator().manual_seed(20260730)
    num_joints = len(_CACHE_TEST_JOINT_NAMES)
    joint_position = torch.zeros((num_envs, num_joints), dtype=torch.float64)
    joint_velocity = torch.randn(
        (num_envs, num_joints),
        generator=generator,
        dtype=torch.float64,
    )
    joint_torque = 20.0 * torch.randn(
        (num_envs, num_joints),
        generator=generator,
        dtype=torch.float64,
    )
    return joint_position, joint_velocity, joint_torque


def test_motor_constraint_cache_matches_uncached_motor_values():
    joint_position, joint_velocity, joint_torque = _cache_test_inputs()
    cache = A3MotorConstraintCache(
        _CACHE_TEST_JOINT_NAMES,
        device="cpu",
        dtype=torch.float64,
    )
    cache.begin_step(
        17,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_torque=joint_torque,
    )

    expected_serial = []
    for family, indices in cache.serial_family_indices.items():
        curve = get_tn_curve(family, dtype=torch.float64, profile="reward_024")
        limit = interpolate_tn_limit(joint_velocity[:, indices], curve)
        expected_serial.append(torch.abs(joint_torque[:, indices]) / limit)
    torch.testing.assert_close(
        cache.serial_utilization(),
        torch.cat(expected_serial, dim=-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    expected_parallel = []
    for table_name, indices in cache.parallel_pair_indices.items():
        table = ParallelJacobianTable.load(table_name, dtype=torch.float64)
        expected = map_parallel_joint_to_motor(
            jacobian=table.interpolate(joint_position[:, indices]),
            joint_velocity=joint_velocity[:, indices],
            joint_torque=joint_torque[:, indices],
            motor_family="PFP78",
            curve=get_tn_curve("PFP78", dtype=torch.float64, profile="reward_024"),
        )
        expected_parallel.append(expected.utilization)
        torch.testing.assert_close(
            cache.parallel_motor_torque(table_name),
            expected.motor_torque,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            cache.parallel_motor_velocity(table_name),
            expected.motor_velocity,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    torch.testing.assert_close(
        cache.parallel_utilization(),
        torch.cat(expected_parallel, dim=-1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_motor_constraint_cache_reuses_parallel_mapping_within_one_step(monkeypatch):
    joint_position, joint_velocity, joint_torque = _cache_test_inputs()
    counters = {"jacobian": 0, "solve": 0, "tn": 0}
    original_interpolate = ParallelJacobianTable.interpolate
    original_solve = motor_model.solve_parallel_motor_velocity_2x2
    original_tn = motor_model.interpolate_tn_limit

    def counted_interpolate(self, position_pr):
        counters["jacobian"] += 1
        return original_interpolate(self, position_pr)

    def counted_solve(jacobian, velocity):
        counters["solve"] += 1
        return original_solve(jacobian, velocity)

    def counted_tn(speed, curve):
        counters["tn"] += 1
        return original_tn(speed, curve)

    monkeypatch.setattr(ParallelJacobianTable, "interpolate", counted_interpolate)
    monkeypatch.setattr(motor_model, "solve_parallel_motor_velocity_2x2", counted_solve)
    monkeypatch.setattr(motor_model, "interpolate_tn_limit", counted_tn)

    cache = A3MotorConstraintCache(
        _CACHE_TEST_JOINT_NAMES,
        device="cpu",
        dtype=torch.float64,
    )
    cache.begin_step(
        23,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_torque=joint_torque,
    )
    cache.parallel_utilization()
    cache.parallel_motor_velocity("left_ankle")
    cache.parallel_motor_torque("left_ankle")
    cache.parallel_utilization()
    cache.serial_utilization()
    cache.serial_utilization()

    assert counters == {
        "jacobian": 3,
        "solve": 3,
        "tn": len(cache.serial_family_indices) + 3,
    }

    cache.begin_step(
        24,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_torque=joint_torque,
    )
    cache.parallel_utilization()
    cache.serial_utilization()
    assert counters == {
        "jacobian": 6,
        "solve": 6,
        "tn": 2 * (len(cache.serial_family_indices) + 3),
    }


def test_motor_constraint_cache_fixed_torque_path_skips_velocity_and_tn(monkeypatch):
    joint_position, joint_velocity, joint_torque = _cache_test_inputs()
    counters = {"solve": 0, "tn": 0}
    original_solve = motor_model.solve_parallel_motor_velocity_2x2
    original_tn = motor_model.interpolate_tn_limit

    def counted_solve(jacobian, velocity):
        counters["solve"] += 1
        return original_solve(jacobian, velocity)

    def counted_tn(speed, curve):
        counters["tn"] += 1
        return original_tn(speed, curve)

    monkeypatch.setattr(motor_model, "solve_parallel_motor_velocity_2x2", counted_solve)
    monkeypatch.setattr(motor_model, "interpolate_tn_limit", counted_tn)

    cache = A3MotorConstraintCache(
        _CACHE_TEST_JOINT_NAMES,
        device="cpu",
        dtype=torch.float64,
    )
    cache.begin_step(
        29,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        joint_torque=joint_torque,
    )
    cache.parallel_motor_torque("left_ankle")
    cache.parallel_motor_torque("right_ankle")

    assert counters == {"solve": 0, "tn": 0}
