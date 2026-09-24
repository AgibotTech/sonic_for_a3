from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.utils.a3_motor_model import (
    ParallelJacobianTable,
    apply_parallel_motor_limits,
    get_tn_curve,
    interpolate_tn_limit,
)


def test_pfp78_curve_uses_60nm_plateau_and_zero_torque_extrapolation():
    curve = get_tn_curve("PFP78", dtype=torch.float64)
    speeds = torch.tensor(
        [0.0, 16.0, 18.84955592153876, curve.speed_rad_s[-1].item()],
        dtype=torch.float64,
    )

    limits = interpolate_tn_limit(speeds, curve)

    assert limits[0].item() == pytest.approx(60.0)
    assert limits[1].item() == pytest.approx(60.0)
    assert limits[2].item() == pytest.approx(44.31, abs=0.05)
    assert limits[3].item() == pytest.approx(0.0)


def test_parallel_limit_clips_each_motor_then_maps_back_to_joint_space():
    jacobian = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    result = apply_parallel_motor_limits(
        jacobian=jacobian,
        joint_velocity=torch.zeros(1, 2, dtype=torch.float64),
        requested_joint_torque=torch.tensor([[80.0, -30.0]], dtype=torch.float64),
        motor_family="PFP78",
    )

    torch.testing.assert_close(
        result.requested_motor_torque,
        torch.tensor([[80.0, -30.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.applied_motor_torque,
        torch.tensor([[60.0, -30.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.applied_joint_torque,
        torch.tensor([[60.0, -30.0]], dtype=torch.float64),
    )
    assert result.saturated.tolist() == [[True, False]]


def test_parallel_mapping_preserves_virtual_power_and_applied_motor_feasibility():
    jacobian = torch.tensor(
        [[[0.55, 0.31], [0.42, -0.64]]],
        dtype=torch.float64,
    )
    joint_velocity = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    requested_joint_torque = torch.tensor([[160.0, 70.0]], dtype=torch.float64)

    result = apply_parallel_motor_limits(
        jacobian=jacobian,
        joint_velocity=joint_velocity,
        requested_joint_torque=requested_joint_torque,
        motor_family="PFP78",
    )

    joint_power = torch.sum(requested_joint_torque * joint_velocity, dim=-1)
    motor_power = torch.sum(
        result.requested_motor_torque * result.motor_velocity,
        dim=-1,
    )
    torch.testing.assert_close(joint_power, motor_power, rtol=1.0e-12, atol=1.0e-12)
    assert torch.all(
        torch.abs(result.applied_motor_torque) <= result.motor_torque_limit + 1.0e-12
    )
    torch.testing.assert_close(
        torch.einsum("...ji,...j->...i", jacobian, result.applied_joint_torque),
        result.applied_motor_torque,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("table_name", ["left_ankle", "right_ankle", "waist"])
def test_parallel_jacobian_table_returns_exact_grid_values(table_name: str):
    table = ParallelJacobianTable.load(table_name, dtype=torch.float64)
    pitch_idx = table.pitch_axis.numel() // 2
    roll_idx = table.roll_axis.numel() // 2
    position = torch.stack(
        (table.pitch_axis[pitch_idx], table.roll_axis[roll_idx])
    ).unsqueeze(0)

    interpolated = table.interpolate(position)

    torch.testing.assert_close(
        interpolated[0],
        table.jacobian[pitch_idx, roll_idx],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert bool(table.reachable[pitch_idx, roll_idx])


def test_parallel_model_matches_solver_idyn_and_dik_at_table_grid_points():
    from gear_sonic.scripts.sim2sim_a3_mujoco import (
        DEFAULT_BRIDGE_CACHE,
        DEFAULT_SOLVER_INCLUDE,
        DEFAULT_SOLVER_LIB,
        A3LoopSolverBridge,
    )

    cases = [
        ("left_ankle", 0),
        ("right_ankle", 1),
        ("waist", None),
    ]
    torque = np.array([37.0, -19.0], dtype=np.float64)
    velocity = np.array([2.5, -1.25], dtype=np.float64)
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
