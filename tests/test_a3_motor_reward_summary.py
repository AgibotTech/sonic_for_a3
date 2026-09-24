from __future__ import annotations

import csv
import json

import pytest

from gear_sonic.scripts.summarize_a3_motor_reward_eval import summarize_dataset


def test_reward_curve_summary_recovers_per_step_penalty_from_dt_weighted_columns(
    tmp_path,
):
    dataset_dir = tmp_path / "good5"
    curve_dir = dataset_dir / "metrics" / "reward_curves"
    curve_dir.mkdir(parents=True)
    curve_path = curve_dir / "000000_walk_reward_curve.csv"
    columns = [
        "step",
        "time_s",
        "total_reward",
        "tracking",
        "action_rate",
        "motor_tn_serial_soft_diag",
        "motor_tn_serial_hard_diag",
        "motor_tn_parallel_soft_diag",
        "motor_tn_parallel_hard_diag",
        "ankle_motor_speed_diag",
        "ankle_motor_power_diag",
        "ankle_motor_fixed_torque_diag",
        "parallel_ankle_cross_diag",
        "parallel_waist_cross_diag",
        "ankle_position_85_diag",
    ]
    rows = [
        [
            0,
            0.02,
            0.058,
            0.10,
            -0.002,
            -0.02,
            -0.01,
            -0.006,
            -0.004,
            -0.002,
            -0.004,
            0.0,
            -0.0002,
            0.0,
            0.0,
        ],
        [
            1,
            0.04,
            0.068,
            0.08,
            -0.002,
            0.0,
            0.0,
            -0.004,
            -0.006,
            0.0,
            -0.002,
            -0.006,
            0.0,
            0.0,
            -0.008,
        ],
    ]
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)

    metrics = {
        "reward_curves": [
            {
                "motion_key": "walk",
                "num_steps": 2,
                "csv": "/remote/path/000000_walk_reward_curve.csv",
            }
        ],
        "eval/all_metrics_dict": {
            "motion_keys": ["walk"],
            "terminated": [False],
        },
        "eval/success/success_rate": 1.0,
        "eval/success/progress_rate": 1.0,
        "eval/success/mpjpe_l": 20.0,
        "eval/success/mpjpe_g": 200.0,
        "eval/success/mpjpe_pa": 10.0,
    }
    with (dataset_dir / "metrics" / "metrics_eval.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metrics, stream)

    summary = summarize_dataset(dataset_dir, "good5")

    assert summary["motion_count"] == 1
    assert summary["frame_count"] == 2
    assert summary["step_dt"] == pytest.approx(0.02)
    assert summary["serial"]["penalty_per_step"]["mean"] == pytest.approx(0.75)
    assert summary["parallel"]["penalty_per_step"]["mean"] == pytest.approx(0.5)
    assert summary["serial"]["soft_nonzero_rate"] == pytest.approx(0.5)
    assert summary["serial"]["hard_trigger_rate"] == pytest.approx(0.5)
    assert summary["parallel"]["soft_nonzero_rate"] == pytest.approx(1.0)
    assert summary["parallel"]["hard_trigger_rate"] == pytest.approx(1.0)
    assert summary["baseline_reward"]["positive_mean_per_step"] == pytest.approx(4.5)
    assert summary["baseline_reward"]["negative_magnitude_mean_per_step"] == pytest.approx(0.1)
    assert summary["constraint_terms"]["ankle_motor_speed"]["penalty_per_step"][
        "mean"
    ] == pytest.approx(0.05)
    assert summary["constraint_terms"]["ankle_motor_speed"][
        "nonzero_rate"
    ] == pytest.approx(0.5)
    assert summary["constraint_terms"]["ankle_position_85"]["penalty_per_step"][
        "mean"
    ] == pytest.approx(0.2)
    assert summary["top_motions"][0]["motion_key"] == "walk"
    assert summary["top_motions"][0]["combined_mean_per_step"] == pytest.approx(
        1.805
    )
    assert summary["candidate_multiplier_impact"][2]["multiplier"] == pytest.approx(
        1.0
    )
    assert summary["candidate_multiplier_impact"][2][
        "penalty_mean_per_step"
    ] == pytest.approx(1.805)
