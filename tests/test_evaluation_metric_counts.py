import numpy as np

from gear_sonic.evaluation.aggregator import aggregate_batches
from gear_sonic.evaluation.metrics import compute_metric_sums_and_counts


def _batch_with_metric_counts():
    return {
        "schema_version": 1,
        "run_id": "test-run",
        "rank": 0,
        "batch_idx": 0,
        "motion_idx": np.asarray([0, 1]),
        "terminated": np.asarray([False, False]),
        "progress": np.asarray([1.0, 1.0]),
        "length_per_motion": np.asarray([20, 10]),
        "sums_by_metric": {
            "mpjpe_g": np.asarray([20.0, 10.0]),
            "vel_dist": np.asarray([19.0, 9.0]),
            "accel_dist": np.asarray([18.0, 8.0]),
        },
        "counts_by_metric": {
            "mpjpe_g": np.asarray([20, 10]),
            "vel_dist": np.asarray([19, 9]),
            "accel_dist": np.asarray([18, 8]),
        },
    }


def test_aggregator_uses_each_metrics_own_sample_count():
    result = aggregate_batches(
        [_batch_with_metric_counts()],
        expected_motion_keys=("motion-0", "motion-1"),
    )

    assert result.metrics_all["mpjpe_g"] == 1.0
    assert result.metrics_all["vel_dist"] == 1.0
    assert result.metrics_all["accel_dist"] == 1.0
    np.testing.assert_allclose(result.all_metrics_dict["vel_dist"], [1.0, 1.0])
    np.testing.assert_allclose(result.all_metrics_dict["accel_dist"], [1.0, 1.0])


def test_aggregator_corrects_known_derivative_counts_in_legacy_payloads():
    batch = _batch_with_metric_counts()
    del batch["counts_by_metric"]

    result = aggregate_batches(
        [batch],
        expected_motion_keys=("motion-0", "motion-1"),
    )

    assert result.metrics_all["mpjpe_g"] == 1.0
    assert result.metrics_all["vel_dist"] == 1.0
    assert result.metrics_all["accel_dist"] == 1.0


def test_metric_counts_follow_each_returned_trajectory_length():
    sums, counts = compute_metric_sums_and_counts(
        {
            "mpjpe_g": [np.ones((20, 14))],
            "vel_dist": [np.ones(19)],
            "accel_dist": [np.ones(18)],
        }
    )

    assert sums["mpjpe_g"].tolist() == [20.0]
    assert sums["vel_dist"].tolist() == [19.0]
    assert sums["accel_dist"].tolist() == [18.0]
    assert counts["mpjpe_g"].tolist() == [20]
    assert counts["vel_dist"].tolist() == [19]
    assert counts["accel_dist"].tolist() == [18]
