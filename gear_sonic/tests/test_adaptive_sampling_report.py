"""Adaptive-sampling report tests with a lightweight MotionLib stub."""

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.trl.callbacks.model_save_callback import ModelSaveCallback
from gear_sonic.utils.adaptive_sampling_index import build_compact_adaptive_index
from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase


def _make_motion_lib(*, full_csv: bool):
    index = build_compact_adaptive_index(torch.tensor([49, 51, 101]), 50)
    sampling_prob = torch.tensor([0.10, 0.40, 0.05, 0.20, 0.15, 0.10])
    motion_prob = torch.tensor([0.10, 0.45, 0.45])
    motion_lib = SimpleNamespace(
        use_adaptive_sampling=True,
        adaptive_sampling_cfg={
            "report_top_k": 2,
            "report_full_csv": full_csv,
        },
        update_adaptive_sampling_motion_sequences=lambda: None,
        _curr_motion_ids=torch.tensor([1]),
        _num_unique_motions=3,
        _motion_data_keys=["motion_a", "motion_b", "motion_c"],
        _motion_data_load={
            "motion_a": {"path": "/data/a.pkl"},
            "motion_b": {"path": "/data/b.pkl"},
            "motion_c": {"path": "/data/c.pkl"},
        },
        adp_samp_bins=index.bins,
        adp_sampling_prob=sampling_prob,
        adp_samp_failure_rate=torch.linspace(0.1, 0.6, 6),
        adp_samp_failure_rate_raw=torch.linspace(0.2, 0.7, 6),
        adp_samp_num_episodes=torch.arange(1, 7, dtype=torch.float32),
        adp_samp_num_failures=torch.arange(6, dtype=torch.float32),
        _sampling_prob=motion_prob,
        adp_samp_motion_bin_starts=index.motion_bin_starts,
        adp_samp_num_bins_per_motion=index.num_bins_per_motion,
        adp_samp_num_frames=index.num_frames,
        adp_samp_num_bins=index.num_bins,
        adp_samp_bin_size=50,
        uniform_sampling_rate=0.1,
    )
    return motion_lib


def test_summary_only_report_does_not_materialize_full_rows():
    report = MotionLibBase.get_adaptive_sampling_report(
        _make_motion_lib(full_csv=False)
    )

    assert report["rows"] == []
    assert report["motions"] == []
    assert report["summary"]["full_csv_enabled"] is False
    assert [row["bin_id"] for row in report["summary"]["top_bins"]] == [1, 3]
    assert report["summary"]["top_bins"][0]["motion_id"] == 1
    assert report["summary"]["top_bins"][0]["segment_index"] == 0
    assert {row["motion_id"] for row in report["summary"]["top_motions"]} == {1, 2}


def test_full_report_preserves_bin_and_motion_csv_rows():
    report = MotionLibBase.get_adaptive_sampling_report(
        _make_motion_lib(full_csv=True)
    )

    assert len(report["rows"]) == 6
    assert len(report["motions"]) == 3
    assert report["summary"]["full_csv_enabled"] is True
    assert [row["bin_id"] for row in report["summary"]["top_bins"]] == [1, 3]
    assert report["motions"][1]["num_bins"] == 2
    assert report["motions"][2]["max_bin_sampling_prob"] == pytest.approx(0.20)


def test_legacy_checkpoint_sampler_state_loads_without_index_conversion():
    sync_calls = []
    motion_lib = SimpleNamespace(
        use_adaptive_sampling=True,
        adp_samp_num_episodes=torch.zeros(6),
        adp_samp_num_failures=torch.zeros(6),
        adp_samp_num_bins=6,
        _device=torch.device("cpu"),
        sync_and_compute_adaptive_sampling=lambda **kwargs: sync_calls.append(kwargs),
    )
    checkpoint_state = {
        "adp_samp_num_episodes": torch.arange(1, 7, dtype=torch.float32),
        "adp_samp_num_failures": torch.arange(6, dtype=torch.float32),
    }

    MotionLibBase.load_state_dict(motion_lib, checkpoint_state)

    torch.testing.assert_close(
        motion_lib.adp_samp_num_episodes,
        checkpoint_state["adp_samp_num_episodes"],
    )
    torch.testing.assert_close(
        motion_lib.adp_samp_num_failures,
        checkpoint_state["adp_samp_num_failures"],
    )
    assert sync_calls == [{"sync_across_gpus": False}]


def test_sampler_state_with_different_bin_count_is_not_loaded():
    motion_lib = SimpleNamespace(
        use_adaptive_sampling=True,
        adp_samp_num_episodes=torch.ones(6),
        adp_samp_num_failures=torch.ones(6),
        adp_samp_num_bins=6,
        _device=torch.device("cpu"),
        sync_and_compute_adaptive_sampling=lambda **kwargs: pytest.fail(
            "mismatched state must not be synchronized"
        ),
    )

    MotionLibBase.load_state_dict(
        motion_lib,
        {
            "adp_samp_num_episodes": torch.zeros(5),
            "adp_samp_num_failures": torch.zeros(5),
        },
    )

    torch.testing.assert_close(motion_lib.adp_samp_num_episodes, torch.ones(6))
    torch.testing.assert_close(motion_lib.adp_samp_num_failures, torch.ones(6))


def test_summary_only_callback_writes_no_full_csv(tmp_path):
    callback = ModelSaveCallback(tmp_path)
    report = MotionLibBase.get_adaptive_sampling_report(
        _make_motion_lib(full_csv=False)
    )

    callback.save_adaptive_sampling_report(report, global_step=2000)

    report_dir = tmp_path / "adaptive_sampling_stats"
    assert (report_dir / "adaptive_sampling_summary_step_002000.json").is_file()
    assert not (report_dir / "adaptive_sampling_bins_step_002000.csv").exists()
    assert not (report_dir / "adaptive_sampling_motions_step_002000.csv").exists()
