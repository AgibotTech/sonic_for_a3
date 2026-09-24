import numpy as np
import torch

from gear_sonic.trl.utils import logger_utils


def test_report_to_normalization_and_trainer_filter():
    assert logger_utils.normalize_report_to("none") == []
    assert logger_utils.normalize_report_to("tensorboard") == ["tensorboard"]
    assert logger_utils.normalize_report_to("[tensorboard,wandb]") == [
        "tensorboard",
        "wandb",
    ]
    assert logger_utils.trainer_report_to(["tensorboard", "wandb"]) == "none"
    assert logger_utils.trainer_report_to("wandb") == "none"


def test_tensorboard_train_logs_use_grouped_tags_and_drop_env_duplicates():
    logs = {
        "objective/rewards": 10.0,
        "loss/policy_avg": 0.1,
        "policy/approxkl_avg": 0.01,
        "val/ratio": 1.0,
        "Env/tracking_anchor_pos": 0.5,
        "tracking_anchor_pos": 0.5,
        "Env/Episode_Reward/tracking_anchor_pos": 0.4,
        "Episode_Reward/tracking_anchor_pos": 0.4,
        "Env/Episode_Termination/anchor_pos": 0.02,
        "Episode_Termination/anchor_pos": 0.02,
        "Env/adp_samp/prob_max": 0.3,
        "Episode/length": 120,
        "scheduled_params/push": 0.2,
        "fps": 1234,
        "experiment_save_dir": "/tmp/run",
    }

    grouped = logger_utils.tensorboard_train_logs(logs)

    assert grouped["Objective/rewards"] == 10.0
    assert grouped["Loss/policy_avg"] == 0.1
    assert grouped["Policy/approxkl_avg"] == 0.01
    assert grouped["Value/ratio"] == 1.0
    assert grouped["Env/tracking_anchor_pos"] == 0.5
    assert "Misc/tracking_anchor_pos" not in grouped
    assert grouped["Rewards/tracking_anchor_pos"] == 0.4
    assert "Misc/Episode_Reward/tracking_anchor_pos" not in grouped
    assert grouped["Terminations/anchor_pos"] == 0.02
    assert "Misc/Episode_Termination/anchor_pos" not in grouped
    assert grouped["AdaptiveSampling/prob_max"] == 0.3
    assert grouped["Episode/length"] == 120
    assert grouped["Scheduled/push"] == 0.2
    assert grouped["Train/fps"] == 1234


def test_filter_scalar_logs_drops_non_scalars():
    logs = {
        "loss": 1.25,
        "np_loss": np.float32(2.5),
        "tensor_loss": torch.tensor(3.5),
        "path": "/tmp/run",
        "hist": [1, 2, 3],
        "enabled": True,
    }

    filtered = logger_utils.filter_scalar_logs(logs)

    assert filtered == {
        "loss": 1.25,
        "np_loss": 2.5,
        "tensor_loss": 3.5,
    }


def test_tensorboard_writer_logs_scalars_text_and_video_fallback(tmp_path):
    writer = logger_utils.create_summary_writer(tmp_path)
    logger_utils.write_tensorboard_values(
        writer,
        {
            "eval/success": 0.75,
            "eval/all_metrics_dict": {
                "motion_keys": ["m0"],
                "terminated": [False],
                "mpjpe": [0.12],
            },
        },
        step=4,
    )
    logger_utils.add_video_or_text(writer, "videos_hard/0000", tmp_path / "missing.mp4", step=4)
    writer.close()

    assert list(tmp_path.glob("events.out.tfevents.*"))
