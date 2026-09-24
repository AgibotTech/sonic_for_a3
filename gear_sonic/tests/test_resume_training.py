"""Tests for full-state resume output-directory selection."""

from omegaconf import OmegaConf
import pytest

from gear_sonic.train_agent_trl import resume_training


def test_resume_into_new_directory_preserves_requested_output(tmp_path):
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    checkpoint = source_dir / "last.pt"
    checkpoint.touch()
    output_dir = tmp_path / "resume_run"
    config = OmegaConf.create(
        {
            "checkpoint": str(checkpoint),
            "experiment_dir": str(output_dir),
            "resume_in_place": False,
        }
    )

    resume_training(config)

    assert config.checkpoint == str(checkpoint)
    assert config.experiment_dir == str(output_dir)


def test_resume_defaults_to_source_checkpoint_directory(tmp_path):
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    checkpoint = source_dir / "last.pt"
    checkpoint.touch()
    config = OmegaConf.create(
        {
            "checkpoint": str(checkpoint),
            "experiment_dir": str(tmp_path / "ignored_output"),
        }
    )

    resume_training(config)

    assert config.experiment_dir == str(source_dir)


def test_new_directory_resume_rejects_source_directory(tmp_path):
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    checkpoint = source_dir / "last.pt"
    checkpoint.touch()
    config = OmegaConf.create(
        {
            "checkpoint": str(checkpoint),
            "experiment_dir": str(source_dir),
            "resume_in_place": False,
        }
    )

    with pytest.raises(ValueError, match="differ from the checkpoint directory"):
        resume_training(config)
