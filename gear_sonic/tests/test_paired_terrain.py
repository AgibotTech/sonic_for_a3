"""Tests for scene-locked terrain pairing without importing IsaacLab."""

import pytest
import torch

from gear_sonic.utils.paired_terrain import (
    condition_bin_probabilities_by_motion,
    rank_uses_terrain,
    sample_paired_motion_time_steps,
    validate_paired_stems,
)


def test_four_rank_partition_is_exactly_three_flat_to_one_terrain():
    spec = {"period": 4, "terrain_remainders": [3]}

    assert [rank_uses_terrain(spec, rank) for rank in range(4)] == [False, False, False, True]


def test_paired_stems_require_identical_sorted_order(tmp_path):
    for stem in ("a", "b"):
        (tmp_path / f"{stem}.usd").touch()

    assert validate_paired_stems(["a", "b"], {"a", "b"}, tmp_path) == ["a", "b"]

    with pytest.raises(ValueError, match="different order"):
        validate_paired_stems(["b", "a"], {"a", "b"}, tmp_path)

    with pytest.raises(ValueError, match="object motion stems"):
        validate_paired_stems(["a", "b"], {"a"}, tmp_path)


def test_paired_adaptive_probabilities_are_normalized_per_motion():
    global_probabilities = torch.tensor([0.1, 0.3, 0.6, 0.0])
    motion_bin_ids = torch.tensor([[0, 1], [2, -1], [3, -1]])

    conditioned = condition_bin_probabilities_by_motion(
        global_probabilities, motion_bin_ids
    )

    torch.testing.assert_close(conditioned[0], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(conditioned[1], torch.tensor([1.0, 0.0]))
    # A zero-probability row falls back to uniform over its valid bins.
    torch.testing.assert_close(conditioned[2], torch.tensor([1.0, 0.0]))


def test_paired_adaptive_sampling_keeps_motion_id_and_terrain_pairing():
    bins = torch.tensor(
        [
            [0, 0, 10],
            [0, 10, 20],
            [1, 30, 40],
        ]
    )
    motion_bin_ids = torch.tensor([[0, 1], [2, -1]])
    conditioned = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    motion_ids = torch.tensor([0, 1, 0, 1])
    generator = torch.Generator().manual_seed(7)

    time_steps = sample_paired_motion_time_steps(
        motion_ids,
        motion_bin_ids,
        conditioned,
        bins,
        generator=generator,
    )

    motion_0_steps = time_steps[motion_ids == 0]
    motion_1_steps = time_steps[motion_ids == 1]
    assert torch.all((motion_0_steps >= 10) & (motion_0_steps < 20))
    assert torch.all((motion_1_steps >= 30) & (motion_1_steps < 40))
