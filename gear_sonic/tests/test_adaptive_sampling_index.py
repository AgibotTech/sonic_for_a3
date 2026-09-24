"""Unit tests for compact adaptive-sampling indexes."""

import pytest
import torch

from gear_sonic.utils.adaptive_sampling_index import (
    aggregate_bin_values_by_motion,
    build_compact_adaptive_index,
    expand_contiguous_ranges,
    map_motion_frames_to_bins,
    sparse_add_bin_counts_,
)


def _legacy_index(num_frames: torch.Tensor, bin_size: int):
    bins = []
    frame_to_bin = []
    motion_to_bins = []
    next_bin = 0
    for motion_id, frames in enumerate(num_frames.tolist()):
        local_bins = []
        local_frame_to_bin = []
        for start in range(0, frames, bin_size):
            end = min(start + bin_size, frames)
            bins.append([motion_id, start, end])
            local_bins.append(next_bin)
            local_frame_to_bin.extend([next_bin] * (end - start))
            next_bin += 1
        motion_to_bins.append(local_bins)
        frame_to_bin.append(local_frame_to_bin)
    return (
        torch.tensor(bins, dtype=torch.long),
        motion_to_bins,
        frame_to_bin,
    )


@pytest.mark.parametrize(
    "frames",
    [
        [1],
        [49, 50, 51],
        [99, 100, 101, 149, 150, 151],
        [7, 63, 128, 203],
    ],
)
def test_compact_index_preserves_legacy_motion_major_bin_order(frames):
    num_frames = torch.tensor(frames, dtype=torch.long)
    legacy_bins, legacy_motion_to_bins, legacy_frame_to_bin = _legacy_index(
        num_frames, 50
    )

    index = build_compact_adaptive_index(num_frames, 50)

    torch.testing.assert_close(index.bins, legacy_bins)
    assert index.num_bins == len(legacy_bins)
    assert index.total_frames == sum(frames)

    for motion_id, expected_bins in enumerate(legacy_motion_to_bins):
        start = int(index.motion_bin_starts[motion_id])
        count = int(index.num_bins_per_motion[motion_id])
        assert list(range(start, start + count)) == expected_bins

        motion_ids = torch.full((frames[motion_id],), motion_id, dtype=torch.long)
        time_steps = torch.arange(frames[motion_id], dtype=torch.long)
        actual = map_motion_frames_to_bins(
            motion_ids,
            time_steps,
            index.motion_bin_starts,
            num_frames,
            50,
        )
        torch.testing.assert_close(
            actual, torch.tensor(legacy_frame_to_bin[motion_id], dtype=torch.long)
        )


def test_map_motion_frames_rejects_out_of_range_steps():
    num_frames = torch.tensor([10, 20], dtype=torch.long)
    index = build_compact_adaptive_index(num_frames, 5)

    with pytest.raises(ValueError, match="outside motion bounds"):
        map_motion_frames_to_bins(
            torch.tensor([0, 1]),
            torch.tensor([10, 3]),
            index.motion_bin_starts,
            num_frames,
            5,
        )


def test_sparse_updates_match_dense_bincount_reference():
    bin_ids = torch.tensor([0, 3, 3, 3, 7, 7], dtype=torch.long)
    bin_lengths = torch.tensor([50, 50, 50, 25, 50, 50, 50, 10], dtype=torch.float32)
    dense = torch.bincount(bin_ids, minlength=8).float() / bin_lengths
    compact = torch.zeros(8)

    sparse_add_bin_counts_(compact, bin_ids, divisors=bin_lengths)

    torch.testing.assert_close(compact, dense)

    failure = torch.tensor([False, True, True, False, True, True])
    dense_failures = torch.bincount(bin_ids[failure], minlength=8).float() * 2.5
    compact_failures = torch.zeros(8)
    sparse_add_bin_counts_(compact_failures, bin_ids[failure], multiplier=2.5)
    torch.testing.assert_close(compact_failures, dense_failures)


def test_sparse_updates_accept_empty_input():
    target = torch.ones(4)
    sparse_add_bin_counts_(target, torch.empty(0, dtype=torch.long))
    torch.testing.assert_close(target, torch.ones(4))


def test_contiguous_range_expansion_and_motion_aggregation_match_legacy_loops():
    starts = torch.tensor([0, 4, 9, 10])
    lengths = torch.tensor([2, 3, 1, 2])
    torch.testing.assert_close(
        expand_contiguous_ranges(starts, lengths),
        torch.tensor([0, 1, 4, 5, 6, 9, 10, 11]),
    )

    frames = torch.tensor([49, 51, 100, 101])
    index = build_compact_adaptive_index(frames, 50)
    values = torch.linspace(0.1, 1.0, index.num_bins, dtype=torch.float64)
    actual = aggregate_bin_values_by_motion(
        values, index.bin_motion_ids, len(frames)
    )
    expected = torch.tensor(
        [
            values[
                int(index.motion_bin_starts[motion_id]) : int(
                    index.motion_bin_starts[motion_id]
                    + index.num_bins_per_motion[motion_id]
                )
            ].sum()
            for motion_id in range(len(frames))
        ],
        dtype=values.dtype,
    )
    torch.testing.assert_close(actual, expected)


def test_paired_matrix_is_built_only_when_requested():
    num_frames = torch.tensor([49, 51, 101], dtype=torch.long)

    compact = build_compact_adaptive_index(num_frames, 50, build_paired=False)
    paired = build_compact_adaptive_index(num_frames, 50, build_paired=True)

    assert compact.motion_bin_ids is None
    assert paired.motion_bin_ids is not None
    torch.testing.assert_close(
        paired.motion_bin_ids,
        torch.tensor([[0, -1, -1], [1, 2, -1], [3, 4, 5]]),
    )


def test_actual_021_shape_eliminates_more_than_two_gib_of_frame_indexes():
    total_frames = 252_808_055
    legacy_frame_index_bytes = total_frames * torch.tensor([], dtype=torch.long).element_size()
    legacy_start_mask_bytes = total_frames * torch.tensor([], dtype=torch.bool).element_size()

    assert legacy_frame_index_bytes + legacy_start_mask_bytes > 2 * 1024**3
