"""Compact tensor indexes used by adaptive motion sampling.

The legacy implementation materialized a dataset-global frame-to-bin tensor.
For large motion unions that tensor is several GiB per rank even though a bin
can be resolved from two small per-motion arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompactAdaptiveSamplingIndex:
    """Motion-major adaptive bin metadata stored on one torch device."""

    num_frames: torch.Tensor
    num_bins_per_motion: torch.Tensor
    motion_bin_starts: torch.Tensor
    bins: torch.Tensor
    bin_motion_ids: torch.Tensor
    bin_lengths: torch.Tensor
    bin_new_motion_mask: torch.Tensor
    num_peer_bins: torch.Tensor
    motion_bin_ids: torch.Tensor | None
    total_frames: int
    num_bins: int


def expand_contiguous_ranges(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Concatenate ``arange(start, start + length)`` without Python loops."""
    if starts.ndim != 1 or lengths.ndim != 1 or starts.shape != lengths.shape:
        raise ValueError("starts and lengths must be same-shaped one-dimensional tensors")
    if starts.dtype != torch.long or lengths.dtype != torch.long:
        raise ValueError("starts and lengths must use torch.long")
    if bool((lengths < 0).any().item()):
        raise ValueError("range lengths must be non-negative")
    total = int(lengths.sum().item())
    if total == 0:
        return torch.empty(0, device=starts.device, dtype=torch.long)

    output_offsets = lengths.cumsum(0) - lengths
    range_bases = torch.repeat_interleave(starts - output_offsets, lengths)
    return range_bases + torch.arange(total, device=starts.device, dtype=torch.long)


def build_compact_adaptive_index(
    num_frames: torch.Tensor,
    bin_size: int,
    *,
    build_paired: bool = False,
) -> CompactAdaptiveSamplingIndex:
    """Build stable motion-major bins from per-motion frame counts."""
    if num_frames.ndim != 1 or num_frames.dtype != torch.long:
        raise ValueError("num_frames must be a one-dimensional torch.long tensor")
    if len(num_frames) == 0:
        raise ValueError("num_frames must contain at least one motion")
    if bool((num_frames <= 0).any().item()):
        raise ValueError("every motion must contain at least one frame")
    if int(bin_size) <= 0:
        raise ValueError("bin_size must be positive")

    bin_size = int(bin_size)
    num_bins_per_motion = torch.div(
        num_frames + bin_size - 1,
        bin_size,
        rounding_mode="floor",
    )
    motion_bin_starts = num_bins_per_motion.cumsum(0) - num_bins_per_motion
    num_bins = int(num_bins_per_motion.sum().item())
    total_frames = int(num_frames.sum().item())

    bin_ids = torch.arange(num_bins, device=num_frames.device, dtype=torch.long)
    bin_motion_ids = torch.repeat_interleave(
        torch.arange(len(num_frames), device=num_frames.device, dtype=torch.long),
        num_bins_per_motion,
    )
    local_bin_ids = bin_ids - motion_bin_starts[bin_motion_ids]
    bin_starts = local_bin_ids * bin_size
    bin_ends = torch.minimum(bin_starts + bin_size, num_frames[bin_motion_ids])
    bins = torch.stack((bin_motion_ids, bin_starts, bin_ends), dim=1)
    # Reuse the first bins column as the retained motion lookup instead of
    # keeping the repeat_interleave allocation alive as a duplicate tensor.
    bin_motion_ids = bins[:, 0]
    bin_lengths = bin_ends - bin_starts
    bin_new_motion_mask = local_bin_ids == 0
    num_peer_bins = num_bins_per_motion[bin_motion_ids]

    motion_bin_ids = None
    if build_paired:
        max_bins_per_motion = int(num_bins_per_motion.max().item())
        local_offsets = torch.arange(
            max_bins_per_motion, device=num_frames.device, dtype=torch.long
        )
        motion_bin_ids = motion_bin_starts[:, None] + local_offsets[None, :]
        motion_bin_ids = motion_bin_ids.masked_fill(
            local_offsets[None, :] >= num_bins_per_motion[:, None], -1
        )

    return CompactAdaptiveSamplingIndex(
        num_frames=num_frames,
        num_bins_per_motion=num_bins_per_motion,
        motion_bin_starts=motion_bin_starts,
        bins=bins,
        bin_motion_ids=bin_motion_ids,
        bin_lengths=bin_lengths,
        bin_new_motion_mask=bin_new_motion_mask,
        num_peer_bins=num_peer_bins,
        motion_bin_ids=motion_bin_ids,
        total_frames=total_frames,
        num_bins=num_bins,
    )


def map_motion_frames_to_bins(
    motion_ids: torch.Tensor,
    motion_time_steps: torch.Tensor,
    motion_bin_starts: torch.Tensor,
    num_frames: torch.Tensor,
    bin_size: int,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Map dataset motion IDs and local frame indices directly to global bins."""
    if motion_ids.shape != motion_time_steps.shape:
        raise ValueError("motion_ids and motion_time_steps must have matching shapes")
    motion_ids = motion_ids.long()
    motion_time_steps = motion_time_steps.long()
    if validate and motion_ids.numel() > 0:
        invalid_motion = (motion_ids < 0) | (motion_ids >= len(num_frames))
        if bool(invalid_motion.any().item()):
            raise ValueError("motion ID outside dataset bounds")
        invalid_step = (motion_time_steps < 0) | (
            motion_time_steps >= num_frames[motion_ids]
        )
        if bool(invalid_step.any().item()):
            raise ValueError("motion time step outside motion bounds")
    return motion_bin_starts[motion_ids] + torch.div(
        motion_time_steps, int(bin_size), rounding_mode="floor"
    )


def sparse_add_bin_counts_(
    target: torch.Tensor,
    bin_ids: torch.Tensor,
    *,
    divisors: torch.Tensor | None = None,
    multiplier: float = 1.0,
) -> None:
    """Accumulate counts only for bins present in ``bin_ids``."""
    if bin_ids.numel() == 0:
        return
    unique_bin_ids, counts = torch.unique(bin_ids.long(), return_counts=True)
    values = counts.to(dtype=target.dtype)
    if divisors is not None:
        values = values / divisors[unique_bin_ids].to(dtype=target.dtype)
    if multiplier != 1.0:
        values = values * float(multiplier)
    target.index_add_(0, unique_bin_ids, values)


def aggregate_bin_values_by_motion(
    values: torch.Tensor,
    bin_motion_ids: torch.Tensor,
    num_motions: int,
) -> torch.Tensor:
    """Sum motion-major bin values into one value per source motion."""
    if values.ndim != 1 or bin_motion_ids.ndim != 1 or values.shape != bin_motion_ids.shape:
        raise ValueError("values and bin_motion_ids must be same-shaped vectors")
    result = torch.zeros(int(num_motions), device=values.device, dtype=values.dtype)
    result.index_add_(0, bin_motion_ids, values)
    return result
