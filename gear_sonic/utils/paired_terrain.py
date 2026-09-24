"""Validation and distributed rank selection for paired terrain training."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def rank_uses_terrain(spec, local_rank=None):
    if spec is None:
        return True
    period = int(spec.get("period", 1))
    if period <= 0:
        raise ValueError(f"rank terrain period must be positive, got {period}")
    remainders = {int(value) for value in spec.get("terrain_remainders", [0])}
    if any(value < 0 or value >= period for value in remainders):
        raise ValueError(
            f"terrain_remainders must be in [0, {period}), got {sorted(remainders)}")
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return int(local_rank) % period in remainders


def usd_stems(path):
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Paired object USD directory does not exist: {root}")
    files = sorted(root.glob("*.usd"))
    if not files:
        raise FileNotFoundError(f"No paired object USD files found under: {root}")
    return [file.stem for file in files]


def validate_paired_stems(motion_keys, object_keys, object_usd_path):
    motion = [str(key) for key in motion_keys]
    objects = {str(key) for key in object_keys}
    usd = usd_stems(object_usd_path)
    if len(motion) != len(set(motion)):
        raise ValueError("Paired terrain motion keys contain duplicates")
    if set(motion) != objects:
        missing = sorted(set(motion) - objects)[:10]
        extra = sorted(objects - set(motion))[:10]
        raise ValueError(
            f"Paired object motion stems do not match robot motions; missing={missing}, extra={extra}")
    if motion != usd:
        missing = sorted(set(motion) - set(usd))[:10]
        extra = sorted(set(usd) - set(motion))[:10]
        if missing or extra:
            raise ValueError(
                f"Paired USD stems do not match robot motions; missing={missing}, extra={extra}")
        raise ValueError(
            "Paired robot motions and USD files contain the same stems but use different order")
    return motion


def condition_bin_probabilities_by_motion(global_probabilities, motion_bin_ids):
    """Normalize adaptive bin probabilities independently for each paired motion."""
    if global_probabilities.ndim != 1:
        raise ValueError("global_probabilities must be one-dimensional")
    if motion_bin_ids.ndim != 2:
        raise ValueError("motion_bin_ids must be a padded two-dimensional tensor")
    valid = motion_bin_ids >= 0
    safe_bin_ids = motion_bin_ids.clamp_min(0)
    probabilities = global_probabilities[safe_bin_ids] * valid
    row_sums = probabilities.sum(dim=1, keepdim=True)
    uniform = valid.to(probabilities.dtype) / valid.sum(dim=1, keepdim=True)
    normalized = probabilities / row_sums.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return torch.where(row_sums > 0, normalized, uniform)


def sample_paired_motion_time_steps(
    dataset_motion_ids,
    motion_bin_ids,
    conditioned_probabilities,
    bins,
    pre_failure_sample_window=0,
    generator=None,
):
    """Sample adaptive time bins without changing a scene-locked motion ID."""
    if dataset_motion_ids.numel() == 0:
        return torch.empty_like(dataset_motion_ids, dtype=torch.int32)

    probabilities = conditioned_probabilities[dataset_motion_ids]
    local_bin_ids = torch.multinomial(
        probabilities, num_samples=1, replacement=True, generator=generator
    ).squeeze(1)
    selected_bin_ids = motion_bin_ids[dataset_motion_ids, local_bin_ids]

    selected_bins = bins[selected_bin_ids]
    bin_start = selected_bins[:, 1]
    bin_end = selected_bins[:, 2]
    widths = bin_end - bin_start

    time_steps = (
        torch.rand(
            len(bin_start),
            device=bin_start.device,
            generator=generator,
        )
        * widths
    ).floor().long() + bin_start
    if pre_failure_sample_window > 0:
        offsets = torch.randint(
            int(pre_failure_sample_window),
            (len(time_steps),),
            device=time_steps.device,
            generator=generator,
        )
        time_steps = (time_steps - offsets).clamp_min(0)
    return time_steps.int()
