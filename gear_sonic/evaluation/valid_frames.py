"""Shared valid-frame rules for Isaac evaluation trajectories."""

from __future__ import annotations

import torch


def valid_eval_frame_mask(
    curr_step: int,
    motion_num_steps: torch.Tensor,
    terminated: torch.Tensor,
    terminating_now: torch.Tensor,
) -> torch.Tensor:
    """Return envs whose current post-step data belongs to the evaluated episode."""
    return (
        (curr_step < motion_num_steps - 1)
        & ~terminated.bool()
        & ~terminating_now.bool()
    )
