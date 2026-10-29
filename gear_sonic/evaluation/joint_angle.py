"""Joint-angle metrics shared by training rewards and evaluation callbacks."""

from __future__ import annotations

import torch


def shortest_angular_difference(
    reference: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    """Return signed ``reference - prediction`` wrapped to [-pi, pi]."""
    difference = reference - prediction
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def resolve_reference_and_robot_joint_indices(
    robot_joint_names: list[str],
    selected_joint_names: list[str],
    body_joint_indices: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Map selected names to motion-reference and robot tensor columns."""
    reference_indices = []
    robot_indices = []
    for joint_name in selected_joint_names:
        if joint_name not in robot_joint_names:
            raise ValueError(f"joint {joint_name!r} is not in robot.joint_names")
        robot_index = robot_joint_names.index(joint_name)
        if body_joint_indices is None:
            reference_index = robot_index
        elif robot_index in body_joint_indices:
            reference_index = body_joint_indices.index(robot_index)
        else:
            raise ValueError(
                f"joint {joint_name!r} is not present in the motion-lib body DOFs"
            )
        robot_indices.append(robot_index)
        reference_indices.append(reference_index)
    return reference_indices, robot_indices


def joint_angle_tracking_reward(
    reference: torch.Tensor, prediction: torch.Tensor, std: float
) -> torch.Tensor:
    """Return exp(-mean(wrapped joint error squared) / std squared)."""
    error = shortest_angular_difference(reference, prediction)
    return torch.exp(-error.square().mean(dim=-1) / std**2)
