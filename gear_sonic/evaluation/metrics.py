"""Per-batch metric calculation for scalable Isaac evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .schemas import SCHEMA_VERSION

BODY_SUBSETS = {
    "legs": (
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    ),
    "vr_3points": (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
    "other_upper_bodies": (
        "pelvis",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
    ),
    "foot": ("left_ankle_roll_link", "right_ankle_roll_link"),
}


def compute_joint_mse_metrics(pred_pos_all, gt_pos_all, root_idx: int = 0):
    """Return squared 3-D joint-position errors in metres squared."""

    if len(pred_pos_all) != len(gt_pos_all):
        raise ValueError("pred_pos_all and gt_pos_all must contain the same trajectories")

    joint_mse_g = []
    joint_mse_l = []
    for pred_pos, gt_pos in zip(pred_pos_all, gt_pos_all, strict=True):
        if pred_pos.shape != gt_pos.shape:
            raise ValueError(
                "predicted and target joint positions must have matching shape, "
                f"got {pred_pos.shape} and {gt_pos.shape}"
            )
        if pred_pos.ndim != 3 or pred_pos.shape[-1] != 3:
            raise ValueError(
                "joint position trajectories must have shape (frames, joints, 3), "
                f"got {pred_pos.shape}"
            )
        if not 0 <= root_idx < pred_pos.shape[1]:
            raise ValueError(f"root_idx={root_idx} is outside {pred_pos.shape[1]} joints")

        joint_mse_g.append(np.square(gt_pos - pred_pos).sum(axis=-1))
        pred_pos_local = pred_pos - pred_pos[:, [root_idx]]
        gt_pos_local = gt_pos - gt_pos[:, [root_idx]]
        joint_mse_l.append(np.square(gt_pos_local - pred_pos_local).sum(axis=-1))

    return {"joint_mse_g": joint_mse_g, "joint_mse_l": joint_mse_l}


def _lookup_body_index(body_names: Sequence[str], requested: str) -> int:
    normalized = {name.lower(): idx for idx, name in enumerate(body_names)}
    key = requested.lower()
    candidates = [key]
    if key.endswith("_link"):
        candidates.append(key[:-5])
    else:
        candidates.append(key + "_link")
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for idx, body_name in enumerate(body_names):
        if body_name.lower().endswith("_" + key):
            return idx
    raise ValueError(f"body {requested!r} was not found; available={list(body_names)!r}")


def resolve_body_subset_indices(body_names: Sequence[str]) -> dict[str, list[int]]:
    """Resolve the callback's standard metric subsets against robot body names."""

    return {
        subset: [_lookup_body_index(body_names, name) for name in requested_names]
        for subset, requested_names in BODY_SUBSETS.items()
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_metric_sums_and_counts(
    metrics: dict[str, list[np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return per-motion sums and the matching temporal sample counts."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for key, trajectories in metrics.items():
        per_motion = []
        per_motion_counts = []
        for trajectory in trajectories:
            values = np.asarray(trajectory)
            if values.ndim == 0:
                per_motion.append(float(values))
                per_motion_counts.append(1)
            elif "mpjpe" in key or "joint_mse" in key:
                joint_count = values.shape[1] if values.ndim > 1 else 1
                per_motion.append(float(values.sum() / joint_count))
                per_motion_counts.append(values.shape[0])
            else:
                per_motion.append(float(values.sum()))
                per_motion_counts.append(values.shape[0])
        sums[key] = np.asarray(per_motion, dtype=np.float64)
        counts[key] = np.asarray(per_motion_counts, dtype=np.int64)
    return sums, counts


def compute_metrics_lite_with_empty(
    pred_trajectories: Sequence[np.ndarray],
    gt_trajectories: Sequence[np.ndarray],
) -> dict[str, list[np.ndarray]]:
    """Run smpl_sim metrics while preserving motions with zero valid frames."""
    if len(pred_trajectories) != len(gt_trajectories):
        raise ValueError("predicted and target trajectory counts must match")
    if not pred_trajectories:
        return {}

    pairs = list(zip(pred_trajectories, gt_trajectories, strict=True))
    if any((len(pred) > 0) != (len(gt) > 0) for pred, gt in pairs):
        raise ValueError("predicted and target trajectories must have matching lengths")
    nonempty_indices = [idx for idx, (pred, _) in enumerate(pairs) if len(pred) > 0]

    from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

    if nonempty_indices:
        computed = compute_metrics_lite(
            [pred_trajectories[idx] for idx in nonempty_indices],
            [gt_trajectories[idx] for idx in nonempty_indices],
            concatenate=False,
            use_tqdm=False,
        )
    else:
        joint_count = pred_trajectories[0].shape[1]
        dummy = np.zeros((3, joint_count, 3), dtype=pred_trajectories[0].dtype)
        computed = compute_metrics_lite(
            [dummy], [dummy], concatenate=False, use_tqdm=False
        )

    metrics: dict[str, list[np.ndarray]] = {}
    for key, values in computed.items():
        nonempty_values = values if nonempty_indices else []
        by_index = {
            motion_idx: np.asarray(value)
            for motion_idx, value in zip(nonempty_indices, nonempty_values, strict=True)
        }
        metrics[key] = [
            by_index.get(idx, np.empty((0,), dtype=np.float64))
            for idx in range(len(pred_trajectories))
        ]
    return metrics


def compute_subset_metrics_lite(
    pred_trajectories: Sequence[np.ndarray],
    gt_trajectories: Sequence[np.ndarray],
    *,
    subset_indices: Sequence[int],
    root_idx: int,
) -> dict[str, list[np.ndarray]]:
    """Compute subset metrics with local MPJPE rooted at the full-body pelvis."""
    subset_pred = [
        trajectory[:, subset_indices, :] for trajectory in pred_trajectories
    ]
    subset_gt = [
        trajectory[:, subset_indices, :] for trajectory in gt_trajectories
    ]
    metrics = compute_metrics_lite_with_empty(subset_pred, subset_gt)
    metrics["mpjpe_l"] = [
        np.linalg.norm(
            (pred - pred[:, [root_idx]])[:, subset_indices, :]
            - (gt - gt[:, [root_idx]])[:, subset_indices, :],
            axis=2,
        )
        * 1000
        for pred, gt in zip(pred_trajectories, gt_trajectories, strict=True)
    ]
    return metrics


def build_batch_payload(
    *,
    pred_pos: torch.Tensor | np.ndarray,
    gt_pos: torch.Tensor | np.ndarray,
    motion_lengths: torch.Tensor | np.ndarray,
    motion_idx: torch.Tensor | np.ndarray,
    terminated: torch.Tensor | np.ndarray,
    progress: torch.Tensor | np.ndarray,
    body_names: Sequence[str],
    run_id: str,
    rank: int,
    batch_idx: int,
    valid_lengths: torch.Tensor | np.ndarray | None = None,
    sampling_prob: torch.Tensor | np.ndarray | None = None,
    obj_pos_err: torch.Tensor | np.ndarray | None = None,
    obj_ori_err: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one compact, trajectory-free batch payload.

    Position tensors are expected as ``(steps, envs, joints, 3)``. Large tensors
    cross the device boundary once here, at the end of a rollout batch.
    """

    if pred_pos.shape != gt_pos.shape:
        raise ValueError("pred_pos and gt_pos must have matching shape")
    if len(pred_pos.shape) != 4 or pred_pos.shape[-1] != 3:
        raise ValueError("positions must have shape (steps, envs, joints, 3)")
    env_count = int(pred_pos.shape[1])
    metadata = {
        "motion_lengths": motion_lengths,
        "motion_idx": motion_idx,
        "terminated": terminated,
        "progress": progress,
    }
    for name, value in metadata.items():
        if int(np.asarray(_to_numpy(value)).size) != env_count:
            raise ValueError(f"{name} must contain one value per environment")
    if len(body_names) != int(pred_pos.shape[2]):
        raise ValueError("body_names must match the joint dimension")

    pred_cpu = _to_numpy(pred_pos)
    gt_cpu = _to_numpy(gt_pos)
    raw_lengths = _to_numpy(motion_lengths).reshape(-1).astype(np.int64)
    if valid_lengths is None:
        trajectory_lengths = raw_lengths - 1
    else:
        trajectory_lengths = _to_numpy(valid_lengths).reshape(-1).astype(np.int64)
        if trajectory_lengths.size != env_count:
            raise ValueError("valid_lengths must contain one value per environment")
    trajectory_lengths = np.clip(trajectory_lengths, 0, pred_cpu.shape[0]).astype(np.int64)
    pred_trajectories = [
        pred_cpu[:length, env] for env, length in enumerate(trajectory_lengths)
    ]
    gt_trajectories = [
        gt_cpu[:length, env] for env, length in enumerate(trajectory_lengths)
    ]

    metrics = compute_metrics_lite_with_empty(pred_trajectories, gt_trajectories)
    metrics.update(compute_joint_mse_metrics(pred_trajectories, gt_trajectories))

    subset_indices = resolve_body_subset_indices(body_names)
    pelvis_idx = _lookup_body_index(body_names, "pelvis")
    for subset, indices in subset_indices.items():
        subset_metrics = compute_subset_metrics_lite(
            pred_trajectories,
            gt_trajectories,
            subset_indices=indices,
            root_idx=pelvis_idx,
        )
        metrics.update({f"{key}_{subset}": value for key, value in subset_metrics.items()})

    sums_by_metric, counts_by_metric = compute_metric_sums_and_counts(metrics)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "rank": int(rank),
        "batch_idx": int(batch_idx),
        "motion_idx": _to_numpy(motion_idx).reshape(-1).astype(np.int64),
        "terminated": _to_numpy(terminated).reshape(-1).astype(bool),
        "progress": _to_numpy(progress).reshape(-1).astype(np.float32),
        "length_per_motion": trajectory_lengths,
        "sums_by_metric": sums_by_metric,
        "counts_by_metric": counts_by_metric,
    }
    if sampling_prob is not None:
        sampling_cpu = _to_numpy(sampling_prob).reshape(-1)
        if sampling_cpu.size != env_count:
            raise ValueError("sampling_prob must contain one value per environment")
        payload["sampling_prob"] = sampling_cpu.astype(np.float32)

    if obj_pos_err is not None or obj_ori_err is not None:
        if obj_pos_err is None or obj_ori_err is None:
            raise ValueError("object position and orientation errors must be provided together")
        pos_cpu = _to_numpy(obj_pos_err)
        ori_cpu = _to_numpy(obj_ori_err)
        if pos_cpu.shape[:2] != pred_cpu.shape[:2] or ori_cpu.shape[:2] != pred_cpu.shape[:2]:
            raise ValueError("object error tensors must have shape (steps, envs)")
        payload["obj_pos_err"] = np.asarray(
            [
                pos_cpu[:length, env].mean() if length else np.nan
                for env, length in enumerate(trajectory_lengths)
            ],
            dtype=np.float32,
        )
        payload["obj_ori_err"] = np.asarray(
            [
                ori_cpu[:length, env].mean() if length else np.nan
                for env, length in enumerate(trajectory_lengths)
            ],
            dtype=np.float32,
        )
    return payload
