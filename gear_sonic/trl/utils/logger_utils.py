"""Shared logging helpers for Trainer integrations and eval post-processing."""

from __future__ import annotations

from collections.abc import Iterable
import json
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

NO_REPORTING_VALUES = {"", "none", "null", "false", "disabled"}


def normalize_report_to(report_to: Any) -> list[str]:
    """Normalize a Hydra/HF ``report_to`` value into integration names."""
    if report_to is None:
        return []
    if isinstance(report_to, str):
        value = report_to.strip()
        if value.lower() in NO_REPORTING_VALUES:
            return []
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        parts = re.split(r"[\s,]+", value)
    elif isinstance(report_to, Iterable) and not isinstance(report_to, dict):
        parts = []
        for item in report_to:
            parts.extend(normalize_report_to(item))
    else:
        parts = [str(report_to)]

    normalized = []
    for part in parts:
        name = str(part).strip().strip("'\"").lower()
        if not name or name in NO_REPORTING_VALUES:
            continue
        if name not in normalized:
            normalized.append(name)
    return normalized


def report_to_contains(report_to: Any, integration: str) -> bool:
    return integration.lower() in normalize_report_to(report_to)


def trainer_report_to(report_to: Any) -> str | list[str]:
    """Return integrations that HuggingFace callbacks should own.

    W&B and TensorBoard are initialized and logged by SONIC-specific callbacks,
    so they are intentionally removed here to avoid duplicate records.
    """
    integrations = [
        name for name in normalize_report_to(report_to) if name not in {"wandb", "tensorboard"}
    ]
    if not integrations:
        return "none"
    if len(integrations) == 1:
        return integrations[0]
    return integrations


def is_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return True
    if isinstance(value, np.generic):
        return np.issubdtype(value.dtype, np.number) and value.shape == ()
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and not value.dtype == torch.bool
    return False


def scalar_value(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def filter_scalar_logs(logs: dict[str, Any]) -> dict[str, float]:
    """Keep only scalar logs that HF TensorBoard can write without warnings."""
    return {key: scalar_value(value) for key, value in logs.items() if is_scalar(value)}


def tensorboard_train_logs(logs: dict[str, Any]) -> dict[str, float]:
    """Prepare training logs with stable TensorBoard groups.

    HuggingFace's default TensorBoard integration rewrites every training tag
    under ``train/``. SONIC already has meaningful W&B-style prefixes, so keep
    those groups visible and fold unprefixed bookkeeping into ``Train/``.
    """
    scalar_logs = filter_scalar_logs(logs)
    normalized = {}
    for key, value in scalar_logs.items():
        if _is_duplicate_raw_env_key(key, scalar_logs):
            continue
        tag = tensorboard_train_tag(key)
        if tag is not None:
            normalized[tag] = value
    return normalized


def tensorboard_train_tag(key: str) -> str | None:
    if key in {"total_flos"}:
        return None
    env_tag = _env_metric_tag(key)
    if env_tag is not None:
        return env_tag
    if key.startswith("objective/"):
        return "Objective/" + key.removeprefix("objective/")
    if key.startswith("loss/"):
        return "Loss/" + key.removeprefix("loss/")
    if key.startswith("policy/"):
        return "Policy/" + key.removeprefix("policy/")
    if key.startswith("val/"):
        return "Value/" + key.removeprefix("val/")
    if key.startswith("Policy/"):
        return key
    if key.startswith("Episode/"):
        return key
    if key.startswith("scheduled_params/"):
        return "Scheduled/" + key.removeprefix("scheduled_params/")
    if key in {
        "collection_time",
        "learn_time",
        "tot_timesteps",
        "tot_time",
        "it",
        "fps",
        "eps",
        "lr",
        "episode",
        "batch_idx",
        "num_total_batches",
        "epoch",
    }:
        return "Train/" + key
    return "Misc/" + key


def create_summary_writer(log_dir: str | Path):
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def write_tensorboard_values(writer, values: dict[str, Any], step: int) -> None:
    for tag, value in values.items():
        _write_tensorboard_value(writer, tag, value, step)
    writer.flush()


def _write_tensorboard_value(writer, tag: str, value: Any, step: int) -> None:
    safe_tag = _safe_tag(tag)
    if is_scalar(value):
        writer.add_scalar(safe_tag, scalar_value(value), step)
    elif isinstance(value, dict):
        scalar_children = {k: v for k, v in value.items() if is_scalar(v)}
        complex_children = {k: v for k, v in value.items() if not is_scalar(v)}
        for child_key, child_value in scalar_children.items():
            _write_tensorboard_value(writer, f"{safe_tag}/{child_key}", child_value, step)
        if complex_children:
            writer.add_text(safe_tag, metrics_dict_to_text(value), step)
    elif isinstance(value, (list, tuple, np.ndarray)):
        writer.add_text(safe_tag, metrics_dict_to_text(value), step)
    elif value is not None:
        writer.add_text(safe_tag, str(value), step)


def metrics_dict_to_text(value: Any) -> str:
    if isinstance(value, dict) and "motion_keys" in value:
        table = motion_metrics_to_markdown(value)
        if table is not None:
            return table
    return _json_text(value)


def motion_metrics_to_markdown(metrics_dict: dict[str, Any]) -> str | None:
    motion_keys = metrics_dict.get("motion_keys") or []
    if not motion_keys:
        return None

    metric_names = [key for key in metrics_dict if key not in {"motion_keys", "terminated"}]
    header = ["motion_key", "terminated", *metric_names]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    terminated = metrics_dict.get("terminated", [None] * len(motion_keys))
    for idx, motion_key in enumerate(motion_keys):
        row = [str(motion_key), _format_cell(_safe_index(terminated, idx))]
        for metric_name in metric_names:
            row.append(_format_cell(_safe_index(metrics_dict.get(metric_name), idx)))
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def add_video_or_text(writer, tag: str, video_path: str | Path, step: int, max_frames: int = 300) -> None:
    video_path = Path(video_path)
    try:
        video_tensor, fps = load_video_tensor(video_path, max_frames=max_frames)
        writer.add_video(_safe_tag(tag), video_tensor, step, fps=fps)
    except Exception as exc:  # noqa: BLE001
        writer.add_text(
            _safe_tag(f"{tag}/decode_error"),
            f"Could not decode video for TensorBoard: {video_path}\n\n{exc}",
            step,
        )


def load_video_tensor(video_path: str | Path, max_frames: int = 300) -> tuple[torch.Tensor, int]:
    video_path = str(video_path)
    try:
        return _load_video_with_cv2(video_path, max_frames=max_frames)
    except Exception:
        return _load_video_with_imageio(video_path, max_frames=max_frames)


def _load_video_with_cv2(video_path: str, max_frames: int) -> tuple[torch.Tensor, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 20))
    frames = []
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    return _frames_to_video_tensor(frames), max(fps, 1)


def _load_video_with_imageio(video_path: str, max_frames: int) -> tuple[torch.Tensor, int]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = int(round(meta.get("fps", 20) or 20))
    frames = []
    try:
        for frame in reader:
            if len(frames) >= max_frames:
                break
            frames.append(np.asarray(frame[..., :3]))
    finally:
        reader.close()

    return _frames_to_video_tensor(frames), max(fps, 1)


def _frames_to_video_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    if not frames:
        raise ValueError("Video contained no decodable frames")
    array = np.stack(frames).astype(np.uint8)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor


def _json_text(value: Any) -> str:
    try:
        return json.dumps(_to_jsonable(value), ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _safe_index(values: Any, idx: int) -> Any:
    try:
        return values[idx]
    except Exception:  # noqa: BLE001
        return None


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _safe_tag(tag: str) -> str:
    return str(tag).strip("/")


def _is_duplicate_raw_env_key(key: str, scalar_logs: dict[str, float]) -> bool:
    return f"Env/{key}" in scalar_logs


def _env_metric_tag(key: str) -> str | None:
    env_key = key.removeprefix("Env/") if key.startswith("Env/") else key
    env_key_lower = env_key.lower()

    grouped_prefixes = [
        ("episode_reward/", "Rewards/"),
        ("reward/", "Rewards/"),
        ("rewards/", "Rewards/"),
        ("episode_termination/", "Terminations/"),
        ("termination/", "Terminations/"),
        ("terminations/", "Terminations/"),
        ("episode_curriculum/", "Curriculum/"),
        ("curriculum/", "Curriculum/"),
        ("adp_samp/", "AdaptiveSampling/"),
        ("adaptive_sampling/", "AdaptiveSampling/"),
        ("command/", "Commands/"),
        ("commands/", "Commands/"),
        ("action/", "Actions/"),
        ("actions/", "Actions/"),
        ("event/", "Events/"),
        ("events/", "Events/"),
    ]
    for source_prefix, target_prefix in grouped_prefixes:
        if env_key_lower.startswith(source_prefix):
            return target_prefix + env_key[len(source_prefix) :]

    if key.startswith("Env/"):
        return "Env/" + env_key
    return None
