"""Pure offline aggregation for scalable Isaac evaluation batches."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .batch_store import BatchStore
from .bucketing import BucketPlan, remap_bucket_payload
from .schemas import SCHEMA_VERSION, RunManifest


class IncompleteEvaluationError(RuntimeError):
    """Raised when complete aggregation is requested with missing motions."""


class DuplicateMotionError(RuntimeError):
    """Raised when two payload rows report the same motion index."""


@dataclass(frozen=True)
class AggregatedEvaluation:
    metrics_all: Mapping[str, float]
    metrics_success: Mapping[str, float]
    all_metrics_dict: Mapping[str, np.ndarray]
    failed_metrics_dict: Mapping[str, np.ndarray]
    failed_keys: tuple[str, ...]
    success_keys: tuple[str, ...]
    failed_idxes: np.ndarray
    success_rate: float
    progress_rate: float
    partial: bool
    missing_motion_keys: tuple[str, ...]
    evaluated_motion_count: int
    expected_motion_count: int
    completed_batch_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _one_dimensional(batch: Mapping[str, Any], key: str, count: int) -> np.ndarray:
    if key not in batch:
        raise ValueError(f"batch is missing required field {key!r}")
    values = np.asarray(batch[key]).reshape(-1)
    if values.size != count:
        raise ValueError(
            f"batch field {key!r} has {values.size} rows, expected {count}"
        )
    return values


def _legacy_metric_counts(key: str, lengths: np.ndarray) -> np.ndarray:
    """Infer counts for payloads written before counts_by_metric was added."""
    if "accel_dist" in key:
        return np.maximum(lengths - 2, 0)
    if "vel_dist" in key:
        return np.maximum(lengths - 1, 0)
    return lengths.copy()


def aggregate_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    expected_motion_keys: Sequence[str],
    alias_to_canonical: Mapping[str, str] | None = None,
    allow_partial: bool = False,
) -> AggregatedEvaluation:
    """Aggregate compact payloads using legacy frame-weighted semantics."""

    if not batches:
        raise IncompleteEvaluationError("no evaluation batch payloads were found")
    expected_keys = tuple(str(key) for key in expected_motion_keys)
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("expected_motion_keys contains duplicates")
    alias_to_canonical = dict(alias_to_canonical or {})

    rows: list[dict[str, Any]] = []
    metric_keys: tuple[str, ...] | None = None
    object_metrics_present: bool | None = None
    sampling_prob_present: bool | None = None
    for batch in batches:
        if int(batch.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("batch schema_version does not match")
        motion_idx = np.asarray(batch.get("motion_idx", [])).reshape(-1).astype(np.int64)
        count = motion_idx.size
        if count == 0:
            raise ValueError("batch contains no motion rows")
        terminated = _one_dimensional(batch, "terminated", count).astype(bool)
        progress = _one_dimensional(batch, "progress", count).astype(np.float64)
        lengths = _one_dimensional(batch, "length_per_motion", count).astype(np.int64)
        if np.any(lengths < 0):
            raise ValueError("length_per_motion values must be non-negative")

        sums_by_metric = batch.get("sums_by_metric")
        if not isinstance(sums_by_metric, Mapping) or not sums_by_metric:
            raise ValueError("batch sums_by_metric must be a non-empty mapping")
        current_metric_keys = tuple(sorted(str(key) for key in sums_by_metric))
        if metric_keys is None:
            metric_keys = current_metric_keys
        elif current_metric_keys != metric_keys:
            raise ValueError(
                f"batch metric keys differ: {current_metric_keys} != {metric_keys}"
            )
        metric_arrays = {
            key: _one_dimensional(sums_by_metric, key, count).astype(np.float64)
            for key in metric_keys
        }
        for key, values in metric_arrays.items():
            if not np.isfinite(values).all():
                raise ValueError(f"batch metric {key!r} contains non-finite values")

        counts_by_metric = batch.get("counts_by_metric")
        if counts_by_metric is None:
            metric_count_arrays = {
                key: _legacy_metric_counts(key, lengths) for key in metric_keys
            }
        else:
            if not isinstance(counts_by_metric, Mapping):
                raise ValueError("batch counts_by_metric must be a mapping")
            current_count_keys = tuple(sorted(str(key) for key in counts_by_metric))
            if current_count_keys != metric_keys:
                raise ValueError(
                    "batch counts_by_metric keys differ from sums_by_metric: "
                    f"{current_count_keys} != {metric_keys}"
                )
            metric_count_arrays = {
                key: _one_dimensional(counts_by_metric, key, count).astype(np.int64)
                for key in metric_keys
            }
        for key, values in metric_count_arrays.items():
            if np.any(values < 0):
                raise ValueError(f"batch metric {key!r} contains negative sample counts")

        has_object = "obj_pos_err" in batch or "obj_ori_err" in batch
        if object_metrics_present is None:
            object_metrics_present = has_object
        elif object_metrics_present != has_object:
            raise ValueError("object metric presence differs between batches")
        if has_object:
            if "obj_pos_err" not in batch or "obj_ori_err" not in batch:
                raise ValueError("object position and orientation metrics must appear together")
            obj_pos = _one_dimensional(batch, "obj_pos_err", count).astype(np.float64)
            obj_ori = _one_dimensional(batch, "obj_ori_err", count).astype(np.float64)
        else:
            obj_pos = obj_ori = None

        has_sampling_prob = "sampling_prob" in batch
        if sampling_prob_present is None:
            sampling_prob_present = has_sampling_prob
        elif sampling_prob_present != has_sampling_prob:
            raise ValueError("sampling_prob presence differs between batches")
        sampling_prob = (
            _one_dimensional(batch, "sampling_prob", count).astype(np.float64)
            if has_sampling_prob
            else None
        )

        for row_index, index in enumerate(motion_idx.tolist()):
            if not 0 <= index < len(expected_keys):
                raise ValueError(
                    f"motion index {index} is outside expected range [0, {len(expected_keys)})"
                )
            row = {
                "motion_idx": index,
                "terminated": bool(terminated[row_index]),
                "progress": float(progress[row_index]),
                "length": int(lengths[row_index]),
                "sums": {key: values[row_index] for key, values in metric_arrays.items()},
                "counts": {
                    key: values[row_index] for key, values in metric_count_arrays.items()
                },
            }
            if has_object:
                row["obj_pos_err"] = float(obj_pos[row_index])
                row["obj_ori_err"] = float(obj_ori[row_index])
            if has_sampling_prob:
                row["sampling_prob"] = float(sampling_prob[row_index])
            rows.append(row)

    rows_by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["motion_idx"])
        if index in rows_by_index:
            raise DuplicateMotionError(f"motion index {index} appears in multiple batches")
        rows_by_index[index] = row

    missing_indices = [index for index in range(len(expected_keys)) if index not in rows_by_index]
    missing_keys = tuple(expected_keys[index] for index in missing_indices)
    if missing_keys and not allow_partial:
        raise IncompleteEvaluationError(
            f"evaluation is incomplete: missing {len(missing_keys)} motions; "
            f"first missing keys={list(missing_keys[:10])}"
        )

    ordered_rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    motion_indices = np.asarray([row["motion_idx"] for row in ordered_rows], dtype=np.int64)
    canonical_keys = np.asarray(
        [
            alias_to_canonical.get(expected_keys[index], expected_keys[index])
            for index in motion_indices
        ],
        dtype=object,
    )
    terminated = np.asarray([row["terminated"] for row in ordered_rows], dtype=bool)
    progress = np.asarray([row["progress"] for row in ordered_rows], dtype=np.float64)
    progress = progress.copy()
    progress[~terminated] = 1.0
    lengths = np.asarray([row["length"] for row in ordered_rows], dtype=np.int64)
    success_mask = ~terminated

    all_metrics_dict: dict[str, np.ndarray] = {}
    metrics_all: dict[str, float] = {}
    metrics_success: dict[str, float] = {}
    assert metric_keys is not None
    for key in metric_keys:
        sums = np.asarray([row["sums"][key] for row in ordered_rows], dtype=np.float64)
        counts = np.asarray([row["counts"][key] for row in ordered_rows], dtype=np.int64)
        all_metrics_dict[key] = np.divide(
            sums,
            counts,
            out=np.full(sums.shape, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        total_count = int(counts.sum())
        metrics_all[key] = float(sums.sum() / total_count) if total_count > 0 else math.nan
        success_samples = int(counts[success_mask].sum())
        metrics_success[key] = (
            float(sums[success_mask].sum() / success_samples)
            if success_samples > 0
            else math.nan
        )

    all_metrics_dict["terminated"] = terminated
    all_metrics_dict["progress"] = progress
    all_metrics_dict["motion_keys"] = canonical_keys
    if sampling_prob_present:
        all_metrics_dict["sampling_prob"] = np.asarray(
            [row["sampling_prob"] for row in ordered_rows], dtype=np.float64
        )

    if object_metrics_present:
        obj_pos = np.asarray([row["obj_pos_err"] for row in ordered_rows], dtype=np.float64)
        obj_ori = np.asarray([row["obj_ori_err"] for row in ordered_rows], dtype=np.float64)
        all_metrics_dict["obj_pos_error"] = obj_pos
        all_metrics_dict["obj_ori_error"] = obj_ori
        metrics_all["obj_pos_error"] = float(obj_pos.mean())
        metrics_all["obj_ori_error"] = float(obj_ori.mean())
        metrics_success["obj_pos_error"] = (
            float(obj_pos[success_mask].mean()) if success_mask.any() else math.nan
        )
        metrics_success["obj_ori_error"] = (
            float(obj_ori[success_mask].mean()) if success_mask.any() else math.nan
        )

    failed_metrics_dict = {
        key: values[terminated]
        for key, values in all_metrics_dict.items()
        if key not in {"terminated", "progress"}
    }
    failed_keys = tuple(str(key) for key in canonical_keys[terminated].tolist())
    success_keys = tuple(str(key) for key in canonical_keys[success_mask].tolist())
    return AggregatedEvaluation(
        metrics_all=metrics_all,
        metrics_success=metrics_success,
        all_metrics_dict=all_metrics_dict,
        failed_metrics_dict=failed_metrics_dict,
        failed_keys=failed_keys,
        success_keys=success_keys,
        failed_idxes=np.nonzero(terminated)[0].astype(np.int64),
        success_rate=float(success_mask.mean()),
        progress_rate=float(progress.mean()),
        partial=bool(missing_keys),
        missing_motion_keys=missing_keys,
        evaluated_motion_count=len(ordered_rows),
        expected_motion_count=len(expected_keys),
        completed_batch_count=len(batches),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def to_metrics_eval_json(result: AggregatedEvaluation) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in result.metrics_success.items():
        payload[f"eval/success/{key}"] = float(value)
    payload["eval/success/success_rate"] = result.success_rate
    payload["eval/success/progress_rate"] = result.progress_rate
    for key, value in result.metrics_all.items():
        payload[f"eval/all/{key}"] = float(value)
    payload["eval/all_metrics_dict"] = _json_value(result.all_metrics_dict)
    payload["eval/failed_metrics_dict"] = _json_value(result.failed_metrics_dict)
    payload["failed_keys"] = list(result.failed_keys)
    payload["failed_idxes"] = result.failed_idxes.tolist()
    payload["eval/meta"] = {
        "schema_version": SCHEMA_VERSION,
        "partial": result.partial,
        "evaluated_motion_count": result.evaluated_motion_count,
        "expected_motion_count": result.expected_motion_count,
        "missing_motion_count": len(result.missing_motion_keys),
        "missing_motion_keys": list(result.missing_motion_keys),
        "completed_batch_count": result.completed_batch_count,
        **dict(result.metadata),
    }
    return payload


def load_remapped_bucket_batches(
    plan: BucketPlan,
    bucket_roots: Mapping[int, Path],
    *,
    allow_partial: bool,
) -> list[dict[str, Any]]:
    """Load child stores and map every bucket-local index to a global index."""

    remapped_batches: list[dict[str, Any]] = []
    seen_global_indices: set[int] = set()
    for assignment in plan.non_empty_assignments():
        root_value = bucket_roots.get(assignment.ordinal)
        if root_value is None:
            if allow_partial:
                continue
            raise IncompleteEvaluationError(
                f"evaluation bucket {assignment.name!r} has no output directory"
            )
        root = Path(root_value).expanduser().resolve()
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            if allow_partial:
                continue
            raise IncompleteEvaluationError(
                f"evaluation bucket {assignment.name!r} has no run manifest"
            )
        child_manifest = RunManifest.read(manifest_path)
        if child_manifest.status != "complete" and not allow_partial:
            raise IncompleteEvaluationError(
                f"evaluation bucket {assignment.name!r} is not complete"
            )
        if len(child_manifest.expected_motion_keys) != assignment.motion_count:
            raise ValueError(
                f"bucket {assignment.name!r} expected_motion_keys count does not match plan"
            )

        batches = BatchStore(
            root / "batch_results", run_id=child_manifest.run_id
        ).load_all()
        if not batches:
            if allow_partial:
                continue
            raise IncompleteEvaluationError(
                f"evaluation bucket {assignment.name!r} has no batch payloads"
            )

        seen_local_indices: set[int] = set()
        for payload in batches:
            local_indices = np.asarray(payload.get("motion_idx", [])).reshape(-1)
            for local_index in local_indices.astype(np.int64).tolist():
                if local_index in seen_local_indices:
                    raise DuplicateMotionError(
                        f"bucket {assignment.name!r} local motion index "
                        f"{local_index} appears in multiple batches"
                    )
                seen_local_indices.add(local_index)
            remapped = remap_bucket_payload(
                payload, assignment.local_to_global_indices
            )
            global_indices = np.asarray(remapped["motion_idx"]).reshape(-1)
            for global_index in global_indices.astype(np.int64).tolist():
                if global_index in seen_global_indices:
                    raise DuplicateMotionError(
                        f"global motion index {global_index} appears in multiple buckets"
                    )
                seen_global_indices.add(global_index)
            remapped_batches.append(remapped)

        if not allow_partial and seen_local_indices != set(range(assignment.motion_count)):
            missing = sorted(set(range(assignment.motion_count)) - seen_local_indices)
            raise IncompleteEvaluationError(
                f"evaluation bucket {assignment.name!r} is missing local indices "
                f"{missing[:10]}"
            )
    return remapped_batches


def _completed_bucket_count(
    plan: BucketPlan, bucket_roots: Mapping[int, Path]
) -> int:
    count = 0
    for assignment in plan.non_empty_assignments():
        root_value = bucket_roots.get(assignment.ordinal)
        if root_value is None:
            continue
        root = Path(root_value).expanduser().resolve()
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = RunManifest.read(manifest_path)
        if manifest.status != "complete":
            continue
        if any((root / "batch_results").glob("rank_*/batch_*.pt")):
            count += 1
    return count


def aggregate_bucket_outputs(
    parent_manifest: RunManifest,
    plan: BucketPlan,
    bucket_roots: Mapping[int, Path],
    *,
    allow_partial: bool = False,
) -> AggregatedEvaluation:
    """Aggregate bucket stores with the existing exact frame-level formulas."""

    batches = load_remapped_bucket_batches(
        plan,
        bucket_roots,
        allow_partial=allow_partial,
    )
    result = aggregate_batches(
        batches,
        expected_motion_keys=parent_manifest.expected_motion_keys,
        allow_partial=allow_partial,
    )
    non_empty_count = len(plan.non_empty_assignments())
    return replace(
        result,
        metadata={
            "bucketed": True,
            "bucket_count": non_empty_count,
            "completed_bucket_count": _completed_bucket_count(plan, bucket_roots),
        },
    )


def write_metrics_eval_json(result: AggregatedEvaluation, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(to_metrics_eval_json(result), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
