"""Validation of complete or explicitly partial evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bucketing import BucketPlan
from .schemas import RunManifest


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    partial: bool
    expected_motion_count: int
    evaluated_motion_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_evaluation(
    run_manifest: RunManifest,
    metrics: Mapping[str, Any],
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    all_metrics = metrics.get("eval/all_metrics_dict")
    if not isinstance(all_metrics, Mapping):
        errors.append("eval/all_metrics_dict is missing or is not a mapping")
        all_metrics = {}

    motion_keys = [str(key) for key in all_metrics.get("motion_keys", [])]
    terminated = np.asarray(all_metrics.get("terminated", []), dtype=bool).reshape(-1)
    evaluated_count = len(motion_keys)
    expected_keys = tuple(run_manifest.expected_motion_keys)
    meta = metrics.get("eval/meta", {})
    partial = bool(meta.get("partial", False)) if isinstance(meta, Mapping) else False

    if len(set(motion_keys)) != len(motion_keys):
        errors.append("motion_keys contains duplicates")
    if terminated.size != evaluated_count:
        errors.append(
            f"terminated length {terminated.size} does not match motion_keys {evaluated_count}"
        )
    for key, values in all_metrics.items():
        if key == "motion_keys":
            continue
        try:
            length = len(values)
        except TypeError:
            errors.append(f"per-motion field {key!r} is not a sequence")
            continue
        if length != evaluated_count:
            errors.append(
                f"per-motion field {key!r} has length {length}, expected {evaluated_count}"
            )

    actual_set = set(motion_keys)
    expected_set = set(expected_keys)
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    if extra:
        errors.append(f"metrics contains {len(extra)} unexpected motion keys")
    if missing and not partial:
        errors.append(f"metrics is missing {len(missing)} motion keys but partial=false")
    if missing and partial:
        warnings.append(f"partial evaluation is missing {len(missing)} motion keys")
    if not missing and partial:
        errors.append("eval/meta.partial=true but motion coverage is complete")

    scalar_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("eval/all/") or key.startswith("eval/success/")
    }
    for key, value in scalar_metrics.items():
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            errors.append(f"scalar metric {key!r} is not finite")

    reported_success = metrics.get("eval/success/success_rate")
    if terminated.size == evaluated_count and evaluated_count > 0:
        computed_success = float((~terminated).mean())
        try:
            matches = math.isclose(
                float(reported_success), computed_success, rel_tol=1e-9, abs_tol=1e-9
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(
                "eval/success/success_rate does not equal mean(not terminated): "
                f"reported={reported_success!r}, computed={computed_success}"
            )

    return ValidationReport(
        valid=not errors,
        partial=partial,
        expected_motion_count=len(expected_keys),
        evaluated_motion_count=evaluated_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_bucket_plan(
    parent_manifest: RunManifest,
    plan: BucketPlan,
    metrics: Mapping[str, Any],
    child_roots: Mapping[int, Path],
) -> ValidationReport:
    """Validate exact bucket coverage and child/parent output compatibility."""

    base = validate_evaluation(parent_manifest, metrics)
    errors = list(base.errors)
    warnings = list(base.warnings)
    if plan.dataset_fingerprint != parent_manifest.dataset_fingerprint:
        errors.append("bucket plan dataset fingerprint does not match parent manifest")
    if plan.config_fingerprint != parent_manifest.config_fingerprint:
        errors.append("bucket plan config fingerprint does not match parent manifest")

    mapped_indices = [
        index
        for assignment in plan.assignments
        for index in assignment.local_to_global_indices
    ]
    expected_indices = list(range(len(parent_manifest.expected_motion_keys)))
    if len(set(mapped_indices)) != len(mapped_indices):
        errors.append("bucket plan maps at least one global index more than once")
    if sorted(mapped_indices) != expected_indices:
        errors.append("bucket plan does not exactly cover the parent motion index range")
    for assignment in plan.assignments:
        if assignment.motion_count != len(assignment.local_to_global_indices):
            errors.append(
                f"bucket {assignment.name!r} motion_count does not match its index map"
            )

    meta = metrics.get("eval/meta", {})
    partial = base.partial
    if not isinstance(meta, Mapping) or meta.get("bucketed") is not True:
        errors.append("eval/meta.bucketed is not true")
        meta = {}
    non_empty = plan.non_empty_assignments()
    if meta.get("bucket_count") != len(non_empty):
        errors.append("eval/meta.bucket_count does not match the bucket plan")

    completed_count = 0
    for assignment in non_empty:
        root_value = child_roots.get(assignment.ordinal)
        if root_value is None:
            message = f"bucket {assignment.name!r} output is missing"
            (warnings if partial else errors).append(message)
            continue
        root = Path(root_value).expanduser().resolve()
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            message = f"bucket {assignment.name!r} run manifest is missing"
            (warnings if partial else errors).append(message)
            continue
        try:
            child = RunManifest.read(manifest_path)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"bucket {assignment.name!r} run manifest is invalid: {exc}")
            continue
        if child.checkpoint_fingerprint != parent_manifest.checkpoint_fingerprint:
            errors.append(f"bucket {assignment.name!r} checkpoint differs from parent")
        expected_keys = tuple(
            parent_manifest.expected_motion_keys[index]
            for index in assignment.local_to_global_indices
        )
        if child.expected_motion_keys != expected_keys:
            errors.append(f"bucket {assignment.name!r} expected motion keys differ")
        if child.world_size != 1:
            errors.append(f"bucket {assignment.name!r} world_size is not 1")
        if child.status == "complete":
            completed_count += 1
        elif not partial:
            errors.append(f"bucket {assignment.name!r} is not complete")

    if meta.get("completed_bucket_count") != completed_count:
        errors.append("eval/meta.completed_bucket_count does not match child outputs")
    all_metrics = metrics.get("eval/all_metrics_dict", {})
    motion_keys = (
        [str(key) for key in all_metrics.get("motion_keys", [])]
        if isinstance(all_metrics, Mapping)
        else []
    )
    expected_order = [
        key for key in parent_manifest.expected_motion_keys if key in set(motion_keys)
    ]
    if motion_keys != expected_order:
        errors.append("bucketed motion_keys are not in canonical global order")

    return ValidationReport(
        valid=not errors,
        partial=partial,
        expected_motion_count=base.expected_motion_count,
        evaluated_motion_count=base.evaluated_motion_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
