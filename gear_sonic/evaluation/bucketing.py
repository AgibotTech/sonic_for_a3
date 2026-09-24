"""Deterministic length-bucket plans for single-GPU evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import PreparedDataset, materialize_entries, safe_stem
from .schemas import (
    DatasetEntry,
    DatasetManifest,
    ManifestMismatchError,
    _atomic_write_json,
    canonical_json_fingerprint,
)


BUCKET_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BucketSpec:
    name: str
    max_frames: int | None
    num_envs: int


@dataclass(frozen=True)
class BucketAssignment:
    ordinal: int
    name: str
    min_frames_exclusive: int
    max_frames_inclusive: int | None
    num_envs: int
    local_to_global_indices: tuple[int, ...]
    motion_count: int
    real_frame_count: int
    estimated_padded_frame_count: int
    estimated_batch_count: int
    estimated_serial_steps: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BucketAssignment:
        max_frames = payload.get("max_frames_inclusive")
        return cls(
            ordinal=int(payload["ordinal"]),
            name=str(payload["name"]),
            min_frames_exclusive=int(payload["min_frames_exclusive"]),
            max_frames_inclusive=(int(max_frames) if max_frames is not None else None),
            num_envs=int(payload["num_envs"]),
            local_to_global_indices=tuple(
                int(index) for index in payload["local_to_global_indices"]
            ),
            motion_count=int(payload["motion_count"]),
            real_frame_count=int(payload["real_frame_count"]),
            estimated_padded_frame_count=int(payload["estimated_padded_frame_count"]),
            estimated_batch_count=int(payload["estimated_batch_count"]),
            estimated_serial_steps=int(payload["estimated_serial_steps"]),
        )


@dataclass(frozen=True)
class BucketPlan:
    schema_version: int
    dataset_fingerprint: str
    config_fingerprint: str
    assignments: tuple[BucketAssignment, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return canonical_json_fingerprint(self.to_dict())

    def non_empty_assignments(self) -> tuple[BucketAssignment, ...]:
        return tuple(assignment for assignment in self.assignments if assignment.motion_count)

    def write(self, path: str | Path) -> None:
        _atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: str | Path) -> BucketPlan:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        schema_version = int(payload["schema_version"])
        if schema_version != BUCKET_PLAN_SCHEMA_VERSION:
            raise ManifestMismatchError(
                "bucket plan schema_version="
                f"{schema_version}, expected {BUCKET_PLAN_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=schema_version,
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            assignments=tuple(
                BucketAssignment.from_dict(item) for item in payload["assignments"]
            ),
        )


@dataclass(frozen=True)
class BucketView:
    assignment: BucketAssignment
    root: Path
    motion_dir: Path
    manifest_path: Path
    manifest: DatasetManifest
    reused: bool


def validate_bucket_specs(specs: Sequence[BucketSpec]) -> None:
    if not specs:
        raise ValueError("at least one bucket is required")

    names: set[str] = set()
    previous_max = 0
    for index, spec in enumerate(specs):
        if not spec.name.strip():
            raise ValueError("bucket names must be non-empty")
        if spec.name in names:
            raise ValueError("bucket names must be unique")
        names.add(spec.name)

        if spec.num_envs < 1:
            raise ValueError("bucket num_envs must be positive")
        if spec.max_frames is None:
            if index != len(specs) - 1:
                raise ValueError("only the final bucket may omit max_frames")
            continue
        if spec.max_frames < 1:
            raise ValueError("bucket max_frames must be positive")
        if spec.max_frames <= previous_max:
            raise ValueError("bucket max_frames values must be strictly increasing")
        previous_max = spec.max_frames

    if specs[-1].max_frames is not None:
        raise ValueError("the final bucket must omit max_frames")


def _estimate_assignment_cost(
    frame_counts: Sequence[int], num_envs: int
) -> tuple[int, int, int]:
    batch_count = (len(frame_counts) + num_envs - 1) // num_envs
    serial_steps = sum(
        max(frame_counts[start : start + num_envs])
        for start in range(0, len(frame_counts), num_envs)
    )
    return batch_count, serial_steps * num_envs, serial_steps


def build_bucket_plan(
    manifest: DatasetManifest,
    specs: Sequence[BucketSpec],
    config_fingerprint: str,
) -> BucketPlan:
    validate_bucket_specs(specs)
    frame_counts = [entry.frame_count for entry in manifest.entries]
    if any(frame_count < 1 for frame_count in frame_counts):
        raise ValueError("dataset entries must have a positive frame_count")

    assignments: list[BucketAssignment] = []
    previous_max = 0
    for ordinal, spec in enumerate(specs):
        indices = tuple(
            index
            for index, frame_count in enumerate(frame_counts)
            if frame_count > previous_max
            and (spec.max_frames is None or frame_count <= spec.max_frames)
        )
        selected_counts = [frame_counts[index] for index in indices]
        batch_count, padded_frames, serial_steps = _estimate_assignment_cost(
            selected_counts, spec.num_envs
        )
        assignments.append(
            BucketAssignment(
                ordinal=ordinal,
                name=spec.name,
                min_frames_exclusive=previous_max,
                max_frames_inclusive=spec.max_frames,
                num_envs=spec.num_envs,
                local_to_global_indices=indices,
                motion_count=len(indices),
                real_frame_count=sum(selected_counts),
                estimated_padded_frame_count=padded_frames,
                estimated_batch_count=batch_count,
                estimated_serial_steps=serial_steps,
            )
        )
        if spec.max_frames is not None:
            previous_max = spec.max_frames

    return BucketPlan(
        schema_version=BUCKET_PLAN_SCHEMA_VERSION,
        dataset_fingerprint=manifest.source_fingerprint,
        config_fingerprint=str(config_fingerprint),
        assignments=tuple(assignments),
    )


def _safe_bucket_name(name: str) -> str:
    return safe_stem(Path(name))


def _bucket_entries(
    prepared: PreparedDataset, assignment: BucketAssignment
) -> tuple[DatasetEntry, ...]:
    selected = [
        prepared.manifest.entries[index]
        for index in assignment.local_to_global_indices
    ]
    width = max(8, len(str(max(0, len(selected) - 1))))
    return tuple(
        DatasetEntry(
            alias=f"{local_index:0{width}d}__{safe_stem(Path(entry.source_path))}",
            canonical_key=entry.canonical_key,
            source_path=entry.source_path,
            frame_count=entry.frame_count,
            source_size=entry.source_size,
            source_mtime_ns=entry.source_mtime_ns,
        )
        for local_index, entry in enumerate(selected)
    )


def _bucket_manifest_matches(
    existing: DatasetManifest, candidate: DatasetManifest
) -> bool:
    return (
        existing.source_root == candidate.source_root
        and existing.source_fingerprint == candidate.source_fingerprint
        and existing.sorting_mode == candidate.sorting_mode
        and existing.materialization == candidate.materialization
        and existing.entries == candidate.entries
    )


def materialize_bucket_views(
    prepared: PreparedDataset,
    plan: BucketPlan,
    output_root: str | Path,
    materialization: str,
) -> tuple[BucketView, ...]:
    """Create deterministic bucket views from an already scanned manifest."""

    if materialization not in {"symlink", "copy"}:
        raise ValueError("materialization must be 'symlink' or 'copy'")
    if plan.dataset_fingerprint != prepared.manifest.source_fingerprint:
        raise ManifestMismatchError(
            "bucket plan dataset fingerprint does not match prepared dataset"
        )

    output = Path(output_root).expanduser().resolve()
    views: list[BucketView] = []
    for assignment in plan.assignments:
        root = output / f"{assignment.ordinal:03d}_{_safe_bucket_name(assignment.name)}"
        motion_dir = root / "motions"
        manifest_path = root / "dataset_manifest.json"
        entries = _bucket_entries(prepared, assignment)
        candidate = DatasetManifest.create(
            source_root=prepared.manifest.source_root,
            entries=entries,
            sorting_mode="bucket",
            materialization=materialization,
        )

        reused = False
        manifest = candidate
        if manifest_path.exists():
            existing = DatasetManifest.read(manifest_path)
            if not _bucket_manifest_matches(existing, candidate):
                raise ManifestMismatchError(
                    f"bucket view differs from existing manifest: {root}"
                )
            expected_paths = [
                motion_dir / f"{entry.alias}.pkl" for entry in existing.entries
            ]
            if not all(path.exists() for path in expected_paths):
                raise ManifestMismatchError(
                    f"bucket manifest exists but materialized files are incomplete: {root}"
                )
            reused = True
            manifest = existing
        else:
            root.mkdir(parents=True, exist_ok=True)
            if entries:
                materialize_entries(entries, motion_dir, materialization)
            candidate.write(manifest_path)

        views.append(
            BucketView(
                assignment=assignment,
                root=root,
                motion_dir=motion_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                reused=reused,
            )
        )
    return tuple(views)


def remap_bucket_payload(
    payload: Mapping[str, Any], local_to_global: Sequence[int]
) -> dict[str, Any]:
    """Return a shallow payload copy with bucket-local motion indices remapped."""

    if "motion_idx" not in payload:
        raise ValueError("bucket payload is missing required field 'motion_idx'")
    local_indices = np.asarray(payload["motion_idx"])
    flattened = local_indices.reshape(-1).astype(np.int64)
    if np.any(flattened < 0) or np.any(flattened >= len(local_to_global)):
        raise ValueError(
            f"motion index is outside bucket-local range [0, {len(local_to_global)})"
        )
    lookup = np.asarray(local_to_global, dtype=np.int64)
    remapped = dict(payload)
    remapped["motion_idx"] = lookup[flattened].reshape(local_indices.shape)
    return remapped
