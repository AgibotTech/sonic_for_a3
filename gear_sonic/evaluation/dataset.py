"""Generic motion-dataset indexing and length-aware materialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Literal, Sequence

import joblib

from .schemas import DatasetEntry, DatasetManifest, ManifestMismatchError


_LENGTH_FIELDS = (
    "root_trans_offset",
    "pose_aa",
    "dof",
    "root_trans",
    "root_pos",
)


@dataclass(frozen=True)
class SortingOptions:
    mode: Literal["off", "on", "auto"] = "auto"
    order: Literal["ascending"] = "ascending"
    materialization: Literal["symlink", "copy"] = "symlink"
    auto_min_motions: int = 2048
    auto_padding_ratio: float = 1.5
    workers: int = 8

    def __post_init__(self) -> None:
        if self.mode not in {"off", "on", "auto"}:
            raise ValueError("sorting mode must be one of: off, on, auto")
        if self.order != "ascending":
            raise ValueError("only ascending motion sorting is supported")
        if self.materialization not in {"symlink", "copy"}:
            raise ValueError("materialization must be 'symlink' or 'copy'")
        if self.auto_min_motions < 1:
            raise ValueError("auto_min_motions must be at least 1")
        if self.auto_padding_ratio < 1.0:
            raise ValueError("auto_padding_ratio must be at least 1.0")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")


@dataclass(frozen=True)
class LongMotionOptions:
    enabled: bool = False
    max_frames: int = 4000

    def __post_init__(self) -> None:
        if self.max_frames < 1:
            raise ValueError("max_frames must be at least 1")


@dataclass(frozen=True)
class PreparedDataset:
    motion_dir: Path
    manifest_path: Path
    manifest: DatasetManifest
    long_motion_dir: Path | None
    reused: bool


@dataclass(frozen=True)
class _SourceMotion:
    path: Path
    canonical_key: str
    embedded_key: str
    frame_count: int
    source_size: int
    source_mtime_ns: int


def resolve_prepared_output(configured: str | Path | None, eval_output: str | Path) -> Path:
    if configured is not None:
        return Path(configured).expanduser().resolve()
    return Path(eval_output).expanduser().resolve() / "prepared_dataset"


def infer_frame_count(path: str | Path) -> tuple[int, str]:
    """Read one motion PKL and return ``(frame_count, embedded_key)``."""

    motion_path = Path(path)
    payload = joblib.load(motion_path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"motion PKL must contain a non-empty mapping: {motion_path}")

    if any(field in payload for field in _LENGTH_FIELDS):
        embedded_key = motion_path.stem
        entry = payload
    else:
        embedded_key, entry = next(iter(payload.items()))
        if not isinstance(entry, dict):
            raise ValueError(f"motion entry must be a mapping: {motion_path}")

    for field in _LENGTH_FIELDS:
        value = entry.get(field)
        if value is None:
            continue
        try:
            frame_count = int(value.shape[0])
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                f"motion length field {field!r} has no leading frame dimension: {motion_path}"
            ) from exc
        if frame_count < 1:
            raise ValueError(f"motion contains no frames: {motion_path}")
        return frame_count, str(embedded_key)

    raise ValueError(
        f"cannot infer frame count from {motion_path}; expected one of {_LENGTH_FIELDS}"
    )


def estimate_padding_ratio(frame_counts: Sequence[int], batch_size: int) -> float:
    """Estimate current-order cost divided by length-sorted batch cost."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not frame_counts:
        return 1.0
    if any(int(count) < 1 for count in frame_counts):
        raise ValueError("frame counts must be positive")

    def padded_cost(values: Sequence[int]) -> int:
        return sum(
            max(values[start : start + batch_size])
            for start in range(0, len(values), batch_size)
        )

    current_cost = padded_cost(frame_counts)
    sorted_cost = padded_cost(sorted(frame_counts))
    return float(current_cost / sorted_cost) if sorted_cost else 1.0


def safe_stem(path: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return safe or "motion"


def _scan_one(source_root: Path, path: Path) -> _SourceMotion:
    frame_count, embedded_key = infer_frame_count(path)
    stat = path.stat()
    canonical_key = path.relative_to(source_root).with_suffix("").as_posix()
    return _SourceMotion(
        path=path.resolve(),
        canonical_key=canonical_key,
        embedded_key=embedded_key,
        frame_count=frame_count,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )


def _scan_dataset(source_root: Path, workers: int) -> list[_SourceMotion]:
    paths = sorted(
        path
        for path in source_root.rglob("*.pkl")
        if path.name not in {"metadata.pkl", "dataset_manifest.pkl"}
    )
    if not paths:
        raise ValueError(f"no motion PKL files found under {source_root}")
    if workers == 1:
        return [_scan_one(source_root, path) for path in paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda path: _scan_one(source_root, path), paths))


def _entries_for(
    motions: Sequence[_SourceMotion], *, sorted_view: bool, index_offset: int = 0
) -> tuple[DatasetEntry, ...]:
    ordered = (
        sorted(motions, key=lambda item: (item.frame_count, item.canonical_key))
        if sorted_view
        else sorted(motions, key=lambda item: item.canonical_key)
    )
    width = max(8, len(str(max(0, len(ordered) - 1))))
    entries = []
    for index, motion in enumerate(ordered):
        if sorted_view:
            alias = f"{index + index_offset:0{width}d}__{safe_stem(motion.path)}"
        else:
            alias = motion.path.stem
        entries.append(
            DatasetEntry(
                alias=alias,
                canonical_key=motion.canonical_key,
                source_path=str(motion.path),
                frame_count=motion.frame_count,
                source_size=motion.source_size,
                source_mtime_ns=motion.source_mtime_ns,
            )
        )
    aliases = [entry.alias for entry in entries]
    if len(set(aliases)) != len(aliases):
        raise ValueError(
            "duplicate motion basenames require sorting/materialization so aliases remain unique"
        )
    return tuple(entries)


def materialize_entries(
    entries: Sequence[DatasetEntry], motion_dir: Path, mode: str
) -> None:
    motion_dir.mkdir(parents=True, exist_ok=True)
    existing = list(motion_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"prepared motion directory is not empty and has no reusable manifest: {motion_dir}"
        )
    for entry in entries:
        destination = motion_dir / f"{entry.alias}.pkl"
        source = Path(entry.source_path)
        if mode == "copy":
            shutil.copyfile(source, destination)
        else:
            os.symlink(source, destination)


def _manifest_matches(existing: DatasetManifest, candidate: DatasetManifest) -> bool:
    return (
        existing.source_root == candidate.source_root
        and existing.source_fingerprint == candidate.source_fingerprint
        and existing.sorting_mode == candidate.sorting_mode
        and existing.materialization == candidate.materialization
        and existing.entries == candidate.entries
    )


def prepare_dataset(
    source_root: str | Path,
    output_dir: str | Path,
    sorting: SortingOptions | None = None,
    *,
    batch_size: int,
    long_motion: LongMotionOptions | None = None,
) -> PreparedDataset:
    """Create or reuse a deterministic evaluation dataset view."""

    sorting = sorting or SortingOptions()
    long_motion = long_motion or LongMotionOptions()
    source = Path(source_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"dataset source is not a directory: {source}")

    motions = _scan_dataset(source, sorting.workers)
    original_counts = [motion.frame_count for motion in motions]
    if sorting.mode == "on":
        use_sorted_view = True
    elif sorting.mode == "off":
        use_sorted_view = False
    else:
        use_sorted_view = (
            len(motions) >= sorting.auto_min_motions
            or estimate_padding_ratio(original_counts, batch_size)
            >= sorting.auto_padding_ratio
        )

    primary_motions = [
        motion
        for motion in motions
        if not long_motion.enabled or motion.frame_count <= long_motion.max_frames
    ]
    long_motions = [
        motion
        for motion in motions
        if long_motion.enabled and motion.frame_count > long_motion.max_frames
    ]
    if not primary_motions:
        raise ValueError("long-motion filtering removed every motion from the primary dataset")

    effective_mode = "on" if use_sorted_view else "off"
    entries = _entries_for(primary_motions, sorted_view=use_sorted_view)
    candidate = DatasetManifest.create(
        source_root=source,
        entries=entries,
        sorting_mode=effective_mode,
        materialization=sorting.materialization,
    )
    manifest_path = output / "dataset_manifest.json"
    motion_dir = output / "motions" if use_sorted_view else source
    long_motion_dir = output / "long_motions" if long_motion.enabled else None

    if manifest_path.exists():
        existing = DatasetManifest.read(manifest_path)
        if not _manifest_matches(existing, candidate):
            raise ManifestMismatchError(
                f"source dataset changed or preparation options differ for {output}"
            )
        expected_paths = (
            [motion_dir / f"{entry.alias}.pkl" for entry in existing.entries]
            if use_sorted_view
            else []
        )
        if all(path.exists() for path in expected_paths):
            return PreparedDataset(
                motion_dir=motion_dir,
                manifest_path=manifest_path,
                manifest=existing,
                long_motion_dir=long_motion_dir,
                reused=True,
            )
        raise ManifestMismatchError(
            f"prepared manifest exists but materialized files are incomplete: {output}"
        )

    output.mkdir(parents=True, exist_ok=True)
    if use_sorted_view:
            materialize_entries(entries, motion_dir, sorting.materialization)
    if long_motion.enabled:
        long_entries = _entries_for(
            long_motions,
            sorted_view=True,
            index_offset=len(entries),
        )
        if long_entries:
            materialize_entries(long_entries, long_motion_dir, sorting.materialization)
        else:
            long_motion_dir.mkdir(parents=True, exist_ok=True)
        DatasetManifest.create(
            source_root=source,
            entries=long_entries,
            sorting_mode="on",
            materialization=sorting.materialization,
        ).write(output / "long_dataset_manifest.json")
    candidate.write(manifest_path)

    return PreparedDataset(
        motion_dir=motion_dir,
        manifest_path=manifest_path,
        manifest=candidate,
        long_motion_dir=long_motion_dir,
        reused=False,
    )
