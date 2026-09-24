"""Versioned manifests shared by evaluation preparation and aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid


SCHEMA_VERSION = 1


class ManifestMismatchError(RuntimeError):
    """Raised when existing evaluation artifacts belong to another run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 for a JSON-compatible value."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class DatasetEntry:
    alias: str
    canonical_key: str
    source_path: str
    frame_count: int
    source_size: int
    source_mtime_ns: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetEntry:
        return cls(
            alias=str(payload["alias"]),
            canonical_key=str(payload["canonical_key"]),
            source_path=str(payload["source_path"]),
            frame_count=int(payload["frame_count"]),
            source_size=int(payload["source_size"]),
            source_mtime_ns=int(payload["source_mtime_ns"]),
        )


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    source_root: str
    source_fingerprint: str
    sorting_mode: str
    materialization: str
    entries: tuple[DatasetEntry, ...]

    @classmethod
    def create(
        cls,
        *,
        source_root: str | Path,
        entries: Sequence[DatasetEntry],
        sorting_mode: str,
        materialization: str,
    ) -> DatasetManifest:
        normalized_entries = tuple(entries)
        fingerprint_input = [
            {
                "canonical_key": entry.canonical_key,
                "source_path": entry.source_path,
                "frame_count": entry.frame_count,
                "source_size": entry.source_size,
                "source_mtime_ns": entry.source_mtime_ns,
            }
            for entry in normalized_entries
        ]
        return cls(
            schema_version=SCHEMA_VERSION,
            source_root=str(Path(source_root).expanduser().resolve()),
            source_fingerprint=canonical_json_fingerprint(fingerprint_input),
            sorting_mode=sorting_mode,
            materialization=materialization,
            entries=normalized_entries,
        )

    def alias_to_canonical(self) -> dict[str, str]:
        return {entry.alias: entry.canonical_key for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        _atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: str | Path) -> DatasetManifest:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        schema_version = int(payload["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ManifestMismatchError(
                f"dataset manifest schema_version={schema_version}, expected {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=schema_version,
            source_root=str(payload["source_root"]),
            source_fingerprint=str(payload["source_fingerprint"]),
            sorting_mode=str(payload["sorting_mode"]),
            materialization=str(payload["materialization"]),
            entries=tuple(DatasetEntry.from_dict(item) for item in payload["entries"]),
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    checkpoint: str
    checkpoint_fingerprint: str
    dataset_fingerprint: str
    dataset_manifest_path: str | None
    config_fingerprint: str
    expected_motion_keys: tuple[str, ...]
    world_size: int
    status: str
    started_at: str
    finished_at: str | None
    wall_time_seconds: float | None

    @classmethod
    def create(
        cls,
        *,
        checkpoint: str | Path,
        checkpoint_fingerprint: str,
        dataset_fingerprint: str,
        config_fingerprint: str,
        expected_motion_keys: Sequence[str],
        world_size: int,
        dataset_manifest_path: str | Path | None = None,
    ) -> RunManifest:
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        return cls(
            schema_version=SCHEMA_VERSION,
            run_id=uuid.uuid4().hex,
            checkpoint=str(Path(checkpoint).expanduser().resolve()),
            checkpoint_fingerprint=checkpoint_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            dataset_manifest_path=(
                str(Path(dataset_manifest_path).expanduser().resolve())
                if dataset_manifest_path is not None
                else None
            ),
            config_fingerprint=config_fingerprint,
            expected_motion_keys=tuple(str(key) for key in expected_motion_keys),
            world_size=int(world_size),
            status="running",
            started_at=_utc_now(),
            finished_at=None,
            wall_time_seconds=None,
        )

    def assert_compatible(self, candidate: RunManifest) -> None:
        fields = (
            "schema_version",
            "checkpoint_fingerprint",
            "dataset_fingerprint",
            "dataset_manifest_path",
            "config_fingerprint",
            "expected_motion_keys",
            "world_size",
        )
        mismatches = [
            field for field in fields if getattr(self, field) != getattr(candidate, field)
        ]
        if mismatches:
            details = ", ".join(
                f"{field}: existing={getattr(self, field)!r}, candidate={getattr(candidate, field)!r}"
                for field in mismatches
            )
            raise ManifestMismatchError(f"run manifest mismatch ({details})")

    def with_completion(self, *, status: str, wall_time_seconds: float) -> RunManifest:
        if status not in {"complete", "failed"}:
            raise ValueError("completion status must be 'complete' or 'failed'")
        if wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be non-negative")
        return replace(
            self,
            status=status,
            finished_at=_utc_now(),
            wall_time_seconds=float(wall_time_seconds),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        _atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        schema_version = int(payload["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ManifestMismatchError(
                f"run manifest schema_version={schema_version}, expected {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=schema_version,
            run_id=str(payload["run_id"]),
            checkpoint=str(payload["checkpoint"]),
            checkpoint_fingerprint=str(payload["checkpoint_fingerprint"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            dataset_manifest_path=(
                str(payload["dataset_manifest_path"])
                if payload.get("dataset_manifest_path") is not None
                else None
            ),
            config_fingerprint=str(payload["config_fingerprint"]),
            expected_motion_keys=tuple(str(key) for key in payload["expected_motion_keys"]),
            world_size=int(payload["world_size"]),
            status=str(payload["status"]),
            started_at=str(payload["started_at"]),
            finished_at=(
                str(payload["finished_at"]) if payload.get("finished_at") is not None else None
            ),
            wall_time_seconds=(
                float(payload["wall_time_seconds"])
                if payload.get("wall_time_seconds") is not None
                else None
            ),
        )


def checkpoint_fingerprint(path: str | Path, sample_bytes: int = 1024 * 1024) -> str:
    """Fingerprint a checkpoint without rereading the entire large file."""

    checkpoint_path = Path(path).expanduser().resolve()
    stat = checkpoint_path.stat()
    digest = hashlib.sha256()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with checkpoint_path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if stat.st_size > sample_bytes:
            stream.seek(max(0, stat.st_size - sample_bytes))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()
