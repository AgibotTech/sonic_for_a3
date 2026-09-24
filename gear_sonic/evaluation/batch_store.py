"""Atomic per-rank persistence for scalable evaluation batches."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import torch

from .schemas import ManifestMismatchError, SCHEMA_VERSION


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode())
            _update_digest(digest, value[key])
    elif isinstance(value, np.ndarray):
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    elif isinstance(value, torch.Tensor):
        _update_digest(digest, value.detach().cpu().numpy())
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(repr(value).encode())


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, {key: value for key, value in payload.items() if key != "fingerprint"})
    return digest.hexdigest()


class BatchStore:
    """Store compact evaluation payloads under rank-specific directories."""

    def __init__(self, root: str | Path, run_id: str):
        self.root = Path(root).expanduser().resolve()
        self.run_id = str(run_id)

    def path_for(self, rank: int, batch_idx: int) -> Path:
        if rank < 0 or batch_idx < 0:
            raise ValueError("rank and batch_idx must be non-negative")
        return self.root / f"rank_{rank:03d}" / f"batch_{batch_idx:06d}.pt"

    def _validate(self, payload: Mapping[str, Any], *, rank: int, batch_idx: int) -> None:
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ManifestMismatchError("batch schema_version does not match")
        if str(payload.get("run_id")) != self.run_id:
            raise ManifestMismatchError(
                f"batch run_id={payload.get('run_id')!r} does not match {self.run_id!r}"
            )
        if int(payload.get("rank", -1)) != rank:
            raise ManifestMismatchError("batch rank does not match its storage path")
        if int(payload.get("batch_idx", -1)) != batch_idx:
            raise ManifestMismatchError("batch_idx does not match its storage path")

    def write(
        self,
        *,
        rank: int,
        batch_idx: int,
        payload: Mapping[str, Any],
    ) -> Path:
        self._validate(payload, rank=rank, batch_idx=batch_idx)
        final_path = self.path_for(rank, batch_idx)
        fingerprint = _payload_fingerprint(payload)
        if final_path.exists():
            existing = torch.load(final_path, map_location="cpu", weights_only=False)
            self._validate(existing, rank=rank, batch_idx=batch_idx)
            existing_fingerprint = existing.get("fingerprint") or _payload_fingerprint(existing)
            if existing_fingerprint != fingerprint:
                raise ManifestMismatchError(f"batch already exists with different contents: {final_path}")
            return final_path

        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
        saved_payload = dict(payload)
        saved_payload["fingerprint"] = fingerprint
        try:
            with temporary.open("wb") as stream:
                torch.save(saved_payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return final_path

    def completed_indices(self, rank: int) -> set[int]:
        rank_dir = self.root / f"rank_{rank:03d}"
        pattern = re.compile(r"batch_(\d{6})\.pt$")
        return {
            int(match.group(1))
            for path in rank_dir.glob("batch_*.pt")
            if (match := pattern.match(path.name)) is not None
        }

    def load_all(self) -> list[dict[str, Any]]:
        batches = []
        for path in sorted(self.root.glob("rank_*/batch_*.pt")):
            rank_match = re.fullmatch(r"rank_(\d{3})", path.parent.name)
            batch_match = re.fullmatch(r"batch_(\d{6})\.pt", path.name)
            if rank_match is None or batch_match is None:
                continue
            rank = int(rank_match.group(1))
            batch_idx = int(batch_match.group(1))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self._validate(payload, rank=rank, batch_idx=batch_idx)
            expected_fingerprint = payload.get("fingerprint")
            if expected_fingerprint != _payload_fingerprint(payload):
                raise ManifestMismatchError(f"batch payload fingerprint mismatch: {path}")
            batches.append(payload)
        return sorted(batches, key=lambda item: (int(item["rank"]), int(item["batch_idx"])))
