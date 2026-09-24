from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.data_process.augment_motion_speed import build_dataset, validate_dataset


def _write_motion(path: Path, *, key: str, fps: float = 30.0, frames: int = 7) -> None:
    values = np.arange(frames, dtype=np.float32)
    motion = {
        "root_trans_offset": np.stack((values, values + 1, values + 2), axis=-1),
        "pose_aa": np.zeros((frames, 2, 3), dtype=np.float32),
        "dof": values[:, None],
        "root_rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
        "fps": fps,
        "label": "preserved",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({key: motion}, path, compress=3)


def test_build_dataset_materializes_five_speed_variants_and_preserves_payload(tmp_path: Path):
    good6 = tmp_path / "good6"
    gqs = tmp_path / "gqs"
    output = tmp_path / "speed5"
    _write_motion(good6 / "walking_demo.pkl", key="walking_demo", fps=120.0)
    _write_motion(gqs / "date_a" / "motion.pkl", key="motion", fps=30.0)

    summary = build_dataset(
        sources={"good6_30s": good6, "filtered_gqs_30s": gqs},
        expected_counts={"good6_30s": 1, "filtered_gqs_30s": 1},
        output_root=output,
        speeds=(0.9, 0.95, 1.0, 1.05, 1.1),
        workers=1,
        compression=3,
        resume=False,
    )

    assert summary["status"] == "complete"
    assert summary["source_motions"] == 2
    assert summary["output_motions"] == 10
    assert summary["source_frames"] == 14
    assert summary["output_frames"] == 70
    assert (output / "COMPLETE.json").is_file()

    slow_path = (
        output / "filtered_gqs_30s" / "speed_0p95" / "date_a" / "motion__src_filtered_gqs_30s__speed_0p95.pkl"
    )
    assert slow_path.is_file()
    assert not slow_path.is_symlink()
    payload = joblib.load(slow_path)
    assert list(payload) == ["motion__src_filtered_gqs_30s__speed_0p95"]
    augmented = next(iter(payload.values()))
    original = next(iter(joblib.load(gqs / "date_a" / "motion.pkl").values()))
    assert augmented["fps"] == pytest.approx(28.5)
    assert augmented["label"] == "preserved"
    for field in ("root_trans_offset", "pose_aa", "dof", "root_rot"):
        np.testing.assert_array_equal(augmented[field], original[field])

    manifest_rows = [json.loads(line) for line in (output / "semantic_manifest.jsonl").read_text().splitlines()]
    assert [row["source"] for row in manifest_rows] == [
        "filtered_gqs_30s",
        "good6_30s",
    ]
    assert len(manifest_rows[0]["variants"]) == 5


def test_symlink_source_becomes_regular_output_and_resume_repairs_missing_file(tmp_path: Path):
    backing = tmp_path / "backing"
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_motion(backing / "linked.pkl", key="linked", frames=5)
    source.mkdir()
    (source / "linked.pkl").symlink_to(backing / "linked.pkl")

    kwargs = dict(
        sources={"gqs": source},
        expected_counts={"gqs": 1},
        output_root=output,
        speeds=(0.9, 1.0),
        workers=1,
        compression=3,
    )
    build_dataset(**kwargs, resume=False)
    output_file = output / "gqs" / "speed_1p00" / "linked__src_gqs__speed_1p00.pkl"
    output_file.unlink()
    (output / "COMPLETE.json").unlink()

    summary = build_dataset(**kwargs, resume=True)

    assert summary["output_motions"] == 2
    assert output_file.is_file()
    assert not output_file.is_symlink()
    assert (
        validate_dataset(
            sources={"gqs": source},
            output_root=output,
            workers=1,
        )["output_motions"]
        == 2
    )


def test_build_dataset_rejects_duplicate_filename_stems(tmp_path: Path):
    source = tmp_path / "source"
    _write_motion(source / "a" / "duplicate.pkl", key="duplicate-a")
    _write_motion(source / "b" / "duplicate.pkl", key="duplicate-b")

    with pytest.raises(ValueError, match="duplicate filename stem"):
        build_dataset(
            sources={"gqs": source},
            expected_counts={"gqs": 2},
            output_root=tmp_path / "output",
            speeds=(1.0,),
            workers=1,
            compression=3,
            resume=False,
        )


def test_build_dataset_refuses_output_inside_source(tmp_path: Path):
    source = tmp_path / "source"
    _write_motion(source / "motion.pkl", key="motion")

    with pytest.raises(ValueError, match="must not contain"):
        build_dataset(
            sources={"gqs": source},
            expected_counts={"gqs": 1},
            output_root=source / "augmented",
            speeds=(1.0,),
            workers=1,
            compression=3,
            resume=False,
        )
