#!/usr/bin/env python3
"""Tests for deterministic failure-top50 (adaptive sampler proxy) manifests."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from build_adaptive_top50_eval_manifest import build_top50_manifest


def write_csv(path: Path) -> None:
    path.write_text("Frame\n0\n1\n", encoding="utf-8")


class BuildAdaptiveTop50EvalManifestTest(unittest.TestCase):
    def test_active_filter_sorting_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            pkl = root / "pkl"
            source.mkdir()
            pkl.mkdir()
            for stem in ("alpha", "beta", "gamma", "inactive"):
                write_csv(source / f"{stem}.csv")
                (pkl / f"{stem}.pkl").touch()
            adaptive = root / "adaptive_sampling_motions_step_008000.csv"
            with adaptive.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("motion_key", "motion_sampling_prob", "max_bin_sampling_prob", "is_active"),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"motion_key": "beta", "motion_sampling_prob": "0.9", "max_bin_sampling_prob": "0.7", "is_active": "true"},
                        {"motion_key": "alpha", "motion_sampling_prob": "0.9", "max_bin_sampling_prob": "0.6", "is_active": "1"},
                        {"motion_key": "gamma", "motion_sampling_prob": "0.1", "max_bin_sampling_prob": "0.2", "is_active": "True"},
                        {"motion_key": "inactive", "motion_sampling_prob": "1.0", "max_bin_sampling_prob": "1.0", "is_active": "false"},
                    )
                )
            result = build_top50_manifest(adaptive, source, root / "out", top=3, training_pkl_dir=pkl)
            self.assertEqual(result["checkpoint_step"], 8000)
            with Path(result["manifest_path"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["motion_key"] for row in rows], ["alpha", "beta", "gamma"])
            self.assertEqual([row["is_active"] for row in rows], ["true", "true", "true"])
            self.assertTrue(all((root / "out" / row["symlink_csv"]).is_file() for row in rows))

    def test_missing_source_fails_instead_of_silently_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            pkl = root / "pkl"
            source.mkdir()
            pkl.mkdir()
            write_csv(source / "other.csv")
            (pkl / "other.pkl").touch()
            adaptive = root / "adaptive.csv"
            adaptive.write_text("motion_key,motion_sampling_prob,is_active\nmissing,1.0,true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot resolve"):
                build_top50_manifest(adaptive, source, root / "out", top=1, training_pkl_dir=pkl)


if __name__ == "__main__":
    unittest.main()
