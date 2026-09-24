#!/usr/bin/env python3
"""Tests for the deterministic symlink-only Kimodo fixed eval-set builder."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from build_kimodo_training_eval_set import build_eval_set


def write_csv(path: Path, rows: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame"])
        writer.writerows([[index] for index in range(rows)])


class BuildKimodoTrainingEvalSetTest(unittest.TestCase):
    def test_seed_stability_symlinks_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "kimodo"
            pkl = root / "kimodo_pkl" / "kimodo"
            source.mkdir(parents=True)
            pkl.mkdir(parents=True)
            for name, rows in (("a", 30), ("b", 45), ("c", 60), ("d", 75)):
                write_csv(source / f"{name}.csv", rows)
                (pkl / f"{name}.pkl").touch()
            first = build_eval_set(source, root / "out_one", target_seconds=4.0, seed=7, training_pkl_dir=pkl)
            rebuilt = build_eval_set(source, root / "out_one", target_seconds=4.0, seed=7, training_pkl_dir=pkl)
            second = build_eval_set(source, root / "out_two", target_seconds=4.0, seed=7, training_pkl_dir=pkl)
            self.assertGreaterEqual(first["total_seconds"], 4.0)
            self.assertEqual(first["clip_count"], rebuilt["clip_count"])
            self.assertEqual(first["clip_count"], second["clip_count"])
            with Path(first["manifest_path"]).open(newline="") as handle:
                first_rows = list(csv.DictReader(handle))
            with Path(second["manifest_path"]).open(newline="") as handle:
                second_rows = list(csv.DictReader(handle))
            self.assertEqual([row["stem"] for row in first_rows], [row["stem"] for row in second_rows])
            self.assertEqual(len({row["stem"] for row in first_rows}), len(first_rows))
            for row in first_rows:
                link = root / "out_one" / row["symlink_csv"]
                self.assertTrue(link.is_symlink())
                self.assertTrue(link.is_file())
                self.assertEqual(Path(row["source_csv"]), link.resolve())

    def test_missing_training_pkl_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            pkl = root / "pkl"
            source.mkdir()
            pkl.mkdir()
            write_csv(source / "only.csv", 30)
            (pkl / "different.pkl").touch()
            with self.assertRaisesRegex(ValueError, "stems must match exactly"):
                build_eval_set(source, root / "out", target_seconds=1.0, training_pkl_dir=pkl)


if __name__ == "__main__":
    unittest.main()
