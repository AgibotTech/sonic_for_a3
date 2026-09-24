#!/usr/bin/env python3
"""Regression tests for native-30fps CSV loading in the A3 MuJoCo runner."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_SCRIPT = REPO_ROOT / "gear_sonic/scripts/sim2sim_a3_mujoco.py"
SPEC = importlib.util.spec_from_file_location("sim2sim_a3_mujoco", SIM_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SIM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIM
SPEC.loader.exec_module(SIM)


def write_flat_csv(path: Path, *, rows: int, use_dof_aliases: bool, duplicate_first_joint: bool = False) -> None:
    headers = [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    headers.extend(f"{name}_dof" if use_dof_aliases else name for name in SIM.A3_CSV_JOINT_NAMES)
    if duplicate_first_joint:
        headers.append(SIM.A3_CSV_JOINT_NAMES[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for idx in range(rows):
            row = {name: "0" for name in headers}
            row["root_translateX"] = str(idx)
            writer.writerow(row)


class KimodoSim2SimCsvIoTest(unittest.TestCase):
    def test_real_kimodo_dof_alias_and_native_duration(self) -> None:
        source = REPO_ROOT / "a3_data/a3_csv_boxing_generated_by_kimodo3000/000000.csv"
        root_pos, root_quat, dof, raw_rows = SIM.load_a3_flat_csv(
            source, source_fps=30.0, frame_stride=1
        )
        self.assertEqual(root_pos.shape[0], raw_rows)
        self.assertEqual(root_quat.shape, (raw_rows, 4))
        self.assertEqual(dof.shape, (raw_rows, SIM.NUM_POLICY_DOFS))
        self.assertAlmostEqual(SIM.csv_reference_duration_seconds(raw_rows, 30.0), raw_rows / 30.0)

    def test_bare_columns_and_legacy_stride_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy_120fps.csv"
            write_flat_csv(source, rows=8, use_dof_aliases=False)
            root_pos, _, dof, raw_rows = SIM.load_a3_flat_csv(
                source, source_fps=30.0, frame_stride=4
            )
            self.assertEqual(raw_rows, 8)
            self.assertEqual(root_pos.shape[0], 2)
            self.assertEqual(dof.shape, (2, SIM.NUM_POLICY_DOFS))
            # Legacy behavior: 120fps input sampled every fourth row becomes a
            # 30fps reference stream, so 8 raw rows span 2/30 seconds.
            self.assertAlmostEqual(
                SIM.csv_reference_duration_seconds(root_pos.shape[0], 30.0), 8.0 / (4.0 * 30.0)
            )

    def test_ambiguous_bare_and_dof_columns_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ambiguous.csv"
            write_flat_csv(source, rows=2, use_dof_aliases=True, duplicate_first_joint=True)
            with self.assertRaisesRegex(ValueError, "ambiguous A3 joint columns"):
                SIM.load_a3_flat_csv(source, source_fps=30.0, frame_stride=1)


if __name__ == "__main__":
    unittest.main()
