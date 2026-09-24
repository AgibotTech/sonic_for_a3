#!/usr/bin/env python3
"""Regression check for the hull-only A3 loop MJCF variant."""

from __future__ import annotations

import copy
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
MJCF_DIR = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf"
ORIGINAL = MJCF_DIR / "a3_t2d5_loop.xml"
DERIVED = MJCF_DIR / "a3_t2d5_loop_hull_optimized.xml"
MESH_RENAMES = {
    "left_ankle_roll_Link_hull": (
        "left_ankle_roll_Link_hull_optimized",
        "left_ankle_roll_Link_hull_optimized.STL",
    ),
    "right_ankle_roll_Link_hull": (
        "right_ankle_roll_Link_hull_optimized",
        "right_ankle_roll_Link_hull_optimized.STL",
    ),
}
RAW_XML_REPLACEMENTS = (
    (
        'name="left_ankle_roll_Link_hull" content_type="model/stl" file="left_ankle_roll_Link_hull.STL"',
        'name="left_ankle_roll_Link_hull_optimized" content_type="model/stl" file="left_ankle_roll_Link_hull_optimized.STL"',
    ),
    (
        'name="right_ankle_roll_Link_hull" content_type="model/stl" file="right_ankle_roll_Link_hull.STL"',
        'name="right_ankle_roll_Link_hull_optimized" content_type="model/stl" file="right_ankle_roll_Link_hull_optimized.STL"',
    ),
    ('mesh="left_ankle_roll_Link_hull"', 'mesh="left_ankle_roll_Link_hull_optimized"'),
    ('mesh="right_ankle_roll_Link_hull"', 'mesh="right_ankle_roll_Link_hull_optimized"'),
)


def canonical_xml(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8")


def expected_derived_root(original_root: ET.Element) -> ET.Element:
    expected = copy.deepcopy(original_root)
    asset = expected.find("asset")
    assert asset is not None
    for mesh in asset.findall("mesh"):
        replacement = MESH_RENAMES.get(mesh.get("name"))
        if replacement is not None:
            mesh.set("name", replacement[0])
            mesh.set("file", replacement[1])
    for geom in expected.findall(".//geom"):
        replacement = MESH_RENAMES.get(geom.get("mesh"))
        if replacement is not None:
            geom.set("mesh", replacement[0])
    return expected


class HullOptimizedMjcfTest(unittest.TestCase):
    def test_only_two_ankle_roll_collision_meshes_change(self) -> None:
        self.assertTrue(ORIGINAL.is_file())
        self.assertTrue(DERIVED.is_file())

        expected_text = ORIGINAL.read_text(encoding="utf-8")
        for original, replacement in RAW_XML_REPLACEMENTS:
            self.assertEqual(expected_text.count(original), 1)
            expected_text = expected_text.replace(original, replacement)
        self.assertEqual(DERIVED.read_text(encoding="utf-8"), expected_text)

        # Parse complete XML through MuJoCo, including both STL mesh assets.
        mujoco.MjModel.from_xml_path(str(ORIGINAL))
        mujoco.MjModel.from_xml_path(str(DERIVED))

        original_root = ET.parse(ORIGINAL).getroot()
        derived_root = ET.parse(DERIVED).getroot()
        original_asset = original_root.find("asset")
        derived_asset = derived_root.find("asset")
        assert original_asset is not None and derived_asset is not None

        original_meshes = original_asset.findall("mesh")
        derived_meshes = derived_asset.findall("mesh")
        self.assertEqual(len(original_meshes), len(derived_meshes))
        changed_file_references = [
            (old.get("file"), new.get("file"))
            for old, new in zip(original_meshes, derived_meshes)
            if old.get("file") != new.get("file")
        ]
        self.assertEqual(
            changed_file_references,
            [
                ("left_ankle_roll_Link_hull.STL", "left_ankle_roll_Link_hull_optimized.STL"),
                ("right_ankle_roll_Link_hull.STL", "right_ankle_roll_Link_hull_optimized.STL"),
            ],
        )
        self.assertEqual(canonical_xml(expected_derived_root(original_root)), canonical_xml(derived_root))


if __name__ == "__main__":
    unittest.main()
