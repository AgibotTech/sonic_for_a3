"""Tests for GRAIL terrain profile MJCF generation."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gear_sonic.utils.terrain_profile import patch_mjcf_with_terrain_profile


def test_patch_mjcf_adds_transformed_stair_boxes(tmp_path):
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text("<mujoco><worldbody><geom name='floor' type='plane'/></worldbody></mujoco>")
    profile = tmp_path / "stairs.json"
    profile.write_text(json.dumps({
        "version": 1,
        "object_position": [1.0, 2.0, 0.0],
        "object_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "boxes": [
            {"center": [0.5, 0.0, 0.1], "half_extents": [0.2, 0.4, 0.1]},
            {"center": [0.9, 0.0, 0.2], "half_extents": [0.2, 0.4, 0.2]},
        ],
    }))

    output = patch_mjcf_with_terrain_profile(mjcf, profile, tmp_path / "out.xml")
    root = ET.parse(output).getroot()
    geoms = root.findall("./worldbody/body[@name='grail_terrain']/geom")
    option = root.find("option")

    assert [geom.get("name") for geom in geoms] == ["grail_step_000", "grail_step_001"]
    np.testing.assert_allclose(np.fromstring(geoms[0].get("pos"), sep=" "), [1.5, 2.0, 0.1])
    np.testing.assert_allclose(np.fromstring(geoms[1].get("size"), sep=" "), [0.2, 0.4, 0.2])
    assert option is not None
    assert option.attrib == {"solver": "Newton", "cone": "elliptic", "impratio": "5"}
    assert option.find("flag").attrib == {"multiccd": "enable"}
    assert geoms[0].get("condim") == "3"
    assert geoms[0].get("priority") == "1"
    assert geoms[0].get("friction") == "1 0.005 0.0001"
    assert geoms[0].get("margin") == "0.001"
    assert geoms[0].get("solref") == "0.008 1"
    assert geoms[0].get("solimp") == "0.85 0.95 0.003 0.5 2"


def test_patch_mjcf_rejects_nonpositive_box_size(tmp_path):
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text("<mujoco><worldbody/></mujoco>")
    profile = tmp_path / "bad.json"
    profile.write_text(json.dumps({
        "version": 1,
        "boxes": [{"center": [0, 0, 0], "half_extents": [1, 0, 1]}],
    }))

    with pytest.raises(ValueError, match="positive"):
        patch_mjcf_with_terrain_profile(mjcf, profile)


def test_patch_mjcf_deduplicates_identical_collision_boxes(tmp_path):
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text("<mujoco><worldbody/></mujoco>")
    profile = tmp_path / "stairs.json"
    box = {"center": [0.0, -1.0, 0.1], "half_extents": [1.0, 1.0, 0.1]}
    profile.write_text(json.dumps({"version": 1, "boxes": [box, dict(box)]}))

    output = patch_mjcf_with_terrain_profile(mjcf, profile, tmp_path / "out.xml")
    geoms = ET.parse(output).getroot().findall(
        "./worldbody/body[@name='grail_terrain']/geom"
    )

    assert [geom.get("name") for geom in geoms] == ["grail_step_000"]
