"""Convert an assembled GRAIL stair profile into collision-only MJCF geoms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


_PATCH_VERSION = b"terrain-contact-v5-contact-margin"


def load_terrain_profile(path):
    path = Path(path).expanduser().resolve()
    profile = json.loads(path.read_text())
    if profile.get("version") != 1:
        raise ValueError(f"Unsupported terrain profile version: {profile.get('version')}")
    boxes = profile.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("Terrain profile must contain at least one box")
    for index, box in enumerate(boxes):
        center = np.asarray(box.get("center"), dtype=np.float64)
        half_extents = np.asarray(box.get("half_extents"), dtype=np.float64)
        if center.shape != (3,) or half_extents.shape != (3,):
            raise ValueError(f"Terrain box {index} center/half_extents must be XYZ vectors")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half_extents)):
            raise ValueError(f"Terrain box {index} contains non-finite values")
        if np.any(half_extents <= 0.0):
            raise ValueError(f"Terrain box {index} half_extents must be positive")
    return profile


def _absolute_compiler_paths(mjcf_path, compiler):
    if compiler is None:
        return
    for attribute in ("meshdir", "texturedir"):
        value = compiler.get(attribute)
        if value and not Path(value).is_absolute():
            compiler.set(attribute, str((mjcf_path.parent / value).resolve()))


def _unique_boxes(boxes):
    unique = []
    seen = set()
    for box in boxes:
        center = np.asarray(box["center"], dtype=np.float64)
        half_extents = np.asarray(box["half_extents"], dtype=np.float64)
        key = tuple(np.round(np.concatenate((center, half_extents)), decimals=9))
        if key in seen:
            continue
        seen.add(key)
        unique.append(box)
    return unique


def _configure_terrain_solver(root):
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        root.insert(1 if compiler is not None else 0, option)
    option.set("solver", "Newton")
    option.set("cone", "elliptic")
    option.set("impratio", "5")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    # Convex foot meshes need a multi-point manifold on flat stair-box faces;
    # a single CCD contact lets the sole pivot and translate around one point.
    flag.set("multiccd", "enable")


def patch_mjcf_with_terrain_profile(mjcf_path, profile_path, output_path=None):
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    profile_path = Path(profile_path).expanduser().resolve()
    profile = load_terrain_profile(profile_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    _configure_terrain_solver(root)
    _absolute_compiler_paths(mjcf_path, root.find("compiler"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {mjcf_path}")

    object_position = np.asarray(profile.get("object_position", [0.0, 0.0, 0.0]), dtype=np.float64)
    object_quat = np.asarray(
        profile.get("object_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if object_position.shape != (3,) or object_quat.shape != (4,):
        raise ValueError("object_position and object_quaternion_wxyz have invalid shape")
    norm = float(np.linalg.norm(object_quat))
    if norm <= 1e-8:
        raise ValueError("object_quaternion_wxyz must be non-zero")
    object_quat = object_quat / norm
    rotation = Rotation.from_quat(
        [object_quat[1], object_quat[2], object_quat[3], object_quat[0]])

    terrain_body = ET.SubElement(worldbody, "body", {"name": "grail_terrain"})
    for index, box in enumerate(_unique_boxes(profile["boxes"])):
        local_center = np.asarray(box["center"], dtype=np.float64)
        center = object_position + rotation.apply(local_center)
        half_extents = np.asarray(box["half_extents"], dtype=np.float64)
        ET.SubElement(
            terrain_body,
            "geom",
            {
                "name": f"grail_step_{index:03d}",
                "type": "box",
                "pos": " ".join(f"{value:.9g}" for value in center),
                "size": " ".join(f"{value:.9g}" for value in half_extents),
                "quat": " ".join(f"{value:.9g}" for value in object_quat),
                "contype": "1",
                "conaffinity": "7",
                "condim": "3",
                "priority": "1",
                "friction": "1 0.005 0.0001",
                # Keep a small speculative-contact shell, matching the contact
                # persistence that PhysX provides around the stair surface.
                "margin": "0.001",
                "solref": "0.008 1",
                "solimp": "0.85 0.95 0.003 0.5 2",
                "rgba": "0.55 0.58 0.62 1",
            },
        )

    if output_path is None:
        digest = hashlib.sha256(
            _PATCH_VERSION + mjcf_path.read_bytes() + profile_path.read_bytes()
        ).hexdigest()[:16]
        output_path = Path(tempfile.gettempdir()) / "a3_grail_terrain_mjcf" / f"{digest}.xml"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
