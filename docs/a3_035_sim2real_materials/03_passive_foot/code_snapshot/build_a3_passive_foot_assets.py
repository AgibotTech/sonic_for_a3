#!/usr/bin/env python3
"""Build A3 passive-foot assets from v8 collision meshes and draft foot joints."""

from __future__ import annotations

import argparse
import copy
import math
from collections import defaultdict
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
A3_URDF_DIR = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/a3"
A3_MJCF_DIR = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf"
FOOT_DRAFT_DIR = REPO_ROOT / "urdfs/a3_compliant_foot_draft1"

BASE_URDF = A3_URDF_DIR / "model_collision_v8_foot.urdf"
DRAFT_URDF = FOOT_DRAFT_DIR / "model.urdf"
OUT_URDF = A3_URDF_DIR / "model_collision_optimized_passive_foot.urdf"

DRAFT_MJCF = FOOT_DRAFT_DIR / "a3_t2d5.xml"
DRAFT_LOOP_MJCF = FOOT_DRAFT_DIR / "a3_t2d5_loop.xml"
OUT_MJCF = A3_MJCF_DIR / "a3_t2d5_passive_foot.xml"
OUT_LOOP_MJCF = A3_MJCF_DIR / "a3_t2d5_loop_passive_foot.xml"

VISUAL_FOOT_DIR = "meshes_compliant_foot"
COLLISION_DIR = "meshes_collision_optimized"
VISUAL_DIR = "meshes"

FOOT_PARTS = ("rear", "forefoot", "toe")
PASSIVE_FOOT_JOINTS = (
    "left_foot_forefoot_joint",
    "left_foot_toe_joint",
    "right_foot_forefoot_joint",
    "right_foot_toe_joint",
)
FOOT_SELF_EXCLUDE_PAIRS = (
    ("left_ankle_roll_Link", "left_foot_forefoot_Link"),
    ("left_foot_forefoot_Link", "left_foot_toe_Link"),
    ("left_ankle_roll_Link", "left_foot_toe_Link"),
    ("right_ankle_roll_Link", "right_foot_forefoot_Link"),
    ("right_foot_forefoot_Link", "right_foot_toe_Link"),
    ("right_ankle_roll_Link", "right_foot_toe_Link"),
)
COLLISION_MODE = "optimized"
FITTED_COLLISION_MODES = ("fitted", "fitted-optimized")
FULL_HULL_PASSIVE_DAMPING_SCALE = 2.0
FULL_HULL_FOREFOOT_ARMATURE = 1.0e-4
FULL_HULL_TOE_ARMATURE = 3.0e-5
FULL_HULL_FOOT_CONTACT_SOLREF = "0.010 1"
FITTED_FOREFOOT_STIFFNESS_NM_RAD = 2396.666
FITTED_TOE_STIFFNESS_NM_RAD = 27.58407
FITTED_FOREFOOT_FLEXION_LIMIT_RAD = 0.06606093
FITTED_TOE_FLEXION_LIMIT_RAD = 0.2305825
FITTED_FOOT_CONTACT_SOLREF = "0.018 1"
FITTED_FOOT_CONTACT_SOLIMP = "0.80 0.95 0.0129123 0.5 2"
FITTED_FOOT_CONTACT_MARGIN = "0.001"


def _configure_outputs(collision_mode: str) -> None:
    global OUT_URDF, OUT_MJCF, OUT_LOOP_MJCF, COLLISION_DIR, COLLISION_MODE

    COLLISION_MODE = collision_mode
    if collision_mode == "optimized":
        OUT_URDF = A3_URDF_DIR / "model_collision_optimized_passive_foot.urdf"
        OUT_MJCF = A3_MJCF_DIR / "a3_t2d5_passive_foot.xml"
        OUT_LOOP_MJCF = A3_MJCF_DIR / "a3_t2d5_loop_passive_foot.xml"
        COLLISION_DIR = "meshes_collision_optimized"
    elif collision_mode == "full-hull":
        OUT_URDF = A3_URDF_DIR / "model_collision_optimized_passive_foot_twostage_fullhull.urdf"
        OUT_MJCF = A3_MJCF_DIR / "a3_t2d5_passive_foot_twostage_fullhull.xml"
        OUT_LOOP_MJCF = A3_MJCF_DIR / "a3_t2d5_loop_passive_foot_twostage_fullhull.xml"
        COLLISION_DIR = "meshes_collision_twostage_fullhull"
    elif collision_mode == "fitted":
        OUT_URDF = A3_URDF_DIR / "model_collision_optimized_passive_foot_twostage_fit.urdf"
        OUT_MJCF = A3_MJCF_DIR / "a3_t2d5_passive_foot_twostage_fit.xml"
        OUT_LOOP_MJCF = A3_MJCF_DIR / "a3_t2d5_loop_passive_foot_twostage_fit.xml"
        COLLISION_DIR = "meshes_collision_twostage_fullhull"
    elif collision_mode == "fitted-optimized":
        OUT_URDF = A3_URDF_DIR / "model_collision_optimized_passive_foot_twostage_fit_optimized.urdf"
        OUT_MJCF = A3_MJCF_DIR / "a3_t2d5_passive_foot_twostage_fit_optimized.xml"
        OUT_LOOP_MJCF = A3_MJCF_DIR / "a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
        COLLISION_DIR = "meshes_collision_optimized"
    else:
        raise ValueError(f"unknown collision mode: {collision_mode}")


def _copy_full_hull_collision_meshes() -> None:
    if COLLISION_MODE not in ("full-hull", "fitted"):
        return
    dest_dir = A3_URDF_DIR / COLLISION_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale_mesh in dest_dir.glob("*.STL"):
        stale_mesh.unlink()
    for side in ("left", "right"):
        for part in FOOT_PARTS:
            src = FOOT_DRAFT_DIR / "meshes" / f"{side}_foot_{part}_collision.STL"
            dst = dest_dir / f"{side}_foot_{part}_collision_convex.STL"
            shutil.copyfile(src, dst)


def _link(root: ET.Element, name: str) -> ET.Element:
    elem = root.find(f"link[@name='{name}']")
    if elem is None:
        raise ValueError(f"missing link: {name}")
    return elem


def _joint(root: ET.Element, name: str) -> ET.Element:
    elem = root.find(f"joint[@name='{name}']")
    if elem is None:
        raise ValueError(f"missing joint: {name}")
    return elem


def _remove_children(elem: ET.Element, tag: str) -> None:
    for child in list(elem.findall(tag)):
        elem.remove(child)


def _rewrite_mesh_dir_references(root: ET.Element) -> None:
    for mesh in root.findall(".//mesh"):
        attr = "filename" if "filename" in mesh.attrib else "file" if "file" in mesh.attrib else None
        if attr is None:
            continue
        value = mesh.attrib[attr]
        if value.startswith("meshes_compliant_foot_v8/"):
            mesh.attrib[attr] = value.replace("meshes_compliant_foot_v8/", f"{VISUAL_FOOT_DIR}/", 1)
        elif value.startswith("meshes_collision_v8_foot/"):
            collision_dir = COLLISION_DIR if "_foot_" in Path(value).name else "meshes_collision_optimized"
            mesh.attrib[attr] = value.replace("meshes_collision_v8_foot/", f"{collision_dir}/", 1)


def _rewrite_link_meshes(link: ET.Element, side: str, part: str | None = None) -> None:
    for visual in link.findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        name = mesh.attrib["filename"]
        if part is None:
            part_from_name = Path(name).stem.split("_foot_")[-1].replace("_visual", "")
        else:
            part_from_name = part
        mesh.attrib["filename"] = f"{VISUAL_FOOT_DIR}/{side}_foot_{part_from_name}_visual.STL"

    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        name = mesh.attrib["filename"]
        if part is None:
            part_from_name = Path(name).stem.split("_foot_")[-1].replace("_collision", "")
        else:
            part_from_name = part
        mesh.attrib["filename"] = (
            f"{COLLISION_DIR}/{side}_foot_{part_from_name}_collision_convex.STL"
        )


def _is_passive_foot_joint(joint_name: str) -> bool:
    return joint_name in PASSIVE_FOOT_JOINTS


def _is_foot_collision_mesh(mesh_name: str) -> bool:
    return any(
        mesh_name.endswith(f"_foot_{part}_collision_convex") for part in FOOT_PARTS
    )


def _format_float(value: float) -> str:
    return f"{value:.9g}"


def _fitted_joint_values(joint_name: str) -> tuple[float, float, float, float]:
    if "forefoot_joint" in joint_name:
        return (
            160.0,
            FITTED_FOREFOOT_STIFFNESS_NM_RAD,
            FITTED_FOREFOOT_FLEXION_LIMIT_RAD,
            FULL_HULL_FOREFOOT_ARMATURE,
        )
    if "toe_joint" in joint_name:
        return (
            64.0,
            FITTED_TOE_STIFFNESS_NM_RAD,
            FITTED_TOE_FLEXION_LIMIT_RAD,
            FULL_HULL_TOE_ARMATURE,
        )
    raise ValueError(f"unknown fitted passive foot joint: {joint_name}")


def _fitted_damping(joint_name: str, source_damping: float) -> float:
    source_stiffness, fitted_stiffness, _, _ = _fitted_joint_values(joint_name)
    return (
        source_damping
        * FULL_HULL_PASSIVE_DAMPING_SCALE
        * math.sqrt(fitted_stiffness / source_stiffness)
    )


def _apply_full_hull_urdf_joint_params(joint: ET.Element) -> None:
    joint_name = joint.attrib["name"]
    if COLLISION_MODE not in ("full-hull", *FITTED_COLLISION_MODES) or not _is_passive_foot_joint(joint_name):
        return
    dynamics = joint.find("dynamics")
    if dynamics is None:
        dynamics = ET.SubElement(joint, "dynamics")
    if "damping" in dynamics.attrib:
        source_damping = float(dynamics.attrib["damping"])
        if COLLISION_MODE in FITTED_COLLISION_MODES:
            damping = _fitted_damping(joint_name, source_damping)
        else:
            damping = source_damping * FULL_HULL_PASSIVE_DAMPING_SCALE
        dynamics.attrib["damping"] = _format_float(damping)

    if COLLISION_MODE in FITTED_COLLISION_MODES:
        _, _, flexion_limit, _ = _fitted_joint_values(joint_name)
        limit = joint.find("limit")
        if limit is not None:
            limit.attrib["lower"] = _format_float(-flexion_limit)
            limit.attrib["upper"] = _format_float(flexion_limit)


def _apply_full_hull_mjcf_joint_params(joint: ET.Element) -> None:
    if COLLISION_MODE not in ("full-hull", *FITTED_COLLISION_MODES):
        return
    joint_name = joint.attrib.get("name", "")
    if not _is_passive_foot_joint(joint_name):
        return

    if COLLISION_MODE in FITTED_COLLISION_MODES:
        _, fitted_stiffness, flexion_limit, armature = _fitted_joint_values(joint_name)
        joint.attrib["stiffness"] = _format_float(fitted_stiffness)
        joint.attrib["range"] = f"{_format_float(-flexion_limit)} {_format_float(flexion_limit)}"
        if "damping" in joint.attrib:
            joint.attrib["damping"] = _format_float(
                _fitted_damping(joint_name, float(joint.attrib["damping"]))
            )
        joint.attrib["armature"] = _format_float(armature)
    else:
        if "damping" in joint.attrib:
            joint.attrib["damping"] = _format_float(
                float(joint.attrib["damping"]) * FULL_HULL_PASSIVE_DAMPING_SCALE
            )
        if "forefoot_joint" in joint_name:
            joint.attrib["armature"] = _format_float(FULL_HULL_FOREFOOT_ARMATURE)
        elif "toe_joint" in joint_name:
            joint.attrib["armature"] = _format_float(FULL_HULL_TOE_ARMATURE)


def build_urdf() -> None:
    base_tree = ET.parse(BASE_URDF)
    base_root = base_tree.getroot()
    _rewrite_mesh_dir_references(base_root)
    draft_root = ET.parse(DRAFT_URDF).getroot()

    for side in ("left", "right"):
        ankle_name = f"{side}_ankle_roll_Link"
        ankle_link = _link(base_root, ankle_name)
        draft_ankle = _link(draft_root, ankle_name)

        _remove_children(ankle_link, "inertial")
        _remove_children(ankle_link, "visual")
        _remove_children(ankle_link, "collision")

        ankle_link.insert(0, copy.deepcopy(draft_ankle.find("inertial")))
        for visual in draft_ankle.findall("visual"):
            elem = copy.deepcopy(visual)
            mesh = elem.find("geometry/mesh")
            mesh.attrib["filename"] = f"{VISUAL_FOOT_DIR}/{side}_foot_rear_visual.STL"
            ankle_link.append(elem)
        for collision in draft_ankle.findall("collision"):
            elem = copy.deepcopy(collision)
            mesh = elem.find("geometry/mesh")
            mesh.attrib["filename"] = f"{COLLISION_DIR}/{side}_foot_rear_collision_convex.STL"
            ankle_link.append(elem)

        for link_name in (f"{side}_foot_forefoot_Link", f"{side}_foot_toe_Link"):
            new_link = copy.deepcopy(_link(draft_root, link_name))
            _rewrite_link_meshes(new_link, side)
            base_root.append(new_link)

        for joint_name in (f"{side}_foot_forefoot_joint", f"{side}_foot_toe_joint"):
            joint = copy.deepcopy(_joint(draft_root, joint_name))
            _apply_full_hull_urdf_joint_params(joint)
            base_root.append(joint)

    ET.indent(base_tree, space="  ")
    OUT_URDF.write_text(
        ET.tostring(base_root, encoding="unicode", short_empty_elements=True) + "\n"
    )


def _collision_records_from_urdf(path: Path) -> dict[str, list[dict[str, str]]]:
    root = ET.parse(path).getroot()
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for idx, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            origin = collision.find("origin")
            filename = mesh.attrib["filename"]
            records[link_name].append(
                {
                    "name": collision.attrib.get("name", f"{link_name}_collision_{idx:02d}"),
                    "filename": filename,
                    "xyz": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                    "rpy": origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                }
            )
    return records


def _rpy_to_quat_wxyz(rpy: str) -> str | None:
    roll, pitch, yaw = (float(v) for v in rpy.split())
    if abs(roll) + abs(pitch) + abs(yaw) < 1e-12:
        return None
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    quat = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return " ".join(f"{v:.9g}" for v in quat)


def _prefix_visual_mesh_file(mesh: ET.Element) -> None:
    file_attr = mesh.attrib.get("file")
    if not file_attr or "/" in file_attr:
        return
    stem = Path(file_attr).stem
    if "_foot_" in stem and stem.endswith("_visual"):
        mesh.attrib["file"] = f"{VISUAL_FOOT_DIR}/{file_attr}"
    else:
        mesh.attrib["file"] = f"{VISUAL_DIR}/{file_attr}"


def _ensure_foot_self_collision_excludes(root: ET.Element) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.Element("contact")
        equality = root.find("equality")
        if equality is None:
            root.append(contact)
        else:
            root.insert(list(root).index(equality), contact)

    existing = {
        frozenset((exclude.attrib.get("body1", ""), exclude.attrib.get("body2", "")))
        for exclude in contact.findall("exclude")
    }
    for body1, body2 in FOOT_SELF_EXCLUDE_PAIRS:
        if frozenset((body1, body2)) not in existing:
            ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})


def build_mjcf(source: Path, output: Path, collision_records: dict[str, list[dict[str, str]]]) -> None:
    tree = ET.parse(source)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib["meshdir"] = "../urdf/a3"

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"{source} has no <asset> section")

    for mesh in list(asset.findall("mesh")):
        name = mesh.attrib.get("name", "")
        if "_foot_" in name and name.endswith("_collision"):
            asset.remove(mesh)

    existing_mesh_names = {mesh.attrib["name"] for mesh in asset.findall("mesh")}
    for mesh in asset.findall("mesh"):
        _prefix_visual_mesh_file(mesh)

    for records in collision_records.values():
        for record in records:
            mesh_name = Path(record["filename"]).stem
            if mesh_name in existing_mesh_names:
                continue
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": mesh_name,
                    "content_type": "model/stl",
                    "file": record["filename"],
                },
            )
            existing_mesh_names.add(mesh_name)

    for body in root.iter("body"):
        body_name = body.attrib.get("name")
        for joint in body.findall("joint"):
            _apply_full_hull_mjcf_joint_params(joint)
        for geom in list(body.findall("geom")):
            if geom.attrib.get("class") == "collision":
                body.remove(geom)
        if body_name not in collision_records:
            continue
        for record in collision_records[body_name]:
            geom_attrs = {
                "class": "collision",
                "type": "mesh",
                "mesh": Path(record["filename"]).stem,
            }
            if record["xyz"] != "0 0 0":
                geom_attrs["pos"] = record["xyz"]
            quat = _rpy_to_quat_wxyz(record["rpy"])
            if quat is not None:
                geom_attrs["quat"] = quat
            if COLLISION_MODE == "full-hull" and _is_foot_collision_mesh(geom_attrs["mesh"]):
                geom_attrs["solref"] = FULL_HULL_FOOT_CONTACT_SOLREF
            elif COLLISION_MODE in FITTED_COLLISION_MODES and _is_foot_collision_mesh(geom_attrs["mesh"]):
                geom_attrs["solref"] = FITTED_FOOT_CONTACT_SOLREF
                geom_attrs["solimp"] = FITTED_FOOT_CONTACT_SOLIMP
                geom_attrs["margin"] = FITTED_FOOT_CONTACT_MARGIN
            ET.SubElement(body, "geom", geom_attrs)

    _ensure_foot_self_collision_excludes(root)

    ET.indent(tree, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collision-mode",
        choices=("optimized", "full-hull", "fitted", "fitted-optimized"),
        default="optimized",
        help=(
            "optimized uses the existing low-face collision meshes; full-hull "
            "copies the original draft rear/forefoot/toe collision hulls for "
            "two-stage rolling-contact tests; fitted keeps the full-hull "
            "meshes and applies the F-x fitted two-stage passive-foot params; "
            "fitted-optimized applies the same params to low-face meshes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_outputs(args.collision_mode)
    _copy_full_hull_collision_meshes()
    build_urdf()
    collision_records = _collision_records_from_urdf(OUT_URDF)
    build_mjcf(DRAFT_MJCF, OUT_MJCF, collision_records)
    build_mjcf(DRAFT_LOOP_MJCF, OUT_LOOP_MJCF, collision_records)

    print(f"Wrote {OUT_URDF.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MJCF.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_LOOP_MJCF.relative_to(REPO_ROOT)}")
    print(f"Collision mode: {COLLISION_MODE}")
    print(f"Passive foot joints: {', '.join(PASSIVE_FOOT_JOINTS)}")


if __name__ == "__main__":
    main()
