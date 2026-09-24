"""Generate IsaacLab <-> MuJoCo order mappings for A3.

This script extracts:
1) IsaacLab-like body traversal order from A3 URDF
2) MuJoCo body/DOF order from A3 MJCF
3) Bidirectional mapping indices between the two conventions

The output is formatted as Python lists so it can be pasted into `a3.py`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/a3/model.urdf"
DEFAULT_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5.xml"

NON_ACTUATED_URDF_JOINT_TYPES = {"fixed"}


def normalize_name(name: str) -> str:
    """Normalize names for robust matching between URDF/MJCF exports."""
    return name.strip().lower()


def _format_py_list(name: str, values: list[int] | list[str]) -> str:
    if not values:
        return f"{name} = []"
    if isinstance(values[0], str):
        items = ",\n".join(f'    "{v}"' for v in values)
    else:
        items = ",\n".join(f"    {v}" for v in values)
    return f"{name} = [\n{items},\n]"


def _inverse_mapping(forward: list[int], target_size: int, mapping_name: str) -> list[int]:
    """Build inverse mapping with explicit validation and clear errors."""
    inverse = [-1] * target_size
    for source_idx, target_idx in enumerate(forward):
        if target_idx < 0 or target_idx >= target_size:
            raise ValueError(
                f"{mapping_name}: target index out of range: "
                f"source[{source_idx}] -> {target_idx}, target_size={target_size}"
            )
        if inverse[target_idx] != -1:
            raise ValueError(
                f"{mapping_name}: duplicate target index {target_idx} "
                f"(existing source={inverse[target_idx]}, new source={source_idx})"
            )
        inverse[target_idx] = source_idx

    missing = [i for i, src in enumerate(inverse) if src == -1]
    if missing:
        raise ValueError(
            f"{mapping_name}: mapping is not bijective, missing target indices: {missing}. "
            "This usually means source and target name lists are not aligned."
        )
    return inverse


def load_urdf_orders(urdf_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (all_links_bfs, kinematic_links_bfs, actuated_joint_names).

    Uses BFS with alphabetical sibling sorting to match Isaac Lab's
    URDF import order (not DFS/XML-order like MuJoCo).
    """
    from collections import deque

    root = ET.parse(urdf_path).getroot()

    all_links = [e.attrib["name"] for e in root.findall("link")]
    child_links = set()
    parent_to_joints: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for joint in root.findall("joint"):
        joint_name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "")
        parent_link = joint.find("parent").attrib["link"]
        child_link = joint.find("child").attrib["link"]
        child_links.add(child_link)
        parent_to_joints[parent_link].append((joint_name, joint_type, child_link))

    root_candidates = [link for link in all_links if link not in child_links]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one URDF root link, got {root_candidates}")
    root_link = root_candidates[0]

    all_links_bfs: list[str] = []
    kinematic_links_bfs: list[str] = [root_link]
    actuated_joint_names: list[str] = []

    queue: deque[str] = deque([root_link])
    while queue:
        link_name = queue.popleft()
        all_links_bfs.append(link_name)
        children = sorted(parent_to_joints.get(link_name, []), key=lambda x: x[2].lower())
        for joint_name, joint_type, child_link in children:
            if joint_type not in NON_ACTUATED_URDF_JOINT_TYPES:
                actuated_joint_names.append(joint_name)
                kinematic_links_bfs.append(child_link)
            queue.append(child_link)

    return all_links_bfs, kinematic_links_bfs, actuated_joint_names


def _load_mjcf_orders_with_mujoco(mjcf_path: Path) -> tuple[list[str], list[str]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    body_names = []
    for body_id in range(1, model.nbody):  # exclude world
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            body_names.append(name)

    joint_names = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            joint_names.append(name)

    return body_names, joint_names


def _load_mjcf_orders_fallback(mjcf_path: Path) -> tuple[list[str], list[str]]:
    """Fallback parser when mujoco python package is unavailable."""
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody> section")

    root_body = worldbody.find("body")
    if root_body is None or "name" not in root_body.attrib:
        raise ValueError("MJCF has no named root body under <worldbody>")

    body_names: list[str] = []
    joint_names: list[str] = []

    def dfs(body_elem: ET.Element):
        body_names.append(body_elem.attrib["name"])

        for joint in body_elem.findall("joint"):
            joint_type = joint.attrib.get("type", "hinge")
            if joint_type not in {"free", "fixed"}:
                joint_names.append(joint.attrib["name"])

        for child_body in body_elem.findall("body"):
            if "name" in child_body.attrib:
                dfs(child_body)

    dfs(root_body)
    return body_names, joint_names


def load_mjcf_orders(mjcf_path: Path) -> tuple[list[str], list[str], str]:
    try:
        body_names, joint_names = _load_mjcf_orders_with_mujoco(mjcf_path)
        return body_names, joint_names, "mujoco-python"
    except Exception as err:
        print(f"# Warning: MuJoCo parser failed, fallback to XML parser. reason: {err}")
        body_names, joint_names = _load_mjcf_orders_fallback(mjcf_path)
        return body_names, joint_names, "xml-fallback"


def compute_mapping(source_names: list[str], target_names: list[str]) -> tuple[list[int], list[str]]:
    target_idx = {normalize_name(name): i for i, name in enumerate(target_names)}
    mapping: list[int] = []
    missing: list[str] = []
    for name in source_names:
        key = normalize_name(name)
        if key not in target_idx:
            missing.append(name)
            continue
        mapping.append(target_idx[key])
    return mapping, missing


def select_urdf_bodies_for_mjcf(all_urdf_links: list[str], mjcf_bodies: list[str]) -> list[str]:
    """Keep URDF DFS order, but only bodies that exist in MJCF."""
    mjcf_name_set = {normalize_name(name) for name in mjcf_bodies}
    selected = [name for name in all_urdf_links if normalize_name(name) in mjcf_name_set]
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate A3 IsaacLab<->MuJoCo order mappings.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to A3 URDF.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="Path to A3 MJCF.")
    args = parser.parse_args()

    all_urdf_links, kinematic_urdf_links, isaaclab_dofs = load_urdf_orders(args.urdf)
    mujoco_bodies, mujoco_dofs, mjcf_loader = load_mjcf_orders(args.mjcf)
    isaaclab_links = select_urdf_bodies_for_mjcf(kinematic_urdf_links, mujoco_bodies)
    source_dof_set = {normalize_name(name) for name in isaaclab_dofs}
    mujoco_dofs_for_mapping = [name for name in mujoco_dofs if normalize_name(name) in source_dof_set]
    dropped_mujoco_dofs = [name for name in mujoco_dofs if normalize_name(name) not in source_dof_set]

    # Filter MuJoCo bodies to only those present in IsaacLab kinematic links
    isaaclab_body_set = {normalize_name(name) for name in isaaclab_links}
    mujoco_bodies_for_mapping = [name for name in mujoco_bodies if normalize_name(name) in isaaclab_body_set]
    dropped_mujoco_bodies = [name for name in mujoco_bodies if normalize_name(name) not in isaaclab_body_set]

    # compute_mapping(source, target) returns mapping[source_pos] = target_idx
    # Convention: X_TO_Y means "converts FROM X TO Y via gather"
    #   mj_data[mujoco_to_isaaclab] → il_data
    #   il_data[isaaclab_to_mujoco] → mj_data (filtered subset)

    # Body forward: values are indices into the FULL MuJoCo body array (32 bodies)
    # so fancy indexing on FK output (32 bodies) selects+reorders to 30 IL bodies
    mujoco_to_isaaclab_body, missing_bodies = compute_mapping(isaaclab_links, mujoco_bodies)
    # Body inverse: values are indices into the 30-body IsaacLab array
    isaaclab_to_mujoco_body, _ = compute_mapping(mujoco_bodies_for_mapping, isaaclab_links)
    # DOF forward: values are indices into the FULL MuJoCo DOF array (31 DOFs)
    # so fancy indexing on FK dof_pos (31) selects+reorders to 29 IL DOFs
    mujoco_to_isaaclab_dof, missing_dofs = compute_mapping(isaaclab_dofs, mujoco_dofs)
    # DOF inverse: values are indices into the 29-DOF IsaacLab array
    isaaclab_to_mujoco_dof, _ = compute_mapping(mujoco_dofs_for_mapping, isaaclab_dofs)

    if missing_bodies:
        raise ValueError(f"Failed to map IsaacLab bodies to MuJoCo: {missing_bodies}")
    if missing_dofs:
        raise ValueError(f"Failed to map IsaacLab DOFs to MuJoCo: {missing_dofs}")

    print(f"# URDF parsed links (all): {len(all_urdf_links)}")
    print(f"# URDF parsed links (kinematic): {len(kinematic_urdf_links)}")
    print(f"# URDF links selected for body mapping: {len(isaaclab_links)}")
    print(f"# URDF parsed actuated joints: {len(isaaclab_dofs)}")
    print(f"# MJCF parsed bodies: {len(mujoco_bodies)}")
    print(f"# MJCF bodies used for body mapping: {len(mujoco_bodies_for_mapping)}")
    if dropped_mujoco_bodies:
        print(f"# MJCF bodies dropped from body mapping: {dropped_mujoco_bodies}")
    print(f"# MJCF parsed joints (non-free, non-fixed): {len(mujoco_dofs)}")
    print(f"# MJCF joints used for DOF mapping: {len(mujoco_dofs_for_mapping)}")
    if dropped_mujoco_dofs:
        print(f"# MJCF joints dropped from DOF mapping: {dropped_mujoco_dofs}")
    print(f"# MJCF loader: {mjcf_loader}")
    print()
    print(_format_py_list("A3_ISAACLAB_JOINTS", isaaclab_links))
    print()
    print(_format_py_list("A3_ISAACLAB_TO_MUJOCO_DOF", isaaclab_to_mujoco_dof))
    print()
    print(_format_py_list("A3_MUJOCO_TO_ISAACLAB_DOF", mujoco_to_isaaclab_dof))
    print()
    print(_format_py_list("A3_ISAACLAB_TO_MUJOCO_BODY", isaaclab_to_mujoco_body))
    print()
    print(_format_py_list("A3_MUJOCO_TO_ISAACLAB_BODY", mujoco_to_isaaclab_body))
    print()
    print(
        "A3_ISAACLAB_TO_MUJOCO_MAPPING = {\n"
        '    "isaaclab_joints": A3_ISAACLAB_JOINTS,\n'
        '    "isaaclab_to_mujoco_dof": A3_ISAACLAB_TO_MUJOCO_DOF,\n'
        '    "mujoco_to_isaaclab_dof": A3_MUJOCO_TO_ISAACLAB_DOF,\n'
        '    "isaaclab_to_mujoco_body": A3_ISAACLAB_TO_MUJOCO_BODY,\n'
        '    "mujoco_to_isaaclab_body": A3_MUJOCO_TO_ISAACLAB_BODY,\n'
        "}"
    )


if __name__ == "__main__":
    main()
