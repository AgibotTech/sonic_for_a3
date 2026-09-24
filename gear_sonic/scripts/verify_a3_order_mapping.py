"""Verify A3 IsaacLab <-> MuJoCo DOF/body mapping correctness.

Loads the A3 robot in Isaac Lab (GPU required) and MuJoCo, compares:
  1. Isaac Lab actual joint order vs BFS-alphabetical prediction
  2. Mapping arrays in a3.py vs mapping derived from actual Isaac Lab order

Usage:
    ACCEPT_EULA=Y path/to/env_isaaclab/bin/python gear_sonic/scripts/verify_a3_order_mapping.py

Results are written to /tmp/verify_a3_result.txt (stdout is swallowed by IsaacSim).
"""

from __future__ import annotations

from collections import defaultdict, deque
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ["ACCEPT_EULA"] = "Y"

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "gear_sonic" / "data" / "assets"
DEFAULT_URDF = ASSET_DIR / "robot_description" / "urdf" / "a3" / "model.urdf"
DEFAULT_MJCF = ASSET_DIR / "robot_description" / "mjcf" / "a3_t2d5.xml"
RESULT_PATH = Path("/tmp/verify_a3_result.txt")


def _bootstrap_isaaclab():
    """Start IsaacSim."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    return app


def _load_robot_in_isaaclab():
    """Load A3 in Isaac Lab using the same UrdfFileCfg as a3.py."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    import isaaclab.sim as sim_utils

    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=str(DEFAULT_URDF),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.09),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_cfg

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    return list(robot.joint_names), list(robot.body_names), sim


def _bfs_alphabetical_dofs(urdf_path: Path) -> list[str]:
    """Compute DOF order using BFS with alphabetical sibling sorting (Isaac Lab convention)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    all_links = [e.attrib["name"] for e in root.findall("link")]
    ptc: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    child_set: set[str] = set()
    for j in root.findall("joint"):
        p = j.find("parent").attrib["link"]
        c = j.find("child").attrib["link"]
        ptc[p].append((j.attrib["name"], j.attrib.get("type", ""), c))
        child_set.add(c)
    root_link = [l for l in all_links if l not in child_set][0]

    bfs_dofs: list[str] = []
    queue: deque[str] = deque([root_link])
    while queue:
        link = queue.popleft()
        children = sorted(ptc.get(link, []), key=lambda x: x[2].lower())
        for jn, jt, cl in children:
            if jt != "fixed":
                bfs_dofs.append(jn)
            queue.append(cl)
    return bfs_dofs


def _mujoco_dof_order(mjcf_path: Path, filter_set: set[str]) -> list[str]:
    """Get MuJoCo DOF order, filtered to joints in filter_set."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mj_joints: list[str] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            mj_joints.append(name)
    return [j for j in mj_joints if j.lower() in filter_set]


def main():
    app = _bootstrap_isaaclab()

    out = open(RESULT_PATH, "w")

    isaac_joints, isaac_bodies, sim = _load_robot_in_isaaclab()

    out.write(f"=== Isaac Lab joint_names ({len(isaac_joints)}) ===\n")
    for i, name in enumerate(isaac_joints):
        out.write(f"  {i}: {name}\n")

    out.write(f"\n=== Isaac Lab body_names ({len(isaac_bodies)}) ===\n")
    for i, name in enumerate(isaac_bodies):
        out.write(f"  {i}: {name}\n")

    # --- Test 1: BFS-alphabetical prediction ---
    bfs_dofs = _bfs_alphabetical_dofs(DEFAULT_URDF)

    out.write(f"\n{'=' * 70}\n")
    out.write("TEST 1: DOF order — Isaac Lab actual vs BFS-alphabetical prediction\n")
    out.write(f"{'=' * 70}\n")
    max_len = max(len(isaac_joints), len(bfs_dofs))
    dof_match = 0
    for i in range(max_len):
        actual = isaac_joints[i] if i < len(isaac_joints) else "N/A"
        computed = bfs_dofs[i] if i < len(bfs_dofs) else "N/A"
        ok = actual == computed
        if ok:
            dof_match += 1
        out.write(f"  {i:>2}: {actual:<35} vs {computed:<35} {'OK' if ok else 'MISMATCH'}\n")
    out.write(f"\nMatched: {dof_match}/{max_len}\n")

    # --- Test 2: mapping values ---
    isaac_set = {d.lower() for d in isaac_joints}
    mj_dofs = _mujoco_dof_order(DEFAULT_MJCF, isaac_set)
    mj_idx_map = {n.lower(): i for i, n in enumerate(mj_dofs)}
    actual_i2m = [mj_idx_map.get(n.lower(), -1) for n in isaac_joints]

    # Expected mapping from a3.py
    from gear_sonic.scripts.generate_a3_order_mapping import (
        compute_mapping,
        load_mjcf_orders,
        load_urdf_orders,
    )

    all_urdf_links, _, isaaclab_dofs_from_script = load_urdf_orders(DEFAULT_URDF)
    mujoco_bodies, mujoco_dofs_raw, _ = load_mjcf_orders(DEFAULT_MJCF)
    source_dof_set = {n.lower() for n in isaaclab_dofs_from_script}
    mujoco_dofs_for_mapping = [n for n in mujoco_dofs_raw if n.lower() in source_dof_set]
    expected_i2m, _ = compute_mapping(isaaclab_dofs_from_script, mujoco_dofs_for_mapping)

    out.write(f"\n{'=' * 70}\n")
    out.write("TEST 2: Mapping values — actual vs generate_a3_order_mapping.py\n")
    out.write(f"{'=' * 70}\n")
    map_ok = actual_i2m == expected_i2m
    out.write(f"Actual:   {actual_i2m}\n")
    out.write(f"Expected: {expected_i2m}\n")
    out.write(f"Match: {'YES' if map_ok else 'NO'}\n")

    # --- Summary ---
    out.write(f"\n{'=' * 70}\n")
    all_ok = dof_match == max_len and map_ok
    if all_ok:
        out.write(">>> ALL CHECKS PASSED <<<\n")
    else:
        out.write(">>> ISSUES FOUND <<<\n")
        if not map_ok and len(actual_i2m) == len(mj_dofs):
            inv = [-1] * len(mj_dofs)
            for s, t in enumerate(actual_i2m):
                if 0 <= t < len(inv):
                    inv[t] = s
            out.write("\nCorrect mapping based on Isaac Lab actual order:\n")
            out.write(f"A3_ISAACLAB_TO_MUJOCO_DOF = {actual_i2m}\n")
            out.write(f"A3_MUJOCO_TO_ISAACLAB_DOF = {inv}\n")

    out.close()
    print(f"Results written to {RESULT_PATH}")

    sim.stop()
    app.close()


if __name__ == "__main__":
    main()
