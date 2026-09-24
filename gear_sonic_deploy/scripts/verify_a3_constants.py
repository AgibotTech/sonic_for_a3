#!/usr/bin/env python3
# Copyright (c) 2026, AgiBot Inc. All rights reserved.
"""
Verify that the C++ A3 policy constants in
    src/g1/g1_deploy_onnx_ref/include/a3_policy_parameters.hpp
match the Python source of truth in
    gear_sonic/envs/manager_env/robots/a3.py
and
    gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3.yaml

See notes/a3_backend_plan.md §5 (PR 2/10).

Special case: a3.py's A3_MUJOCO_TO_ISAACLAB_DOF has a bug (contains values
29 and 30, should only be [0..28]).  We instead verify the C++ value against
numpy.argsort(A3_ISAACLAB_TO_MUJOCO_DOF).

Exit 0 on success, 1 on any mismatch.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # GR00T-WholeBodyControl/
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"
A3_PY = REPO_ROOT / "gear_sonic/envs/manager_env/robots/a3.py"
A3_YAML = (
    REPO_ROOT
    / "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3.yaml"
)
HPP = (
    DEPLOY_ROOT
    / "src/g1/g1_deploy_onnx_ref/include/a3_policy_parameters.hpp"
)
A3_LAYOUT_HPP = (
    DEPLOY_ROOT
    / "src/g1/g1_deploy_onnx_ref/include/robot_io/a3_layout_extra.hpp"
)
A3_LAYOUT_CPP = (
    DEPLOY_ROOT
    / "src/g1/g1_deploy_onnx_ref/src/robot_io/a3_layout_extra.cpp"
)

# 29-DOF MuJoCo policy-view order: the 31-DOF MuJoCo real order with neck
# slots [3..4] removed. This is the storage convention used by
# a3_policy_parameters.hpp after the 2026-04 A3 realignment.
POLICY_VIEW_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
LAYOUT31_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    *POLICY_VIEW_JOINT_NAMES[3:17],
    *POLICY_VIEW_JOINT_NAMES[17:],
]
POLICY_TO_SDK_IDX = [0, 1, 2, *range(5, 31)]

# -----------------------------------------------------------------------------
# Parse the C++ header: each `constexpr std::array<...> NAME = { ... };`
# -----------------------------------------------------------------------------
def parse_cpp_arrays(text: str) -> dict[str, list[float]]:
    # Strip // line comments (not /* */ -- header has none).
    no_comments = re.sub(r"//[^\n]*", "", text)
    out: dict[str, list[float]] = {}
    pattern = re.compile(
        r"constexpr\s+std::array<[^>]*(?:<[^>]*>[^>]*)?>\s+(\w+)\s*=\s*"
        r"(\{\{?.*?\}\}?)\s*;",
        re.DOTALL,
    )
    for m in pattern.finditer(no_comments):
        name = m.group(1)
        body = m.group(2)
        # Extract numeric literals (support ints, floats, negatives).
        nums = re.findall(r"-?\d+\.\d*|-?\d+", body)
        out[name] = [float(n) for n in nums]
    return out


def parse_cpp_int_array(text: str, name: str) -> list[int]:
    no_comments = re.sub(r"//[^\n]*", "", text)
    m = re.search(rf"{re.escape(name)}\s*=\s*\{{(.*?)\}}\s*;", no_comments, re.DOTALL)
    if not m:
        return []
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def parse_layout31_names(text: str) -> list[str]:
    m = re.search(r"JointLayout\s+layout\{\{(.*?)\}\};", text, re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


# -----------------------------------------------------------------------------
# Load a3.py as a module WITHOUT importing isaaclab.  We monkeypatch the
# relevant modules with stubs so that only dataclass-style configuration
# code runs.
# -----------------------------------------------------------------------------
def load_a3_module():
    import types

    class _Stub:
        def __init__(self, *a, **kw):
            self.__dict__.update(kw)

        def __call__(self, *a, **kw):
            obj = _Stub()
            obj.__dict__.update(kw)
            return obj

        def __getattr__(self, name):
            # Any attribute access returns another _Stub.
            val = _Stub()
            self.__dict__[name] = val
            return val

    # ArticulationCfg(...) should retain init_state/actuators kwargs.
    class ArticulationCfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        class InitialStateCfg:
            def __init__(self, **kw):
                self.__dict__.update(kw)

    # ImplicitActuatorCfg(...) -- retain stiffness/damping/
    # joint_names_expr/effort_limit_sim.
    class ImplicitActuatorCfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    isaaclab_mod = types.ModuleType("isaaclab")
    actuators_mod = types.ModuleType("isaaclab.actuators")
    actuators_mod.ImplicitActuatorCfg = ImplicitActuatorCfg
    assets_mod = types.ModuleType("isaaclab.assets")
    articulation_mod = types.ModuleType("isaaclab.assets.articulation")
    articulation_mod.ArticulationCfg = ArticulationCfg
    sim_mod = types.ModuleType("isaaclab.sim")
    # sim_utils.UrdfFileCfg, RigidBodyPropertiesCfg, etc.
    sim_mod.UrdfFileCfg = _Stub()
    sim_mod.RigidBodyPropertiesCfg = _Stub()
    sim_mod.ArticulationRootPropertiesCfg = _Stub()
    sim_mod.UrdfConverterCfg = _Stub()
    sys.modules["isaaclab"] = isaaclab_mod
    sys.modules["isaaclab.actuators"] = actuators_mod
    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.assets.articulation"] = articulation_mod
    sys.modules["isaaclab.sim"] = sim_mod

    gs_mod = types.ModuleType("gear_sonic")
    envs_mod = types.ModuleType("gear_sonic.envs")
    mgr_mod = types.ModuleType("gear_sonic.envs.manager_env")
    mdp_mod = types.ModuleType("gear_sonic.envs.manager_env.mdp")
    act_mod = types.ModuleType("gear_sonic.envs.manager_env.mdp.actuators")
    act_mod.ImplicitActuatorCfg = ImplicitActuatorCfg
    act_mod.DelayedImplicitActuatorCfg = ImplicitActuatorCfg
    sys.modules["gear_sonic"] = gs_mod
    sys.modules["gear_sonic.envs"] = envs_mod
    sys.modules["gear_sonic.envs.manager_env"] = mgr_mod
    sys.modules["gear_sonic.envs.manager_env.mdp"] = mdp_mod
    sys.modules["gear_sonic.envs.manager_env.mdp.actuators"] = act_mod

    spec = importlib.util.spec_from_file_location("a3_source", str(A3_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_joint_dict(d: dict | float, joint_name: str, names_expr: list[str]):
    """Resolve a per-joint stiffness/damping value: keys are regexes."""
    if not isinstance(d, dict):
        return d
    for pat, val in d.items():
        if re.fullmatch(pat, joint_name):
            return val
    return None


def expected_from_a3(a3):
    """Produce the reference 29-element arrays from a3.py contents."""
    cfg = a3.A3_CYLINDER_CFG
    joint_pos = cfg.init_state.joint_pos
    actuators = cfg.actuators

    default = [float(joint_pos[n]) for n in POLICY_VIEW_JOINT_NAMES]

    kps: list[float] = []
    kds: list[float] = []
    action_scale: list[float] = []
    for n in POLICY_VIEW_JOINT_NAMES:
        got_kp, got_kd, got_effort = None, None, None
        for actu in actuators.values():
            names_expr = actu.joint_names_expr
            matched = any(re.fullmatch(p, n) for p in names_expr)
            if not matched:
                continue
            got_kp = _resolve_joint_dict(actu.stiffness, n, names_expr)
            got_kd = _resolve_joint_dict(actu.damping, n, names_expr)
            got_effort = _resolve_joint_dict(actu.effort_limit_sim, n, names_expr)
            break
        if got_kp is None or got_kd is None or got_effort is None:
            raise RuntimeError(f"Could not resolve actuator params for {n}")
        kps.append(float(got_kp))
        kds.append(float(got_kd))
        action_scale.append(0.25 * float(got_effort) / float(got_kp))

    itm = list(a3.A3_ISAACLAB_TO_MUJOCO_DOF)  # 29 elements
    # Use numpy argsort to derive the correct inverse (a3.py's
    # A3_MUJOCO_TO_ISAACLAB_DOF is buggy -- contains 29 and 30).
    mti_correct = list(np.argsort(np.array(itm)).tolist())

    return {
        "a3_default_angles": default,
        "a3_kps": kps,
        "a3_kds": kds,
        "a3_action_scale": action_scale,
        "a3_isaaclab_to_mujoco": itm,
        "a3_mujoco_to_isaaclab": mti_correct,
    }


def parse_yaml_vr3(path: Path):
    text = path.read_text()
    # vr_3point_body_offset: [[...],[...],[...]] -- find the full outer list
    # on the line and then extract three inner triples.
    m = re.search(r"vr_3point_body_offset:\s*(\[.*\])\s*$", text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find vr_3point_body_offset in yaml")
    triples = re.findall(r"\[\s*([^\[\]]+?)\s*\]", m.group(1))
    if len(triples) != 3:
        raise RuntimeError(
            f"Expected 3 triples in vr_3point_body_offset, got {len(triples)}")
    offsets: list[float] = []
    for t in triples:
        nums = [float(x.strip().lstrip("+")) for x in t.split(",")]
        offsets.extend(nums)
    # vr_3point_body names -> indices via A3_ISAACLAB_JOINTS lookup.
    m2 = re.search(r'vr_3point_body:\s*\[([^\]]+)\]', text)
    if not m2:
        raise RuntimeError("Could not find vr_3point_body in yaml")
    names = [n.strip().strip('"').strip("'") for n in m2.group(1).split(",")]
    return names, offsets


def compare(name: str, got: list[float], exp: list[float], tol: float = 1e-9):
    if len(got) != len(exp):
        return [f"{name}: length mismatch cpp={len(got)} py={len(exp)}"]
    diffs = []
    for i, (g, e) in enumerate(zip(got, exp)):
        if abs(g - e) > tol:
            diffs.append(f"{name}[{i}]: cpp={g} py={e}")
    return diffs


def main() -> int:
    cpp_text = HPP.read_text()
    cpp_arrays = parse_cpp_arrays(cpp_text)
    layout_hpp_text = A3_LAYOUT_HPP.read_text()
    layout_cpp_text = A3_LAYOUT_CPP.read_text()

    a3 = load_a3_module()
    expected = expected_from_a3(a3)

    vr_names, vr_offsets = parse_yaml_vr3(A3_YAML)
    isaaclab_joints = list(a3.A3_ISAACLAB_JOINTS)
    vr_idx_expected = [isaaclab_joints.index(n) for n in vr_names]
    expected["a3_vr_3point_index"] = vr_idx_expected
    expected["a3_vr_3point_body_offset"] = vr_offsets  # flattened 9 floats

    all_diffs: list[str] = []
    for name, exp in expected.items():
        if name not in cpp_arrays:
            all_diffs.append(f"{name}: not found in C++ header")
            continue
        all_diffs.extend(compare(name, cpp_arrays[name], exp))

    policy_to_sdk = parse_cpp_int_array(layout_hpp_text, "kA3PolicyToSdkIdx")
    if policy_to_sdk != POLICY_TO_SDK_IDX:
        all_diffs.append(
            f"kA3PolicyToSdkIdx mismatch cpp={policy_to_sdk} "
            f"expected={POLICY_TO_SDK_IDX}"
        )

    layout31_names = parse_layout31_names(layout_cpp_text)
    if layout31_names != LAYOUT31_JOINT_NAMES:
        all_diffs.append(
            f"MakeA3Layout31 names mismatch cpp={layout31_names} "
            f"expected={LAYOUT31_JOINT_NAMES}"
        )

    # Sanity: verify a3.py bug is real (report, do not fail).
    buggy = list(a3.A3_MUJOCO_TO_ISAACLAB_DOF)
    max_val = max(buggy)
    note = ""
    if max_val > 28:
        note = (f"  (note: a3.py A3_MUJOCO_TO_ISAACLAB_DOF has out-of-range "
                f"values up to {max_val}; using argsort instead)")

    if all_diffs:
        print("FAIL: A3 constants inconsistent:")
        for d in all_diffs:
            print(f"  - {d}")
        if note:
            print(note)
        return 1

    print("OK: all A3 constants consistent")
    verified = sorted(expected.keys()) + ["MakeA3Layout31", "kA3PolicyToSdkIdx"]
    print(f"  verified: {verified}")
    if note:
        print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
