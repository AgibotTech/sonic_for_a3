#!/usr/bin/env python3
# Copyright (c) 2026, AgiBot Inc. All rights reserved.
#
# offline_export_a3_tokenizer_bin.py — PR 9.6
#
# Bakes the "reference motion" tokenizer input (640 floats per tick) for
# offline replay by the C++ deploy binary (a3_deploy_onnx_ref, PR 9.5).
#
# Rationale (from notes/a3_pr9_spec.md §4):
#   At training time the A3 SONIC policy consumes
#     [command_multi_future_nonflat(580) | motion_anchor_ori_b_mf_nonflat(60)]
#   sliced from a motion_lib pkl at every tick. We do NOT port motion_lib +
#   command_multi_future packing to C++ — instead we dump the raw 640-float
#   tensor here, once per tick, to a flat binary. The backend just memcpys.
#
# This script must run inside the SAME Python environment as training so that
# Isaac Lab, the A3 robot config, and the pkl loader all exist. After activating
# an environment with Isaac Lab and `pip install -e "gear_sonic/[training]"`:
#
#     cd /path/to/repository/gear_sonic_deploy
#     python scripts/offline_export_a3_tokenizer_bin.py \
#         --pkl  /path/to/a3_motionlib/clip.pkl \
#         --out  /tmp/a3_reference_motion/clip \
#         --num-ticks 500
#
# Output directory layout:
#   {out}/tokenizer.bin   [num_ticks, 640] float32 little-endian, no header
#   {out}/meta.json       num_ticks, dt_ns, provenance
#
# See the README alongside this file for a detailed setup guide.

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_isaaclab_available() -> None:
    """Fail early with an actionable message if isaaclab isn't importable.

    The Isaac Lab env activation is the caller's responsibility (see the
    docstring). Without it the script can't construct the ManagerBasedEnv
    and the error from a missing `isaaclab` import is less helpful.
    """
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "ERROR: Isaac Lab is not importable. Activate a compatible Isaac Lab\n"
            "environment and install this checkout with:\n"
            "  python -m pip install -e 'gear_sonic/[training]'\n"
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bake the A3 tokenizer reference-motion stream for offline replay by a3_deploy_onnx_ref.",
    )
    p.add_argument(
        "--pkl",
        required=True,
        help="Path to a motion_lib pkl clip (single file).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory. Will be created if missing. Produces tokenizer.bin + meta.json.",
    )
    p.add_argument(
        "--num-ticks",
        type=int,
        default=500,
        help="Number of 50Hz policy ticks to dump. Default: 500 (10s).",
    )
    p.add_argument(
        "--ckpt",
        default="checkpoints/035_step200000/model_step_200000.pt",
        help="Training checkpoint — ONLY used to locate the companion "
        "training config.yaml (to reconstruct the env bit-exactly). "
        "Weights are NOT loaded.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the env — fixes the initial motion sampling so runs are deterministic.",
    )
    return p.parse_args()


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    _ensure_isaaclab_available()
    args = _parse_args()

    pkl_path = Path(args.pkl).resolve()
    out_dir = Path(args.out).resolve()
    if not pkl_path.is_file():
        print(f"ERROR: pkl not found: {pkl_path}", file=sys.stderr)
        return 2
    if not Path(args.ckpt).is_file():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Endianness warning — x86 hosts + aarch64 robots are both little-endian,
    # so no byte-swap is needed. Emit a loud warning on unexpected hosts so
    # the bin isn't silently mis-produced for a big-endian consumer.
    if sys.byteorder != "little":
        print(
            f"WARNING: host byteorder is '{sys.byteorder}', but the C++ loader "
            "assumes little-endian float32. Aborting to avoid a silent "
            "byte-order mismatch.",
            file=sys.stderr,
        )
        return 3

    # --- Bring up Isaac Sim (headless) -------------------------------------
    # Import ordering matters: AppLauncher must run BEFORE `torch`, or CUDA
    # contexts get initialised on the wrong GPU. Mirror what eval_agent_trl
    # does for a clean boot.
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app_args, _ = ap.parse_known_args(
        [
            "--headless",
            "--kit_args=--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error",
        ]
    )
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app  # noqa: F841

    # --- Imports that require Isaac Sim to have booted ---------------------
    import numpy as np
    import omegaconf
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gear_sonic"))
    # ^ Make the training tree importable. Not strictly required if
    # An editable install is the normal setup, but this makes the script
    # self-contained when invoked directly from a checkout.
    from gear_sonic import train_agent_trl  # noqa: E402
    from gear_sonic.envs.manager_env.mdp.observations import (  # noqa: E402
        command_multi_future,
        motion_anchor_ori_b_mf,
    )
    from gear_sonic.utils import (
        common as rl_utils_common,  # noqa: E402
        config_utils,  # noqa: E402
    )

    # The training config uses a few custom OmegaConf resolvers (eval, if,
    # eq, …). Register them before loading the config.
    config_utils.register_rl_resolvers()
    # Training configs also contain `${hydra:runtime.choices.exp}` (and
    # similar) — but we don't run under @hydra.main, so the built-in
    # `hydra:` resolver is absent. Register a stub that returns a fixed
    # placeholder; the value ends up in fields we don't read (notably
    # `save_rendering_dir`). Tolerate re-registration in case a caller
    # already registered one.
    try:
        omegaconf.OmegaConf.register_new_resolver("hydra", lambda *_: "offline_export", replace=False)
    except (ValueError, Exception):
        pass
    # `${now:...}` is a hydra plugin; stub it similarly.
    try:
        omegaconf.OmegaConf.register_new_resolver("now", lambda *_: "19700101_000000", replace=False)
    except (ValueError, Exception):
        pass

    # --- Load training config from the checkpoint sibling ------------------
    ckpt_dir = Path(args.ckpt).parent
    cfg_path = ckpt_dir / "config.yaml"
    if not cfg_path.is_file():
        cfg_path = ckpt_dir.parent / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: config.yaml not found near {args.ckpt}", file=sys.stderr)
        return 2
    print(f"Loading training config: {cfg_path}")

    raw = cfg_path.read_text()
    # Backward-compat rewrites (copied from eval_agent_trl.py).
    raw = (
        raw.replace("groot.rl.trl.", "gear_sonic.trl.")
        .replace("groot.rl.envs.", "gear_sonic.envs.")
        .replace("groot.rl.utils.", "gear_sonic.utils.")
        .replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
        .replace("groot.rl.agents.", "gear_sonic.trl.")
        .replace("groot/rl/data/", "gear_sonic/data/")
        .replace("assets/bm/unitree_description/", "assets/robot_description/")
        .replace("1215_bones_seed_filtered", "bones_seed_smpl")
    )
    import io

    cfg = omegaconf.OmegaConf.load(io.StringIO(raw))

    # Shrink the env to the smallest footprint that can still run the
    # observation terms deterministically.
    with omegaconf.open_dict(cfg):
        cfg.num_envs = 1
        cfg.seed = args.seed
        cfg.headless = True
        cfg.multi_gpu = False
        # Restrict motion_lib to exactly the pkl Zach passed, so the env
        # samples this single clip on reset.
        motion_cmd = cfg.manager_env.commands.motion
        motion_cmd.filter_motion_keys = [pkl_path.stem]
        if "motion_lib_cfg" in motion_cmd:
            motion_cmd.motion_lib_cfg.filter_motion_keys = [pkl_path.stem]
            # Point motion_lib at the pkl's parent directory — it scans
            # recursively for .pkl files and the filter_motion_keys above
            # narrows down to the one we want.
            motion_cmd.motion_lib_cfg.motion_file = str(pkl_path.parent)

    rl_utils_common.seeding(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The A3 robot config uses a relative ASSET_DIR ("gear_sonic/data/assets")
    # that IsaacLab resolves from the process CWD. We need CWD to be the
    # PARENT of gear_sonic/ (matches `python -m gear_sonic.train_agent_trl`
    # invoked from the dev tree root).
    gear_sonic_parent = REPO_ROOT
    expected = gear_sonic_parent / "gear_sonic" / "data" / "assets"
    if not expected.is_dir():
        raise RuntimeError(
            f"A3 assets were not found at {expected}. Run this script from a complete repository checkout."
        )
    os.chdir(str(gear_sonic_parent))
    print(f"chdir to {gear_sonic_parent} so gear_sonic/data/assets/… resolves")

    # create_manager_env wants an args_cli with .headless; construct a
    # minimal stand-in. (We cannot reuse `app_args` directly because it
    # already consumed AppLauncher's slots.)
    class _FakeArgs:
        headless = True

    env = train_agent_trl.create_manager_env(cfg, device, _FakeArgs())

    # --- Reset + sanity checks --------------------------------------------
    _ = env.reset(flatten_dict_obs=False)
    inner_env = env.env  # ManagerBasedRLEnv

    # --- Capture initial robot state (first frame of motion clip) ---------
    # After reset, the robot is placed at the motion clip's first frame. We
    # export this state so runtime diagnostics can compare against the same
    # initial pose used during training.
    robot = inner_env.scene["robot"]
    # joint_pos is in IsaacLab order (shape [1, num_joints]).
    init_joint_pos_il = robot.data.joint_pos[0].detach().cpu().numpy().tolist()
    init_joint_vel_il = robot.data.joint_vel[0].detach().cpu().numpy().tolist()
    # Root state in world frame.
    init_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().tolist()
    # IsaacLab stores quat as (w,x,y,z).
    init_root_quat_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy().tolist()
    # MuJoCo free joint qvel: [lin_vel_world(3), ang_vel_body(3)].
    # IsaacLab root_lin_vel_w is world frame → matches MuJoCo qvel[0:3].
    # IsaacLab root_ang_vel_b is body frame → matches MuJoCo qvel[3:6].
    # (root_ang_vel_w would be WRONG for MuJoCo!)
    init_root_lin_vel = robot.data.root_lin_vel_w[0].detach().cpu().numpy().tolist()
    init_root_ang_vel = robot.data.root_ang_vel_b[0].detach().cpu().numpy().tolist()

    # Convert joint_pos from IsaacLab order → MuJoCo 29-DOF policy view.
    from gear_sonic.envs.manager_env.robots.a3 import A3_ISAACLAB_TO_MUJOCO_DOF
    init_joint_pos_mujoco_29 = [
        float(init_joint_pos_il[A3_ISAACLAB_TO_MUJOCO_DOF[i]])
        for i in range(29)
    ]
    init_joint_vel_mujoco_29 = [
        float(init_joint_vel_il[A3_ISAACLAB_TO_MUJOCO_DOF[i]])
        for i in range(29)
    ]

    print(f"  initial root_pos = {[f'{v:.4f}' for v in init_root_pos]}")
    print(f"  initial root_quat_wxyz = {[f'{v:.4f}' for v in init_root_quat_wxyz]}")
    print(f"  initial joint_pos_mujoco[0:6] = {[f'{v:.4f}' for v in init_joint_pos_mujoco_29[:6]]}")

    # Training-side sanity: make sure the observation-manager knows the two
    # terms we're about to call directly. Catches typos in case the term
    # names change in a future training revision. (Uses a defensive
    # getattr — the obs-manager internal attribute name has varied across
    # Isaac Lab minor versions.)
    tok_cfgs = None
    for attr in ("_group_obs_class_term_cfgs", "_group_obs_class_term_names"):
        if hasattr(inner_env.observation_manager, attr):
            tok_cfgs = getattr(inner_env.observation_manager, attr).get("tokenizer", None)
            break
    if tok_cfgs is not None:
        # tok_cfgs may be a list of cfgs or a list of names depending on
        # the attr above. Convert to a set of term-name strings.
        if tok_cfgs and not isinstance(tok_cfgs[0], str):
            term_names = {getattr(c, "name", str(c)) for c in tok_cfgs}
        else:
            term_names = set(tok_cfgs)
        for required in ("motion_anchor_ori_b_mf_nonflat", "command_multi_future_nonflat"):
            if required not in term_names:
                print(
                    f"WARNING: '{required}' not in tokenizer obs terms; the offline dump may not match training.",
                    file=sys.stderr,
                )

    # --- Dump tokenizer slices --------------------------------------------
    tokenizer_buf = np.zeros((args.num_ticks, 640), dtype=np.float32, order="C")

    # No-op action: all zeros in the policy's action space. motion_lib
    # advancement inside _update_command is driven by the env step count,
    # not by the action vector, so a zero action is a deterministic
    # "advance one tick" request.
    action_dim = inner_env.action_space.shape[-1]
    noop = torch.zeros((1, action_dim), dtype=torch.float32, device=device)

    print(f"Dumping {args.num_ticks} ticks × 640 floats …")
    for t in range(args.num_ticks):
        anchor_t = motion_anchor_ori_b_mf(inner_env, "motion", non_flatten=True)
        cmd_t = command_multi_future(inner_env, "motion", non_flatten=True)

        # anchor_t: [num_envs, 10, 6] → 60 floats per env
        # cmd_t:    [num_envs, 10, 58] → 580 floats per env
        if anchor_t.shape[0] != 1 or cmd_t.shape[0] != 1:
            raise RuntimeError(f"expected num_envs=1; got anchor={anchor_t.shape} cmd={cmd_t.shape}")
        # Order matches tokenizer_obs_names (= ONNX input order):
        # command_multi_future_nonflat(580) then motion_anchor_ori(60).
        flat = torch.cat([cmd_t.reshape(1, -1), anchor_t.reshape(1, -1)], dim=-1)
        if flat.shape[-1] != 640:
            raise RuntimeError(f"expected 640-flat slice; got shape {flat.shape}")
        tokenizer_buf[t] = flat[0].detach().cpu().numpy().astype(np.float32)

        # Advance motion by one policy tick.
        inner_env.step(noop)

        if (t + 1) % 50 == 0:
            print(f"  tick {t + 1:4d} / {args.num_ticks}")

    # --- Write outputs -----------------------------------------------------
    bin_path = out_dir / "tokenizer.bin"
    meta_path = out_dir / "meta.json"

    # C-contiguous float32 little-endian — matches
    # src/g1/g1_deploy_onnx_ref/include/a3_deploy/a3_tokenizer_replay.hpp.
    tokenizer_buf.tofile(str(bin_path))
    expected_bytes = args.num_ticks * 640 * 4
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        print(f"ERROR: bin size mismatch: wrote {actual_bytes}, expected {expected_bytes}", file=sys.stderr)
        return 4

    meta = {
        "num_ticks": int(args.num_ticks),
        "dt_ns": 20_000_000,  # 50 Hz policy period
        "pkl_source": str(pkl_path),
        "clip_name": pkl_path.stem,
        "checkpoint_path": str(Path(args.ckpt).resolve()),
        "generation_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "initial_state": {
            "root_pos": init_root_pos,
            "root_quat_wxyz": init_root_quat_wxyz,
            "root_lin_vel": init_root_lin_vel,
            "root_ang_vel": init_root_ang_vel,
            "joint_pos_mujoco_29": init_joint_pos_mujoco_29,
            "joint_vel_mujoco_29": init_joint_vel_mujoco_29,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"✓ wrote {bin_path} ({actual_bytes} bytes)")
    print(f"✓ wrote {meta_path}")
    print(f"  num_ticks={args.num_ticks}  dt_ns=20_000_000  clip={pkl_path.stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
