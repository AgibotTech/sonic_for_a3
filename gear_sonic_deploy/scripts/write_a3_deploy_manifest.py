#!/usr/bin/env python3
"""Write an A3 deployment parity manifest.

The manifest is intentionally read-only: it records the exact artifacts that a
deployment run is supposed to match, plus the observed ONNX I/O schema. Use it
before hardware bring-up so obs/action/control diffs can be traced back to a
concrete checkpoint (when supplied), motion, robot asset, and git revision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GEAR_ROOT = REPO_ROOT / "gear_sonic_deploy"
DEFAULT_RUNTIME_CFG = (
    REPO_ROOT
    / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml"
)
DEFAULT_MOTION = (
    REPO_ROOT / "a3_data/agibot_a3/001_walk_front_slow.csv"
)
DEFAULT_MJCF = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
)
DEFAULT_URDF = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf"
)
DEFAULT_SOLVER_LIB = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/solver/a3_loop/lib/liba3_ankle_waist_solver.a"
)


def git_value(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return out.strip()


def load_runtime_cfg(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"runtime cfg is not a mapping: {path}")
    return data


def fmt_shape(shape: list[Any]) -> list[Any]:
    out: list[Any] = []
    for d in shape:
        if d is None:
            out.append("?")
        elif isinstance(d, str):
            out.append(d)
        else:
            out.append(int(d))
    return out


def probe_onnx(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False, "error": "missing"}

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = [
            {"name": i.name, "shape": fmt_shape(list(i.shape)), "dtype": i.type}
            for i in sess.get_inputs()
        ]
        outputs = [
            {"name": o.name, "shape": fmt_shape(list(o.shape)), "dtype": o.type}
            for o in sess.get_outputs()
        ]
    except ImportError:
        try:
            import onnx

            model = onnx.load(str(path))

            def shape_from_value_info(value_info: Any) -> list[Any]:
                dims = value_info.type.tensor_type.shape.dim
                shape: list[Any] = []
                for dim in dims:
                    if dim.HasField("dim_value") and dim.dim_value > 0:
                        shape.append(int(dim.dim_value))
                    elif dim.HasField("dim_param") and dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
                return shape

            def dtype_from_value_info(value_info: Any) -> str:
                elem = value_info.type.tensor_type.elem_type
                return onnx.TensorProto.DataType.Name(elem) if elem else "?"

            inputs = [
                {
                    "name": i.name,
                    "shape": shape_from_value_info(i),
                    "dtype": dtype_from_value_info(i),
                }
                for i in model.graph.input
            ]
            outputs = [
                {
                    "name": o.name,
                    "shape": shape_from_value_info(o),
                    "dtype": dtype_from_value_info(o),
                }
                for o in model.graph.output
            ]
        except ImportError:
            return {
                "path": str(path),
                "exists": True,
                "ok": False,
                "error": "neither onnxruntime nor onnx is importable",
            }
    except Exception as exc:
        return {"path": str(path), "exists": True, "ok": False, "error": str(exc)}

    input_shape = inputs[0]["shape"] if inputs else []
    output_shape = outputs[0]["shape"] if outputs else []
    ok = len(input_shape) == 2 and input_shape[-1] == 1570
    ok = ok and len(output_shape) == 2 and output_shape[-1] == 29
    return {
        "path": str(path),
        "exists": True,
        "ok": ok,
        "expected": {"input": [1, 1570], "output": [1, 29]},
        "inputs": inputs,
        "outputs": outputs,
    }


def runtime_asset_root(runtime_cfg: Path) -> Path:
    """Return the directory from which runtime YAML asset paths are resolved.

    Source-tree YAML paths are relative to ``gear_sonic_deploy``.  A packaged
    runtime YAML lives under ``<package>/config`` and uses the package root
    instead.  Keeping this distinction here makes a manifest match the actual
    deploy binary rather than the repository root.
    """

    resolved = runtime_cfg.resolve()
    try:
        resolved.relative_to(GEAR_ROOT.resolve())
    except ValueError:
        return resolved.parent.parent if resolved.parent.name == "config" else resolved.parent
    return GEAR_ROOT


def as_path(value: Any, fallback: Path, base: Path = REPO_ROOT) -> Path:
    if value is None:
        return fallback
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def resolve_runtime_path(value: Any, fallback: Path, asset_root: Path) -> Path:
    """Resolve a runtime YAML path like the source/package deployment flow.

    Source configs usually keep deploy assets below ``gear_sonic_deploy`` but
    the selected20 normal playlist intentionally lives at repository root.
    Packaged configs instead resolve every relative path from package root.
    Prefer the runtime asset root when it contains the path, then fall back to
    the repository root for source-tree references.
    """

    if value is None:
        return fallback
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = (asset_root / path, REPO_ROOT / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def infer_motion_from_runtime_cfg(
    reference_cfg: dict[str, Any], asset_root: Path
) -> tuple[Path, Path | None]:
    csv_path_raw = reference_cfg.get("csv_path")
    csv_path = (
        resolve_runtime_path(csv_path_raw, Path(), asset_root)
        if csv_path_raw is not None
        else None
    )
    csv_file = str(reference_cfg.get("csv_file") or "").strip()
    if csv_file:
        csv_source = (
            (csv_path / csv_file)
            if csv_path is not None
            else resolve_runtime_path(csv_file, Path(), asset_root)
        )
    else:
        csv_source = csv_path

    stem = csv_source.stem if csv_source is not None else DEFAULT_MOTION.stem
    motion_root = resolve_runtime_path(
        reference_cfg.get("motion_lib_dir"),
        asset_root / "data/agibot_a3_motion_lib_29d",
        asset_root,
    )
    candidates = [
        motion_root / "robot/agibot_a3" / f"{stem}.pkl",
        motion_root / "robot_filtered/agibot_a3" / f"{stem}.pkl",
        asset_root / "data/agibot_a3_motion_lib_29d/robot/agibot_a3" / f"{stem}.pkl",
        asset_root / "data/agibot_a3_motion_lib_29d/robot_filtered/agibot_a3" / f"{stem}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, csv_source
    return candidates[0], csv_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-cfg", type=Path, default=DEFAULT_RUNTIME_CFG)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "a3_deploy_manifest.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional PT checkpoint to record (existence check only)",
    )
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    args.runtime_cfg = args.runtime_cfg.resolve()
    cfg = load_runtime_cfg(args.runtime_cfg)
    asset_root = runtime_asset_root(args.runtime_cfg)
    onnx_cfg = cfg.get("onnx") or {}
    onnx_path = resolve_runtime_path(onnx_cfg.get("model_path"), Path(), asset_root)
    a3_fast_onnx_raw = onnx_cfg.get("a3_fast_model_path")
    a3_fast_onnx_path = (
        resolve_runtime_path(a3_fast_onnx_raw, Path(), asset_root)
        if a3_fast_onnx_raw
        else None
    )
    reference_cfg = cfg.get("reference_motion") or {}
    inferred_motion, csv_source = infer_motion_from_runtime_cfg(reference_cfg, asset_root)
    tokenizer_bin_raw = reference_cfg.get("tokenizer_bin_path")
    tokenizer_path = (
        resolve_runtime_path(tokenizer_bin_raw, Path(), asset_root)
        if tokenizer_bin_raw
        else None
    )
    tokenizer_meta_raw = reference_cfg.get("meta_path")
    tokenizer_meta = (
        resolve_runtime_path(tokenizer_meta_raw, Path(), asset_root)
        if tokenizer_meta_raw
        else None
    )
    backend_cfg = cfg.get("backend") or {}
    loop_mjcf_path = args.mjcf
    external_loop_raw = backend_cfg.get("external_loop_mjcf_path")
    external_loop_mjcf = (
        resolve_runtime_path(external_loop_raw, Path(), asset_root)
        if external_loop_raw
        else None
    )
    checkpoint_path = args.checkpoint
    motion_path = args.motion if args.motion != DEFAULT_MOTION else inferred_motion

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "root": str(REPO_ROOT),
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "source_of_truth": {
            "sim2sim_script": str(REPO_ROOT / "gear_sonic/scripts/sim2sim_a3_mujoco.py"),
            "sim2sim_mode": "closed_chain_loop",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "onnx": str(onnx_path),
            "csv": str(csv_source) if csv_source is not None else None,
            "motion_pkl": str(motion_path) if motion_path.exists() else None,
            "loop_mjcf": str(loop_mjcf_path),
            "external_option4_mjcf": str(external_loop_mjcf) if external_loop_mjcf else None,
            "loop_solver_lib": str(DEFAULT_SOLVER_LIB),
            "urdf": str(args.urdf),
            "action_clip": 20.0,
            "obs_shape": [1, 1570],
            "action_shape": [1, 29],
            "target_fps": float(reference_cfg.get("target_fps", 50.0)),
            "future_frame_skip": int(reference_cfg.get("future_frame_skip", 5)),
            "warmup_ticks": int((cfg.get("policy_driver") or {}).get("warmup_ticks", 100)),
            "on_end": str(reference_cfg.get("on_end", "hold_last")),
        },
        "runtime_cfg": str(args.runtime_cfg),
        "runtime_asset_root": str(asset_root),
        "artifacts": {
            "onnx": probe_onnx(onnx_path),
            "csv": {
                "path": str(csv_source) if csv_source is not None else "",
                "exists": bool(csv_source and csv_source.exists()),
            },
            "loop_mjcf": {"path": str(loop_mjcf_path), "exists": loop_mjcf_path.exists()},
            "loop_solver_lib": {
                "path": str(DEFAULT_SOLVER_LIB),
                "exists": DEFAULT_SOLVER_LIB.exists(),
            },
            "urdf": {"path": str(args.urdf), "exists": args.urdf.exists()},
        },
    }
    if a3_fast_onnx_path is not None:
        manifest["source_of_truth"]["a3_fast_onnx"] = str(a3_fast_onnx_path)
        manifest["artifacts"]["a3_fast_onnx"] = probe_onnx(a3_fast_onnx_path)
    if checkpoint_path is not None:
        manifest["artifacts"]["checkpoint"] = {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
        }
    if tokenizer_path is not None:
        manifest["artifacts"]["tokenizer_bin"] = {
            "path": str(tokenizer_path),
            "exists": tokenizer_path.exists(),
        }
    if tokenizer_meta is not None:
        manifest["artifacts"]["tokenizer_meta"] = {
            "path": str(tokenizer_meta),
            "exists": tokenizer_meta.exists(),
        }
    if external_loop_mjcf is not None:
        manifest["artifacts"]["external_option4_mjcf"] = {
            "path": str(external_loop_mjcf),
            "exists": external_loop_mjcf.exists(),
        }
    if motion_path.exists():
        manifest["artifacts"]["motion_pkl"] = {
            "path": str(motion_path),
            "exists": True,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[write] {args.out}")

    onnx_ok = manifest["artifacts"]["onnx"].get("ok", False)
    missing = [
        name
        for name, item in manifest["artifacts"].items()
        if name != "onnx" and not item.get("exists", False)
    ]
    if not onnx_ok:
        print("[warn] ONNX is missing or does not match [*,1570] -> [*,29]", file=sys.stderr)
    if missing:
        print(f"[warn] missing artifacts: {', '.join(missing)}", file=sys.stderr)
    if args.strict and (not onnx_ok or missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
