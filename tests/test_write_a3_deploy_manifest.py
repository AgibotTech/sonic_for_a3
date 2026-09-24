"""Regression coverage for source-tree A3 runtime asset resolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic_deploy/scripts/write_a3_deploy_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("write_a3_deploy_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
manifest = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest
SPEC.loader.exec_module(manifest)


def test_source_runtime_paths_are_relative_to_gear_sonic_deploy() -> None:
    root = manifest.runtime_asset_root(manifest.DEFAULT_RUNTIME_CFG)
    model = manifest.as_path(
        "assets/a3_runtime/models/035_step200000/model_step_200000_g1.onnx",
        Path(),
        root,
    )

    assert root == manifest.GEAR_ROOT
    assert model == manifest.GEAR_ROOT / "assets/a3_runtime/models/035_step200000/model_step_200000_g1.onnx"


def test_source_runtime_paths_can_fall_back_to_repo_root() -> None:
    root = manifest.runtime_asset_root(manifest.DEFAULT_RUNTIME_CFG)
    csv = manifest.resolve_runtime_path(
        "a3_data/agibot_a3/001_walk_front_slow.csv",
        Path(),
        root,
    )

    assert csv == manifest.REPO_ROOT / "a3_data/agibot_a3/001_walk_front_slow.csv"
    assert csv.exists()
