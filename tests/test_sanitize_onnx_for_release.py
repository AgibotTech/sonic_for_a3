"""Tests for the release-safe ONNX metadata sanitizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic_deploy/scripts/sanitize_onnx_for_release.py"
)
SPEC = importlib.util.spec_from_file_location("sanitize_onnx_for_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sanitizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sanitizer
SPEC.loader.exec_module(sanitizer)


def test_sanitize_removes_nested_trace_docs_and_metadata() -> None:
    home_path = "/" + "home" + "/example"
    mount_path = "/" + "mnt" + "/build"
    node = helper.make_node("Identity", ["obs_dict"], ["action"])
    node.doc_string = f"{home_path}/export.py(12): forward"
    graph = helper.make_graph(
        [node],
        "policy",
        [helper.make_tensor_value_info("obs_dict", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 2])],
    )
    graph.doc_string = f"{mount_path}/graph"
    model = helper.make_model(graph)
    model.doc_string = f"{home_path}/model"
    entry = model.metadata_props.add()
    entry.key = "provenance"
    entry.value = f"{home_path}/metadata"

    assert sanitizer.find_internal_paths(model)
    sanitizer.sanitize_model(model)

    assert not sanitizer.find_internal_paths(model)
    assert not model.doc_string
    assert not model.graph.doc_string
    assert not model.graph.node[0].doc_string
    assert not model.metadata_props
    onnx.checker.check_model(model)
