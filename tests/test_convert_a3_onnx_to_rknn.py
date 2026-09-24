"""Unit tests for the standalone A3 ONNX-to-RKNN conversion entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic_deploy/scripts/convert_a3_onnx_to_rknn.py"
)
SPEC = importlib.util.spec_from_file_location("convert_a3_onnx_to_rknn", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _schema(batch_dim: int | str = "batch") -> dict:
    return {
        "input_name": "obs_dict",
        "input_shape": [batch_dim, 1570],
        "output_name": "action",
        "output_shape": [batch_dim, 29],
        "opset": [{"domain": "ai.onnx", "version": 17}],
        "ir_version": 8,
    }


def test_dynamic_batch_onnx_is_bound_to_one_for_rknn_load() -> None:
    """RKNN receives a concrete one-robot shape without rewriting the ONNX file."""

    model_spec = converter.ModelSpec(
        label="external",
        onnx_key="",
        rknn_key="",
        expected_input_dim=1570,
    )

    kwargs = converter.rknn_load_kwargs(
        model_spec,
        _schema(),
        Path("/tmp/external_motion.onnx"),
    )

    assert kwargs == {
        "model": "/tmp/external_motion.onnx",
        "inputs": ["obs_dict"],
        "input_size_list": [[1, 1570]],
        "outputs": ["action"],
    }


def test_static_batch_onnx_keeps_the_existing_rknn_load_contract() -> None:
    """Existing deployment ONNX files do not receive unnecessary crop arguments."""

    model_spec = converter.ModelSpec(
        label="g1",
        onnx_key="model_path",
        rknn_key="rknn_model_path",
        expected_input_dim=1570,
    )

    kwargs = converter.rknn_load_kwargs(
        model_spec,
        _schema(batch_dim=1),
        Path("/tmp/static_motion.onnx"),
    )

    assert kwargs == {"model": "/tmp/static_motion.onnx"}


def test_external_onnx_directory_is_expanded_in_stable_order(tmp_path: Path) -> None:
    """The generic entrypoint accepts a directory without picking unrelated files."""

    (tmp_path / "zeta.onnx").touch()
    (tmp_path / "alpha.onnx").touch()
    (tmp_path / "notes.txt").touch()

    paths = converter.expand_onnx_inputs([tmp_path])

    assert paths == [tmp_path / "alpha.onnx", tmp_path / "zeta.onnx"]


def test_onnx_doc_string_helper_supports_current_protobuf_descriptor_api() -> None:
    """RKNN's optimizer calls this helper even though doc strings are unused."""

    node = helper.make_node("Identity", ["obs_dict"], ["action"])
    node.doc_string = "export trace"
    model = helper.make_model(
        helper.make_graph(
            [node],
            "policy",
            [helper.make_tensor_value_info("obs_dict", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 2])],
        )
    )

    converter.ensure_onnx_helper_compat()
    onnx.helper.strip_doc_string(model)

    assert not model.graph.node[0].doc_string


def test_onnx_mapping_compat_restores_rknn_constant_fold_maps() -> None:
    """Toolkit2 needs the old bidirectional dtype maps, not just one of them."""

    converter.ensure_onnx_mapping_compat()

    assert onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[TensorProto.FLOAT] is not None
    assert onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[
        onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[TensorProto.FLOAT]
    ] == TensorProto.FLOAT


def test_manifest_paths_are_relative_to_the_sidecar_directory(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "rknn_models" / "release"
    model_path = tmp_path / "models" / "release" / "policy.onnx"

    assert converter.manifest_relative_path(model_path, manifest_dir) == "../../models/release/policy.onnx"
    assert converter.manifest_relative_path(manifest_dir / "policy.rknn", manifest_dir) == "policy.rknn"
