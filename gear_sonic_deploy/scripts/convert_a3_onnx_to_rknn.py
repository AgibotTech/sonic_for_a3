#!/usr/bin/env python3
"""Convert A3 monolithic ONNX policies to RKNN models.

The script intentionally does not edit the runtime YAML by default. It prints
the generated `onnx.rknn_*_model_path` values after conversion so callers can
decide when to switch `onnx.backend` to `rknn`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

GEAR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GEAR_ROOT.parent
DEFAULT_RUNTIME_CFG = (
    GEAR_ROOT
    / "src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml"
)
DEFAULT_OUT_DIR = GEAR_ROOT / "assets/a3_runtime/rknn_models"
SUPPORTED_EXTERNAL_INPUT_DIMS = {1570, 1770}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    onnx_key: str | None
    rknn_key: str | None
    expected_input_dim: int
    expected_action_dim: int = 29


MODEL_SPECS = {
    "g1": ModelSpec(
        label="g1",
        onnx_key="model_path",
        rknn_key="rknn_model_path",
        expected_input_dim=1570,
    ),
    "smpl": ModelSpec(
        label="smpl",
        onnx_key="smpl_model_path",
        rknn_key="smpl_rknn_model_path",
        expected_input_dim=1770,
    ),
    "a3_fast": ModelSpec(
        label="a3_fast",
        onnx_key="a3_fast_model_path",
        rknn_key="a3_fast_rknn_model_path",
        expected_input_dim=1570,
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_relative_to(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def manifest_relative_path(path: Path, manifest_dir: Path) -> str:
    """Return a portable path relative to the generated sidecar location."""

    return Path(os.path.relpath(path.resolve(), start=manifest_dir.resolve())).as_posix()


def resolve_path(raw: str, runtime_cfg: Path) -> Path:
    p = Path(raw).expanduser()
    candidates = (
        [p]
        if p.is_absolute()
        else [REPO_ROOT / p, GEAR_ROOT / p, runtime_cfg.parent / p]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"{raw} does not exist; tried: {tried}")


def load_runtime_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg.get("onnx"), dict):
        raise ValueError("runtime config is missing top-level onnx mapping")
    mode = str(cfg["onnx"].get("mode") or "monolithic").lower().replace("-", "_")
    if mode in ("encoder_decoder", "encoderdecoder", "split"):
        raise ValueError("RKNN conversion supports monolithic A3 policies only")
    return cfg


def probe_onnx_schema(path: Path) -> dict:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required for schema validation: pip install onnx") from exc

    model = onnx.load(str(path))
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1:
        raise ValueError(f"{path} must have exactly 1 input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"{path} must have exactly 1 output, got {len(outputs)}")

    def tensor_shape(value_info) -> list[int | str]:
        dims: list[int | str] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param") and dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
        return dims

    return {
        "input_name": inputs[0].name,
        "input_shape": tensor_shape(inputs[0]),
        "output_name": outputs[0].name,
        "output_shape": tensor_shape(outputs[0]),
        "opset": [
            {"domain": op.domain or "ai.onnx", "version": int(op.version)}
            for op in model.opset_import
        ],
        "ir_version": int(model.ir_version),
    }


def _is_supported_batch_dim(value: int | str) -> bool:
    """Accept the static deploy batch or an ONNX symbolic batch dimension."""
    return value == 1 or isinstance(value, str)


def validate_schema(spec: ModelSpec, schema: dict, path: Path) -> None:
    if schema["input_name"] != "obs_dict":
        raise ValueError(f"{path} input name must be obs_dict, got {schema['input_name']}")
    if schema["output_name"] != "action":
        raise ValueError(f"{path} output name must be action, got {schema['output_name']}")
    if (
        len(schema["input_shape"]) != 2
        or not _is_supported_batch_dim(schema["input_shape"][0])
        or schema["input_shape"][1] != spec.expected_input_dim
    ):
        raise ValueError(
            f"{path} input shape must be [1|batch, {spec.expected_input_dim}], "
            f"got {schema['input_shape']}"
        )
    if (
        len(schema["output_shape"]) != 2
        or not _is_supported_batch_dim(schema["output_shape"][0])
        or schema["output_shape"][1] != spec.expected_action_dim
    ):
        raise ValueError(
            f"{path} output shape must be [1|batch, {spec.expected_action_dim}], "
            f"got {schema['output_shape']}"
        )


def rknn_load_kwargs(spec: ModelSpec, schema: dict, onnx_path: Path) -> dict:
    """Bind a compatible ONNX policy to one robot inference per RKNN call."""
    validate_schema(spec, schema, onnx_path)
    if schema["input_shape"][0] == 1 and schema["output_shape"][0] == 1:
        return {"model": str(onnx_path)}
    return {
        "model": str(onnx_path),
        "inputs": [schema["input_name"]],
        "input_size_list": [[1, spec.expected_input_dim]],
        "outputs": [schema["output_name"]],
    }


def expand_onnx_inputs(inputs: Iterable[Path]) -> list[Path]:
    """Expand explicit ONNX files and one-level ONNX directories deterministically."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser()
        if path.is_dir():
            candidates = sorted(path.glob("*.onnx"))
        elif path.is_file() and path.suffix.lower() == ".onnx":
            candidates = [path]
        else:
            raise FileNotFoundError(
                f"{path} is not an ONNX file or a directory containing ONNX files"
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    if not paths:
        raise FileNotFoundError("no .onnx files were found in --onnx input")
    return paths


def external_model_spec(onnx_path: Path) -> ModelSpec:
    """Infer a supported monolithic A3 interface for a direct ONNX conversion."""
    schema = probe_onnx_schema(onnx_path)
    input_shape = schema["input_shape"]
    if len(input_shape) != 2 or not isinstance(input_shape[1], int):
        raise ValueError(f"{onnx_path} must have a rank-2 input with a fixed feature dimension")
    input_dim = input_shape[1]
    if input_dim not in SUPPORTED_EXTERNAL_INPUT_DIMS:
        raise ValueError(
            f"{onnx_path} input feature dimension {input_dim} is unsupported; "
            f"expected one of {sorted(SUPPORTED_EXTERNAL_INPUT_DIMS)}"
        )
    return ModelSpec(
        label=onnx_path.stem,
        onnx_key=None,
        rknn_key=None,
        expected_input_dim=input_dim,
    )


def ensure_onnx_mapping_compat() -> None:
    """Provide onnx.mapping for RKNN Toolkit2 when running with ONNX >= 1.20."""
    import onnx

    if hasattr(onnx, "mapping"):
        return
    try:
        from onnx import _mapping
    except ImportError as exc:
        raise RuntimeError(
            "RKNN Toolkit2 2.3.2 needs onnx.mapping; install onnx<=1.19 "
            "or use an ONNX build with onnx._mapping"
        ) from exc

    tensor_map = _mapping.TENSOR_TYPE_MAP
    tensor_to_np = {key: value.np_dtype for key, value in tensor_map.items()}
    # Toolkit2 still uses both of these removed ONNX module-level maps while
    # folding constants.  Keep dtype objects (rather than their string names)
    # as keys, matching ONNX <= 1.19.
    np_to_tensor = {}
    for key, value in tensor_map.items():
        np_to_tensor.setdefault(value.np_dtype, key)
    tensor_to_storage = {key: value.storage_dtype for key, value in tensor_map.items()}
    compat = types.SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE=tensor_to_np,
        NP_TYPE_TO_TENSOR_TYPE=np_to_tensor,
        TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE=tensor_to_storage,
        STORAGE_TENSOR_TYPE_TO_FIELD={
            int(onnx.TensorProto.FLOAT): "float_data",
            int(onnx.TensorProto.INT32): "int32_data",
            int(onnx.TensorProto.INT64): "int64_data",
            int(onnx.TensorProto.UINT8): "int32_data",
            int(onnx.TensorProto.UINT16): "int32_data",
            int(onnx.TensorProto.DOUBLE): "double_data",
            int(onnx.TensorProto.COMPLEX64): "float_data",
            int(onnx.TensorProto.COMPLEX128): "double_data",
            int(onnx.TensorProto.UINT32): "uint64_data",
            int(onnx.TensorProto.UINT64): "uint64_data",
            int(onnx.TensorProto.STRING): "string_data",
            int(onnx.TensorProto.BOOL): "int32_data",
        },
    )
    onnx.mapping = compat
    sys.modules["onnx.mapping"] = compat


def ensure_onnx_helper_compat() -> None:
    """Patch ONNX's doc-string walker for protobuf wheels without ``label``.

    RKNN Toolkit2 calls ``onnx.helper.strip_doc_string`` during graph
    optimization.  ONNX 1.21 still implements that helper with the legacy
    ``FieldDescriptor.label`` property, whereas newer protobuf wheels expose
    ``is_repeated``.  The replacement only clears non-semantic doc strings.
    """

    import onnx

    probe = onnx.ModelProto()
    descriptor = probe.DESCRIPTOR.fields[0]
    if hasattr(descriptor, "label"):
        return

    def is_repeated(field) -> bool:
        return bool(getattr(field, "is_repeated", False))

    def strip_doc_string(proto) -> None:
        for field in proto.DESCRIPTOR.fields:
            if field.name == "doc_string":
                proto.ClearField(field.name)
            elif field.type == field.TYPE_MESSAGE:
                value = getattr(proto, field.name)
                if is_repeated(field):
                    for child in value:
                        strip_doc_string(child)
                elif proto.HasField(field.name):
                    strip_doc_string(value)

    onnx.helper.strip_doc_string = strip_doc_string


def convert_one(
    spec: ModelSpec,
    onnx_path: Path,
    out_dir: Path,
    target_platform: str,
    overwrite: bool,
    verbose: bool,
) -> dict:
    ensure_onnx_mapping_compat()
    ensure_onnx_helper_compat()
    from rknn.api import RKNN

    schema = probe_onnx_schema(onnx_path)
    load_kwargs = rknn_load_kwargs(spec, schema, onnx_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rknn_path = out_dir / f"{onnx_path.stem}.rknn"
    manifest_path = out_dir / f"{onnx_path.stem}.rknn.json"
    if rknn_path.exists() and not overwrite:
        raise FileExistsError(f"{rknn_path} exists; pass --overwrite to replace it")

    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with ret={ret}")
        ret = rknn.load_onnx(**load_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with ret={ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with ret={ret}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with ret={ret}")
    finally:
        rknn.release()

    manifest = {
        "label": spec.label,
        "target_platform": target_platform,
        "do_quantization": False,
        "source_onnx": manifest_relative_path(onnx_path, out_dir),
        "source_onnx_sha256": sha256_file(onnx_path),
        "rknn": manifest_relative_path(rknn_path, out_dir),
        "rknn_sha256": sha256_file(rknn_path),
        "schema": schema,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def selected_specs(names: Iterable[str]) -> list[ModelSpec]:
    specs = []
    for name in names:
        try:
            specs.append(MODEL_SPECS[name])
        except KeyError as exc:
            raise ValueError(f"unknown model '{name}'") from exc
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-cfg", type=Path, default=DEFAULT_RUNTIME_CFG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--onnx",
        type=Path,
        nargs="+",
        metavar="PATH",
        help=(
            "Direct ONNX input: one or more monolithic A3 ONNX files or directories. "
            "A symbolic batch dimension is automatically bound to batch=1 for RKNN."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["g1", "smpl", "a3_fast"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (GEAR_ROOT / out_dir).resolve()

    conversion_jobs: list[tuple[ModelSpec, Path]] = []
    if args.onnx:
        for onnx_path in expand_onnx_inputs(args.onnx):
            conversion_jobs.append((external_model_spec(onnx_path), onnx_path))
    else:
        cfg_path = args.runtime_cfg.resolve()
        cfg = load_runtime_cfg(cfg_path)
        for spec in selected_specs(args.models):
            raw = cfg["onnx"].get(spec.onnx_key)
            if not raw:
                print(f"[skip] onnx.{spec.onnx_key} is not configured")
                continue
            conversion_jobs.append((spec, resolve_path(str(raw), cfg_path)))

    manifests = []
    for spec, onnx_path in conversion_jobs:
        print(f"[convert] {spec.label}: {onnx_path}")
        manifest = convert_one(
            spec=spec,
            onnx_path=onnx_path,
            out_dir=out_dir,
            target_platform=args.target_platform,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        manifests.append((spec, manifest))
        print(f"[write] {manifest['rknn']}")
        print(f"[write] {manifest['rknn']}.json")

    if not manifests:
        print("no models converted", file=sys.stderr)
        return 2

    print("\n# Add these under `onnx:` when switching to RKNN:")
    print("backend: rknn")
    print("rknn_core_mask: auto")
    for spec, manifest in manifests:
        if spec.rknn_key is None:
            print(f"# external {spec.label}: {manifest['rknn']}")
            continue
        rknn_path = Path(manifest["rknn"])
        rel = path_relative_to(rknn_path, GEAR_ROOT)
        value = rel if rel is not None else str(rknn_path)
        print(f"{spec.rknn_key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
