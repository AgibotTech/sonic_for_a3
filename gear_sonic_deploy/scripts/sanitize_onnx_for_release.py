#!/usr/bin/env python3
"""Remove non-semantic provenance strings from ONNX artifacts before release.

PyTorch exports can place source locations in nested ONNX ``doc_string``
fields.  They are not used by ONNX Runtime or RKNN, but they can disclose
developer-machine paths.  This tool strips those fields, clears free-form
metadata, verifies the resulting graph and refuses to write an artifact that
still contains an absolute home or mount path in a protobuf string field.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
from typing import Iterator

import onnx
from google.protobuf.message import Message


PATH_PATTERN = re.compile(r"(?:^|[^A-Za-z0-9_])/(?:home|mnt)/")


def _is_repeated(field) -> bool:
    """Support both protobuf descriptor APIs used by ONNX wheels."""

    if hasattr(field, "is_repeated"):
        return bool(field.is_repeated)
    return field.label == field.LABEL_REPEATED


def iter_string_fields(message: Message, prefix: str = "model") -> Iterator[tuple[str, str]]:
    """Yield every non-empty protobuf string field, including nested graphs."""

    for field in message.DESCRIPTOR.fields:
        value = getattr(message, field.name)
        field_path = f"{prefix}.{field.name}"
        if field.type == field.TYPE_STRING:
            if _is_repeated(field):
                for index, item in enumerate(value):
                    if item:
                        yield f"{field_path}[{index}]", item
            elif value:
                yield field_path, value
        elif field.type == field.TYPE_MESSAGE:
            if _is_repeated(field):
                for index, child in enumerate(value):
                    yield from iter_string_fields(child, f"{field_path}[{index}]")
            elif message.HasField(field.name):
                yield from iter_string_fields(value, field_path)


def find_internal_paths(model: onnx.ModelProto) -> list[tuple[str, str]]:
    """Return protobuf string fields that contain a developer filesystem path."""

    return [(path, value) for path, value in iter_string_fields(model) if PATH_PATTERN.search(value)]


def strip_doc_strings(message: Message) -> None:
    """Recursively clear doc strings across protobuf runtime versions.

    ONNX 1.21's helper currently expects the legacy descriptor ``label`` API,
    while newer protobuf wheels expose ``is_repeated`` instead.  Keep the
    traversal local so the release tool works with both versions.
    """

    for field in message.DESCRIPTOR.fields:
        if field.name == "doc_string":
            message.ClearField(field.name)
        elif field.type == field.TYPE_MESSAGE:
            value = getattr(message, field.name)
            if _is_repeated(field):
                for child in value:
                    strip_doc_strings(child)
            elif message.HasField(field.name):
                strip_doc_strings(value)


def sanitize_model(model: onnx.ModelProto) -> None:
    """Clear non-semantic debug fields without changing graph operations or weights."""

    strip_doc_strings(model)
    del model.metadata_props[:]


def sanitize_file(path: Path) -> tuple[int, int]:
    """Sanitize one model atomically and return (paths_before, paths_after)."""

    model = onnx.load(str(path), load_external_data=False)
    before = find_internal_paths(model)
    sanitize_model(model)
    after = find_internal_paths(model)
    if after:
        fields = ", ".join(field for field, _ in after[:5])
        raise RuntimeError(f"{path} still contains internal path(s) in {fields}")
    onnx.checker.check_model(model)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)
    try:
        onnx.save(model, str(temp_path))
        # Verify the serialized payload, not only the in-memory graph.
        written = onnx.load(str(temp_path), load_external_data=False)
        onnx.checker.check_model(written)
        written_after = find_internal_paths(written)
        if written_after:
            fields = ", ".join(field for field, _ in written_after[:5])
            raise RuntimeError(f"serialized {path} still contains internal path(s) in {fields}")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return len(before), len(after)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, nargs="+", help="ONNX files to sanitize in place")
    args = parser.parse_args()

    for path in args.onnx:
        path = path.resolve()
        if not path.is_file() or path.suffix.lower() != ".onnx":
            raise FileNotFoundError(f"not an ONNX file: {path}")
        before, after = sanitize_file(path)
        print(f"sanitized {path}: internal-path fields {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
