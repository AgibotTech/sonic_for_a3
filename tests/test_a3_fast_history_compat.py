from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

COMMANDS = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic/envs/manager_env/mdp/commands.py"
)


def _load_function(name: str):
    tree = ast.parse(COMMANDS.read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(COMMANDS), "exec"), namespace)
    return namespace[name]


def _load_tracking_method(name: str):
    tree = ast.parse(COMMANDS.read_text())
    cls = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "TrackingCommand"
    )
    node = next(
        item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(COMMANDS), "exec"), namespace)
    return namespace[name]


def test_a3_fast_history_slots_and_valid_future_modes():
    build = _load_function("_build_a3_fast_relative_slots")
    history = build(10, 5, None, "cpu")
    assert history.tolist() == list(range(-5, 5))

    clamped = build(10, None, 5, "cpu")
    assert clamped.tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4, 4]

    with pytest.raises(ValueError, match="mutually exclusive"):
        build(10, 5, 5, "cpu")


def test_a3_fast_history_indices_are_clamped_at_motion_start():
    method = _load_tracking_method("a3_fast_future_time_steps")
    fake = SimpleNamespace(
        a3_fast_future_time_steps_init=torch.tensor([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]]),
        time_steps=torch.tensor([0]),
        motion_start_time_steps=torch.tensor([0]),
        motion_num_steps=torch.tensor([3]),
        _cached_reference=lambda _key, factory: factory(),
    )
    assert method(fake).tolist() == [0, 0, 0, 0, 0, 0, 1, 2, 2, 2]


def test_a3_fast_zero_padding_preserves_shape_and_zeros_tail_frames():
    zero_pad = _load_function("_zero_pad_a3_fast_future_frames")
    values = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, -1)

    unchanged = zero_pad(values, 10, None, False)
    assert torch.equal(unchanged, values)

    padded = zero_pad(values, 10, 5, True).reshape(2, 10, 3)
    assert padded.shape == (2, 10, 3)
    assert torch.equal(padded[:, :5], values.reshape(2, 10, 3)[:, :5])
    assert torch.count_nonzero(padded[:, 5:]) == 0

    with pytest.raises(ValueError, match="requires a3_fast_valid_future_frames"):
        zero_pad(values, 10, None, True)
