from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import Mock

import torch

COMMANDS = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic/envs/manager_env/mdp/commands.py"
)


def _quat_inv(value):
    return -value


def _quat_mul(left, right):
    return left + 2 * right


def _quat_apply(quaternion, vector):
    return quaternion[..., :3] + vector


class _TorchTransform:
    @staticmethod
    def get_heading_q(value):
        return 3 * value


METHOD_GLOBALS = {
    "Sequence": Sequence,
    "torch": torch,
    "torch_transform": _TorchTransform,
    "quat_inv": _quat_inv,
    "quat_mul": _quat_mul,
    "quat_apply": _quat_apply,
}


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
    namespace = dict(METHOD_GLOBALS)
    exec(compile(ast.fix_missing_locations(module), str(COMMANDS), "exec"), namespace)
    return namespace[name]


def test_empty_resample_only_refreshes_relative_reference():
    resample_command = _load_tracking_method("_resample_command")
    command = SimpleNamespace(_update_relative_reference=Mock())

    resample_command(command, torch.empty(0, dtype=torch.long))

    command._update_relative_reference.assert_called_once_with()


def test_nonempty_resample_allocates_only_the_reset_subset():
    source = COMMANDS.read_text()
    tree = ast.parse(source)
    tracking_command = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand"
    )
    resample_source = ast.unparse(
        next(
            node
            for node in tracking_command.body
            if isinstance(node, ast.FunctionDef) and node.name == "_resample_command"
        )
    )
    reset_state = source.split("# Reset only the requested environments.", 1)[1].split(
        "# Handle object positioning", 1
    )[0]

    assert "self.body_pos_w[env_ids, 0].clone()" in reset_state
    assert "self.joint_pos[env_ids].clone()" in reset_state
    assert "(len(env_ids), self.motion_lib_num_dof)" in reset_state
    assert "self.num_envs, self.robot_num_dof" not in reset_state
    assert "joint_pos[env_ids] =" not in reset_state
    assert "write_joint_state_to_sim(joint_pos[env_ids]" not in reset_state
    assert "root_pos[env_ids] +=" not in reset_state
    assert "if not self._requires_encoder_availability_filter" in source
    assert "sampling_cases = [(env_ids, self.encoder_sample_probs)]" in source
    assert resample_source.count("get_motion_num_steps") == 1


def test_relative_reference_helper_matches_legacy_formula():
    update_relative_reference = _load_tracking_method("_update_relative_reference")
    torch.manual_seed(7)
    num_envs = 5
    num_bodies = 3
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=[f"body_{index}" for index in range(num_bodies)]),
        anchor_pos_w=torch.randn(num_envs, 3),
        anchor_quat_w=torch.randn(num_envs, 4),
        robot_anchor_pos_w=torch.randn(num_envs, 3),
        robot_anchor_quat_w=torch.randn(num_envs, 4),
        body_pos_w=torch.randn(num_envs, num_bodies, 3),
        body_quat_w=torch.randn(num_envs, num_bodies, 4),
    )

    anchor_pos = command.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat = command.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_pos = command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    delta_pos = robot_anchor_pos
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_quat = _TorchTransform.get_heading_q(
        _quat_mul(robot_anchor_quat, _quat_inv(anchor_quat))
    )
    expected_quat = _quat_mul(delta_quat, command.body_quat_w)
    expected_pos = delta_pos + _quat_apply(delta_quat, command.body_pos_w - anchor_pos)

    update_relative_reference(command)

    torch.testing.assert_close(command.body_quat_relative_w, expected_quat)
    torch.testing.assert_close(command.body_pos_relative_w, expected_pos)


def test_current_reference_cache_reuses_motion_lookup_until_frame_changes():
    cached_reference = _load_tracking_method("_cached_reference")
    invalidate_reference_cache = _load_tracking_method("_invalidate_reference_cache")
    body_pos_w = _load_tracking_method("body_pos_w")
    anchor_pos_w = _load_tracking_method("anchor_pos_w")

    class MotionLib:
        def __init__(self):
            self.calls = 0

        def get_body_pos_w(self, motion_ids, time_steps):
            self.calls += 1
            return torch.arange(30, dtype=torch.float32).view(2, 5, 3)

    command_type = type(
        "CachedCommand",
        (),
        {
            "_cached_reference": cached_reference,
            "_invalidate_reference_cache": invalidate_reference_cache,
            "body_pos_w": property(body_pos_w),
            "anchor_pos_w": property(anchor_pos_w),
        },
    )
    command = command_type()
    command.cache_current_references = True
    command.motion_lib = MotionLib()
    command.motion_ids = torch.tensor([0, 1])
    command.motion_start_time_steps = torch.tensor([0, 0])
    command.time_steps = torch.tensor([2, 3])
    command.motion_anchor_body_index = 1
    command._env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    )

    first = command.body_pos_w
    second = command.body_pos_w
    anchor = command.anchor_pos_w

    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(anchor, first[:, 1])
    assert command.motion_lib.calls == 1

    command._invalidate_reference_cache()
    third = command.body_pos_w
    assert third.data_ptr() != first.data_ptr()
    assert command.motion_lib.calls == 2


def test_current_reference_cache_is_opt_in_for_alias_compatibility():
    cached_reference = _load_tracking_method("_cached_reference")
    command = SimpleNamespace(cache_current_references=False)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return torch.tensor(float(calls))

    first = cached_reference(command, "value", factory)
    second = cached_reference(command, "value", factory)

    assert first.item() == 1
    assert second.item() == 2
    assert calls == 2
