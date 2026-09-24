from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import accelerate
import torch

from gear_sonic.trl.trainer.ppo_trainer import (
    TRLPPOTrainer,
    _compact_completed_episode_stats,
    process_ep_infos,
)

PPO_TRAINER = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic/trl/trainer/ppo_trainer.py"
)


def test_episode_info_reduction_concatenates_once_without_mutating_inputs():
    ep_infos = [
        {
            "reward": torch.tensor([1.0, 3.0]),
            "length": 2.0,
            "integer_metric": torch.tensor([1, 3]),
            "double_metric": torch.tensor([1.0, 3.0], dtype=torch.float64),
        },
        {
            "reward": torch.tensor([5.0]),
            "length": 4.0,
            "integer_metric": torch.tensor([5]),
            "double_metric": torch.tensor([5.0], dtype=torch.float64),
        },
    ]

    reduced = process_ep_infos(ep_infos, torch.device("cpu"))

    torch.testing.assert_close(reduced["reward"], torch.tensor(3.0))
    torch.testing.assert_close(reduced["length"], torch.tensor(3.0))
    torch.testing.assert_close(reduced["integer_metric"], torch.tensor(3.0))
    torch.testing.assert_close(
        reduced["double_metric"], torch.tensor(3.0, dtype=torch.float64)
    )
    assert ep_infos[0]["length"] == 2.0
    assert ep_infos[1]["length"] == 4.0


def test_rollout_loop_has_no_per_step_device_to_host_copy():
    tree = ast.parse(PPO_TRAINER.read_text())
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRLPPOTrainer"
    )
    rollout = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_rollout_step"
    )
    step_loop = next(
        node
        for node in ast.walk(rollout)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "i"
    )

    cpu_calls = [
        node
        for node in ast.walk(ast.Module(body=step_loop.body, type_ignores=[]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cpu"
    ]
    assert cpu_calls == []

    step_source = ast.unparse(ast.Module(body=step_loop.body, type_ignores=[]))
    assert ".nonzero(" not in step_source
    assert "completed_episode_masks.append(completed_mask)" in step_source


def test_dense_episode_snapshots_preserve_legacy_completion_order_and_shape():
    reward_snapshots = [
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[4.0], [5.0], [6.0]]),
    ]
    length_snapshots = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    masks = [torch.tensor([False, True, False]), torch.tensor([True, False, True])]

    reward_sums, episode_lengths = _compact_completed_episode_stats(
        reward_snapshots, length_snapshots, masks
    )

    # Legacy ``nonzero(as_tuple=False)`` indexing produced an extra singleton
    # axis; retain it so checkpoint/log buffer nesting is unchanged.
    torch.testing.assert_close(
        reward_sums, torch.tensor([[[2.0]], [[4.0]], [[6.0]]])
    )
    torch.testing.assert_close(
        episode_lengths, torch.tensor([[2.0], [4.0], [6.0]])
    )


def test_training_stop_still_runs_callback_finalizers():
    tree = ast.parse(PPO_TRAINER.read_text())
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TRLPPOTrainer"
    )
    train = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    train_source = ast.unparse(train)

    assert "callback_handler.on_train_end" in train_source
    assert "if self.control.should_training_stop:\n        return" not in train_source


class _CpuAccelerator:
    device = torch.device("cpu")
    distributed_type = accelerate.DistributedType.NO
    process_index = 0

    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _gradient_test_trainer(model, *, max_grad_norm=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return SimpleNamespace(
        args=SimpleNamespace(max_grad_norm=max_grad_norm, fp16=False),
        model=model,
        optimizer=optimizer,
        accelerator=_CpuAccelerator(),
        use_apex=False,
    )


def test_gradient_clipping_uses_global_norm_as_finite_check():
    model = torch.nn.Linear(4, 2)
    model(torch.ones(3, 4)).sum().backward()
    trainer = _gradient_test_trainer(model)

    grad_norm = TRLPPOTrainer._gradient_clipping(trainer)

    assert grad_norm is not None
    assert torch.isfinite(grad_norm)
    assert torch.nn.utils.get_total_norm(
        [parameter.grad for parameter in model.parameters()]
    ) <= 0.100001


def test_gradient_clipping_skips_nonfinite_update():
    model = torch.nn.Linear(4, 2)
    model(torch.ones(3, 4)).sum().backward()
    model.weight.grad[0, 0] = float("nan")
    trainer = _gradient_test_trainer(model)

    grad_norm = TRLPPOTrainer._gradient_clipping(trainer)

    assert grad_norm is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_finite_gradients_without_clipping_still_allow_optimizer_step():
    model = torch.nn.Linear(4, 2)
    model(torch.ones(3, 4)).sum().backward()
    trainer = _gradient_test_trainer(model, max_grad_norm=0.0)

    grad_norm = TRLPPOTrainer._gradient_clipping(trainer)

    assert grad_norm is not None
    assert torch.isfinite(grad_norm)
