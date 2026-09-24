"""Tests for bounded, rank-aware distributed startup diagnostics."""

from collections import OrderedDict

import pytest
import torch

from gear_sonic.train_agent_trl import _get_distributed_timeout_seconds
from gear_sonic.trl.utils.common import (
    build_module_parameter_signature,
    distributed_stage,
)


def test_module_parameter_signature_is_stable_and_sensitive_to_model_shape():
    torch.manual_seed(7)
    first = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    torch.manual_seed(11)
    same_shape = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    different_shape = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2))

    first_signature = build_module_parameter_signature(
        OrderedDict([("policy", first), ("unused", None)])
    )
    same_shape_signature = build_module_parameter_signature(
        OrderedDict([("policy", same_shape), ("unused", None)])
    )
    different_signature = build_module_parameter_signature(
        OrderedDict([("policy", different_shape), ("unused", None)])
    )

    assert first_signature == same_shape_signature
    assert first_signature["digest"] != different_signature["digest"]
    assert first_signature["parameter_tensors"] == 4
    assert first_signature["parameter_numel"] == 26


def test_distributed_stage_reports_begin_complete_and_failure():
    messages = []

    result = distributed_stage("prepare", lambda: 17, log_fn=messages.append)

    assert result == 17
    assert messages[0].startswith("[DistributedStage] begin stage=prepare")
    assert messages[-1].startswith("[DistributedStage] complete stage=prepare")

    messages.clear()
    with pytest.raises(RuntimeError, match="broken"):
        distributed_stage(
            "prepare",
            lambda: (_ for _ in ()).throw(RuntimeError("broken")),
            log_fn=messages.append,
        )
    assert messages[-1].startswith("[DistributedStage] failed stage=prepare")
    assert "error_type=RuntimeError" in messages[-1]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "not-a-number"])
def test_distributed_timeout_env_must_be_positive_finite(monkeypatch, value):
    monkeypatch.setenv("SONIC_DISTRIBUTED_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="positive"):
        _get_distributed_timeout_seconds()
