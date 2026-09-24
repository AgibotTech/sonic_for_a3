from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from tensordict import TensorDict
import torch
import torch.nn as nn

from gear_sonic.trl.modules.actor_critic_modules import Actor
from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper, TRLPPOTrainer
from gear_sonic.utils.motion_lib import motion_lib_base
from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase, MotionlibMode
from gear_sonic.utils.running_mean_std import RunningMeanStd

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "gear_sonic/envs/wrapper/manager_env_wrapper.py"


def _bare_actor(*, max_rollout_history=1, skip_history_one_mask=False):
    actor = Actor.__new__(Actor)
    nn.Module.__init__(actor)
    actor.max_rollout_history = max_rollout_history
    actor.skip_rollout_attnmask_for_history_one = skip_history_one_mask
    actor.obs_dict_buffer = TensorDict()
    actor.dones_buffer = None
    actor.steps = 0
    return actor


def test_actor_speed_flag_preserves_legacy_backbone_kwargs_position():
    parameters = list(inspect.signature(Actor.__init__).parameters)
    assert parameters.index("backbone_kwargs") < parameters.index(
        "skip_rollout_attnmask_for_history_one"
    )


def test_motion_loader_raises_only_soft_file_descriptor_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        motion_lib_base.resource,
        "getrlimit",
        lambda _kind: (1024, 524288),
    )
    monkeypatch.setattr(
        motion_lib_base.resource,
        "setrlimit",
        lambda kind, limits: calls.append((kind, limits)),
    )

    new_limit = motion_lib_base._raise_nofile_soft_limit()

    assert new_limit == 524288
    assert calls == [
        (motion_lib_base.resource.RLIMIT_NOFILE, (524288, 524288))
    ]


def test_history_one_rollout_replaces_views_without_concat_or_mask():
    actor = _bare_actor(skip_history_one_mask=True)
    previous = torch.randn(7, 5)
    current = torch.randn(7, 5)
    actor.obs_dict_buffer["actor_obs"] = previous.unsqueeze(1)

    mask = Actor._update_obs_buffer(
        actor,
        {"actor_obs": current},
        cur_dones=torch.zeros(7, dtype=torch.bool),
    )

    assert mask is None
    assert actor.obs_dict_buffer["actor_obs"].shape == (7, 1, 5)
    assert actor.obs_dict_buffer["actor_obs"].data_ptr() == current.data_ptr()


def test_history_one_keeps_legacy_attention_mask_when_fastpath_disabled():
    actor = _bare_actor(skip_history_one_mask=False)
    current = torch.randn(7, 5)

    mask = Actor._update_obs_buffer(
        actor,
        {"actor_obs": current},
        cur_dones=torch.zeros(7, dtype=torch.bool),
    )

    assert mask.shape == (7, 1, 1)
    assert not mask.any()


def test_action_std_validation_stays_on_device_and_sanitizes_invalid_values(monkeypatch):
    actor = _bare_actor()
    actor.use_log_std = False
    actor.std = nn.Parameter(torch.tensor([0.1, 1.0e-8, -1.0, float("nan")]))
    actor.clamp_noise_std = False
    actor.algo_config = {}
    actor.forward = MethodType(
        lambda self, obs_dict, **kwargs: obs_dict["actor_obs"], actor
    )

    monkeypatch.setattr(
        torch,
        "any",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("action std validation must not synchronize through torch.any")
        ),
    )
    actor.update_distribution({"actor_obs": torch.zeros(2, 4)})

    torch.testing.assert_close(
        actor.action_std[0], torch.tensor([0.1, 1.0e-6, 1.0e-6, 1.0e-6])
    )
    assert torch.isfinite(actor.action_std).all()


def _load_wrapper_helper(name):
    tree = ast.parse(WRAPPER_PATH.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(WRAPPER_PATH), "exec"), namespace)
    return namespace[name]


def test_adaptive_sampling_metric_helper_reuses_reductions_and_handles_zero_counts():
    collect = _load_wrapper_helper("_collect_adaptive_sampling_metrics")
    motion_lib = SimpleNamespace(
        adp_samp_num_episodes=torch.zeros(4),
        adp_samp_num_failures=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        adp_samp_failure_rate_raw=torch.tensor([0.0, 0.5, 1.0, 1.0]),
        adp_sampling_active_prob=torch.tensor([0.25, 0.75], dtype=torch.float64),
    )

    metrics = collect(motion_lib, motion_command=None)

    assert metrics["adp_samp/num_episodes_mean"] == 0
    assert metrics["adp_samp/episodes_max_over_mean"] == 0
    torch.testing.assert_close(
        metrics["adp_samp/prob_max"], torch.tensor(0.75, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["adp_samp/effective_num_bins"], torch.tensor(1.6, dtype=torch.float64)
    )


def test_wrapper_guards_optional_host_action_copy_and_checks_attribute_before_cuda_any():
    tree = ast.parse(WRAPPER_PATH.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerEnvWrapper"
    )
    step = next(
        node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    source = ast.unparse(step)

    assert "if self.store_env_actions_in_extras" in source
    assert "hasattr(self.env, '_prev_meta_action') and reset_mask.any()" in source


class _ReduceOnlyAccelerator:
    def __init__(self, reduced):
        self.reduced = reduced
        self.reduce_calls = 0
        self.gather_calls = 0

    def reduce(self, tensor, reduction="sum"):
        assert reduction == "sum"
        self.reduce_calls += 1
        return self.reduced.to(device=tensor.device, dtype=tensor.dtype)

    def gather(self, tensor):
        self.gather_calls += 1
        raise AssertionError("full advantage tensors must not be gathered")


def test_advantage_normalization_uses_distributed_sufficient_statistics():
    rank0 = torch.tensor([[[1.0]], [[2.0]]])
    rank1 = torch.tensor([[[3.0]], [[6.0]]])
    global_values = torch.cat([rank0, rank1], dim=1)
    global_sum = global_values.sum(dim=(0, 1), keepdim=True)
    global_square_sum = global_values.square().sum(dim=(0, 1), keepdim=True)
    global_count = torch.full_like(global_sum, global_values.shape[0] * global_values.shape[1])
    accelerator = _ReduceOnlyAccelerator(
        torch.stack([global_sum, global_square_sum, global_count])
    )
    trainer = SimpleNamespace(
        accelerator=accelerator,
        sync_advantage_normalization=True,
    )

    normalized = TRLPPOTrainer._normalize_advantages(trainer, rank0)
    expected = (rank0 - global_values.mean(dim=(0, 1), keepdim=True)) / (
        global_values.std(dim=(0, 1), keepdim=True) + 1.0e-8
    )

    torch.testing.assert_close(normalized, expected)
    assert accelerator.reduce_calls == 1
    assert accelerator.gather_calls == 0


def test_adaptive_sampler_sync_uses_reduce_without_world_sized_gather():
    class Accelerator:
        def __init__(self):
            self.reduce_calls = 0

        def reduce(self, tensor, reduction="mean"):
            assert reduction == "mean"
            self.reduce_calls += 1
            return tensor + 1

        def gather(self, tensor):
            raise AssertionError("adaptive sampler state must not be all-gathered")

    motion_lib = SimpleNamespace(
        use_adaptive_sampling=True,
        adaptive_sampling_sync_across_gpus=True,
        adaptive_sampling_cfg={"use_failure_rate_decay": False},
        adp_samp_num_episodes=torch.tensor([1.0, 2.0]),
        adp_samp_num_failures=torch.tensor([3.0, 4.0]),
        update_adaptive_sampling_probabilities=lambda: None,
    )
    accelerator = Accelerator()

    MotionLibBase.sync_and_compute_adaptive_sampling(
        motion_lib, accelerator=accelerator, sync_across_gpus=True
    )

    assert accelerator.reduce_calls == 1
    torch.testing.assert_close(motion_lib.adp_samp_num_episodes, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(motion_lib.adp_samp_num_failures, torch.tensor([4.0, 5.0]))


def test_validated_semantic_manifest_bypasses_recursive_motion_glob(tmp_path):
    dataset_root = tmp_path / "dataset"
    motion_source = dataset_root / "filtered" / "speed_1p00"
    motion_source.mkdir(parents=True)
    output_file = "filtered/speed_1p00/subdir/motion_a.pkl"
    manifest_contents = json.dumps(
        {
            "variants": [
                {
                    "output_key": "motion_a",
                    "output_file": output_file,
                    "frames": 366,
                    "fps": 30.0,
                }
            ]
        }
    ) + "\n"
    (dataset_root / "semantic_manifest.jsonl").write_text(manifest_contents)
    (dataset_root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": hashlib.sha256(
                    manifest_contents.encode()
                ).hexdigest(),
                "output_motions": 1,
            }
        )
    )
    motion_lib = MotionLibBase.__new__(MotionLibBase)
    motion_lib.m_cfg = {"use_semantic_manifest_metadata": True}
    motion_lib.debug = False

    sources = motion_lib._load_sources_from_semantic_manifests(
        [str(motion_source)]
    )

    assert list(sources) == [str(motion_source)]
    assert sources[str(motion_source)]["motion_a"] == {
        "path": str(dataset_root / output_file),
        "length": 366,
        "fps": 30.0,
    }

    motion_lib._load_single_motion_source = lambda _source: (_ for _ in ()).throw(
        AssertionError("validated manifest path must bypass recursive globbing")
    )
    motion_lib.load_data([str(motion_source)])
    assert motion_lib._motion_data_keys.tolist() == ["motion_a"]
    assert motion_lib._motion_data_list[0]["length"] == 366


def _make_manifest_fallback_fixture(tmp_path, manifest_contents, complete):
    dataset_root = tmp_path / "dataset"
    motion_source = dataset_root / "filtered" / "speed_1p00"
    motion_source.mkdir(parents=True)
    (dataset_root / "semantic_manifest.jsonl").write_text(manifest_contents)
    (dataset_root / "COMPLETE.json").write_text(json.dumps(complete))

    motion_lib = MotionLibBase.__new__(MotionLibBase)
    motion_lib.m_cfg = {"use_semantic_manifest_metadata": True}
    motion_lib.debug = False
    legacy_calls = []

    def legacy_loader(source):
        legacy_calls.append(str(source))
        return {"legacy_motion": {"path": "legacy.pkl"}}, MotionlibMode.directory

    motion_lib._load_single_motion_source = legacy_loader
    return motion_lib, motion_source, legacy_calls


def test_semantic_manifest_requires_checksum_and_count_or_falls_back(tmp_path):
    manifest_contents = json.dumps(
        {
            "variants": [
                {
                    "output_key": "motion_a",
                    "output_file": "filtered/speed_1p00/motion_a.pkl",
                    "frames": 20,
                    "fps": 50.0,
                }
            ]
        }
    ) + "\n"
    motion_lib, motion_source, legacy_calls = _make_manifest_fallback_fixture(
        tmp_path,
        manifest_contents,
        {"status": "complete", "output_motions": 1},
    )

    motion_lib.load_data([str(motion_source)])

    assert legacy_calls == [str(motion_source)]
    assert motion_lib._motion_data_keys.tolist() == ["legacy_motion"]


def test_malformed_semantic_manifest_falls_back_without_aborting_startup(tmp_path):
    manifest_contents = "{malformed json\n"
    motion_lib, motion_source, legacy_calls = _make_manifest_fallback_fixture(
        tmp_path,
        manifest_contents,
        {
            "status": "complete",
            "manifest_sha256": hashlib.sha256(manifest_contents.encode()).hexdigest(),
            "output_motions": 1,
        },
    )

    motion_lib.load_data([str(motion_source)])

    assert legacy_calls == [str(motion_source)]
    assert motion_lib._motion_data_keys.tolist() == ["legacy_motion"]


def test_semantic_manifest_preserves_legacy_overlapping_source_validation(tmp_path):
    dataset_root = tmp_path / "dataset"
    nested_source = dataset_root / "filtered" / "speed_1p00"
    nested_source.mkdir(parents=True)
    manifest_contents = json.dumps(
        {
            "variants": [
                {
                    "output_key": "motion_a",
                    "output_file": "filtered/speed_1p00/motion_a.pkl",
                    "frames": 20,
                    "fps": 50.0,
                }
            ]
        }
    ) + "\n"
    (dataset_root / "semantic_manifest.jsonl").write_text(manifest_contents)
    (dataset_root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": hashlib.sha256(
                    manifest_contents.encode()
                ).hexdigest(),
                "output_motions": 1,
            }
        )
    )
    motion_lib = MotionLibBase.__new__(MotionLibBase)
    motion_lib.m_cfg = {"use_semantic_manifest_metadata": True}
    motion_lib.debug = False

    sources = motion_lib._load_sources_from_semantic_manifests(
        [str(dataset_root), str(nested_source)]
    )

    assert sources == {}


def test_nonfinite_manifest_fps_falls_back_to_legacy_loader(tmp_path):
    manifest_contents = json.dumps(
        {
            "variants": [
                {
                    "output_key": "motion_a",
                    "output_file": "filtered/speed_1p00/motion_a.pkl",
                    "frames": 20,
                    "fps": float("nan"),
                }
            ]
        }
    ) + "\n"
    motion_lib, motion_source, legacy_calls = _make_manifest_fallback_fixture(
        tmp_path,
        manifest_contents,
        {
            "status": "complete",
            "manifest_sha256": hashlib.sha256(manifest_contents.encode()).hexdigest(),
            "output_motions": 1,
        },
    )

    motion_lib.load_data([str(motion_source)])

    assert legacy_calls == [str(motion_source)]
    assert motion_lib._motion_data_keys.tolist() == ["legacy_motion"]


def test_train_metrics_are_packed_into_one_collective_with_unbiased_ratio_variance():
    class Accelerator:
        def __init__(self):
            self.reduce_calls = 0

        def reduce(self, tensor, reduction="sum"):
            assert reduction == "sum"
            self.reduce_calls += 1
            # Simulate a second rank with identical statistics.
            return tensor * 2

        def gather_for_metrics(self, tensor):
            raise AssertionError("metric tensors must be packed before reduction")

    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.accelerator = Accelerator()
    trainer.compute_imgaug_bc_loss = False
    trainer.use_symmetry = False
    trainer.approxkl_stats = torch.tensor([1.0, 3.0])
    trainer.pg_clipfrac_stats = torch.tensor([2.0, 4.0])
    trainer.pg_loss_stats = torch.tensor([3.0, 5.0])
    trainer.vf_loss_stats = torch.tensor([4.0, 6.0])
    trainer.entropy_stats = torch.tensor([5.0, 7.0])
    trainer.weighted_ppo_loss_stats = torch.tensor([6.0, 8.0])
    trainer.vf_clipfrac_stats = torch.tensor([7.0, 9.0])
    trainer.ratio_stats = torch.tensor([1.0, 3.0])
    trainer.advantage_mean_stats = torch.tensor([8.0, 10.0])
    trainer.advantage_std_stats = torch.tensor([9.0, 11.0])

    metrics = trainer._get_train_metrics()

    assert trainer.accelerator.reduce_calls == 1
    assert metrics["policy/approxkl_avg"] == 2.0
    assert metrics["loss/entropy_avg"] == 6.0
    assert metrics["objective/entropy"] == 6.0
    torch.testing.assert_close(
        torch.tensor(metrics["val/ratio_var"]),
        torch.tensor([1.0, 3.0, 1.0, 3.0]).var(),
    )


def test_policy_update_builds_distribution_without_sampling_unused_actions():
    class Policy(nn.Module):
        has_aux_loss = False

        def __init__(self):
            super().__init__()
            self.updated = False
            self.action_mean = torch.tensor([[0.1, 0.2]])
            self.action_std = torch.tensor([[0.3, 0.4]])
            self.entropy = torch.tensor([0.5])

        def act(self, **kwargs):
            raise AssertionError("policy update must not sample an unused action")

        def update_distribution(self, *, is_training, **kwargs):
            assert is_training is True
            self.updated = True

        def get_actions_log_prob(self, actions):
            return actions.sum(dim=-1)

    policy = Policy()
    wrapper = PolicyAndValueWrapper(
        policy,
        value_model=None,
        sample_actions_in_policy_update=False,
    )
    actions = torch.tensor([[1.0, 2.0]])

    result = wrapper.forward_component(
        "policy", actions=actions, obs_dict={"actor_obs": torch.zeros(1, 1, 2)}
    )

    assert policy.updated is True
    torch.testing.assert_close(result["logprobs"], torch.tensor([3.0]))
    torch.testing.assert_close(result["action_mean"], policy.action_mean)


def test_running_mean_std_uses_fused_var_mean_with_legacy_moments():
    normalizer = RunningMeanStd((4,), per_channel=True)
    values = torch.randn(17, 3, 4)
    flat_values = values.reshape(-1, values.shape[-1])
    expected_mean = flat_values.mean(dim=0)
    expected_var = flat_values.var(dim=0)
    expected_running = normalizer._update_mean_var_count_from_moments(
        normalizer.running_mean,
        normalizer.running_var,
        normalizer.count,
        expected_mean,
        expected_var,
        flat_values.shape[0],
    )

    normalizer(values)

    torch.testing.assert_close(normalizer.running_mean, expected_running[0])
    torch.testing.assert_close(normalizer.running_var, expected_running[1])
    torch.testing.assert_close(normalizer.count, expected_running[2])


def test_running_mean_std_sync_uses_reduce_not_gather():
    class Accelerator:
        num_processes = 2

        def __init__(self):
            self.reduce_calls = 0

        def reduce(self, tensor, reduction="mean"):
            assert reduction == "mean"
            self.reduce_calls += 1
            return tensor + 2

        def gather(self, tensor):
            raise AssertionError("normalizer state must not be all-gathered")

    normalizer = RunningMeanStd((3,), per_channel=True)
    accelerator = Accelerator()
    normalizer.sync_across_gpus(accelerator)

    assert accelerator.reduce_calls == 1
    torch.testing.assert_close(normalizer.running_mean, torch.full((3,), 2.0))
    torch.testing.assert_close(normalizer.running_var, torch.full((3,), 3.0))
