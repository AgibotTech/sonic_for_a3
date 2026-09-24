from __future__ import annotations

from copy import deepcopy

from omegaconf import OmegaConf
import pytest
import torch

from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule


def _base_module(*, temporal=True):
    return {
        "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
        "input_dim": None,
        "output_dim": None,
        "num_input_temporal_dims": 1 if temporal else None,
        "num_output_temporal_dims": 1 if temporal else None,
        "module_config_dict": {
            "type": "MLP",
            "layer_config": {
                "type": "MLP",
                "hidden_dims": [8],
                "activation": "SiLU",
            },
        },
    }


def _make_module(
    *,
    rollout_decoders=None,
    cache_encoded_cpu=True,
    active_decoders=("g1_dyn", "g1_kin"),
):
    env_config = OmegaConf.create(
        {
            "obs": {
                "group_obs_dims": {
                    "tokenizer": {
                        "encoder_index": [1],
                        "motion": [1, 4],
                        "recon": [1, 4],
                    }
                },
                "group_obs_names": {
                    "tokenizer": ["encoder_index", "motion", "recon"]
                },
            },
            "robot": {
                "actions_dim": 3,
                "algo_obs_dim_dict": {"actor_obs": 5, "tokenizer": 9},
            },
        }
    )
    encoder = _base_module()
    decoders = {
        "g1_dyn": {
            "inputs": ["token_flattened", "proprioception"],
            "outputs": ["action"],
            "has_temporal_dim": False,
            "params": _base_module(temporal=False),
        },
        "g1_kin": {
            "inputs": ["token"],
            "outputs": ["recon"],
            "has_temporal_dim": True,
            "params": _base_module(),
        },
    }
    return UniversalTokenModule(
        env_config=env_config,
        algo_config=OmegaConf.create({}),
        proprioception_features=["actor_obs"],
        num_fsq_levels=4,
        max_num_tokens=1,
        num_future_frames=1,
        quantizer=None,
        encoders=OmegaConf.create(
            {"g1": {"inputs": ["motion"], "params": deepcopy(encoder)}}
        ),
        decoders=OmegaConf.create(decoders),
        encoder_sample_probs=OmegaConf.create({"g1": 1.0}),
        active_encoders=["g1"],
        active_decoders=list(active_decoders),
        rollout_decoders=rollout_decoders,
        cache_encoded_cpu=cache_encoded_cpu,
    )


def _example_input():
    batch_size, sequence_length = 7, 3
    encoder_index = torch.ones(batch_size, sequence_length, 1)
    motion = torch.randn(batch_size, sequence_length, 1, 4)
    recon = torch.randn(batch_size, sequence_length, 1, 4)
    return {
        "actor_obs": torch.randn(batch_size, sequence_length, 5),
        "tokenizer": torch.cat(
            [encoder_index, motion.flatten(-2), recon.flatten(-2)], dim=-1
        ),
    }


def test_action_only_forward_skips_kinematic_decoder_with_exact_action_parity():
    torch.manual_seed(11)
    reference = _make_module()
    candidate = _make_module(
        rollout_decoders=["g1_dyn"], cache_encoded_cpu=False
    )
    candidate.load_state_dict(reference.state_dict())
    input_data = _example_input()
    calls = {"reference_kin": 0, "candidate_kin": 0}

    def count(name):
        def hook(*_):
            calls[name] += 1

        return hook

    reference.decoders["g1_kin"].register_forward_hook(count("reference_kin"))
    candidate.decoders["g1_kin"].register_forward_hook(count("candidate_kin"))

    with torch.no_grad():
        reference_action = reference(input_data)
        candidate_action = candidate(input_data)

    assert torch.equal(candidate_action, reference_action)
    assert calls == {"reference_kin": 1, "candidate_kin": 0}
    assert candidate._last_encoded_tokens is None
    assert candidate._last_encoded_latents is None
    assert candidate._last_full_latent_flat is not None
    assert reference._last_encoded_tokens["g1"].device.type == "cpu"


def test_rich_and_auxiliary_forwards_keep_all_active_decoders():
    module = _make_module(rollout_decoders=["g1_dyn"], cache_encoded_cpu=False)
    input_data = _example_input()

    rich = module(input_data, return_dict=True)
    auxiliary = module(input_data, compute_aux_loss=True)

    assert list(rich["decoded_outputs"]) == ["g1_dyn", "g1_kin"]
    assert list(auxiliary["decoded_outputs"]) == ["g1_dyn", "g1_kin"]


@pytest.mark.parametrize("rollout_decoders", [["missing"], ["g1_kin"]])
def test_invalid_rollout_decoder_sets_fail_fast(rollout_decoders):
    with pytest.raises(ValueError, match="rollout_decoders"):
        _make_module(rollout_decoders=rollout_decoders)


def test_unknown_active_decoder_fails_fast():
    with pytest.raises(ValueError, match="active_decoders contains unknown"):
        _make_module(active_decoders=("g1_dyn", "typo_decoder"))


def _make_coactivated_module(*, rollout_encoder_exclusions=None):
    env_config = OmegaConf.create(
        {
            "obs": {
                "group_obs_dims": {
                    "tokenizer": {
                        "encoder_index": [2],
                        "g1_motion": [1, 4],
                        "a3_motion": [1, 4],
                    }
                },
                "group_obs_names": {
                    "tokenizer": ["encoder_index", "g1_motion", "a3_motion"]
                },
            },
            "robot": {
                "actions_dim": 3,
                "algo_obs_dim_dict": {"actor_obs": 5, "tokenizer": 10},
            },
        }
    )
    encoder = _base_module()
    return UniversalTokenModule(
        env_config=env_config,
        algo_config=OmegaConf.create({}),
        proprioception_features=["actor_obs"],
        num_fsq_levels=4,
        max_num_tokens=1,
        num_future_frames=1,
        quantizer=None,
        encoders=OmegaConf.create(
            {
                "g1": {"inputs": ["g1_motion"], "params": deepcopy(encoder)},
                "a3_fast": {
                    "inputs": ["a3_motion"],
                    "params": deepcopy(encoder),
                },
            }
        ),
        decoders=OmegaConf.create(
            {
                "g1_dyn": {
                    "inputs": ["token_flattened", "proprioception"],
                    "outputs": ["action"],
                    "has_temporal_dim": False,
                    "params": _base_module(temporal=False),
                }
            }
        ),
        encoder_sample_probs=OmegaConf.create({"g1": 1.0, "a3_fast": 1.0}),
        active_encoders=["g1", "a3_fast"],
        active_decoders=["g1_dyn"],
        rollout_decoders=["g1_dyn"],
        rollout_encoder_exclusions=rollout_encoder_exclusions,
        cache_encoded_cpu=False,
    )


def _coactivated_input():
    batch_size = 6
    encoder_index = torch.tensor(
        [[[1.0, 0.0]], [[1.0, 1.0]]] * (batch_size // 2)
    )
    return {
        "actor_obs": torch.randn(batch_size, 1, 5),
        "tokenizer": torch.cat(
            [
                encoder_index,
                torch.randn(batch_size, 1, 4),
                torch.randn(batch_size, 1, 4),
            ],
            dim=-1,
        ),
    }


def test_action_rollout_skips_coactivated_rows_that_are_overwritten():
    torch.manual_seed(23)
    reference = _make_coactivated_module()
    candidate = _make_coactivated_module(
        rollout_encoder_exclusions={"g1": ["a3_fast"]}
    )
    candidate.load_state_dict(reference.state_dict())
    input_data = _coactivated_input()
    batch_sizes = {"reference_g1": [], "candidate_g1": []}

    reference.encoders["g1"].register_forward_pre_hook(
        lambda _module, args: batch_sizes["reference_g1"].append(args[0].shape[0])
    )
    candidate.encoders["g1"].register_forward_pre_hook(
        lambda _module, args: batch_sizes["candidate_g1"].append(args[0].shape[0])
    )

    with torch.no_grad():
        reference_action = reference(input_data)
        candidate_action = candidate(input_data)

    torch.testing.assert_close(candidate_action, reference_action)
    assert batch_sizes == {"reference_g1": [6], "candidate_g1": [3]}

    rich = candidate(input_data, return_dict=True)
    auxiliary = candidate(input_data, compute_aux_loss=True)
    assert rich["encoder_masks"]["g1"].sum() == 6
    assert auxiliary["encoder_masks"]["g1"].sum() == 6
    assert batch_sizes["candidate_g1"] == [3, 6, 6]


def test_invalid_rollout_encoder_exclusion_fails_fast():
    with pytest.raises(ValueError, match="rollout_encoder_exclusions"):
        _make_coactivated_module(
            rollout_encoder_exclusions={"g1": ["missing_encoder"]}
        )
