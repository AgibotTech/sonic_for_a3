import torch
from torch import nn

from gear_sonic.trl.modules.input_projectors import build_projector_from_config
from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule


def test_height_map_projector_preserves_leading_dimensions():
    projector, _ = build_projector_from_config(
        {
            "type": "conv2d",
            "input_shape": [1, 11, 11],
            "channels": [16, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "use_maxpool": True,
            "hidden_dims": [128],
            "output_dim": 128,
            "activation": "ReLU",
        }
    )

    output = projector(torch.randn(3, 4, 121))

    assert output.shape == (3, 4, 128)


def test_checkpoint_input_expansion_is_per_frame_and_zero_for_new_features():
    temporal_dim = 3
    old_frame_dim = 2
    new_frame_dim = 5
    loaded = torch.arange(2 * temporal_dim * old_frame_dim, dtype=torch.float32).reshape(
        2, temporal_dim * old_frame_dim
    )
    target = torch.empty(2, temporal_dim * new_frame_dim)

    expanded = UniversalTokenModule._expand_temporal_input_weight(
        loaded, target, temporal_dim
    ).reshape(2, temporal_dim, new_frame_dim)

    assert torch.equal(
        expanded[:, :, :old_frame_dim], loaded.reshape(2, temporal_dim, old_frame_dim)
    )
    assert torch.count_nonzero(expanded[:, :, old_frame_dim:]) == 0


def test_height_map_embedding_is_broadcast_across_motion_frames():
    module = UniversalTokenModule.__new__(UniversalTokenModule)
    nn.Module.__init__(module)
    projector, _ = build_projector_from_config(
        {
            "type": "conv2d",
            "input_shape": [1, 11, 11],
            "channels": [16, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "use_maxpool": True,
            "hidden_dims": [128],
            "output_dim": 128,
            "activation": "ReLU",
        }
    )
    module.encoders = nn.ModuleDict({"g1": nn.Identity()})
    module.sub_encoders = {"g1": []}
    module.encoder_input_features = {
        "g1": ["motion", "anchor", "height_map_z_flat"]
    }
    module.encoder_input_projectors = nn.ModuleDict(
        {"g1": nn.ModuleDict({"height_map_z_flat": projector})}
    )
    module.encoder_mask_features = {"g1": []}
    observations = {
        "motion": torch.randn(2, 3, 10, 58),
        "anchor": torch.randn(2, 3, 10, 6),
        "height_map_z_flat": torch.randn(2, 3, 121),
    }

    encoded_input = module._encode_single("g1", observations)

    assert encoded_input.shape == (6, 10, 192)
    assert torch.equal(encoded_input[..., :58], observations["motion"].reshape(6, 10, 58))
    assert torch.equal(encoded_input[..., 58:64], observations["anchor"].reshape(6, 10, 6))
    assert torch.equal(encoded_input[:, 0, 64:], encoded_input[:, -1, 64:])
