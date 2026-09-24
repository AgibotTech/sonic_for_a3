"""Small input adapters used before universal-token motion encoders."""

from __future__ import annotations

from torch import nn


class Conv2dProjector(nn.Module):
    """Project a flattened spatial observation to a compact feature vector."""

    def __init__(self, conv_layers, mlp_head, input_shape):
        super().__init__()
        self.conv_layers = conv_layers
        self.mlp_head = mlp_head
        self.input_shape = tuple(input_shape)

    def forward(self, x):
        channels, height, width = self.input_shape
        flat_dim = channels * height * width
        leading_shape = x.shape[:-1] if x.shape[-1] == flat_dim else x.shape[:-3]
        x = x.reshape(-1, channels, height, width)
        x = self.conv_layers(x)
        x = self.mlp_head(x.reshape(x.shape[0], -1))
        return x.reshape(*leading_shape, -1)


def _build_conv2d_projector(
    input_shape,
    channels,
    kernel_sizes,
    strides,
    paddings,
    use_maxpool,
    hidden_dims,
    output_dim,
    activation_cls,
):
    in_channels, height, width = input_shape
    conv_layers = []
    for out_channels, kernel, stride, padding in zip(
        channels, kernel_sizes, strides, paddings, strict=True
    ):
        conv_layers.extend(
            [nn.Conv2d(in_channels, out_channels, kernel, stride, padding), activation_cls()]
        )
        if use_maxpool:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        height = (height + 2 * padding - kernel) // stride + 1
        width = (width + 2 * padding - kernel) // stride + 1
        if use_maxpool:
            height //= 2
            width //= 2
        in_channels = out_channels

    mlp_layers = []
    input_dim = in_channels * height * width
    for hidden_dim in hidden_dims:
        mlp_layers.extend([nn.Linear(input_dim, hidden_dim), activation_cls()])
        input_dim = hidden_dim
    mlp_layers.append(nn.Linear(input_dim, output_dim))
    return Conv2dProjector(
        conv_layers=nn.Sequential(*conv_layers),
        mlp_head=nn.Sequential(*mlp_layers),
        input_shape=input_shape,
    )


def build_projector_from_config(projector_config, feat_dim=None):
    """Build an input projector from a Hydra/OmegaConf mapping."""
    projector_type = projector_config.get("type", "mlp")
    output_dim = projector_config["output_dim"]
    activation_cls = getattr(nn, projector_config.get("activation", "SiLU"))

    if projector_type == "conv2d":
        channels = projector_config.get("channels", [16, 32])
        kernel_sizes = projector_config.get("kernel_sizes", [3, 3])
        projector = _build_conv2d_projector(
            input_shape=projector_config["input_shape"],
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=projector_config.get("strides", [1] * len(channels)),
            paddings=projector_config.get("paddings", [kernel // 2 for kernel in kernel_sizes]),
            use_maxpool=projector_config.get("use_maxpool", True),
            hidden_dims=projector_config.get("hidden_dims", [256]),
            output_dim=output_dim,
            activation_cls=activation_cls,
        )
        description = (
            f"Conv2d: input_shape={projector_config['input_shape']} "
            f"channels={channels} -> {output_dim}"
        )
        return projector, description

    if projector_type != "mlp":
        raise ValueError(f"Unsupported input projector type: {projector_type}")
    if feat_dim is None:
        raise ValueError("feat_dim is required for an MLP input projector")

    layers = []
    input_dim = feat_dim
    hidden_dims = projector_config.get("hidden_dims", [256])
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(input_dim, hidden_dim), activation_cls()])
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers), f"MLP: {feat_dim} -> {hidden_dims} -> {output_dim}"
