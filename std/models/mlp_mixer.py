from functools import partial
from typing import Optional

import torch
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_
from torch import nn

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim: int,
        dropout: float = 0.0,
        mixing_type: str = "channel",
        distillation: bool = False,
    ):
        super().__init__()
        layer_dense = nn.Linear if mixing_type == "channel" else partial(nn.Conv1d, kernel_size=1)
        self.dense1 = layer_dense(dim, hidden_dim)
        self.dense2 = layer_dense(hidden_dim, dim) if not distillation else layer_dense(hidden_dim, 1)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self, dim, n_patches, spatial_scale: float = 0.5, channel_scale: float = 4, dropout: float = 0.0
    ):
        super().__init__()
        token_hidden = int(dim * spatial_scale)
        channel_hidden = int(dim * channel_scale)
        self.token_mixer = PreNormResidual(
            dim, FFN(n_patches, token_hidden, dropout, mixing_type="token")
        )
        self.channel_mixer = PreNormResidual(
            dim, FFN(dim, channel_hidden, dropout, mixing_type="channel")
        )

    def forward(self, z: torch.Tensor):
        u = self.token_mixer(z)
        z = self.channel_mixer(u)

        return z


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size,
        channels,
        patch_size,
        dim,
        depth,
        num_classes,
        f_spatial_expansion: float = 0.5,
        f_channel_expansion: float = 4,
        dropout=0.0,
    ):
        super().__init__()
        h, w = pair(image_size)
        assert (image_size % patch_size) == 0 and (w % h) == 0, "image must be divisible by patch size"
        n_patches = (h // patch_size) * (w // patch_size)

        self.depth = depth

        self.patchifier = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )  # Extract patches
        self.per_patch_fc = nn.Linear((patch_size ** 2) * channels, dim)
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(dim, n_patches, f_spatial_expansion, f_channel_expansion, dropout)
                for _ in range(depth)
            ]
        )
        self.ln = nn.LayerNorm(dim)
        self.gap = Reduce("b n c -> b c", "mean")
        self.classifier = nn.Linear(dim, num_classes)

    def forward_features(self, x):
        B = x.shape[0]  # n_batch
        x = self.patchifier(x)
        z = self.per_patch_fc(x)
        for i, layer in enumerate(self.mixer_blocks):
            z = layer(z)
        z = self.ln(z)
        z = self.gap(z)
        return z

    def forward(self, x):
        B = x.shape[0]  # n_batch
        z = self.forward_features(x)
        outputs = self.classifier(z)
        return outputs


if __name__ == "__main__":
    import torch
    from timm.models.mlp_mixer import MlpMixer
    from torchinfo import summary

    from std.mine import build_mine, mine_regularization

    torch.manual_seed(3)

    device = torch.device("cuda")
    image_size = 32
    mixer = MLPMixer(
        image_size=image_size, channels=3, patch_size=4, dim=512, depth=8, num_classes=100
    ).to(device)
    input_size = (2, 3, image_size, image_size)  # b,c,h,w
    summary(mixer, input_size=input_size, device=device)
    torch.manual_seed(3)
    x = torch.randn(*input_size, device=device)
    y, y_kd = mixer(x)
    print(y)
    print(x.shape, y.shape, y_kd.shape)

    # dim_spatial = 512
    # dim_channel = 196
    #
    # model_optimizer, mine_network, mine_optimizer, objective = build_mine(
    #     mixer, dim_spatial, dim_channel, device
    # )
    # mine_regularization(mixer, mine_network, model_optimizer, mine_optimizer, objective, x)
