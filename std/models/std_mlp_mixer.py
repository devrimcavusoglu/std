from functools import partial

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

        self.spatial_dist = FFN(
            n_patches + 1, token_hidden, dropout, mixing_type="token", distillation=True
        )
        self.spatial_dist_norm = nn.LayerNorm(dim)

        self.channel_dist = FFN(
            dim + 1, channel_hidden, dropout, mixing_type="channel", distillation=True
        )
        self.channel_dist_norm = nn.LayerNorm(dim + 1)

    def forward(self, z, ts, tc):
        u = self.token_mixer(z)
        z = self.channel_mixer(u)

        zts = torch.cat((z, ts), 1)
        ts = self.spatial_dist(self.spatial_dist_norm(zts)) + ts

        ztc = torch.cat((z, tc), -1)
        tc = self.channel_dist(self.channel_dist_norm(ztc)) + tc

        return z, ts, tc


class STDMLPMixer(nn.Module):
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

        self.spatial_dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.channel_dist_token = nn.Parameter(torch.zeros(1, n_patches, 1))

        trunc_normal_(self.spatial_dist_token, std=0.2)
        trunc_normal_(self.channel_dist_token, std=0.2)

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
        self.classifier_dist = nn.Linear(dim + n_patches, num_classes)

    def forward_features(self, x):
        B = x.shape[0]  # n_batch
        x = self.patchifier(x)
        z = self.per_patch_fc(x)
        ts, tc = self.spatial_dist_token.expand(B, -1, -1), self.channel_dist_token.expand(B, -1, -1)
        for layer in self.mixer_blocks:
            z, ts, tc = layer(z, ts, tc)
        z = self.ln(z)
        z = self.gap(z)
        return z, ts, tc

    def forward(self, x):
        B = x.shape[0]  # n_batch
        z, ts, tc = self.forward_features(x)
        outputs = self.classifier(z)
        t_dist = torch.cat((ts.view(B, -1), tc.view(B, -1)), dim=-1)
        outputs_dist = self.classifier_dist(t_dist)
        if self.training:
            return outputs, outputs_dist
        return (outputs + outputs_dist) / 2


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    from std.mine import build_mine, mine_regularization

    torch.manual_seed(3)

    device = torch.device("cuda")
    image_size = 224
    mixer = STDMLPMixer(
        image_size=image_size, channels=3, patch_size=16, dim=512, depth=2, num_classes=10
    ).to(device)
    input_size = (2, 3, image_size, image_size)  # b,c,h,w
    summary(mixer, input_size=input_size, device=device)
    torch.manual_seed(3)
    x = torch.randn(*input_size, device=device)
    y, y_kd = mixer(x)
    print(y)
    print(x.shape, y.shape, y_kd.shape)

    dim_spatial = 512
    dim_channel = 196

    model_optimizer, mine_network, mine_optimizer, objective = build_mine(
        mixer, dim_spatial, dim_channel, device
    )
    mine_regularization(mixer, mine_network, model_optimizer, mine_optimizer, objective, x)
