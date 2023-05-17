import torch
from timm.models.layers import trunc_normal_
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, distillation: str = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.distillation = distillation

    def forward(self, x, t):
        if self.distillation == "spatial":
            zt = torch.cat([x, t], 1)
            zt = self.fn(self.norm(zt)) + t
            return zt[:, :-1, :], zt[:, -2:-1, :]  # (z, ts)

        zt = torch.cat([x, t], -1)
        zt = self.fn(self.norm(zt)) + t
        return zt[:, :, :-1], zt[:, :, -2:-1]  # (z, ts)


class FFN(nn.Module):
    def __init__(self, dim, inner_dim_factor: float = 4, dropout: float = 0.0, mixing_type: str = "channel"):
        super().__init__()
        inner_dim = int(dim * inner_dim_factor)
        layer_dense = nn.Linear if mixing_type == "channel" else partial(nn.Conv1d, kernel_size=1)
        self.dense1 = layer_dense(dim, inner_dim)
        self.dense2 = layer_dense(inner_dim, dim)
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
    def __init__(self, dim, n_patches, f_spatial_expansion: float = 4, f_channel_expansion: float = 4, dropout: float = 0.0):
        super().__init__()
        self.token_mixer = PreNormResidual(dim, FFN(n_patches+1, f_spatial_expansion, dropout, mixing_type="spatial"), distillation="spatial")
        self.channel_mixer = PreNormResidual(dim+1, FFN(dim+1, f_channel_expansion, dropout, mixing_type="channel"), distillation="channel")

    def forward(self, x, ts, tc):
        x, ts = self.token_mixer(x, ts)
        x, tc = self.channel_mixer(x, tc)
        return x, ts, tc


class STDMLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, f_spatial_expansion: float = 4, f_channel_expansion: float = 0.5, dropout = 0.):
        super().__init__()
        h, w = pair(image_size)
        assert (image_size % patch_size) == 0 and (w % h) == 0, "image must be divisible by patch size"
        self.n_patches = (h // patch_size) * (w // patch_size)

        self.spatial_dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.channel_dist_token = nn.Parameter(torch.zeros(1, self.n_patches, 1))

        trunc_normal_(self.spatial_dist_token, std=0.2)
        trunc_normal_(self.channel_dist_token, std=0.2)

        self.patchifier = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)  # Extract patches
        self.per_patch_fc = nn.Linear((patch_size ** 2) * channels, dim)
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, self.n_patches, f_spatial_expansion, f_channel_expansion, dropout) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.gap = Reduce('b n c -> b c', 'mean')
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patchifier(x)
        x = self.per_patch_fc(x)
        ts, tc = self.spatial_dist_token, self.channel_dist_token
        for layer in self.mixer_blocks:
            x, ts, tc = layer(x, ts, tc)
        x = self.ln(x)
        x = self.gap(x)
        return self.classifier(x)


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    torch.manual_seed(3)

    device = torch.device("cuda")
    image_size = 224
    mixer = STDMLPMixer(image_size=image_size, channels=3, patch_size=16, dim=512, depth=2).to(device)
    input_size = (1, 3, image_size, image_size)  # b,c,h,w
    # summary(mixer, input_size=input_size, device=device)
    torch.manual_seed(3)
    x = torch.randn(*input_size, device=device)
    y = mixer(x)
    print(y)
    print(x.shape, y.shape)
