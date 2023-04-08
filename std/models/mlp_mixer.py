from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # Extract patches
        # nn.Linear((patch_size ** 2) * channels, dim),  # Per-patch FC
        nn.Conv2d(3, dim, 1),
        nn.Conv2d(dim, dim, patch_size, stride=patch_size, groups=dim),
        Rearrange('b c p1 p2 -> b (p1 p2) c'),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),  # Token-mixing
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))  # Channel-mixing
        ) for _ in range(depth)],
        nn.LayerNorm(dim),  # LN
        Reduce('b n c -> b c', 'mean'),  # GAP
        nn.Linear(dim, num_classes)  # Classification head
    )


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    device = torch.device("cuda")
    image_size = 224
    mixer = MLPMixer(image_size=image_size, channels=3, patch_size=56, dim=512, depth=1, num_classes=1000).to(device)
    input_size = (1, 3, image_size, image_size)  # b,c,h,w
    summary(mixer, input_size=input_size, device=device)
    # x = torch.randn(*input_size, device=device)
    # y = mixer(x)
    # print(x.shape, y.shape)
