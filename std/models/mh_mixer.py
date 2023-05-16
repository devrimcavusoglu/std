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
        h = self.fn(self.norm(x))
        return h + x


class MHInner(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, halve = True):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        out_dim = dim//2 if halve else dim
        self.dropout = nn.Dropout(dropout)
        self.dense1 = dense(dim, inner_dim)
        self.dense2 = dense(inner_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.gelu(self.dense1(x)))
        x = self.dropout(self.dense2(x))
        return x


class MHMixer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth=1, num_classes=1000, expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
        super().__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        self.depth = depth

        self.tokenizer = nn.Sequential(
                nn.Conv2d(3, dim, 1),
                nn.Conv2d(dim, dim, patch_size, stride=patch_size, groups=dim),
                Rearrange('b c p1 p2 -> b (p1 p2) c'),
        )

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.mh_inner_token = PreNormResidual(dim, MHInner(num_patches, expansion_factor, dropout, dense=chan_first, halve=False))
        self.mh_inner_channel = PreNormResidual(dim, MHInner(dim, expansion_factor_token, dropout, dense=chan_last, halve=False))
        self.mh_fuse = PreNormResidual(dim, MHInner(dim, expansion_factor_token, dropout, dense=chan_last, halve=False))
        self.output = nn.Sequential(
                nn.LayerNorm(dim),
                Reduce('b n c -> b c', 'mean')
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.tokenizer(x)
        for _ in range(self.depth):
            h1 = self.mh_inner_token(x)
            h2 = self.mh_inner_channel(x)
            h = torch.hstack((h1, h2))
            x = self.mh_fuse(h)
        x = self.output(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    device = torch.device("cuda")
    image_size = 448
    mixer = MHMixer(image_size=image_size, patch_size=16, dim=512, depth=1, num_classes=1000)
    mixer = mixer.to(device)
    input_size = (1, 3, image_size, image_size)  # b,c,h,w
    summary(mixer, input_size=input_size, device=device)
    x = torch.randn(*input_size, device=device)
    y = mixer(x)
    print(x.shape, y.shape)
