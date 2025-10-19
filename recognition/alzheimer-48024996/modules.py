"""
Originally written by Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou;
and modified for Alzheimer's Disease recognition.

https://github.com/raoyongming/GFNet/
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    Standard MLP (Feed-Forward Network) block used in Transformer architectures.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    """
    The core GFNet block. Applies a learnable filter in the frequency domain.
    """

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        # 2D Fourier Transform
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        # Multiply by learnable complex weights
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        # Inverse 2D Fourier Transform
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        x = x.reshape(B, N, C)
        return x


class BlockLayerScale(nn.Module):
    """
    A single GFNet block with LayerScale applied.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        h=14,
        w=8,
        init_values=1e-5,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x))))
        )
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using a single convolutional layer.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DownLayer(nn.Module):
    """
    Downsampling Layer used in GFNetPyramid to reduce spatial dimensions
    and increase channel dimensions between stages.
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.proj.out_channels)
        return x


class GFNetPyramid(nn.Module):
    """
    The GFNetPyramid model for Alzheimer's Disease classification.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        num_classes=2,
        embed_dim=[64, 128, 256, 512],
        depth=[2, 2, 10, 4],
        mlp_ratio=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        init_values=0.001,
        dropcls=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Initial Patch Embedding
        self.patch_embed = nn.ModuleList()
        first_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim[0]
        )
        self.patch_embed.append(first_patch_embed)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, first_patch_embed.num_patches, embed_dim[0])
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Calculate spatial sizes for each stage of the pyramid
        sizes = [img_size // patch_size]
        for _ in range(3):
            sizes.append(sizes[-1] // 2)

        # Create downsampling layers for subsequent stages
        for i in range(3):
            down_layer = DownLayer(sizes[i], embed_dim[i], embed_dim[i + 1])
            self.patch_embed.append(down_layer)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.blocks = nn.ModuleList()
        cur = 0

        # Build the blocks for each stage
        for i in range(4):
            h = sizes[i]
            w = h // 2 + 1  # Width for rfft2

            stage_blocks = nn.Sequential(
                *[
                    BlockLayerScale(
                        dim=embed_dim[i],
                        mlp_ratio=mlp_ratio[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        h=h,
                        w=w,
                        init_values=init_values,
                    )
                    for j in range(depth[i])
                ]
            )
            self.blocks.append(stage_blocks)
            cur += depth[i]

        # Classifier head
        self.norm = norm_layer(embed_dim[-1])
        self.head = nn.Linear(self.num_features, num_classes)
        self.final_dropout = nn.Dropout(p=dropcls) if dropcls > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(4):
            x = self.patch_embed[i](x)
            if i == 0:
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = self.blocks[i](x)

        return self.norm(x).mean(1)  # Global Average Pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x
