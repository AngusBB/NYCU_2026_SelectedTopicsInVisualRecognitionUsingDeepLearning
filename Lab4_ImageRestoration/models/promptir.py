"""PromptIR model adapted for the HW4 rain/snow restoration task.

This implementation is adapted from the official PromptIR code:
https://github.com/va1shn9v/PromptIR

Paper:
Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan,
"PromptIR: Prompting for All-in-One Blind Image Restoration", NeurIPS 2023.

Use is intended for this non-commercial academic lab. See LICENSE_PROMPTIR.md.
"""

from __future__ import annotations

import numbers
from typing import Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    """LayerNorm over channels without a bias term."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("BiasFreeLayerNorm expects a 1D shape.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """LayerNorm over channels with scale and bias."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("WithBiasLayerNorm expects a 1D shape.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """LayerNorm wrapper for BCHW tensors."""

    def __init__(self, dim: int, norm_type: str) -> None:
        super().__init__()
        if norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError("norm_type must be 'BiasFree' or 'WithBias'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), height, width)


class FeedForward(nn.Module):
    """Gated depth-wise convolution feed-forward network."""

    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float,
        bias: bool,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim,
            hidden_features * 2,
            kernel_size=1,
            bias=bias,
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    """Multi-DConv head transposed self-attention."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(out)


class Downsample(nn.Module):
    """Spatial downsample by 2 with pixel unshuffle."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Spatial upsample by 2 with pixel shuffle."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TransformerBlock(nn.Module):
    """Restormer transformer block used by PromptIR."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping 3x3 convolution patch embedding."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PromptGenBlock(nn.Module):
    """Generate a weighted prompt tensor from decoder features."""

    def __init__(
        self,
        prompt_dim: int,
        prompt_len: int,
        prompt_size: int,
        lin_dim: int,
    ) -> None:
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim,
            prompt_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = (
            prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.prompt_param.expand(batch, -1, -1, -1, -1)
        )
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(
            prompt,
            (height, width),
            mode="bilinear",
            align_corners=False,
        )
        return self.conv3x3(prompt)


class PromptIR(nn.Module):
    """PromptIR for blind rain/snow image restoration."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        decoder: bool = True,
    ) -> None:
        super().__init__()
        if dim != 48 and decoder:
            raise ValueError(
                "The PromptIR decoder prompt dimensions are defined for dim=48."
            )
        self.model_args = {
            "inp_channels": inp_channels,
            "out_channels": out_channels,
            "dim": dim,
            "num_blocks": list(num_blocks),
            "num_refinement_blocks": num_refinement_blocks,
            "heads": list(heads),
            "ffn_expansion_factor": ffn_expansion_factor,
            "bias": bias,
            "layer_norm_type": layer_norm_type,
            "decoder": decoder,
        }

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64,
                prompt_len=5,
                prompt_size=64,
                lin_dim=96,
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=128,
                prompt_len=5,
                prompt_size=32,
                lin_dim=192,
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=320,
                prompt_len=5,
                prompt_size=16,
                lin_dim=384,
            )

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)

        level2_dim = int(dim * 2)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level2_dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(level2_dim)

        level3_dim = int(dim * 4)
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level3_dim,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(level3_dim)

        latent_dim = int(dim * 8)
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=latent_dim,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.noise_level3 = TransformerBlock(
            dim=level3_dim + 512,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            level3_dim + 512,
            level3_dim,
            kernel_size=1,
            bias=bias,
        )
        self.up4_3 = Upsample(level3_dim)
        self.reduce_chan_level3 = nn.Conv2d(
            level2_dim + level3_dim,
            level3_dim,
            kernel_size=1,
            bias=bias,
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level3_dim,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.noise_level2 = TransformerBlock(
            dim=level2_dim + 224,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            level2_dim + 224,
            level3_dim,
            kernel_size=1,
            bias=bias,
        )
        self.up3_2 = Upsample(level3_dim)
        self.reduce_chan_level2 = nn.Conv2d(
            level3_dim,
            level2_dim,
            kernel_size=1,
            bias=bias,
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level2_dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.noise_level1 = TransformerBlock(
            dim=level2_dim + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            level2_dim + 64,
            level2_dim,
            kernel_size=1,
            bias=bias,
        )
        self.up2_1 = Upsample(level2_dim)
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level2_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level2_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        self.output = nn.Conv2d(
            level2_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], dim=1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], dim=1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], dim=1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        return self.output(out_dec_level1) + inp_img
