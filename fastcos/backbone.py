import torch
import torch.nn as nn
import torch.nn.functional as F


class RepSpatialMixing(nn.Module):
    """
    Reparameterized spatial mixing:
    Depthwise convolution applied after reshaping tokens to grid.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dim,
            bias=True,
        )
        self.pwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, tokens, H, W):
        """
        tokens: [B, N, C]
        """
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.pwconv(self.dwconv(x))
        x = x.flatten(2).transpose(1, 2)
        return x


class FastCOSBlock(nn.Module):
    """
    One Fast-COS block:
    - Global self-attention
    - Local spatial mixing
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.spatial = RepSpatialMixing(embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, tokens, H, W):
        """
        tokens: [B, N, C]
        """

        # --- Global self-attention ---
        x = self.norm1(tokens)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        tokens = tokens + attn_out

        # --- Spatial mixing ---
        x = self.norm2(tokens)
        spatial_out = self.spatial(x, H, W)
        tokens = tokens + spatial_out

        # --- Feed-forward ---
        x = self.norm3(tokens)
        tokens = tokens + self.mlp(x)

        return tokens


class FastCOSBackbone(nn.Module):
    """
    Stack of Fast-COS blocks.
    Operates on fused tokens.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                FastCOSBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, tokens, H, W):
        """
        Args:
            tokens: [B, N, C]
            H, W: patch grid size

        Returns:
            tokens: [B, N, C]
        """
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        return tokens
