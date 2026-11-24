import torch
import torch.nn as nn


class CLFTFusionBlock(nn.Module):
    """
    CLFT-style token fusion:
    - Camera tokens query
    - LiDAR tokens provide key/value
    - Residual + normalization
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, cam_tokens, lidar_tokens):
        """
        Args:
            cam_tokens:   [B, N, C]
            lidar_tokens: [B, N, C]

        Returns:
            fused_tokens: [B, N, C]
        """

        # --- Cross-attention ---
        # Camera queries LiDAR
        cam_norm = self.norm1(cam_tokens)
        lid_norm = self.norm1(lidar_tokens)

        attn_out, _ = self.cross_attn(
            query=cam_norm,
            key=lid_norm,
            value=lid_norm,
            need_weights=False,
        )

        # Residual connection
        x = cam_tokens + attn_out

        # --- Feed-forward refinement ---
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x
