import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Shared patch embedding for Camera and LiDAR.
    Guarantees identical spatial tokenization.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        patch_size: int = 16,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor [B, C, H, W]

        Returns:
            tokens: [B, N, C]
            H_p: patch height
            W_p: patch width
        """

        B, C, H, W = x.shape

        # Enforce divisibility (important for fusion)
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {(H, W)} not divisible by patch size {self.patch_size}"

        x = self.proj(x)          # [B, C, H/P, W/P]
        H_p, W_p = x.shape[2], x.shape[3]

        # Flatten to tokens
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)

        return x, H_p, W_p
