# src/models/vit/patch_embed.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding.
    Supports variable input image resolutions (e.g. 224x224 global crops and 96x96 local crops).
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Verify only that the image dimensions are divisible by the patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input spatial dimensions ({H}, {W}) must be divisible by patch size ({self.patch_size})"

        # [B, C, H, W] -> [B, embed_dim, H/patch, W/patch] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x