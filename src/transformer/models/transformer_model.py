"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2024
File: transformer_model.py

Description:
    This file defines the Transformer model used for image classification
    from sequences of flattened image patches.
"""

import torch
from torch import Tensor, nn

from .encoder_block import TransformerEncoderBlock


class VisionTransformer(nn.Module):
    """Transformer model for classification from patch sequences."""

    def __init__(
        self,
        patch_dim: int = 16 * 16 * 3,
        seq_len: int = 14 * 14,
        embed_dim: int = 384,
        num_classes: int = 20,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.05,
        use_patch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.patch_dim = patch_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_norm = nn.LayerNorm(patch_dim) if use_patch_norm else nn.Identity()
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )

        self.norm_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits from a batch of patch sequences."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, L, D), but got {tuple(x.shape)}."
            )

        batch_size, seq_len, patch_dim = x.shape

        if patch_dim != self.patch_dim:
            raise ValueError(
                f"Expected patch dimension {self.patch_dim}, but got {patch_dim}."
            )
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, but got {seq_len}."
            )

        x = self.patch_norm(x)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        x = self.blocks(x)
        x = self.norm_final(x)

        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits
