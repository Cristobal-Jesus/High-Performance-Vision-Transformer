"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2024
File: encoder_block.py

Description:
    This file defines a Transformer encoder block with self-attention,
    feed-forward layers, residual connections, and stochastic depth.
"""

from torch import Tensor, nn

from .drop_path import DropPath
from .attention import Attention
from .mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with pre-normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.05,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.path2 = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention and feed-forward layers with residual connections."""
        x = x + self.path1(self.dropout1(self.attention(self.norm1(x))))
        x = x + self.path2(self.mlp(self.norm2(x)))
        return x
