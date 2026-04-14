"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: Image Classification Using CNNs and Transformers
Author: Cristobal Jesus Sarmiento Rodriguez
Supervisor: Supervisor Name
Date: 2026-02-03
File: attention.py

Description:
    This file defines the multi-head self-attention module used in the
    Transformer architecture of the project.

References:
    - https://www.ibm.com/think/topics/attention-mechanism
"""

import math

from torch import Tensor, nn


class Attention(nn.Module):
    """Multi-head scaled dot-product self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention to the input tensor."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, L, D), but got {tuple(x.shape)}."
            )

        batch_size, seq_len, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, but got {embed_dim}."
            )

        # Compute query, key, and value projections.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split the embedding dimension across multiple attention heads.
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention.
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Weight the values using the attention distribution.
        output = attention_weights @ v

        # Merge all heads back into the original embedding dimension.
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, embed_dim)

        output = self.proj(output)
        output = self.proj_dropout(output)

        return output
