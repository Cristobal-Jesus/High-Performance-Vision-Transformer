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
    Transformer architecture of the project. Uses PyTorch's fused
    scaled_dot_product_attention, which selects the fastest available
    backend at runtime (math kernel on V100, Flash Attention 2 on
    H100/H200), and falls back to the manual implementation only when
    SDPA is unavailable (PyTorch < 2.0).

References:
    - https://www.ibm.com/think/topics/attention-mechanism
    - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


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
        self.dropout_p = dropout

        # Fused QKV projection saves one GEMM and improves Tensor Core utilisation.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
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

        # Single fused projection -> (B, L, 3*D), reshape to (3, B, H, L, Dh)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if _HAS_SDPA:
            # Picks Flash Attention / memory-efficient / math automatically.
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            output = self._manual_attention(q, k, v)

        # Merge heads and project
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, embed_dim)
        output = self.proj(output)
        output = self.proj_dropout(output)
        return output

    def _manual_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Fallback path for PyTorch < 2.0 (no SDPA available)."""
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        return attention_weights @ v
