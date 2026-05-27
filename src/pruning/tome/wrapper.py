"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/tome/wrapper.py

Description:
    Wrapper que envuelve un TransformerEncoderBlock con Token Merging.

    En cada bloque se ejecuta un paso de fusión de tokens ANTES de pasar
    el tensor a través de la atención y la MLP.  Los tokens fusionados
    permanecen fusionados para el resto de la red, reduciendo el cómputo
    en capas posteriores.

    El número efectivo de tokens disminuye en r por bloque, así que en
    una red de depth=12 con r=8 se eliminan hasta 12×8 = 96 tokens del
    total de 196 parches (≈49% de reducción de longitud de secuencia).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from transformer.models.encoder_block import TransformerEncoderBlock

from .merger import bipartite_soft_matching


class ToMeEncoderBlock(nn.Module):
    """Envuelve un TransformerEncoderBlock aplicando Token Merging.

    No modifica los pesos del bloque original; solo intercepta el
    forward para fusionar tokens antes de la autoatención.

    Args:
        block: Bloque encoder a envolver.
        r:     Número de tokens a fusionar en este bloque.
    """

    def __init__(self, block: TransformerEncoderBlock, r: int) -> None:
        super().__init__()
        self.block     = block
        self.r         = r
        self._num_heads = block.attention.num_heads
        self._head_dim  = block.attention.head_dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward con fusión de tokens.

        1. Extrae las claves K del bloque de atención para calcular la
           métrica de similitud (sin gradientes: solo se usa para decidir
           qué tokens fusionar).
        2. Calcula el esquema BSM y fusiona los ``r`` pares más parecidos.
        3. Ejecuta el forward completo del bloque sobre los tokens fusionados.

        Args:
            x: ``(B, N, C)`` — tokens de entrada.

        Returns:
            ``(B, N-r, C)`` — tokens de salida (secuencia más corta).
        """
        if self.r <= 0:
            return self.block(x)

        B, N, C = x.shape

        # --- 1. Extraer claves K para la métrica de similitud ---
        # Se hace sin gradientes porque solo nos importa la decisión de fusión.
        with torch.no_grad():
            normed = self.block.norm1(x)                         # (B, N, C)
            qkv = self.block.attention.qkv(normed)               # (B, N, 3C)
            qkv = qkv.reshape(B, N, 3, self._num_heads, self._head_dim)
            k = qkv[:, :, 1, :, :].mean(dim=2)                  # (B, N, head_dim)

        # --- 2. Esquema de fusión (BSM) ---
        merge = bipartite_soft_matching(k, r=self.r)

        # --- 3. Fusionar tokens de entrada ---
        x = merge(x)   # (B, N-r, C)

        # --- 4. Forward del bloque sobre tokens fusionados ---
        x = x + self.block.path1(self.block.dropout1(
            self.block.attention(self.block.norm1(x))
        ))
        x = x + self.block.path2(self.block.mlp(self.block.norm2(x)))

        return x


def apply_tome(
    model: nn.Module,
    r: int,
) -> nn.Module:
    """Reemplaza todos los TransformerEncoderBlock del modelo por ToMeEncoderBlock.

    La sustitución se hace in-place sobre ``model.blocks`` (Sequential de bloques).

    Args:
        model: VisionTransformer original (se modifica in-place).
        r:     Tokens a fusionar por bloque.

    Returns:
        El mismo modelo con los bloques reemplazados.
    """
    new_blocks = nn.Sequential(*[
        ToMeEncoderBlock(block, r=r)
        for block in model.blocks
    ])
    model.blocks = new_blocks
    return model
