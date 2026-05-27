"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/structured/head_pruner.py

Description:
    Poda estructurada de cabezas de atención en el VisionTransformer.

    Algoritmo:
        1. Para cada bloque encoder, calcula la puntuación de importancia
           de cada cabeza como la norma L1 de sus columnas V en la
           proyección QKV fusionada.
        2. Ordena las cabezas por importancia (ascendente) y anula los
           pesos de las ``floor(num_heads * prune_ratio)`` menos importantes.
        3. Anular una cabeza significa poner a cero las filas V del peso
           QKV y la columna correspondiente del peso de proyección.

    La poda se aplica in-place; los pesos no se eliminan físicamente del
    tensor (el shape se mantiene), pero los gradientes de las columnas
    anuladas son cero, por lo que el modelo actúa como si esas cabezas no
    existiesen.

    Nota: esta es la forma habitual de "probar" la poda estructurada antes
    de una reconstrucción del modelo con menos cabezas.  Para la destilación
    posterior, el modelo podado se usa directamente como estudiante.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from transformer.models.transformer_model import VisionTransformer


def _head_importance_scores(
    qkv_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Calcula la importancia de cada cabeza como norma L1 de sus pesos V.

    En el peso QKV fusionado de forma ``(3*D, D)``, las filas de la
    parte V (bloque [2*D : 3*D]) se dividen entre cabezas:
    cabeza h → filas ``[2*D + h*Dh : 2*D + (h+1)*Dh]``.

    Args:
        qkv_weight: Peso de la proyección QKV, forma ``(3*D, D)``.
        num_heads:  Número de cabezas de atención.
        head_dim:   Dimensión por cabeza.

    Returns:
        Tensor de forma ``(num_heads,)`` con la puntuación de cada cabeza.
    """
    embed_dim = num_heads * head_dim
    # Extraer bloque V: filas [2*D, 3*D]
    v_weight = qkv_weight[2 * embed_dim: 3 * embed_dim, :]   # (D, D)
    # Dividir en cabezas
    v_heads = v_weight.reshape(num_heads, head_dim, -1)       # (H, Dh, D)
    # Norma L1 por cabeza
    return v_heads.abs().sum(dim=(1, 2))                       # (H,)


def _zero_head(
    qkv_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    head_idx: int,
    head_dim: int,
    num_heads: int,
) -> None:
    """Anula los pesos de la cabeza ``head_idx`` en QKV y en la proyección.

    Se modifica in-place sin gradientes.

    Args:
        qkv_weight:  Peso QKV, forma ``(3*D, D)``.
        proj_weight: Peso de salida, forma ``(D, D)``.
        head_idx:    Índice de la cabeza a anular (0-indexado).
        head_dim:    Dimensión por cabeza.
        num_heads:   Número total de cabezas.
    """
    embed_dim = num_heads * head_dim

    with torch.no_grad():
        # Anular Q, K, V de esta cabeza en el tensor QKV
        for part in range(3):
            start = part * embed_dim + head_idx * head_dim
            end   = start + head_dim
            qkv_weight[start:end, :] = 0.0

        # Anular la columna correspondiente en la proyección de salida
        col_start = head_idx * head_dim
        col_end   = col_start + head_dim
        proj_weight[:, col_start:col_end] = 0.0


class HeadPruner:
    """Poda las cabezas de atención menos importantes del VisionTransformer.

    Calcula la importancia de cada cabeza y anula las ``prune_ratio``
    menos importantes en cada bloque encoder.

    Args:
        prune_ratio: Fracción de cabezas a podar en cada bloque (0 < r < 1).
    """

    def __init__(self, prune_ratio: float = 0.25) -> None:
        if not (0.0 < prune_ratio < 1.0):
            raise ValueError(f"prune_ratio debe estar en (0, 1), recibido: {prune_ratio}")
        self.prune_ratio = prune_ratio

    def compute_importances(
        self,
        model: VisionTransformer,
    ) -> Dict[int, torch.Tensor]:
        """Devuelve las puntuaciones de importancia por bloque.

        Args:
            model: VisionTransformer (puede estar en cualquier dispositivo).

        Returns:
            Dict ``{bloque_idx: tensor (num_heads,)}`` con puntuaciones L1.
        """
        importances: Dict[int, torch.Tensor] = {}
        for idx, block in enumerate(model.blocks):
            qkv_w = block.attention.qkv.weight   # (3D, D)
            scores = _head_importance_scores(
                qkv_w,
                num_heads=block.attention.num_heads,
                head_dim=block.attention.head_dim,
            )
            importances[idx] = scores.detach().cpu()
        return importances

    def prune(self, model: VisionTransformer) -> Tuple[VisionTransformer, Dict[int, List[int]]]:
        """Aplica la poda in-place al modelo.

        Args:
            model: VisionTransformer a podar.

        Returns:
            Tupla ``(model_podado, heads_podadas)`` donde ``heads_podadas``
            es un dict ``{bloque_idx: [head_idx, ...]}``.
        """
        heads_pruned: Dict[int, List[int]] = {}

        for idx, block in enumerate(model.blocks):
            num_heads = block.attention.num_heads
            head_dim  = block.attention.head_dim
            n_to_prune = max(1, math.floor(num_heads * self.prune_ratio))

            qkv_weight  = block.attention.qkv.weight
            proj_weight = block.attention.proj.weight

            scores = _head_importance_scores(qkv_weight, num_heads, head_dim)
            # Cabezas menos importantes primero
            sorted_heads = scores.argsort().tolist()
            to_prune = sorted_heads[:n_to_prune]

            for h in to_prune:
                _zero_head(qkv_weight, proj_weight, h, head_dim, num_heads)

            heads_pruned[idx] = to_prune
            print(
                f"  Bloque {idx:2d}: podadas {n_to_prune} cabezas "
                f"(índices {to_prune})"
            )

        return model, heads_pruned
