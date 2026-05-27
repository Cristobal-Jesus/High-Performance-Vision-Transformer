"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/tome/merger.py

Description:
    Implementación del algoritmo de Bipartite Soft Matching (BSM) de ToMe
    (Bolya et al., 2022).  Devuelve una función que fusiona r pares de
    tokens similares, reduciendo la longitud de secuencia de N a N-r.

    El token CLS (índice 0) nunca se fusiona: siempre permanece en el
    conjunto de destino.

References:
    - Bolya et al. "Token Merging: Your ViT But Faster" (ICLR 2023)
      https://arxiv.org/abs/2210.09461
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def bipartite_soft_matching(
    metric: Tensor,
    r: int,
) -> Callable[[Tensor], Tensor]:
    """Calcula el esquema de fusión y devuelve una función ``merge``.

    El algoritmo parte los tokens en dos conjuntos (A: índices pares,
    B: índices impares), calcula la similitud coseno entre ellos y
    fusiona los ``r`` pares más parecidos sumando el token B al token A
    más cercano y haciendo la media.  Los tokens B no fusionados se
    concatenan al final.

    El token CLS (posición 0 en el tensor de entrada) siempre queda
    protegido en el conjunto A.

    Args:
        metric: ``(B, N, C)`` — vector de similitud por token (e.g. claves K
                promediadas sobre las cabezas).
        r:      Número de tokens a eliminar.  Si ``r >= |B|`` se recorta al
                máximo posible.

    Returns:
        ``merge``: función ``(B, N, C) -> (B, N-r, C)`` que aplica la
        fusión a cualquier tensor con la misma estructura espacial que
        ``metric``.
    """
    B, N, _ = metric.shape

    # Proteger CLS: solo se trabaja con los tokens de parche (índices 1…N-1)
    patch = F.normalize(metric[:, 1:, :], p=2, dim=-1)   # (B, N-1, C)
    n_patch = N - 1

    a = patch[:, ::2, :]    # conjunto A: parches en posiciones pares
    b = patch[:, 1::2, :]   # conjunto B: parches en posiciones impares
    n_b = b.shape[1]

    # Recortar r al máximo sensato
    r = min(r, n_b)

    with torch.no_grad():
        # Similitud coseno (B, |A|, |B|) — ya normalizados arriba
        scores = a @ b.transpose(-1, -2)

        # Para cada token de B, el token A más parecido
        node_max, node_idx = scores.max(dim=-2)   # (B, |B|)

        # Ordenar B por similitud descendente
        order = node_max.argsort(dim=-1, descending=True)  # (B, |B|)
        src_idx = order[..., :r]    # (B, r) — tokens B a fusionar
        unm_idx = order[..., r:]    # (B, |B|-r) — tokens B que quedan

        # Índice del token A destino para cada token B fusionado
        dst_idx = node_idx.gather(dim=-1, index=src_idx)   # (B, r)

    n_a = a.shape[1]

    def merge(x: Tensor) -> Tensor:
        """Fusiona r pares de tokens: ``(B, N, C) -> (B, N-r, C)``.

        Args:
            x: Tensor de tokens de forma ``(B, N, C)``.

        Returns:
            Tensor fusionado de forma ``(B, N-r, C)``.
        """
        B_x, N_x, C = x.shape

        cls   = x[:, :1, :]        # (B, 1, C) — siempre protegido
        patches = x[:, 1:, :]      # (B, N-1, C)

        xa = patches[:, ::2, :]    # (B, |A|, C)
        xb = patches[:, 1::2, :]   # (B, |B|, C)

        # Acumular tokens B fusionados en sus destinos de A (suma)
        exp = dst_idx.unsqueeze(-1).expand(B_x, -1, C)
        src_tokens = xb.gather(dim=1, index=src_idx.unsqueeze(-1).expand(B_x, -1, C))
        xa = xa.clone()
        xa.scatter_add_(1, exp, src_tokens)

        # Normalizar destinos que recibieron contribuciones (media)
        ones   = torch.ones(B_x, r, 1, device=x.device, dtype=x.dtype)
        counts = torch.ones(B_x, n_a, 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, dst_idx.unsqueeze(-1), ones)
        xa = xa / counts

        # Tokens B no fusionados
        unm_tokens = xb.gather(
            dim=1,
            index=unm_idx.unsqueeze(-1).expand(B_x, -1, C),
        )

        return torch.cat([cls, xa, unm_tokens], dim=1)   # (B, N-r, C)

    return merge
