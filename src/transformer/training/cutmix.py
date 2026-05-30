"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: training/cutmix.py

Description:
    CutMix data augmentation (Yun et al., 2019).

    A diferencia de Mixup, que mezcla píxeles linealmente, CutMix corta
    un rectángulo aleatorio de una imagen y lo pega sobre otra.  Las
    etiquetas se mezclan proporcionalmente al área cortada.

    Ventaja respecto a Mixup: preserva regiones locales intactas, lo que
    ayuda a que el modelo aprenda características más robustas sin las
    mezclas de píxeles que confunden la textura.

    Se aplica sobre imágenes en crudo (antes de parchificar), con forma
    ``(B, C, H, W)``.

References:
    - Yun et al. "CutMix: Training Strategy that Makes Use of Sample
      Mixing" (ICCV 2019) https://arxiv.org/abs/1905.04899
"""

from __future__ import annotations

import typing as t

import numpy as np
import torch
from torch import Tensor


class CutMixAugmentor:
    """Aplica CutMix a lotes de imágenes en crudo ``(B, C, H, W)``."""

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Args:
            alpha: Parámetro de la distribución Beta para muestrear lambda
                   (fracción de área conservada de la imagen original).
                   ``alpha=1.0`` es el valor estándar de DeiT.
                   ``alpha=0.0`` desactiva CutMix.
        """
        if alpha < 0.0:
            raise ValueError("alpha debe ser mayor o igual a 0.")
        self.alpha = alpha

    @property
    def enabled(self) -> bool:
        """True si CutMix está activado."""
        return self.alpha > 0.0

    def apply(
        self,
        x: Tensor,
        y: Tensor,
    ) -> t.Tuple[Tensor, Tensor, Tensor, float]:
        """Aplica CutMix a un lote de imágenes.

        Args:
            x: Tensor de imágenes ``(B, C, H, W)``.
            y: Tensor de etiquetas ``(B,)``.

        Returns:
            Tupla ``(x_mixed, y_a, y_b, lam)``:
                - ``x_mixed``: imágenes con el parche pegado.
                - ``y_a``: etiquetas originales.
                - ``y_b``: etiquetas del lote permutado (fuente del parche).
                - ``lam``: fracción de área perteneciente a la imagen original.
        """
        if not self.enabled:
            return x, y, y, 1.0

        lam = float(np.random.beta(self.alpha, self.alpha))
        B, C, H, W = x.shape
        indices = torch.randperm(B, device=x.device)

        # Dimensiones del rectángulo a cortar
        cut_h = int(H * np.sqrt(1.0 - lam))
        cut_w = int(W * np.sqrt(1.0 - lam))

        # Centro aleatorio del corte
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]

        # Lambda real según el área efectivamente cortada
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

        return x_mixed, y, y[indices], lam
