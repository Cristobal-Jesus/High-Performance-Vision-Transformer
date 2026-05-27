"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/losses.py

Description:
    Función de pérdida para Knowledge Distillation.

    Combina la divergencia KL sobre targets suaves (soft targets del
    profesor) con la entropía cruzada sobre los targets duros (etiquetas
    reales).

    Fórmula:
        L = α · KL(log_softmax(s/T), softmax(t/T)) · T²
          + (1-α) · CE(s, y)

    Referencias:
        - Hinton et al. "Distilling the Knowledge in a Neural Network"
          https://arxiv.org/abs/1503.02531
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KnowledgeDistillationLoss(nn.Module):
    """Pérdida de destilación KD = α·KL(soft) + (1-α)·CE(hard).

    Args:
        temperature:     Temperatura T para suavizar los logits del profesor.
                         A mayor T, la distribución del profesor es más suave
                         y transfiere más "conocimiento oscuro".
        alpha:           Peso de la pérdida KL (0 ≤ α ≤ 1).
                         El complemento (1-α) pondera la CE con etiquetas hard.
        label_smoothing: Suavizado de etiquetas en la CE (0 = ninguno).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.70,
        label_smoothing: float = 0.10,
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha debe estar en [0, 1], recibido: {alpha}")

        self.temperature     = temperature
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

        self._ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Calcula la pérdida KD.

        Args:
            student_logits: ``(B, C)`` — logits del estudiante.
            teacher_logits: ``(B, C)`` — logits del profesor (sin gradientes).
            labels:         ``(B,)`` — etiquetas de clase.

        Returns:
            Escalar con la pérdida combinada.
        """
        T = self.temperature

        # Pérdida de distilación: KL sobre distribuciones suavizadas
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)

        # Pérdida con etiquetas hard
        ce_loss = self._ce(student_logits, labels)

        return self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
