"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfonmance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2024
File: focal_loss.py

Description:
    This file defines the Focal Loss function with optional label smoothing
    for multi-class classification tasks.

References:
    - https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional label smoothing."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05) -> None:
        super().__init__()

        if gamma < 0.0:
            raise ValueError("gamma must be greater than or equal to 0.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0.0, 1.0).")

        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss from logits and class indices."""
        if logits.ndim != 2:
            raise ValueError(
                f"logits must have shape (batch_size, num_classes), got {tuple(logits.shape)}."
            )
        if targets.ndim != 1:
            raise ValueError(
                f"targets must have shape (batch_size,), got {tuple(targets.shape)}."
            )
        if logits.size(0) != targets.size(0):
            raise ValueError("logits and targets must contain the same number of samples.")

        num_classes = logits.size(1)
        if num_classes < 2:
            raise ValueError("logits must contain at least two classes.")
        if self.label_smoothing > 0.0 and num_classes == 1:
            raise ValueError("Label smoothing requires at least two classes.")

        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits,
                fill_value=self.label_smoothing / (num_classes - 1),
            )
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.label_smoothing,
            )

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Down-weight well-classified examples and focus more on hard ones.
        focal_factor = (1.0 - probs) ** self.gamma
        loss = -smooth_targets * focal_factor * log_probs

        return loss.sum(dim=1).mean()
