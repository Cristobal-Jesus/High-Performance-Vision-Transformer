"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: mixup.py

Description:
    This file defines the class responsible for applying mixup during
    training.
"""

import typing as t

import numpy as np
import torch
from torch import Tensor


class MixupAugmentor:
    """Apply mixup to batches of samples and labels."""

    def __init__(self, alpha: float = 0.0) -> None:
        if alpha < 0.0:
            raise ValueError("alpha must be greater than or equal to 0.")

        self.alpha = alpha

    @property
    def enabled(self) -> bool:
        """Return whether mixup is enabled."""
        return self.alpha > 0.0

    def apply(self, x: Tensor, y: Tensor) -> t.Tuple[Tensor, Tensor, Tensor, float]:
        """Apply mixup to one batch."""
        if not self.enabled:
            return x, y, y, 1.0

        lam = float(np.random.beta(self.alpha, self.alpha))
        indices = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[indices]
        return mixed_x, y, y[indices], lam
