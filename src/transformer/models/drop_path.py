"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfonmance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th June 2024
File: drop_path.py

Description:
    This file defines the DropPath regularization layer, also known as
    stochastic depth, used to randomly drop residual paths during training.

References:
    - https://www.shadecoder.com/topics/drop-path-a-comprehensive-guide-for-2025
"""

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """Drop residual paths during training using stochastic depth."""

    def __init__(self, drop_prob: float = 0.1) -> None:
        super().__init__()

        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in the range [0.0, 1.0).")

        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic depth to the input tensor during training."""
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Create one binary mask per sample and broadcast it across the remaining dimensions.
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(mask_shape, device=x.device, dtype=x.dtype)
        binary_mask = (random_tensor < keep_prob).to(x.dtype)

        return x * binary_mask / keep_prob
