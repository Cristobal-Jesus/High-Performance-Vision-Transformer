"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: schedulers.py

Description:
    This file defines learning-rate scheduler factories used during
    training.
"""

import math

import torch


class SchedulerFactory:
    """Build schedulers used by the training application."""

    @staticmethod
    def build_warmup_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        epochs: int,
        warmup_epochs: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create a warmup + cosine decay schedule."""

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            t = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
