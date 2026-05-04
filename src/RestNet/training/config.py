"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: config.py

Description:
    This file defines the configuration object used by the ResNet
    training application.
"""

import typing as t
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResNetTrainingConfig:
    """Store all hyperparameters and paths used during training."""

    root_dir: str = "/var/tmp/nhernang/dataset_256"
    train_batch_size: int = 128
    val_batch_size: int = 128
    epochs: int = 200
    warmup_epochs: int = 5
    image_size: int = 224
    num_classes: int = 20

    model_name: str = "resnet50"
    pretrained: bool = True

    learning_rate: float = 5e-5
    weight_decay: float = 0.10
    label_smoothing: float = 0.15
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    mixup_alpha: float = 0.0

    patience: int = 25
    min_delta: float = 0.0005

    num_threads: int = 32
    device_id: int = 0
    drop_last: bool = True
    train_prefetch_queue_depth: int = 2
    val_prefetch_queue_depth: int = 2

    eml_gpu_index: t.Optional[int] = None

    checkpoint_path: Path = Path("checkpoints/resnet/best_resnet50.pth")
    figure_path: Path = Path("outputs/figures/convolutional/V100/training_curves_resnet50.png")

    @property
    def use_mixup(self) -> bool:
        """Return whether mixup should be enabled."""
        return self.mixup_alpha > 0.0
