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
    This file defines the configuration object used by the Transformer
    training application.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TransformerTrainingConfig:
    """Store all hyperparameters and paths used during training."""

    root_dir: str = "/home/almeida/Cristobal/Images/dataset2"
    train_batch_size: int = 1536
    val_batch_size: int = 1536
    epochs: int = 5
    warmup_epochs: int = 10
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 20
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    mixup_alpha: float = 0.0
    patience: int = 25
    min_delta: float = 0.0005
    num_threads: int = 32
    device_id: int = 0
    drop_last: bool = True
    train_prefetch_queue_depth: int = 10
    val_prefetch_queue_depth: int = 6
    eml_gpu_index: int | None = None
    checkpoint_path: Path = Path("checkpoints/transformer/best_transformer_20_classes_h200.pth")
    figure_path: Path = Path("outputs/figures/training_curves_20_classes_h200.png")

    @property
    def patch_dim(self) -> int:
        """Return the flattened size of one image patch."""
        return self.patch_size * self.patch_size * 3

    @property
    def seq_len(self) -> int:
        """Return the number of patches per image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def use_mixup(self) -> bool:
        """Return whether mixup should be enabled."""
        return self.mixup_alpha > 0.0
