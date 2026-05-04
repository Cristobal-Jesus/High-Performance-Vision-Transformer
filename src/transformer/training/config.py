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

import typing as t
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransformerTrainingConfig:
    """Store all hyperparameters and paths used during training."""

    root_dir: str = "/var/tmp/nhernang/dataset_256"
    train_batch_size: int = 192
    val_batch_size: int = 192
    epochs: int = 200
    warmup_epochs: int = 10
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 20
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.20
    learning_rate: float = 3e-4
    weight_decay: float = 0.10 
    label_smoothing: float = 0.15
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    mixup_alpha: float = 0.0
    patience: int = 25
    min_delta: float = 0.0005
    validate_every: int = 5
    max_val_batches: t.Optional[int] = None
    compile_model: bool = True
    show_progress_bar: bool = False
    num_threads: int = 32
    device_id: int = 0
    drop_last: bool = True
    train_prefetch_queue_depth: int = 4
    val_prefetch_queue_depth: int = 2
    auto_tune_for_gpu: bool = True
    eml_gpu_index: t.Optional[int] = None
    checkpoint_path: Path = Path("checkpoints/transformer/best_transformer.pth")
    figure_path: Path = Path("outputs/figures/transformer/V100/training_curves.png")

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
