"""Configuration object for pretrained ViT-B training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ViTBTrainingConfig:
    """Store all settings needed by the pretrained ViT-B trainer."""

    train_dir: Path
    val_dir: Path
    output_path: Path = Path("checkpoints/transformer/vit_b_16_pretrained.pth")
    num_classes: int = 20
    image_size: int = 224
    epochs: int = 200
    batch_size: int = 256
    num_workers: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 8
    min_delta: float = 0.0005
    pretrained: bool = True
    freeze_backbone: bool = False
    mixed_precision: bool = True
    compile_model: bool = False
    device: str = "cuda"
    eml_gpu_index: int | None = None
