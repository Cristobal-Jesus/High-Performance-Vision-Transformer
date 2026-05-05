"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: constants.py

Description:
    Shared constants for the INT8 dynamic quantization comparison pipeline.
"""

from __future__ import annotations

from pathlib import Path

CLASS_NAMES: list[str] = [
    "Anade_azulon",
    "Bald_eagle",
    "Bee",
    "Camachuelo_mexicano",
    "Downy_woodpecker",
    "Firebug",
    "Garceta_grande",
    "Garza_azulada",
    "Mariquita",
    "Martinete_comun",
    "Mirlo_comun",
    "Monarch",
    "Northern_cardinal",
    "Painted_lady",
    "Paloma",
    "Red_spotted_admirial",
    "Small_white",
    "White_heron",
    "White_tailed_deer",
    "Zorzal_americano",
]

LABEL_MAP: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

CHECKPOINT_PATH: Path = Path(
    "checkpoints/transformer/best_transformer.pth"
)
INT8_CHECKPOINT_PATH: Path = Path(
    "checkpoints/transformer/best_transformer_int8_cpu.pt"
)

IMAGE_SIZE: int = 224
PATCH_SIZE: int = 16
EVAL_BATCH_SIZE: int = 64

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

VARIANT_LABEL: dict[str, str] = {
    "fp32": "FP32",
    "int8": "INT8 (Dynamic)",
}