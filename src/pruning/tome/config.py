"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/tome/config.py

Description:
    Configuración del experimento Token Merging (ToMe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToMeConfig:
    """Parámetros para el experimento de Token Merging.

    Attributes:
        checkpoint_path: Ruta al checkpoint entrenado del VisionTransformer.
        r:               Número de pares de tokens a fusionar por bloque.
                         Valores típicos: 4, 8, 16.  A mayor ``r``, mayor
                         reducción de cómputo pero posible caída de accuracy.
        output_path:     Ruta donde guardar el modelo con ToMe aplicado.
        figure_path:     Ruta donde guardar la figura comparativa.
        root_dir:        Directorio raíz del dataset (para evaluación).
        val_batch_size:  Tamaño de batch de validación.
        device_id:       Índice de la GPU (0 = primera GPU).
        max_val_batches: Número máximo de batches de validación a evaluar
                         (``None`` = todos).
        image_size:      Tamaño de imagen cuadrada de entrada.
        patch_size:      Tamaño de parche.
        num_classes:     Número de clases.
        embed_dim:       Dimensión del embedding.
        depth:           Número de bloques del encoder.
        num_heads:       Número de cabezas de atención.
        mlp_ratio:       Ratio MLP.
        dropout:         Dropout de activaciones.
        drop_path_rate:  DropPath estocástico.
    """

    # --- rutas ---
    checkpoint_path: Path = Path("checkpoints/transformer/best_transformer.pth")
    output_path: Path = Path("checkpoints/pruning/tome/tome_model.pth")
    figure_path: Path = Path("outputs/figures/pruning/tome/comparison.png")

    # --- ToMe ---
    r: int = 8   # tokens a fusionar por bloque

    # --- datos ---
    root_dir: str = "/home/almeida/Cristobal/Images/dataset_256"
    val_batch_size: int = 128
    device_id: int = 0
    max_val_batches: int | None = None
    num_threads: int = 4

    # --- arquitectura (debe coincidir con el checkpoint) ---
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 20
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    drop_path_rate: float = 0.20
