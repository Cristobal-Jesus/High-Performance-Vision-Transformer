"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/structured/config.py

Description:
    Configuración del experimento de poda estructurada de cabezas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class StructuredPruningConfig:
    """Parámetros para el experimento de poda estructurada de cabezas.

    Attributes:
        checkpoint_path:  Ruta al checkpoint del VisionTransformer.
        prune_ratio:      Fracción de cabezas a podar en cada capa.
                          0.25 → se poda 1 de cada 4 cabezas por bloque.
        output_path:      Ruta donde guardar el modelo podado.
        figure_path:      Ruta donde guardar la figura comparativa.
        root_dir:         Directorio raíz del dataset.
        val_batch_size:   Tamaño de batch de validación.
        device_id:        Índice de la GPU.
        max_val_batches:  Número máximo de batches de validación (None = todos).
        num_threads:      Hilos DALI.
        image_size:       Tamaño de imagen.
        patch_size:       Tamaño de parche.
        num_classes:      Número de clases.
        embed_dim:        Dimensión embedding.
        depth:            Profundidad del encoder.
        num_heads:        Número de cabezas.
        mlp_ratio:        Ratio MLP.
        dropout:          Dropout de activaciones.
        drop_path_rate:   DropPath.
    """

    # --- rutas ---
    checkpoint_path: Path = Path("checkpoints/transformer/best_transformer.pth")
    output_path: Path = Path("checkpoints/pruning/structured/structured_model.pth")
    figure_path: Path = Path("outputs/figures/pruning/structured/comparison.png")

    # --- poda ---
    prune_ratio: float = 0.25   # fracción de cabezas a podar por capa

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
