"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/config.py

Description:
    Configuración del experimento de Knowledge Distillation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DistillationConfig:
    """Parámetros para el experimento de Knowledge Distillation.

    Attributes:
        teacher_path:      Ruta al checkpoint del modelo profesor
                           (VisionTransformer original entrenado).
        student_path:      Ruta al checkpoint del modelo estudiante
                           (modelo podado, output de pruning/).
        output_path:       Ruta donde guardar el estudiante destilado.
        figure_path:       Ruta donde guardar la figura de entrenamiento.

        temperature:       Temperatura T para suavizar los logits del
                           profesor.  Valores típicos: 3–6.
        alpha:             Peso de la pérdida KL (distilación).
                           (1-alpha) es el peso de la CE con etiquetas hard.

        epochs:            Épocas de destilación.
        learning_rate:     Tasa de aprendizaje del estudiante.
        weight_decay:      Regularización L2.
        warmup_epochs:     Épocas de calentamiento del LR.
        grad_clip_norm:    Norma máxima del gradiente (clipping).

        root_dir:          Directorio raíz del dataset.
        train_batch_size:  Tamaño de batch de entrenamiento.
        val_batch_size:    Tamaño de batch de validación.
        num_threads:       Hilos DALI.
        device_id:         Índice de la GPU.
        validate_every:    Validar cada N épocas.
        max_val_batches:   Batches máximos de validación (None = todos).

        image_size:        Tamaño de imagen de entrada.
        patch_size:        Tamaño de parche.
        num_classes:       Número de clases.
        embed_dim:         Dimensión del embedding.
        depth:             Bloques del encoder.
        num_heads:         Cabezas de atención.
        mlp_ratio:         Ratio MLP.
        dropout:           Dropout de activaciones.
        drop_path_rate:    DropPath estocástico.
        tome_r:            Si > 0, aplica ToMe con este r al estudiante
                           al cargarlo (solo si el student_path es el
                           checkpoint base, no el modelo ToMe ya guardado).
    """

    # --- rutas ---
    teacher_path: Path = Path("checkpoints/transformer/best_transformer.pth")
    student_path: Path = Path("checkpoints/pruning/tome/tome_model.pth")
    output_path: Path = Path("checkpoints/distillation/student_distilled.pth")
    figure_path: Path = Path("outputs/figures/distillation/training_curves.png")

    # --- destilación ---
    temperature: float = 4.0
    alpha: float = 0.70       # peso KL; 0.30 → CE con etiquetas hard

    # --- bucle de entrenamiento ---
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.10

    # --- datos ---
    root_dir: str = "/home/almeida/Cristobal/Images/dataset_256"
    train_batch_size: int = 128
    val_batch_size: int = 128
    num_threads: int = 4
    device_id: int = 0
    validate_every: int = 5
    max_val_batches: int | None = None
    drop_last: bool = True
    train_prefetch_queue_depth: int = 3
    val_prefetch_queue_depth: int = 2

    # --- arquitectura (debe coincidir con los checkpoints) ---
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 20
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    drop_path_rate: float = 0.20

    # --- ToMe en el estudiante (0 = desactivado) ---
    tome_r: int = 0
