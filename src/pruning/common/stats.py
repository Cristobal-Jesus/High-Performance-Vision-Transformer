"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/common/stats.py

Description:
    Dataclasses para almacenar los resultados de cada experimento de pruning.
    Sirven de interfaz entre el evaluador y el plotter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PruningStats:
    """Resultados de un experimento de pruning.

    Attributes:
        label:          Etiqueta descriptiva (ej. "Original", "ToMe r=8").
        accuracy:       Accuracy top-1 en validación (porcentaje).
        total_params:   Número total de parámetros del modelo.
        nonzero_params: Parámetros distintos de cero.
        ms_per_image:   Latencia media de inferencia en ms/imagen.
        sparsity:       Fracción de pesos nulos (0.0 = denso, 1.0 = todo cero).
    """

    label: str
    accuracy: float
    total_params: int
    nonzero_params: int
    ms_per_image: float
    sparsity: float = 0.0

    @property
    def size_mb(self) -> float:
        """Tamaño aproximado en MB asumiendo FP32 (4 bytes/param)."""
        return self.total_params * 4 / (1024 ** 2)

    @property
    def effective_size_mb(self) -> float:
        """Tamaño efectivo en MB contando solo parámetros no nulos."""
        return self.nonzero_params * 4 / (1024 ** 2)
