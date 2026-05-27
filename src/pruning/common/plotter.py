"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/common/plotter.py

Description:
    Visualización comparativa de resultados de pruning.
    Genera un gráfico de dos paneles: accuracy y tamaño de modelo,
    comparando el modelo original con las variantes podadas.

References:
    - https://matplotlib.org/stable/
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from .stats import PruningStats

_BAR_COLORS: list[str] = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


class PruningPlotter:
    """Genera una figura comparativa para experimentos de pruning."""

    def plot(
        self,
        stats: List[PruningStats],
        output_path: str | Path,
        title: str = "Pruning Comparison",
    ) -> None:
        """Guarda una figura de dos paneles en *output_path*.

        Panel 1: Top-1 accuracy (%).
        Panel 2: Tamaño del modelo en MB (parámetros totales × 4 bytes).

        Args:
            stats:       Lista de :class:`PruningStats`, una por variante.
            output_path: Ruta de destino para la figura PNG.
            title:       Título de la figura.
        """
        labels = [s.label for s in stats]
        accuracy = [s.accuracy for s in stats]
        size_mb = [s.size_mb for s in stats]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        self._bar(axes[0], labels, accuracy, "Top-1 Accuracy (%)")
        self._bar(axes[1], labels, size_mb, "Model Size (MB)")

        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[plotter] Figura guardada en {output_path}")

    @staticmethod
    def _bar(
        ax,
        labels: list[str],
        values: list[float],
        ylabel: str,
    ) -> None:
        """Dibuja un panel de barras anotado."""
        colors = _BAR_COLORS[: len(labels)]
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(values) * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
