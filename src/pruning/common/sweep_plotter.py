"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/common/sweep_plotter.py

Description:
    Plotter para experimentos de barrido de intensidad de poda.
    Genera una figura de dos paneles (accuracy y métrica secundaria)
    frente al parámetro de poda, mostrando la curva de compromiso.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


@dataclass
class SweepPoint:
    """Resultado de un nivel de poda en el barrido.

    Attributes:
        param_value:      Valor del parámetro de poda (r, ratio, sparsity…).
        accuracy:         Top-1 accuracy (%).
        secondary_metric: Métrica secundaria (speedup, sparsidad real…).
        label:            Etiqueta corta para el punto en la gráfica.
    """

    param_value: float
    accuracy: float
    secondary_metric: float
    label: str = ""


class PruningSweepPlotter:
    """Genera la gráfica de barrido de intensidad de poda.

    Args:
        x_label:         Etiqueta del eje X (parámetro de poda).
        secondary_label: Etiqueta del eje Y del panel derecho.
        title:           Título de la figura.
    """

    def __init__(
        self,
        x_label: str = "Intensidad de poda",
        secondary_label: str = "Métrica secundaria",
        title: str = "Barrido de intensidad de poda",
    ) -> None:
        self._x_label = x_label
        self._secondary_label = secondary_label
        self._title = title

    def plot(
        self,
        points: list[SweepPoint],
        output_path: str | Path,
        baseline_accuracy: float | None = None,
    ) -> None:
        """Guarda la figura de barrido en *output_path*.

        Args:
            points:            Lista de puntos del barrido, ordenados por
                               ``param_value`` ascendente.
            output_path:       Ruta de destino (PNG).
            baseline_accuracy: Accuracy del modelo original sin podar.
                               Se dibuja como línea de referencia discontinua.
        """
        xs  = [p.param_value      for p in points]
        acc = [p.accuracy         for p in points]
        sec = [p.secondary_metric for p in points]

        fig, (ax_acc, ax_sec) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(self._title, fontsize=13, fontweight="bold")

        # ── Panel izquierdo: Accuracy ──────────────────────────────────
        ax_acc.plot(xs, acc, color="black", lw=2.0,
                    marker="o", ms=6, zorder=3)
        if baseline_accuracy is not None:
            ax_acc.axhline(
                baseline_accuracy,
                color="gray", lw=1.2, linestyle="--",
                label=f"Original ({baseline_accuracy:.2f}%)",
            )
            ax_acc.legend(fontsize=8, framealpha=0.9)

        # Anotar cada punto
        for p in points:
            ax_acc.annotate(
                f"{p.accuracy:.2f}%",
                xy=(p.param_value, p.accuracy),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center", fontsize=7.5,
            )

        ax_acc.set_xlabel(self._x_label, fontsize=10)
        ax_acc.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
        ax_acc.set_xticks(xs)
        ax_acc.grid(True, axis="y", color="#e0e0e0")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)

        # ── Panel derecho: métrica secundaria ─────────────────────────
        ax_sec.plot(xs, sec, color="black", lw=2.0,
                    marker="s", ms=6, linestyle="--", zorder=3)

        for p in points:
            ax_sec.annotate(
                f"{p.secondary_metric:.2f}",
                xy=(p.param_value, p.secondary_metric),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center", fontsize=7.5,
            )

        ax_sec.set_xlabel(self._x_label, fontsize=10)
        ax_sec.set_ylabel(self._secondary_label, fontsize=10)
        ax_sec.set_xticks(xs)
        ax_sec.grid(True, axis="y", color="#e0e0e0")
        ax_sec.spines["top"].set_visible(False)
        ax_sec.spines["right"].set_visible(False)

        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[sweep] Figura guardada en {output_path}")
