"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/visualization/plotter.py

Description:
    Clase de visualización para el entrenamiento de Knowledge Distillation.
    Genera una figura de dos paneles:
        1. Pérdida KD (train + val).
        2. Accuracy de validación (estudiante vs punto de inicio).

References:
    - https://matplotlib.org/stable/
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt


class DistillationPlotter:
    """Almacena el historial de destilación y genera la figura resumen."""

    def __init__(self) -> None:
        self.epochs:       t.List[int]            = []
        self.train_loss:   t.List[float]           = []
        self.train_acc:    t.List[float]           = []
        self.val_loss:     t.List[t.Optional[float]] = []
        self.val_acc:      t.List[t.Optional[float]] = []
        self.best_val_acc: t.Optional[float]       = None
        self.best_epoch:   t.Optional[int]         = None

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: t.Optional[float] = None,
        val_acc: t.Optional[float] = None,
        is_best: bool = False,
    ) -> None:
        """Registra las métricas de un epoch.

        Args:
            epoch:      Número de epoch.
            train_loss: Pérdida KD de entrenamiento.
            train_acc:  Accuracy de entrenamiento (%).
            val_loss:   Pérdida KD de validación (None si no se validó).
            val_acc:    Accuracy de validación (None si no se validó).
            is_best:    True si es el mejor epoch hasta ahora.
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

        if is_best and val_acc is not None:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch

    def plot(
        self,
        save_path: t.Union[str, Path],
        teacher_accuracy: t.Optional[float] = None,
    ) -> None:
        """Genera y guarda la figura de entrenamiento de destilación.

        Panel 1: Pérdida KD (train + val).
        Panel 2: Accuracy del estudiante (val) con línea de referencia
                 del profesor si se proporciona ``teacher_accuracy``.

        Args:
            save_path:         Ruta de destino para la figura PNG.
            teacher_accuracy:  Accuracy del modelo profesor (línea de
                               referencia en el panel de accuracy).
        """
        if not self.epochs:
            raise RuntimeError("No hay historial disponible para graficar.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Knowledge Distillation — Training", fontsize=14, fontweight="bold")

        self._plot_loss(axes[0])
        self._plot_accuracy(axes[1], teacher_accuracy)

        fig.tight_layout()

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[plotter] Figura guardada en {output_path}")

    # ------------------------------------------------------------------
    # Paneles individuales
    # ------------------------------------------------------------------

    def _plot_loss(self, ax) -> None:
        """Pinta las curvas de pérdida KD."""
        ax.plot(self.epochs, self.train_loss, label="Train KD Loss", color="tab:blue")

        val_epochs = [e for e, v in zip(self.epochs, self.val_loss) if v is not None]
        val_values = [v for v in self.val_loss if v is not None]
        if val_epochs:
            ax.plot(val_epochs, val_values, label="Val KD Loss",
                    color="tab:orange", linestyle="--", marker="o", markersize=3)

        ax.set_title("KD Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    def _plot_accuracy(
        self,
        ax,
        teacher_accuracy: t.Optional[float],
    ) -> None:
        """Pinta la accuracy de validación del estudiante."""
        val_epochs = [e for e, v in zip(self.epochs, self.val_acc) if v is not None]
        val_values = [v for v in self.val_acc if v is not None]

        if val_epochs:
            ax.plot(val_epochs, val_values, label="Student Val Acc",
                    color="tab:blue", marker="o", markersize=3)

        if teacher_accuracy is not None:
            ax.axhline(
                teacher_accuracy,
                color="tab:red",
                linestyle="--",
                linewidth=1.5,
                label=f"Teacher ({teacher_accuracy:.2f}%)",
            )

        if self.best_epoch is not None and self.best_val_acc is not None:
            ax.scatter(
                self.best_epoch,
                self.best_val_acc,
                color="red",
                zorder=5,
                label=f"Best ({self.best_val_acc:.2f}%)",
            )

        ax.set_title("Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
