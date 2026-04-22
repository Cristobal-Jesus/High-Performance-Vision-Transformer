"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th Marh 2026
File: training_plotter.py

Description:
    This file defines the class responsible for storing training history
    and generating plots for accuracy, loss, and optional energy metrics.

References:
    - https://matplotlib.org/stable/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


class TrainingPlotter:
    """Store training history and generate summary plots."""

    def __init__(self) -> None:
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0

    def update(
        self,
        train_acc: float,
        val_acc: float,
        train_loss: float,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Store the metrics of one epoch."""
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        if is_best:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.val_acc)

    def plot(
        self,
        save_path: str = "outputs/figures/transformer/training_curves_ViT-B_16.png",
        energy_stats: Optional[Dict[str, Any]] = None,
        device_info: Optional[str] = None,
    ) -> None:
        """Plot training curves and optionally save them to disk."""
        self._validate_history()

        epochs = list(range(1, len(self.train_acc) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        self._plot_accuracy(axes[0], epochs)
        self._plot_loss(axes[1], epochs)

        if energy_stats is not None:
            self._add_energy_text(fig, energy_stats, device_info)

        fig.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    def _plot_accuracy(self, axis, epochs: List[int]) -> None:
        """Plot training and validation accuracy."""
        axis.plot(epochs, self.train_acc, label="Train Accuracy")
        axis.plot(epochs, self.val_acc, label="Validation Accuracy")

        if self.best_epoch is not None and self.best_val_acc is not None:
            axis.scatter(
                self.best_epoch,
                self.best_val_acc,
                color="red",
                zorder=5,
                label="Best Validation Accuracy",
            )
            axis.annotate(
                f"{self.best_val_acc:.2f}%",
                (self.best_epoch, self.best_val_acc),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="red",
            )

        axis.set_title("Accuracy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.legend()

    def _plot_loss(self, axis, epochs: List[int]) -> None:
        """Plot training and validation loss."""
        axis.plot(epochs, self.train_loss, label="Train Loss")
        axis.plot(epochs, self.val_loss, label="Validation Loss")
        axis.set_title("Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()

    def _add_energy_text(
        self,
        figure,
        energy_stats: Dict[str, Any],
        device_info: Optional[str],
    ) -> None:
        """Add energy and device information to the figure."""
        gpu_joules, cpu_joules, elapsed_seconds = self._extract_energy_stats(energy_stats)
        total_joules = gpu_joules + cpu_joules

        text = (
            f"Device: {device_info or 'Unknown'}\n"
            f"GPU Energy: {gpu_joules:.4f} J\n"
            f"CPU Energy: {cpu_joules:.4f} J\n"
            f"Total Energy: {total_joules:.4f} J\n"
            f"Time: {elapsed_seconds / 60:.2f} min"
        )

        figure.text(
            0.02,
            0.02,
            text,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    def _extract_energy_stats(
        self,
        energy_stats: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """Extract GPU energy, CPU energy, and elapsed time from the metrics dictionary."""
        measurements = energy_stats.get("meas", {})
        elapsed_seconds = float(energy_stats.get("elapsed", 0.0))

        gpu_joules = 0.0
        cpu_joules = 0.0

        if "gpu_energy_j" in measurements:
            gpu_joules = float(measurements["gpu_energy_j"])
        elif "gpu_energy_uj" in measurements:
            gpu_joules = float(measurements["gpu_energy_uj"]) / 1e6

        for device, values in measurements.items():
            if "rapl" not in str(device).lower():
                continue

            if isinstance(values, dict):
                consumed_microjoules = float(values.get("consumed", 0.0))
            elif isinstance(values, (int, float)):
                consumed_microjoules = float(values)
            else:
                consumed_microjoules = 0.0

            cpu_joules += consumed_microjoules / 1e6

        return gpu_joules, cpu_joules, elapsed_seconds

    def _validate_history(self) -> None:
        """Validate that the metric history is consistent before plotting."""
        history_lengths = {
            len(self.train_acc),
            len(self.val_acc),
            len(self.train_loss),
            len(self.val_loss),
        }

        if len(history_lengths) != 1:
            raise RuntimeError("Training history lists must have the same length.")
        if not self.train_acc:
            raise RuntimeError("No training history is available to plot.")
