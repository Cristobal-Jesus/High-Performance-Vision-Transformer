"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: plotter.py

Description:
    Plotting class that generates a two-panel bar chart comparing
    accuracy and model size on disk between the FP32 baseline and the
    INT8 dynamically quantized model.

References:
    - https://matplotlib.org/stable/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .quantization_stats import QuantizationStats

_BAR_COLORS: list[str] = ["#4C72B0", "#DD8452"]


class QuantizationPlotter:
    """Generate a comparison figure for FP32 vs INT8 benchmarks."""

    def plot(
        self,
        stats: list[QuantizationStats],
        output_path: str | Path,
    ) -> None:
        """Save a two-panel bar chart to *output_path*.

        The two panels show:
          1. Top-1 accuracy (%)
          2. Model size on disk (MB)

        Args:
            stats: One ``QuantizationStats`` per variant, in display order.
            output_path: Destination path for the saved figure (PNG).
        """
        labels = [s.label for s in stats]
        accuracy = [s.accuracy for s in stats]
        disk_size = [s.disk_size_mb for s in stats]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            "Quantization Comparison: FP32 vs INT8 (Dynamic)",
            fontsize=14,
            fontweight="bold",
        )

        self._bar(axes[0], labels, accuracy, "Top-1 Accuracy (%)")
        self._bar(axes[1], labels, disk_size, "Model Size on Disk (MB)")

        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Figure saved to {output_path}")

    @staticmethod
    def _bar(
        ax,
        labels: list[str],
        values: list[float],
        ylabel: str,
    ) -> None:
        """Draw a single annotated bar chart panel."""
        bars = ax.bar(labels, values, color=_BAR_COLORS[: len(labels)], width=0.4)
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
