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
    Plotting class that generates a three-panel bar chart comparing
    top-1 accuracy, model size and inference throughput across the
    FP32, FP16 and BF16 precision variants.

References:
    - https://matplotlib.org/stable/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .precision_stats import PrecisionStats

_BAR_COLORS: list[str] = ["#4C72B0", "#DD8452", "#55A868"]


class ComparisonPlotter:
    """Generate a side-by-side comparison figure for precision benchmarks."""

    def plot(
        self,
        stats: list[PrecisionStats],
        output_path: str | Path,
    ) -> None:
        """Save a three-panel bar chart to *output_path*.

        The three panels show:
          1. Top-1 accuracy (%)
          2. Model size (MB)
          3. Inference throughput (images / second)

        Args:
            stats: One ``PrecisionStats`` per dtype variant, in display order.
            output_path: Destination path for the saved figure (PNG).
        """
        labels = [s.label for s in stats]
        accuracy = [s.accuracy for s in stats]
        size_mb = [s.size_mb for s in stats]
        throughput = [s.throughput for s in stats]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            "Precision Comparison: FP32 vs FP16 vs BF16",
            fontsize=14,
            fontweight="bold",
        )

        self._bar(axes[0], labels, accuracy, "Top-1 Accuracy (%)")
        self._bar(axes[1], labels, size_mb, "Model Size (MB)")
        self._bar(axes[2], labels, throughput, "Throughput (img/s)")

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
        bars = ax.bar(labels, values, color=_BAR_COLORS[: len(labels)], width=0.5)
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
