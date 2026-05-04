"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: visualization.py

Description:
    This file defines the object responsible for creating table and chart
    visualizations from PyTorch model metrics.
"""


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .model_metrics import ModelMetrics


class ComparisonVisualizer:
    """Creates polished table and chart artifacts from model metrics."""

    def __init__(self, output_dir: Path) -> None:
        """Initializes the visualizer."""
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, metrics: list[ModelMetrics]) -> None:
        """Saves CSV, table image and accuracy chart for the comparison."""
        dataframe = self._build_dataframe(metrics)
        dataframe.to_csv(self._output_dir / "model_summary.csv", index=False)
        self._save_table(dataframe)
        self._save_accuracy_chart(dataframe)

    def _build_dataframe(self, metrics: list[ModelMetrics]) -> pd.DataFrame:
        """Builds a presentation-ready dataframe."""
        rows = []
        for item in metrics:
            rows.append(
                {
                    "Modelo": item.name,
                    "Accuracy (%)": item.accuracy,
                    "Peso (MB)": round(item.file_size_mb, 2),
                    "Parámetros totales": item.total_parameters,
                    "Tensores": item.tensor_count,
                    "Dtypes": item.dtype_summary,
                }
            )

        return pd.DataFrame(rows)

    def _save_table(self, dataframe: pd.DataFrame) -> None:
        """Saves a visually polished table as a PNG image."""
        display_dataframe = dataframe.copy()
        display_dataframe["Accuracy (%)"] = display_dataframe["Accuracy (%)"].map(
            "{:.2f}".format
        )
        display_dataframe["Peso (MB)"] = display_dataframe["Peso (MB)"].map(
            "{:.2f}".format
        )
        display_dataframe["Parámetros totales"] = display_dataframe[
            "Parámetros totales"
        ].map("{:,}".format)


        row_count = max(len(display_dataframe), 1)
        fig_height = 1.25 + row_count * 0.55
        fig, ax = plt.subplots(figsize=(15, fig_height))
        fig.patch.set_facecolor("#f6f8fb")
        ax.axis("off")

        table = ax.table(
            cellText=display_dataframe.values,
            colLabels=display_dataframe.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.65)

        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d8dee9")
            if row == 0:
                cell.set_facecolor("#1f2937")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#edf2f7")
            else:
                cell.set_facecolor("#ffffff")

        ax.set_title(
            "Características de los modelos",
            fontsize=17,
            fontweight="bold",
            color="#111827",
            pad=20,
        )
        plt.tight_layout()
        fig.savefig(self._output_dir / "model_comparison_table.png", dpi=220)
        plt.close(fig)

    def _save_accuracy_chart(self, dataframe: pd.DataFrame) -> None:
        """Saves an accuracy comparison bar chart as a PNG image."""
        sorted_dataframe = dataframe.sort_values("Accuracy (%)", ascending=True)
        colors = ["#5b8def", "#22a06b", "#f59e0b"]

        fig, ax = plt.subplots(figsize=(10, 5.8))
        fig.patch.set_facecolor("#f6f8fb")
        ax.set_facecolor("#ffffff")

        bars = ax.barh(
            sorted_dataframe["Modelo"],
            sorted_dataframe["Accuracy (%)"],
            color=colors[: len(sorted_dataframe)],
            edgecolor="#111827",
            linewidth=0.8,
        )

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#111827",
            )

        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy en test (%)", fontsize=11, color="#374151")
        ax.set_title(
            "Accuracy en el conjunto de test",
            fontsize=17,
            fontweight="bold",
            color="#111827",
            pad=16,
        )
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", colors="#374151")

        plt.tight_layout()
        fig.savefig(self._output_dir / "accuracy_comparison.png", dpi=220)
        plt.close(fig)
