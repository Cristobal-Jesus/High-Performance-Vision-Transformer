"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/compare_all/compare_all.py

Description:
    Entry point for the combined quantization comparison.
    Benchmarks FP32, INT8, INT4, INT2 and INT1 in a single run and
    saves a publication-quality three-panel comparison figure.

Usage:
    python -m quantization.compare_all.compare_all
"""

from __future__ import annotations

from pathlib import Path

from transformer.training.config import TransformerTrainingConfig
from .comparator import AllQuantizationComparator
from .plotter import AllQuantizationPlotter


_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../High-Performance-Vision-Transformer


def main() -> None:
    """Run the full quantization comparison and save the figure."""
    config = TransformerTrainingConfig()

    comparator = AllQuantizationComparator(
        test_dir="/home/almeida/Cristobal/Test",
        config=config,
        checkpoint_path=_PROJECT_ROOT / config.checkpoint_path,
        batch_size=64,
    )

    all_stats = comparator.run()

    AllQuantizationPlotter().plot(
        stats=all_stats,
        output_path=_PROJECT_ROOT / "outputs/figures/quantization/all_quantization_comparison.png",
    )


if __name__ == "__main__":
    main()
