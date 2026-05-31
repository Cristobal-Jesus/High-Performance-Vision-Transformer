"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int2/compare_quantization.py

Description:
    Entry point for the INT2 weight-only quantization comparison.

Usage:
    python -m quantization.int2.compare_quantization
"""

from __future__ import annotations

from transformer.training.config import TransformerTrainingConfig
from .comparator import QuantizationComparator


def main() -> None:
    """Run the FP32 vs INT2 benchmark and save the comparison figure."""
    config = TransformerTrainingConfig()

    comparator = QuantizationComparator(
        test_dir="/home/almeida/Cristobal/Test",
        config=config,
        checkpoint_path=config.checkpoint_path,
        batch_size=64,
        output_path="outputs/figures/quantization/quantization_int2_comparison.png",
    )
    comparator.compare()


if __name__ == "__main__":
    main()
