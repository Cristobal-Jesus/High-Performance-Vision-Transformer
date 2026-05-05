"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: compare_precision.py

Description:
    Entry point for the FP32 / FP16 / BF16 precision comparison benchmark.

Usage:
    python -m src.quantization.16bits.compare_precision
"""

from __future__ import annotations

import torch

from .comparator import PrecisionComparator
from transformer.training.config import TransformerTrainingConfig


def main() -> None:
    """Run the precision comparison and save the comparison figure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = TransformerTrainingConfig()

    comparator = PrecisionComparator(
        device=device,
        test_dir="/home/bejeque/nhernang/Cristobal/pytorch_models/assets/Test",
        config=config,
        checkpoint_path=config.checkpoint_path,
        batch_size=128,
        output_path="outputs/figures/quantization/precision_comparison.png",
    )
    comparator.compare()


if __name__ == "__main__":
    main()
