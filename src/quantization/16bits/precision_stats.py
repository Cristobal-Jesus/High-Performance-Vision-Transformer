"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: precision_stats.py

Description:
    Dataclass that stores the benchmark results for a single numeric
    precision variant (FP32, FP16 or BF16).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrecisionStats:
    """Hold inference benchmark results for one dtype configuration.

    Attributes:
        label: Human-readable dtype label (e.g. "FP32").
        total_params: Total number of trainable parameters.
        size_mb: Model size in megabytes.
        accuracy: Top-1 accuracy on the test set (percentage).
        throughput: Inference throughput in images per second.
    """

    label: str
    total_params: int
    size_mb: float
    accuracy: float
    throughput: float
