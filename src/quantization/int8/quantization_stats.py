"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: quantization_stats.py

Description:
    Dataclass that stores the benchmark results for a single model variant
    (FP32 baseline or INT8 dynamically quantized).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantizationStats:
    """Hold benchmark results for one quantization variant.

    Attributes:
        label: Human-readable variant label (e.g. "FP32", "INT8 (Dynamic)").
        total_params: Total number of visible parameters.
        disk_size_mb: Model size on disk in megabytes.
        accuracy: Top-1 accuracy on the test set (percentage).
        elapsed_sec: Total inference time in seconds.
    """

    label: str
    total_params: int
    disk_size_mb: float
    accuracy: float
    elapsed_sec: float
