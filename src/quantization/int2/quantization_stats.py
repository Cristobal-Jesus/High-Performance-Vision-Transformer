"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int2/quantization_stats.py

Description:
    Dataclass that stores benchmark results for a single model variant
    (FP32 baseline or INT2 weight-only quantized).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantizationStats:
    """Benchmark results for one quantization variant.

    Attributes:
        label: Human-readable variant name (e.g. ``"FP32"``, ``"INT2 (W2A32)"``).
        total_params: Total number of model parameters.
        disk_size_mb: Model size in megabytes.  For FP32 this is the actual
            checkpoint size on disk; for INT2 it is the theoretical packed
            size if weights were stored at 2 bits each.
        accuracy: Top-1 accuracy on the test set (percentage).
        elapsed_sec: Total inference time in seconds.
    """

    label: str
    total_params: int
    disk_size_mb: float
    accuracy: float
    elapsed_sec: float
