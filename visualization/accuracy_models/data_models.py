"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: data_models.py

Description:
    This file defines the data objects used to configure and summarize
    PyTorch model comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Configuration needed to describe and load a model checkpoint."""

    name: str
    checkpoint_path: Path
    accuracy: float


@dataclass(frozen=True)
class ModelMetrics:
    """Comparable metrics extracted from a model checkpoint."""

    name: str
    checkpoint_path: Path
    accuracy: float
    file_size_mb: float
    total_parameters: int
    tensor_count: int
    dtype_summary: str
