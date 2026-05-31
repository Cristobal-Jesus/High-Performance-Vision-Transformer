"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/compare_all/stats.py

Description:
    Unified dataclass for storing benchmark results across all quantization
    variants (FP32, INT8, INT4, INT2, INT1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VariantStats:
    """Benchmark results for one quantization variant.

    Attributes:
        label: Short human-readable name shown on plots (e.g. ``"INT4"``).
        full_label: Longer name with method detail (e.g. ``"INT4 (W4A32)"``).
        bits: Effective bits per weight (32 for FP32, 8 for INT8, etc.).
        accuracy: Top-1 accuracy on the test set (percentage).
        size_mb: Model size in megabytes.  Actual file size for FP32 and
            INT8; theoretical packed size for INT4, INT2 and INT1.
        theoretical: ``True`` when *size_mb* is a theoretical estimate.
        device: Inference device used (``"CPU"`` or ``"GPU"``).
    """

    label: str
    full_label: str
    bits: int
    accuracy: float
    size_mb: float
    theoretical: bool = field(default=False)
    device: str = field(default="GPU")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to a 32-bit FP32 model of the same size.

        Computed as ``32 / self.bits``.  For FP32 this is 1.0.
        """
        return 32.0 / self.bits
