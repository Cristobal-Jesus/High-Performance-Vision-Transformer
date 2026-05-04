"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: model_metrics.py

Description:
    This file defines the coordinator object that collects metrics from
    several PyTorch model checkpoints.
"""


from __future__ import annotations

from .checkpoint_inspector import CheckpointInspector
from .data_models import ModelConfig, ModelMetrics



class ModelComparison:
    """Coordinates metric extraction for several model checkpoints."""

    def __init__(
        self,
        configs: list[ModelConfig],
        inspector: CheckpointInspector | None = None,
    ) -> None:
        """Initializes the comparison workflow.

        Args:
            configs: Model configurations to compare.
            inspector: Optional checkpoint inspector dependency.
        """
        self._configs = configs
        self._inspector = inspector or CheckpointInspector()

    def collect_metrics(self) -> list[ModelMetrics]:
        """Collects metrics for all configured models.

        Returns:
            Metrics extracted from each configured checkpoint.
        """
        return [self._inspector.inspect(config) for config in self._configs]
