"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: checkpoint_inspector.py

Description:
    This file defines the object responsible for loading PyTorch checkpoints
    and extracting comparable model metrics.
"""

from __future__ import annotations

from typing import Any

import torch

from .data_models import ModelConfig, ModelMetrics


class CheckpointInspector:
    """Loads PyTorch checkpoints and extracts summary metrics."""

    _STATE_DICT_KEYS = (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
    )

    def inspect(self, config: ModelConfig) -> ModelMetrics:
        """Loads a checkpoint and returns comparable metrics.

        Args:
            config: Model configuration with name, type, path and accuracy.

        Returns:
            Metrics extracted from the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            RuntimeError: If PyTorch cannot read the checkpoint.
        """
        checkpoint_path = config.checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state = self._extract_state_dict(checkpoint)

        tensor_values = [value for value in state.values() if torch.is_tensor(value)]
        total_parameters = sum(tensor.numel() for tensor in tensor_values)
        dtype_summary = self._summarize_dtypes(tensor_values)

        return ModelMetrics(
            name=config.name,
            checkpoint_path=checkpoint_path,
            accuracy=config.accuracy,
            file_size_mb=checkpoint_path.stat().st_size / (1024**2),
            total_parameters=total_parameters,
            tensor_count=len(tensor_values),
            dtype_summary=dtype_summary,
        )

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        """Extracts a tensor dictionary from common PyTorch checkpoint formats."""
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()

        if isinstance(checkpoint, dict):
            if self._looks_like_state_dict(checkpoint):
                return checkpoint

            for key in self._STATE_DICT_KEYS:
                value = checkpoint.get(key)
                if isinstance(value, torch.nn.Module):
                    return value.state_dict()
                if isinstance(value, dict) and self._looks_like_state_dict(value):
                    return value

        raise RuntimeError(
            "No se pudo encontrar un state_dict válido dentro del checkpoint."
        )

    def _count_trainable_parameters(self, checkpoint: Any) -> int | None:
        """Counts trainable parameters when a full torch module is available."""
        if isinstance(checkpoint, torch.nn.Module):
            return sum(
                parameter.numel()
                for parameter in checkpoint.parameters()
                if parameter.requires_grad
            )

        if isinstance(checkpoint, dict):
            for key in self._STATE_DICT_KEYS:
                value = checkpoint.get(key)
                if isinstance(value, torch.nn.Module):
                    return sum(
                        parameter.numel()
                        for parameter in value.parameters()
                        if parameter.requires_grad
                    )

        return None

    def _looks_like_state_dict(self, value: dict[Any, Any]) -> bool:
        """Checks whether a dictionary is probably a PyTorch state_dict."""
        return bool(value) and all(torch.is_tensor(item) for item in value.values())

    def _summarize_dtypes(self, tensors: list[torch.Tensor]) -> str:
        """Builds a compact dtype summary for display."""
        dtype_counts: dict[str, int] = {}
        for tensor in tensors:
            dtype_name = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        return ", ".join(
            f"{dtype_name} ({count})"
            for dtype_name, count in sorted(dtype_counts.items())
        )
