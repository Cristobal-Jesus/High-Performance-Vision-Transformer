"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: model_factory.py

Description:
    Factory class responsible for loading a VisionTransformer checkpoint
    and casting it to a requested numeric precision (FP32, FP16 or BF16).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from transformer.models.transformer_model import VisionTransformer
from transformer.training.config import TransformerTrainingConfig
from .constants import CHECKPOINT_PATH


class ModelFactory:
    """Load and cast VisionTransformer checkpoints to a target dtype."""

    _DTYPE_MAP: dict[str, torch.dtype] = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(
        self,
        config: TransformerTrainingConfig,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        )

    def load(self, dtype: str, device: torch.device) -> nn.Module:
        """Return a VisionTransformer cast to *dtype* with loaded weights.

        Args:
            dtype: One of ``"fp32"``, ``"fp16"``, or ``"bf16"``.
            device: Target device (e.g. ``torch.device("cuda")``).

        Returns:
            The model in evaluation mode with the requested dtype.

        Raises:
            KeyError: If *dtype* is not one of the supported keys.
        """
        if dtype not in self._DTYPE_MAP:
            raise KeyError(
                f"Unsupported dtype '{dtype}'. "
                f"Choose one of: {sorted(self._DTYPE_MAP.keys())}."
            )

        cfg = self._config

        model = VisionTransformer(
            patch_dim=cfg.patch_dim,
            seq_len=cfg.seq_len,
            embed_dim=cfg.embed_dim,
            num_classes=cfg.num_classes,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=0.0,
            drop_path_rate=0.0,
        )

        state_dict = torch.load(self._checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        torch_dtype = self._DTYPE_MAP[dtype]
        model = model.to(device=device, dtype=torch_dtype)
        model.eval()

        return model

    @staticmethod
    def compute_stats(model: nn.Module) -> tuple[int, float]:
        """Return the parameter count and memory footprint of *model*.

        Args:
            model: Any ``nn.Module``.

        Returns:
            A tuple ``(total_params, size_mb)`` where *size_mb* reflects
            the actual dtype of the stored parameters.
        """
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            / (1024 ** 2)
        )
        return total_params, size_mb
