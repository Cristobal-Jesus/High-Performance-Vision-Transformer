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
    Factory class responsible for loading the FP32 VisionTransformer,
    applying dynamic INT8 quantization, saving checkpoints and computing
    model statistics.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.quantization
from torch import nn

from transformer.models.transformer_model import VisionTransformer
from transformer.training.config import TransformerTrainingConfig
from .constants import CHECKPOINT_PATH, INT8_CHECKPOINT_PATH


class ModelFactory:
    """Load, quantize and persist VisionTransformer checkpoints.

    Args:
        config: Training config used to reconstruct the model architecture.
        checkpoint_path: Path to the FP32 model checkpoint.
        int8_checkpoint_path: Destination path for the quantized model.
    """

    def __init__(
        self,
        config: TransformerTrainingConfig,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        int8_checkpoint_path: str | Path = INT8_CHECKPOINT_PATH,
    ) -> None:
        self._config = config
        self._checkpoint_path = Path(checkpoint_path)
        self._int8_checkpoint_path = Path(int8_checkpoint_path)

    def load_fp32(self) -> nn.Module:
        """Return the FP32 VisionTransformer with loaded weights on CPU.

        Returns:
            The model placed on CPU in evaluation mode.
        """
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

        state_dict = torch.load(
            self._checkpoint_path, map_location="cpu", weights_only=True
        )
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def quantize_int8(self, model_fp32: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization to all ``nn.Linear`` layers.

        Dynamic quantization converts weights to INT8 at load time and
        quantizes activations on-the-fly during each forward pass.
        No calibration dataset is required.

        Args:
            model_fp32: FP32 model in evaluation mode.

        Returns:
            Quantized INT8 model on CPU.
        """
        model_fp32.eval()
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return model_int8

    def save_int8(self, model_int8: nn.Module) -> Path:
        """Persist the full INT8 model object to disk.

        Quantized models must be saved with ``torch.save(model, ...)``
        rather than ``state_dict`` because their internal structure
        differs from a standard ``nn.Module``.

        Args:
            model_int8: Quantized model to persist.

        Returns:
            Path where the model was saved.
        """
        self._int8_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_int8, self._int8_checkpoint_path)
        return self._int8_checkpoint_path

    def load_int8(self) -> nn.Module:
        """Reload a previously saved INT8 model from disk.

        Returns:
            The INT8 model in evaluation mode.
        """
        model = torch.load(
            self._int8_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.eval()
        return model

    @staticmethod
    def compute_param_stats(model: nn.Module) -> tuple[int, float]:
        """Return parameter count and in-memory footprint.

        Args:
            model: Any ``nn.Module``.

        Returns:
            A tuple ``(total_params, param_mem_mb)``.
        """
        total_params = sum(p.numel() for p in model.parameters())
        param_mem_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            / (1024 ** 2)
        )
        return total_params, param_mem_mb

    @staticmethod
    def disk_size_mb(path: str | Path) -> float:
        """Return the file size on disk in megabytes.

        Args:
            path: Path to the saved model file.

        Returns:
            File size in megabytes.
        """
        return os.path.getsize(path) / (1024 ** 2)
    

    @property
    def checkpoint_path(self) -> Path:
        """Return the path to the FP32 checkpoint."""
        return self._checkpoint_path
