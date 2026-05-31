"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int2/model_factory.py

Description:
    Factory responsible for loading the FP32 VisionTransformer, applying
    INT2 weight-only quantization and computing model statistics.

    INT2 quantization uses symmetric per-tensor quantization on every
    ``nn.Linear`` weight matrix.  Weights are mapped to the signed 2-bit
    range [-2, 1] using a per-tensor scale factor, then immediately
    dequantized back to FP32.  This *simulated quantization* accurately
    measures the accuracy impact without requiring specialised hardware.

    Because the result is a standard FP32 model, inference runs on GPU.
    The reported size is the theoretical packed size (2 bits per weight
    value) rather than the size of the saved FP32 file.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn

from transformer.models.transformer_model import VisionTransformer
from transformer.training.config import TransformerTrainingConfig
from .constants import CHECKPOINT_PATH


class ModelFactory:
    """Load, INT2-quantize and inspect VisionTransformer checkpoints.

    Args:
        config: Training configuration used to reconstruct the architecture.
        checkpoint_path: Path to the FP32 (or EMA) model checkpoint.
    """

    # 2-bit signed symmetric range
    _INT2_MAX: int = 1
    _INT2_MIN: int = -2

    def __init__(
        self,
        config: TransformerTrainingConfig,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
    ) -> None:
        self._config = config
        self._checkpoint_path = Path(checkpoint_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_fp32(self) -> nn.Module:
        """Return the FP32 VisionTransformer with loaded weights on CPU.

        Strips the ``_orig_mod.`` prefix that ``torch.compile`` adds to
        state-dict keys before loading.

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
        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize_int2(self, model_fp32: nn.Module) -> nn.Module:
        """Return a deep copy of *model_fp32* with INT2-simulated weights.

        Every ``nn.Linear`` weight tensor is quantized to the signed 2-bit
        range [-2, 1] via symmetric per-tensor scaling, then dequantized
        back to FP32.  All other parameters are left unchanged.

        Args:
            model_fp32: FP32 model in evaluation mode.

        Returns:
            New FP32 model whose Linear weights carry only 2-bit information.
        """
        model_q = copy.deepcopy(model_fp32)
        model_q.eval()
        with torch.no_grad():
            for module in model_q.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = self._quantize_tensor(
                        module.weight.data
                    )
        return model_q

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def theoretical_size_mb(self, model: nn.Module) -> float:
        """Compute the theoretical packed size at 2 bits per weight value.

        ``nn.Linear`` weight matrices are counted at 2 bits each; all other
        parameters remain at 32 bits.

        Args:
            model: Model whose parameter shapes are inspected.

        Returns:
            Theoretical model size in megabytes.
        """
        weight_bits = 0
        other_bytes = 0
        for name, param in model.named_parameters():
            if "weight" in name and param.ndim >= 2:
                weight_bits += param.numel() * 2
            else:
                other_bytes += param.numel() * 4
        return ((weight_bits + 7) // 8 + other_bytes) / (1024 ** 2)

    @staticmethod
    def compute_param_stats(model: nn.Module) -> tuple[int, float]:
        """Return parameter count and in-memory footprint (FP32 assumed).

        Args:
            model: Any ``nn.Module``.

        Returns:
            A tuple ``(total_params, param_mem_mb)``.
        """
        total = sum(p.numel() for p in model.parameters())
        mem_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            / (1024 ** 2)
        )
        return total, mem_mb

    @staticmethod
    def disk_size_mb(path: str | Path) -> float:
        """Return the actual file size on disk in megabytes.

        Args:
            path: Path to the saved model file.

        Returns:
            File size in megabytes.
        """
        return os.path.getsize(path) / (1024 ** 2)

    @property
    def checkpoint_path(self) -> Path:
        """Path to the FP32 checkpoint."""
        return self._checkpoint_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize *tensor* to INT2 range then dequantize to FP32.

        Args:
            tensor: Weight tensor of any shape.

        Returns:
            Dequantized tensor with the same shape and dtype as *tensor*.
        """
        abs_max = tensor.abs().max()
        if abs_max == 0:
            return tensor
        scale = abs_max / self._INT2_MAX
        quantized = torch.clamp(
            torch.round(tensor / scale), self._INT2_MIN, self._INT2_MAX
        )
        return (quantized * scale).to(tensor.dtype)
