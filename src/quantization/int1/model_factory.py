"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int1/model_factory.py

Description:
    Factory responsible for loading the FP32 VisionTransformer, applying
    INT1 binary weight quantization and computing model statistics.

    INT1 uses XNOR-Net style binary quantization on every ``nn.Linear``
    weight matrix.  Each weight is replaced by its sign multiplied by the
    per-tensor mean absolute value (alpha), which minimises the L2
    distance between the full-precision and binary weight matrix:

        W_b = sign(W) * alpha,  where alpha = mean(|W|)

    Weights are then dequantized back to FP32 for inference.  This
    *simulated quantization* accurately measures the accuracy impact of
    extreme 1-bit compression without requiring specialised hardware.

    The reported size is the theoretical packed size (1 bit per weight
    value plus one FP32 scale factor per layer).

References:
    Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary
    Convolutional Neural Networks", ECCV 2016.
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
    """Load, INT1-quantize and inspect VisionTransformer checkpoints.

    Args:
        config: Training configuration used to reconstruct the architecture.
        checkpoint_path: Path to the FP32 (or EMA) model checkpoint.
    """

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

    def quantize_int1(self, model_fp32: nn.Module) -> nn.Module:
        """Return a deep copy of *model_fp32* with binary-quantized weights.

        Every ``nn.Linear`` weight tensor is binarized using the XNOR-Net
        formula ``W_b = sign(W) * mean(|W|)``, then stored as FP32.  All
        other parameters are left unchanged.

        Args:
            model_fp32: FP32 model in evaluation mode.

        Returns:
            New FP32 model whose Linear weights carry only 1-bit information.
        """
        model_q = copy.deepcopy(model_fp32)
        model_q.eval()
        with torch.no_grad():
            for module in model_q.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = self._binarize_tensor(
                        module.weight.data
                    )
        return model_q

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def theoretical_size_mb(self, model: nn.Module) -> float:
        """Compute the theoretical packed size at 1 bit per weight value.

        ``nn.Linear`` weight matrices are counted at 1 bit each plus one
        FP32 scale factor (alpha) per layer; all other parameters remain
        at 32 bits.

        Args:
            model: Model whose parameter shapes are inspected.

        Returns:
            Theoretical model size in megabytes.
        """
        weight_bits = 0
        scale_bytes = 0
        other_bytes = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_bits += module.weight.numel() * 1
                scale_bytes += 4  # one FP32 alpha per layer
        for name, param in model.named_parameters():
            if not ("weight" in name and param.ndim >= 2):
                other_bytes += param.numel() * 4
        return (
            (weight_bits + 7) // 8 + scale_bytes + other_bytes
        ) / (1024 ** 2)

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

    @staticmethod
    def _binarize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Binarize *tensor* using XNOR-Net scaling.

        Computes ``W_b = sign(W) * mean(|W|)``.  The scale factor alpha
        minimises the Frobenius norm ``||W - alpha * sign(W)||_F``.

        Args:
            tensor: Weight tensor of any shape.

        Returns:
            Binarized tensor with the same shape and dtype as *tensor*.
        """
        alpha = tensor.abs().mean()
        if alpha == 0:
            return tensor
        return (tensor.sign() * alpha).to(tensor.dtype)
