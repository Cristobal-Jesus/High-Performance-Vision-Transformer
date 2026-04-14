"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: batch_processor.py

Description:
    This file defines the class responsible for unpacking DALI batches and
    converting images into patch sequences for the Transformer model.
"""

import torch
from torch import Tensor


class PatchBatchProcessor:
    """Prepare DALI batches for a patch-based Transformer."""

    def __init__(self, patch_size: int = 16) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than 0.")

        self.patch_size = patch_size

    def prepare_batch(
        self,
        data,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Extract, move, and patchify one DALI batch."""
        images, labels = self.unpack_dali_batch(data)
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze(-1).long().to(device, non_blocking=True)
        patches = self.patchify(images)
        return patches, labels

    @staticmethod
    def unpack_dali_batch(data) -> tuple[Tensor, Tensor]:
        """Extract images and labels from one DALI iterator item."""
        batch = data[0]
        image_key = "images" if "images" in batch else "x"
        label_key = "labels" if "labels" in batch else "y"
        return batch[image_key], batch[label_key]

    def patchify(self, images: Tensor) -> Tensor:
        """Convert a batch of images into a sequence of flattened patches."""
        batch_size, channels, height, width = images.shape

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Image size ({height}x{width}) is not divisible by patch_size={self.patch_size}."
            )

        patches_h = height // self.patch_size
        patches_w = width // self.patch_size

        images = images.unfold(2, self.patch_size, self.patch_size)
        images = images.unfold(3, self.patch_size, self.patch_size)
        images = images.permute(0, 2, 3, 1, 4, 5).contiguous()

        return images.view(
            batch_size,
            patches_h * patches_w,
            channels * self.patch_size * self.patch_size,
        )
