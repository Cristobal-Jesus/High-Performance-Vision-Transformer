"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: attention_dataset.py

Description:
    This file defines the dataset class used to load images, apply
    preprocessing transforms, and split each image into non-overlapping
    patches for attention-based models.

References:
    - https://pytorch.org/docs/stable/data.html
    - https://pytorch.org/vision/stable/index.html
"""

from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class AttnDataset(Dataset):
    """Dataset for patch-based image inputs used by attention models."""

    def __init__(
        self,
        x_paths: Sequence[str | Path],
        y: Sequence[int],
        transform: Callable | None = None,
        patch_size: int = 16,
    ) -> None:
        if len(x_paths) != len(y):
            raise ValueError("x_paths and y must have the same length.")
        if patch_size <= 0:
            raise ValueError("patch_size must be greater than 0.")

        self.x_paths = x_paths
        self.y = y
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load one image, split it into flattened patches, and return its label."""
        image = Image.open(self.x_paths[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            raise ValueError(
                "transform must convert the image to a tensor before extracting patches."
            )

        channels, height, width = image.shape
        patch_size = self.patch_size

        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "Image height and width must be divisible by patch_size."
            )

        # Split the image into non-overlapping patches.
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

        # Rearrange dimensions so each patch becomes one row.
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()

        # Flatten each patch to shape (num_patches, channels * patch_size * patch_size).
        patches = patches.view(-1, channels * patch_size * patch_size)

        label = torch.tensor(self.y[index], dtype=torch.long)
        return patches, label
