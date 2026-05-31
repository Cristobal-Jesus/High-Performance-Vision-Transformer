"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int2/benchmark.py

Description:
    Benchmarking class that measures top-1 accuracy and total inference
    time for a VisionTransformer over a flat 20-class test directory.

    INT2 simulated quantization produces a standard FP32 model, so
    inference runs on GPU when CUDA is available, falling back to CPU
    otherwise.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from transformer.training.batch_processor import PatchBatchProcessor
from .constants import (
    EVAL_BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
    PATCH_SIZE,
)


def _label_from_filename(filename: str, label_map: dict[str, int]) -> int:
    """Infer the integer class label from a filename.

    Expects the convention ``ClassName_<index>.<ext>``.

    Args:
        filename: Bare filename without directory component.
        label_map: Mapping from class-name string to integer label.

    Returns:
        Integer label for the image.

    Raises:
        ValueError: If the inferred class name is not in *label_map*.
    """
    stem = os.path.splitext(filename)[0]
    class_name = "_".join(stem.split("_")[:-1])
    if class_name in label_map:
        return label_map[class_name]
    raise ValueError(
        f"Could not infer class from filename '{filename}'. "
        f"Expected one of: {sorted(label_map.keys())}."
    )


class _FlatLabeledDataset(Dataset):
    """Dataset that loads images from a flat directory.

    Labels are inferred from filenames using ``_label_from_filename``.

    Args:
        images_dir: Path to the flat test-image directory.
        label_map: Mapping from class-name string to integer label.
        transform: Torchvision transform applied to each loaded image.
    """

    _SUPPORTED: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        images_dir: str,
        label_map: dict[str, int],
        transform,
    ) -> None:
        self._transform = transform
        self._samples: list[tuple[str, int]] = []
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(self._SUPPORTED):
                continue
            self._samples.append(
                (
                    os.path.join(images_dir, fname),
                    _label_from_filename(fname, label_map),
                )
            )
        if not self._samples:
            raise RuntimeError(
                f"No supported images found in '{images_dir}'."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, label


class InferenceBenchmark:
    """Measure top-1 accuracy and inference time for an INT2-quantized ViT.

    Runs on CUDA when available; falls back to CPU automatically.

    Args:
        patch_size: Patch size used by ``PatchBatchProcessor``.
        batch_size: Batch size for the DataLoader.
    """

    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        batch_size: int = EVAL_BATCH_SIZE,
    ) -> None:
        self._processor = PatchBatchProcessor(patch_size=patch_size)
        self._batch_size = batch_size
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)
            ),
        ])

    def run(self, model: nn.Module, images_dir: str) -> tuple[float, float]:
        """Run a full inference pass over *images_dir*.

        Args:
            model: Model in evaluation mode (moved to ``self._device``
                internally).
            images_dir: Path to the flat test-image directory.

        Returns:
            A tuple ``(accuracy_percent, elapsed_sec)``.
        """
        loader = DataLoader(
            _FlatLabeledDataset(images_dir, LABEL_MAP, self._transform),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(self._device.type == "cuda"),
        )
        model = model.to(self._device)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                patches = self._processor.patchify(images)
                preds = model(patches).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        return 100.0 * correct / max(1, total), elapsed
