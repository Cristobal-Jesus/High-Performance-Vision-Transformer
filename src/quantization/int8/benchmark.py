"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: benchmark.py

Description:
    Benchmarking class that measures top-1 accuracy and total inference
    time for a VisionTransformer over a flat 20-class test directory.
    Runs on CPU because INT8 dynamic quantization is CPU-only.
"""

from __future__ import annotations

import os
import time

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from transformer.training.batch_processor import PatchBatchProcessor
from .constants import EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABEL_MAP, PATCH_SIZE


def _label_from_filename(filename: str, label_map: dict[str, int]) -> int:
    """Infer the integer class label from a filename.

    The expected convention is ``ClassName_<index>.<ext>`` where the class
    name may itself contain underscores (e.g. ``Anade_azulon_001.jpg``).
    The last token is treated as the numeric index and stripped.

    Args:
        filename: Bare filename (no directory component).
        label_map: Mapping from class name string to integer label.

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
    """Load images from a flat directory with labels encoded in filenames.

    Args:
        images_dir: Path to the flat test image directory.
        label_map: Mapping from class name string to integer label.
        transform: Torchvision transform applied to each image.
    """

    _SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        images_dir: str | Path,
        label_map: dict[str, int],
        transform,
    ) -> None:
        self._transform = transform
        self._samples: list[tuple[str, int]] = []

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(self._SUPPORTED_EXTENSIONS):
                continue
            label = _label_from_filename(fname, label_map)
            self._samples.append((os.path.join(str(images_dir), fname), label))

        if not self._samples:
            raise RuntimeError(f"No supported images found in '{images_dir}'.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, label


class InferenceBenchmark:
    """Measure top-1 accuracy and inference time for a patch-based model on CPU.

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
        self._transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(IMAGENET_MEAN),
                    std=list(IMAGENET_STD),
                ),
            ]
        )

    def run(
        self,
        model: nn.Module,
        images_dir: str,
    ) -> tuple[float, float]:
        """Run a full inference pass over *images_dir* on CPU.

        Args:
            model: Model in evaluation mode placed on CPU.
            images_dir: Path to the flat test image directory.

        Returns:
            A tuple ``(accuracy_percent, elapsed_sec)``.
        """
        loader = self._build_loader(images_dir)
        # torch.quantization.quantize_dynamic produce modelos qint8 que solo
        # funcionan en CPU. No se puede mover a CUDA — limitación de PyTorch.
        device = torch.device("cpu")

        correct = 0
        total = 0
        t_start = time.time()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                patches = self._processor.patchify(images)
                logits = model(patches)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        elapsed = time.time() - t_start
        accuracy = 100.0 * correct / max(1, total)
        return accuracy, elapsed

    def _build_loader(self, images_dir: str) -> DataLoader:
        """Build a DataLoader from a flat 20-class test directory."""
        dataset = _FlatLabeledDataset(
            images_dir=images_dir,
            label_map=LABEL_MAP,
            transform=self._transform,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=0,
        )
