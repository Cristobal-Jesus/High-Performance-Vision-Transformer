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
    Benchmarking class that measures top-1 accuracy and inference
    throughput (images/second) for a VisionTransformer model over a
    DataLoader of test images.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer.training.batch_processor import PatchBatchProcessor


class InferenceBenchmark:
    """Measure inference accuracy and throughput for a patch-based model.

    Args:
        patch_size: Size (in pixels) of each image patch.
        warmup_batches: Number of batches to discard before timing starts.
    """

    def __init__(self, patch_size: int, warmup_batches: int = 5) -> None:
        self._processor = PatchBatchProcessor(patch_size=patch_size)
        self._warmup_batches = warmup_batches

    def run(
        self,
        model: nn.Module,
        loader: DataLoader,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[float, float]:
        """Run a full inference pass over *loader*.

        Args:
            model: Model already placed on *device* and cast to *torch_dtype*.
            loader: DataLoader yielding ``(images, labels)`` pairs.
            torch_dtype: Target dtype for the input tensor cast.
            device: Device used for inference.

        Returns:
            A tuple ``(accuracy_percent, throughput)`` where *throughput*
            is measured in images per second.
        """
        self._warmup(model, loader, torch_dtype, device)

        correct = 0
        total = 0

        if device.type == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device=device, dtype=torch_dtype, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                patches = self._processor.patchify(images)
                logits = model(patches)
                preds = logits.argmax(dim=1)

                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t_start
        accuracy = 100.0 * correct / max(1, total)
        throughput = total / max(1e-9, elapsed)

        return accuracy, throughput

    def _warmup(
        self,
        model: nn.Module,
        loader: DataLoader,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Run a few batches to warm up the GPU before timing."""
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if i >= self._warmup_batches:
                    break
                images = images.to(device=device, dtype=torch_dtype, non_blocking=True)
                patches = self._processor.patchify(images)
                _ = model(patches)

        if device.type == "cuda":
            torch.cuda.synchronize()
