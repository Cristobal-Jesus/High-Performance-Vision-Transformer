"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: comparator.py

Description:
    Orchestrator class that benchmarks a VisionTransformer across FP32,
    FP16 and BF16 numeric precisions. Images are loaded from a flat test
    directory where the class label is encoded in each filename
    (e.g. ``Anade_azulon_001.jpg``). A comparison figure is saved at the
    end of the run.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from transformer.training.config import TransformerTrainingConfig
from .benchmark import InferenceBenchmark
from .constants import (
    CHECKPOINT_PATH,
    DTYPE_LABEL,
    EVAL_BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_MAP,
)
from .model_factory import ModelFactory
from .plotter import ComparisonPlotter
from .precision_stats import PrecisionStats


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
        images_dir: Path to the flat image directory.
        label_map: Mapping from class name string to integer label.
        transform: Optional torchvision transform applied to each image.
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


class PrecisionComparator:
    """Benchmark FP32, FP16 and BF16 inference and generate a comparison report.

    Args:
        device: Target device for inference.
        test_dir: Path to a flat directory of labelled test images.
        config: Training config used to reconstruct the model architecture.
                Defaults to ``TransformerTrainingConfig()``.
        checkpoint_path: Path to the saved model checkpoint.
                         Defaults to ``constants.CHECKPOINT_PATH``.
        batch_size: Batch size used for the DataLoader.
        output_path: Destination path for the saved comparison figure.
    """

    _DTYPE_KEYS: tuple[str, ...] = ("fp32", "fp16", "bf16")

    _DTYPE_MAP: dict[str, torch.dtype] = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(
        self,
        device: torch.device,
        test_dir: str | Path,
        config: TransformerTrainingConfig = None,
        checkpoint_path: str | Path = None,
        batch_size: int = EVAL_BATCH_SIZE,
        output_path: str | Path = "outputs/figures/quantization/precision_comparison.png",
    ) -> None:
        self._device = device
        self._test_dir = Path(test_dir)
        self._config = config or TransformerTrainingConfig()
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
        )
        self._batch_size = batch_size
        self._output_path = Path(output_path)

    def run(self) -> list[PrecisionStats]:
        """Benchmark all dtype variants and return the collected statistics."""
        loader = self._build_loader()
        factory = ModelFactory(self._config, self._checkpoint_path)
        benchmark = InferenceBenchmark(patch_size=self._config.patch_size)

        all_stats = []

        for key in self._DTYPE_KEYS:
            label = DTYPE_LABEL[key]
            torch_dtype = self._DTYPE_MAP[key]
            print(f"\n--- Benchmarking {label} ---")

            model = factory.load(key, self._device)
            total_params, size_mb = ModelFactory.compute_stats(model)
            accuracy, throughput = benchmark.run(model, loader, torch_dtype, self._device)

            stats = PrecisionStats(
                label=label,
                total_params=total_params,
                size_mb=size_mb,
                accuracy=accuracy,
                throughput=throughput,
            )
            all_stats.append(stats)

            print(
                f"  Accuracy   : {accuracy:.2f}%\n"
                f"  Model size : {size_mb:.1f} MB\n"
                f"  Throughput : {throughput:.0f} img/s\n"
                f"  Params     : {total_params:,}"
            )

            del model
            torch.cuda.empty_cache()

        return all_stats

    def compare(self) -> None:
        """Run all benchmarks and save the comparison figure to disk."""
        all_stats = self.run()
        ComparisonPlotter().plot(all_stats, self._output_path)

    def _build_loader(self) -> DataLoader:
        """Build a DataLoader over the flat test directory."""
        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(IMAGENET_MEAN),
                    std=list(IMAGENET_STD),
                ),
            ]
        )

        dataset = _FlatLabeledDataset(
            images_dir=self._test_dir,
            label_map=LABEL_MAP,
            transform=transform,
        )

        print(f"Test set: {len(dataset)} images from '{self._test_dir}'")

        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=self._device.type == "cuda",
            drop_last=False,
        )
