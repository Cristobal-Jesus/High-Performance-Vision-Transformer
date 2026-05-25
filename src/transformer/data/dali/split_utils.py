"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristobal Jesus Sarmiento Rodriguez
Date: 17th March 2026
File: split_utils.py

Description:
    This file defines the class responsible for scanning dataset folders,
    assigning labels, splitting samples into training and validation
    subsets, and generating temporary DALI file lists.

References:
    - https://docs.python.org/3/library/tempfile.html
"""

import os
import random
import tempfile
import typing as t
from pathlib import Path

from .validation import DaliImageValidator


class DatasetSplitManager:
    """Scan the dataset, split the samples, and generate DALI file lists."""

    def __init__(
        self,
        root_dir: str,
        validator: DaliImageValidator,
        train_ratio: float = 0.8,
        seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")

        self.root_dir = Path(root_dir)
        self.validator = validator
        self.train_ratio = train_ratio
        self.seed = seed
        self.show_progress = show_progress

    def create_split(
        self,
    ) -> t.Tuple[t.List[str], t.List[int], t.List[str], t.List[int], t.Dict[str, int], t.List[t.Tuple[str, str]]]:
        """Create the train/validation split and return metadata."""
        image_paths, labels, label_map, invalid_files = self._scan_dataset()
        x_train, y_train, x_val, y_val = self._split_samples(image_paths, labels)
        return x_train, y_train, x_val, y_val, label_map, invalid_files

    def write_file_list(
        self,
        image_paths: t.Sequence[str],
        labels: t.Sequence[int],
        suffix: str,
    ) -> str:
        """Write a temporary DALI file list and return its path."""
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length.")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            encoding="utf-8",
        ) as file:
            file.write("".join(f"{path} {label}\n" for path, label in zip(image_paths, labels)))
            return file.name

    def _scan_dataset(
        self,
    ) -> t.Tuple[t.List[str], t.List[int], t.Dict[str, int], t.List[t.Tuple[str, str]]]:
        """Scan the dataset root directory and collect image paths and labels.

        Image validation via DALI is intentionally skipped: the dataset is
        assumed to be clean.  Any corrupted files will raise an error when
        the training pipeline first tries to decode them.
        """
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        with os.scandir(self.root_dir) as it:
            class_entries = sorted(
                (entry for entry in it if entry.is_dir()),
                key=lambda entry: entry.name,
            )

        if not class_entries:
            raise RuntimeError("No class directories were found in the dataset root.")

        label_map = {entry.name: index for index, entry in enumerate(class_entries)}

        image_paths: t.List[str] = []
        labels: t.List[int] = []

        for class_entry in class_entries:
            label = label_map[class_entry.name]
            with os.scandir(class_entry.path) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        image_paths.append(entry.path)
                        labels.append(label)

        if not image_paths:
            raise RuntimeError("No image files were found in the dataset root.")

        return image_paths, labels, label_map, []

    def _split_samples(
        self,
        image_paths: t.Sequence[str],
        labels: t.Sequence[int],
    ) -> t.Tuple[t.List[str], t.List[int], t.List[str], t.List[int]]:
        """Shuffle the dataset and split it into training and validation subsets."""
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length.")
        if len(image_paths) < 2:
            raise ValueError("At least two samples are required to create a split.")

        combined = list(zip(image_paths, labels))
        random.Random(self.seed).shuffle(combined)

        train_size = int(len(combined) * self.train_ratio)
        train_size = min(max(train_size, 1), len(combined) - 1)

        train_samples = combined[:train_size]
        val_samples = combined[train_size:]

        x_train = [path for path, _ in train_samples]
        y_train = [label for _, label in train_samples]
        x_val = [path for path, _ in val_samples]
        y_val = [label for _, label in val_samples]

        return x_train, y_train, x_val, y_val