"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: split_utils.py

Description:
    This file defines the class responsible for scanning dataset folders,
    assigning labels, splitting samples into training and validation
    subsets, and generating temporary DALI file lists.

References:
    - https://docs.python.org/3/library/tempfile.html
"""

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
    ) -> None:
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1.")

        self.root_dir = Path(root_dir)
        self.validator = validator
        self.train_ratio = train_ratio
        self.seed = seed

    def create_split(
        self,
    ) -> t.Tuple[
        t.List[str],
        t.List[int],
        t.List[str],
        t.List[int],
        t.Dict[str, int],
        t.List[t.Tuple[str, str]],
    ]:
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
            for path, label in zip(image_paths, labels):
                file.write(f"{path} {label}\n")

            return file.name

    def _scan_dataset(
        self,
    ) -> t.Tuple[t.List[str], t.List[int], t.Dict[str, int], t.List[t.Tuple[str, str]]]:
        """Scan the dataset root directory and collect valid image paths and labels."""
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        class_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        if not class_dirs:
            raise RuntimeError("No class directories were found in the dataset root.")

        label_map = {class_dir.name: index for index, class_dir in enumerate(class_dirs)}

        image_paths: t.List[str] = []
        labels: t.List[int] = []
        invalid_files: t.List[t.Tuple[str, str]] = []

        candidates: t.List[t.Tuple[str, int]] = []

        for class_dir in class_dirs:
            label = label_map[class_dir.name]

            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file():
                    continue

                candidates.append((str(image_path), label))

        if not candidates:
            raise RuntimeError("No image files were found in the dataset root.")

        path_to_label = {path: label for path, label in candidates}

        for path, (is_valid, info) in self.validator.validate_many(
            path for path, _ in candidates
        ):
            if is_valid:
                image_paths.append(path)
                labels.append(path_to_label[path])
            else:
                invalid_files.append((path, info))

        if not image_paths:
            raise RuntimeError("No valid images were found for DALI.")

        return image_paths, labels, label_map, invalid_files

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
