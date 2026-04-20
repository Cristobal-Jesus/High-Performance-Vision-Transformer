"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 20th April 2026
File: validation.py

Description:
    This file defines the class responsible for validating whether an
    image file can be safely processed by NVIDIA DALI.

References:
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

from __future__ import annotations

import os
import typing as t
from pathlib import Path

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def


@pipeline_def
def _dali_validation_pipeline():
    encoded = fn.external_source(name="encoded", device="cpu")
    images = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB,
    )
    return images


class DaliImageValidator:
    """Validate whether an image file can be processed by NVIDIA DALI."""

    DEFAULT_SUPPORTED_FORMATS = {
        "JPEG",
        "PNG",
        "BMP",
        "TIFF",
        "PPM",
        "PGM",
        "PBM",
        "WEBP",
        "JPEG2000",
    }

    _SUFFIX_TO_FORMAT = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".ppm": "PPM",
        ".pgm": "PGM",
        ".pbm": "PBM",
        ".webp": "WEBP",
        ".jp2": "JPEG2000",
        ".j2k": "JPEG2000",
        ".j2c": "JPEG2000",
        ".jpc": "JPEG2000",
    }

    def __init__(
        self,
        supported_formats: t.Optional[t.Iterable[str]] = None,
        num_threads: t.Optional[int] = None,
        device_id: t.Optional[int] = None,
        delete_invalid: bool = True,
        batch_size: int = 256,
    ) -> None:
        self.supported_formats = set(supported_formats or self.DEFAULT_SUPPORTED_FORMATS)

        cpu_count = os.cpu_count() or 4
        self.num_threads = num_threads or min(8, max(2, cpu_count // 2))
        self.device_id = 0 if device_id is None else int(device_id)
        self.delete_invalid = delete_invalid
        self.batch_size = max(1, int(batch_size))

        self._pipe = _dali_validation_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            exec_pipelined=True,
            exec_async=True,
            prefetch_queue_depth=4,
        )
        self._pipe.build()

        self._single_pipe = _dali_validation_pipeline(
            batch_size=1,
            num_threads=max(1, min(2, self.num_threads)),
            device_id=self.device_id,
            exec_pipelined=True,
            exec_async=True,
            prefetch_queue_depth=1,
        )
        self._single_pipe.build()

    def _guess_format_from_path(self, image_path: Path) -> str:
        return self._SUFFIX_TO_FORMAT.get(image_path.suffix.lower(), "UNKNOWN")

    def _delete_file(self, image_path: Path) -> str:
        if not self.delete_invalid:
            return ""

        try:
            if image_path.exists() and image_path.is_file():
                image_path.unlink()
                return " [DELETED]"
        except Exception as exc:
            return f" [DELETE FAILED: {exc}]"

        return ""

    def _validate_single_path(self, image_path: Path) -> t.Tuple[bool, str]:
        if not image_path.is_file():
            return False, "Invalid or corrupted image: file does not exist"

        image_format = self._guess_format_from_path(image_path)
        if image_format not in self.supported_formats:
            delete_msg = self._delete_file(image_path)
            return False, f"Unsupported DALI format: {image_format}{delete_msg}"

        try:
            with image_path.open("rb") as file:
                encoded = file.read()

            self._single_pipe.feed_input("encoded", [encoded])
            self._single_pipe.run()

            return True, image_format

        except Exception as exc:
            delete_msg = self._delete_file(image_path)
            return False, f"Invalid or corrupted image: {exc}{delete_msg}"

    def validate(self, path: t.Union[str, Path]) -> t.Tuple[bool, str]:
        """Return whether the image is valid for DALI and an explanatory message."""
        return self._validate_single_path(Path(path))

    def validate_many(
        self,
        paths: t.Iterable[t.Union[str, Path]],
    ) -> t.Iterator[t.Tuple[str, t.Tuple[bool, str]]]:
        """
        Validate many image paths in batches.

        It tries to decode a whole batch with DALI first. If the batch fails,
        it falls back to validating each file in that batch individually.
        """
        batch_items: t.List[t.Tuple[Path, str, bytes]] = []

        def flush_batch() -> t.Iterator[t.Tuple[str, t.Tuple[bool, str]]]:
            if not batch_items:
                return

            batch_paths = [item[0] for item in batch_items]
            batch_formats = [item[1] for item in batch_items]
            batch_encoded = [item[2] for item in batch_items]

            try:
                self._pipe.feed_input("encoded", batch_encoded)
                self._pipe.run()

                for image_path, image_format in zip(batch_paths, batch_formats):
                    yield str(image_path), (True, image_format)

            except Exception:
                for image_path in batch_paths:
                    yield str(image_path), self._validate_single_path(image_path)

        for raw_path in paths:
            image_path = Path(raw_path)

            if not image_path.is_file():
                yield str(image_path), (
                    False,
                    "Invalid or corrupted image: file does not exist",
                )
                continue

            image_format = self._guess_format_from_path(image_path)
            if image_format not in self.supported_formats:
                delete_msg = self._delete_file(image_path)
                yield str(image_path), (
                    False,
                    f"Unsupported DALI format: {image_format}{delete_msg}",
                )
                continue

            try:
                with image_path.open("rb") as file:
                    encoded = file.read()
            except Exception as exc:
                delete_msg = self._delete_file(image_path)
                yield str(image_path), (
                    False,
                    f"Invalid or corrupted image: {exc}{delete_msg}",
                )
                continue

            batch_items.append((image_path, image_format, encoded))

            if len(batch_items) >= self.batch_size:
                yield from flush_batch()
                batch_items.clear()

        if batch_items:
            yield from flush_batch()
