"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristobal Jesus Sarmiento Rodriguez
Date: 20th April 2026
File: validation.py

Description:
    This file defines the class responsible for validating whether an
    image file can be safely processed by NVIDIA DALI.
"""

from __future__ import annotations

import os
import typing as t
from pathlib import Path

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def


@pipeline_def
def _dali_single_image_validation_pipeline():
    """Reusable DALI pipeline that decodes one encoded image at a time."""
    encoded = fn.external_source(
        name="encoded",
        device="cpu",
        dtype=types.UINT8,
    )

    decoded = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB,
        hw_decoder_load=0.9,
    )

    return decoded


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
        batch_size: int = 128,
        progress_callback: t.Optional[t.Callable[[int], None]] = None,
    ) -> None:
        self.supported_formats = set(supported_formats or self.DEFAULT_SUPPORTED_FORMATS)

        cpu_count = os.cpu_count() or 4
        self.num_threads = num_threads or max(4, cpu_count)
        self.device_id = 0 if device_id is None else int(device_id)
        self.delete_invalid = delete_invalid
        self.batch_size = max(1, int(batch_size))  # kept for compatibility
        self.progress_callback = progress_callback

        self._validation_pipe = None

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

    def _detect_unsupported_signature(self, image_path: Path) -> t.Optional[str]:
        try:
            with image_path.open("rb") as f:
                header = f.read(12)
        except Exception:
            return None

        if header.startswith((b"GIF87a", b"GIF89a")):
            return "GIF"

        return None

    def _summarize_exception(self, exc: Exception) -> str:
        text = str(exc)

        if "GIF images are not supported" in text:
            return "GIF images are not supported"

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if line in {
                "Critical error in pipeline:",
                "Current pipeline object is no longer valid.",
                ". File:",
            }:
                continue

            if line.startswith("Error when executing "):
                continue

            if line.startswith("Stacktrace") or line.startswith("[frame"):
                continue

            if "] " in line:
                line = line.rsplit("] ", 1)[-1]

            return line

        return exc.__class__.__name__

    def _get_validation_pipe(self):
        if self._validation_pipe is None:
            self._validation_pipe = _dali_single_image_validation_pipeline(
                batch_size=1,
                num_threads=self.num_threads,
                device_id=self.device_id,
                exec_pipelined=False,
                exec_async=False,
                prefetch_queue_depth=1,
            )
            self._validation_pipe.build()

        return self._validation_pipe

    def _drop_validation_pipe(self) -> None:
        pipe = self._validation_pipe
        self._validation_pipe = None

        if pipe is not None:
            try:
                pipe.reset()
            except Exception:
                pass

    def _run_pipeline_for_path(self, image_path: Path) -> None:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError("empty file")

        pipe = self._get_validation_pipe()

        try:
            pipe.feed_input("encoded", [encoded])
            outputs = pipe.run()
            del outputs
        except Exception:
            self._drop_validation_pipe()
            raise

    def _validate_path(self, image_path: Path) -> t.Tuple[bool, str]:
        if not image_path.is_file():
            return False, "Invalid or corrupted image: file does not exist"

        image_format = self._guess_format_from_path(image_path)
        if image_format not in self.supported_formats:
            delete_msg = self._delete_file(image_path)
            return False, f"Unsupported DALI format: {image_format}{delete_msg}"

        disguised_format = self._detect_unsupported_signature(image_path)
        if disguised_format is not None:
            delete_msg = self._delete_file(image_path)
            return (
                False,
                f"Unsupported DALI format: {disguised_format} disguised as {image_path.suffix or 'unknown'}{delete_msg}",
            )

        try:
            self._run_pipeline_for_path(image_path)
            return True, image_format
        except Exception as exc:
            delete_msg = self._delete_file(image_path)
            reason = self._summarize_exception(exc)
            return False, f"Invalid or corrupted image: {reason}{delete_msg}"

    def validate(self, path: t.Union[str, Path]) -> t.Tuple[bool, str]:
        """Return whether the image is valid for DALI and an explanatory message."""
        return self._validate_path(Path(path))

    def validate_many(
        self,
        paths: t.Iterable[t.Union[str, Path]],
    ) -> t.Iterator[t.Tuple[str, t.Tuple[bool, str]]]:
        """Validate many image paths using one reusable mixed DALI pipeline."""
        for raw_path in paths:
            image_path = Path(raw_path)
            result = self._validate_path(image_path)

            if self.progress_callback is not None:
                self.progress_callback(1)

            yield str(image_path), result
