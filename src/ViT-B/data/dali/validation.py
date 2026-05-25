"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: validation.py

Description:
    This file defines the class responsible for validating whether an
    image file can be safely processed by NVIDIA DALI.

References:
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, UnidentifiedImageError


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

    def __init__(self, supported_formats: Iterable[str] | None = None) -> None:
        self.supported_formats = set(supported_formats or self.DEFAULT_SUPPORTED_FORMATS)

    def validate(self, path: str | Path) -> tuple[bool, str]:
        """Return whether the image is valid for DALI and an explanatory message."""
        image_path = Path(path)

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image_format = image.format or "UNKNOWN"

            if image_format in self.supported_formats:
                return True, image_format

            return False, f"Unsupported DALI format: {image_format}"

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return False, f"Invalid or corrupted image: {exc}"
