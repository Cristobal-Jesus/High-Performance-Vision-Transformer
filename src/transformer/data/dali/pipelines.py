"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: pipelines.py

Description:
    This file defines the class responsible for building the NVIDIA DALI
    pipelines used for training and validation image preprocessing.

References:
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""


import typing as t
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def


class DaliPipelineFactory:
    """Build NVIDIA DALI pipelines for training and validation."""

    def __init__(
        self,
        mean: t.Optional[t.Sequence[float]] = None,
        std: t.Optional[t.Sequence[float]] = None,
        horizontal_flip_probability: float = 0.5,
        grayscale_probability: float = 0.15,
        erasing_probability: float = 0.25,
        brightness_range: t.Tuple[float, float] = (0.85, 1.15),
        contrast_range: t.Tuple[float, float] = (0.85, 1.15),
        saturation_range: t.Tuple[float, float] = (0.80, 1.20),
        hue_range: t.Tuple[float, float] = (-10.0, 10.0),
        erasing_area_range: t.Tuple[float, float] = (0.02, 0.20),
    ) -> None:
        self.mean = list(mean or [0.485 * 255, 0.456 * 255, 0.406 * 255])
        self.std = list(std or [0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.horizontal_flip_probability = horizontal_flip_probability
        self.grayscale_probability = grayscale_probability
        self.erasing_probability = erasing_probability

        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.erasing_area_range = erasing_area_range

    def create_train_pipeline(
        self,
        *,
        batch_size: int,
        num_threads: int,
        device_id: int,
        file_list: str,
        crop: int = 224,
        prefetch_queue_depth: int = 6,
    ):
        """Create the DALI pipeline used during training."""
        mean = self.mean
        std = self.std

        horizontal_flip_probability = self.horizontal_flip_probability
        grayscale_probability = self.grayscale_probability
        erasing_probability = self.erasing_probability

        brightness_range = self.brightness_range
        contrast_range = self.contrast_range
        saturation_range = self.saturation_range
        hue_range = self.hue_range
        erasing_area_range = self.erasing_area_range

        @pipeline_def(enable_conditionals=True)
        def pipeline(file_list: str, crop: int = 224):
            images, labels = fn.readers.file(
                file_list=file_list,
                random_shuffle=True,
                name="Reader",
                stick_to_shard=True,
            )

            images = fn.decoders.image(
                images,
                device="mixed",
                output_type=types.RGB,
            )

            images = fn.random_resized_crop(
                images,
                device="gpu",
                size=(crop, crop),
                random_area=(0.8, 1.0),
                random_aspect_ratio=(0.75, 1.33),
            )

            # Apply color jitter.
            brightness = fn.random.uniform(range=brightness_range)
            contrast = fn.random.uniform(range=contrast_range)
            saturation = fn.random.uniform(range=saturation_range)
            hue = fn.random.uniform(range=hue_range)

            images = fn.color_twist(
                images,
                device="gpu",
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )

            # Apply grayscale augmentation by fully desaturating the image.
            # This keeps the image in RGB format instead of converting it to 1 channel.
            do_grayscale = fn.random.coin_flip(probability=grayscale_probability)
            if do_grayscale:
                images = fn.color_twist(
                    images,
                    device="gpu",
                    saturation=0.0,
                )

            # Apply random erasing with normalized coordinates.
            do_erase = fn.random.coin_flip(probability=erasing_probability)
            if do_erase:
                erase_anchor = fn.random.uniform(range=(0.0, 0.80), shape=[2])
                erase_shape = fn.random.uniform(range=erasing_area_range, shape=[2])

                images = fn.erase(
                    images,
                    device="gpu",
                    anchor=erase_anchor,
                    shape=erase_shape,
                    axis_names="HW",
                    fill_value=0,
                    normalized_anchor=True,
                    normalized_shape=True,
                )

            # Apply horizontal flip.
            mirror = fn.random.coin_flip(probability=horizontal_flip_probability)

            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mirror=mirror,
                mean=mean,
                std=std,
            )

            return images, labels

        return pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            file_list=file_list,
            crop=crop,
            prefetch_queue_depth=prefetch_queue_depth,
        )

    def create_val_pipeline(
        self,
        *,
        batch_size: int,
        num_threads: int,
        device_id: int,
        file_list: str,
        size: int = 224,
        prefetch_queue_depth: int = 2,
    ):
        """Create the DALI pipeline used during validation."""
        mean = self.mean
        std = self.std

        @pipeline_def
        def pipeline(file_list: str, size: int = 224):
            images, labels = fn.readers.file(
                file_list=file_list,
                random_shuffle=False,
                name="Reader",
                stick_to_shard=True,
            )

            images = fn.decoders.image(
                images,
                device="mixed",
                output_type=types.RGB,
            )

            images = fn.resize(
                images,
                device="gpu",
                resize_x=size,
                resize_y=size,
            )

            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=mean,
                std=std,
            )

            return images, labels

        return pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            file_list=file_list,
            size=size,
            prefetch_queue_depth=prefetch_queue_depth,
        )
