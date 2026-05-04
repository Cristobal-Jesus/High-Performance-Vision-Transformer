"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: dali_eval_pipeline.py

Description:
    This file defines the DALI pipeline and iterator used during
    evaluation. It loads images and their integer labels in batches,
    applying the standard ImageNet normalisation expected by the model.
"""

from __future__ import annotations

import typing as t

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@pipeline_def
def _eval_pipeline(
    image_paths: t.List[str],
    labels: t.List[int],
    shard_id: int = 0,
    num_shards: int = 1,
) -> t.Tuple:
    """DALI pipeline that loads, decodes, and normalises images for eval.

    Args:
        image_paths: Ordered list of absolute paths to valid image files.
        labels: Integer label for each image, aligned with image_paths.
        shard_id: Index of the current shard (0 for single-GPU eval).
        num_shards: Total number of shards (1 for single-GPU eval).

    Returns:
        A tuple (images, labels) ready for the DALIGenericIterator.
    """
    # -- file reading --------------------------------------------------------
    encoded, labels_out = fn.readers.file(
        files=image_paths,
        labels=labels,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
        name="Reader",
    )

    # -- decoding ------------------------------------------------------------
    images = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB,
        hw_decoder_load=0.9,
    )

    # -- resize --------------------------------------------------------------
    images = fn.resize(
        images,
        device="gpu",
        resize_x=224,
        resize_y=224,
        interp_type=types.INTERP_LINEAR,
    )

    # -- layout: HWC -> CHW, uint8 -> float32 normalised ---------------------
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_layout="CHW",
    )

    return images, labels_out


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_eval_iterator(
    image_paths: t.List[str],
    labels: t.List[int],
    batch_size: int = 128,
    num_threads: int = 4,
    device_id: int = 0,
) -> DALIGenericIterator:
    """Builds and returns a DALI iterator for evaluation.

    Args:
        image_paths: Ordered list of absolute paths to valid image files.
        labels: Integer label for each image, aligned with image_paths.
        batch_size: Number of samples per batch.
        num_threads: CPU threads used by the pipeline.
        device_id: CUDA device index. Must be >= 0 because this pipeline uses
            mixed/GPU DALI operators.

    Returns:
        A DALIGenericIterator yielding dicts with keys 'images' and 'labels'.
    """
    if len(image_paths) != len(labels):
        raise ValueError(
            "image_paths and labels must have the same length "
            f"({len(image_paths)} != {len(labels)})"
        )

    if device_id is None or device_id < 0:
        raise ValueError(
            "This DALI pipeline uses device='mixed' and device='gpu', "
            "so device_id must be a valid CUDA device index, e.g. 0."
        )

    pipe = _eval_pipeline(
        image_paths=image_paths,
        labels=labels,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )
    pipe.build()

    return DALIGenericIterator(
        pipe,
        output_map=["images", "labels"],
        size=len(image_paths),
        auto_reset=True,
    )
