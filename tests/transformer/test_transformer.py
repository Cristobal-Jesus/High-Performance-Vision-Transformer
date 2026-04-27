"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
File: evaluator.py

Description:
    This module defines the Evaluator class, responsible for loading a
    trained model and computing its accuracy over a directory of test
    images. Images are validated with NVIDIA DALI before inference, and
    batches are prepared with PatchBatchProcessor.
"""

from __future__ import annotations

import os
import typing as t
from pathlib import Path

import torch
from tqdm import tqdm

from src.transformer.training.batch_processor import PatchBatchProcessor
from src.transformer.models.transformer_model import VisionTransformer
from src.transformer.data.dali.validation import DaliImageValidator
from .dali_eval_pipeline import build_eval_iterator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: t.List[str] = [
    "Anade_azulon",
    "Bald_eagle",
    "Bee",
    "Camachuelo_mexicano",
    "Downy_woodpecker",
    "Firebug",
    "Garceta_grande",
    "Garza_azulada",
    "Mariquita",
    "Martinete_comun",
    "Mirlo_comun",
    "Monarch",
    "Northern_cardinal",
    "Painted_lady",
    "Paloma",
    "Red_spotted_admirial",
    "Small_white",
    "White_heron",
    "White_tailed_deer",
    "Zorzal_americano",
]

LABEL_MAP: t.Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Loads a trained model and evaluates it over a directory of images.

    Images are first validated with NVIDIA DALI. Only valid images are
    fed to the evaluation pipeline. Batches are patchified by
    PatchBatchProcessor before being passed to the model.

    Attributes:
        device: The torch device used for inference.
        model: The loaded model in evaluation mode.
        validator: A DaliImageValidator instance used to pre-filter images.
        processor: A PatchBatchProcessor that converts images to patches.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: t.Optional[torch.device] = None,
        validator: t.Optional[DaliImageValidator] = None,
        patch_size: int = 16,
    ) -> None:
        """Initialises the Evaluator.

        Args:
            checkpoint_path: Path to the model checkpoint (.pth file).
            device: Torch device to use. Defaults to CUDA if available,
                otherwise CPU.
            validator: A DaliImageValidator instance. If None, a default
                one is created with delete_invalid=True.
            patch_size: Patch size passed to PatchBatchProcessor.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model(checkpoint_path)
        self.validator = validator or DaliImageValidator(delete_invalid=True)
        self.processor = PatchBatchProcessor(patch_size=patch_size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str) -> VisionTransformer:
        """Loads model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pth checkpoint.

        Returns:
            The model in evaluation mode, moved to self.device.
        """
        model = VisionTransformer().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def _label_from_filename(
        filename: str,
        label_map: t.Dict[str, int] = LABEL_MAP,
    ) -> int:
        """Infers the integer label from an image filename.

        Filenames are expected to follow the pattern
        ``<ClassName>_<index>.<ext>``, e.g. ``Bee_003.jpg``.

        Args:
            filename: The bare filename (no directory component).
            label_map: Mapping from class name string to integer label.

        Returns:
            The integer label corresponding to the class encoded in
            the filename.

        Raises:
            ValueError: If the class name cannot be found in label_map.
        """
        stem = os.path.splitext(filename)[0]
        class_name = "_".join(stem.split("_")[:-1])

        if class_name in label_map:
            return label_map[class_name]

        raise ValueError(
            f"Could not infer class from filename '{filename}'. "
            f"Expected one of: {sorted(label_map.keys())}"
        )

    def _collect_valid_images(
        self,
        images_dir: str,
    ) -> t.Tuple[t.List[str], t.List[int]]:
        """Scans a directory, validates images with DALI, and collects paths.

        Args:
            images_dir: Directory containing the image files.

        Returns:
            A tuple (image_paths, labels) containing only entries for
            images that passed DALI validation.
        """
        supported_extensions = (".jpg", ".jpeg", ".png")
        candidate_paths = [
            os.path.join(images_dir, fname)
            for fname in sorted(os.listdir(images_dir))
            if fname.lower().endswith(supported_extensions)
        ]

        image_paths: t.List[str] = []
        labels: t.List[int] = []
        skipped = 0

        progress = tqdm(
            self.validator.validate_many(candidate_paths),
            total=len(candidate_paths),
            desc="Validating",
            unit="img",
        )

        for path_str, (is_valid, message) in progress:
            if not is_valid:
                progress.write(f"  [SKIP] {Path(path_str).name} — {message}")
                skipped += 1
                continue

            fname = Path(path_str).name
            image_paths.append(path_str)
            labels.append(self._label_from_filename(fname))

        if skipped:
            print(f"  {skipped} image(s) skipped after DALI validation.\n")

        return image_paths, labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model_stats(self) -> t.Tuple[int, float]:
        """Returns the total parameter count and model size in megabytes.

        Returns:
            A tuple (total_params, size_mb).
        """
        total_params = 0
        total_bytes = 0

        for param in self.model.parameters():
            num = param.numel()
            total_params += num
            total_bytes += num * param.element_size()

        size_mb = total_bytes / (1024 ** 2)
        return total_params, size_mb

    def evaluate(
        self,
        images_dir: str,
        batch_size: int = 128,
        num_threads: int = 4,
        ) -> float:
        """Evaluates the model accuracy over images in a directory.

        Only images that pass DALI validation are included. Each batch
        is patchified by PatchBatchProcessor before being passed to the
        model.

        Args:
            images_dir: Path to the directory containing test images.
            batch_size: Number of samples per inference batch.
            num_threads: CPU threads used by the DALI pipeline.

        Returns:
            Top-1 accuracy as a percentage (0-100).

        Raises:
            RuntimeError: If no valid images are found after validation.
        """
        print("Validating images with DALI...")
        image_paths, labels = self._collect_valid_images(images_dir)

        if not image_paths:
            raise RuntimeError(
                f"No valid images found in '{images_dir}' after DALI validation."
            )

        if self.device.type != "cuda":
            raise RuntimeError(
                "The DALI evaluation pipeline uses GPU operators "
                "(device='mixed' and device='gpu'), but the evaluator is using CPU. "
                "Use a CUDA device or create a CPU-only DALI pipeline."
            )

        device_id = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )

        loader = build_eval_iterator(
            image_paths=image_paths,
            labels=labels,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(loader, desc="Inference", unit="batch"):
                patches, targets = self.processor.prepare_batch(data, self.device)

                outputs = self.model(patches)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return 100.0 * correct / total
