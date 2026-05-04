"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
File: evaluator.py

Description:
    This module defines the Evaluator class, responsible for loading a
    trained ResNet50 model and computing its accuracy over a directory of test
    images. Images are validated with NVIDIA DALI before inference.
"""

from __future__ import annotations

import os
import typing as t
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50
from tqdm import tqdm

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
    """Loads a trained ResNet50 model and evaluates it over a directory of images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: t.Optional[torch.device] = None,
        validator: t.Optional[DaliImageValidator] = None,
        num_classes: int = 20,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model(checkpoint_path, num_classes)
        self.validator = validator or DaliImageValidator(delete_invalid=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str, num_classes: int = 20) -> nn.Module:
        """Load ResNet50 model trained on custom dataset."""
        model = resnet50(weights=None)
        # Adapt the final layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def _label_from_filename(
        filename: str,
        label_map: t.Dict[str, int] = LABEL_MAP,
    ) -> int:
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
                batch = data[0]
                inputs = batch["images"].to(self.device, non_blocking=True)
                targets = batch["labels"].squeeze(-1).long().to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return 100.0 * correct / total
