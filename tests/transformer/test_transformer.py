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
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
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
    """Loads a trained model and evaluates it over a directory of images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: t.Optional[torch.device] = None,
        validator: t.Optional[DaliImageValidator] = None,
        patch_size: int = 16,
        scratch: bool = None,                          # Fix: None → False
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scratch = scratch                          # Fix 4: guardar como atributo
        self.model = self._load_model(checkpoint_path, scratch)
        self.validator = validator or DaliImageValidator(delete_invalid=True)
        self.processor = PatchBatchProcessor(patch_size=patch_size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str, scratch: bool) -> nn.Module:
        if scratch:
            # Arquitectura idéntica a la usada en el entrenamiento (config.py)
            model = VisionTransformer(
                patch_dim=16 * 16 * 3,       # patch_size=16
                seq_len=(224 // 16) ** 2,    # 196 tokens
                embed_dim=384,
                num_classes=len(CLASS_NAMES),
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                dropout=0.0,                 # eval: dropout desactivado
                drop_path_rate=0.0,          # eval: drop_path desactivado
                use_patch_norm=True,
            ).to(self.device)

            state_dict = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            # El checkpoint puede tener el prefijo "_orig_mod." si el modelo
            # fue guardado desde un torch.compile (OptimizedModule).
            state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
            model.eval()
            return model

        # --- torchvision ViT-B/16 (checkpoint legacy) ---
        vit_model = vit_b_16(weights=None)
        vit_model.heads.head = nn.Linear(
            vit_model.heads.head.in_features,
            len(CLASS_NAMES),
        )

        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        vit_model.load_state_dict(state_dict)
        vit_model.to(self.device)

        model = vit_model
        try:
            model = torch.compile(vit_model)
        except Exception as exc:
            print(f"[torch.compile] Disabled: {exc}")

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
                # Fix 4: cada modelo espera un formato de entrada diferente
                if self.scratch:
                    inputs, targets = self.processor.prepare_batch(data, self.device)
                else:
                    batch = data[0]
                    inputs = batch["images"].to(self.device, non_blocking=True)
                    targets = batch["labels"].squeeze(-1).long().to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return 100.0 * correct / total
