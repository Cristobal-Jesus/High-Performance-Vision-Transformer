"""
evaluate_best_transformer.py

Evaluates best_transformer.pth on a flat Test folder whose image names follow:
<ClassName>_<index>.jpg

Example:
Bee_003.jpg
Northern_cardinal_12.png
White_tailed_deer_99.jpeg
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = "checkpoints/transformer/best_transformer.pth"
IMAGES_DIR = Path("/home/bejeque/nhernang/Cristobal/pytorch_models/assets/Test")


# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.transformer.models.transformer_model import VisionTransformer
except ImportError:
    from transformer.models.transformer_model import VisionTransformer


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

CLASS_NAMES = [
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

LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def label_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    class_name = "_".join(stem.split("_")[:-1])

    if class_name not in LABEL_MAP:
        raise ValueError(
            f"No se pudo inferir la clase desde '{filename}'. "
            f"Clase detectada: '{class_name}'"
        )

    return LABEL_MAP[class_name]


class FlatImageDataset(Dataset):
    def __init__(self, images_dir: Path) -> None:
        self.images_dir = images_dir
        self.extensions = {".jpg", ".jpeg", ".png"}

        self.paths = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.extensions
        )

        if not self.paths:
            raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        label = label_from_filename(path.name)

        return image, label, path.name


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------

def patchify(images: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    batch_size, channels, height, width = images.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Tamaño inválido: {height}x{width}")

    patches_h = height // patch_size
    patches_w = width // patch_size

    images = images.unfold(2, patch_size, patch_size)
    images = images.unfold(3, patch_size, patch_size)
    images = images.permute(0, 2, 3, 1, 4, 5).contiguous()

    return images.view(
        batch_size,
        patches_h * patches_w,
        channels * patch_size * patch_size,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def clean_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value

    return cleaned


def load_model(checkpoint_path: Path, device: torch.device) -> VisionTransformer:
    print(f"Cargando checkpoint: {checkpoint_path}")

    model = VisionTransformer().to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = clean_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    dataset = FlatImageDataset(IMAGES_DIR)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Imágenes encontradas: {len(dataset)}")

    model = load_model(CHECKPOINT_PATH, device)

    correct = 0
    total = 0

    class_correct = Counter()
    class_total = Counter()
    wrong_examples = []

    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            patches = patchify(images, patch_size=16)

            outputs = model(patches)
            preds = outputs.argmax(dim=1)

            matches = preds.eq(labels)

            correct += matches.sum().item()
            total += labels.size(0)

            for name, target, pred, ok in zip(
                names,
                labels.cpu().tolist(),
                preds.cpu().tolist(),
                matches.cpu().tolist(),
            ):
                class_total[target] += 1

                if ok:
                    class_correct[target] += 1
                elif len(wrong_examples) < 30:
                    wrong_examples.append((name, target, pred))

    accuracy = 100.0 * correct / total

    print()
    print(f"Accuracy total: {accuracy:.2f}% ({correct}/{total})")

    print()
    print("Accuracy por clase:")
    for idx, class_name in IDX_TO_CLASS.items():
        n = class_total[idx]
        c = class_correct[idx]

        if n == 0:
            print(f"  {idx:02d} {class_name}: sin muestras")
        else:
            print(f"  {idx:02d} {class_name}: {100.0 * c / n:.2f}% ({c}/{n})")

    if wrong_examples:
        print()
        print("Primeros errores:")
        for name, target, pred in wrong_examples:
            print(
                f"  {name}: "
                f"real={target:02d} {IDX_TO_CLASS[target]} | "
                f"pred={pred:02d} {IDX_TO_CLASS.get(pred, '<unknown>')}"
            )


if __name__ == "__main__":
    evaluate()
