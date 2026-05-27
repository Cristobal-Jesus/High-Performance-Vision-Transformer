"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/common/checkpoint.py

Description:
    Funciones para cargar el VisionTransformer desde un checkpoint y
    guardar modelos podados.  Cada función hace exactamente una cosa.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from transformer.models.transformer_model import VisionTransformer


def load_vision_transformer(
    checkpoint_path: Path | str,
    *,
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 20,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    dropout: float = 0.10,
    drop_path_rate: float = 0.20,
    device: torch.device | None = None,
) -> VisionTransformer:
    """Instancia y carga un VisionTransformer desde un checkpoint .pth.

    Args:
        checkpoint_path: Ruta al archivo ``.pth`` con los pesos guardados.
        image_size:       Tamaño de la imagen cuadrada de entrada (píxeles).
        patch_size:       Tamaño de cada parche cuadrado.
        num_classes:      Número de clases de salida.
        embed_dim:        Dimensión del embedding.
        depth:            Número de bloques del encoder.
        num_heads:        Número de cabezas de atención.
        mlp_ratio:        Ratio de expansión de la MLP.
        dropout:          Tasa de dropout de activaciones.
        drop_path_rate:   Tasa de DropPath estocástico.
        device:           Dispositivo destino; si es ``None`` usa CPU.

    Returns:
        Modelo en modo evaluación sobre el dispositivo indicado.
    """
    if device is None:
        device = torch.device("cpu")

    patch_dim = patch_size * patch_size * 3
    seq_len = (image_size // patch_size) ** 2

    model = VisionTransformer(
        patch_dim=patch_dim,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_classes=num_classes,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        use_patch_norm=True,
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # El trainer guarda un dict con clave "model_state_dict" o el state_dict directo.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Si se entrenó con torch.compile, los pesos tienen prefijo "_orig_mod."
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_model(
    model: nn.Module,
    output_path: Path | str,
    *,
    metadata: dict | None = None,
) -> None:
    """Guarda el state_dict del modelo y metadatos opcionales.

    Args:
        model:       Modelo cuyo state_dict se va a guardar.
        output_path: Ruta de destino (.pth).
        metadata:    Diccionario extra que se añade al fichero (ej.
                     sparsidad, número de cabezas podadas, etc.).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {"model_state_dict": model.state_dict()}
    if metadata:
        payload.update(metadata)

    torch.save(payload, output_path)
    print(f"[checkpoint] Modelo guardado en {output_path}")
