"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/common/evaluator.py

Description:
    Evaluador de modelos sobre el conjunto de validación.
    Devuelve accuracy top-1, latencia media y número de parámetros
    efectivos (no nulos).
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


def evaluate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    """Evalúa el modelo sobre el val_loader y devuelve (accuracy, ms/img).

    Args:
        model:       Modelo en modo eval ya sobre ``device``.
        val_loader:  Iterador DALI de validación.
        device:      Dispositivo de inferencia.
        max_batches: Si se especifica, evalúa solo los primeros N batches.

    Returns:
        Tupla ``(top1_accuracy_pct, ms_per_image)``.
    """
    model.eval()
    correct = total = 0
    elapsed = 0.0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # DALI devuelve una lista de dicts con claves "data" y "label"
            images = batch[0]["data"].to(device, non_blocking=True)
            labels = batch[0]["label"].squeeze(-1).long().to(device, non_blocking=True)

            t0 = time.perf_counter()
            logits = model(images)
            torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - t0

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    ms_per_img = (elapsed / total * 1000) if total > 0 else 0.0
    return accuracy, ms_per_img


def count_nonzero_params(model: nn.Module) -> Tuple[int, int]:
    """Cuenta parámetros totales y no nulos del modelo.

    Args:
        model: Modelo PyTorch.

    Returns:
        Tupla ``(total_params, nonzero_params)``.
    """
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += param.count_nonzero().item()
    return total, nonzero


def model_sparsity(model: nn.Module) -> float:
    """Devuelve la fracción de pesos iguales a cero en el modelo.

    Args:
        model: Modelo PyTorch.

    Returns:
        Fracción en [0, 1] de pesos cero.
    """
    total, nonzero = count_nonzero_params(model)
    if total == 0:
        return 0.0
    return 1.0 - nonzero / total
