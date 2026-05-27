"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/unstructured/weight_pruner.py

Description:
    Poda no estructurada por magnitud sobre las capas lineales del
    VisionTransformer.

    Utiliza ``torch.nn.utils.prune.l1_unstructured`` para calcular la
    máscara de poda (los pesos de menor norma L1 se anulan), luego llama
    a ``remove`` para hacer la poda permanente (elimina los hooks y
    almacena el peso efectivo directamente).

    Capas podadas:
        - Todos los ``nn.Linear`` excepto la capa de clasificación (``head``)
          para no distorsionar la salida del modelo.

References:
    - https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune

from transformer.models.transformer_model import VisionTransformer


class WeightPruner:
    """Aplica poda no estructurada por magnitud a las capas lineales.

    Args:
        sparsity: Fracción de pesos a anular (ej. 0.5 → 50% de ceros).
    """

    def __init__(self, sparsity: float = 0.50) -> None:
        if not (0.0 < sparsity < 1.0):
            raise ValueError(f"sparsity debe estar en (0, 1), recibido: {sparsity}")
        self.sparsity = sparsity

    def prune(self, model: VisionTransformer) -> VisionTransformer:
        """Aplica poda L1 in-place a todos los ``nn.Linear`` del modelo.

        La capa de clasificación (``model.head``) se excluye para
        preservar la proyección al espacio de clases.

        Args:
            model: VisionTransformer a podar.

        Returns:
            El mismo modelo con los pesos podados (permanente).
        """
        n_layers = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Excluir la capa de clasificación
            if name == "head":
                continue

            prune.l1_unstructured(module, name="weight", amount=self.sparsity)
            prune.remove(module, "weight")  # hace la máscara permanente
            n_layers += 1

        print(
            f"  Poda no estructurada aplicada a {n_layers} capas lineales "
            f"(sparsidad objetivo: {self.sparsity:.0%})"
        )
        return model
