"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: training/ema.py

Description:
    Exponential Moving Average (EMA) de los pesos del modelo.

    Después de cada paso del optimizador se actualiza una copia "sombra"
    del modelo según:

        ema_w = decay * ema_w + (1 - decay) * w

    La copia EMA se usa exclusivamente para validación e inferencia.
    Esto suaviza la trayectoria del entrenamiento y reduce la varianza
    de las predicciones, obteniendo en la práctica +0.5–1.5% de accuracy
    sobre la validación sin cambiar el entrenamiento.

    Implementación compatible con DDP: accede siempre al modelo base
    mediante ``getattr(model, "module", model)`` para ignorar el wrapper
    DistributedDataParallel.

References:
    - Touvron et al. "Training data-efficient image transformers" (DeiT)
      https://arxiv.org/abs/2012.12877
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMAModel:
    """Mantiene y actualiza una copia EMA de los pesos del modelo.

    Args:
        model: Modelo de partida (puede estar envuelto en DDP).
        decay: Factor de suavizado.  Valores habituales: 0.9998 (DeiT)
               o 0.9999 para datasets grandes.  A mayor decay, más
               lentamente cambia la copia EMA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay debe estar en (0, 1), recibido: {decay}")

        self.decay = decay

        # La copia EMA es siempre el modelo base (sin wrapper DDP)
        source = getattr(model, "module", model)
        self._ema: nn.Module = copy.deepcopy(source)
        self._ema.eval()
        for param in self._ema.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update(self, model: nn.Module) -> None:
        """Actualiza la copia EMA con los pesos actuales del modelo.

        Debe llamarse tras cada paso del optimizador.

        Args:
            model: Modelo en entrenamiento (puede ser DDP-wrapped).
        """
        source = getattr(model, "module", model)
        decay  = self.decay

        with torch.no_grad():
            # Parámetros (pesos y sesgos)
            for ema_p, src_p in zip(
                self._ema.parameters(), source.parameters()
            ):
                ema_p.data.mul_(decay).add_(src_p.data, alpha=1.0 - decay)

            # Buffers (ej. running_mean de BatchNorm)
            for ema_b, src_b in zip(
                self._ema.buffers(), source.buffers()
            ):
                ema_b.copy_(src_b)

    @property
    def model(self) -> nn.Module:
        """Devuelve el modelo EMA (en modo eval, sin gradientes)."""
        return self._ema

    def state_dict(self) -> dict:
        """State dict de la copia EMA (para guardar checkpoint)."""
        return self._ema.state_dict()
