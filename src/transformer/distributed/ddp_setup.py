"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distributed/ddp_setup.py

Description:
    Inicialización del proceso distribuido y funciones de consulta de rango
    para entrenamiento con DistributedDataParallel (DDP).

    torchrun establece automáticamente estas variables de entorno:
        RANK        — rango global del proceso entre todos los nodos
        LOCAL_RANK  — rango local en el nodo actual (= índice de la GPU)
        WORLD_SIZE  — número total de procesos en ejecución
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Ciclo de vida del grupo de procesos
# ---------------------------------------------------------------------------

def init_process_group(backend: str = "nccl") -> None:
    """Inicializa el grupo de procesos distribuidos y asigna la GPU local.

    Debe llamarse una sola vez al inicio de cada proceso DDP, antes de
    cualquier operación colectiva o de envolver el modelo con DDP.

    Args:
        backend: Backend de comunicación.  Usa ``"nccl"`` para entrenamiento
            en GPU (por defecto) y ``"gloo"`` solo para pruebas en CPU.
    """
    dist.init_process_group(backend=backend)
    # Vincula el proceso a su GPU local para que las operaciones CUDA
    # vayan al dispositivo correcto sin especificarlo en cada llamada.
    torch.cuda.set_device(get_local_rank())


def cleanup() -> None:
    """Destruye el grupo de procesos distribuidos.

    Debe llamarse en un bloque ``finally`` para garantizar que se ejecuta
    incluso si el entrenamiento termina con error.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Consultas de rango — cada función expone un único dato del proceso
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Devuelve el rango global de este proceso.

    El rango 0 es el proceso principal (master): el único que guarda
    checkpoints, imprime métricas y genera gráficas.

    Returns:
        0 si el grupo de procesos no está inicializado (ejecución mono-GPU).
    """
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank() -> int:
    """Devuelve el rango local de este proceso, es decir, el índice de su GPU.

    En un nodo con 2 GPUs: proceso 0 → LOCAL_RANK=0, proceso 1 → LOCAL_RANK=1.

    Returns:
        0 si la variable de entorno LOCAL_RANK no está definida.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Devuelve el número total de procesos participando en el entrenamiento.

    Returns:
        1 si el grupo de procesos no está inicializado (ejecución mono-GPU).
    """
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Indica si este proceso es el principal (rango 0).

    Usa este guard para operaciones que deben ocurrir exactamente una vez:
    imprimir métricas, guardar checkpoints, medir energía, generar gráficas.

    Returns:
        ``True`` solo en el proceso de rango 0.
    """
    return get_rank() == 0