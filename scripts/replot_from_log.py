"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: scripts/replot_from_log.py

Description:
    Regenera una gráfica de curvas de entrenamiento a partir del LOG de una
    ejecución antigua, aplicando el formato unificado (el mismo que usan las
    gráficas de H200/H100): 2 paneles (Accuracy + Loss) y un recuadro de
    energía de una sola línea centrado bajo las gráficas.

    Sirve para reconvertir gráficas viejas (recuadro multilínea en la esquina)
    al formato nuevo SIN tener que re-entrenar: los datos por época se leen del
    log real, así que no se inventa ni se altera ningún valor.

    El log debe contener las líneas que imprime el trainer, del estilo:
        Epoch [10/35] Train Loss: 1.0700 | Train Acc: 87.50% | Val Loss: 1.0100 | Val Acc: 90.12%

Ejemplo de uso:
    python scripts/replot_from_log.py \\
        --log logs/cnn_v100.out \\
        --output outputs/figures/convolutional/V100/training_curves_cnn_V100.png \\
        --device "Tesla V100-PCIE-32GB" \\
        --gpu-energy-j 2669492.0120 \\
        --time-min 235.11
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# Captura: epoch, total, train_loss, train_acc, val_loss, val_acc
EPOCH_RE = re.compile(
    r"Epoch\s*\[(\d+)/(\d+)\].*?"
    r"Train Loss:\s*([\d.]+).*?"
    r"Train Acc:\s*([\d.]+)%.*?"
    r"Val Loss:\s*([\d.]+).*?"
    r"Val Acc:\s*([\d.]+)%"
)

# Líneas de energía/best epoch que el trainer imprime al final (opcionales).
BEST_EPOCH_RE = re.compile(r"Best epoch:\s*(\d+)")
GPU_ENERGY_RE = re.compile(r"GPU energy\s*\([^)]*\):\s*([\d.]+)\s*J")


def parse_log(log_path: Path) -> Dict[str, List[float]]:
    """Extrae las curvas por época del log de entrenamiento.

    Args:
        log_path: Ruta al fichero de log (.out de SLURM o stdout capturado).

    Returns:
        Diccionario con las listas paralelas train_acc, val_acc, train_loss,
        val_loss en orden de época.

    Raises:
        ValueError: si no se encuentra ninguna línea de época en el log.
    """
    train_acc: List[float] = []
    val_acc: List[float] = []
    train_loss: List[float] = []
    val_loss: List[float] = []

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        train_loss.append(float(match.group(3)))
        train_acc.append(float(match.group(4)))
        val_loss.append(float(match.group(5)))
        val_acc.append(float(match.group(6)))

    if not train_acc:
        raise ValueError(
            f"No se encontró ninguna línea 'Epoch [x/y] ...' en {log_path}. "
            "Comprueba que el log es el correcto."
        )

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def parse_energy(log_path: Path) -> Dict[str, Optional[float]]:
    """Intenta leer best_epoch y GPU energy del log (pueden no estar)."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    best_epoch_match = BEST_EPOCH_RE.search(text)
    gpu_energy_match = GPU_ENERGY_RE.search(text)

    return {
        "best_epoch": int(best_epoch_match.group(1)) if best_epoch_match else None,
        "gpu_energy_j": float(gpu_energy_match.group(1)) if gpu_energy_match else None,
    }


def resolve_best(
    history: Dict[str, List[float]],
    best_epoch_override: Optional[int],
    best_epoch_from_log: Optional[int],
) -> tuple[int, float]:
    """Determina la mejor época y su accuracy de validación.

    La mejor época es la de menor val_loss (misma regla que el trainer).
    Se puede forzar con --best-epoch o con el valor leído del log.

    Returns:
        (best_epoch_1based, best_val_acc)
    """
    if best_epoch_override is not None:
        best_idx = best_epoch_override - 1
    elif best_epoch_from_log is not None:
        best_idx = best_epoch_from_log - 1
    else:
        # argmin de val_loss
        best_idx = min(
            range(len(history["val_loss"])),
            key=lambda i: history["val_loss"][i],
        )

    best_idx = max(0, min(best_idx, len(history["val_acc"]) - 1))
    return best_idx + 1, history["val_acc"][best_idx]


def plot(
    history: Dict[str, List[float]],
    best_epoch: int,
    best_val_acc: float,
    output_path: Path,
    device: str,
    gpu_energy_j: Optional[float],
    time_min: Optional[float],
) -> None:
    """Genera la figura con el formato unificado (2 paneles + recuadro energía).

    Replica exactamente el formato de los TrainingPlotter del proyecto:
        - Eje Y de accuracy etiquetado como "Accuracy (%)".
        - Punto rojo en la mejor accuracy de validación con su anotación.
        - Recuadro de energía de una línea, centrado bajo las gráficas.
    """
    epochs = list(range(1, len(history["train_acc"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel de accuracy ---
    axes[0].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[0].plot(epochs, history["val_acc"], label="Validation Accuracy")
    axes[0].scatter(
        best_epoch, best_val_acc, color="red", zorder=5,
        label="Best Validation Accuracy",
    )
    axes[0].annotate(
        "{:.2f}%".format(best_val_acc),
        (best_epoch, best_val_acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        color="red",
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()

    # --- Panel de loss ---
    axes[1].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # --- Recuadro de energía (una línea, centrado abajo) ---
    has_energy = gpu_energy_j is not None or time_min is not None
    if has_energy:
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        text = "Device: {}   |   GPU Energy: {:.4f} J   |   Time: {:.2f} min".format(
            device or "Unknown",
            gpu_energy_j if gpu_energy_j is not None else 0.0,
            time_min if time_min is not None else 0.0,
        )
        fig.text(
            0.5, 0.04, text,
            fontsize=9, ha="center", verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en: {output_path}  ({len(epochs)} épocas)")


def main() -> None:
    """Punto de entrada de la herramienta de regeneración."""
    parser = argparse.ArgumentParser(
        description="Regenera una gráfica de entrenamiento desde el log con el formato unificado.",
    )
    parser.add_argument("--log", required=True, type=Path, help="Ruta al log de la ejecución.")
    parser.add_argument("--output", required=True, type=Path, help="Ruta del PNG de salida.")
    parser.add_argument("--device", default="Unknown", help="Nombre del dispositivo para el recuadro.")
    parser.add_argument("--gpu-energy-j", type=float, default=None, help="Energía GPU en julios.")
    parser.add_argument("--time-min", type=float, default=None, help="Tiempo total en minutos.")
    parser.add_argument("--best-epoch", type=int, default=None, help="Forzar la mejor época (1-based).")
    args = parser.parse_args()

    history = parse_log(args.log)
    energy_from_log = parse_energy(args.log)

    best_epoch, best_val_acc = resolve_best(
        history, args.best_epoch, energy_from_log["best_epoch"],
    )

    gpu_energy_j = args.gpu_energy_j
    if gpu_energy_j is None:
        gpu_energy_j = energy_from_log["gpu_energy_j"]

    plot(
        history=history,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        output_path=args.output,
        device=args.device,
        gpu_energy_j=gpu_energy_j,
        time_min=args.time_min,
    )


if __name__ == "__main__":
    main()
