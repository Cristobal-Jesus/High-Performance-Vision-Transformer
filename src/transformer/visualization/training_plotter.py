"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th Marh 2026
File: training_plotter.py

Description:
    This file defines the class responsible for storing training history
    and generating plots for accuracy, loss, and optional energy metrics.

References:
    - https://matplotlib.org/stable/
"""

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt


class TrainingPlotter:
    """Almacena el historial de entrenamiento y genera gráficas resumen."""

    def __init__(self) -> None:
        self.train_acc: t.List[float] = []
        self.val_acc: t.List[float] = []
        self.train_loss: t.List[float] = []
        self.val_loss: t.List[float] = []
        # Potencia GPU media por epoch: None en epochs sin validación.
        self.train_gpu_power_w: t.List[float] = []
        self.val_gpu_power_w: t.List[t.Optional[float]] = []
        self.best_val_acc: t.Optional[float] = None
        self.best_epoch: t.Optional[int] = None

    def update(
        self,
        train_acc: float,
        val_acc: float,
        train_loss: float,
        val_loss: float,
        is_best: bool = False,
        train_gpu_power_w: float = 0.0,
        val_gpu_power_w: t.Optional[float] = None,
    ) -> None:
        """Almacena las métricas de un epoch."""
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_gpu_power_w.append(train_gpu_power_w)
        self.val_gpu_power_w.append(val_gpu_power_w)

        if is_best:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.val_acc)

    def plot(
        self,
        save_path: t.Optional[t.Union[str, Path]] = "outputs/figures/transformer/V100/training_curves.png",
        energy_stats: t.Optional[t.Dict[str, t.Any]] = None,
        device_info: t.Optional[str] = None,
    ) -> None:
        """Genera y guarda la gráfica de curvas de entrenamiento.

        Crea 2 paneles (accuracy + loss) si no hay datos de potencia,
        o 3 paneles (accuracy + loss + potencia GPU) si los hay.
        """
        self._validate_history()

        epochs = list(range(1, len(self.train_acc) + 1))
        has_power = any(p > 0.0 for p in self.train_gpu_power_w)

        if has_power:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            self._plot_accuracy(axes[0], epochs)
            self._plot_loss(axes[1], epochs)
            self._plot_gpu_power(axes[2], epochs)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            self._plot_accuracy(axes[0], epochs)
            self._plot_loss(axes[1], epochs)

        if energy_stats is not None:
            self._add_energy_text(fig, energy_stats, device_info)

        fig.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    # ------------------------------------------------------------------
    # Paneles individuales
    # ------------------------------------------------------------------

    def _plot_accuracy(self, axis, epochs: t.List[int]) -> None:
        """Pinta las curvas de accuracy de entrenamiento y validación."""
        axis.plot(epochs, self.train_acc, label="Train Accuracy")
        axis.plot(epochs, self.val_acc, label="Validation Accuracy")

        if self.best_epoch is not None and self.best_val_acc is not None:
            axis.scatter(
                self.best_epoch,
                self.best_val_acc,
                color="red",
                zorder=5,
                label="Best Validation Accuracy",
            )
            axis.annotate(
                "{:.2f}%".format(self.best_val_acc),
                (self.best_epoch, self.best_val_acc),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="red",
            )

        axis.set_title("Accuracy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy (%)")
        axis.legend()

    def _plot_loss(self, axis, epochs: t.List[int]) -> None:
        """Pinta las curvas de pérdida de entrenamiento y validación."""
        axis.plot(epochs, self.train_loss, label="Train Loss")
        axis.plot(epochs, self.val_loss, label="Validation Loss")
        axis.set_title("Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()

    def _plot_gpu_power(self, axis, epochs: t.List[int]) -> None:
        """Pinta la potencia GPU media durante entrenamiento y validación.

        El entrenamiento (forward + backward + optimizer) consume significativamente
        más potencia que la validación (solo forward con torch.inference_mode).
        La caída de potencia durante la validación es visible como valles periódicos.
        """
        axis.plot(
            epochs,
            self.train_gpu_power_w,
            label="Entrenamiento (forward+backward)",
            color="tab:blue",
        )

        # La validación no ocurre en todos los epochs: filtrar los None.
        val_epochs = [
            e for e, p in zip(epochs, self.val_gpu_power_w)
            if p is not None and p > 0.0
        ]
        val_powers = [p for p in self.val_gpu_power_w if p is not None and p > 0.0]

        if val_epochs:
            axis.plot(
                val_epochs,
                val_powers,
                label="Validación (solo forward)",
                color="tab:orange",
                linestyle="--",
                marker="o",
                markersize=3,
            )

        axis.set_title("Potencia GPU")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Potencia (W)")
        axis.legend()

    # ------------------------------------------------------------------
    # Anotación de energía
    # ------------------------------------------------------------------

    def _add_energy_text(
        self,
        figure,
        energy_stats: t.Dict[str, t.Any],
        device_info: t.Optional[str],
    ) -> None:
        """Añade un recuadro con las métricas de energía y dispositivo a la figura."""
        gpu_joules, cpu_joules, elapsed_seconds = self._extract_energy_stats(energy_stats)
        total_joules = gpu_joules + cpu_joules

        text = (
            "Device: {}\n"
            "GPU Energy: {:.4f} J\n"
            "CPU Energy: {:.4f} J\n"
            "Total Energy: {:.4f} J\n"
            "Time: {:.2f} min"
        ).format(
            device_info or "Unknown",
            gpu_joules,
            cpu_joules,
            total_joules,
            elapsed_seconds / 60,
        )

        figure.text(
            0.02,
            0.02,
            text,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    def _extract_energy_stats(
        self,
        energy_stats: t.Dict[str, t.Any],
    ) -> t.Tuple[float, float, float]:
        """Extrae energía GPU, energía CPU y tiempo transcurrido del dict de métricas."""
        measurements = energy_stats.get("meas", {})
        elapsed_seconds = float(energy_stats.get("elapsed", 0.0))

        gpu_joules = 0.0
        cpu_joules = 0.0

        if "gpu_energy_j" in measurements:
            gpu_joules = float(measurements["gpu_energy_j"])
        elif "gpu_energy_uj" in measurements:
            gpu_joules = float(measurements["gpu_energy_uj"]) / 1e6

        for device, values in measurements.items():
            if "rapl" not in str(device).lower():
                continue

            if isinstance(values, dict):
                consumed_microjoules = float(values.get("consumed", 0.0))
            elif isinstance(values, (int, float)):
                consumed_microjoules = float(values)
            else:
                consumed_microjoules = 0.0

            cpu_joules += consumed_microjoules / 1e6

        return gpu_joules, cpu_joules, elapsed_seconds

    # ------------------------------------------------------------------
    # Validación del historial
    # ------------------------------------------------------------------

    def _validate_history(self) -> None:
        """Comprueba que todas las listas de historial tienen la misma longitud."""
        history_lengths = {
            len(self.train_acc),
            len(self.val_acc),
            len(self.train_loss),
            len(self.val_loss),
            len(self.train_gpu_power_w),
            len(self.val_gpu_power_w),
        }

        if len(history_lengths) != 1:
            raise RuntimeError("Training history lists must have the same length.")
        if not self.train_acc:
            raise RuntimeError("No training history is available to plot.")
    