"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: comparator.py

Description:
    Orchestrator class that benchmarks the FP32 VisionTransformer baseline
    against its INT8 dynamically quantized variant, then saves a comparison
    figure. Both variants run on CPU because PyTorch dynamic quantization
    is CPU-only.
"""

from __future__ import annotations

from pathlib import Path

from transformer.training.config import TransformerTrainingConfig
from .benchmark import InferenceBenchmark
from .constants import (
    CHECKPOINT_PATH,
    EVAL_BATCH_SIZE,
    INT8_CHECKPOINT_PATH,
    VARIANT_LABEL,
)
from .model_factory import ModelFactory
from .plotter import QuantizationPlotter
from .quantization_stats import QuantizationStats


class QuantizationComparator:
    """Benchmark FP32 vs INT8 dynamic quantization and generate a report.

    Args:
        test_dir: Path to the flat directory of 20-class test images.
        config: Training config used to reconstruct the model architecture.
                Defaults to ``TransformerTrainingConfig()``.
        checkpoint_path: Path to the FP32 model checkpoint.
        int8_checkpoint_path: Destination path for the INT8 model file.
        batch_size: Batch size used during inference.
        output_path: Destination path for the comparison figure.
    """

    def __init__(
        self,
        test_dir: str | Path,
        config: TransformerTrainingConfig = None,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        int8_checkpoint_path: str | Path = INT8_CHECKPOINT_PATH,
        batch_size: int = EVAL_BATCH_SIZE,
        output_path: str | Path = "outputs/figures/quantization/quantization_int8_comparison.png",
    ) -> None:
        self._test_dir = str(test_dir)
        self._config = config or TransformerTrainingConfig()
        self._factory = ModelFactory(
            self._config, checkpoint_path, int8_checkpoint_path
        )
        self._benchmark = InferenceBenchmark(batch_size=batch_size)
        self._output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[QuantizationStats]:
        """Benchmark both variants and return the collected statistics.

        Returns:
            A list with two ``QuantizationStats``: FP32 first, INT8 second.
        """
        all_stats = []

        # --- FP32 ---
        print("\n--- Benchmarking FP32 ---")
        model_fp32 = self._factory.load_fp32()
        total_params, _ = ModelFactory.compute_param_stats(model_fp32)
        disk_mb_fp32 = ModelFactory.disk_size_mb(self._factory.checkpoint_path)

        acc_fp32, elapsed_fp32 = self._benchmark.run(model_fp32, self._test_dir)
        all_stats.append(
            QuantizationStats(
                label=VARIANT_LABEL["fp32"],
                total_params=total_params,
                disk_size_mb=disk_mb_fp32,
                accuracy=acc_fp32,
                elapsed_sec=elapsed_fp32,
            )
        )
        self._print_stats(all_stats[-1])

        # --- INT8 ---
        print("\n--- Benchmarking INT8 (Dynamic) ---")
        model_int8 = self._factory.quantize_int8(model_fp32)
        int8_path = self._factory.save_int8(model_int8)
        model_int8 = self._factory.load_int8()
        total_params_int8, _ = ModelFactory.compute_param_stats(model_int8)
        disk_mb_int8 = ModelFactory.disk_size_mb(int8_path)

        acc_int8, elapsed_int8 = self._benchmark.run(model_int8, self._test_dir)
        all_stats.append(
            QuantizationStats(
                label=VARIANT_LABEL["int8"],
                total_params=total_params_int8,
                disk_size_mb=disk_mb_int8,
                accuracy=acc_int8,
                elapsed_sec=elapsed_int8,
            )
        )
        self._print_stats(all_stats[-1])

        self._print_summary(all_stats)
        return all_stats

    def compare(self) -> None:
        """Run all benchmarks and save the comparison figure to disk."""
        all_stats = self.run()
        QuantizationPlotter().plot(all_stats, self._output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_stats(stats: QuantizationStats) -> None:
        """Print the benchmark results for one variant.

        Args:
            stats: Results to display.
        """
        print(
            f"  Accuracy   : {stats.accuracy:.2f}%\n"
            f"  Disk size  : {stats.disk_size_mb:.2f} MB\n"
            f"  Time       : {stats.elapsed_sec:.2f}s\n"
            f"  Params     : {stats.total_params:,}"
        )

    @staticmethod
    def _print_summary(stats: list[QuantizationStats]) -> None:
        """Print a side-by-side summary with derived metrics.

        Args:
            stats: List containing FP32 stats at index 0 and INT8 at index 1.
        """
        fp32, int8 = stats[0], stats[1]
        size_reduction = (1.0 - int8.disk_size_mb / fp32.disk_size_mb) * 100.0
        speedup = fp32.elapsed_sec / max(1e-9, int8.elapsed_sec)
        acc_delta = int8.accuracy - fp32.accuracy

        print("\n================ RESUMEN ================")
        print(
            f"FP32 -> Acc: {fp32.accuracy:.2f}% | "
            f"Time: {fp32.elapsed_sec:.2f}s | "
            f"Disk: {fp32.disk_size_mb:.2f} MB"
        )
        print(
            f"INT8 -> Acc: {int8.accuracy:.2f}% | "
            f"Time: {int8.elapsed_sec:.2f}s | "
            f"Disk: {int8.disk_size_mb:.2f} MB"
        )
        print("---")
        print(f"Reducción de tamaño : {size_reduction:.1f}%")
        print(f"Speedup en CPU      : {speedup:.2f}x")
        print(f"Diferencia accuracy : {acc_delta:+.2f}%")
        print("=========================================")
