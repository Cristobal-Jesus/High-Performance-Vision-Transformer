"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int4/comparator.py

Description:
    Orchestrator that benchmarks the FP32 VisionTransformer baseline
    against its INT4 weight-only quantized variant and saves a comparison
    figure.

    INT4 simulated quantization:
        Each ``nn.Linear`` weight tensor is mapped to the signed 4-bit
        range [-8, 7] with a symmetric per-tensor scale, then dequantized
        to FP32.  Inference runs on GPU.  The reported model size is the
        theoretical packed size (4 bits per weight value).
"""

from __future__ import annotations

from pathlib import Path

from transformer.training.config import TransformerTrainingConfig
from .benchmark import InferenceBenchmark
from .constants import (
    CHECKPOINT_PATH,
    EVAL_BATCH_SIZE,
    VARIANT_LABEL,
)
from .model_factory import ModelFactory
from .plotter import QuantizationPlotter
from .quantization_stats import QuantizationStats


class QuantizationComparator:
    """Benchmark FP32 vs INT4 weight quantization and generate a report.

    Args:
        test_dir: Flat directory of 20-class test images.
        config: Training configuration for model reconstruction.
            Defaults to ``TransformerTrainingConfig()``.
        checkpoint_path: Path to the FP32 model checkpoint.
        batch_size: Batch size used during inference.
        output_path: Destination path for the comparison figure.
    """

    def __init__(
        self,
        test_dir: str | Path,
        config: TransformerTrainingConfig | None = None,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        batch_size: int = EVAL_BATCH_SIZE,
        output_path: str | Path = (
            "outputs/figures/quantization/quantization_int4_comparison.png"
        ),
    ) -> None:
        self._test_dir = str(test_dir)
        self._config = config or TransformerTrainingConfig()
        self._factory = ModelFactory(self._config, checkpoint_path)
        self._benchmark = InferenceBenchmark(batch_size=batch_size)
        self._output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[QuantizationStats]:
        """Benchmark both variants and return collected statistics.

        Returns:
            A list with two ``QuantizationStats``: FP32 first, INT4 second.
        """
        all_stats: list[QuantizationStats] = []

        # --- FP32 baseline ---
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

        # --- INT4 ---
        print("\n--- Benchmarking INT4 (W4A32) ---")
        model_int4 = self._factory.quantize_int4(model_fp32)
        theo_mb = self._factory.theoretical_size_mb(model_fp32)
        acc_int4, elapsed_int4 = self._benchmark.run(model_int4, self._test_dir)
        all_stats.append(
            QuantizationStats(
                label=VARIANT_LABEL["int4"],
                total_params=total_params,
                disk_size_mb=theo_mb,
                accuracy=acc_int4,
                elapsed_sec=elapsed_int4,
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
        """Print benchmark results for one variant.

        Args:
            stats: Results to display.
        """
        print(
            f"  Accuracy  : {stats.accuracy:.2f}%\n"
            f"  Size      : {stats.disk_size_mb:.2f} MB\n"
            f"  Time      : {stats.elapsed_sec:.2f}s\n"
            f"  Params    : {stats.total_params:,}"
        )

    @staticmethod
    def _print_summary(stats: list[QuantizationStats]) -> None:
        """Print a side-by-side summary with derived metrics.

        Args:
            stats: List with FP32 at index 0 and INT4 at index 1.
        """
        fp32, int4 = stats[0], stats[1]
        size_reduction = (1.0 - int4.disk_size_mb / fp32.disk_size_mb) * 100.0
        speedup = fp32.elapsed_sec / max(1e-9, int4.elapsed_sec)
        acc_delta = int4.accuracy - fp32.accuracy

        print("\n================ RESUMEN ================")
        print(
            f"FP32  -> Acc: {fp32.accuracy:.2f}% | "
            f"Time: {fp32.elapsed_sec:.2f}s | "
            f"Size: {fp32.disk_size_mb:.2f} MB"
        )
        print(
            f"INT4  -> Acc: {int4.accuracy:.2f}% | "
            f"Time: {int4.elapsed_sec:.2f}s | "
            f"Size: {int4.disk_size_mb:.2f} MB"
        )
        print("---")
        print(f"Reducción de tamaño : {size_reduction:.1f}%")
        print(f"Speedup             : {speedup:.2f}x")
        print(f"Diferencia accuracy : {acc_delta:+.2f}%")
        print("=========================================")
