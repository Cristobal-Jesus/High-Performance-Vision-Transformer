"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/int1/comparator.py

Description:
    Orchestrator that benchmarks the FP32 VisionTransformer baseline
    against its INT1 binary weight quantized variant and saves a
    comparison figure.

    INT1 simulated quantization (XNOR-Net):
        Each ``nn.Linear`` weight tensor is replaced by
        ``sign(W) * mean(|W|)``.  Inference runs on GPU.  The reported
        model size is the theoretical packed size (1 bit per weight value
        plus one FP32 scale factor per layer).
"""

from __future__ import annotations

from pathlib import Path

from transformer.training.config import TransformerTrainingConfig
from .benchmark import InferenceBenchmark
from .constants import CHECKPOINT_PATH, EVAL_BATCH_SIZE, VARIANT_LABEL
from .model_factory import ModelFactory
from .plotter import QuantizationPlotter
from .quantization_stats import QuantizationStats


class QuantizationComparator:
    """Benchmark FP32 vs INT1 binary quantization and generate a report.

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
            "outputs/figures/quantization/quantization_int1_comparison.png"
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
            A list with two ``QuantizationStats``: FP32 first, INT1 second.
        """
        all_stats: list[QuantizationStats] = []

        print("\n--- Benchmarking FP32 ---")
        model_fp32 = self._factory.load_fp32()
        total_params, _ = ModelFactory.compute_param_stats(model_fp32)
        disk_mb_fp32 = ModelFactory.disk_size_mb(self._factory.checkpoint_path)
        acc_fp32, elapsed_fp32 = self._benchmark.run(model_fp32, self._test_dir)
        all_stats.append(QuantizationStats(
            label=VARIANT_LABEL["fp32"],
            total_params=total_params,
            disk_size_mb=disk_mb_fp32,
            accuracy=acc_fp32,
            elapsed_sec=elapsed_fp32,
        ))
        self._print_stats(all_stats[-1])

        print("\n--- Benchmarking INT1 (Binary, XNOR-Net) ---")
        model_int1 = self._factory.quantize_int1(model_fp32)
        theo_mb = self._factory.theoretical_size_mb(model_fp32)
        acc_int1, elapsed_int1 = self._benchmark.run(model_int1, self._test_dir)
        all_stats.append(QuantizationStats(
            label=VARIANT_LABEL["int1"],
            total_params=total_params,
            disk_size_mb=theo_mb,
            accuracy=acc_int1,
            elapsed_sec=elapsed_int1,
        ))
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
            stats: List with FP32 at index 0 and INT1 at index 1.
        """
        fp32, int1 = stats[0], stats[1]
        size_reduction = (1.0 - int1.disk_size_mb / fp32.disk_size_mb) * 100.0
        speedup = fp32.elapsed_sec / max(1e-9, int1.elapsed_sec)
        acc_delta = int1.accuracy - fp32.accuracy

        print("\n================ RESUMEN ================")
        print(
            f"FP32  -> Acc: {fp32.accuracy:.2f}% | "
            f"Time: {fp32.elapsed_sec:.2f}s | "
            f"Size: {fp32.disk_size_mb:.2f} MB"
        )
        print(
            f"INT1  -> Acc: {int1.accuracy:.2f}% | "
            f"Time: {int1.elapsed_sec:.2f}s | "
            f"Size: {int1.disk_size_mb:.2f} MB"
        )
        print("---")
        print(f"Reducción de tamaño : {size_reduction:.1f}%")
        print(f"Speedup             : {speedup:.2f}x")
        print(f"Diferencia accuracy : {acc_delta:+.2f}%")
        print("=========================================")
