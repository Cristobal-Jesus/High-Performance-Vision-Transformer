"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/compare_all/comparator.py

Description:
    Orchestrator that benchmarks all seven quantization variants
    (FP32, FP16, BF16, INT8, INT4, INT2, INT1) of the VisionTransformer
    using a single FP32 model load and returns a unified list of
    ``VariantStats``.

    Device policy:
        - FP32 and INT8 run on CPU (``quantize_dynamic`` is CPU-only).
        - FP16 and BF16 run on GPU (16-bit inference needs CUDA).
        - INT4, INT2 and INT1 run on GPU when CUDA is available because
          simulated quantization produces standard FP32 models.

    Size policy:
        - FP32 and INT8: actual checkpoint / saved-model file size.
        - FP16, BF16: actual in-memory footprint (2 bytes per parameter).
        - INT4, INT2, INT1: theoretical packed size (N bits per weight).
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import torch

from transformer.training.config import TransformerTrainingConfig

# INT8 (dynamic, CPU-only)
from quantization.int8.model_factory import (
    ModelFactory as Int8Factory,
)
from quantization.int8.benchmark import InferenceBenchmark as Int8Benchmark
from quantization.int8.constants import INT8_CHECKPOINT_PATH

# INT4 / INT2 / INT1 (simulated, GPU)
from quantization.int4.model_factory import ModelFactory as Int4Factory
from quantization.int4.benchmark import InferenceBenchmark as Int4Benchmark
from quantization.int2.model_factory import ModelFactory as Int2Factory
from quantization.int2.benchmark import InferenceBenchmark as Int2Benchmark
from quantization.int1.model_factory import ModelFactory as Int1Factory
from quantization.int1.benchmark import InferenceBenchmark as Int1Benchmark

from .stats import VariantStats


class AllQuantizationComparator:
    """Benchmark FP32, FP16, BF16, INT8, INT4, INT2 and INT1 in a single run.

    Args:
        test_dir: Flat directory of 20-class test images.
        config: Training configuration for architecture reconstruction.
            Defaults to ``TransformerTrainingConfig()``.
        checkpoint_path: Path to the FP32 (or EMA) model checkpoint.
        batch_size: Inference batch size for all variants.
    """

    def __init__(
        self,
        test_dir: str | Path,
        config: TransformerTrainingConfig | None = None,
        checkpoint_path: str | Path = "checkpoints/transformer/best_transformer_e384.pth",
        batch_size: int = 64,
    ) -> None:
        self._test_dir = str(test_dir)
        self._config = config or TransformerTrainingConfig()
        self._checkpoint_path = Path(checkpoint_path)
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[VariantStats]:
        """Benchmark all variants and return collected statistics.

        Returns:
            List of ``VariantStats`` in order:
            FP32, FP16, BF16, INT8, INT4, INT2, INT1.
        """
        all_stats: list[VariantStats] = []

        # --- FP32 baseline (shared model load) ---
        print("\n[1/7] FP32 baseline (CPU)...")
        int4_factory = Int4Factory(self._config, self._checkpoint_path)
        model_fp32 = int4_factory.load_fp32()
        fp32_size = Int4Factory.disk_size_mb(self._checkpoint_path)
        acc_fp32, _ = Int4Benchmark(batch_size=self._batch_size).run(
            model_fp32, self._test_dir
        )
        all_stats.append(VariantStats(
            label="FP32",
            full_label="FP32",
            bits=32,
            accuracy=acc_fp32,
            size_mb=fp32_size,
            theoretical=False,
            device="GPU",
        ))
        self._log(all_stats[-1])

        # --- FP16 and BF16 (cast of the already-loaded FP32 model) ---
        self._add_16bit_variants(model_fp32, all_stats)

        # --- INT8 dynamic (CPU-only) ---
        print("\n[4/7] INT8 Dynamic (CPU)...")
        int8_factory = Int8Factory(self._config, self._checkpoint_path, INT8_CHECKPOINT_PATH)
        model_int8 = int8_factory.quantize_int8(int8_factory.load_fp32())
        int8_path = int8_factory.save_int8(model_int8)
        model_int8 = int8_factory.load_int8()
        int8_size = Int8Factory.disk_size_mb(int8_path)
        acc_int8, _ = Int8Benchmark(batch_size=self._batch_size).run(
            model_int8, self._test_dir
        )
        all_stats.append(VariantStats(
            label="INT8",
            full_label="INT8 (Dynamic)",
            bits=8,
            accuracy=acc_int8,
            size_mb=int8_size,
            theoretical=False,
            device="CPU",
        ))
        self._log(all_stats[-1])

        # --- INT4 (simulated, GPU) ---
        print("\n[5/7] INT4 W4A32 (GPU)...")
        model_int4 = int4_factory.quantize_int4(model_fp32)
        acc_int4, _ = Int4Benchmark(batch_size=self._batch_size).run(
            model_int4, self._test_dir
        )
        all_stats.append(VariantStats(
            label="INT4",
            full_label="INT4 (W4A32)",
            bits=4,
            accuracy=acc_int4,
            size_mb=int4_factory.theoretical_size_mb(model_fp32),
            theoretical=True,
            device="GPU",
        ))
        self._log(all_stats[-1])

        # --- INT2 (simulated, GPU) ---
        print("\n[6/7] INT2 W2A32 (GPU)...")
        int2_factory = Int2Factory(self._config, self._checkpoint_path)
        model_int2 = int2_factory.quantize_int2(model_fp32)
        acc_int2, _ = Int2Benchmark(batch_size=self._batch_size).run(
            model_int2, self._test_dir
        )
        all_stats.append(VariantStats(
            label="INT2",
            full_label="INT2 (W2A32)",
            bits=2,
            accuracy=acc_int2,
            size_mb=int2_factory.theoretical_size_mb(model_fp32),
            theoretical=True,
            device="GPU",
        ))
        self._log(all_stats[-1])

        # --- INT1 binary (simulated, GPU) ---
        print("\n[7/7] INT1 Binary XNOR-Net (GPU)...")
        int1_factory = Int1Factory(self._config, self._checkpoint_path)
        model_int1 = int1_factory.quantize_int1(model_fp32)
        acc_int1, _ = Int1Benchmark(batch_size=self._batch_size).run(
            model_int1, self._test_dir
        )
        all_stats.append(VariantStats(
            label="INT1",
            full_label="INT1 (Binary)",
            bits=1,
            accuracy=acc_int1,
            size_mb=int1_factory.theoretical_size_mb(model_fp32),
            theoretical=True,
            device="GPU",
        ))
        self._log(all_stats[-1])

        self._print_summary(all_stats)
        return all_stats

    # ------------------------------------------------------------------
    # 16-bit variants (FP16 / BF16)
    # ------------------------------------------------------------------

    def _add_16bit_variants(
        self,
        model_fp32,
        all_stats: list[VariantStats],
    ) -> None:
        """Benchmark FP16 and BF16 by casting the already-loaded FP32 model.

        Reutiliza el modelo FP32 ya cargado (sin el prefijo ``_orig_mod.`` de
        ``torch.compile``) y lo castea a cada formato de 16 bits.  La inferencia
        usa el benchmark del módulo ``16bits``, que adapta el dtype de las
        imágenes de entrada al del modelo.

        El módulo ``16bits`` no puede importarse con ``import`` normal porque
        su nombre empieza por dígito, así que se carga con ``importlib``.

        Args:
            model_fp32: Modelo FP32 en CPU, ya con los pesos cargados.
            all_stats: Lista de resultados a la que se añaden FP16 y BF16.
        """
        benchmark_module = importlib.import_module("quantization.16bits.benchmark")
        InferenceBenchmark16 = benchmark_module.InferenceBenchmark

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = self._build_test_loader(device)
        benchmark = InferenceBenchmark16(patch_size=self._config.patch_size)

        variants = (
            ("[2/7]", "FP16", torch.float16),
            ("[3/7]", "BF16", torch.bfloat16),
        )
        for position, label, torch_dtype in variants:
            print(f"\n{position} {label} (GPU)...")

            model = copy.deepcopy(model_fp32).to(device=device, dtype=torch_dtype)
            model.eval()

            # Tamaño real en memoria: cada parámetro ocupa 2 bytes en 16 bits.
            size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 ** 2)

            accuracy, _ = benchmark.run(model, loader, torch_dtype, device)

            all_stats.append(VariantStats(
                label=label,
                full_label=label,
                bits=16,
                accuracy=accuracy,
                size_mb=size_mb,
                theoretical=False,
                device="GPU" if device.type == "cuda" else "CPU",
            ))
            self._log(all_stats[-1])

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _build_test_loader(self, device: torch.device):
        """Construye un ``DataLoader`` sobre el directorio plano de test.

        Reutiliza el ``Dataset`` y las constantes del módulo INT4, que ya
        codifica la etiqueta de cada imagen en su nombre de fichero.

        Args:
            device: Dispositivo de inferencia (para fijar ``pin_memory``).

        Returns:
            ``DataLoader`` que itera ``(images, labels)`` sin barajar.
        """
        from torch.utils.data import DataLoader
        from torchvision import transforms

        from quantization.int4.benchmark import _FlatLabeledDataset
        from quantization.int4.constants import (
            IMAGE_SIZE,
            IMAGENET_MEAN,
            IMAGENET_STD,
            LABEL_MAP,
        )

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ])
        dataset = _FlatLabeledDataset(self._test_dir, LABEL_MAP, transform)

        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log(stats: VariantStats) -> None:
        """Print a one-line result for one variant.

        Args:
            stats: Results to display.
        """
        size_note = "(theoretical)" if stats.theoretical else "(actual)"
        print(
            f"  {stats.full_label:<22} | "
            f"Acc: {stats.accuracy:.2f}% | "
            f"Size: {stats.size_mb:.1f} MB {size_note} | "
            f"Device: {stats.device}"
        )

    @staticmethod
    def _print_summary(stats: list[VariantStats]) -> None:
        """Print a comparison table for all variants.

        Args:
            stats: Full list of collected statistics.
        """
        fp32_acc = stats[0].accuracy
        fp32_size = stats[0].size_mb

        print("\n╔══════════════════════════════════════════════════════╗")
        print("║          RESUMEN DE CUANTIZACIÓN COMPLETO            ║")
        print("╠══════════════╦════════════╦════════════╦═════════════╣")
        print("║ Variante     ║ Accuracy   ║ Δ Accuracy ║ Compresión  ║")
        print("╠══════════════╬════════════╬════════════╬═════════════╣")
        for s in stats:
            delta = s.accuracy - fp32_acc
            ratio = fp32_size / max(s.size_mb, 1e-9)
            print(
                f"║ {s.full_label:<12} ║ {s.accuracy:>7.2f}%  ║ "
                f"{delta:>+8.2f}%  ║ {ratio:>8.1f}×   ║"
            )
        print("╚══════════════╩════════════╩════════════╩═════════════╝")
