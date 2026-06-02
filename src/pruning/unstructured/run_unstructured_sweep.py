"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/unstructured/run_unstructured_sweep.py

Description:
    Barrido de intensidad de poda no estructurada por magnitud.

    Evalúa el modelo con sparsity = 0.0, 0.20, 0.40, 0.60, 0.80, 0.90
    y genera una gráfica que muestra la curva accuracy vs esparsidad
    real conforme aumenta la fracción de pesos anulados.

Uso:
    PYTHONPATH=src python -m pruning.unstructured.run_unstructured_sweep
"""

from __future__ import annotations

import torch

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory

from pruning.common.checkpoint import load_vision_transformer
from pruning.common.evaluator import evaluate, model_sparsity
from pruning.common.sweep_plotter import PruningSweepPlotter, SweepPoint

from .config import UnstructuredPruningConfig
from .weight_pruner import WeightPruner

# Fracciones de pesos a anular (0.0 = original)
SPARSITY_LEVELS = [0.0, 0.20, 0.40, 0.60, 0.80, 0.90]


def _build_val_loader(cfg: UnstructuredPruningConfig, device_id: int):
    """Construye el iterador DALI de validación."""
    dm = DaliDataModule(
        root_dir=cfg.root_dir,
        train_batch_size=cfg.val_batch_size,
        val_batch_size=cfg.val_batch_size,
        image_size=cfg.image_size,
        num_threads=cfg.num_threads,
        device_id=device_id,
        drop_last=False,
        train_prefetch_queue_depth=2,
        val_prefetch_queue_depth=2,
        validator_batch_size=cfg.val_batch_size,
        pipeline_factory=DaliPipelineFactory(),
    )
    dm.setup()
    return dm, dm.val_dataloader()


def main() -> None:
    """Ejecuta el barrido de poda no estructurada."""
    cfg = UnstructuredPruningConfig()
    device = torch.device("cuda", cfg.device_id)

    points: list[SweepPoint] = []
    baseline_accuracy: float | None = None

    for sparsity in SPARSITY_LEVELS:
        label = f"{int(sparsity*100)}%" if sparsity > 0 else "Original"
        print(f"\n[Unstructured sweep] Evaluando sparsity={sparsity:.2f} ({label})...")

        model = load_vision_transformer(
            cfg.checkpoint_path,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            drop_path_rate=cfg.drop_path_rate,
            device=device,
        )

        if sparsity > 0:
            pruner = WeightPruner(sparsity=sparsity)
            model = pruner.prune(model)

        real_sparsity = model_sparsity(model)

        dm, val_loader = _build_val_loader(cfg, cfg.device_id)
        acc, ms = evaluate(
            model, val_loader, device,
            patch_size=cfg.patch_size,
            max_batches=cfg.max_val_batches,
        )
        dm.teardown()

        if baseline_accuracy is None:
            baseline_accuracy = acc

        print(f"  Accuracy       : {acc:.2f}%")
        print(f"  Sparsidad real : {real_sparsity:.4f}")

        points.append(SweepPoint(
            param_value=sparsity * 100,
            accuracy=acc,
            secondary_metric=real_sparsity * 100,
            label=label,
        ))

    PruningSweepPlotter(
        x_label="Pesos anulados (% objetivo)",
        secondary_label="Esparsidad real del modelo (%)",
        title="Poda no estructurada por magnitud — Barrido de intensidad",
    ).plot(
        points=points,
        output_path="outputs/figures/pruning/unstructured/sweep.png",
        baseline_accuracy=baseline_accuracy,
    )

    print("\n[Unstructured sweep] Completado.")


if __name__ == "__main__":
    main()
