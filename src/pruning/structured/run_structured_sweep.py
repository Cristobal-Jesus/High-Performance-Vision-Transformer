"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/structured/run_structured_sweep.py

Description:
    Barrido de intensidad de poda estructurada de cabezas.

    Evalúa el modelo con prune_ratio = 0.0, 0.10, 0.25, 0.50, 0.75
    y genera una gráfica que muestra la curva accuracy vs esparsidad
    conforme aumenta la fracción de cabezas podadas.

Uso:
    PYTHONPATH=src python -m pruning.structured.run_structured_sweep
"""

from __future__ import annotations

import torch

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory

from pruning.common.checkpoint import load_vision_transformer
from pruning.common.evaluator import evaluate, model_sparsity
from pruning.common.sweep_plotter import PruningSweepPlotter, SweepPoint

from .config import StructuredPruningConfig
from .head_pruner import HeadPruner

# Fracciones de cabezas a podar (0.0 = original)
PRUNE_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75]


def _build_val_loader(cfg: StructuredPruningConfig, device_id: int):
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
    """Ejecuta el barrido de poda estructurada."""
    cfg = StructuredPruningConfig()
    device = torch.device("cuda", cfg.device_id)

    points: list[SweepPoint] = []
    baseline_accuracy: float | None = None

    for ratio in PRUNE_RATIOS:
        label = f"{int(ratio*100)}%" if ratio > 0 else "Original"
        print(f"\n[Structured sweep] Evaluando ratio={ratio:.2f} ({label})...")

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

        if ratio > 0:
            pruner = HeadPruner(prune_ratio=ratio)
            model, heads_pruned = pruner.prune(model)
            print(f"  Cabezas podadas: {heads_pruned}")

        sparsity = model_sparsity(model)

        dm, val_loader = _build_val_loader(cfg, cfg.device_id)
        acc, ms = evaluate(
            model, val_loader, device,
            patch_size=cfg.patch_size,
            max_batches=cfg.max_val_batches,
        )
        dm.teardown()

        if baseline_accuracy is None:
            baseline_accuracy = acc

        print(f"  Accuracy  : {acc:.2f}%")
        print(f"  Sparsidad : {sparsity:.4f}")

        points.append(SweepPoint(
            param_value=ratio * 100,   # en porcentaje para el eje X
            accuracy=acc,
            secondary_metric=sparsity * 100,
            label=label,
        ))

    PruningSweepPlotter(
        x_label="Cabezas podadas por capa (%)",
        secondary_label="Esparsidad del modelo (%)",
        title="Poda estructurada de cabezas — Barrido de intensidad",
    ).plot(
        points=points,
        output_path="outputs/figures/pruning/structured/sweep.png",
        baseline_accuracy=baseline_accuracy,
    )

    print("\n[Structured sweep] Completado.")


if __name__ == "__main__":
    main()
