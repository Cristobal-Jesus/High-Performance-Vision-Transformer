"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/tome/run_tome_sweep.py

Description:
    Barrido de intensidad de Token Merging.

    Evalúa el modelo con r = 0 (original), 4, 8, 12, 16, 20 y genera
    una gráfica que muestra la curva accuracy vs speedup conforme
    aumenta el número de tokens fusionados por bloque.

Uso:
    PYTHONPATH=src python -m pruning.tome.run_tome_sweep
"""

from __future__ import annotations

import torch

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory

from pruning.common.checkpoint import load_vision_transformer
from pruning.common.evaluator import evaluate
from pruning.common.sweep_plotter import PruningSweepPlotter, SweepPoint

from .config import ToMeConfig
from .wrapper import apply_tome

# Valores de r a explorar (0 = modelo original sin ToMe)
R_VALUES = [0, 4, 8, 12, 16, 20]


def _build_val_loader(cfg: ToMeConfig, device_id: int):
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
    """Ejecuta el barrido de Token Merging."""
    cfg = ToMeConfig()
    device = torch.device("cuda", cfg.device_id)

    points: list[SweepPoint] = []

    for r in R_VALUES:
        label = f"r={r}" if r > 0 else "Original"
        print(f"\n[ToMe sweep] Evaluando {label}...")

        # Cargar modelo limpio en cada iteración
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

        if r > 0:
            apply_tome(model, r=r)

        dm, val_loader = _build_val_loader(cfg, cfg.device_id)
        acc, ms = evaluate(
            model, val_loader, device,
            patch_size=cfg.patch_size,
            max_batches=cfg.max_val_batches,
        )
        dm.teardown()

        # Para r=0 guardamos ms de referencia para calcular speedup
        if r == 0:
            ms_baseline = ms

        speedup = ms_baseline / ms if ms > 0 else 1.0
        tokens_final = (cfg.image_size // cfg.patch_size) ** 2 - r * cfg.depth

        print(f"  Accuracy : {acc:.2f}%")
        print(f"  ms/img   : {ms:.3f}")
        print(f"  Speedup  : {speedup:.2f}x")
        print(f"  Tokens   : {tokens_final}")

        points.append(SweepPoint(
            param_value=r,
            accuracy=acc,
            secondary_metric=speedup,
            label=label,
        ))

    PruningSweepPlotter(
        x_label="r (tokens fusionados por bloque)",
        secondary_label="Speedup (× respecto al original)",
        title="Token Merging — Barrido de intensidad (r)",
    ).plot(
        points=points,
        output_path="outputs/figures/pruning/tome/sweep.png",
        baseline_accuracy=points[0].accuracy,
    )

    print("\n[ToMe sweep] Completado.")


if __name__ == "__main__":
    main()
