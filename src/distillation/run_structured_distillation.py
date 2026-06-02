"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/run_structured_distillation.py

Description:
    Destilación del modelo con poda estructurada al 25 % y al 50 %.

    Para cada ratio:
        1. Carga el modelo original (profesor).
        2. Aplica la poda estructurada al ratio indicado (estudiante).
        3. Guarda el modelo podado como checkpoint intermedio.
        4. Ejecuta Knowledge Distillation durante cfg.epochs épocas.
        5. Guarda el estudiante destilado y genera la figura de curvas.

    El profesor es siempre el modelo original sin podar (90 % acc).
    El estudiante recupera parte de la accuracy perdida por la poda
    aprendiendo de las distribuciones suaves del profesor.

Uso:
    PYTHONPATH=src python -m distillation.run_structured_distillation
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.optim as optim

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory
from transformer.training.schedulers import SchedulerFactory

from pruning.common.checkpoint import load_vision_transformer, save_model
from pruning.structured.head_pruner import HeadPruner

from .config import DistillationConfig
from .distiller import Distiller
from .losses import KnowledgeDistillationLoss
from .visualization.plotter import DistillationPlotter

# Ratios a destilar
PRUNE_RATIOS = [0.25, 0.50]


def _build_data_module(cfg: DistillationConfig) -> DaliDataModule:
    """Construye el módulo DALI."""
    return DaliDataModule(
        root_dir=cfg.root_dir,
        train_batch_size=cfg.train_batch_size,
        val_batch_size=cfg.val_batch_size,
        image_size=cfg.image_size,
        num_threads=cfg.num_threads,
        device_id=cfg.device_id,
        drop_last=cfg.drop_last,
        train_prefetch_queue_depth=cfg.train_prefetch_queue_depth,
        val_prefetch_queue_depth=cfg.val_prefetch_queue_depth,
        validator_batch_size=cfg.val_batch_size,
        pipeline_factory=DaliPipelineFactory(),
    )


def _load(cfg: DistillationConfig, path: Path, device: torch.device):
    """Carga un VisionTransformer desde ``path``."""
    return load_vision_transformer(
        path,
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


def _run_one(cfg: DistillationConfig, ratio: float, device: torch.device) -> None:
    """Poda al ``ratio`` dado y destila. Guarda modelo y figura."""
    tag = f"structured_{int(ratio * 100)}"
    print(f"\n{'='*60}")
    print(f"  Poda estructurada {int(ratio*100)}%  →  Destilación")
    print(f"{'='*60}")

    # ── 1. Profesor (original, frozen) ────────────────────────────────
    print("[KD] Cargando profesor…")
    teacher = _load(cfg, cfg.teacher_path, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── 2. Estudiante (podado) ─────────────────────────────────────────
    print(f"[KD] Aplicando poda estructurada {int(ratio*100)}%…")
    student = _load(cfg, cfg.teacher_path, device)   # parte del original
    pruner = HeadPruner(prune_ratio=ratio)
    student, heads_pruned = pruner.prune(student)
    print(f"[KD] Cabezas podadas: {heads_pruned}")

    # Guardar modelo podado (sin destilar) para referencia
    pruned_path = Path(f"checkpoints/pruning/structured/{tag}.pth")
    save_model(student, pruned_path, metadata={"prune_ratio": ratio,
                                                "heads_pruned": heads_pruned})

    # ── 3. Dataloaders ─────────────────────────────────────────────────
    print("[KD] Construyendo dataloaders…")
    dm = _build_data_module(cfg)
    dm.setup()

    # ── 4. Optimizador y scheduler ─────────────────────────────────────
    optimizer = optim.AdamW(
        student.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = SchedulerFactory.build_warmup_cosine_scheduler(
        optimizer=optimizer,
        epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
    )

    # ── 5. Destilación ─────────────────────────────────────────────────
    criterion = KnowledgeDistillationLoss(
        temperature=cfg.temperature,
        alpha=cfg.alpha,
        label_smoothing=cfg.label_smoothing,
    )
    plotter = DistillationPlotter()

    output_path = Path(f"checkpoints/distillation/{tag}_distilled.pth")
    figure_path = Path(f"outputs/figures/distillation/{tag}_curves.png")

    distiller = Distiller(
        teacher=teacher,
        student=student,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_path=output_path,
        plotter=plotter,
        validate_every=cfg.validate_every,
        max_val_batches=cfg.max_val_batches,
        grad_clip_norm=cfg.grad_clip_norm,
    )

    print(f"[KD] Iniciando destilación ({cfg.epochs} épocas)…")
    result = distiller.fit(cfg.epochs)
    dm.teardown()

    # ── 6. Resultados y figura ──────────────────────────────────────────
    print(f"\n[KD] Mejor val_loss : {result['best_val_loss']:.4f}")
    print(f"[KD] Mejor epoch    : {result['best_epoch']}")
    print(f"[KD] Checkpoint     : {output_path}")

    plotter.plot(save_path=figure_path, teacher_accuracy=None)


def main() -> None:
    """Ejecuta la destilación para los ratios 25% y 50%."""
    cfg    = DistillationConfig()
    device = torch.device("cuda", cfg.device_id)

    for ratio in PRUNE_RATIOS:
        _run_one(cfg, ratio, device)

    print("\n[KD] Destilación estructurada completada.")


if __name__ == "__main__":
    main()
