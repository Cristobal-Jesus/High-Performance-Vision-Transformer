"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/run_distillation.py

Description:
    Punto de entrada para el experimento de Knowledge Distillation.

    El profesor es el VisionTransformer original entrenado.
    El estudiante es el modelo podado guardado en pruning/.

    Flujo:
        1. Carga el profesor (frozen) y el estudiante desde sus checkpoints.
        2. Si cfg.tome_r > 0, aplica ToMe al estudiante en memoria.
        3. Construye los dataloaders DALI.
        4. Ejecuta el bucle de destilación (Distiller.fit).
        5. Genera la figura comparativa (loss + accuracy).
        6. Imprime métricas finales.

    Uso:
        python -m distillation.run_distillation

    Personalización:
        Edita ``DistillationConfig`` en ``distillation/config.py`` para
        cambiar las rutas de checkpoints, temperatura, alpha, epochs, etc.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory
from transformer.training.schedulers import SchedulerFactory

from pruning.common.checkpoint import load_vision_transformer
from pruning.tome.wrapper import apply_tome

from .config import DistillationConfig
from .distiller import Distiller
from .losses import KnowledgeDistillationLoss
from .visualization.plotter import DistillationPlotter


def _build_data_module(cfg: DistillationConfig) -> DaliDataModule:
    """Construye el módulo DALI con los parámetros de configuración."""
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


def _load_model(cfg: DistillationConfig, path, device: torch.device):
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


def main() -> None:
    """Ejecuta el pipeline completo de Knowledge Distillation."""
    cfg    = DistillationConfig()
    device = torch.device("cuda", cfg.device_id)

    # --- 1. Cargar modelos ---
    print("[KD] Cargando profesor…")
    teacher = _load_model(cfg, cfg.teacher_path, device)
    teacher.eval()

    print("[KD] Cargando estudiante…")
    student = _load_model(cfg, cfg.student_path, device)

    # Opcionalmente aplica ToMe al estudiante en memoria
    if cfg.tome_r > 0:
        print(f"[KD] Aplicando ToMe r={cfg.tome_r} al estudiante…")
        apply_tome(student, r=cfg.tome_r)

    # Poner al estudiante en modo entrenamiento
    student.train()

    # --- 2. Dataloaders ---
    print("[KD] Construyendo dataloaders…")
    dm = _build_data_module(cfg)
    dm.setup()

    # --- 3. Optimizador y scheduler ---
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

    # --- 4. Pérdida y plotter ---
    criterion = KnowledgeDistillationLoss(
        temperature=cfg.temperature,
        alpha=cfg.alpha,
        label_smoothing=cfg.label_smoothing,
    )
    plotter = DistillationPlotter()

    # --- 5. Entrenamiento ---
    print(f"[KD] Iniciando destilación ({cfg.epochs} épocas)…")
    distiller = Distiller(
        teacher=teacher,
        student=student,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_path=cfg.output_path,
        plotter=plotter,
        validate_every=cfg.validate_every,
        max_val_batches=cfg.max_val_batches,
        grad_clip_norm=cfg.grad_clip_norm,
    )

    result = distiller.fit(cfg.epochs)
    dm.teardown()

    # --- 6. Resultados ---
    print("\n" + "=" * 50)
    print(f"[KD] Mejor val_loss : {result['best_val_loss']:.4f}")
    print(f"[KD] Mejor epoch    : {result['best_epoch']}")
    print("=" * 50 + "\n")

    # --- 7. Figura ---
    plotter.plot(
        save_path=cfg.figure_path,
        teacher_accuracy=None,   # Se puede rellenar con la acc del profesor
    )


if __name__ == "__main__":
    main()
