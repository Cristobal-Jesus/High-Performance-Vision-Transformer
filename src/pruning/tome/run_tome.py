"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/tome/run_tome.py

Description:
    Punto de entrada para el experimento Token Merging (ToMe).

    Flujo:
        1. Carga el VisionTransformer desde el checkpoint entrenado.
        2. Evalúa el modelo original (línea base).
        3. Aplica ToMe (reemplaza bloques por ToMeEncoderBlock).
        4. Evalúa el modelo con ToMe.
        5. Imprime la comparativa y genera la figura.
        6. Guarda el modelo con ToMe para usarlo en distillation.

    Uso:
        python -m pruning.tome.run_tome
"""

from __future__ import annotations

import torch

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory

from pruning.common.checkpoint import load_vision_transformer, save_model
from pruning.common.evaluator import count_nonzero_params, evaluate, model_sparsity
from pruning.common.plotter import PruningPlotter
from pruning.common.stats import PruningStats

from .config import ToMeConfig
from .wrapper import apply_tome


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


def _print_comparison(original: PruningStats, tome: PruningStats) -> None:
    """Imprime la tabla comparativa en consola."""
    print("\n" + "=" * 55)
    print(f"{'Métrica':<30} {'Original':>10} {'ToMe':>10}")
    print("-" * 55)
    print(f"{'Accuracy (%):':<30} {original.accuracy:>10.2f} {tome.accuracy:>10.2f}")
    print(f"{'Parámetros (M):':<30} {original.total_params/1e6:>10.2f} {tome.total_params/1e6:>10.2f}")
    print(f"{'Tamaño (MB):':<30} {original.size_mb:>10.2f} {tome.size_mb:>10.2f}")
    print(f"{'ms/imagen:':<30} {original.ms_per_image:>10.2f} {tome.ms_per_image:>10.2f}")
    speedup = original.ms_per_image / tome.ms_per_image if tome.ms_per_image > 0 else 0
    print(f"{'Speedup:':<30} {speedup:>10.2f}×")
    print("=" * 55 + "\n")


def main() -> None:
    """Ejecuta el pipeline completo de ToMe."""
    cfg = ToMeConfig()
    device = torch.device("cuda", cfg.device_id)

    # --- 1. Cargar modelo original ---
    print("[ToMe] Cargando modelo original…")
    model_orig = load_vision_transformer(
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

    # --- 2. Construir dataloader ---
    print("[ToMe] Construyendo dataloader de validación…")
    dm, val_loader = _build_val_loader(cfg, cfg.device_id)

    # --- 3. Evaluar modelo original ---
    print("[ToMe] Evaluando modelo original…")
    acc_orig, ms_orig = evaluate(
        model_orig, val_loader, device,
        patch_size=cfg.patch_size, max_batches=cfg.max_val_batches,
    )
    total_orig, nz_orig = count_nonzero_params(model_orig)
    stats_orig = PruningStats(
        label="Original",
        accuracy=acc_orig,
        total_params=total_orig,
        nonzero_params=nz_orig,
        ms_per_image=ms_orig,
    )

    # --- 4. Aplicar ToMe ---
    print(f"[ToMe] Aplicando Token Merging con r={cfg.r}…")
    # Reconstruir el modelo para no modificar el original
    model_tome = load_vision_transformer(
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
    apply_tome(model_tome, r=cfg.r)

    # --- 5. Evaluar modelo con ToMe ---
    print("[ToMe] Evaluando modelo con ToMe…")
    # Reiniciar iterador DALI
    dm.teardown()
    dm, val_loader = _build_val_loader(cfg, cfg.device_id)
    acc_tome, ms_tome = evaluate(
        model_tome, val_loader, device,
        patch_size=cfg.patch_size, max_batches=cfg.max_val_batches,
    )
    total_tome, nz_tome = count_nonzero_params(model_tome)
    stats_tome = PruningStats(
        label=f"ToMe r={cfg.r}",
        accuracy=acc_tome,
        total_params=total_tome,
        nonzero_params=nz_tome,
        ms_per_image=ms_tome,
    )

    dm.teardown()

    # --- 6. Imprimir comparativa ---
    _print_comparison(stats_orig, stats_tome)

    # --- 7. Guardar modelo ToMe ---
    save_model(
        model_tome,
        cfg.output_path,
        metadata={"tome_r": cfg.r, "accuracy": acc_tome},
    )

    # --- 8. Generar figura ---
    PruningPlotter().plot(
        stats=[stats_orig, stats_tome],
        output_path=cfg.figure_path,
        title=f"Token Merging (r={cfg.r}) vs Original",
    )


if __name__ == "__main__":
    main()
