"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: pruning/unstructured/run_unstructured.py

Description:
    Punto de entrada para la poda no estructurada por magnitud.

    Flujo:
        1. Carga el VisionTransformer desde el checkpoint.
        2. Evalúa la línea base (modelo original).
        3. Poda los pesos de menor magnitud según ``sparsity``.
        4. Evalúa el modelo podado.
        5. Imprime la comparativa y genera la figura.
        6. Guarda el modelo podado para la destilación.

    Uso:
        python -m pruning.unstructured.run_unstructured
"""

from __future__ import annotations

import torch

from transformer.data.dali.datamodule import DaliDataModule
from transformer.data.dali.pipelines import DaliPipelineFactory

from pruning.common.checkpoint import load_vision_transformer, save_model
from pruning.common.evaluator import count_nonzero_params, evaluate, model_sparsity
from pruning.common.plotter import PruningPlotter
from pruning.common.stats import PruningStats

from .config import UnstructuredPruningConfig
from .weight_pruner import WeightPruner


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


def _print_comparison(original: PruningStats, pruned: PruningStats) -> None:
    """Imprime la tabla comparativa en consola."""
    print("\n" + "=" * 60)
    print(f"{'Métrica':<30} {'Original':>14} {'Podado':>14}")
    print("-" * 60)
    print(f"{'Accuracy (%):':<30} {original.accuracy:>14.2f} {pruned.accuracy:>14.2f}")
    print(f"{'Parámetros (M):':<30} {original.total_params/1e6:>14.2f} {pruned.total_params/1e6:>14.2f}")
    print(f"{'Sparsidad real:':<30} {original.sparsity:>14.4f} {pruned.sparsity:>14.4f}")
    print(f"{'ms/imagen:':<30} {original.ms_per_image:>14.2f} {pruned.ms_per_image:>14.2f}")
    print("=" * 60 + "\n")


def main() -> None:
    """Ejecuta el pipeline completo de poda no estructurada."""
    cfg = UnstructuredPruningConfig()
    device = torch.device("cuda", cfg.device_id)

    # --- 1. Cargar modelo original ---
    print("[Unstructured] Cargando modelo original…")
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

    # --- 2. Construir dataloader ---
    print("[Unstructured] Construyendo dataloader de validación…")
    dm, val_loader = _build_val_loader(cfg, cfg.device_id)

    # --- 3. Evaluar línea base ---
    print("[Unstructured] Evaluando modelo original…")
    acc_orig, ms_orig = evaluate(
        model, val_loader, device, max_batches=cfg.max_val_batches
    )
    total_orig, nz_orig = count_nonzero_params(model)
    stats_orig = PruningStats(
        label="Original",
        accuracy=acc_orig,
        total_params=total_orig,
        nonzero_params=nz_orig,
        ms_per_image=ms_orig,
        sparsity=model_sparsity(model),
    )

    # --- 4. Aplicar poda no estructurada ---
    print(f"[Unstructured] Podando pesos (sparsity={cfg.sparsity:.0%})…")
    pruner = WeightPruner(sparsity=cfg.sparsity)
    model = pruner.prune(model)

    # --- 5. Evaluar modelo podado ---
    print("[Unstructured] Evaluando modelo podado…")
    dm.teardown()
    dm, val_loader = _build_val_loader(cfg, cfg.device_id)
    acc_pruned, ms_pruned = evaluate(
        model, val_loader, device, max_batches=cfg.max_val_batches
    )
    total_pruned, nz_pruned = count_nonzero_params(model)
    stats_pruned = PruningStats(
        label=f"Unstructured {int(cfg.sparsity*100)}%",
        accuracy=acc_pruned,
        total_params=total_pruned,
        nonzero_params=nz_pruned,
        ms_per_image=ms_pruned,
        sparsity=model_sparsity(model),
    )
    dm.teardown()

    # --- 6. Imprimir comparativa ---
    _print_comparison(stats_orig, stats_pruned)

    # --- 7. Guardar modelo podado ---
    save_model(
        model,
        cfg.output_path,
        metadata={
            "sparsity": cfg.sparsity,
            "real_sparsity": stats_pruned.sparsity,
            "accuracy": acc_pruned,
        },
    )

    # --- 8. Generar figura ---
    PruningPlotter().plot(
        stats=[stats_orig, stats_pruned],
        output_path=cfg.figure_path,
        title=f"Unstructured Pruning ({int(cfg.sparsity*100)}%) vs Original",
    )


if __name__ == "__main__":
    main()
