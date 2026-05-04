"""Generates visual comparison reports for PyTorch image models."""

from __future__ import annotations

from pathlib import Path

from .model_metrics import ModelComparison, ModelConfig
from .visualization import ComparisonVisualizer


CUSTOM_TRANSFORMER_PATH = Path("checkpoints/transformer/best_transformer.pth")
VIT_B_16_PATH = Path("checkpoints/transformer/best_transformer_vitb16.pth")
RESNET50_PATH = Path("checkpoints/resnet/best_resnet50.pth")
OUTPUT_DIR = Path("outputs/inference/model_comparison")

DEFAULT_ACCURACIES = {
    "custom_transformer": 87.41,
    "vit_b_16": 91.15,
    "resnet50": 91.41,
}


def build_model_configs() -> list[ModelConfig]:
    """Builds the model configuration list used by the report."""
    return [
        ModelConfig(
            name="Transformer propio",
            model_type="Transformer",
            checkpoint_path=CUSTOM_TRANSFORMER_PATH,
            accuracy=DEFAULT_ACCURACIES["custom_transformer"],
        ),
        ModelConfig(
            name="ViT-B/16 preentrenado",
            model_type="Transformer preentrenado",
            checkpoint_path=VIT_B_16_PATH,
            accuracy=DEFAULT_ACCURACIES["vit_b_16"],
        ),
        ModelConfig(
            name="ResNet50 preentrenada",
            model_type="CNN preentrenada",
            checkpoint_path=RESNET50_PATH,
            accuracy=DEFAULT_ACCURACIES["resnet50"],
        ),
    ]


def main() -> None:
    """Runs the model comparison workflow."""
    comparison = ModelComparison(build_model_configs())
    metrics = comparison.collect_metrics()

    visualizer = ComparisonVisualizer(OUTPUT_DIR)
    visualizer.save_report(metrics)

    print(f"Reporte generado en: {OUTPUT_DIR.resolve()}")
    print(f"- {OUTPUT_DIR / 'model_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'model_comparison_table.png'}")
    print(f"- {OUTPUT_DIR / 'accuracy_comparison.png'}")


if __name__ == "__main__":
    main()
