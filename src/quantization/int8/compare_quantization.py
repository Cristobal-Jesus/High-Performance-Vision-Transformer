from __future__ import annotations

from .comparator import QuantizationComparator
from transformer.training.config import TransformerTrainingConfig


def main() -> None:
    config = TransformerTrainingConfig()

    comparator = QuantizationComparator(
        test_dir="/home/bejeque/nhernang/Cristobal/pytorch_models/assets/Test",
        config=config,
        checkpoint_path=config.checkpoint_path,
        batch_size=64,
        output_path="outputs/figures/quantization/quantization_int8_comparison.png",
    )
    comparator.compare()


if __name__ == "__main__":
    main()
    