"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Performance and Machine Learning
File: main.py

Description:
    Entry point. Instantiates the Evaluator and runs inference over the
    test image directory.
"""

import torch

from tests.transformer.test_transformer import Evaluator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "checkpoints/transformer/best_transformer_e384.pth"
IMAGES_DIR = "/home/almeida/Cristobal/Test"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Loads the model and evaluates it over the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    evaluator = Evaluator(checkpoint_path=CHECKPOINT_PATH, device=device, scratch=True)
    print("Model loaded.")

    total_params, size_mb = evaluator.model_stats()
    precision = next(iter(evaluator.model.parameters())).dtype

    print(f"Total parameters:  {total_params:,}")
    print(f"Model size in RAM: {size_mb:.2f} MB")
    print(f"Parameter dtype:   {precision}")

    accuracy = evaluator.evaluate(images_dir=IMAGES_DIR)

    print(f"\nAccuracy on test images: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
