"""Train a pretrained ViT-B/16 with explicit train and validation splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.ViT-B.data.train_val_image_folder_data_module import (
    TrainValImageFolderDataModule,
)
from src.ViT-B.training.pretrained_vit_b import PretrainedViTBFactory
from src.ViT-B.training.vit_b_trainer import ViTBTrainer
from src.ViT-B.training.vit_b_training_config import ViTBTrainingConfig


class PretrainedViTBTrainingApplication:
    """Build and run the pretrained ViT-B training workflow."""

    def __init__(self, config: ViTBTrainingConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Execute the full training workflow."""
        device = self._resolve_device()
        data_module = self._build_data_module()
        data_module.setup()

        model = self._build_model(device)
        trainer = self._build_trainer(model, data_module, device)
        result = trainer.fit(self.config.epochs)

        print(f"Best epoch: {result['best_epoch']}")
        print(f"Best val loss: {result['best_val_loss']:.4f}")
        print(f"Best val acc: {result['best_val_acc']:.2f}%")

    def _build_data_module(self) -> TrainValImageFolderDataModule:
        return TrainValImageFolderDataModule(
            train_dir=self.config.train_dir,
            val_dir=self.config.val_dir,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            expected_num_classes=self.config.num_classes,
        )

    def _build_model(self, device: torch.device) -> nn.Module:
        model = PretrainedViTBFactory(
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            freeze_backbone=self.config.freeze_backbone,
        ).build()
        model = model.to(device)

        if self.config.compile_model:
            try:
                model = torch.compile(model)
            except Exception as exc:
                print(f"[torch.compile] Disabled: {exc}")

        return model

    def _build_trainer(
        self,
        model: nn.Module,
        data_module: TrainValImageFolderDataModule,
        device: torch.device,
    ) -> ViTBTrainer:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing,
        )
        optimizer = torch.optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.config.epochs),
        )

        return ViTBTrainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_path=self.config.output_path,
            class_names=data_module.class_names,
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mixed_precision=self.config.mixed_precision,
        )

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(self.config.device)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train pretrained ViT-B/16.")
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("checkpoints/transformer/vit_b_16_pretrained.pth"),
        type=Path,
    )
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.1, type=float)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--min-delta", default=0.0005, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ViTBTrainingConfig:
    """Build a strongly typed training configuration from CLI arguments."""
    return ViTBTrainingConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_path=args.output,
        num_classes=args.num_classes,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        min_delta=args.min_delta,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        mixed_precision=not args.no_mixed_precision,
        compile_model=args.compile_model,
        device=args.device,
    )


def main() -> None:
    """Run the pretrained ViT-B training application."""
    app = PretrainedViTBTrainingApplication(build_config(parse_args()))
    app.run()


if __name__ == "__main__":
    main()
