"""Trainer for pretrained ViT-B classifiers."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm


class ViTBTrainer:
    """Train and validate a ViT-B classifier with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        output_path: str | Path,
        class_names: list[str],
        patience: int = 8,
        min_delta: float = 0.0005,
        mixed_precision: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_path = Path(output_path)
        self.class_names = class_names
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = mixed_precision and device.type == "cuda"

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(self, epochs: int) -> dict[str, float | int]:
        """Run training and return the best validation state."""
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            train_loss, train_acc = self._train_one_epoch(epoch, epochs)
            val_loss, val_acc = self._validate_one_epoch()
            self.scheduler.step()

            improved = val_loss < best_val_loss - self.min_delta
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, best_val_loss, best_val_acc)
            else:
                epochs_without_improvement += 1

            elapsed = time.perf_counter() - start_time
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                f"time={elapsed:.2f}s"
            )

            if epochs_without_improvement >= self.patience:
                print("Early stopping triggered.")
                break

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
        }

    def _train_one_epoch(self, epoch: int, epochs: int) -> tuple[float, float]:
        self.model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(
            self.train_loader,
            desc=f"Train {epoch}/{epochs}",
            leave=False,
        ):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += batch_size

        return loss_sum / max(1, total), 100.0 * correct / max(1, total)

    @torch.no_grad()
    def _validate_one_epoch(self) -> tuple[float, float]:
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.val_loader, desc="Val", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self._autocast_context():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += batch_size

        return loss_sum / max(1, total), 100.0 * correct / max(1, total)

    def _save_checkpoint(
        self,
        epoch: int,
        best_val_loss: float,
        best_val_acc: float,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "class_names": self.class_names,
                "num_classes": len(self.class_names),
                "model_name": "vit_b_16",
                "pretrained": True,
            },
            self.output_path,
        )
        print(f"Saved best model: {self.output_path}")

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()

        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
