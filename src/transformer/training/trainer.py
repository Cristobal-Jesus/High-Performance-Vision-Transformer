"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: trainer.py

Description:
    This file defines the class responsible for the Transformer training
    and validation loops.
"""

import typing as t
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .batch_processor import PatchBatchProcessor
from .mixup import MixupAugmentor


class TransformerTrainer:
    """Train and validate a Transformer classification model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler,
        plotter,
        device: torch.device,
        batch_processor: t.Optional[PatchBatchProcessor],
        mixup_augmentor: MixupAugmentor,
        checkpoint_path: t.Union[str, Path],
        patience: int = 25,
        min_delta: float = 0.0005,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.plotter = plotter
        self.device = device
        self.batch_processor = batch_processor
        self.mixup_augmentor = mixup_augmentor
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.min_delta = min_delta

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(self, epochs: int) -> t.Dict[str, t.Union[float, int]]:
        """Run the full training process."""
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_metrics = self._train_one_epoch(epoch, epochs)
            val_metrics = self._validate_one_epoch()
            self._step_scheduler(val_metrics["val_loss"])

            is_best = False
            if val_metrics["val_loss"] < best_val_loss - self.min_delta:
                best_val_loss = val_metrics["val_loss"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                is_best = True
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_without_improvement += 1

            self.plotter.update(
                train_acc=train_metrics["train_acc"],
                val_acc=val_metrics["val_acc"],
                train_loss=train_metrics["train_loss"],
                val_loss=val_metrics["val_loss"],
                is_best=is_best,
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

            if epochs_without_improvement >= self.patience:
                print("Early stopping triggered.")
                break

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }

    def _prepare_inputs(self, data):
        """Prepare model inputs and labels from one loader batch."""
        if self.batch_processor is not None:
            return self.batch_processor.prepare_batch(data, self.device)

        batch = data[0]
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels"].squeeze(-1).long().to(self.device, non_blocking=True)
        return images, labels

    def _train_one_epoch(self, epoch: int, epochs: int) -> t.Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        train_loss_sum = 0.0
        train_correct = 0.0
        train_total = 0

        for data in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            inputs, labels = self._prepare_inputs(data)
            inputs, y_a, y_b, lam = self.mixup_augmentor.apply(inputs, labels)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_context():
                outputs = self.model(inputs)
                if self.mixup_augmentor.enabled:
                    loss = lam * self.criterion(outputs, y_a) + (
                        1.0 - lam
                    ) * self.criterion(outputs, y_b)
                else:
                    loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size

            _, preds = outputs.max(1)
            if self.mixup_augmentor.enabled:
                train_correct += (
                    lam * preds.eq(y_a).sum().item()
                    + (1.0 - lam) * preds.eq(y_b).sum().item()
                )
            else:
                train_correct += preds.eq(labels).sum().item()

            train_total += batch_size

        return {
            "train_loss": train_loss_sum / max(1, train_total),
            "train_acc": 100.0 * train_correct / max(1, train_total),
        }

    def _validate_one_epoch(self) -> t.Dict[str, float]:
        """Evaluate the model on the validation split."""
        self.model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = self._prepare_inputs(data)

                with self._autocast_context():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += batch_size

        return {
            "val_loss": val_loss_sum / max(1, val_total),
            "val_acc": 100.0 * val_correct / max(1, val_total),
        }

    def _step_scheduler(self, val_loss: float) -> None:
        """Advance the scheduler after each epoch."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _autocast_context(self):
        """Return the autocast context used during forward passes."""
        if self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()
