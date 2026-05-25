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
from itertools import islice

import torch
from torch import nn
from tqdm import tqdm

from .batch_processor import PatchBatchProcessor
from .hardware import GpuProfile
from .mixup import MixupAugmentor
from .config import TransformerTrainingConfig 


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
        gpu_profile: t.Optional[GpuProfile] = None,
        patience: int = 25,
        min_delta: float = 0.0005,
        validate_every: int = 1,
        max_val_batches: t.Optional[int] = None,
        show_progress_bar: bool = True,
        grad_clip_norm: float = 1.0,
        config: TransformerTrainingConfig = None,
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
        self.gpu_profile = gpu_profile
        self.patience = patience
        self.min_delta = min_delta
        self.validate_every = max(1, validate_every)
        if max_val_batches is not None and max_val_batches <= 0:
            raise ValueError("max_val_batches must be greater than 0 or None.")
        self.max_val_batches = max_val_batches
        self.show_progress_bar = show_progress_bar
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self._amp_dtype = self._select_amp_dtype()
        # GradScaler is required when training in float16 to avoid gradient
        # underflow. With bfloat16 (Ampere+) it must NOT be used.
        use_scaler = (
            self._amp_dtype == torch.float16
            and self.device.type == "cuda"
        )
        self._scaler = torch.cuda.amp.GradScaler() if use_scaler else None
        self.config = config

    def _select_amp_dtype(self) -> t.Optional[torch.dtype]:
        if self.device.type != "cuda":
            return None
        if self.gpu_profile is not None and self.gpu_profile.amp_dtype is not None:
            return self.gpu_profile.amp_dtype
        # Fallback: detect from compute capability if profile not provided.
        cap = torch.cuda.get_device_capability(self.device)
        return torch.bfloat16 if cap[0] >= 8 else torch.float16

    def fit(self, epochs: int) -> t.Dict[str, t.Union[float, int]]:
        """Run the full training process."""
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        last_val_metrics = {
            "val_loss": float("nan"),
            "val_acc": float("nan"),
        }

        for epoch in range(epochs):
            train_metrics = self._train_one_epoch(epoch, epochs)
            should_validate = self._should_validate(epoch, epochs)

            if should_validate:
                val_metrics = self._validate_one_epoch()
                last_val_metrics = val_metrics
                self._step_scheduler(val_metrics["val_loss"])
            else:
                val_metrics = last_val_metrics
                self._step_scheduler(None)

            is_best = False
            if should_validate and val_metrics["val_loss"] < best_val_loss - self.min_delta:
                best_val_loss = val_metrics["val_loss"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                is_best = True
                torch.save(self.model.state_dict(), self.checkpoint_path)
            elif should_validate:
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
                f"{self._format_validation_metrics(val_metrics, should_validate)}"
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
        train_correct = torch.zeros((), device=self.device, dtype=torch.float32)
        train_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        train_total = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            disable=not self.show_progress_bar,
        )

        for data in progress:
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

            # Reemplaza el bloque if self._scaler por:
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.detach().float() * batch_size

            _, preds = outputs.max(1)
            if self.mixup_augmentor.enabled:
                train_correct += (
                    lam * preds.eq(y_a).sum().float()
                    + (1.0 - lam) * preds.eq(y_b).sum().float()
                )
            else:
                train_correct += preds.eq(labels).sum().float()

            train_total += batch_size

        return {
            "train_loss": (train_loss_sum / max(1, train_total)).item(),
            "train_acc": (100.0 * train_correct / max(1, train_total)).item(),
        }

    def _validate_one_epoch(self) -> t.Dict[str, float]:
        """Evaluate the model on the validation split."""
        self.model.eval()
        val_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        val_correct = torch.zeros((), device=self.device, dtype=torch.float32)
        val_total = 0

        val_iterable = self.val_loader
        if self.max_val_batches is not None:
            self._reset_loader(self.val_loader)
            val_iterable = islice(self.val_loader, self.max_val_batches)

        try:
            with torch.inference_mode():
                for data in val_iterable:
                    inputs, labels = self._prepare_inputs(data)

                    with self._autocast_context():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    batch_size = labels.size(0)
                    val_loss_sum += loss.detach().float() * batch_size
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().float()
                    val_total += batch_size
        finally:
            if self.max_val_batches is not None:
                self._reset_loader(self.val_loader)

        return {
            "val_loss": (val_loss_sum / max(1, val_total)).item(),
            "val_acc": (100.0 * val_correct / max(1, val_total)).item(),
        }

    def _step_scheduler(self, val_loss: t.Optional[float]) -> None:
        """Advance the scheduler after each epoch."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _should_validate(self, epoch: int, epochs: int) -> bool:
        """Return whether validation should run after the current epoch."""
        epoch_number = epoch + 1
        return (
            epoch_number == 1
            or epoch_number == epochs
            or epoch_number % self.validate_every == 0
        )

    @staticmethod
    def _reset_loader(loader) -> None:
        """Reset iterators such as DALI when the epoch was only partially consumed."""
        reset = getattr(loader, "reset", None)
        if reset is not None:
            reset()

    @staticmethod
    def _format_validation_metrics(
        val_metrics: t.Dict[str, float],
        was_validated: bool,
    ) -> str:
        """Format validation metrics without pretending skipped epochs were evaluated."""
        if not was_validated:
            return "Val: skipped"

        return (
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']:.2f}%"
        )

    def _autocast_context(self):
        """Return the autocast context used during forward passes."""
        if self._amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
