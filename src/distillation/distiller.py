"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distillation/distiller.py

Description:
    Bucle de entrenamiento para Knowledge Distillation.

    El profesor permanece en modo evaluación y sus gradientes se
    desactivan durante todo el entrenamiento.  Solo el estudiante se
    actualiza.

    Estructura del bucle (un epoch):
        1. Forward del profesor sobre el batch (sin gradientes).
        2. Forward del estudiante sobre el mismo batch.
        3. Cálculo de la pérdida KD.
        4. Backward + gradient clipping + paso del optimizador.
        5. Actualización del scheduler al final de cada epoch.
        6. Cada ``validate_every`` epochs: evalúa el estudiante y
           guarda checkpoint si es el mejor val_loss.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from transformer.training.batch_processor import PatchBatchProcessor
from .losses import KnowledgeDistillationLoss
from .visualization.plotter import DistillationPlotter


class Distiller:
    """Gestiona el entrenamiento de destilación profesor → estudiante.

    Args:
        teacher:         Modelo profesor (se congela automáticamente).
        student:         Modelo estudiante (se entrena).
        train_loader:    Iterador DALI de entrenamiento.
        val_loader:      Iterador DALI de validación.
        optimizer:       Optimizador para el estudiante.
        scheduler:       Scheduler de LR.
        criterion:       Instancia de :class:`KnowledgeDistillationLoss`.
        device:          Dispositivo de entrenamiento.
        checkpoint_path: Ruta donde guardar el mejor checkpoint del estudiante.
        plotter:         Instancia de :class:`DistillationPlotter`.
        validate_every:  Validar cada N epochs.
        max_val_batches: Batches máximos de validación (None = todos).
        grad_clip_norm:  Norma máxima del gradiente.
        show_progress_bar: Si True, muestra barra de progreso tqdm.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader,
        val_loader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: KnowledgeDistillationLoss,
        device: torch.device,
        checkpoint_path: Path,
        plotter: DistillationPlotter,
        validate_every: int = 5,
        max_val_batches: t.Optional[int] = None,
        grad_clip_norm: float = 1.0,
        show_progress_bar: bool = True,
    ) -> None:
        self.teacher           = teacher.to(device).eval()
        self.student           = student.to(device)
        self.train_loader      = train_loader
        self.val_loader        = val_loader
        self.optimizer         = optimizer
        self.scheduler         = scheduler
        self.criterion         = criterion
        self.device            = device
        self.checkpoint_path   = Path(checkpoint_path)
        self.plotter           = plotter
        self.validate_every    = validate_every
        self.max_val_batches   = max_val_batches
        self.grad_clip_norm    = grad_clip_norm
        self.show_progress_bar = show_progress_bar
        self._processor        = PatchBatchProcessor(patch_size=16)

        # Congelar profesor
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self._best_val_loss = float("inf")
        self._best_epoch    = 0

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def fit(self, epochs: int) -> dict:
        """Ejecuta el bucle de destilación durante ``epochs`` épocas.

        Args:
            epochs: Número de épocas de entrenamiento.

        Returns:
            Dict con ``{"best_val_loss", "best_epoch"}``.
        """
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch()

            val_loss = val_acc = 0.0
            do_validate = (epoch % self.validate_every == 0) or (epoch == epochs)
            if do_validate:
                val_loss, val_acc = self._val_epoch()
                is_best = val_loss < self._best_val_loss
                if is_best:
                    self._best_val_loss = val_loss
                    self._best_epoch    = epoch
                    self._save_checkpoint()
            else:
                is_best = False

            self.scheduler.step()
            self.plotter.update(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss if do_validate else None,
                val_acc=val_acc if do_validate else None,
                is_best=is_best,
            )

            self._print_epoch(epoch, epochs, train_loss, train_acc,
                              val_loss if do_validate else None,
                              val_acc if do_validate else None)

        return {
            "best_val_loss": self._best_val_loss,
            "best_epoch":    self._best_epoch,
        }

    # ------------------------------------------------------------------
    # Bucles internos
    # ------------------------------------------------------------------

    def _train_epoch(self) -> t.Tuple[float, float]:
        """Ejecuta un epoch de entrenamiento.

        Returns:
            ``(loss_media, accuracy_pct)``
        """
        self.student.train()
        total_loss = correct = total = 0

        for batch in self.train_loader:
            images = batch[0]["images"].to(self.device, non_blocking=True)
            labels = batch[0]["labels"].squeeze(-1).long().to(self.device, non_blocking=True)

            patches = self._processor.patchify(images)
            model_dtype = next(self.student.parameters()).dtype
            patches = patches.to(dtype=model_dtype)

            with torch.inference_mode():
                teacher_logits = self.teacher(patches)

            student_logits = self.student(patches)
            loss = self.criterion(student_logits, teacher_logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (student_logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def _val_epoch(self) -> t.Tuple[float, float]:
        """Ejecuta un epoch de validación del estudiante.

        Returns:
            ``(val_loss_media, val_accuracy_pct)``
        """
        self.student.eval()
        total_loss = correct = total = 0

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader):
                if self.max_val_batches and batch_idx >= self.max_val_batches:
                    break

                images = batch[0]["images"].to(self.device, non_blocking=True)
                labels = batch[0]["labels"].squeeze(-1).long().to(self.device, non_blocking=True)

                patches = self._processor.patchify(images)
                model_dtype = next(self.student.parameters()).dtype
                patches = patches.to(dtype=model_dtype)

                teacher_logits = self.teacher(patches)
                student_logits = self.student(patches)
                loss = self.criterion(student_logits, teacher_logits, labels)

                total_loss += loss.item() * images.size(0)
                correct    += (student_logits.argmax(1) == labels).sum().item()
                total      += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Guarda el mejor checkpoint del estudiante."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.student.state_dict(),
                "best_val_loss":    self._best_val_loss,
                "best_epoch":       self._best_epoch,
            },
            self.checkpoint_path,
        )

    @staticmethod
    def _print_epoch(
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: t.Optional[float],
        val_acc: t.Optional[float],
    ) -> None:
        """Imprime una línea de resumen del epoch."""
        val_str = (
            f" | val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%"
            if val_loss is not None else ""
        )
        print(
            f"  Epoch {epoch:3d}/{total_epochs}"
            f" | train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%"
            f"{val_str}"
        )
