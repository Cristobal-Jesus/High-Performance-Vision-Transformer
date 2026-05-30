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
from .cutmix import CutMixAugmentor
from .ema import EMAModel
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
        ddp_rank: int = 0,
        is_ddp: bool = False,
        cutmix_augmentor: t.Optional[CutMixAugmentor] = None,
        ema_decay: float = 0.0,
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
        self.cutmix_augmentor = cutmix_augmentor or CutMixAugmentor(alpha=0.0)
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
        self.ddp_rank = ddp_rank
        self.is_ddp = is_ddp
        self.config = config

        # EMA: se inicializa solo si decay > 0
        self._ema: t.Optional[EMAModel] = (
            EMAModel(model, decay=ema_decay) if ema_decay > 0.0 else None
        )

        self._amp_dtype = self._select_amp_dtype()
        use_scaler = (
            self._amp_dtype == torch.float16
            and self.device.type == "cuda"
        )
        self._scaler = torch.cuda.amp.GradScaler() if use_scaler else None
        self._nvml_handle = self._init_nvml_handle()

    def _select_amp_dtype(self) -> t.Optional[torch.dtype]:
        """Elige el dtype de precisión mixta según el perfil de GPU."""
        if self.device.type != "cuda":
            return None
        if self.gpu_profile is not None and self.gpu_profile.amp_dtype is not None:
            return self.gpu_profile.amp_dtype
        cap = torch.cuda.get_device_capability(self.device)
        return torch.bfloat16 if cap[0] >= 8 else torch.float16

    def _init_nvml_handle(self):
        """Inicializa pynvml y devuelve el handle de la GPU. None si no disponible."""
        if self.device.type != "cuda":
            return None
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = self.device.index if self.device.index is not None else 0
            return pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            return None

    def _sample_gpu_power_w(self) -> float:
        """Devuelve la potencia instantánea de la GPU en vatios. 0.0 si no disponible."""
        if self._nvml_handle is None:
            return 0.0
        try:
            import pynvml
            return pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
        except Exception:
            return 0.0

    def fit(self, epochs: int) -> t.Dict[str, t.Union[float, int]]:
        """Ejecuta el ciclo completo de entrenamiento."""
        best_val_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        last_val_metrics: t.Dict[str, t.Any] = {
            "val_loss": float("nan"),
            "val_acc": float("nan"),
            "val_gpu_power_w": None,
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
                self._save_checkpoint()
            elif should_validate:
                epochs_without_improvement += 1

            self.plotter.update(
                train_acc=train_metrics["train_acc"],
                val_acc=val_metrics["val_acc"],
                train_loss=train_metrics["train_loss"],
                val_loss=val_metrics["val_loss"],
                is_best=is_best,
                train_gpu_power_w=train_metrics.get("train_gpu_power_w", 0.0),
                val_gpu_power_w=val_metrics.get("val_gpu_power_w") if should_validate else None,
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

    def _save_checkpoint(self) -> None:
        """Guarda el checkpoint. En DDP solo escribe el rango 0.

        Si EMA está activo, guarda los pesos EMA (mejor para inferencia).
        """
        if self.is_ddp and self.ddp_rank != 0:
            return
        if self._ema is not None:
            torch.save(self._ema.state_dict(), self.checkpoint_path)
        else:
            model_state = getattr(self.model, "module", self.model).state_dict()
            torch.save(model_state, self.checkpoint_path)

    def _prepare_inputs(self, data):
        """Prepara las entradas del modelo desde un lote del loader.

        Usado solo en validación (sin augmentación). Para entrenamiento
        con CutMix se usa ``_load_images`` + ``_to_model_input``.
        """
        if self.batch_processor is not None:
            return self.batch_processor.prepare_batch(data, self.device)
        batch = data[0]
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels"].squeeze(-1).long().to(self.device, non_blocking=True)
        return images, labels

    def _load_images(self, data) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Desempaqueta el lote DALI y mueve al dispositivo sin parchificar.

        Devuelve imágenes en crudo ``(B, C, H, W)`` y etiquetas ``(B,)``.
        CutMix necesita la estructura espacial antes de la parchificación.
        """
        batch = data[0]
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels"].squeeze(-1).long().to(self.device, non_blocking=True)
        return images, labels

    def _to_model_input(self, images: torch.Tensor) -> torch.Tensor:
        """Convierte imágenes en crudo ``(B, C, H, W)`` a la entrada del modelo.

        Si hay ``batch_processor``, aplica la parchificación para el ViT.
        De lo contrario devuelve las imágenes tal cual.
        """
        if self.batch_processor is not None:
            return self.batch_processor.patchify(images)
        return images

    def _apply_batch_augmentation(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Aplica CutMix, Mixup o ninguno según la configuración.

        Si ambos están activos, elige aleatoriamente uno de los dos.
        CutMix tiene prioridad por defecto (50/50 si ambos están activos).

        Returns:
            ``(images_aug, y_a, y_b, lam)``
        """
        use_cutmix = self.cutmix_augmentor.enabled
        use_mixup  = self.mixup_augmentor.enabled

        if use_cutmix and use_mixup:
            if torch.rand(1).item() < 0.5:
                return self.cutmix_augmentor.apply(images, labels)
            else:
                return self.mixup_augmentor.apply(images, labels)
        elif use_cutmix:
            return self.cutmix_augmentor.apply(images, labels)
        elif use_mixup:
            return self.mixup_augmentor.apply(images, labels)
        else:
            return images, labels, labels, 1.0

    def _train_one_epoch(self, epoch: int, epochs: int) -> t.Dict[str, t.Any]:
        """Entrena el modelo durante un epoch completo."""
        self.model.train()
        train_correct = torch.zeros((), device=self.device, dtype=torch.float32)
        train_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        train_total = 0
        power_samples: t.List[float] = []

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            disable=not self.show_progress_bar,
        )

        for i, data in enumerate(progress):
            # --- 1. Cargar imágenes en crudo (sin parchificar) ---
            images, labels = self._load_images(data)

            # --- 2. Augmentación (CutMix / Mixup / ninguna) ---
            # Se aplica antes de parchificar porque CutMix necesita la
            # estructura espacial 2D de la imagen.
            images, y_a, y_b, lam = self._apply_batch_augmentation(images, labels)

            # --- 3. Convertir a entrada del modelo (parchificar si ViT) ---
            inputs = self._to_model_input(images)

            # --- 4. Forward + loss ---
            self.optimizer.zero_grad(set_to_none=True)

            mixing_active = lam < 1.0 and not torch.equal(y_a, y_b)

            with self._autocast_context():
                outputs = self.model(inputs)
                if mixing_active:
                    loss = (
                        lam * self.criterion(outputs, y_a)
                        + (1.0 - lam) * self.criterion(outputs, y_b)
                    )
                else:
                    loss = self.criterion(outputs, labels)

            # --- 5. Backward + clip + step ---
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

            # --- 6. Actualizar copia EMA ---
            if self._ema is not None:
                self._ema.update(self.model)

            # --- 7. Métricas ---
            batch_size = labels.size(0)
            train_loss_sum += loss.detach().float() * batch_size

            _, preds = outputs.max(1)
            if mixing_active:
                train_correct += (
                    lam * preds.eq(y_a).sum().float()
                    + (1.0 - lam) * preds.eq(y_b).sum().float()
                )
            else:
                train_correct += preds.eq(labels).sum().float()

            train_total += batch_size

            if i % 25 == 12:
                power_samples.append(self._sample_gpu_power_w())

        avg_power_w = (
            sum(power_samples) / len(power_samples) if power_samples else 0.0
        )

        return {
            "train_loss": (train_loss_sum / max(1, train_total)).item(),
            "train_acc": (100.0 * train_correct / max(1, train_total)).item(),
            "train_gpu_power_w": avg_power_w,
        }

    def _validate_one_epoch(self) -> t.Dict[str, t.Any]:
        """Evalúa el modelo sobre el split de validación.

        Usa la copia EMA si está disponible, ya que sus pesos son más
        estables y suelen dar mejor accuracy que el modelo en entrenamiento.
        """
        eval_model = self._ema.model if self._ema is not None else self.model
        eval_model.eval()
        self.model.eval()   # necesario para DDP / batch norm
        val_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        val_correct = torch.zeros((), device=self.device, dtype=torch.float32)
        val_total = 0
        power_samples: t.List[float] = []

        val_iterable = self.val_loader
        if self.max_val_batches is not None:
            self._reset_loader(self.val_loader)
            val_iterable = islice(self.val_loader, self.max_val_batches)

        try:
            with torch.inference_mode():
                for i, data in enumerate(val_iterable):
                    inputs, labels = self._prepare_inputs(data)
                    with self._autocast_context():
                        outputs = eval_model(inputs)
                        loss = self.criterion(outputs, labels)
                    batch_size = labels.size(0)
                    val_loss_sum += loss.detach().float() * batch_size
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().float()
                    val_total += batch_size
                    if i % 5 == 2:
                        power_samples.append(self._sample_gpu_power_w())
        finally:
            if self.max_val_batches is not None:
                self._reset_loader(self.val_loader)

        avg_power_w = (
            sum(power_samples) / len(power_samples) if power_samples else 0.0
        )

        return {
            "val_loss": (val_loss_sum / max(1, val_total)).item(),
            "val_acc": (100.0 * val_correct / max(1, val_total)).item(),
            "val_gpu_power_w": avg_power_w,
        }

    def _step_scheduler(self, val_loss: t.Optional[float]) -> None:
        """Avanza el scheduler tras cada epoch."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _should_validate(self, epoch: int, epochs: int) -> bool:
        """Indica si se debe validar al final del epoch actual."""
        epoch_number = epoch + 1
        return (
            epoch_number == 1
            or epoch_number == epochs
            or epoch_number % self.validate_every == 0
        )

    @staticmethod
    def _reset_loader(loader) -> None:
        """Reinicia iteradores como DALI cuando el epoch se consumió parcialmente."""
        reset = getattr(loader, "reset", None)
        if reset is not None:
            reset()

    @staticmethod
    def _format_validation_metrics(
        val_metrics: t.Dict[str, t.Any],
        was_validated: bool,
    ) -> str:
        """Formatea las métricas de validación para el log."""
        if not was_validated:
            return "Val: skipped"
        return (
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']:.2f}%"
        )

    def _autocast_context(self):
        """Devuelve el contexto de autocast para los forward passes."""
        if self._amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
