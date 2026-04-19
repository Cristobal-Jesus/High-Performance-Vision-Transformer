"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: train_transformer.py

Description:
    This file defines the object-oriented training application for the
    Transformer model, including DALI data loading and GPU/CPU energy
    measurement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights

from transformer.data.dali.datamodule import DaliDataModule
from transformer.training.losses.focal_loss import FocalLoss
from transformer.models.transformer_model import VisionTransformer
from transformer.training.batch_processor import PatchBatchProcessor
from ViTB16.training.config import TransformerTrainingConfig
from transformer.training.energy.cpu_energy_meter import RAPLCPUEnergyMeter
from transformer.training.energy.gpu_energy_meter import EMLGPUEnergyMeter, SlurmGPUSelector
from transformer.training.mixup import MixupAugmentor
from transformer.training.schedulers import SchedulerFactory
from transformer.training.trainer import TransformerTrainer
from ViTB16.visualization.training_plotter import TrainingPlotter


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class TransformerTrainingApplication:
    """Build, run, and report the full Transformer training workflow."""

    def __init__(self, config: TransformerTrainingConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Execute the training workflow."""
        device = self._get_device()
        data_module = self._build_data_module()
        data_module.setup()

        trainer = self._build_trainer(
            device=device,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
        )

        gpu_meter = EMLGPUEnergyMeter(SlurmGPUSelector(self.config.eml_gpu_index))
        cpu_meter = RAPLCPUEnergyMeter()

        (training_result, eml_measurements, elapsed_total, device_key), cpu_metrics = (
            cpu_meter.measure(lambda: gpu_meter.measure(lambda: trainer.fit(self.config.epochs)))
        )

        gpu_metrics = gpu_meter.extract_metrics(eml_measurements, device_key)
        energy_stats = self._build_energy_stats(
            eml_measurements=eml_measurements,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics,
            device_key=device_key,
        )

        print(f"Best validation loss: {training_result['best_val_loss']:.4f}")
        print(f"Best epoch: {training_result['best_epoch']}")
        print(f"GPU energy ({device_key}): {gpu_metrics['gpu_energy_j']:.4f} J")
        print(f"Average GPU power: {gpu_metrics['gpu_avg_power_w']:.4f} W")
        print(f"CPU energy (RAPL files): {cpu_metrics['cpu_rapl_files_j']:.4f} J")

        trainer.plotter.plot(
            save_path=self.config.figure_path,
            energy_stats={
                "meas": energy_stats,
                "elapsed": elapsed_total,
            },
            device_info=self._get_device_info(),
        )

        data_module.teardown()

    def _build_data_module(self) -> DaliDataModule:
        """Create the DALI data module."""
        return DaliDataModule(
            root_dir=self.config.root_dir,
            train_batch_size=self.config.train_batch_size,
            val_batch_size=self.config.val_batch_size,
            image_size=self.config.image_size,
            num_threads=self.config.num_threads,
            device_id=self.config.device_id,
            drop_last=self.config.drop_last,
            train_prefetch_queue_depth=self.config.train_prefetch_queue_depth,
            val_prefetch_queue_depth=self.config.val_prefetch_queue_depth,
        )

    def _build_trainer(
        self,
        device: torch.device,
        train_loader,
        val_loader,
    ) -> TransformerTrainer:
        """Create the trainer object used for the fit loop."""
        model = self._build_model(device)
        optimizer = self._build_optimizer(model)
        criterion = self._build_criterion()
        scheduler = SchedulerFactory.build_warmup_cosine_scheduler(
            optimizer=optimizer,
            epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
        )
        plotter = TrainingPlotter()

        return TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            plotter=plotter,
            device=device,
            batch_processor=PatchBatchProcessor(self.config.patch_size),
            mixup_augmentor=MixupAugmentor(self.config.mixup_alpha),
            checkpoint_path=self.config.checkpoint_path,
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

    def _build_model(self, device: torch.device) -> nn.Module:
        """Create and optionally compile the Transformer model."""
        weight = ViT_B_16_Weights.DEFAULT
        vit_model = vit_b_16(weights=weight)
        vit_model.heads.head = nn.Linear(
            vit_model.heads.head.in_features,
            self.config.num_classes
        )
        vit_model.to(device)

        try:
            model = torch.compile(vit_model)
        except Exception as exc:
            print(f"[torch.compile] Disabled: {exc}")

        return model

    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create the optimizer used during training."""
        return optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_criterion(self) -> nn.Module:
        """Create the training loss."""
        if self.config.use_focal_loss:

            return FocalLoss(
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing,
            )

        return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

    def _build_energy_stats(
        self,
        eml_measurements: dict,
        gpu_metrics: dict[str, float],
        cpu_metrics: dict[str, float | bool],
        device_key: str,
    ) -> dict:
        """Merge GPU and CPU measurements into one plot-friendly dictionary."""
        energy_stats = dict(eml_measurements)
        energy_stats[device_key] = dict(eml_measurements[device_key])
        energy_stats["gpu_energy_uj"] = gpu_metrics["gpu_energy_uj"]
        energy_stats["gpu_energy_j"] = gpu_metrics["gpu_energy_j"]
        energy_stats["gpu_avg_power_w"] = gpu_metrics["gpu_avg_power_w"]
        energy_stats["cpu_rapl_files_uj"] = cpu_metrics["cpu_rapl_files_uj"]
        energy_stats["cpu_rapl_files_j"] = cpu_metrics["cpu_rapl_files_j"]
        energy_stats["cpu_rapl_available"] = cpu_metrics["cpu_rapl_available"]
        energy_stats["rapl_files"] = {"consumed": cpu_metrics["cpu_rapl_files_uj"]}
        return energy_stats

    @staticmethod
    def _get_device() -> torch.device:
        """Return the device used during training."""
        if not torch.cuda.is_available():
            raise RuntimeError("This script requires CUDA.")

        return torch.device("cuda")

    @staticmethod
    def _get_device_info() -> str:
        """Return a human-readable description of the active GPU."""
        return torch.cuda.get_device_name(0)


def main() -> None:
    """Run the training application."""
    config = TransformerTrainingConfig()
    application = TransformerTrainingApplication(config)
    application.run()


if __name__ == "__main__":
    main()
