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

import typing as t

import torch
import torch.nn as nn
import torch.optim as optim

from transformer.data.dali.datamodule import DaliDataModule
from transformer.training.losses.focal_loss import FocalLoss
from transformer.models.transformer_model import VisionTransformer
from transformer.training.batch_processor import PatchBatchProcessor
from transformer.training.config import TransformerTrainingConfig
from transformer.training.energy.cpu_energy_meter import RAPLCPUEnergyMeter
from transformer.training.energy.gpu_energy_meter import EMLGPUEnergyMeter, SlurmGPUSelector
from transformer.training.mixup import MixupAugmentor
from transformer.training.schedulers import SchedulerFactory
from transformer.training.trainer import TransformerTrainer
from transformer.visualization.training_plotter import TrainingPlotter
from transformer.data.dali.pipelines import DaliPipelineFactory
import nvidia.dali.types as types


class TransformerTrainingApplication:
    """Build, run, and report the full Transformer training workflow."""

    def __init__(self, config: TransformerTrainingConfig) -> None:
        self.config = config

    def run(self) -> None:
        """Execute the training workflow."""
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
    
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
            validator_batch_size=self.config.validator_batch_size,
            pipeline_factory=self._build_pipeline_factory(),
        )

    def _build_trainer(self, device, train_loader, val_loader) -> TransformerTrainer:
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
            gpu_profile=self.config._gpu_profile,      # ← antes faltaba
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            validate_every=self.config.validate_every,  # ← antes faltaba
            max_val_batches=self.config.max_val_batches, # ← antes faltaba
            show_progress_bar=self.config.show_progress_bar, # ← antes faltaba
            grad_clip_norm=self.config.grad_clip_norm,  # ← nuevo
        )

    def _build_model(self, device: torch.device) -> nn.Module:
        """Create and optionally compile the Transformer model."""
        model = VisionTransformer(
            patch_dim=self.config.patch_dim,
            seq_len=self.config.seq_len,
            embed_dim=self.config.embed_dim,
            num_classes=self.config.num_classes,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            dropout=self.config.dropout,
            drop_path_rate=self.config.drop_path_rate,
            use_patch_norm=True,
        ).to(device)

        if self.config.compile_model:
            profile = self.config._gpu_profile
            compile_mode = profile.compile_mode if profile else "default"
            try:
                model = torch.compile(model, mode=compile_mode)
            except Exception as exc:
                print(f"[torch.compile] Disabled ({compile_mode}): {exc}")
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
        eml_measurements: t.Dict[str, t.Any],
        gpu_metrics: t.Dict[str, float],
        cpu_metrics: t.Dict[str, t.Union[float, bool]],
        device_key: str,
    ) -> t.Dict[str, t.Any]:
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
    
    def _build_pipeline_factory(self) -> DaliPipelineFactory:
        # BFLOAT16 en DALI requiere >= 1.14. Si no está, usamos FLOAT16 y
        # dejamos que el autocast de PyTorch haga la conversión a BF16.
        bf16_dtype = getattr(types, "BFLOAT16", types.FLOAT16)

        dtype_map = {
            "bf16": bf16_dtype,
            "fp16": types.FLOAT16,
        }
        output_dtype = dtype_map.get(self.config.dali_output_dtype, types.FLOAT16)

        return DaliPipelineFactory(
            output_dtype=output_dtype,
            hw_decoder_load=self.config.dali_hw_decoder_load,
            device_memory_padding=self.config.dali_device_memory_padding,
            host_memory_padding=self.config.dali_host_memory_padding,
            preallocate_width_hint=self.config.dali_preallocate_width_hint,
            preallocate_height_hint=self.config.dali_preallocate_height_hint,
            decoder_cache_size=self.config.dali_decoder_cache_size,
            decoder_cache_threshold=0,
            train_decoder_cache_size=self.config.dali_train_decoder_cache_size,
            reader_prefetch_queue_depth=self.config.train_prefetch_queue_depth,
        )

    def _get_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("This script requires CUDA.")
        return torch.device("cuda", self.config.device_id)

    def _get_device_info(self) -> str:
        return torch.cuda.get_device_name(self.config.device_id) 


def main() -> None:
    """Run the training application."""
    config = TransformerTrainingConfig()
    application = TransformerTrainingApplication(config)
    application.run()


if __name__ == "__main__":
    main()
