"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: distributed/train_ddp.py

Description:
    Aplicación de entrenamiento multi-GPU con DistributedDataParallel (DDP).

    torchrun lanza un proceso Python independiente por cada GPU.  Todos los
    procesos ejecutan el mismo bucle de entrenamiento en paralelo:

        1. Cada proceso lee únicamente su fragmento del dataset (DALI sharding).
        2. Cada proceso hace forward + backward sobre su lote local.
        3. Tras el backward, NCCL sincroniza los gradientes entre GPUs
           (all-reduce automático del wrapper DDP).
        4. Todos los procesos actualizan sus pesos con los mismos gradientes.

    Solo el proceso de rango 0 mide energía, imprime métricas y guarda
    checkpoints; el resto se mantienen en lockstep para que DDP funcione.

    Comando de lanzamiento (ver batch_ddp.sh para SLURM):
        torchrun --nproc_per_node=2 -m transformer.distributed.train_ddp
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer.data.dali.datamodule import DaliDataModule
from transformer.training.batch_processor import PatchBatchProcessor
from transformer.training.config import TransformerTrainingConfig
from transformer.training.cutmix import CutMixAugmentor
from transformer.training.mixup import MixupAugmentor
from transformer.training.schedulers import SchedulerFactory
from transformer.training.train_transformer import TransformerTrainingApplication
from transformer.training.trainer import TransformerTrainer
from transformer.visualization.training_plotter import TrainingPlotter
from transformer.distributed.ddp_setup import (
    cleanup,
    get_local_rank,
    get_rank,
    get_world_size,
    init_process_group,
    is_main_process,
)


class DDPTrainingApplication(TransformerTrainingApplication):
    """Extiende la aplicación mono-GPU para entrenamiento con DDP.

    Solo sobreescribe los métodos que cambian en un entorno multi-GPU:
        - Selección de dispositivo (LOCAL_RANK en vez de device_id del config).
        - Escalado del learning rate proporcional al número de GPUs.
        - Envoltura del modelo en DDP para sincronización de gradientes.
        - Módulo de datos con sharding de dataset por rango.
        - Trainer con flags de rango DDP para guardado y logging.
        - Bucle principal con medición de energía solo en rango 0.
    """

    def __init__(self, config: TransformerTrainingConfig) -> None:
        """Inicializa el estado DDP y escala el learning rate.

        Args:
            config: Configuración de entrenamiento. El learning rate se
                modifica in-place según la regla de escalado lineal.
        """
        super().__init__(config)
        self._rank       = get_rank()
        self._local_rank = get_local_rank()
        self._world_size = get_world_size()
        self._scale_learning_rate()

    # ------------------------------------------------------------------
    # Escalado del learning rate
    # ------------------------------------------------------------------

    def _scale_learning_rate(self) -> None:
        """Escala el LR linealmente con el número de GPUs.

        Regla de escalado lineal: con N GPUs cada proceso procesa un lote
        completo de batch_size imágenes, por lo que el batch global efectivo
        es batch_size × N.  Para mantener la misma dinámica de convergencia
        que en mono-GPU, el LR debe crecer proporcionalmente.

        Solo actúa cuando hay más de una GPU para no alterar experimentos
        mono-GPU que usen esta clase por error.
        """
        if self._world_size <= 1:
            return

        self.config.learning_rate *= self._world_size

        if is_main_process():
            print(
                f"[DDP] {self._world_size} GPUs — "
                f"LR escalado a {self.config.learning_rate:.2e}"
            )

    # ------------------------------------------------------------------
    # Selección de dispositivo
    # ------------------------------------------------------------------

    def _get_device(self) -> torch.device:
        """Devuelve el dispositivo CUDA asignado a este proceso.

        Cada proceso usa su propia GPU identificada por LOCAL_RANK, que
        es el índice de la GPU dentro del nodo actual.
        """
        return torch.device("cuda", self._local_rank)

    def _get_device_info(self) -> str:
        """Devuelve una descripción legible del entorno multi-GPU."""
        return f"{self._world_size}× {torch.cuda.get_device_name(self._local_rank)}"

    # ------------------------------------------------------------------
    # Construcción del modelo
    # ------------------------------------------------------------------

    def _build_model(self, device: torch.device) -> nn.Module:
        """Construye el modelo y lo envuelve en DDP.

        Primero delega en el padre para crear y compilar el VisionTransformer,
        luego añade el wrapper DDP que sincroniza gradientes tras cada backward.

        Args:
            device: GPU asignada a este proceso.

        Returns:
            Modelo envuelto en ``DistributedDataParallel``.
        """
        # El padre aplica torch.compile si config.compile_model es True.
        # DDP se añade después del compile para máxima compatibilidad.
        model = super()._build_model(device)
        return DDP(model, device_ids=[self._local_rank])

    # ------------------------------------------------------------------
    # Construcción del módulo de datos
    # ------------------------------------------------------------------

    def _build_data_module(self) -> DaliDataModule:
        """Construye el módulo DALI con sharding del dataset por rango.

        Cada proceso recibe un fragmento distinto del dataset: el rango 0
        lee las primeras 1/N imágenes, el rango 1 las siguientes, etc.
        Así no hay duplicación de datos entre GPUs.
        """
        return DaliDataModule(
            root_dir=self.config.root_dir,
            train_batch_size=self.config.train_batch_size,
            val_batch_size=self.config.val_batch_size,
            image_size=self.config.image_size,
            num_threads=self.config.num_threads,
            device_id=self._local_rank,
            drop_last=self.config.drop_last,
            train_prefetch_queue_depth=self.config.train_prefetch_queue_depth,
            val_prefetch_queue_depth=self.config.val_prefetch_queue_depth,
            validator_batch_size=self.config.validator_batch_size,
            pipeline_factory=self._build_pipeline_factory(),
            shard_id=self._rank,          # fragmento que lee este proceso
            num_shards=self._world_size,  # total de fragmentos
        )

    # ------------------------------------------------------------------
    # Construcción del trainer
    # ------------------------------------------------------------------

    def _build_trainer(
        self,
        device: torch.device,
        train_loader,
        val_loader,
    ) -> TransformerTrainer:
        """Ensambla el trainer con los parámetros específicos de DDP.

        La única diferencia respecto al trainer mono-GPU es que se pasan
        ``ddp_rank`` e ``is_ddp=True`` para que el trainer sepa en qué
        proceso está y proteja correctamente el guardado de checkpoints
        y la reducción de métricas de validación entre GPUs.

        Args:
            device:       GPU de este proceso.
            train_loader: Iterador DALI de entrenamiento (shard de este rango).
            val_loader:   Iterador DALI de validación (shard de este rango).

        Returns:
            Instancia configurada de :class:`TransformerTrainer`.
        """
        model     = self._build_model(device)
        optimizer = self._build_optimizer(model)
        criterion = self._build_criterion()
        scheduler = SchedulerFactory.build_warmup_cosine_scheduler(
            optimizer=optimizer,
            epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
        )

        return TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            plotter=TrainingPlotter(),
            device=device,
            batch_processor=PatchBatchProcessor(self.config.patch_size),
            mixup_augmentor=MixupAugmentor(self.config.mixup_alpha),
            cutmix_augmentor=CutMixAugmentor(self.config.cutmix_alpha),
            ema_decay=self.config.ema_decay,
            checkpoint_path=self.config.checkpoint_path,
            gpu_profile=self.config._gpu_profile,
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            validate_every=self.config.validate_every,
            max_val_batches=self.config.max_val_batches,
            # La barra de progreso solo aparece en el proceso principal para
            # evitar N líneas entrelazadas en el log de SLURM.
            show_progress_bar=self.config.show_progress_bar and is_main_process(),
            grad_clip_norm=self.config.grad_clip_norm,
            ddp_rank=self._rank,
            is_ddp=True,
        )

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Ejecuta el entrenamiento DDP midiendo la energía de todas las GPUs.

        Todos los rangos miden su propia GPU con EML.  Al terminar el
        entrenamiento se usa ``dist.all_reduce`` para sumar la energía de
        todas las GPUs en el rango 0, que es el único que imprime y guarda
        la gráfica.
        """
        from transformer.training.energy.cpu_energy_meter import RAPLCPUEnergyMeter
        from transformer.training.energy.gpu_energy_meter import (
            EMLGPUEnergyMeter,
            SlurmGPUSelector,
        )

        self._configure_torch_backends()

        device      = self._get_device()
        data_module = self._build_data_module()
        data_module.setup()

        trainer = self._build_trainer(
            device=device,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader(),
        )

        # Cada rango mide su propia GPU (local_rank como índice explícito)
        gpu_meter = EMLGPUEnergyMeter(SlurmGPUSelector(self._local_rank))

        if is_main_process():
            cpu_meter = RAPLCPUEnergyMeter()
            (result, eml_meas, elapsed, device_key), cpu_metrics = cpu_meter.measure(
                lambda: gpu_meter.measure(lambda: trainer.fit(self.config.epochs))
            )
        else:
            cpu_metrics = None
            result, eml_meas, elapsed, device_key = gpu_meter.measure(
                lambda: trainer.fit(self.config.epochs)
            )

        # Sumar la energía de todas las GPUs en rango 0 vía all_reduce
        local_energy_j = gpu_meter.extract_metrics(eml_meas, device_key)["gpu_energy_j"]
        energy_tensor  = torch.tensor([local_energy_j], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(energy_tensor, op=torch.distributed.ReduceOp.SUM)
        total_gpu_energy_j = energy_tensor.item()

        if is_main_process():
            gpu_metrics = gpu_meter.extract_metrics(eml_meas, device_key)
            # Reemplazar la energía local por la suma total de todas las GPUs
            gpu_metrics["gpu_energy_j"]  = total_gpu_energy_j
            gpu_metrics["gpu_energy_uj"] = total_gpu_energy_j * 1e6
            # Potencia media aproximada: energía total / tiempo transcurrido
            gpu_metrics["gpu_avg_power_w"] = total_gpu_energy_j / max(elapsed, 1e-9)

            self._print_results(result, gpu_metrics, cpu_metrics)
            self._save_plot(trainer, gpu_metrics, eml_meas,
                            cpu_metrics, device_key, elapsed)

        data_module.teardown()

    # ------------------------------------------------------------------
    # Métodos auxiliares privados
    # ------------------------------------------------------------------

    def _configure_torch_backends(self) -> None:
        """Activa las optimizaciones de backends de PyTorch según el config."""
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    def _print_results(
        self,
        result: dict,
        gpu_metrics: dict,
        cpu_metrics: dict,
    ) -> None:
        """Imprime las métricas finales de entrenamiento y energía (rango 0).

        Args:
            result:      Resultado de ``trainer.fit()``.
            gpu_metrics: Métricas GPU con energía total de todas las GPUs.
            cpu_metrics: Métricas de energía CPU (RAPL).
        """
        print(f"[DDP] Best val loss      : {result['best_val_loss']:.4f}")
        print(f"[DDP] Best epoch         : {result['best_epoch']}")
        print(f"[DDP] GPU energy (total) : {gpu_metrics['gpu_energy_j']:.2f} J  "
              f"({self._world_size} GPUs sumadas)")
        print(f"[DDP] GPU avg power      : {gpu_metrics['gpu_avg_power_w']:.2f} W")
        print(f"[DDP] CPU energy         : {cpu_metrics['cpu_rapl_files_j']:.2f} J")

    def _save_plot(
        self,
        trainer: TransformerTrainer,
        gpu_metrics: dict,
        eml_meas: dict,
        cpu_metrics: dict,
        device_key: str,
        elapsed: float,
    ) -> None:
        """Genera y guarda la gráfica de curvas de entrenamiento (rango 0).

        Args:
            trainer:     Trainer con el plotter ya actualizado.
            gpu_metrics: Métricas GPU con energía total de todas las GPUs.
            eml_meas:    Mediciones EML originales del rango 0 (para el dict).
            cpu_metrics: Métricas de energía CPU.
            device_key:  Clave EML del dispositivo del rango 0.
            elapsed:     Tiempo total de entrenamiento en segundos.
        """
        energy_stats = self._build_energy_stats(
            eml_measurements=eml_meas,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics,
            device_key=device_key,
        )
        trainer.plotter.plot(
            save_path=self.config.figure_path,
            energy_stats={"meas": energy_stats, "elapsed": elapsed},
            device_info=self._get_device_info(),
            show_power=False,
        )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada que torchrun invoca una vez por proceso/GPU.

    El grupo de procesos se inicializa antes de construir la aplicación y
    se destruye en el bloque finally para limpiar aunque haya errores.
    """
    init_process_group()
    try:
        config = TransformerTrainingConfig()
        # Las gráficas del entrenamiento DDP se guardan en outputs/figures/ddp/
        # en vez de outputs/figures/transformer/ para distinguirlas del mono-GPU.
        from pathlib import Path
        config.figure_path = Path(
            str(config.figure_path).replace(
                "outputs/figures/transformer/",
                "outputs/figures/ddp/",
            )
        )
        application = DDPTrainingApplication(config)
        application.run()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
