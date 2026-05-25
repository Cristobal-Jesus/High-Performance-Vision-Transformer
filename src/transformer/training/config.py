"""..."""
from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from .hardware import GpuProfile, detect_gpu_profile


@dataclass
class TransformerTrainingConfig:

    # --- rutas (None → se rellenan desde GpuProfile) ---
    root_dir: t.Optional[str] = None
    checkpoint_path: t.Optional[Path] = None
    figure_path: t.Optional[Path] = None

    # --- batch sizes (None → se rellenan desde GpuProfile) ---
    train_batch_size: t.Optional[int] = None
    val_batch_size: t.Optional[int] = None

    # --- arquitectura del modelo ---
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 20
    embed_dim: int = 512        # era 384
    depth: int = 12
    num_heads: int = 8          # era 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.25  # era 0.20

    # --- optimizador ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.10
    label_smoothing: float = 0.15
    grad_clip_norm: float = 1.0   # NUEVO

    # --- loss ---
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # --- augmentation ---
    mixup_alpha: float = 0.0

    # --- bucle de entrenamiento ---
    epochs: int = 200
    warmup_epochs: int = 10
    patience: int = 25
    min_delta: float = 0.0005
    validate_every: int = 1
    max_val_batches: t.Optional[int] = None

    # --- DALI (None → se rellenan desde GpuProfile) ---
    num_threads: int = 32
    device_id: int = 0
    drop_last: bool = True
    train_prefetch_queue_depth: t.Optional[int] = None
    val_prefetch_queue_depth: int = 4
    validator_batch_size: int = 128           # NUEVO
    dali_output_dtype: t.Optional[str] = None          # "fp16" | "bf16"
    dali_hw_decoder_load: t.Optional[float] = None
    dali_device_memory_padding: t.Optional[int] = None
    dali_host_memory_padding: t.Optional[int] = None
    dali_preallocate_width_hint: t.Optional[int] = None
    dali_preallocate_height_hint: t.Optional[int] = None
    dali_decoder_cache_size: t.Optional[int] = None    # MB; 0 = desactivado
    dali_train_decoder_cache_size: t.Optional[int] = None

    # --- hardware / backends ---
    auto_tune_for_gpu: bool = True
    compile_model: bool = True
    allow_tf32: bool = True        # NUEVO
    cudnn_benchmark: bool = True   # NUEVO
    eml_gpu_index: t.Optional[int] = None

    # --- misc ---
    show_progress_bar: bool = True

    # campo no-init: perfil detectado
    _gpu_profile: t.Optional[GpuProfile] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        profile: t.Optional[GpuProfile] = None

        if self.auto_tune_for_gpu:
            try:
                profile = detect_gpu_profile(self.device_id)
                object.__setattr__(self, "_gpu_profile", profile)
            except RuntimeError:
                pass

        def _resolve(value, attr: str, fallback):
            if value is not None:
                return value
            if profile is not None:
                return getattr(profile, attr)
            return fallback

        folder = profile.output_folder if profile else "CPU"

        self.root_dir = _resolve(self.root_dir, "root_dir",
                                 "/home/almeida/Cristobal/Images/dataset_256")

        if self.checkpoint_path is None:
            self.checkpoint_path = Path(
                f"checkpoints/transformer/best_transformer.pth"
            )
        if self.figure_path is None:
            self.figure_path = Path(
                f"outputs/figures/transformer/{folder}/training_curves.png"
            )

        self.train_batch_size = _resolve(
            self.train_batch_size, "recommended_batch_size", 64)
        if self.val_batch_size is None:
            self.val_batch_size = self.train_batch_size

        self.train_prefetch_queue_depth = _resolve(
            self.train_prefetch_queue_depth, "recommended_prefetch", 3)

        self.dali_output_dtype = _resolve(
            self.dali_output_dtype, "dali_output_dtype", "fp16")
        self.dali_hw_decoder_load = _resolve(
            self.dali_hw_decoder_load, "dali_hw_decoder_load", 0.65)
        self.dali_device_memory_padding = _resolve(
            self.dali_device_memory_padding, "dali_device_memory_padding", 240_000)
        self.dali_host_memory_padding = _resolve(
            self.dali_host_memory_padding, "dali_host_memory_padding", 240_000)
        self.dali_preallocate_width_hint = _resolve(
            self.dali_preallocate_width_hint, "dali_preallocate_width_hint", 256)
        self.dali_preallocate_height_hint = _resolve(
            self.dali_preallocate_height_hint, "dali_preallocate_height_hint", 256)
        self.dali_decoder_cache_size = _resolve(
            self.dali_decoder_cache_size, "dali_decoder_cache_size", 0)
        self.dali_train_decoder_cache_size = _resolve(
            self.dali_train_decoder_cache_size, "dali_train_decoder_cache_size", 0)
        
    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * 3

    @property
    def seq_len(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def use_mixup(self) -> bool:
        return self.mixup_alpha > 0.0
