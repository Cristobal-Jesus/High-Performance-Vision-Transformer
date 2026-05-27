"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 30th April 2026
File: hardware.py

Description:
    GPU profile detection. Chooses AMP dtype, GradScaler need, DALI
    pipeline parameters, and training defaults based on the active CUDA
    device. All hardware-dependent knobs live here so that config.py
    never contains hard-coded device paths or tuning values.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch


@dataclass
class GpuProfile:
    """All hardware-dependent tuning parameters for one GPU model."""

    # --- identity ---
    name: str
    compute_capability: t.Tuple[int, int]
    vram_gb: float

    # --- training precision ---
    amp_dtype: t.Optional[torch.dtype]
    use_grad_scaler: bool
    supports_bf16: bool
    supports_fp8: bool

    # --- training throughput ---
    recommended_batch_size: int
    recommended_prefetch: int        # DALIGenericIterator prefetch_queue_depth
    recommended_num_threads: int     # DALI CPU decoder threads
    compile_mode: str                # torch.compile mode string

    # --- dataset paths (cluster-specific) ---
    root_dir: str
    output_folder: str               # subfolder for checkpoints / figures

    # --- DALI pipeline knobs ---
    dali_output_dtype: str
    dali_hw_decoder_load: float
    dali_device_memory_padding: int
    dali_host_memory_padding: int
    dali_preallocate_width_hint: int
    dali_preallocate_height_hint: int

    # Validation decoder cache size in MB (0 = disabled).
    dali_decoder_cache_size: int

    # Training decoder cache size in MB (0 = disabled).
    # Cuando > 0, el pipeline de entrenamiento usa fn.decoders.image (cacheable)
    # + fn.random_resized_crop en GPU. A partir de la época 2 el disco queda
    # inactivo: todas las imágenes se sirven desde VRAM.
    dali_train_decoder_cache_size: int


# ---------------------------------------------------------------------------
_DATASET_PADDING_BYTES: int = 240_000   # ~235 KB por imagen 256×256 RGB


def detect_gpu_profile(device_id: int = 0) -> GpuProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot build GPU profile.")

    name = torch.cuda.get_device_name(device_id)
    cap = torch.cuda.get_device_capability(device_id)
    props = torch.cuda.get_device_properties(device_id)
    vram_gb = props.total_memory / (1024 ** 3)

    supports_bf16 = cap[0] >= 8
    supports_fp8 = cap[0] >= 9
    name_lower = name.lower()

    # H200 — Hopper, 141 GB HBM3e
    if "h200" in name_lower:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.bfloat16,
            use_grad_scaler=False,
            supports_bf16=True,
            supports_fp8=True,
            recommended_batch_size=1024,
            recommended_prefetch=8,
            recommended_num_threads=48,
            compile_mode="max-autotune",
            root_dir="/home/almeida/Cristobal/Images/dataset_256",
            output_folder="H200",
            dali_output_dtype="bf16",
            dali_hw_decoder_load=1.0,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=0,           # 8 GB caché validación
            dali_train_decoder_cache_size=45_000,   # 90 GB caché entrenamiento
        )

    # H100 — Hopper, 80 GB HBM3
    if "h100" in name_lower:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.bfloat16,
            use_grad_scaler=False,
            supports_bf16=True,
            supports_fp8=True,
            recommended_batch_size=1024,
            recommended_prefetch=4,
            recommended_num_threads=32,
            compile_mode="max-autotune",
            root_dir="/home/almeida/Cristobal/Images/dataset_256",
            output_folder="H100",
            dali_output_dtype="bf16",
            dali_hw_decoder_load=1.0,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=0,           # 4 GB caché validación
            dali_train_decoder_cache_size=20_000,   # 40 GB caché entrenamiento
        )

    # V100 — Volta, 16 o 32 GB HBM2
    if "v100" in name_lower:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.float16,
            use_grad_scaler=True,
            supports_bf16=False,
            supports_fp8=False,
            recommended_batch_size=128 if vram_gb < 24 else 192,
            recommended_prefetch=2,
            recommended_num_threads=16,
            compile_mode="default",
            root_dir="/var/tmp/nhernang/dataset_256",
            output_folder="V100",
            dali_output_dtype="fp16",
            dali_hw_decoder_load=0.65,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=0,
            dali_train_decoder_cache_size=0,
        )

    # Fallback genérico — Ampere o más nuevo (BF16)
    if supports_bf16:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.bfloat16,
            use_grad_scaler=False,
            supports_bf16=True,
            supports_fp8=supports_fp8,
            recommended_batch_size=192,
            recommended_prefetch=3,
            recommended_num_threads=16,
            compile_mode="reduce-overhead",
            root_dir="/home/almeida/Cristobal/Images/dataset_256",
            output_folder="Ampere",
            dali_output_dtype="bf16",
            dali_hw_decoder_load=0.85,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=0,
            dali_train_decoder_cache_size=0,
        )

    # Turing / Pascal / antiguas (solo FP16)
    return GpuProfile(
        name=name,
        compute_capability=cap,
        vram_gb=vram_gb,
        amp_dtype=torch.float16,
        use_grad_scaler=True,
        supports_bf16=False,
        supports_fp8=False,
        recommended_batch_size=128,
        recommended_prefetch=2,
        recommended_num_threads=8,
        compile_mode="default",
        root_dir="/var/tmp/nhernang/dataset_256",
        output_folder="GPU",
        dali_output_dtype="fp16",
        dali_hw_decoder_load=0.65,
        dali_device_memory_padding=_DATASET_PADDING_BYTES,
        dali_host_memory_padding=_DATASET_PADDING_BYTES,
        dali_preallocate_width_hint=256,
        dali_preallocate_height_hint=256,
        dali_decoder_cache_size=0,
        dali_train_decoder_cache_size=0,
    )