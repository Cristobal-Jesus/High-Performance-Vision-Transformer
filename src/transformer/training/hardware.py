"""..."""
from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch


@dataclass
class GpuProfile:
    # --- identidad ---
    name: str
    compute_capability: t.Tuple[int, int]
    vram_gb: float

    # --- precisión ---
    amp_dtype: t.Optional[torch.dtype]
    use_grad_scaler: bool
    supports_bf16: bool
    supports_fp8: bool

    # --- throughput ---
    recommended_batch_size: int
    recommended_prefetch: int
    compile_mode: str

    # --- rutas del clúster ---
    root_dir: str
    output_folder: str

    # --- parámetros del pipeline DALI ---
    dali_output_dtype: str           # "fp16" | "bf16"
    dali_hw_decoder_load: float      # fracción del motor HW NVJPEG (0.0–1.0)
    dali_device_memory_padding: int  # bytes por imagen en GPU
    dali_host_memory_padding: int    # bytes por imagen en CPU
    dali_preallocate_width_hint: int
    dali_preallocate_height_hint: int
    dali_decoder_cache_size: int     # MB en VRAM para caché de val (0 = desactivado)


_DATASET_PADDING_BYTES: int = 240_000  # 256×256×3 + 20% margen


def detect_gpu_profile(device_id: int = 0) -> GpuProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot build GPU profile.")

    name  = torch.cuda.get_device_name(device_id)
    cap   = torch.cuda.get_device_capability(device_id)
    props = torch.cuda.get_device_properties(device_id)
    vram_gb    = props.total_memory / (1024 ** 3)
    supports_bf16 = cap[0] >= 8
    supports_fp8  = cap[0] >= 9
    name_lower    = name.lower()

    if "h200" in name_lower:
        return GpuProfile(
            name=name, compute_capability=cap, vram_gb=vram_gb,
            amp_dtype=torch.bfloat16, use_grad_scaler=False,
            supports_bf16=True, supports_fp8=True,
            recommended_batch_size=512, recommended_prefetch=6,
            compile_mode="max-autotune",
            root_dir="/home/bejeque/nhernang/Cristobal/dataset_256",
            output_folder="H200",
            dali_output_dtype="bf16",
            dali_hw_decoder_load=1.0,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=8192,   # 8 GB — sobra en 141 GB
        )

    if "h100" in name_lower:
        return GpuProfile(
            name=name, compute_capability=cap, vram_gb=vram_gb,
            amp_dtype=torch.bfloat16, use_grad_scaler=False,
            supports_bf16=True, supports_fp8=True,
            recommended_batch_size=384, recommended_prefetch=4,
            compile_mode="max-autotune",
            root_dir="/home/bejeque/nhernang/Cristobal/dataset_256",
            output_folder="H100",
            dali_output_dtype="bf16",
            dali_hw_decoder_load=1.0,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=4096,   # 4 GB — seguro en 80 GB
        )

    if "v100" in name_lower:
        return GpuProfile(
            name=name, compute_capability=cap, vram_gb=vram_gb,
            amp_dtype=torch.float16, use_grad_scaler=True,
            supports_bf16=False, supports_fp8=False,
            recommended_batch_size=128 if vram_gb < 24 else 192,
            recommended_prefetch=2,
            compile_mode="default",
            root_dir="/var/tmp/nhernang/dataset_256",
            output_folder="V100",
            dali_output_dtype="fp16",
            dali_hw_decoder_load=0.65,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=0,      # conservador en ≤32 GB
        )

    # Fallback genérico BF16 (Ampere, etc.)
    if supports_bf16:
        return GpuProfile(
            name=name, compute_capability=cap, vram_gb=vram_gb,
            amp_dtype=torch.bfloat16, use_grad_scaler=False,
            supports_bf16=True, supports_fp8=supports_fp8,
            recommended_batch_size=192, recommended_prefetch=3,
            compile_mode="reduce-overhead",
            root_dir="/home/bejeque/nhernang/Cristobal/dataset_256",
            output_folder="Ampere",
            dali_output_dtype="bf16",
            dali_hw_decoder_load=0.85,
            dali_device_memory_padding=_DATASET_PADDING_BYTES,
            dali_host_memory_padding=_DATASET_PADDING_BYTES,
            dali_preallocate_width_hint=256,
            dali_preallocate_height_hint=256,
            dali_decoder_cache_size=0,
        )

    # Fallback genérico FP16 (Turing, Pascal, etc.)
    return GpuProfile(
        name=name, compute_capability=cap, vram_gb=vram_gb,
        amp_dtype=torch.float16, use_grad_scaler=True,
        supports_bf16=False, supports_fp8=False,
        recommended_batch_size=128, recommended_prefetch=2,
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
    )
