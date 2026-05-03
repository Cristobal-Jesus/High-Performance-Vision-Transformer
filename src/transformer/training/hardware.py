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
    GPU profile detection. Chooses AMP dtype, GradScaler need, and
    DALI / training defaults based on the active CUDA device.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch


@dataclass
class GpuProfile:
    name: str
    compute_capability: t.Tuple[int, int]
    vram_gb: float
    amp_dtype: t.Optional[torch.dtype]
    use_grad_scaler: bool
    recommended_batch_size: int
    recommended_prefetch: int
    hw_decoder_load: float
    compile_mode: str
    supports_bf16: bool
    supports_fp8: bool


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

    if "v100" in name_lower:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.float16,
            use_grad_scaler=True,
            recommended_batch_size=128 if vram_gb < 24 else 192,
            recommended_prefetch=2,
            hw_decoder_load=0.65,
            compile_mode="default",
            supports_bf16=False,
            supports_fp8=False,
        )

    if "h200" in name_lower:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.bfloat16,
            use_grad_scaler=False,
            recommended_batch_size=512,
            recommended_prefetch=4,
            hw_decoder_load=1.0,
            compile_mode="max-autotune",
            supports_bf16=True,
            supports_fp8=True,
        )

    if "h100" in name_lower:
        return GpuProfile(
            name=name,
            compute_capability=cap,
            vram_gb=vram_gb,
            amp_dtype=torch.bfloat16,
            use_grad_scaler=False,
            recommended_batch_size=384,
            recommended_prefetch=4,
            hw_decoder_load=1.0,
            compile_mode="max-autotune",
            supports_bf16=True,
            supports_fp8=True,
        )

    # Generic fallback: pick by compute capability
    if supports_bf16:
        return GpuProfile(
            name=name, compute_capability=cap, vram_gb=vram_gb,
            amp_dtype=torch.bfloat16, use_grad_scaler=False,
            recommended_batch_size=192, recommended_prefetch=3,
            hw_decoder_load=0.85, compile_mode="reduce-overhead",
            supports_bf16=True, supports_fp8=supports_fp8,
        )

    return GpuProfile(
        name=name, compute_capability=cap, vram_gb=vram_gb,
        amp_dtype=torch.float16, use_grad_scaler=True,
        recommended_batch_size=128, recommended_prefetch=2,
        hw_decoder_load=0.65, compile_mode="default",
        supports_bf16=False, supports_fp8=False,
    )
