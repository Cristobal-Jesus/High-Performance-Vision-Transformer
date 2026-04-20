"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: gpu_energy_meter.py

Description:
    This file defines the classes responsible for selecting the GPU
    assigned by Slurm and measuring its energy with EML using the NVML
    backend.
"""

import os
import typing as t

from pyeml import measure_energy
from pyeml.devices import nvml
from pyeml.units import uj

T = t.TypeVar("T")


class SlurmGPUSelector:
    """Resolve the GPU index that should be measured inside a Slurm job."""

    def __init__(self, explicit_index: t.Optional[int] = None) -> None:
        self.explicit_index = explicit_index

    def resolve_index(self) -> int:
        """Return the physical GPU index that EML should measure."""
        if self.explicit_index is not None:
            return self.explicit_index

        for env_name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
            value = os.environ.get(env_name, "").strip()
            index = self._extract_first_numeric_token(value)
            if index is not None:
                return index

        return 0

    @staticmethod
    def _extract_first_numeric_token(value: str) -> t.Optional[int]:
        """Extract the first GPU index from a comma-separated environment variable."""
        if not value:
            return None

        first_token = value.split(",")[0].strip()
        return int(first_token) if first_token.isdigit() else None


class EMLGPUEnergyMeter:
    """Measure GPU energy with EML using an NVML device decorator."""

    def __init__(self, gpu_selector: SlurmGPUSelector) -> None:
        self.gpu_selector = gpu_selector

    def measure(
        self,
        function: t.Callable[[], T],
    ) -> t.Tuple[T, t.Dict[str, t.Any], float, str]:
        """Measure GPU energy while executing a callable."""
        gpu_index = self.gpu_selector.resolve_index()

        @measure_energy(devices=(nvml(gpu_index),), unit=uj)
        def measured_function() -> T:
            return function()

        result, measurements, elapsed = measured_function()
        device_key = "nvml{}".format(gpu_index)

        if device_key not in measurements:
            raise RuntimeError(
                "EML did not return measurements for {}. Available keys: {}".format(
                    device_key,
                    list(measurements.keys()),
                )
            )

        return result, measurements, elapsed, device_key

    @staticmethod
    def extract_metrics(
        measurements: t.Dict[str, t.Any],
        device_key: str,
    ) -> t.Dict[str, float]:
        """Return normalized GPU metrics in joules, watts, and seconds."""
        device_measurements = measurements[device_key]

        gpu_energy_uj = float(device_measurements["consumed"])
        gpu_power_uw = float(device_measurements["power"])
        gpu_elapsed_s = float(device_measurements["elapsed"])

        return {
            "gpu_energy_uj": gpu_energy_uj,
            "gpu_energy_j": gpu_energy_uj / 1e6,
            "gpu_avg_power_uw": gpu_power_uw,
            "gpu_avg_power_w": gpu_power_uw / 1e6,
            "gpu_elapsed_s": gpu_elapsed_s,
        }
