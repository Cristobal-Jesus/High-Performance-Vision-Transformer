"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: cpu_energy_meter.py

Description:
    This file defines a CPU energy meter that reads Intel RAPL counters
    directly from Linux powercap files.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class RAPLCPUEnergyMeter:
    """Measure CPU energy using Linux RAPL sysfs counters."""

    def __init__(self, powercap_root: str | Path = "/sys/class/powercap") -> None:
        self.powercap_root = Path(powercap_root)

    def measure(self, function: Callable[[], T]) -> tuple[T, dict[str, float | bool]]:
        """Measure CPU energy while executing a callable."""
        domains = self._discover_package_domains()
        if not domains:
            result = function()
            return result, {
                "cpu_rapl_available": False,
                "cpu_rapl_files_uj": 0.0,
                "cpu_rapl_files_j": 0.0,
            }

        start_snapshot = self._read_snapshot(domains)
        result = function()
        end_snapshot = self._read_snapshot(domains)
        energy_uj = self._compute_energy_uj(domains, start_snapshot, end_snapshot)

        return result, {
            "cpu_rapl_available": True,
            "cpu_rapl_files_uj": energy_uj,
            "cpu_rapl_files_j": energy_uj / 1e6,
        }

    def _discover_package_domains(self) -> list[Path]:
        """Discover top-level Intel RAPL package domains."""
        if not self.powercap_root.exists():
            return []

        domains: list[Path] = []
        for domain in sorted(self.powercap_root.iterdir()):
            if not domain.is_dir():
                continue
            if not domain.name.startswith("intel-rapl:"):
                continue
            if domain.name.count(":") != 1:
                continue
            if (domain / "energy_uj").exists() and (domain / "max_energy_range_uj").exists():
                domains.append(domain)

        return domains

    def _read_snapshot(self, domains: list[Path]) -> dict[Path, int]:
        """Read one snapshot of all package energy counters."""
        snapshot: dict[Path, int] = {}
        for domain in domains:
            snapshot[domain] = self._read_int(domain / "energy_uj")
        return snapshot

    def _compute_energy_uj(
        self,
        domains: list[Path],
        start_snapshot: dict[Path, int],
        end_snapshot: dict[Path, int],
    ) -> float:
        """Compute consumed energy in microjoules, handling counter wraparound."""
        total_uj = 0.0
        for domain in domains:
            start_value = start_snapshot[domain]
            end_value = end_snapshot[domain]

            if end_value >= start_value:
                delta = end_value - start_value
            else:
                max_range = self._read_int(domain / "max_energy_range_uj")
                delta = (max_range - start_value) + end_value

            total_uj += float(delta)

        return total_uj

    @staticmethod
    def _read_int(path: Path) -> int:
        """Read an integer from a sysfs file."""
        return int(path.read_text(encoding="utf-8").strip())
