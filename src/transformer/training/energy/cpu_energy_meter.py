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

import typing as t
from pathlib import Path

T = t.TypeVar("T")


class RAPLCPUEnergyMeter:
    """Measure CPU energy using Linux RAPL sysfs counters."""

    def __init__(self, powercap_root: t.Union[str, Path] = "/sys/class/powercap") -> None:
        self.powercap_root = Path(powercap_root)

    def measure(
        self,
        function: t.Callable[[], T],
    ) -> t.Tuple[T, t.Dict[str, t.Union[float, bool]]]:
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

    def _discover_package_domains(self) -> t.List[Path]:
        """Discover top-level Intel RAPL package domains."""
        if not self.powercap_root.exists():
            return []

        domains = []
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

    def _read_snapshot(self, domains: t.List[Path]) -> t.Dict[Path, int]:
        """Read one snapshot of all package energy counters."""
        snapshot = {}
        for domain in domains:
            snapshot[domain] = self._read_int(domain / "energy_uj")
        return snapshot

    def _compute_energy_uj(
        self,
        domains: t.List[Path],
        start_snapshot: t.Dict[Path, int],
        end_snapshot: t.Dict[Path, int],
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
