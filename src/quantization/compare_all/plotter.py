"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Performance Computing and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
File: quantization/compare_all/plotter.py

Description:
    Publication-quality figures comparing all quantization variants
    (FP32, FP16, BF16, INT8, INT4, INT2, INT1).  Each metric is saved as
    its own separate figure:

        1. Accuracy — vertical bar chart with FP32 reference line
           and per-bar accuracy-drop annotation.
        2. Model Size — vertical bar chart with compression-ratio labels
           and a note distinguishing actual from theoretical sizes.
        3. Accuracy vs. Compression Trade-off — scatter + line plot that
           shows the Pareto curve of accuracy against compression ratio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from .stats import VariantStats

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# One colour per variant — gradient from precision (blue) to aggression (red)
_COLORS: dict[str, str] = {
    "FP32": "#2B7BB9",
    "FP16": "#5DADE2",
    "BF16": "#48C9B0",
    "INT8": "#27AE60",
    "INT4": "#F39C12",
    "INT2": "#E67E22",
    "INT1": "#C0392B",
}

_EDGE_COLOR = "#2C3E50"
_GRID_COLOR = "#ECF0F1"
_TEXT_COLOR = "#2C3E50"
_REF_LINE_COLOR = "#7F8C8D"

_FONT_FAMILY = "DejaVu Sans"


class AllQuantizationPlotter:
    """Generate three separate publication-quality comparison figures.

    The figures compare FP32, FP16, BF16, INT8, INT4, INT2 and INT1 across:
        - Accuracy.
        - Model size on disk / theoretical packed size.
        - Accuracy vs. compression trade-off curve.
    """

    def plot(
        self,
        stats: list[VariantStats],
        output_dir: str | Path,
        basename: str = "all_quantization_comparison",
    ) -> list[Path]:
        """Save the comparison as three separate figures in *output_dir*.

        Args:
            stats: One ``VariantStats`` per variant, ordered FP32 → INT1.
            output_dir: Destination directory for the three PNG figures.
            basename: Common prefix for the file names.  The suffixes
                ``_accuracy``, ``_size`` and ``_tradeoff`` are appended.

        Returns:
            The list of paths written, in the order accuracy, size, trade-off.
        """
        self._apply_style()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        panels = (
            ("accuracy", self._plot_accuracy, "VisionTransformer — Accuracy"),
            ("size", self._plot_size, "VisionTransformer — Model Size"),
            (
                "tradeoff",
                self._plot_tradeoff,
                "VisionTransformer — Accuracy vs. Compression Trade-off",
            ),
        )

        saved: list[Path] = []
        for suffix, panel_fn, title in panels:
            path = output_dir / f"{basename}_{suffix}.png"
            self._make_single_figure(panel_fn, stats, path, title)
            saved.append(path)

        return saved

    # ------------------------------------------------------------------
    # Figure assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_style() -> None:
        """Apply the shared Matplotlib style for all figures."""
        plt.rcParams.update({
            "font.family": _FONT_FAMILY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": _GRID_COLOR,
            "grid.linewidth": 0.8,
            "text.color": _TEXT_COLOR,
            "axes.labelcolor": _TEXT_COLOR,
            "xtick.color": _TEXT_COLOR,
            "ytick.color": _TEXT_COLOR,
        })

    def _make_single_figure(
        self,
        panel_fn,
        stats: list[VariantStats],
        output_path: Path,
        title: str,
    ) -> None:
        """Build one standalone figure for a single metric panel.

        Args:
            panel_fn: One of the ``_plot_*`` panel-drawing methods.
            stats: All variant statistics.
            output_path: Destination path for the PNG figure.
            title: Figure-specific title shown at the top.
        """
        fp32_acc = stats[0].accuracy

        fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="white")
        panel_fn(ax, stats)

        fig.suptitle(
            title,
            fontsize=15, fontweight="bold", color=_TEXT_COLOR,
        )
        fig.text(
            0.5, 0.915,
            f"20-class dataset · FP32 baseline: {fp32_acc:.2f}%",
            ha="center", va="top",
            fontsize=9, color=_REF_LINE_COLOR, style="italic",
        )

        self._add_legend(fig, stats)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.20)

        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Figure saved → {output_path}")

    # ------------------------------------------------------------------
    # Panel 1 — Accuracy
    # ------------------------------------------------------------------

    def _plot_accuracy(self, ax, stats: list[VariantStats]) -> None:
        """Draw the Accuracy bar chart.

        Args:
            ax: Target Axes.
            stats: All variant statistics.
        """
        labels = [s.label for s in stats]
        accs = [s.accuracy for s in stats]
        colors = [_COLORS[s.label] for s in stats]
        fp32_acc = stats[0].accuracy

        bars = ax.bar(
            labels, accs,
            color=colors,
            width=0.55,
            edgecolor=_EDGE_COLOR,
            linewidth=0.8,
            zorder=3,
        )

        # FP32 reference dashed line
        ax.axhline(
            fp32_acc,
            color=_REF_LINE_COLOR,
            linestyle="--",
            linewidth=1.2,
            zorder=2,
            label=f"FP32 baseline ({fp32_acc:.2f}%)",
        )

        # Y-axis zoom: start slightly below min accuracy
        y_min = max(0.0, min(accs) - 5.0)
        y_max = fp32_acc + 4.0
        ax.set_ylim(y_min, y_max)

        # Value + delta annotations
        for bar, acc, stat in zip(bars, accs, stats):
            delta = acc - fp32_acc
            delta_str = "" if stat.bits == 32 else f"\n({delta:+.1f}%)"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.3,
                f"{acc:.2f}%{delta_str}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=_TEXT_COLOR,
                zorder=4,
            )

        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Panel 2 — Model Size
    # ------------------------------------------------------------------

    def _plot_size(self, ax, stats: list[VariantStats]) -> None:
        """Draw the Model Size bar chart with compression ratio labels.

        Args:
            ax: Target Axes.
            stats: All variant statistics.
        """
        labels = [s.label for s in stats]
        sizes = [s.size_mb for s in stats]
        colors = [_COLORS[s.label] for s in stats]
        fp32_size = stats[0].size_mb

        bars = ax.bar(
            labels, sizes,
            color=colors,
            width=0.55,
            edgecolor=_EDGE_COLOR,
            linewidth=0.8,
            zorder=3,
        )

        ax.set_ylim(0, fp32_size * 1.35)

        for bar, size, stat in zip(bars, sizes, stats):
            ratio = fp32_size / max(size, 1e-9)
            ratio_str = "1×" if stat.bits == 32 else f"{ratio:.1f}×"
            size_note = "†" if stat.theoretical else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + fp32_size * 0.015,
                f"{size:.1f} MB\n{ratio_str}{size_note}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color=_TEXT_COLOR,
                zorder=4,
            )

        # Footnote for theoretical sizes
        ax.text(
            0.98, 0.02,
            "† theoretical packed size",
            transform=ax.transAxes,
            fontsize=7.5, color=_REF_LINE_COLOR,
            ha="right", va="bottom",
            style="italic",
        )

        ax.set_ylabel("Size (MB)", fontsize=10)
        ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Panel 3 — Trade-off curve
    # ------------------------------------------------------------------

    def _plot_tradeoff(self, ax, stats: list[VariantStats]) -> None:
        """Draw the Accuracy vs. Compression ratio trade-off scatter.

        Args:
            ax: Target Axes.
            stats: All variant statistics.
        """
        ratios = [s.compression_ratio for s in stats]
        accs = [s.accuracy for s in stats]
        colors = [_COLORS[s.label] for s in stats]
        fp32_acc = stats[0].accuracy

        # Connecting line (grey, behind points)
        ax.plot(
            ratios, accs,
            color="#BDC3C7",
            linewidth=1.8,
            linestyle="-",
            zorder=2,
        )

        # Scatter points
        for ratio, acc, color, stat in zip(ratios, accs, colors, stats):
            ax.scatter(
                ratio, acc,
                s=130,
                color=color,
                edgecolors=_EDGE_COLOR,
                linewidths=0.8,
                zorder=5,
            )
            # Label offset: avoid overlapping points that share a ratio.
            # FP16 and BF16 both sit at 2× → split them vertically.
            offset_x = ratio * 0.08
            offset_y = 0.4
            if stat.label == "FP32":
                offset_x = -ratio * 0.05
                offset_y = -1.2
            elif stat.label == "FP16":
                offset_y = 0.6
            elif stat.label == "BF16":
                offset_y = -1.4
            ax.text(
                ratio + offset_x,
                acc + offset_y,
                stat.label,
                fontsize=8.5,
                fontweight="bold",
                color=color,
                zorder=6,
            )

        # FP32 reference horizontal line
        ax.axhline(
            fp32_acc,
            color=_REF_LINE_COLOR,
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )

        # Shade the area of acceptable accuracy loss (within 2%)
        ax.axhspan(
            fp32_acc - 2.0, fp32_acc,
            alpha=0.07,
            color="#27AE60",
            zorder=0,
            label="Within 2% of FP32",
        )

        y_min = min(accs) - 4.0
        y_max = fp32_acc + 3.0
        ax.set_ylim(max(0.0, y_min), y_max)
        ax.set_xlim(0.5, max(ratios) * 1.15)

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}×")
        )
        ax.set_xticks([1, 2, 4, 8, 16, 32])

        ax.set_xlabel("Compression ratio (log₂ scale)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.grid(True, which="both", color=_GRID_COLOR, linewidth=0.7)
        ax.set_axisbelow(True)

        ax.legend(
            fontsize=8, loc="lower left",
            framealpha=0.85, edgecolor="#BDC3C7",
        )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    @staticmethod
    def _add_legend(fig, stats: list[VariantStats]) -> None:
        """Add a shared colour legend below the panel.

        Args:
            fig: The Figure object.
            stats: Variants to include in the legend.
        """
        patches = [
            mpatches.Patch(
                facecolor=_COLORS[s.label],
                edgecolor=_EDGE_COLOR,
                linewidth=0.6,
                label=f"{s.full_label}  ({s.bits}-bit)",
            )
            for s in stats
        ]
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(len(stats), 4),
            fontsize=9,
            framealpha=0.9,
            edgecolor="#BDC3C7",
            bbox_to_anchor=(0.5, 0.005),
        )
