"""Reproduce Figure 2: dataset composition and organic-cation chemical space.

Panels
------
(a) Counts of 0D and non-0D compounds, with non-0D split into 1D/2D/3D.
(b) Dimensionality composition of the F, Cl, Br, and I subsets.
(c) UMAP of RDKit descriptors for original and newly proposed cations.

The script reads the curated compound workbook for panels (a) and (b), and the UMAP
coordinate CSVs created by generate_umap_coordinates.py for panel (c). It does not
require the large model-training feature table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMS = ["0D", "1D", "2D", "3D"]
PURE_HALIDES = ["F", "Cl", "Br", "I"]
DIM_COLOR = {"0D": "#E69F00", "1D": "#0072B2", "2D": "#009E73", "3D": "#D55E00"}
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e3e2dc", "#c7c6bf"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 19,
            "axes.linewidth": 0.8,
            "axes.labelsize": 20,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
        }
    )


def style_axes(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)


def declutter(centers: list[float], min_gap: float) -> list[float]:
    positions = list(centers)
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < min_gap:
            positions[index] = positions[index - 1] + min_gap
    for index in range(len(positions) - 2, -1, -1):
        if positions[index + 1] - positions[index] < min_gap:
            positions[index] = positions[index + 1] - min_gap
    return positions


def plot_panel_a(data: pd.DataFrame, output: Path) -> None:
    counts = data["dimensionality"].value_counts().reindex(DIMS, fill_value=0)
    total = int(counts.sum())
    nonzero_dims = ["1D", "2D", "3D"]
    nonzero_total = int(counts[nonzero_dims].sum())

    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    fig.patch.set_facecolor("white")
    bar_width, x_0d, x_nonzero = 0.4, 0.0, 1.0
    y_max = max(int(counts["0D"]), nonzero_total) * 1.18
    ax.bar(x_0d, counts["0D"], width=bar_width, color=DIM_COLOR["0D"], zorder=3)

    # A small visual floor keeps the 3D segment visible. Counts and labels remain exact.
    true_heights = {dim: int(counts[dim]) for dim in nonzero_dims}
    min_visual = y_max * 0.009
    deficits = {dim: max(min_visual - value, 0) for dim, value in true_heights.items()}
    visual_heights = dict(true_heights)
    total_deficit = sum(deficits.values())
    if total_deficit:
        donor = max(true_heights, key=true_heights.get)
        visual_heights[donor] -= total_deficit
        for dim, deficit in deficits.items():
            visual_heights[dim] += deficit

    bottom = 0.0
    segments = []
    for dim in nonzero_dims:
        height = visual_heights[dim]
        ax.bar(x_nonzero, height, width=bar_width, bottom=bottom, color=DIM_COLOR[dim], zorder=3)
        segments.append((dim, true_heights[dim], bottom + height / 2))
        bottom += height

    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.55, x_nonzero + 0.75)
    ax.set_xticks([x_0d, x_nonzero], ["0D", "non-0D"])
    ax.set_ylabel("number of compounds")
    style_axes(ax)
    ax.text(
        x_0d,
        int(counts["0D"]) + y_max * 0.015,
        f"{int(counts['0D'])}\n{counts['0D'] / total * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=19,
        linespacing=1.25,
    )

    right_edge = x_nonzero + bar_width / 2
    label_y = declutter([center for _, _, center in segments], y_max * 0.14)
    for (dim, value, center), y_value in zip(segments, label_y):
        ax.plot([right_edge, right_edge + 0.08], [center, y_value], color=INK_SOFT, lw=0.7, zorder=4)
        ax.text(
            right_edge + 0.11,
            y_value,
            f"{dim}: {value}\n{value / total * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=18,
            linespacing=1.2,
        )
    fig.tight_layout()
    fig.savefig(output, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_panel_b(data: pd.DataFrame, output: Path) -> None:
    subset = data[data["Sb_halide"].isin(PURE_HALIDES)]
    counts = (
        subset.groupby(["Sb_halide", "dimensionality"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=PURE_HALIDES, columns=DIMS, fill_value=0)
    )
    totals = counts.sum(axis=1)
    fractions = counts.div(totals, axis=0)

    # As in panel (a), preserve tiny nonzero classes with a visual floor.
    visual = fractions.copy()
    min_visual = 1.10 * 0.009
    for halide in PURE_HALIDES:
        row = fractions.loc[halide]
        deficits = {dim: min_visual - row[dim] for dim in DIMS if 0 < row[dim] < min_visual}
        if deficits:
            visual.loc[halide, row.idxmax()] -= sum(deficits.values())
            for dim, deficit in deficits.items():
                visual.loc[halide, dim] += deficit

    x_values = np.arange(len(PURE_HALIDES))
    bottoms = np.zeros(len(PURE_HALIDES))
    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    fig.patch.set_facecolor("white")
    for dim in DIMS:
        visible_values = visual[dim].to_numpy()
        true_values = fractions[dim].to_numpy()
        ax.bar(
            x_values,
            visible_values,
            0.52,
            bottom=bottoms,
            color=DIM_COLOR[dim],
            label=dim,
            zorder=3,
        )
        for index, (visible, true) in enumerate(zip(visible_values, true_values)):
            if true > 0.06:
                ax.text(
                    x_values[index],
                    bottoms[index] + visible / 2,
                    f"{true * 100:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=18,
                    fontweight="medium",
                )
        bottoms += visible_values

    for index, total in enumerate(totals.to_numpy()):
        ax.text(x_values[index], 1.02, f"n = {int(total)}", ha="center", va="bottom", fontsize=18, color=INK_SOFT)
    ax.set_ylim(0, 1.09)
    ax.set_xticks(x_values, PURE_HALIDES)
    ax.set_ylabel("percentage of compounds")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        handleheight=1.2,
        columnspacing=1.4,
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_panel_c(original: pd.DataFrame, proposed: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    fig.patch.set_facecolor("white")
    style_axes(ax, grid_axis=None)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.scatter(
        original["UMAP_1"], original["UMAP_2"], s=22, alpha=0.9,
        label="Original cations", color="#D55E00", edgecolors="none", zorder=2,
    )
    new_only = proposed[~proposed["in_original"]]
    overlap = proposed[proposed["in_original"]]
    ax.scatter(
        new_only["UMAP_1"], new_only["UMAP_2"], s=44, marker="o",
        label="New cations", facecolors="none", edgecolors="#0072B2", linewidth=1.8, zorder=3,
    )
    ax.scatter(
        overlap["UMAP_1"], overlap["UMAP_2"], s=44, marker="s",
        label="Overlapping cations", facecolors="none", edgecolors="#009E73", linewidth=1.8, zorder=3,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def compose_figure(panel_paths: list[Path], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.0))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.995, top=0.995, bottom=0.005, wspace=0.03)
    label_x = {"a": -0.045, "b": 0.01, "c": 0.01}
    for ax, label, path in zip(axes, ["a", "b", "c"], panel_paths):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.text(
            label_x[label], 0.99, f"({label})", transform=ax.transAxes,
            fontsize=24, fontweight="bold", ha="left", va="top", clip_on=False,
        )
    fig.savefig(output_dir / "figure_2.png", dpi=600, facecolor="white")
    fig.savefig(output_dir / "figure_2.pdf", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[1]
    workspace_root = script_dir.parents[2]

    def default_input(filename: str) -> Path:
        """Support both the public-repository layout and this analysis workspace."""
        candidates = [
            repository_root / filename,
            repository_root / "data" / filename,
            workspace_root / filename,
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=default_input("merged_compound_cif_cation.xlsx"),
        help="Curated compound workbook (default: data/merged_compound_cif_cation.xlsx).",
    )
    parser.add_argument(
        "--coordinates-dir",
        type=Path,
        default=script_dir / "reference_coordinates",
        help=(
            "Directory containing umap_original_cations.csv and "
            "umap_new_cations.csv (default: published reference_coordinates)."
        ),
    )
    parser.add_argument("--output", type=Path, default=script_dir / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.data)
    required = {"dimensionality", "Sb_halide"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{args.data} is missing required columns: {sorted(missing)}")
    data["dimensionality"] = data["dimensionality"].astype(str).str.strip()
    data = data[data["dimensionality"].isin(DIMS)].copy()
    if len(data) != 400:
        print(f"Warning: expected 400 known-dimensionality compounds, found {len(data)}")

    panel_a = args.output / "panel_a_dimensionality_counts.png"
    panel_b = args.output / "panel_b_dimensionality_by_halide.png"
    panel_c = args.output / "panel_c_cation_umap.png"
    plot_panel_a(data, panel_a)
    plot_panel_b(data, panel_b)
    reference_original = args.coordinates_dir / "umap_original_cations.csv"
    reference_new = args.coordinates_dir / "umap_new_cations.csv"
    missing_coordinates = [path for path in (reference_original, reference_new) if not path.exists()]
    if missing_coordinates:
        raise FileNotFoundError(
            f"Missing UMAP coordinate file(s): {missing_coordinates}. "
            "Run figures/figrue_2/generate_umap_coordinates.py first."
        )
    original = pd.read_csv(reference_original)
    proposed = pd.read_csv(reference_new)
    print("Using the saved UMAP coordinate datasets for panel (c).")
    original[["canonical_unique_cation", "UMAP_1", "UMAP_2"]].to_csv(
        args.output / "umap_original_cations.csv", index=False
    )
    proposed[["cation_name", "SMILES", "in_original", "UMAP_1", "UMAP_2"]].to_csv(
        args.output / "umap_new_cations.csv", index=False
    )
    plot_panel_c(original, proposed, panel_c)
    compose_figure([panel_a, panel_b, panel_c], args.output)
    print(f"Saved Figure 2 and its component panels to {args.output.resolve()}")


if __name__ == "__main__":
    main()
