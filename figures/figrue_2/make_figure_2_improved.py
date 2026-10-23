"""Improved Figure 2: dataset composition and organic-cation chemical space.

A redesign of make_figure_2.py following Knaflic, *Storytelling with Data* (ch. 1-5).

Changes from the original
-------------------------
* Panels (a) and (b) are merged into one 100% bar chart ("All" + each halide), so the
  reader compares halide families against the whole dataset in one place.
* 0D (the majority) is neutral gray; only the non-0D classes get color, using the
  colorblind-safe Okabe-Ito hues so 1D, 2D and 3D are easy to tell apart. In the original, orange meant "0D" in (a)/(b) but
  "original cations" in (c).
* Non-0D share is labeled at the end of every bar, so the F -> I trend needs no arithmetic.
* Sample sizes live in the row labels (the "n =" labels overlapped in the original).
* In the UMAP panel, existing cations are recessive gray and the new cations are the
  highlight; the legend no longer sits on top of data points. UMAP tick values carry no
  physical meaning, so the ticks are removed.
* Action titles state the takeaway. Everything is drawn in one matplotlib figure instead
  of stitching PNGs together, so text stays sharp and sizes stay consistent.

Usage:  python figures/figrue_2/make_figure_2_improved.py
Output: figures/figrue_2/output_improved/figure_2_improved.png (+ .pdf)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIMS = ["0D", "1D", "2D", "3D"]
HALIDES = ["F", "Cl", "Br", "I"]
COLOR = {"0D": "#DCDAD5", "1D": "#0072B2", "2D": "#009E73", "3D": "#D55E00"}  # Okabe-Ito, colorblind-safe
INK, MUTED, FAINT = "#1F1F1F", "#6B6B6B", "#C9C7C1"
HIGHLIGHT = "#7B3294"  # new cations; distinct from the dimensionality colors


def plot_composition(ax, data: pd.DataFrame) -> None:
    rows = [("All", data)] + [(h, data[data["Sb_halide"] == h]) for h in HALIDES]
    labels, y_positions = [], []
    for i, (name, subset) in enumerate(rows):
        y = -i - (0.35 if i > 0 else 0)  # small gap between "All" and the halide rows
        counts = subset["dimensionality"].value_counts().reindex(DIMS, fill_value=0)
        total = counts.sum()
        left = 0.0
        for dim in DIMS:
            width = counts[dim] / total
            if width == 0:
                continue
            ax.barh(y, width, left=left, height=0.62, color=COLOR[dim],
                    edgecolor="white", linewidth=1.5)
            if width >= 0.08:
                ax.text(left + width / 2, y, f"{width:.0%}", ha="center", va="center",
                        fontsize=11, color=INK if dim in ("0D", "2D") else "white")
            left += width
        non0d = 1 - counts["0D"] / total
        ax.text(1.03, y, f"{non0d:.0%}", ha="left", va="center", fontsize=12,
                fontweight="bold", color=INK if non0d > 0 else MUTED)
        labels.append(f"{name}  (n = {total})")
        y_positions.append(y)

    ax.set_yticks(y_positions, labels, fontsize=12, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_xlim(0, 1.14)
    ax.set_ylim(min(y_positions) - 0.6, 1.05)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compact legend above the bars, plus a header for the non-0D column.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[d], label=d) for d in DIMS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4,
              frameon=False, fontsize=11, handlelength=1.1, handleheight=1.1,
              columnspacing=1.2, borderaxespad=0.2)
    ax.text(1.03, y_positions[0] + 0.55, "non-0D", ha="left", va="bottom", fontsize=11,
            color=INK, fontweight="bold")

    ax.set_title("(a)  Non-0D share rises from F to I", loc="left", fontsize=13.5,
                 fontweight="bold", color=INK, pad=34)


def plot_cation_space(ax, original: pd.DataFrame, proposed: pd.DataFrame) -> None:
    ax.scatter(original["UMAP_1"], original["UMAP_2"], s=18, color=FAINT,
               edgecolors="none", zorder=2)
    new_only = proposed[~proposed["in_original"]]
    overlap = proposed[proposed["in_original"]]
    ax.scatter(new_only["UMAP_1"], new_only["UMAP_2"], s=70, facecolors="white",
               edgecolors=HIGHLIGHT, linewidth=2, zorder=3)
    ax.scatter(overlap["UMAP_1"], overlap["UMAP_2"], s=70, marker="s", color=HIGHLIGHT, zorder=3)

    # Direct labels instead of a legend that covers points.
    ax.annotate(f"{len(original)} cations in the\nliterature dataset", xy=(9.3, 2.1),
                xytext=(8.0, 5.6), fontsize=11, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-", color=FAINT, lw=1))
    ax.annotate(f"{len(new_only)} new proposed cations\n(open circles)", xy=(5.65, 3.2),
                xytext=(6.4, 2.1), fontsize=11, color=HIGHLIGHT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=HIGHLIGHT, lw=1))
    ax.annotate(f"{len(overlap)} proposed cations already\nin the dataset (squares)", xy=(3.75, 5.55),
                xytext=(2.6, 7.75), fontsize=11, color=HIGHLIGHT,
                arrowprops=dict(arrowstyle="-", color=HIGHLIGHT, lw=1))

    ax.set_xlim(2.4, 10.8)
    ax.set_ylim(0.5, 8.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=11, color=MUTED, loc="left")
    ax.set_ylabel("UMAP 2", fontsize=11, color=MUTED, loc="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.set_title("(b)  New cations sit inside the known chemical space", loc="left",
                 fontsize=13.5, fontweight="bold", color=INK, pad=33)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "merged_compound_cif_cation.xlsx")
    parser.add_argument("--coordinates-dir", type=Path, default=script_dir / "reference_coordinates")
    parser.add_argument("--output", type=Path, default=script_dir / "output_improved")
    args = parser.parse_args()

    data = pd.read_excel(args.data)
    data["dimensionality"] = data["dimensionality"].astype(str).str.strip()
    data = data[data["dimensionality"].isin(DIMS)].copy()
    original = pd.read_csv(args.coordinates_dir / "umap_original_cations.csv")
    proposed = pd.read_csv(args.coordinates_dir / "umap_new_cations.csv")

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6.4),
                                     gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.12, wspace=0.12)

    plot_composition(ax_a, data)
    plot_cation_space(ax_b, original, proposed)

    fig.text(0.015, 0.95, "About a third of Sb-halide compounds are non-0D, and heavier halides "
             "form more extended structures", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.015, 0.905, f"Dimensionality of {len(data)} literature compounds, and where the "
             "proposed organic cations fall in descriptor space", fontsize=12.5, color=MUTED)
    fig.text(0.015, 0.03, "\"All\" also includes 10 mixed-halide compounds not shown as "
             "separate rows. F has only 12 compounds, so its 0% should be read with caution.",
             fontsize=10, color=MUTED)

    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "figure_2_improved.png", dpi=300, facecolor="white")
    fig.savefig(args.output / "figure_2_improved.pdf", facecolor="white")
    print(f"Saved improved Figure 2 to {args.output.resolve()}")


if __name__ == "__main__":
    main()
