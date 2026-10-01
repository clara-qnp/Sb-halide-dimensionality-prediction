"""Reproduce Figure 4: feature importance for the best-performing ML and LLM configurations.

Panels
------
(a) Permutation importance for the best ML model (feature_importance/permutation_importance.py's
    figures/ml_permutation_importance_no_title.png).
(b) Grounded LLM feature importance, Borda score by model, for the paired "with labeled examples"
    condition (llm_feature_importance/stage2_plot.py's
    llm_feature_importance/results_with_data/figures/figure1_feature_importance_heatmap.png).

This script only composes the two already-rendered panel images; it does not regenerate either
panel itself. See the README for how to produce them first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

BG = "#ffffff"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-a",
        type=Path,
        default=repository_root / "figures" / "ml_permutation_importance_no_title.png",
        help="ML permutation-importance panel (default: figures/ml_permutation_importance_no_title.png).",
    )
    parser.add_argument(
        "--panel-b",
        type=Path,
        default=repository_root / "llm_feature_importance" / "results_with_data" / "figures"
        / "figure1_feature_importance_heatmap.png",
        help=(
            "Grounded LLM feature-importance heatmap (default: "
            "llm_feature_importance/results_with_data/figures/figure1_feature_importance_heatmap.png)."
        ),
    )
    parser.add_argument("--output", type=Path, default=script_dir / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, path in (("--panel-a", args.panel_a), ("--panel-b", args.panel_b)):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} image: {path}. See the README for how to generate it first.")

    args.output.mkdir(parents=True, exist_ok=True)
    panels = [("a", args.panel_a), ("b", args.panel_b)]
    images = [mpimg.imread(path) for _, path in panels]

    # Give each panel a width proportional to its own image's aspect ratio (at a shared row
    # height) so imshow doesn't stretch or squash either one -- the two source panels have
    # quite different native aspect ratios (a tall bar chart vs. a wide heatmap).
    height_in = 6.6
    aspects = [image.shape[1] / image.shape[0] for image in images]
    width_in = height_in * sum(aspects)

    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in), gridspec_kw={"width_ratios": aspects})
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.04)

    for ax, image, (label, _) in zip(axes, images, panels):
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.01, 0.99, f"({label})", transform=ax.transAxes,
            fontsize=20, fontweight="bold", color="black", ha="left", va="top", clip_on=False,
        )

    fig.savefig(args.output / "figure_4.png", dpi=600, facecolor=BG)
    fig.savefig(args.output / "figure_4.pdf", facecolor=BG)
    plt.close(fig)
    print(f"Saved Figure 4 to {args.output.resolve()}")


if __name__ == "__main__":
    main()
