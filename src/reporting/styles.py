from pathlib import Path

import matplotlib as mpl


CLASS_COLORS = {
    "AF": "#4C78A8",
    "N": "#54A24B",
    "PAT": "#E45756",
}

SPLIT_COLORS = {
    "val": "#4C78A8",
    "external": "#E45756",
    "train": "#72B7B2",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig, output_dir: str | Path, stem: str, dpi: int = 600) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".svg", ".pdf"]:
        fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")

