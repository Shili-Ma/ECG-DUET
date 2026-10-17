import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.styles import SPLIT_COLORS, apply_publication_style, save_figure


MAIN_COLOR = "#4C78A8"
DROP_COLOR = "#E45756"
NEUTRAL = "#6B7280"
ACCENT = "#72B7B2"
MODULE_COLUMNS = [
    ("use_time_branch", "Time"),
    ("use_spatial_branch", "Spatial"),
    ("use_idcnn", "IDCNN"),
    ("use_dynamic_fusion", "Fusion"),
    ("use_moe", "MoE"),
    ("use_fourier_block", "Fourier"),
]
ABLATION_LABELS = {
    "Main_Model": "Full model",
    "2_No_Time_Branch": "w/o temporal branch",
    "3_No_Spatial_Branch": "w/o spatial branch",
    "4_No_IDCNN": "w/o IDCNN block",
    "5_No_Dynamic_Fusion": "w/o dynamic fusion",
    "6_No_MoE": "w/o MoE module",
    "7_No_Fourier_Block": "w/o Fourier block",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone ablation plot design options")
    parser.add_argument("--ablation_summary", default="results/ablations/ablation_summary.csv")
    parser.add_argument("--output_dir", default="results/figures/ablation_options")
    args = parser.parse_args()

    apply_publication_style()
    df = pd.read_csv(args.ablation_summary)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.append(plot_ablation_drop(df, out))
    generated.append(plot_ablation_dumbbell(df, out))
    generated.append(plot_ablation_module_matrix(df, out))
    generated.append(plot_ablation_composite(df, out))
    print("Generated ablation plot options:", ", ".join(generated))


def plot_ablation_drop(df: pd.DataFrame, out: Path) -> str:
    plot_df, main_score = _ablation_rows(df)
    plot_df["drop"] = main_score - plot_df["external_macro_f1"]
    plot_df = plot_df.sort_values("drop", ascending=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    colors = [DROP_COLOR if x > 0 else ACCENT for x in plot_df["drop"]]
    ax.barh(plot_df["label"], plot_df["drop"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    for y, value in enumerate(plot_df["drop"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.004 if value >= 0 else -0.004
        ax.text(value + offset, y, f"{value:.3f}", va="center", ha=ha, fontsize=6)
    ax.set_title("Ablation contribution")
    ax.set_xlabel("External macro-F1 decrease from main model")
    ax.set_ylabel("")
    _set_xmargin(ax, np.r_[plot_df["drop"].to_numpy(), 0], pad=0.02)
    fig.tight_layout()
    save_figure(fig, out, "ablation_option1_drop_contribution")
    plt.close(fig)
    return "ablation_option1_drop_contribution"


def plot_ablation_dumbbell(df: pd.DataFrame, out: Path) -> str:
    plot_df, _ = _ablation_rows(df)
    plot_df = plot_df.sort_values("external_macro_f1", ascending=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.plot(
            [row.external_macro_f1, row.val_macro_f1],
            [idx, idx],
            color="#9CA3AF",
            linewidth=1.1,
            zorder=1,
        )
    ax.scatter(plot_df["val_macro_f1"], y, s=22, color=SPLIT_COLORS["val"], label="Internal test", zorder=2)
    ax.scatter(plot_df["external_macro_f1"], y, s=22, color=SPLIT_COLORS["external"], label="External test", zorder=2)
    ax.set_yticks(y, plot_df["label"])
    ax.set_title("Ablation internal-external shift")
    ax.set_xlabel("Macro-F1")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    _set_score_xlim(ax, plot_df[["val_macro_f1", "external_macro_f1"]].to_numpy().ravel())
    fig.tight_layout()
    save_figure(fig, out, "ablation_option2_dumbbell")
    plt.close(fig)
    return "ablation_option2_dumbbell"


def plot_ablation_module_matrix(df: pd.DataFrame, out: Path) -> str:
    plot_df = df.copy()
    plot_df["label"] = plot_df["experiment"].map(_clean_ablation_label)
    plot_df["external_macro_f1"] = pd.to_numeric(plot_df["external_macro_f1"], errors="coerce")
    plot_df = plot_df.dropna(subset=["external_macro_f1"]).sort_values("external_macro_f1", ascending=True)

    fig, (ax_matrix, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(5.2, 3.2),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.08},
        sharey=True,
    )
    module_cols = [(col, label) for col, label in MODULE_COLUMNS if col in plot_df.columns]
    matrix = plot_df[[col for col, _ in module_cols]].astype(bool).to_numpy(dtype=int)
    ax_matrix.imshow(matrix, aspect="auto", cmap=_module_cmap(), vmin=0, vmax=1)
    ax_matrix.set_yticks(np.arange(len(plot_df)), plot_df["label"])
    ax_matrix.set_xticks(np.arange(len(module_cols)), [label for _, label in module_cols], rotation=35, ha="right")
    ax_matrix.set_title("Module retained")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_matrix.text(j, i, "Y" if matrix[i, j] else "N", ha="center", va="center", fontsize=5.5, color="white" if matrix[i, j] else "#111111")

    colors = [MAIN_COLOR if str(name).lower() == "main_model" else ACCENT for name in plot_df["experiment"]]
    ax_bar.barh(np.arange(len(plot_df)), plot_df["external_macro_f1"], color=colors)
    ax_bar.set_title("Performance")
    ax_bar.set_xlabel("External macro-F1")
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    _set_score_xlim(ax_bar, plot_df["external_macro_f1"])
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.23, top=0.88, wspace=0.08)
    save_figure(fig, out, "ablation_option3_module_matrix")
    plt.close(fig)
    return "ablation_option3_module_matrix"


def plot_ablation_composite(df: pd.DataFrame, out: Path) -> str:
    plot_df = df.copy()
    plot_df["label"] = plot_df["experiment"].map(_clean_ablation_label)
    for col in ["external_macro_f1", "val_macro_f1"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    main = plot_df[plot_df["experiment"].astype(str).str.lower().isin(["main_model", "proposed"])]
    main_external = float(main["external_macro_f1"].iloc[0]) if not main.empty else float(plot_df["external_macro_f1"].max())
    main_internal = float(main["val_macro_f1"].iloc[0]) if not main.empty else float(plot_df["val_macro_f1"].max())
    plot_df["external_drop"] = main_external - plot_df["external_macro_f1"]
    plot_df["internal_drop"] = main_internal - plot_df["val_macro_f1"]
    plot_df = plot_df.dropna(subset=["external_macro_f1", "val_macro_f1", "external_drop", "internal_drop"]).sort_values("external_macro_f1", ascending=True)
    y = np.arange(len(plot_df))

    fig, (ax_module, ax_perf, ax_drop, ax_shift) = plt.subplots(
        1,
        4,
        figsize=(7.2, 3.6),
        gridspec_kw={"width_ratios": [0.9, 1.15, 1.15, 1.35], "wspace": 0.12},
        sharey=True,
    )

    module_cols = [(col, label) for col, label in MODULE_COLUMNS if col in plot_df.columns]
    matrix = plot_df[[col for col, _ in module_cols]].astype(bool).to_numpy(dtype=int)
    ax_module.imshow(matrix, aspect="auto", cmap=_module_cmap(), vmin=0, vmax=1)
    ax_module.set_yticks(y, plot_df["label"])
    ax_module.set_xticks(np.arange(len(module_cols)), [label for _, label in module_cols], rotation=35, ha="right")
    ax_module.set_title("")
    ax_module.tick_params(axis="y", length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_module.text(
                j,
                i,
                "Y" if matrix[i, j] else "N",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if matrix[i, j] else "#111111",
            )

    perf_offset = 0.12
    for values, offset, color, label in [
        (plot_df["val_macro_f1"], perf_offset, SPLIT_COLORS["val"], "Internal test"),
        (plot_df["external_macro_f1"], -perf_offset, SPLIT_COLORS["external"], "External test"),
    ]:
        ax_perf.plot(values, y + offset, color="#CBD5E1", linewidth=0.8, zorder=1)
        for idx, value in enumerate(values):
            ax_perf.hlines(idx + offset, value - 0.010, value + 0.010, color=color, linewidth=2.6, zorder=2)
            ax_perf.scatter(value, idx + offset, s=14, color=color, edgecolor="white", linewidth=0.5, zorder=3, label=label if idx == 0 else None)
            ax_perf.text(
                value + 0.018,
                idx + offset,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=5.2,
                zorder=10,
                clip_on=False,
            )
    ax_perf.set_title("")
    ax_perf.set_xlabel("Macro-F1")
    _hide_shared_y_axis(ax_perf)
    _set_score_xlim(ax_perf, plot_df[["val_macro_f1", "external_macro_f1"]].to_numpy().ravel())
    left, right = ax_perf.get_xlim()
    ax_perf.set_xlim(left, min(1.04, right + 0.025))

    ax_drop.axvline(0, color="#333333", linewidth=0.8)
    drop_offset = 0.12
    for values, offset, color, label in [
        (plot_df["internal_drop"], drop_offset, SPLIT_COLORS["val"], "Internal test"),
        (plot_df["external_drop"], -drop_offset, SPLIT_COLORS["external"], "External test"),
    ]:
        for idx, value in enumerate(values):
            ax_drop.hlines(idx + offset, 0, value, color=color, linewidth=1.6, zorder=1)
            ax_drop.scatter(value, idx + offset, s=14, color=color, edgecolor="white", linewidth=0.5, zorder=2, label=label if idx == 0 else None)
            ax_drop.text(
                value + 0.004,
                idx + offset,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=5.2,
                zorder=10,
                clip_on=False,
            )
    ax_drop.set_title("")
    ax_drop.set_xlabel("Delta macro-F1\nvs full model")
    _hide_shared_y_axis(ax_drop)
    _set_xmargin(ax_drop, np.r_[plot_df["internal_drop"].to_numpy(), plot_df["external_drop"].to_numpy(), 0], pad=0.02)

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax_shift.plot(
            [row.external_macro_f1, row.val_macro_f1],
            [idx, idx],
            color="#9CA3AF",
            linewidth=1.0,
            zorder=1,
        )
    ax_shift.scatter(plot_df["val_macro_f1"], y, s=18, color=SPLIT_COLORS["val"], label="Internal test", zorder=2)
    ax_shift.scatter(plot_df["external_macro_f1"], y, s=18, color=SPLIT_COLORS["external"], label="External test", zorder=2)
    ax_shift.set_title("")
    ax_shift.set_xlabel("Macro-F1")
    _hide_shared_y_axis(ax_shift)
    ax_shift.legend(loc="upper left", frameon=True, framealpha=0.9)
    _set_score_xlim(ax_shift, plot_df[["val_macro_f1", "external_macro_f1"]].to_numpy().ravel())

    for ax in [ax_module, ax_perf, ax_drop, ax_shift]:
        ax.set_ylim(-0.6, len(plot_df) - 0.4)

    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.24, top=0.96, wspace=0.12)
    save_figure(fig, out, "ablation_option4_composite")
    plt.close(fig)
    return "ablation_option4_composite"


def _ablation_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    plot_df = df.copy()
    plot_df["label"] = plot_df["experiment"].map(_clean_ablation_label)
    for col in ["external_macro_f1", "val_macro_f1"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    main = plot_df[plot_df["experiment"].astype(str).str.lower().isin(["main_model", "proposed"])]
    main_score = float(main["external_macro_f1"].iloc[0]) if not main.empty else float(plot_df["external_macro_f1"].max())
    plot_df = plot_df[~plot_df["experiment"].astype(str).str.lower().isin(["main_model", "proposed"])]
    return plot_df.dropna(subset=["external_macro_f1", "val_macro_f1"]), main_score


def _clean_ablation_label(name: object) -> str:
    text = str(name)
    if text in ABLATION_LABELS:
        return ABLATION_LABELS[text]
    text = text.replace("_", " ")
    if text[:2].isdigit() and " " in text:
        text = text.split(" ", 1)[1]
    return text


def _set_score_xlim(ax, values) -> None:
    values = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").dropna()
    if values.empty:
        ax.set_xlim(0, 1.02)
        return
    low = max(0.0, float(values.min()) - 0.03)
    high = min(1.01, float(values.max()) + 0.03)
    if high - low < 0.08:
        center = (low + high) / 2
        low = max(0.0, center - 0.04)
        high = min(1.01, center + 0.04)
    ax.set_xlim(low, high)


def _hide_shared_y_axis(ax) -> None:
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    ax.spines["left"].set_visible(False)


def _set_xmargin(ax, values, pad: float) -> None:
    values = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").dropna()
    low = min(0.0, float(values.min())) - pad
    high = max(0.0, float(values.max())) + pad
    ax.set_xlim(low, high)


def _module_cmap():
    from matplotlib.colors import ListedColormap

    return ListedColormap(["#E5E7EB", MAIN_COLOR])


if __name__ == "__main__":
    main()
