from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .styles import CLASS_COLORS, SPLIT_COLORS, apply_publication_style, save_figure


MAIN_COLOR = "#4C78A8"
ALT_COLOR = "#72B7B2"
DROP_COLOR = "#E45756"
NEUTRAL = "#6B7280"
PALETTE = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2", "#9D755D", "#BAB0AC"]
EXCLUDED_BASELINE_MODELS = {"vit", "swintransformer"}
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


def make_all_figures(results_dir: str = "results", output_dir: str | None = None) -> List[str]:
    apply_publication_style()
    results_path = Path(results_dir)
    output_path = Path(output_dir) if output_dir else results_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    main_dir = _find_main_result_dir(results_path)
    if main_dir is not None and make_figure2(main_dir, output_path):
        written.append("figure2_main_performance")
    if make_figure3(results_path, output_path):
        written.append("figure3_comparison_ablation")
    if make_figure4(results_path, output_path):
        written.append("figure4_reliability_robustness")
    return written


def make_figure2(main_dir: Path, output_dir: Path) -> bool:
    val_df = _load_prediction_table(main_dir, "val")
    ext_df = _load_prediction_table(main_dir, "external")
    if val_df is None or ext_df is None:
        return False

    class_names = _class_names_from_predictions(val_df)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    _plot_confusion(axes[0, 0], val_df, class_names, "Internal test")
    _plot_confusion(axes[0, 1], ext_df, class_names, "External test")
    _plot_macro_split_bars(axes[0, 2], val_df, ext_df, "Overall performance")
    _plot_per_class_metric_lines(axes[1, 0], val_df, class_names, "Internal class metrics")
    internal_values = _prediction_metrics(val_df, class_names)[["sensitivity", "specificity", "precision", "f1"]].to_numpy()
    axes[1, 0].set_ylim(max(0.98, float(np.nanmin(internal_values)) - 0.002), 1.005)
    _plot_per_class_metric_lines(axes[1, 1], ext_df, class_names, "External class metrics")
    _plot_per_class_f1_compare(axes[1, 2], val_df, ext_df, class_names, "Class-wise F1")
    _label_panels(axes.ravel())
    fig.tight_layout(w_pad=1.5, h_pad=1.6)
    save_figure(fig, output_dir, "figure2_main_performance")
    plt.close(fig)
    return True


def make_figure3(results_dir: Path, output_dir: Path) -> bool:
    baseline = _safe_read_csv(results_dir / "baselines" / "baseline_summary_updated.csv")
    if baseline is None:
        baseline = _safe_read_csv(results_dir / "baselines" / "baseline_summary.csv")
    baseline = _filter_baseline_models(baseline)
    ablation = _safe_read_csv(results_dir / "ablations" / "ablation_summary.csv")
    main_dir = _find_main_result_dir(results_dir)
    history = _safe_read_csv((main_dir or Path("")) / "metrics_history.csv")

    if baseline is None and ablation is None and history is None:
        return False

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.34, wspace=0.35)
    ax_history = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_params = fig.add_subplot(gs[1, 1])
    bottom = gs[2, :].subgridspec(1, 4, width_ratios=[0.9, 1.15, 1.15, 1.35], wspace=0.12)
    ax_module = fig.add_subplot(bottom[0, 0])
    ax_perf = fig.add_subplot(bottom[0, 1], sharey=ax_module)
    ax_delta = fig.add_subplot(bottom[0, 2], sharey=ax_module)
    ax_shift = fig.add_subplot(bottom[0, 3], sharey=ax_module)

    _plot_history(ax_history, history, "Training trajectory")
    _plot_generalization_gap(ax_gap, baseline, "model_name", "Baseline generalization gap")
    _plot_metric_heatmap(
        ax_heatmap,
        baseline,
        "model_name",
        ["best_val_accuracy", "val_macro_f1", "external_accuracy", "external_macro_f1"],
        "Baseline metric matrix",
    )
    _plot_params_vs_performance(ax_params, baseline, ablation, "Parameters vs performance")
    _plot_ablation_composite_axes(ax_module, ax_perf, ax_delta, ax_shift, ablation)
    ax_module.text(
        0.0,
        1.03,
        "Ablation analysis",
        transform=ax_module.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="normal",
    )
    _label_panels([ax_history, ax_gap, ax_heatmap, ax_params, ax_module])
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.08, top=0.97, hspace=0.34, wspace=0.35)
    save_figure(fig, output_dir, "figure3_comparison_ablation")
    plt.close(fig)
    return True


def make_figure4(results_dir: Path, output_dir: Path) -> bool:
    robustness = _safe_read_csv(results_dir / "robustness" / "robustness_summary.csv")
    robustness_auc = _safe_read_csv(results_dir / "robustness" / "robustness_auc.csv")
    calibration_summary = _safe_read_csv(results_dir / "calibration" / "calibration_summary.csv")
    selected = _safe_read_csv(results_dir / "interpretability" / "selected_cases.csv")
    cases = _safe_read_csv(results_dir / "interpretability" / "gradcam_cases.csv")
    confidence = _safe_read_csv(results_dir / "calibration" / "confidence_cases.csv")
    selected_dir = results_dir / "interpretability" / "gradcam_selected"
    main_dir = _find_main_result_dir(results_dir)
    predictions = None if main_dir is None else _load_prediction_table(main_dir, "external")

    if all(x is None for x in [robustness, robustness_auc, calibration_summary, selected, cases, confidence]):
        return False

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    _plot_robustness_by_severity(axes[0, 0], robustness, "Perturbation response")
    _plot_robustness_auc(axes[0, 1], robustness_auc, "Robustness summary")
    _plot_calibration_bars(axes[0, 2], calibration_summary, "Calibration error")
    _plot_gradcam_group(axes[1, 0], selected if selected is not None else cases, True, "Correct predictions", selected_dir)
    _plot_gradcam_group(axes[1, 1], selected if selected is not None else cases, False, "Incorrect predictions", selected_dir)
    _plot_confidence_gap(axes[1, 2], confidence, predictions, "Confidence by outcome")
    _label_panels(axes.ravel())
    fig.tight_layout(w_pad=1.3, h_pad=1.5)
    save_figure(fig, output_dir, "figure4_reliability_robustness")
    plt.close(fig)
    return True


def make_figure5(results_dir: Path, output_dir: Path) -> bool:
    selected = _safe_read_csv(results_dir / "interpretability" / "selected_cases.csv")
    cases = _safe_read_csv(results_dir / "interpretability" / "gradcam_cases.csv")
    confidence = _safe_read_csv(results_dir / "calibration" / "confidence_cases.csv")
    selected_dir = results_dir / "interpretability" / "gradcam_selected"
    main_dir = _find_main_result_dir(results_dir)
    predictions = None if main_dir is None else _load_prediction_table(main_dir, "external")
    if selected is None and cases is None and predictions is None and confidence is None:
        return False

    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.35)
    ax_correct = fig.add_subplot(gs[0, 0])
    ax_incorrect = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[0, 2])

    _plot_gradcam_group(ax_correct, selected if selected is not None else cases, True, "Correct predictions", selected_dir)
    _plot_gradcam_group(ax_incorrect, selected if selected is not None else cases, False, "Incorrect predictions", selected_dir)
    _plot_confidence_gap(ax_conf, confidence, predictions, "Confidence by outcome")
    _label_panels([ax_correct, ax_incorrect, ax_conf])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.35)
    _shrink_axis_height(ax_correct, 0.70)
    _shrink_axis_height(ax_incorrect, 0.70)
    _shrink_axis_height(ax_conf, 0.70)
    save_figure(fig, output_dir, "figure5_interpretability_error")
    plt.close(fig)
    return True


def _find_main_result_dir(results_dir: Path) -> Optional[Path]:
    candidates = [
        results_dir / "main",
        results_dir / "cached_train" / "checkpoints",
        results_dir / "cached_train",
        results_dir / "checkpoints",
    ]
    for candidate in candidates:
        if candidate.exists() and (
            (candidate / "metrics_summary.json").exists()
            or list(candidate.glob("val_epoch*.xlsx"))
            or (candidate / "predictions_val.xlsx").exists()
        ):
            return candidate
    return None


def _load_prediction_table(result_dir: Path, split: str) -> Optional[pd.DataFrame]:
    preferred = result_dir / f"predictions_{split}.xlsx"
    if preferred.exists():
        return pd.read_excel(preferred)
    pred_dir = result_dir / "predictions"
    epoch_files = sorted(pred_dir.glob(f"{split}_epoch*.xlsx"), key=_epoch_sort_key) if pred_dir.exists() else []
    if not epoch_files:
        epoch_files = sorted(result_dir.glob(f"{split}_epoch*.xlsx"), key=_epoch_sort_key)
    if not epoch_files:
        return None
    return pd.read_excel(epoch_files[-1])


def _class_names_from_predictions(df: pd.DataFrame) -> List[str]:
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if prob_cols:
        return [col.replace("prob_", "") for col in prob_cols]
    if "label_name" in df.columns:
        names = df[["label", "label_name"]].drop_duplicates().sort_values("label")
        return names["label_name"].astype(str).tolist()
    return [str(x) for x in sorted(df["label"].unique())]


def _prediction_metrics(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    labels = df["label"].to_numpy()
    preds = df["pred"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    specificity = _specificity_per_class(labels, preds, len(class_names))
    return pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support,
        }
    )


def _specificity_per_class(labels, preds, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    total = cm.sum()
    out = []
    for idx in range(n_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        out.append(0.0 if tn + fp == 0 else tn / (tn + fp))
    return np.asarray(out)


def _plot_confusion(ax, df: pd.DataFrame, class_names: List[str], title: str) -> None:
    cm = confusion_matrix(df["label"], df["pred"], labels=list(range(len(class_names))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "#111111"
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=color)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_macro_split_bars(ax, val_df: pd.DataFrame, ext_df: pd.DataFrame, title: str) -> None:
    class_names = _class_names_from_predictions(val_df)
    rows = []
    for split, df in [("Internal test", val_df), ("External test", ext_df)]:
        metrics = _prediction_metrics(df, class_names)
        rows.extend(
            [
                {"split": split, "metric": "Accuracy", "value": float((df["label"] == df["pred"]).mean())},
                {"split": split, "metric": "Macro-F1", "value": float(metrics["f1"].mean())},
                {"split": split, "metric": "Mean sensitivity", "value": float(metrics["sensitivity"].mean())},
                {"split": split, "metric": "Mean specificity", "value": float(metrics["specificity"].mean())},
            ]
        )
    plot_df = pd.DataFrame(rows)
    metrics_order = ["Accuracy", "Macro-F1", "Mean sensitivity", "Mean specificity"]
    x = np.arange(len(metrics_order))
    width = 0.36
    for offset, split, color in [(-width / 2, "Internal test", SPLIT_COLORS["val"]), (width / 2, "External test", SPLIT_COLORS["external"])]:
        vals = [plot_df[(plot_df["split"] == split) & (plot_df["metric"] == m)]["value"].iloc[0] for m in metrics_order]
        ax.bar(x + offset, vals, width=width, label=split, color=color)
    ax.set_title(title)
    ax.set_xticks(x, metrics_order, rotation=35, ha="right")
    _set_zoomed_unit_ylim(ax, plot_df["value"], upper=1.01, top_room=0.18)
    ax.set_ylabel("Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True, framealpha=0.9)


def _plot_per_class_metric_lines(ax, df: pd.DataFrame, class_names: list[str], title: str) -> None:
    metrics = _prediction_metrics(df, class_names)
    metric_cols = ["sensitivity", "specificity", "precision", "f1"]
    x = np.arange(len(class_names))
    offsets = np.linspace(-0.06, 0.06, len(metric_cols))
    for idx, metric in enumerate(metric_cols):
        ax.plot(
            x + offsets[idx],
            metrics[metric],
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=metric.capitalize(),
            color=PALETTE[idx],
        )
    ax.set_title(title)
    ax.set_xticks(x, class_names)
    _set_zoomed_unit_ylim(ax, metrics[metric_cols].to_numpy().ravel())
    ax.set_ylabel("Score")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.2,
        columnspacing=0.9,
    )


def _plot_per_class_f1_compare(ax, val_df: pd.DataFrame, ext_df: pd.DataFrame, class_names: list[str], title: str) -> None:
    val_metrics = _prediction_metrics(val_df, class_names)
    ext_metrics = _prediction_metrics(ext_df, class_names)
    x = np.arange(len(class_names))
    width = 0.36
    ax.bar(x - width / 2, val_metrics["f1"], width, color=SPLIT_COLORS["val"], label="Internal test")
    ax.bar(x + width / 2, ext_metrics["f1"], width, color=SPLIT_COLORS["external"], label="External test")
    ax.set_title(title)
    ax.set_xticks(x, class_names)
    _set_zoomed_unit_ylim(ax, np.r_[val_metrics["f1"].to_numpy(), ext_metrics["f1"].to_numpy()], upper=1.01, top_room=0.18)
    ax.set_ylabel("F1-score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True, framealpha=0.9)


def _set_zoomed_unit_ylim(ax, values, pad: float = 0.015, min_span: float = 0.04, upper: float | None = None, top_room: float = 0.08) -> None:
    values = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").dropna()
    if values.empty:
        ax.set_ylim(0, 1.02)
        return
    low = float(values.min())
    high = float(values.max())
    if upper is not None:
        y1 = upper
        data_span = max(y1 - low, min_span)
        y0 = max(0.0, low - pad - data_span * top_room)
        if high > y1:
            y1 = min(1.02, high + pad)
        ax.set_ylim(y0, y1)
        return
    span = max(high - low, min_span)
    center = (high + low) / 2
    y0 = max(0.0, center - span / 2 - pad)
    y1 = min(1.02, center + span / 2 + pad)
    if y1 - y0 < min_span:
        y0 = max(0.0, y1 - min_span)
    ax.set_ylim(y0, y1)


def _plot_history(ax, history: Optional[pd.DataFrame], title: str) -> None:
    if history is None:
        _empty_panel(ax, "No training history")
        return
    for col, label, color in [
        ("train_acc", "Train", SPLIT_COLORS["train"]),
        ("val_acc", "Internal test", SPLIT_COLORS["val"]),
        ("external_acc", "External test", SPLIT_COLORS["external"]),
    ]:
        if col in history:
            ax.plot(history["epoch"], history[col], label=label, color=color, linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend()


def _plot_ranked_metric(ax, df: Optional[pd.DataFrame], name_col: str, metric: str, title: str) -> None:
    if df is None or name_col not in df or metric not in df:
        _empty_panel(ax, "Result file pending")
        return
    plot_df = df[[name_col, metric]].dropna().copy()
    if plot_df.empty:
        _empty_panel(ax, "Metric pending")
        return
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna().sort_values(metric, ascending=True).tail(12)
    colors = [MAIN_COLOR if str(name).lower() in {"proposed", "main_model"} else ALT_COLOR for name in plot_df[name_col]]
    ax.barh(plot_df[name_col].astype(str), plot_df[metric], color=colors)
    ax.set_title(title)
    ax.set_xlabel(metric)
    xmin = 0.4 if metric == "external_macro_f1" else 0
    ax.set_xlim(xmin, 1.02)


def _plot_generalization_gap(ax, df: Optional[pd.DataFrame], name_col: str, title: str) -> None:
    if df is None or name_col not in df:
        _empty_panel(ax, "Gap data pending")
        return
    if {"val_macro_f1", "external_macro_f1"}.issubset(df.columns):
        internal_col = "val_macro_f1"
        external_col = "external_macro_f1"
        xlabel = "Macro-F1"
    elif {"best_val_accuracy", "external_accuracy"}.issubset(df.columns):
        internal_col = "best_val_accuracy"
        external_col = "external_accuracy"
        xlabel = "Accuracy"
    else:
        _empty_panel(ax, "Gap data pending")
        return
    plot_df = df[[name_col, internal_col, external_col]].dropna().copy()
    plot_df[internal_col] = pd.to_numeric(plot_df[internal_col], errors="coerce")
    plot_df[external_col] = pd.to_numeric(plot_df[external_col], errors="coerce")
    plot_df["gap"] = plot_df[internal_col] - plot_df[external_col]
    plot_df = plot_df.dropna().sort_values("gap", ascending=True).tail(12)
    y = np.arange(len(plot_df))
    names = plot_df[name_col].astype(str).to_numpy()
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        name = str(getattr(row, name_col))
        internal = float(getattr(row, internal_col))
        external = float(getattr(row, external_col))
        line_color = MAIN_COLOR if name.lower() in {"proposed", "main_model"} else "#9CA3AF"
        ax.plot([external, internal], [idx, idx], color=line_color, linewidth=1.1, zorder=1)
        ax.annotate(
            "",
            xy=(external, idx),
            xytext=(internal, idx),
            arrowprops=dict(arrowstyle="->", color=line_color, lw=1.0, shrinkA=4, shrinkB=4),
            zorder=2,
        )
    ax.scatter(plot_df[internal_col], y, s=22, color=SPLIT_COLORS["val"], label="Internal test", zorder=3)
    ax.scatter(plot_df[external_col], y, s=22, color=SPLIT_COLORS["external"], label="External test", zorder=3)
    ax.set_yticks(y, names)
    values = pd.to_numeric(pd.Series(plot_df[[internal_col, external_col]].to_numpy().ravel()), errors="coerce").dropna()
    if not values.empty:
        xmin = max(0.0, float(values.min()) - 0.03)
        xmax = min(1.01, float(values.max()) + 0.03)
        if xmax - xmin < 0.08:
            center = (xmax + xmin) / 2
            xmin = max(0.0, center - 0.04)
            xmax = min(1.01, center + 0.04)
        ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(plot_df) - 0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)


def _plot_metric_heatmap(ax, df: Optional[pd.DataFrame], name_col: str, metrics: list[str], title: str) -> None:
    if df is None or name_col not in df:
        _empty_panel(ax, "Result file pending")
        return
    available = [m for m in metrics if m in df.columns]
    if not available:
        _empty_panel(ax, "Metric matrix pending")
        return
    plot_df = df[[name_col, *available]].copy()
    for col in available:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=available, how="all").tail(10)
    values = plot_df[available].to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_yticks(range(len(plot_df)), plot_df[name_col].astype(str))
    metric_labels = {
        "best_val_accuracy": "Int. Acc",
        "val_macro_f1": "Int. F1",
        "external_accuracy": "Ext. Acc",
        "external_macro_f1": "Ext. F1",
    }
    ax.set_xticks(range(len(available)), [metric_labels.get(m, m.replace("_", " ")) for m in available], rotation=0, ha="center")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] >= 0.72 else "#111111"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5, color=text_color)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_ablation_drop(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or "experiment" not in df or "external_macro_f1" not in df:
        _empty_panel(ax, "Ablation data pending")
        return
    plot_df = df[["experiment", "external_macro_f1"]].dropna().copy()
    plot_df["external_macro_f1"] = pd.to_numeric(plot_df["external_macro_f1"], errors="coerce")
    main = plot_df[plot_df["experiment"].astype(str).str.lower().isin(["main_model", "proposed"])]
    if main.empty:
        ref = float(plot_df["external_macro_f1"].max())
    else:
        ref = float(main["external_macro_f1"].iloc[0])
    plot_df = plot_df[~plot_df["experiment"].astype(str).str.lower().isin(["main_model", "proposed"])]
    plot_df["drop"] = ref - plot_df["external_macro_f1"]
    plot_df = plot_df.sort_values("drop", ascending=True)
    ax.barh(plot_df["experiment"].astype(str), plot_df["drop"], color=DROP_COLOR)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Main - ablation macro-F1")


def _plot_params_vs_performance(ax, baseline: Optional[pd.DataFrame], ablation: Optional[pd.DataFrame], title: str) -> None:
    frames = []
    for group, df, name_col in [("Baseline", baseline, "model_name"), ("Ablation", ablation, "experiment")]:
        if df is not None and {name_col, "model_params_M", "external_macro_f1"}.issubset(df.columns):
            tmp = df[[name_col, "model_params_M", "external_macro_f1"]].copy()
            tmp["group"] = group
            tmp = tmp.rename(columns={name_col: "name"})
            frames.append(tmp)
    if not frames:
        _empty_panel(ax, "Parameter data pending")
        return
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["model_params_M"] = pd.to_numeric(plot_df["model_params_M"], errors="coerce")
    plot_df["external_macro_f1"] = pd.to_numeric(plot_df["external_macro_f1"], errors="coerce")
    plot_df = plot_df.dropna()
    for group, color in [("Baseline", ALT_COLOR), ("Ablation", "#F58518")]:
        group_df = plot_df[plot_df["group"] == group]
        if not group_df.empty:
            ax.scatter(group_df["model_params_M"], group_df["external_macro_f1"], s=22, color=color, alpha=0.85, label=group)
    main_df = plot_df[plot_df["name"].astype(str).str.lower().isin(["proposed", "main_model"])]
    if not main_df.empty:
        ax.scatter(
            main_df["model_params_M"],
            main_df["external_macro_f1"],
            s=34,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.9,
            label="Full model",
            zorder=4,
        )
    ax.set_title(title)
    ax.set_xlabel("Parameters, M")
    ax.set_ylabel("External macro-F1")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)


def _plot_ablation_composite_axes(ax_module, ax_perf, ax_delta, ax_shift, df: Optional[pd.DataFrame]) -> None:
    if df is None or not {"experiment", "val_macro_f1", "external_macro_f1"}.issubset(df.columns):
        _empty_panel(ax_module, "Ablation data pending")
        for ax in [ax_perf, ax_delta, ax_shift]:
            ax.axis("off")
        return

    plot_df = df.copy()
    plot_df["label"] = plot_df["experiment"].map(_clean_ablation_label)
    for col in ["external_macro_f1", "val_macro_f1"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    main = plot_df[plot_df["experiment"].astype(str).str.lower().isin(["main_model", "proposed"])]
    main_external = float(main["external_macro_f1"].iloc[0]) if not main.empty else float(plot_df["external_macro_f1"].max())
    main_internal = float(main["val_macro_f1"].iloc[0]) if not main.empty else float(plot_df["val_macro_f1"].max())
    plot_df["external_delta"] = main_external - plot_df["external_macro_f1"]
    plot_df["internal_delta"] = main_internal - plot_df["val_macro_f1"]
    plot_df = plot_df.dropna(subset=["external_macro_f1", "val_macro_f1", "external_delta", "internal_delta"])
    plot_df = plot_df.sort_values("external_macro_f1", ascending=True)
    if plot_df.empty:
        _empty_panel(ax_module, "Ablation data pending")
        for ax in [ax_perf, ax_delta, ax_shift]:
            ax.axis("off")
        return

    y = np.arange(len(plot_df))
    module_cols = [(col, label) for col, label in MODULE_COLUMNS if col in plot_df.columns]
    if module_cols:
        matrix = plot_df[[col for col, _ in module_cols]].astype(bool).to_numpy(dtype=int)
        ax_module.imshow(matrix, aspect="auto", cmap=ListedColormap(["#E5E7EB", MAIN_COLOR]), vmin=0, vmax=1)
        ax_module.set_xticks(np.arange(len(module_cols)), [label for _, label in module_cols], rotation=35, ha="right")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax_module.text(j, i, "Y" if matrix[i, j] else "N", ha="center", va="center", fontsize=5.2, color="white" if matrix[i, j] else "#111111")
    else:
        _empty_panel(ax_module, "Module data pending")
    ax_module.set_yticks(y, plot_df["label"])
    ax_module.tick_params(axis="y", length=0)

    offset = 0.12
    for values, y_offset, color, label in [
        (plot_df["val_macro_f1"], offset, SPLIT_COLORS["val"], "Internal test"),
        (plot_df["external_macro_f1"], -offset, SPLIT_COLORS["external"], "External test"),
    ]:
        ax_perf.plot(values, y + y_offset, color="#CBD5E1", linewidth=0.8, zorder=1)
        for idx, value in enumerate(values):
            ax_perf.hlines(idx + y_offset, value - 0.010, value + 0.010, color=color, linewidth=2.4, zorder=2)
            ax_perf.scatter(value, idx + y_offset, s=13, color=color, edgecolor="white", linewidth=0.5, zorder=3, label=label if idx == 0 else None)
            ax_perf.text(value + 0.018, idx + y_offset, f"{value:.3f}", va="center", ha="left", fontsize=5.0, zorder=10, clip_on=False)
    ax_perf.set_xlabel("Macro-F1")
    _hide_shared_y_axis(ax_perf)
    _set_score_xlim(ax_perf, plot_df[["val_macro_f1", "external_macro_f1"]].to_numpy().ravel())
    left, right = ax_perf.get_xlim()
    ax_perf.set_xlim(left, min(1.04, right + 0.025))

    ax_delta.axvline(0, color="#333333", linewidth=0.8)
    for values, y_offset, color, label in [
        (plot_df["internal_delta"], offset, SPLIT_COLORS["val"], "Internal test"),
        (plot_df["external_delta"], -offset, SPLIT_COLORS["external"], "External test"),
    ]:
        for idx, value in enumerate(values):
            ax_delta.hlines(idx + y_offset, 0, value, color=color, linewidth=1.5, zorder=1)
            ax_delta.scatter(value, idx + y_offset, s=13, color=color, edgecolor="white", linewidth=0.5, zorder=2, label=label if idx == 0 else None)
            ax_delta.text(value + 0.004, idx + y_offset, f"{value:.3f}", va="center", ha="left", fontsize=5.0, zorder=10, clip_on=False)
    ax_delta.set_xlabel("Delta macro-F1\nvs full model")
    _hide_shared_y_axis(ax_delta)
    _set_xmargin(ax_delta, np.r_[plot_df["internal_delta"].to_numpy(), plot_df["external_delta"].to_numpy(), 0], pad=0.02)

    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax_shift.plot([row.external_macro_f1, row.val_macro_f1], [idx, idx], color="#9CA3AF", linewidth=1.0, zorder=1)
    ax_shift.scatter(plot_df["val_macro_f1"], y, s=16, color=SPLIT_COLORS["val"], label="Internal test", zorder=2)
    ax_shift.scatter(plot_df["external_macro_f1"], y, s=16, color=SPLIT_COLORS["external"], label="External test", zorder=2)
    ax_shift.set_xlabel("Macro-F1")
    _hide_shared_y_axis(ax_shift)
    ax_shift.legend(loc="upper left", frameon=True, framealpha=0.9)
    _set_score_xlim(ax_shift, plot_df[["val_macro_f1", "external_macro_f1"]].to_numpy().ravel())

    for ax in [ax_module, ax_perf, ax_delta, ax_shift]:
        ax.set_ylim(-0.6, len(plot_df) - 0.4)


def _plot_robustness_by_severity(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or not {"perturbation", "severity", "macro_f1"}.issubset(df.columns):
        _empty_panel(ax, "Robustness data pending")
        return
    use = df[df["perturbation"].isin(["brightness", "contrast", "blur", "downsample", "rotation"])].copy()
    if use.empty:
        _empty_panel(ax, "Robustness data pending")
        return
    clean = df[df["perturbation"] == "clean"]
    clean_score = float(clean["macro_f1"].iloc[0]) if not clean.empty else np.nan
    for idx, (name, group) in enumerate(use.groupby("perturbation")):
        group = group.copy()
        group["severity"] = pd.to_numeric(group["severity"], errors="coerce")
        group = group.dropna(subset=["severity", "macro_f1"])
        if "level" in group.columns:
            group["level"] = pd.to_numeric(group["level"], errors="coerce")
        else:
            group["level"] = _perturbation_level(name, group["severity"])
        group = group.dropna(subset=["level"])
        group = group.groupby("level", as_index=False)["macro_f1"].mean().sort_values("level")
        if not np.isnan(clean_score):
            group = pd.concat([pd.DataFrame({"level": [0], "macro_f1": [clean_score]}), group], ignore_index=True)
        ax.plot(group["level"], group["macro_f1"], marker="o", linewidth=1.1, markersize=3, color=PALETTE[idx % len(PALETTE)], label=name)
    if not np.isnan(clean_score):
        ax.axhline(clean_score, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Perturbation level")
    ax.set_ylabel("External macro-F1")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Clean", "Mild", "Moderate", "Severe"])
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2)


def _perturbation_level(name: str, severity: pd.Series) -> pd.Series:
    severity = pd.to_numeric(severity, errors="coerce")
    if name in {"brightness", "contrast"}:
        strength = (severity - 1.0).abs().round(6)
    elif name in {"rotation", "translate"}:
        strength = severity.abs().round(6)
    elif name in {"downsample", "jpeg"}:
        strength = (-severity).round(6)
    else:
        strength = severity.round(6)
    levels = {value: idx + 1 for idx, value in enumerate(sorted(strength.dropna().unique()))}
    return strength.map(levels)


def _plot_robustness_auc(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or "perturbation" not in df or "macro_f1_robustness_auc" not in df:
        _empty_panel(ax, "Robustness AUC pending")
        return
    plot_df = df.sort_values("macro_f1_robustness_auc", ascending=True)
    ax.barh(plot_df["perturbation"], plot_df["macro_f1_robustness_auc"], color=ALT_COLOR)
    ax.set_title(title)
    ax.set_xlabel("Mean robustness macro-F1")
    ax.set_xlim(0, 1.02)


def _plot_calibration_bars(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or not {"split", "ece", "brier_score"}.issubset(df.columns):
        _empty_panel(ax, "Calibration summary pending")
        return
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["ece"], width, color=MAIN_COLOR, label="ECE")
    ax.bar(x + width / 2, df["brier_score"], width, color=DROP_COLOR, label="Brier")
    ax.set_title(title)
    ax.set_xticks(x, df["split"].astype(str))
    ax.set_ylabel("Error")
    ax.legend()


def _plot_reliability_diagram(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or not {"split", "mean_confidence", "accuracy"}.issubset(df.columns):
        _empty_panel(ax, "Reliability data pending")
        return
    for idx, (split, group) in enumerate(df.groupby("split")):
        group = group.dropna(subset=["mean_confidence", "accuracy"])
        if group.empty:
            continue
        group = group.sort_values("mean_confidence")
        sizes = _calibration_marker_sizes(group)
        ax.plot(group["mean_confidence"], group["accuracy"], linewidth=0.9, color=PALETTE[idx], alpha=0.75)
        ax.scatter(
            group["mean_confidence"],
            group["accuracy"],
            s=sizes,
            color=PALETTE[idx],
            edgecolor="white",
            linewidth=0.4,
            label=split,
            alpha=0.9,
            zorder=3,
        )
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()


def _calibration_marker_sizes(group: pd.DataFrame) -> np.ndarray:
    if "count" not in group.columns:
        return np.full(len(group), 18.0)
    counts = pd.to_numeric(group["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    if counts.size == 0 or counts.max() <= 0:
        return np.full(len(group), 18.0)
    scaled = np.sqrt(counts / counts.max())
    return 12.0 + 42.0 * scaled


def _plot_coverage_accuracy(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or not {"split", "coverage", "accuracy"}.issubset(df.columns):
        _empty_panel(ax, "Coverage data pending")
        return
    for idx, (split, group) in enumerate(df.groupby("split")):
        group = group.dropna(subset=["coverage", "accuracy"]).sort_values("coverage")
        ax.plot(group["coverage"], group["accuracy"], marker="o", linewidth=1.1, markersize=2.5, label=split, color=PALETTE[idx])
    ax.set_title(title)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()


def _plot_ci_intervals(ax, df: Optional[pd.DataFrame], title: str) -> None:
    if df is None or not {"split", "metric", "estimate", "ci_low", "ci_high"}.issubset(df.columns):
        _empty_panel(ax, "CI data pending")
        return
    plot_df = df[(df["split"] == "external") & (df["class_name"] == "overall")].copy()
    if plot_df.empty:
        plot_df = df[df["class_name"] == "overall"].copy()
    plot_df = plot_df.sort_values("estimate")
    y = np.arange(len(plot_df))
    x = plot_df["estimate"].astype(float)
    xerr = np.vstack([x - plot_df["ci_low"].astype(float), plot_df["ci_high"].astype(float) - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", color=MAIN_COLOR, ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.set_yticks(y, plot_df["metric"].astype(str))
    ax.set_title(title)
    ax.set_xlabel("Estimate with 95% CI")
    ax.set_xlim(0, 1.02)


def _plot_gradcam_images(axes: list, cases: Optional[pd.DataFrame]) -> None:
    if cases is None or "cam_path" not in cases:
        for ax in axes:
            _empty_panel(ax, "Grad-CAM pending")
        return
    plot_cases = cases.head(len(axes)).copy()
    for ax, (_, row) in zip(axes, plot_cases.iterrows()):
        path = Path(str(row["cam_path"]))
        if path.exists():
            img = mpimg.imread(path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, path.name, ha="center", va="center")
        ax.set_title(f"{row.get('label_name', '')} -> {row.get('pred_name', '')}")
        ax.axis("off")
    for ax in axes[len(plot_cases):]:
        ax.axis("off")


def _plot_gradcam_group(ax, cases: Optional[pd.DataFrame], correct: bool, title: str, selected_dir: Path | None = None) -> None:
    ax.axis("off")
    ax.set_title(title)
    selected_files = _gradcam_selected_files(selected_dir, correct)
    if selected_files:
        group = pd.DataFrame(selected_files)
    else:
        group = None
    if cases is None or "cam_path" not in cases or "correct" not in cases:
        if group is None:
            ax.text(0.5, 0.5, "Grad-CAM pending", ha="center", va="center")
            return
    if group is None:
        group = cases[cases["correct"].astype(str).str.lower().isin(["true", "1"]) == correct].head(3).copy()
    if group.empty:
        ax.text(0.5, 0.5, "No cases", ha="center", va="center")
        return

    n = len(group)
    gap = 0.010
    height = (0.98 - gap * (n - 1)) / n
    for idx, (_, row) in enumerate(group.iterrows()):
        y0 = 0.0 + (n - 1 - idx) * (height + gap)
        child = ax.inset_axes([0.02, y0, 0.96, height])
        path = Path(str(row["cam_path"]))
        if path.exists():
            child.imshow(mpimg.imread(path))
        else:
            child.text(0.5, 0.5, path.name, ha="center", va="center", fontsize=5)
        child.set_title(f"{row.get('label_name', '')} -> {row.get('pred_name', '')}", fontsize=6, pad=1.5)
        child.axis("off")


def _gradcam_selected_files(selected_dir: Path | None, correct: bool) -> list[dict[str, object]]:
    if selected_dir is None or not selected_dir.exists():
        return []
    rows = []
    for path in sorted(selected_dir.glob("*.png")):
        label, pred = _parse_gradcam_filename(path.name)
        is_correct = bool(label and pred and label == pred)
        if is_correct != correct:
            continue
        rows.append({"cam_path": str(path), "label_name": label, "pred_name": pred, "correct": is_correct})
    return rows


def _parse_gradcam_filename(name: str) -> tuple[str, str]:
    stem = Path(name).stem
    if "_to_" not in stem:
        return "", ""
    left, pred = stem.rsplit("_to_", 1)
    label = left.split("_")[-1]
    return label, pred


def _plot_error_transition(ax, df: pd.DataFrame, title: str) -> None:
    errors = df[df["label"] != df["pred"]].copy()
    if errors.empty:
        _empty_panel(ax, "No errors")
        return
    class_names = _class_names_from_predictions(df)
    cm = confusion_matrix(errors["label"], errors["pred"], labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)
    im = ax.imshow(cm, cmap="Reds")
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_confidence_gap(ax, confidence: Optional[pd.DataFrame], predictions: Optional[pd.DataFrame], title: str) -> None:
    if confidence is None:
        if predictions is None:
            _empty_panel(ax, "Confidence data pending")
            return
        prob_cols = [col for col in predictions.columns if col.startswith("prob_")]
        confidence = predictions.copy()
        confidence["confidence"] = confidence[prob_cols].max(axis=1)
        confidence["correct"] = confidence["label"] == confidence["pred"]
        confidence["split"] = "external"
    ext = confidence[confidence.get("split", "external") == "external"].copy()
    if ext.empty:
        ext = confidence.copy()
    values = [ext.loc[ext["correct"].astype(bool), "confidence"], ext.loc[~ext["correct"].astype(bool), "confidence"]]
    bp = ax.boxplot(values, labels=["Correct", "Incorrect"], patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], [MAIN_COLOR, DROP_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_title(title)
    ax.set_ylabel("Maximum probability")
    ax.set_ylim(0.6, 1.02)


def _plot_error_pairs(ax, cases: pd.DataFrame, title: str) -> None:
    if not {"label_name", "pred_name", "correct"}.issubset(cases.columns):
        _empty_panel(ax, "Case data pending")
        return
    errors = cases[~cases["correct"].astype(bool)].copy()
    if errors.empty:
        _empty_panel(ax, "No selected errors")
        return
    errors["pair"] = errors["label_name"].astype(str) + " -> " + errors["pred_name"].astype(str)
    counts = errors["pair"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color=DROP_COLOR)
    ax.set_title(title)
    ax.set_xlabel("Selected cases")


def _plot_case_table(ax, cases: Optional[pd.DataFrame], title: str) -> None:
    ax.axis("off")
    if cases is None or not {"label_name", "pred_name", "correct"}.issubset(cases.columns):
        ax.text(0.5, 0.5, "Selected cases pending", ha="center", va="center")
        return
    use = cases.head(8).copy()
    text = []
    for _, row in use.iterrows():
        mark = "OK" if bool(row["correct"]) else "Err"
        text.append(f"{mark}: {row['label_name']} -> {row['pred_name']}")
    ax.set_title(title)
    ax.text(0.02, 0.98, "\n".join(text), ha="left", va="top", fontsize=6, transform=ax.transAxes)


def _clean_ablation_label(name: object) -> str:
    text = str(name)
    if text in ABLATION_LABELS:
        return ABLATION_LABELS[text]
    text = text.replace("_", " ")
    if text[:2].isdigit() and " " in text:
        text = text.split(" ", 1)[1]
    return text


def _hide_shared_y_axis(ax) -> None:
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    ax.spines["left"].set_visible(False)


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


def _set_xmargin(ax, values, pad: float) -> None:
    values = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").dropna()
    if values.empty:
        ax.set_xlim(-pad, pad)
        return
    low = min(0.0, float(values.min())) - pad
    high = max(0.0, float(values.max())) + pad
    ax.set_xlim(low, high)


def _shrink_axis_height(ax, scale: float) -> None:
    pos = ax.get_position()
    new_height = pos.height * scale
    y0 = pos.y0 + (pos.height - new_height) / 2
    ax.set_position([pos.x0, y0, pos.width, new_height])


def _empty_panel(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _filter_baseline_models(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or "model_name" not in df:
        return df
    keep = ~df["model_name"].astype(str).str.lower().isin(EXCLUDED_BASELINE_MODELS)
    return df.loc[keep].copy()


def _epoch_sort_key(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _label_panels(axes: Iterable) -> None:
    for idx, ax in enumerate(axes):
        ax.text(-0.12, 1.08, chr(ord("a") + idx), transform=ax.transAxes, fontweight="bold", fontsize=9)
