from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


EXCLUDED_BASELINE_MODELS = {"vit", "swintransformer"}
SPLIT_LABELS = {"val": "Internal test", "external": "External test"}
ABLATION_LABELS = {
    "Main_Model": "Full model",
    "2_No_Time_Branch": "w/o temporal branch",
    "3_No_Spatial_Branch": "w/o spatial branch",
    "4_No_IDCNN": "w/o IDCNN block",
    "5_No_Dynamic_Fusion": "w/o dynamic fusion",
    "6_No_MoE": "w/o MoE module",
    "7_No_Fourier_Block": "w/o Fourier block",
}


def make_all_tables(results_dir: str = "results", output_dir: str | None = None) -> List[str]:
    results_path = Path(results_dir)
    output_path = Path(output_dir) if output_dir else results_path / "tables"
    output_path.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    table1 = make_table1_dataset(results_path)
    _write_table(table1, output_path / "table1_dataset_characteristics")
    written.append("table1_dataset_characteristics")

    table2 = make_table2_main_performance(results_path)
    if table2 is not None:
        _write_table(table2, output_path / "table2_main_performance")
        written.append("table2_main_performance")

    table3 = make_table3_comparison_ablation(results_path)
    if table3 is not None:
        _write_table(table3, output_path / "table3_comparison_ablation")
        written.append("table3_comparison_ablation")
    return written


def make_table1_dataset(results_dir: Path) -> pd.DataFrame:
    rows = []
    for split, root, purpose in [
        ("BuildModel", Path("Dataset/BuildModel"), "model development"),
        ("ExternalTest", Path("Dataset/ExternalTest"), "external testing"),
    ]:
        counts = _class_counts(root)
        rows.append(
            {
                "split": split,
                "source": str(root),
                "AF": counts.get("AF", 0),
                "N": counts.get("N", 0),
                "PAT": counts.get("PAT", 0),
                "total": sum(counts.values()),
                "image_size": "1064 x 245 after white-padding resize",
                "preprocessing": "RGB conversion, aspect-ratio preserving white padding, normalization",
                "purpose": purpose,
            }
        )
    return pd.DataFrame(rows)


def make_table2_main_performance(results_dir: Path) -> Optional[pd.DataFrame]:
    main_dir = _find_main_result_dir(results_dir)
    if main_dir is None:
        return None
    rows = []
    for split in ["val", "external"]:
        df = _load_prediction_table(main_dir, split)
        if df is None:
            continue
        rows.extend(_performance_rows(df, split))
    return pd.DataFrame(rows) if rows else None


def make_table3_comparison_ablation(results_dir: Path) -> Optional[pd.DataFrame]:
    frames = []
    baseline = _safe_read_csv(results_dir / "baselines" / "baseline_summary_updated.csv")
    if baseline is None:
        baseline = _safe_read_csv(results_dir / "baselines" / "baseline_summary.csv")
    if baseline is not None:
        baseline = baseline.copy()
        baseline = baseline[~baseline["model_name"].astype(str).str.lower().isin(EXCLUDED_BASELINE_MODELS)]
        baseline["experiment_group"] = "baseline"
        baseline["model_or_variant"] = baseline["model_name"].map(_model_display_name)
        baseline["key_change"] = baseline["model_name"].map(
            lambda name: "proposed model" if str(name).lower() == "proposed" else "alternative model architecture"
        )
        frames.append(baseline)

    ablation = _safe_read_csv(results_dir / "ablations" / "ablation_summary.csv")
    if ablation is not None:
        ablation = ablation.copy()
        ablation["experiment_group"] = "ablation"
        ablation["model_or_variant"] = ablation["experiment"].map(_ablation_display_name)
        ablation["key_change"] = ablation.apply(_ablation_change, axis=1)
        frames.append(ablation)

    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True, sort=False)
    cols = [
        "experiment_group",
        "model_or_variant",
        "key_change",
        "model_params_M",
        "internal_accuracy",
        "internal_macro_f1",
        "external_accuracy",
        "external_macro_f1",
        "external_auroc_macro",
    ]
    if "internal_accuracy" not in merged and "best_val_accuracy" in merged:
        merged["internal_accuracy"] = merged["best_val_accuracy"]
    if "internal_macro_f1" not in merged and "val_macro_f1" in merged:
        merged["internal_macro_f1"] = merged["val_macro_f1"]
    for col in cols:
        if col not in merged:
            merged[col] = None
    return merged[cols]


def _performance_rows(df: pd.DataFrame, split: str) -> List[Dict[str, object]]:
    class_names = _class_names_from_predictions(df)
    labels = df["label"].to_numpy()
    preds = df["pred"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    rows = []
    for idx, name in enumerate(class_names):
        rows.append(
            {
                "dataset": split,
                "class_or_average": name,
                "accuracy": accuracy_score(labels, preds),
                "sensitivity": recall[idx],
                "specificity": _specificity(labels, preds, idx),
                "precision": precision[idx],
                "f1_score": f1[idx],
                "support": int(support[idx]),
            }
        )
    rows.append(
        {
            "dataset": SPLIT_LABELS.get(split, split),
            "class_or_average": "macro average",
            "accuracy": accuracy_score(labels, preds),
            "sensitivity": float(pd.Series(recall).mean()),
            "specificity": float(pd.Series([_specificity(labels, preds, idx) for idx in range(len(class_names))]).mean()),
            "precision": float(pd.Series(precision).mean()),
            "f1_score": f1_score(labels, preds, average="macro", zero_division=0),
            "support": len(labels),
        }
    )
    for row in rows[:-1]:
        row["dataset"] = SPLIT_LABELS.get(split, split)
    return rows


def _specificity(labels, preds, class_idx: int) -> float:
    labels = pd.Series(labels)
    preds = pd.Series(preds)
    tn = ((labels != class_idx) & (preds != class_idx)).sum()
    fp = ((labels != class_idx) & (preds == class_idx)).sum()
    return 0.0 if tn + fp == 0 else float(tn / (tn + fp))


def _write_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{stem}.csv", index=False)
    df.to_excel(f"{stem}.xlsx", index=False)


def _class_counts(root: Path) -> Dict[str, int]:
    counts = {}
    for cls_dir in sorted(root.iterdir()) if root.exists() else []:
        if cls_dir.is_dir():
            counts[cls_dir.name] = len([p for p in cls_dir.iterdir() if p.is_file()])
    return counts


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
    files = sorted(result_dir.glob(f"{split}_epoch*.xlsx"), key=_epoch_sort_key)
    pred_dir = result_dir / "predictions"
    if pred_dir.exists():
        files.extend(sorted(pred_dir.glob(f"{split}_epoch*.xlsx"), key=_epoch_sort_key))
    if not files:
        return None
    return pd.read_excel(files[-1])


def _class_names_from_predictions(df: pd.DataFrame) -> List[str]:
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if prob_cols:
        return [col.replace("prob_", "") for col in prob_cols]
    return [str(x) for x in sorted(df["label"].unique())]


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _epoch_sort_key(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def _ablation_change(row: pd.Series) -> str:
    disabled = []
    for col, label in [
        ("use_time_branch", "without time branch"),
        ("use_spatial_branch", "without spatial branch"),
        ("use_idcnn", "without IDCNN"),
        ("use_dynamic_fusion", "without dynamic fusion"),
        ("use_moe", "without MoE"),
        ("use_fourier_block", "without Fourier block"),
    ]:
        if col in row and _is_false(row[col]):
            disabled.append(label)
    return "full proposed model" if not disabled else ", ".join(disabled)


def _model_display_name(name: object) -> str:
    text = str(name)
    aliases = {
        "resnet18": "ResNet18",
        "densenet121": "DenseNet121",
        "mobilenet_v2": "MobileNetV2",
        "efficientnet_b0": "EfficientNet-B0",
        "se_ecgnet": "SE-ECGNet",
        "lstm": "LSTM",
        "gru": "GRU",
        "Proposed": "Proposed",
    }
    return aliases.get(text, text)


def _ablation_display_name(name: object) -> str:
    text = str(name)
    return ABLATION_LABELS.get(text, text)


def _is_false(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return bool(value) is False
