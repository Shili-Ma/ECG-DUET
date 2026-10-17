from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def class_names_from_prediction_table(df: pd.DataFrame) -> list[str]:
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if prob_cols:
        return [col.replace("prob_", "") for col in prob_cols]
    if "label_name" in df.columns:
        names = df[["label", "label_name"]].drop_duplicates().sort_values("label")
        return names["label_name"].astype(str).tolist()
    return [str(x) for x in sorted(df["label"].unique())]


def prediction_arrays(df: pd.DataFrame, class_names: Iterable[str] | None = None):
    names = list(class_names) if class_names is not None else class_names_from_prediction_table(df)
    prob_cols = [f"prob_{name}" for name in names]
    missing = [col for col in prob_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    labels = df["label"].to_numpy(dtype=int)
    preds = df["pred"].to_numpy(dtype=int)
    probs = df[prob_cols].to_numpy(dtype=float)
    return labels, preds, probs, names


def expected_calibration_error(labels, preds, probs, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    confidence = np.max(probs, axis=1)
    correct = (np.asarray(preds) == np.asarray(labels)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n_total = len(confidence)
    for bin_idx in range(n_bins):
        lo, hi = bins[bin_idx], bins[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            acc = np.nan
            conf = np.nan
            gap = np.nan
            weight = 0.0
        else:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            gap = abs(acc - conf)
            weight = count / n_total
            ece += weight * gap
        rows.append(
            {
                "bin": bin_idx + 1,
                "bin_lower": lo,
                "bin_upper": hi,
                "count": count,
                "accuracy": acc,
                "mean_confidence": conf,
                "abs_gap": gap,
                "weight": weight,
            }
        )
    return float(ece), pd.DataFrame(rows)


def multiclass_brier_score(labels, probs, n_classes: int) -> float:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def coverage_accuracy_curve(labels, preds, probs, n_points: int = 21) -> pd.DataFrame:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    confidence = np.max(probs, axis=1)
    thresholds = np.linspace(0.0, 1.0, n_points)
    rows = []
    for threshold in thresholds:
        mask = confidence >= threshold
        coverage = float(mask.mean())
        if mask.any():
            acc = float(accuracy_score(labels[mask], preds[mask]))
            error_rate = 1.0 - acc
            mean_conf = float(confidence[mask].mean())
        else:
            acc = np.nan
            error_rate = np.nan
            mean_conf = np.nan
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accuracy": acc,
                "error_rate": error_rate,
                "mean_confidence": mean_conf,
                "n_cases": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def confidence_table(df: pd.DataFrame) -> pd.DataFrame:
    labels, preds, probs, names = prediction_arrays(df)
    confidence = np.max(probs, axis=1)
    out = df.copy()
    out["confidence"] = confidence
    out["correct"] = labels == preds
    out["margin"] = np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]
    if "label_name" not in out.columns:
        out["label_name"] = [names[i] for i in labels]
    if "pred_name" not in out.columns:
        out["pred_name"] = [names[i] for i in preds]
    return out


def calibration_outputs(df: pd.DataFrame, split: str, n_bins: int = 10) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels, preds, probs, class_names = prediction_arrays(df)
    ece, bins = expected_calibration_error(labels, preds, probs, n_bins=n_bins)
    brier = multiclass_brier_score(labels, probs, n_classes=len(class_names))
    curve = coverage_accuracy_curve(labels, preds, probs)
    conf = confidence_table(df)
    summary = {
        "split": split,
        "n_cases": int(len(df)),
        "accuracy": float(accuracy_score(labels, preds)),
        "ece": ece,
        "brier_score": brier,
        "mean_confidence": float(np.max(probs, axis=1).mean()),
        "mean_confidence_correct": float(conf.loc[conf["correct"], "confidence"].mean()),
        "mean_confidence_incorrect": float(conf.loc[~conf["correct"], "confidence"].mean())
        if (~conf["correct"]).any()
        else np.nan,
    }
    bins.insert(0, "split", split)
    curve.insert(0, "split", split)
    conf.insert(0, "split", split)
    return summary, bins, curve, conf
