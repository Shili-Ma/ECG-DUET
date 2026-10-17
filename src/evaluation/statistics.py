from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .calibration import prediction_arrays


def metric_value(labels, preds, metric: str, class_idx: int | None = None) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    if metric == "accuracy":
        return float(accuracy_score(labels, preds))
    if metric == "macro_f1":
        return float(f1_score(labels, preds, average="macro", zero_division=0))
    if metric == "weighted_f1":
        return float(f1_score(labels, preds, average="weighted", zero_division=0))
    if metric == "class_f1":
        if class_idx is None:
            raise ValueError("class_idx is required for class_f1")
        values = f1_score(labels, preds, labels=[class_idx], average=None, zero_division=0)
        return float(values[0])
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(
    labels,
    preds,
    metric: str,
    n_boot: int = 1000,
    seed: int = 42,
    class_idx: int | None = None,
    alpha: float = 0.05,
) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    rng = np.random.default_rng(seed)
    n = len(labels)
    observed = metric_value(labels, preds, metric, class_idx=class_idx)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(metric_value(labels[idx], preds[idx], metric, class_idx=class_idx))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric if class_idx is None else f"class_{class_idx}_f1",
        "estimate": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
    }


def prediction_metric_ci_table(df: pd.DataFrame, split: str, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    labels, preds, _, class_names = prediction_arrays(df)
    rows = []
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        row = bootstrap_ci(labels, preds, metric, n_boot=n_boot, seed=seed)
        row.update({"split": split, "class_name": "overall"})
        rows.append(row)
    for idx, name in enumerate(class_names):
        row = bootstrap_ci(labels, preds, "class_f1", n_boot=n_boot, seed=seed + idx + 1, class_idx=idx)
        row.update({"split": split, "class_name": name, "metric": "class_f1"})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    labels,
    preds_a,
    preds_b,
    metric: str = "macro_f1",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    if len(labels) != len(preds_a) or len(labels) != len(preds_b):
        raise ValueError("Paired predictions must have the same length.")
    rng = np.random.default_rng(seed)
    n = len(labels)
    observed = metric_value(labels, preds_a, metric) - metric_value(labels, preds_b, metric)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(metric_value(labels[idx], preds_a[idx], metric) - metric_value(labels[idx], preds_b[idx], metric))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "difference": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
    }


def mcnemar_test(labels, preds_a, preds_b) -> dict:
    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    a_correct = preds_a == labels
    b_correct = preds_b == labels
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        statistic = 0.0
        p_value = 1.0
    elif n < 25:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
        statistic = float((abs(b - c) - 1) ** 2 / n) if n else 0.0
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n)
        p_value = float(math.erfc(math.sqrt(statistic / 2.0)))
    return {
        "main_correct_baseline_wrong": b,
        "main_wrong_baseline_correct": c,
        "mcnemar_n": n,
        "mcnemar_statistic": statistic,
        "mcnemar_p_value": p_value,
    }


def compare_prediction_tables(
    main_df: pd.DataFrame,
    other_df: pd.DataFrame,
    other_name: str,
    split: str = "external",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    labels, main_preds, _, _ = prediction_arrays(main_df)
    other_labels, other_preds, _, _ = prediction_arrays(other_df)
    if not np.array_equal(labels, other_labels):
        raise ValueError(f"Labels are not aligned for comparison with {other_name}")
    diff = paired_bootstrap_difference(labels, main_preds, other_preds, n_boot=n_boot, seed=seed)
    mcnemar = mcnemar_test(labels, main_preds, other_preds)
    return {
        "split": split,
        "comparison": other_name,
        **diff,
        **mcnemar,
    }
