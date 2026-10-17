from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass
class EvaluationResult:
    split: str
    loss: float
    labels: np.ndarray
    preds: np.ndarray
    probs: np.ndarray
    class_names: List[str]

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.labels, self.preds))

    @property
    def cm(self) -> np.ndarray:
        return confusion_matrix(self.labels, self.preds, labels=list(range(len(self.class_names))))

    @property
    def report(self) -> str:
        return classification_report(
            self.labels,
            self.preds,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )

    def prediction_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.probs, columns=[f"prob_{name}" for name in self.class_names])
        df["pred"] = self.preds
        df["pred_name"] = [self.class_names[i] for i in self.preds]
        df["label"] = self.labels
        df["label_name"] = [self.class_names[i] for i in self.labels]
        return df

    def save_predictions(self, path: str) -> None:
        self.prediction_table().to_excel(path, index=False)

    def summary_dict(self) -> Dict[str, Any]:
        n_classes = len(self.class_names)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.labels,
            self.preds,
            labels=list(range(n_classes)),
            zero_division=0,
        )
        specificity = _specificity_per_class(self.cm)
        summary: Dict[str, Any] = {
            "split": self.split,
            "loss": float(self.loss),
            "accuracy": self.accuracy,
            "macro_f1": float(f1_score(self.labels, self.preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(self.labels, self.preds, average="weighted", zero_division=0)),
            "per_class": {},
        }
        for idx, name in enumerate(self.class_names):
            summary["per_class"][name] = {
                "precision": float(precision[idx]),
                "sensitivity": float(recall[idx]),
                "specificity": float(specificity[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        summary.update(_probability_metrics(self.labels, self.probs, self.class_names))
        return summary


def build_evaluation_result(
    split: str,
    loss: float,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> EvaluationResult:
    return EvaluationResult(
        split=split,
        loss=float(loss),
        labels=np.asarray(labels),
        preds=np.asarray(preds),
        probs=np.asarray(probs),
        class_names=list(class_names),
    )


def _specificity_per_class(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    specificity = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificity.append(0.0 if denom == 0 else tn / denom)
    return np.asarray(specificity)


def _probability_metrics(labels: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    n_classes = len(class_names)
    out: Dict[str, Any] = {"auroc_macro": None, "auprc_macro": None, "per_class_auroc": {}, "per_class_auprc": {}}
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    try:
        out["auroc_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    try:
        out["auprc_macro"] = float(average_precision_score(y_bin, probs, average="macro"))
    except ValueError:
        pass
    for idx, name in enumerate(class_names):
        try:
            out["per_class_auroc"][name] = float(roc_auc_score(y_bin[:, idx], probs[:, idx]))
        except ValueError:
            out["per_class_auroc"][name] = None
        try:
            out["per_class_auprc"][name] = float(average_precision_score(y_bin[:, idx], probs[:, idx]))
        except ValueError:
            out["per_class_auprc"][name] = None
    return out

