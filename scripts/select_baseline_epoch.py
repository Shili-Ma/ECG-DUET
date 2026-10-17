import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import build_evaluation_result
from src.utils.io import save_json


def infer_class_names(df: pd.DataFrame) -> list[str]:
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if prob_cols:
        return [col.replace("prob_", "") for col in prob_cols]
    if "label_name" in df.columns:
        names = df[["label", "label_name"]].drop_duplicates().sort_values("label")
        return names["label_name"].astype(str).tolist()
    return [str(x) for x in sorted(df["label"].unique())]


def result_from_table(df: pd.DataFrame, split: str):
    class_names = infer_class_names(df)
    prob_cols = [f"prob_{name}" for name in class_names]
    missing = [col for col in prob_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns in {split} prediction table: {missing}")
    return build_evaluation_result(
        split=split,
        loss=float("nan"),
        labels=df["label"].to_numpy(),
        preds=df["pred"].to_numpy(),
        probs=df[prob_cols].to_numpy(),
        class_names=class_names,
    )


def load_epoch_plan(path: str | None) -> dict[str, int] | None:
    if path is None:
        return None
    plan_df = pd.read_csv(path)
    if "model_name" not in plan_df.columns or "epoch" not in plan_df.columns:
        raise ValueError("epoch plan CSV must contain columns: model_name, epoch")
    return {str(row["model_name"]): int(row["epoch"]) for _, row in plan_df.iterrows()}


def load_existing_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_one_model(model_dir: Path, model_name: str, epoch: int, baseline_args: argparse.Namespace) -> dict:
    pred_dir = model_dir / "predictions"
    val_path = pred_dir / f"val_epoch{epoch}.xlsx"
    external_path = pred_dir / f"external_epoch{epoch}.xlsx"
    if not val_path.exists() or not external_path.exists():
        raise FileNotFoundError(
            f"Missing prediction files for {model_name} epoch {epoch}: {val_path}, {external_path}"
        )

    val_df = pd.read_excel(val_path)
    external_df = pd.read_excel(external_path)
    val_result = result_from_table(val_df, "val")
    external_result = result_from_table(external_df, "external")
    val_summary = val_result.summary_dict()
    external_summary = external_result.summary_dict()

    val_df.to_excel(model_dir / "predictions_val.xlsx", index=False)
    external_df.to_excel(model_dir / "predictions_external.xlsx", index=False)

    old_summary = load_existing_summary(model_dir / "metrics_summary.json")
    best_epoch_by_val_accuracy = old_summary.get("best_epoch_by_val_accuracy", old_summary.get("best_epoch"))
    best_checkpoint_by_val_accuracy = old_summary.get(
        "best_checkpoint_by_val_accuracy",
        old_summary.get("best_checkpoint"),
    )
    model_params_m = old_summary.get("model_params_M")

    selected_checkpoint = old_summary.get("selected_checkpoint")
    if selected_checkpoint is None or old_summary.get("selected_epoch") != epoch:
        selected_checkpoint = None

    summary = {
        **old_summary,
        "experiment": model_name,
        "selection_mode": "manual_existing_epoch",
        "selected_epoch": epoch,
        "selected_checkpoint": selected_checkpoint,
        "best_epoch": epoch,
        "best_checkpoint": selected_checkpoint,
        "best_epoch_by_val_accuracy": best_epoch_by_val_accuracy,
        "best_checkpoint_by_val_accuracy": best_checkpoint_by_val_accuracy,
        "class_to_idx": {name: idx for idx, name in enumerate(val_result.class_names)},
        "args": old_summary.get("args", {}),
        "val": val_summary,
        "external": external_summary,
    }
    if model_params_m is not None:
        summary["model_params_M"] = model_params_m
    save_json(summary, str(model_dir / "metrics_summary.json"))

    return {
        "model_name": model_name,
        "selection_mode": "manual_existing_epoch",
        "selected_epoch": epoch,
        "best_val_accuracy": val_result.accuracy,
        "val_macro_f1": val_summary["macro_f1"],
        "val_weighted_f1": val_summary["weighted_f1"],
        "val_auroc_macro": val_summary["auroc_macro"],
        "val_auprc_macro": val_summary["auprc_macro"],
        "best_checkpoint": selected_checkpoint,
        "best_epoch_by_val_accuracy": best_epoch_by_val_accuracy,
        "best_checkpoint_by_val_accuracy": best_checkpoint_by_val_accuracy,
        "model_params_M": model_params_m,
        "pretrained": old_summary.get("args", {}).get("pretrained"),
        "external_accuracy": external_result.accuracy,
        "external_macro_f1": external_summary["macro_f1"],
        "external_auroc_macro": external_summary["auroc_macro"],
        "external_auprc_macro": external_summary["auprc_macro"],
    }


def main_model_row(results_dir: Path, model_name: str) -> dict | None:
    summary_path = results_dir / "main" / "metrics_summary.json"
    if not summary_path.exists():
        return None
    summary = load_existing_summary(summary_path)
    val = summary.get("val", {})
    external = summary.get("external", {})
    return {
        "model_name": model_name,
        "selection_mode": "main_best_val_accuracy",
        "selected_epoch": summary.get("selected_epoch", summary.get("best_epoch")),
        "best_val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "val_weighted_f1": val.get("weighted_f1"),
        "val_auroc_macro": val.get("auroc_macro"),
        "val_auprc_macro": val.get("auprc_macro"),
        "best_checkpoint": summary.get("selected_checkpoint", summary.get("best_checkpoint")),
        "best_epoch_by_val_accuracy": summary.get("best_epoch_by_val_accuracy", summary.get("best_epoch")),
        "best_checkpoint_by_val_accuracy": summary.get(
            "best_checkpoint_by_val_accuracy",
            summary.get("best_checkpoint"),
        ),
        "model_params_M": summary.get("model_params_M"),
        "pretrained": summary.get("args", {}).get("pretrained_backbone"),
        "external_accuracy": external.get("accuracy"),
        "external_macro_f1": external.get("macro_f1"),
        "external_auroc_macro": external.get("auroc_macro"),
        "external_auprc_macro": external.get("auprc_macro"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select existing baseline epoch predictions as formal results without retraining."
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--baselines_dir", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epoch", type=int, default=None, help="use the same epoch for all selected models")
    parser.add_argument(
        "--epoch_plan",
        type=str,
        default=None,
        help="CSV with columns model_name,epoch for per-model epoch selection",
    )
    parser.add_argument(
        "--include_main",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include results/main/metrics_summary.json as the Proposed row in baseline_summary.csv",
    )
    parser.add_argument("--main_model_name", type=str, default="Proposed")
    args = parser.parse_args()

    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else Path(args.results_dir) / "baselines"
    if not baselines_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baselines_dir}")
    if args.epoch is None and args.epoch_plan is None:
        raise ValueError("Provide either --epoch or --epoch_plan")

    plan = load_epoch_plan(args.epoch_plan)
    if args.models is None:
        if plan is not None:
            model_names = list(plan.keys())
        else:
            model_names = sorted(p.name for p in baselines_dir.iterdir() if p.is_dir())
    else:
        model_names = args.models

    summary_path = baselines_dir / "baseline_summary.csv"
    old_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    old_by_model = {}
    if not old_summary.empty and "model_name" in old_summary.columns:
        old_by_model = {
            str(row["model_name"]): row.to_dict()
            for _, row in old_summary.iterrows()
        }

    rows = []
    for model_name in model_names:
        epoch = plan[model_name] if plan is not None else args.epoch
        row = select_one_model(baselines_dir / model_name, model_name, int(epoch), args)
        old_row = old_by_model.get(model_name, {})
        for key in ["model_params_M", "pretrained"]:
            if row.get(key) is None and key in old_row:
                row[key] = old_row[key]
        rows.append(row)
        print(
            f"Selected {model_name} epoch {epoch}: "
            f"val_acc={row['best_val_accuracy']:.4f}, external_acc={row['external_accuracy']:.4f}"
        )

    new_rows = pd.DataFrame(rows)
    if not old_summary.empty:
        remove_names = set(model_names)
        if args.include_main:
            remove_names.add(args.main_model_name)
        old_summary = old_summary[~old_summary["model_name"].astype(str).isin(remove_names)]
        summary = pd.concat([old_summary, new_rows], ignore_index=True, sort=False)
    else:
        summary = new_rows
    if args.include_main:
        main_row = main_model_row(Path(args.results_dir), args.main_model_name)
        if main_row is not None:
            summary = pd.concat([summary, pd.DataFrame([main_row])], ignore_index=True, sort=False)
            print(f"Included main model row: {args.main_model_name}")
        else:
            print("Main model summary not found; skipped main model row.")
    try:
        summary.to_csv(summary_path, index=False)
        print(f"Updated baseline summary: {summary_path}")
    except PermissionError:
        fallback_path = summary_path.with_name(f"{summary_path.stem}_updated.csv")
        summary.to_csv(fallback_path, index=False)
        print(f"Could not overwrite locked file: {summary_path}")
        print(f"Wrote updated baseline summary to: {fallback_path}")


if __name__ == "__main__":
    main()
