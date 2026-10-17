import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import build_tensor_cache
from src.data.datasets import CachedECGDataset
from src.data.splits import split_indices
from src.models.baselines import DEFAULT_BASELINES, get_baseline_model
from src.training.engine import evaluate, train_one_epoch
from src.training.losses import class_weights_from_labels
from src.utils.io import save_json
from src.utils.seed import set_seed


def maybe_main_model_row(results_dir: str, model_name: str = "Proposed") -> dict | None:
    summary_path = Path(results_dir) / "main" / "metrics_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
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
        "best_val_accuracy_by_val_accuracy": val.get("accuracy"),
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
    parser = argparse.ArgumentParser(description="Baseline ECG experiments with cached data")
    parser.add_argument("--data_root", type=str, default="Dataset/BuildModel")
    parser.add_argument("--external_root", type=str, default="Dataset/ExternalTest")
    parser.add_argument("--target_width", type=int, default=1064)
    parser.add_argument("--target_height", type=int, default=245)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained weights")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--models", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument(
        "--select_epoch",
        type=int,
        default=None,
        help="use this epoch as the formal baseline result; default uses the best validation-accuracy epoch",
    )
    args = parser.parse_args()
    if args.select_epoch is not None and not (1 <= args.select_epoch <= args.epochs):
        raise ValueError(f"--select_epoch must be between 1 and --epochs ({args.epochs})")

    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    experiments_dir = args.save_dir
    if experiments_dir is None:
        experiments_dir = os.path.join(args.results_dir, "baselines")
    elif not os.path.isabs(experiments_dir) and not os.path.normpath(experiments_dir).startswith(os.path.normpath(args.results_dir) + os.sep):
        experiments_dir = os.path.join(args.results_dir, experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    cache_root = args.cache_root or os.path.join(args.results_dir, "cache")
    os.makedirs(cache_root, exist_ok=True)

    data_root = Path(args.data_root)
    external_root = Path(args.external_root)
    cache_root = Path(cache_root)
    train_cache_root = cache_root / f"BuildModel_{args.target_width}x{args.target_height}"
    external_cache_root = cache_root / f"ExternalTest_{args.target_width}x{args.target_height}"

    train_mapping = build_tensor_cache(data_root, train_cache_root, args.target_width, args.target_height, args.rebuild_cache)
    external_mapping = build_tensor_cache(external_root, external_cache_root, args.target_width, args.target_height, args.rebuild_cache)
    if train_mapping != external_mapping:
        raise ValueError(f"External mapping {external_mapping} does not match training mapping {train_mapping}")

    train_full = CachedECGDataset(str(train_cache_root), train=True)
    val_full = CachedECGDataset(str(train_cache_root), train=False)
    external_ds = CachedECGDataset(str(external_cache_root), train=False)
    class_names = train_full.classes

    train_indices, val_indices = split_indices(len(train_full), args.val_split, args.seed)
    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    train_labels = [train_full.samples[i][1] for i in train_indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights_from_labels(train_labels, len(class_names), device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    external_loader = DataLoader(external_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    summary_rows = []

    for model_name in args.models:
        print(f"\n{'='*20} Baseline: {model_name} {'='*20}")
        experiment_dir = os.path.join(experiments_dir, model_name)
        predictions_dir = os.path.join(experiment_dir, "predictions")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)

        model = get_baseline_model(
            model_name=model_name,
            num_classes=len(class_names),
            img_height=args.target_height,
            img_width=args.target_width,
            pretrained=args.pretrained,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        history = []
        best_epoch = None
        best_acc = -1.0
        best_path = None
        best_val_result = None
        best_external_result = None
        selected_path = None
        selected_val_result = None
        selected_external_result = None
        model_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, device, epoch, class_weights, scheduler
            )
            val_result = evaluate(model, val_loader, device, class_names, split="val")
            external_result = evaluate(model, external_loader, device, class_names, split="external")
            val_summary = val_result.summary_dict()
            external_summary = external_result.summary_dict()
            val_result.save_predictions(os.path.join(predictions_dir, f"val_epoch{epoch}.xlsx"))
            external_result.save_predictions(os.path.join(predictions_dir, f"external_epoch{epoch}.xlsx"))

            history.append(
                {
                    "experiment": model_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_result.loss,
                    "val_acc": val_result.accuracy,
                    "val_macro_f1": val_summary["macro_f1"],
                    "val_auroc_macro": val_summary["auroc_macro"],
                    "external_loss": external_result.loss,
                    "external_acc": external_result.accuracy,
                    "external_macro_f1": external_summary["macro_f1"],
                    "external_auroc_macro": external_summary["auroc_macro"],
                }
            )
            pd.DataFrame(history).to_csv(os.path.join(experiment_dir, "metrics_history.csv"), index=False)

            print(
                f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_result.accuracy:.4f} "
                f"external_acc={external_result.accuracy:.4f}"
            )

            if val_result.accuracy > best_acc:
                best_epoch = epoch
                best_acc = val_result.accuracy
                best_path = os.path.join(experiment_dir, f"best_epoch{epoch}_acc{best_acc:.4f}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "class_to_idx": train_full.class_to_idx,
                        "args": vars(args),
                    },
                    best_path,
                )
                best_val_result = val_result
                best_external_result = external_result

            if args.select_epoch is not None and epoch == args.select_epoch:
                selected_path = os.path.join(experiment_dir, f"selected_epoch{epoch}_acc{val_result.accuracy:.4f}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "class_to_idx": train_full.class_to_idx,
                        "args": vars(args),
                        "selection_mode": "manual_epoch",
                    },
                    selected_path,
                )
                selected_val_result = val_result
                selected_external_result = external_result

        if args.select_epoch is None:
            formal_epoch = best_epoch
            formal_checkpoint = best_path
            formal_val_result = best_val_result
            formal_external_result = best_external_result
            selection_mode = "best_val_accuracy"
        else:
            formal_epoch = args.select_epoch
            formal_checkpoint = selected_path
            formal_val_result = selected_val_result
            formal_external_result = selected_external_result
            selection_mode = "manual_epoch"

        if formal_val_result is None or formal_external_result is None:
            raise RuntimeError(f"No formal result was selected for baseline {model_name}.")

        formal_val_summary = formal_val_result.summary_dict()
        formal_external_summary = formal_external_result.summary_dict()
        formal_val_result.save_predictions(os.path.join(experiment_dir, "predictions_val.xlsx"))
        formal_external_result.save_predictions(os.path.join(experiment_dir, "predictions_external.xlsx"))
        save_json(
            {
                "experiment": model_name,
                "selection_mode": selection_mode,
                "selected_epoch": formal_epoch,
                "selected_checkpoint": formal_checkpoint,
                "best_epoch": formal_epoch,
                "best_checkpoint": formal_checkpoint,
                "best_epoch_by_val_accuracy": best_epoch,
                "best_checkpoint_by_val_accuracy": best_path,
                "best_val_accuracy_by_val_accuracy": best_acc,
                "model_params_M": model_params_m,
                "class_to_idx": train_full.class_to_idx,
                "args": vars(args),
                "val": formal_val_summary,
                "external": formal_external_summary,
            },
            os.path.join(experiment_dir, "metrics_summary.json"),
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "selection_mode": selection_mode,
                "selected_epoch": formal_epoch,
                "best_val_accuracy": formal_val_result.accuracy,
                "val_macro_f1": formal_val_summary["macro_f1"],
                "val_weighted_f1": formal_val_summary["weighted_f1"],
                "val_auroc_macro": formal_val_summary["auroc_macro"],
                "val_auprc_macro": formal_val_summary["auprc_macro"],
                "best_checkpoint": formal_checkpoint,
                "best_epoch_by_val_accuracy": best_epoch,
                "best_val_accuracy_by_val_accuracy": best_acc,
                "best_checkpoint_by_val_accuracy": best_path,
                "model_params_M": f"{model_params_m:.2f}",
                "pretrained": args.pretrained,
                "external_accuracy": formal_external_result.accuracy,
                "external_macro_f1": formal_external_summary["macro_f1"],
                "external_auroc_macro": formal_external_summary["auroc_macro"],
                "external_auprc_macro": formal_external_summary["auprc_macro"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    main_row = maybe_main_model_row(args.results_dir)
    if main_row is not None:
        summary_df = pd.concat([pd.DataFrame([main_row]), summary_df], ignore_index=True, sort=False)
    summary_path = os.path.join(experiments_dir, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 30 + " Baseline Summary " + "=" * 30)
    print(summary_df.to_string(index=False))
    print(f"\nBaseline summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
