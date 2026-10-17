import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.cache import build_tensor_cache
from src.data.datasets import CachedECGDataset
from src.data.splits import split_indices
from src.models.ecg_model import ECGThreeClassModel, count_params
from src.training.engine import evaluate, train_one_epoch
from src.training.losses import class_weights_from_labels, make_weighted_sampler
from src.utils.io import save_json
from src.utils.paths import resolve_results_paths
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Cached ECG training with external test evaluation")
    parser.add_argument("--data_root", type=str, default="Dataset/BuildModel")
    parser.add_argument("--external_root", type=str, default="Dataset/ExternalTest")
    parser.add_argument("--target_width", type=int, default=1064)
    parser.add_argument("--target_height", type=int, default=245)
    parser.add_argument("--seq_len", type=int, default=256, help="sequence length used by the time branch")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--use_sampler", action="store_true", help="use WeightedRandomSampler for class imbalance")
    args = parser.parse_args()

    set_seed(args.seed)
    save_dir, cache_root = resolve_results_paths(args.results_dir, args.save_dir, args.cache_root)
    predictions_dir = os.path.join(save_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

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
    print("Class mapping:", train_full.class_to_idx)
    print(f"Cached train tensors: {len(train_full)}")
    print(f"Cached external tensors: {len(external_ds)}")

    train_indices, val_indices = split_indices(len(train_full), args.val_split, args.seed)
    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    train_labels = [train_full.samples[i][1] for i in train_indices]
    sampler = make_weighted_sampler(train_labels, len(class_names)) if args.use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    external_loader = DataLoader(external_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights_from_labels(train_labels, len(class_names), device)
    print("Class weights:", {name: float(class_weights[i].cpu()) for i, name in enumerate(class_names)})

    model = ECGThreeClassModel(
        img_size=args.seq_len,
        t_embed=128,
        t_layers=2,
        s_idcnn=128,
        fusion_out=256,
        num_classes=len(class_names),
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    model_params_m = count_params(model) / 1e6
    print(f"Model params: {model_params_m:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc, best_path = -1.0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, class_weights, scheduler
        )
        val_result = evaluate(model, val_loader, device, class_names, split="val")
        ext_result = evaluate(model, external_loader, device, class_names, split="external")
        val_summary = val_result.summary_dict()
        ext_summary = ext_result.summary_dict()
        val_result.save_predictions(os.path.join(predictions_dir, f"val_epoch{epoch}.xlsx"))
        ext_result.save_predictions(os.path.join(predictions_dir, f"external_epoch{epoch}.xlsx"))

        history.append(
            {
                "experiment": "cached_main",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_result.loss,
                "val_acc": val_result.accuracy,
                "val_macro_f1": val_summary["macro_f1"],
                "val_auroc_macro": val_summary["auroc_macro"],
                "external_loss": ext_result.loss,
                "external_acc": ext_result.accuracy,
                "external_macro_f1": ext_summary["macro_f1"],
                "external_auroc_macro": ext_summary["auroc_macro"],
            }
        )
        pd.DataFrame(history).to_csv(os.path.join(save_dir, "metrics_history.csv"), index=False)

        print(
            f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_result.accuracy:.4f} "
            f"external_acc={ext_result.accuracy:.4f} val_loss={val_result.loss:.4f} external_loss={ext_result.loss:.4f}"
        )
        print("Validation Confusion Matrix:\n", val_result.cm)
        print("Validation Report:\n", val_result.report)
        print("External Confusion Matrix:\n", ext_result.cm)
        print("External Report:\n", ext_result.report)

        if val_result.accuracy > best_acc:
            best_acc = val_result.accuracy
            best_path = os.path.join(save_dir, f"best_epoch{epoch}_acc{best_acc:.4f}.pt")
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
            val_result.save_predictions(os.path.join(save_dir, "predictions_val.xlsx"))
            ext_result.save_predictions(os.path.join(save_dir, "predictions_external.xlsx"))
            save_json(
                {
                    "experiment": "cached_main",
                    "best_epoch": epoch,
                    "best_checkpoint": best_path,
                    "model_params_M": model_params_m,
                    "class_to_idx": train_full.class_to_idx,
                    "data_root": str(data_root),
                    "external_root": str(external_root),
                    "train_size": len(train_ds),
                    "val_size": len(val_ds),
                    "external_size": len(external_ds),
                    "args": vars(args),
                    "val": val_summary,
                    "external": ext_summary,
                },
                os.path.join(save_dir, "metrics_summary.json"),
            )
            print(f"Saved best to {best_path}")

    print(f"Training done. Best val acc={best_acc:.4f} @ {best_path}")


if __name__ == "__main__":
    main()
