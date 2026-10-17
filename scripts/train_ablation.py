import argparse
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
from src.models.ablation_model import AblationModel, count_params, get_ablation_configs
from src.training.engine import evaluate, train_one_epoch
from src.training.losses import class_weights_from_labels, make_weighted_sampler
from src.utils.io import save_json
from src.utils.seed import set_seed


def resolve_experiment_dirs(results_dir: str, save_dir: str | None, cache_root: str | None):
    os.makedirs(results_dir, exist_ok=True)
    if save_dir is None:
        save_dir = os.path.join(results_dir, "ablations")
    elif not os.path.isabs(save_dir) and not os.path.normpath(save_dir).startswith(os.path.normpath(results_dir) + os.sep):
        save_dir = os.path.join(results_dir, save_dir)
    if cache_root is None:
        cache_root = os.path.join(results_dir, "cache")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)
    return save_dir, cache_root


def run_single_ablation(
    config,
    args,
    train_loader,
    val_loader,
    external_loader,
    class_names,
    class_to_idx,
    class_weights,
    train_size,
    val_size,
    external_size,
    device,
):
    print(f"\n{'=' * 20} Ablation: {config.name} {'=' * 20}")
    experiment_dir = os.path.join(args.experiments_dir, config.name)
    predictions_dir = os.path.join(experiment_dir, "predictions")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    model = AblationModel(
        img_height=args.target_height,
        img_width=args.target_width,
        t_embed=args.t_embed,
        t_layers=args.t_layers,
        s_idcnn=args.s_idcnn,
        fusion_out=args.fusion_out,
        num_classes=len(class_names),
        pretrained_backbone=args.pretrained_backbone,
        **config.model_kwargs(),
    ).to(device)
    model_params_m = count_params(model) / 1e6
    print(f"Config: {config.model_kwargs()}")
    print(f"Model params: {model_params_m:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = []
    best_acc = -1.0
    best_path = None
    best_val_summary = None
    best_external_summary = None

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
                "experiment": config.name,
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
            f"external_acc={external_result.accuracy:.4f} external_macro_f1={external_summary['macro_f1']:.4f}"
        )

        if val_result.accuracy > best_acc:
            best_acc = val_result.accuracy
            best_path = os.path.join(experiment_dir, f"best_epoch{epoch}_acc{best_acc:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "class_to_idx": class_to_idx,
                    "config": config.model_kwargs(),
                    "args": vars(args),
                },
                best_path,
            )
            best_val_summary = val_summary
            best_external_summary = external_summary
            val_result.save_predictions(os.path.join(experiment_dir, "predictions_val.xlsx"))
            external_result.save_predictions(os.path.join(experiment_dir, "predictions_external.xlsx"))
            save_json(
                {
                    "experiment": config.name,
                    "best_epoch": epoch,
                    "best_checkpoint": best_path,
                    "config": config.model_kwargs(),
                    "model_params_M": model_params_m,
                    "class_to_idx": class_to_idx,
                    "data_root": args.data_root,
                    "external_root": args.external_root,
                    "train_size": train_size,
                    "val_size": val_size,
                    "external_size": external_size,
                    "args": vars(args),
                    "val": val_summary,
                    "external": external_summary,
                },
                os.path.join(experiment_dir, "metrics_summary.json"),
            )
            print(f"Saved best to {best_path}")

    return {
        "experiment": config.name,
        "best_val_accuracy": best_acc,
        "best_checkpoint": best_path,
        "model_params_M": f"{model_params_m:.2f}",
        "use_time_branch": config.use_time_branch,
        "use_spatial_branch": config.use_spatial_branch,
        "use_idcnn": config.use_idcnn,
        "use_dynamic_fusion": config.use_dynamic_fusion,
        "use_moe": config.use_moe,
        "use_fourier_block": config.use_fourier_block,
        "external_accuracy": None if best_external_summary is None else best_external_summary["accuracy"],
        "external_macro_f1": None if best_external_summary is None else best_external_summary["macro_f1"],
        "external_auroc_macro": None if best_external_summary is None else best_external_summary["auroc_macro"],
        "val_macro_f1": None if best_val_summary is None else best_val_summary["macro_f1"],
        "val_auroc_macro": None if best_val_summary is None else best_val_summary["auroc_macro"],
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation ECG experiments with cached data and external testing")
    parser.add_argument("--data_root", type=str, default="Dataset/BuildModel")
    parser.add_argument("--external_root", type=str, default="Dataset/ExternalTest")
    parser.add_argument("--target_width", type=int, default=1064)
    parser.add_argument("--target_height", type=int, default=245)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
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
    parser.add_argument("--configs", nargs="+", default=None, help="ablation config names, or all")
    parser.add_argument("--t_embed", type=int, default=128)
    parser.add_argument("--t_layers", type=int, default=2)
    parser.add_argument("--s_idcnn", type=int, default=128)
    parser.add_argument("--fusion_out", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    experiments_dir, cache_root = resolve_experiment_dirs(args.results_dir, args.save_dir, args.cache_root)
    args.experiments_dir = experiments_dir
    os.makedirs(experiments_dir, exist_ok=True)

    selected_configs = get_ablation_configs(args.configs)
    save_json(
        {"configs": [{"name": config.name, **config.model_kwargs()} for config in selected_configs]},
        os.path.join(experiments_dir, "ablation_plan.json"),
    )

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
    print("Selected ablations:", ", ".join(config.name for config in selected_configs))

    summary_rows = []
    for config in selected_configs:
        set_seed(args.seed)
        summary_rows.append(
            run_single_ablation(
                config=config,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                external_loader=external_loader,
                class_names=class_names,
                class_to_idx=train_full.class_to_idx,
                class_weights=class_weights,
                train_size=len(train_ds),
                val_size=len(val_ds),
                external_size=len(external_ds),
                device=device,
            )
        )
        pd.DataFrame(summary_rows).to_csv(os.path.join(experiments_dir, "ablation_summary.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(experiments_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n" + "=" * 30 + " Ablation Summary " + "=" * 30)
    print(summary_df.to_string(index=False))
    print(f"\nAblation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
