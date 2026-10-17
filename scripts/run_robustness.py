import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import build_tensor_cache
from src.evaluation.robustness import (
    PerturbedCachedECGDataset,
    evaluate_loader_for_robustness,
    load_main_model,
    parse_perturbation_grid,
    robustness_auc,
)


def main():
    parser = argparse.ArgumentParser(description="Run robustness perturbation analysis for the main model.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--split", type=str, default="external", choices=["external"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--perturbation_grid", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, summary, device = load_main_model(results_dir)
    model_args = summary.get("args", {})
    target_width = int(model_args.get("target_width", 1064))
    target_height = int(model_args.get("target_height", 245))
    cache_root = Path(args.cache_root) if args.cache_root else results_dir / "cache"
    source_root = Path(summary.get("external_root", model_args.get("external_root", "Dataset/ExternalTest")))
    cache_dir = cache_root / f"ExternalTest_{target_width}x{target_height}"
    build_tensor_cache(source_root, cache_dir, target_width, target_height, rebuild=False)

    grid = parse_perturbation_grid(args.perturbation_grid)
    pd.DataFrame(grid).to_csv(output_dir / "perturbation_plan.csv", index=False)

    rows = []
    for spec in grid:
        perturbation = spec["perturbation"]
        severity = spec["severity"]
        level = spec.get("level", pd.NA)
        level_name = spec.get("level_name", "")
        dataset = PerturbedCachedECGDataset(cache_dir, perturbation=perturbation, severity=severity)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        pred_df, metrics = evaluate_loader_for_robustness(model, loader, device, class_names)
        row = {
            "split": args.split,
            "perturbation": perturbation,
            "severity": severity,
            "level": level,
            "level_name": level_name,
            **metrics,
        }
        rows.append(row)
        if args.save_predictions:
            pred_path = output_dir / "predictions" / f"{perturbation}_level{level}_{str(severity).replace('.', 'p')}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(pred_path, index=False)
        print(
            f"{perturbation} level={level} severity={severity}: "
            f"acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "robustness_summary.csv", index=False)
    robustness_auc(summary_df).to_csv(output_dir / "robustness_auc.csv", index=False)
    print(f"Robustness outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
