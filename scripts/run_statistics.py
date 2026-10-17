import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistics import compare_prediction_tables, prediction_metric_ci_table


def compare_folder(
    main_df: pd.DataFrame,
    folder: Path,
    name_col: str,
    split: str,
    n_boot: int,
    seed: int,
    skip_names: set[str] | None = None,
):
    rows = []
    skip_names = skip_names or set()
    if not folder.exists():
        return rows
    for child in sorted(p for p in folder.iterdir() if p.is_dir()):
        if child.name in skip_names:
            continue
        path = child / f"predictions_{split}.xlsx"
        if not path.exists():
            continue
        try:
            other = pd.read_excel(path)
            rows.append(compare_prediction_tables(main_df, other, child.name, split=split, n_boot=n_boot, seed=seed))
        except Exception as exc:
            print(f"Skip comparison with {child.name}: {exc}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run bootstrap confidence intervals and paired comparisons.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    ci_frames = []
    for split in ["val", "external"]:
        path = results_dir / "main" / f"predictions_{split}.xlsx"
        if not path.exists():
            print(f"Skip missing prediction table: {path}")
            continue
        df = pd.read_excel(path)
        ci_frames.append(prediction_metric_ci_table(df, split=split, n_boot=args.n_boot, seed=args.seed))
    if ci_frames:
        pd.concat(ci_frames, ignore_index=True).to_csv(output_dir / "main_metric_ci.csv", index=False)

    external_main = results_dir / "main" / "predictions_external.xlsx"
    if external_main.exists():
        main_df = pd.read_excel(external_main)
        baseline_rows = compare_folder(
            main_df,
            results_dir / "baselines",
            name_col="model_name",
            split="external",
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if baseline_rows:
            pd.DataFrame(baseline_rows).to_csv(output_dir / "main_vs_baselines.csv", index=False)
        ablation_rows = compare_folder(
            main_df,
            results_dir / "ablations",
            name_col="experiment",
            split="external",
            n_boot=args.n_boot,
            seed=args.seed,
            skip_names={"1_Full_Model"},
        )
        if ablation_rows:
            pd.DataFrame(ablation_rows).to_csv(output_dir / "main_vs_ablations.csv", index=False)

    print(f"Statistics outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
