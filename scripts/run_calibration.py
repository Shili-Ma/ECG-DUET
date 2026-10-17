import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import calibration_outputs


def main():
    parser = argparse.ArgumentParser(description="Run multiclass calibration and uncertainty analysis.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--prediction_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    prediction_dir = Path(args.prediction_dir) if args.prediction_dir else results_dir / "main"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, bins, curves, confidences = [], [], [], []
    for split in ["val", "external"]:
        path = prediction_dir / f"predictions_{split}.xlsx"
        if not path.exists():
            print(f"Skip missing prediction table: {path}")
            continue
        df = pd.read_excel(path)
        summary, bin_df, curve_df, confidence_df = calibration_outputs(df, split=split, n_bins=args.n_bins)
        summaries.append(summary)
        bins.append(bin_df)
        curves.append(curve_df)
        confidences.append(confidence_df)

    if not summaries:
        raise FileNotFoundError(f"No prediction tables found under {prediction_dir}")

    pd.DataFrame(summaries).to_csv(output_dir / "calibration_summary.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(output_dir / "calibration_bins.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "coverage_accuracy_curve.csv", index=False)
    pd.concat(confidences, ignore_index=True).to_csv(output_dir / "confidence_cases.csv", index=False)
    print(f"Calibration outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
