import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.make_figures import make_all_figures


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures from experiment results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    written = make_all_figures(args.results_dir, args.output_dir)
    if written:
        print("Generated figures:", ", ".join(written))
    else:
        print("No figures generated. Run the main/baseline/ablation experiments first.")


if __name__ == "__main__":
    main()

