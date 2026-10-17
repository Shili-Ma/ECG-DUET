import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.make_tables import make_all_tables


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript tables from experiment results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    written = make_all_tables(args.results_dir, args.output_dir)
    print("Generated tables:", ", ".join(written))


if __name__ == "__main__":
    main()

