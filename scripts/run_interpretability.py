import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import build_tensor_cache
from src.evaluation.gradcam import (
    GradCAM,
    RawCachedECGDataset,
    gradcam_target_layer,
    save_gradcam_overlay,
    select_case_indices,
)
from src.evaluation.robustness import load_main_model


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM cases for the main ECG model.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--prediction_table", type=str, default=None)
    parser.add_argument("--target_layer", type=str, default="spatial.cnn.layer4")
    parser.add_argument("--per_class_correct", type=int, default=2)
    parser.add_argument("--error_cases", type=int, default=6)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "interpretability"
    cam_dir = output_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, summary, device = load_main_model(results_dir)
    model_args = summary.get("args", {})
    target_width = int(model_args.get("target_width", 1064))
    target_height = int(model_args.get("target_height", 245))
    cache_root = Path(args.cache_root) if args.cache_root else results_dir / "cache"
    source_root = Path(summary.get("external_root", model_args.get("external_root", "Dataset/ExternalTest")))
    cache_dir = cache_root / f"ExternalTest_{target_width}x{target_height}"
    build_tensor_cache(source_root, cache_dir, target_width, target_height, rebuild=False)

    prediction_table = Path(args.prediction_table) if args.prediction_table else results_dir / "main" / "predictions_external.xlsx"
    predictions = pd.read_excel(prediction_table)
    dataset = RawCachedECGDataset(cache_dir)
    target_layer = gradcam_target_layer(model, args.target_layer)
    cam_runner = GradCAM(model, target_layer)

    rows = []
    try:
        for index in select_case_indices(predictions, args.per_class_correct, args.error_cases):
            image, raw, label, source = dataset[index]
            row = predictions.iloc[index]
            target_class = int(row["pred"])
            cam, pred_class, probs = cam_runner(image.unsqueeze(0).to(device), target_class=target_class)
            label_name = class_names[int(label)]
            pred_name = class_names[pred_class]
            confidence = float(probs[pred_class].item())
            cam_path = cam_dir / f"case_{index:04d}_{label_name}_to_{pred_name}.png"
            save_gradcam_overlay(raw, cam, cam_path)
            rows.append(
                {
                    "index": index,
                    "image_path": source,
                    "cam_path": str(cam_path),
                    "label": int(label),
                    "label_name": label_name,
                    "pred": pred_class,
                    "pred_name": pred_name,
                    "confidence": confidence,
                    "correct": bool(pred_class == int(label)),
                }
            )
            print(f"Saved Grad-CAM case {index}: {label_name} -> {pred_name}")
    finally:
        cam_runner.close()

    pd.DataFrame(rows).to_csv(output_dir / "gradcam_cases.csv", index=False)
    print(f"Interpretability outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
