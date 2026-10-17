"""Exercise model forward/backward and metric export using synthetic data only."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.evaluation.metrics import build_evaluation_result
from src.models.ablation_model import AblationModel, get_ablation_configs
from src.models.ecg_model import ECGThreeClassModel


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    images = torch.randn(2, 3, 32, 64)
    labels = torch.tensor([0, 2])

    model = ECGThreeClassModel(img_size=32)
    logits = model(images)
    assert logits.shape == (2, 3)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()
    del model

    for config in get_ablation_configs():
        model = AblationModel(img_height=32, img_width=64, **config.model_kwargs()).eval()
        with torch.no_grad():
            output = model(images)
        assert output.shape == (2, 3), config.name
        assert torch.isfinite(output).all(), config.name
        del model

    labels_np = torch.tensor([0, 1, 2]).numpy()
    probabilities = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]).numpy()
    result = build_evaluation_result(
        split="synthetic", loss=0.0, labels=labels_np, preds=labels_np,
        probs=probabilities, class_names=["AF", "N", "PAT"],
    )
    assert result.accuracy == 1.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.xlsx"
        result.save_predictions(str(path))
        assert len(pd.read_excel(path)) == 3
    print("PASS: main forward/backward, seven ablation forwards, metrics and Excel export")


if __name__ == "__main__":
    main()
