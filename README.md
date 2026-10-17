# ECG Image Classification

A PyTorch research pipeline for classifying ECG images into **AF**, **N**, and **PAT**, using temporal and spatial features with dynamic fusion and a mixture-of-experts classifier.

The repository includes training, external evaluation, baseline comparisons, ablation studies, robustness and calibration analysis, bootstrap statistics, Grad-CAM, and figure/table generation. **21 real ECG sample images** are included for a small end-to-end demonstration. Full datasets and trained weights are not included.

[Quick start](#quick-start) · [Model](#model) · [Data preparation](#data-preparation) · [Experiments](#experiments) · [Outputs](#outputs)

## Quick start

### 1. Install dependencies

Use **Python 3.10+** (3.11 recommended). Download or clone the repository, open its root directory, and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install a matching PyTorch/torchvision pair using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). The code uses CUDA when available and otherwise runs on CPU.

### 2. Run the real ECG example

Run this command from the repository root; no additional dataset is needed:

```bash
python main.py --data_root examples/sample_data/BuildModel --external_root examples/sample_data/ExternalTest --results_dir results/sample_demo --epochs 1 --batch_size 2 --num_workers 0 --val_split 0.4
```

The example contains 5 development images and 2 external images per class. The command uses **9 training, 6 validation, and 6 external images** to exercise preprocessing, caching, one training epoch, evaluation, and result export.

Outputs are saved to `results/sample_demo/main/`, including a checkpoint, metrics, and prediction spreadsheets. These small-sample metrics are only useful for checking the software workflow. See the [sample data notes](examples/sample_data/README.md) for selection and processing details.

An additional check uses synthetic tensors and requires no image files:

```bash
python scripts/smoke_test.py
```

## Model

The main model combines two feature branches:

```text
                         ECG image
                             |
              +--------------+--------------+
              |                             |
       Image-to-sequence              ResNet18 + IDCNN
       Fourier + Transformer          Spatial features
              |                             |
              +--------------+--------------+
                             |
                       Dynamic fusion
                             |
                   Mixture of experts
                             |
                       AF / N / PAT
```

The temporal branch converts image columns into a 256-position sequence; the spatial branch combines global ResNet18 features with local dilated-convolution features. A learned gate fuses both branches, followed by four experts with top-2 routing. The default model has approximately **13.30 million trainable parameters**.

Implementation: [`src/models/ecg_model.py`](src/models/ecg_model.py).

## Data preparation

For experiments on your own dataset, arrange images as follows:

```text
Dataset/
├── BuildModel/
│   ├── AF/
│   ├── N/
│   └── PAT/
└── ExternalTest/
    ├── AF/
    ├── N/
    └── PAT/
```

Place PNG, JPG, JPEG, or BMP files directly inside the class folders. Development and external datasets must have matching class names. Use unique filename stems within each class.

Images are resized with aspect-ratio-preserving white padding to **1064 × 245 pixels** and cached as tensors. The default development split is 85% training and 15% validation, with seed 42. This is an image-level random split, without patient grouping or class stratification; establish patient independence before scientific evaluation.

## Experiments

### Main-model training

```bash
python main.py --epochs 40 --batch_size 64 --num_workers 4
```

| Option | Purpose |
| --- | --- |
| `--data_root`, `--external_root` | Set the development and external image directories |
| `--results_dir` | Set the experiment output root; default: `results` |
| `--batch_size` | Reduce this value if memory is insufficient |
| `--num_workers 0` | Load data in the main process |
| `--rebuild_cache` | Recreate cached tensors after changing images |
| `--use_sampler` | Enable weighted sampling for class imbalance |
| `--pretrained_backbone` | Load ImageNet-pretrained ResNet18 weights |

The best checkpoint is selected by **validation accuracy**. Keep external predictions independent of hyperparameter and epoch selection. Use a separate results directory for each independent run.

### Baselines and ablations

```bash
python scripts/train_baselines.py --epochs 40
python scripts/train_ablation.py --epochs 40 --configs all
```

Baselines include ResNet18, DenseNet121, MobileNetV2, EfficientNet-B0, SE-ECGNet, ViT, Swin Transformer, LSTM, and GRU. Ablations study the temporal branch, spatial branch, IDCNN, dynamic fusion, MoE, and Fourier block.

**Current limitation:** the ablation implementation uses a different image-to-sequence encoder from the main model. Its full-model configuration is therefore not identical to the main model. See [experiment notes](docs/experiment_notes.md) before interpreting component comparisons. Baseline and ablation scripts default to 5 epochs; the commands above set the budget explicitly.

### Analysis and visualization

Run these commands after main-model training:

| Task | Command |
| --- | --- |
| Perturbation robustness | `python scripts/run_robustness.py` |
| Calibration and uncertainty | `python scripts/run_calibration.py` |
| Bootstrap intervals and paired comparisons | `python scripts/run_statistics.py --n_boot 1000` |
| Grad-CAM | `python scripts/run_interpretability.py` |
| Figures | `python scripts/make_figures.py` |
| Tables | `python scripts/make_tables.py` |

Robustness and Grad-CAM require a trained checkpoint and images/cache; calibration and statistics read prediction spreadsheets. Comparison outputs also require baseline or ablation results. Pass the same `--results_dir` to downstream commands when using a custom output root.

All experiment entry points support `--help`. Details on reporting behavior, manual epoch selection, and cross-platform paths are in [experiment notes](docs/experiment_notes.md).

## Outputs

A main-model run creates:

```text
results/main/
├── best_epoch*_acc*.pt        # Best validation checkpoint
├── metrics_history.csv       # Metrics for every epoch
├── metrics_summary.json      # Selected-model summary
├── predictions_val.xlsx      # Selected validation predictions
├── predictions_external.xlsx # Selected external predictions
└── predictions/              # Per-epoch prediction spreadsheets
```

Caches, baseline runs, ablations, analyses, figures, and tables occupy separate subdirectories under `results/`. Generated outputs and full datasets are excluded from Git by `.gitignore`.

## Repository structure

```text
main.py                 Main-model training
src/
  data/                 Preprocessing, cache, datasets, and splits
  models/               Main model, baselines, and ablations
  training/             Training/evaluation loops and losses
  evaluation/           Metrics and analysis methods
  reporting/            Figure and table generation
  utils/                Seeds, paths, and serialization
scripts/                Experiment commands and smoke check
examples/sample_data/   Real ECG samples and manifest
docs/                  Experiment and implementation notes
requirements.txt        Dependency constraints
.github/workflows/      Automated syntax, CLI, and model checks
```

## Reproducibility and data availability

The sample images enable workflow testing; the full dataset, historical results, and pretrained study checkpoints are not distributed. Dependency ranges are provided in `requirements.txt`; record exact installed versions for each experiment. The current split, cache behavior, and reporting assumptions are documented in [experiment notes](docs/experiment_notes.md).

Sample labels are inherited from the original dataset. Images have generic filenames and no embedded metadata; public redistribution permission must be established by the data owner. See the [sample documentation](examples/sample_data/README.md).
