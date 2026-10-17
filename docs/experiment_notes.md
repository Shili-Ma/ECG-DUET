# Experiment and implementation notes

This document describes the current implementation and reporting behavior.
For installation and the sample run, see [the README](../README.md).

## Baselines and ablations

```bash
python scripts/train_baselines.py --epochs 40 --models resnet18 densenet121 mobilenet_v2 efficientnet_b0 se_ecgnet vit swintransformer lstm gru
python scripts/train_ablation.py --epochs 40 --configs all
```

The scripts' defaults are **5 epochs**, so set the training budget explicitly for comparisons. Equal epoch counts alone do not establish convergence or comparison fairness. Available ablation configurations are:

| Configuration | Change |
| --- | --- |
| `1_Full_Model` | Full ablation implementation |
| `2_No_Time_Branch` | Remove temporal branch |
| `3_No_Spatial_Branch` | Remove spatial branch |
| `4_No_IDCNN` | Remove IDCNN |
| `5_No_Dynamic_Fusion` | Replace dynamic fusion with a linear layer |
| `6_No_MoE` | Replace MoE with an MLP |
| `7_No_Fourier_Block` | Remove Fourier block |

**Implementation distinction:** the ablation model uses a learned full-height column encoder and a width-length sequence, while the main model uses height averaging and a 256-position sequence. Consequently, `1_Full_Model` is not architecturally identical to the main model. Comparisons against the main model are not controlled single-component ablations. This release preserves that existing behavior.

Optional offline selection utilities can promote saved per-epoch prediction tables:

```bash
python scripts/select_baseline_epoch.py --models resnet18 --epoch 4
python scripts/select_ablation_epoch.py --experiments 4_No_IDCNN --epoch 5 --no-include_main
```

Select epochs using a predefined validation-only rule. These utilities update formal predictions and summaries; a matching weight checkpoint may not exist for a manually selected epoch. The ablation selector replaces its full-model summary row with the main-model result by default; use `--no-include_main` to retain the ablation implementation's full-model row.


## Evaluation and reporting

After main-model training:

```bash
python scripts/run_robustness.py
python scripts/run_calibration.py
python scripts/run_statistics.py --n_boot 1000
python scripts/run_interpretability.py
```

Robustness and Grad-CAM require the main checkpoint, its summary, and local images/cache. Calibration and statistics consume saved prediction tables. Paired comparisons require identical sample order and labels across experiments. Maintain consistent datasets and seeds.

Then generate figures and tables:

```bash
python scripts/make_figures.py
python scripts/make_tables.py
python scripts/make_ablation_plot_options.py
```

The current figure entry point exports Figures 2–4 when their required inputs exist. A Figure 5 helper exists in the source but is not called by the entry point. Current comparison figures/tables exclude ViT and Swin; their training scripts remain available. Reporting currently prefers `baseline_summary_updated.csv` when present, otherwise `baseline_summary.csv`. The dataset-characteristics table reads the default `Dataset/` paths. Plot layouts and some axis limits are tailored to the original study and may need adjustment for new results.

For a headless Linux machine, set `MPLBACKEND=Agg` before plotting. All entry points support `--help`. When using a custom `--results_dir`, pass the same value to downstream scripts. Keep generated paths valid when moving runs between machines; saved summaries/checkpoints are not automatically relocated or converted between Windows and POSIX path conventions.


## Data and runtime details

- The temporal input is derived from image columns, not calibrated ECG voltages.
- Splitting uses a seeded image-level permutation, without stratification or patient grouping. Check patient independence and duplicates before scientific evaluation.
- Cache reuse checks tensor counts, not content hashes. Rebuild after replacing images, moving files, or changing class folders. Use unique filename stems within each class to avoid cache collisions.
- Model selection uses validation accuracy. External predictions are exported each epoch, but should not be used to select epochs or tune hyperparameters.
- Dependency ranges are not a lockfile for the original training environment. Record installed versions with `python -m pip freeze > environment-lock.txt`.
- CUDA is selected when available, otherwise CPU; Apple MPS is not currently selected. ImageNet weights are downloaded only when pretrained options are enabled.
- The synthetic smoke check covers the main model forward/backward pass, seven ablation forwards, metrics, and Excel export. It is not a full training or scientific validation test.
