# Real ECG sample images

These 21 ECG images are a small subset of the project's existing local dataset, intended only to exercise the software pipeline.

| Split | AF | N | PAT | Total |
| --- | ---: | ---: | ---: | ---: |
| BuildModel | 5 | 5 | 5 | 15 |
| ExternalTest | 2 | 2 | 2 | 6 |

Samples retain their original dataset split and class label. Labels have not been independently clinically adjudicated. Selection used the first filenames in sorted order within each split/class, not model predictions or performance. This is not a representative evaluation cohort.

Images were visually inspected: no visible names, patient identifiers, or report headers were observed. Files were renamed to generic class-based names and re-encoded as RGB PNGs without metadata. Image dimensions and RGB pixels were preserved. No original filenames or source paths are included. Exact RGB-pixel duplicates among the 21 samples were checked and none were found; this does not establish patient independence.

`manifest.csv` lists the distributed filenames, labels, splits, and dimensions. The full dataset is not included. Public redistribution rights cannot be established from the image files alone and must be confirmed by the dataset owner before publication; no data license is asserted here.

From the repository root:

```bash
python main.py --data_root examples/sample_data/BuildModel --external_root examples/sample_data/ExternalTest --results_dir results/sample_demo --epochs 1 --batch_size 2 --num_workers 0 --val_split 0.4
```

The demo uses a seeded image-level split of the 15 development images into 9 training and 6 validation images. Class coverage in the validation subset is not guaranteed. Outputs are written separately under `results/sample_demo/`. Do not use demo metrics as evidence of clinical or research performance.
