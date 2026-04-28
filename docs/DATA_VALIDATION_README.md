# Data Validation Guide (Karnataka Crop Pipeline)

This document explains exactly how dataset download, curation, splitting, and validation were separated and how teammates can run the same process.

## What Was Implemented

Two separate scripts were created:

1. [download_datasets.py](download_datasets.py)
- Only downloads and extracts source datasets.
- Does not do crop filtering, splitting, training, or validation metrics.

2. [validate_karnataka_datasets.py](validate_karnataka_datasets.py)
- Curates Karnataka crops from source datasets.
- Splits class-wise into train/test with 70:30 ratio.
- Runs data validation metrics and leakage checks.
- Generates validation visualizations.

This separation is the proof that download and validation are independent stages.

## Separation Proof (Code + Behavior)

### Stage A: Download only
Run:

```bash
python3 download_datasets.py
```

Observed behavior:
- Downloads and extracts source archives to [data/downloads](data/downloads) and [data/raw](data/raw).
- It does not create curated train/test splits or validation reports.

### Stage B: Validation only (from already-downloaded data)
Run:

```bash
python3 validate_karnataka_datasets.py --source-dirs data/raw/dataset1_plantvillage
```

Observed behavior:
- Reads downloaded data.
- Creates curated split in [data/karnataka_curated/dataset](data/karnataka_curated/dataset).
- Creates numeric reports and plots.

This confirms validation is a separate step from download.

## Current Run Proof (PlantVillage-Only Run)

Executed command:

```bash
python3 validate_karnataka_datasets.py --source-dirs data/raw/dataset1_plantvillage
```

Result summary from run:
- Total images: 92104
- Total classes: 23
- Train/Test: 64465 / 27639 (69.99% / 30.01%)
- Corrupted images: 0
- Cross-split duplicate groups: 19

## Output Files Generated

Validation artifacts created:

1. [data/karnataka_curated/curation_summary.json](data/karnataka_curated/curation_summary.json)
- Curation counts by class and dataset source.

2. [data/karnataka_curated/validation_report.json](data/karnataka_curated/validation_report.json)
- Split ratio checks (expected vs actual)
- Per-class split drift
- Class imbalance statistics
- Corruption count
- Leakage/duplicate checks

3. [data/karnataka_curated/validation_visualizations](data/karnataka_curated/validation_visualizations)
- train_test_distribution.png
- class_distribution_top20.png
- split_class_matrix_heatmap.png

## Team Setup Checklist

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Download datasets:

```bash
python3 download_datasets.py
```

3. If using all 4 datasets, update URLs/placeholders in [download_datasets.py](download_datasets.py) or provide a sources JSON.

4. Run validation for available sources:

```bash
python3 validate_karnataka_datasets.py --source-dirs data/raw/dataset1_plantvillage data/raw/dataset2 data/raw/dataset3 data/raw/dataset4
```

If only PlantVillage is currently available, run with only that source dir.

## Karnataka Crop Filtering Logic

Default crop keywords used in validation script:
- rice
- maize / corn
- cotton
- tomato
- potato
- chilli / pepper
- grape
- sugarcane
- pomegranate
- banana

Matching is done from class labels (normalized text and primary crop token).

## 70:30 Split Policy

The split policy implemented is strict train/test only:
- Train ratio: 0.70
- Test ratio: 0.30

Split is performed class-wise to preserve class distribution as much as possible.

## Notes for Demo Preparation

1. For immediate demo, PlantVillage-only validation is already complete.
2. Add remaining 3 datasets when URLs are available.
3. Re-run the same validation command including all 4 source directories.
4. Use generated JSON and plots as evidence in presentation.
