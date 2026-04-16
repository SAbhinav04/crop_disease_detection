#!/usr/bin/env python3
"""
Validation + curation pipeline for Karnataka crops.

Approach implemented:
1) Select Karnataka crops from all source datasets
2) Build curated pool
3) Split class-wise into train/test (70:30)
4) Evaluate split quality and dataset health
5) Save numerical report and visualizations

Example:
  python validate_karnataka_datasets.py --source-dirs data/raw/dataset1_plantvillage data/raw/dataset2 data/raw/dataset3 data/raw/dataset4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import enhanced_data_pipeline as edp


RANDOM_SEED = 42
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

DEFAULT_KARNATAKA_CROPS = [
    "rice",
    "maize",
    "corn",
    "cotton",
    "tomato",
    "potato",
    "chilli",
    "pepper",
    "grape",
    "sugarcane",
    "pomegranate",
    "banana",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate Karnataka crops, split 70/30, and validate dataset"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root folder. Defaults to script folder.",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="List of source dataset folders (use your 4 datasets).",
    )
    parser.add_argument(
        "--karnataka-crops",
        nargs="+",
        default=DEFAULT_KARNATAKA_CROPS,
        help="Crop names to retain from the source datasets.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualizations.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    curated_root = root / "data" / "karnataka_curated"
    paths = {
        "curated_root": curated_root,
        "curated_original": curated_root / "original",
        "curated_cleaned": curated_root / "cleaned",
        "dataset_dir": curated_root / "dataset",
        "train_dir": curated_root / "dataset" / "train",
        "test_dir": curated_root / "dataset" / "test",
        "metadata_csv": curated_root / "dataset_metadata.csv",
        "report_file": curated_root / "validation_report.json",
        "summary_file": curated_root / "curation_summary.json",
        "viz_dir": curated_root / "validation_visualizations",
    }
    for p in paths.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return paths


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_target_crop(class_name: str, crops: List[str]) -> bool:
    normalized = norm(class_name)
    primary = norm(class_name.split("___")[0])
    for crop in crops:
        c = norm(crop)
        if c in normalized or c in primary:
            return True
    return False


def unique_name(src: Path, source_tag: str) -> str:
    h = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:12]
    return f"{source_tag}_{h}{src.suffix.lower()}"


def curate_sources(
    source_dirs: List[Path], curated_original: Path, crops: List[str]
) -> Dict[str, object]:
    clear_dir(curated_original)

    selected = 0
    skipped_non_crop = 0
    skipped_corrupt = 0
    by_class: Dict[str, int] = defaultdict(int)
    by_dataset: Dict[str, int] = defaultdict(int)

    for idx, src_dir in enumerate(source_dirs, start=1):
        src = src_dir.resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source dataset folder not found: {src}")

        source_tag = f"ds{idx}"
        print(f"[curate] scanning: {src}")

        for file in src.rglob("*"):
            if not file.is_file() or not is_image(file):
                continue

            class_name = file.parent.name
            if not is_target_crop(class_name, crops):
                skipped_non_crop += 1
                continue

            try:
                with Image.open(file) as img:
                    img.verify()

                out_class = curated_original / class_name
                out_class.mkdir(parents=True, exist_ok=True)
                out_path = out_class / unique_name(file, source_tag)
                shutil.copy2(file, out_path)

                selected += 1
                by_class[class_name] += 1
                by_dataset[source_tag] += 1
            except Exception:
                skipped_corrupt += 1

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_count": len(source_dirs),
        "selected_crops": crops,
        "selected_images": selected,
        "skipped_non_crop": skipped_non_crop,
        "skipped_corrupted": skipped_corrupt,
        "by_class": dict(sorted(by_class.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def split_train_test(curated_original: Path, dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    clear_dir(dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    split_counts: Dict[str, Dict[str, int]] = {}

    for class_dir in sorted(curated_original.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted([p for p in class_dir.iterdir() if p.is_file() and is_image(p)])
        if not images:
            continue

        random.shuffle(images)
        train_end = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:train_end]
        test_imgs = images[train_end:]

        class_train = train_dir / class_dir.name
        class_test = test_dir / class_dir.name
        class_train.mkdir(parents=True, exist_ok=True)
        class_test.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, class_train / img.name)
        for img in test_imgs:
            shutil.copy2(img, class_test / img.name)

        split_counts[class_dir.name] = {"train": len(train_imgs), "test": len(test_imgs)}

    return split_counts


def sha1_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gather_stats(dataset_dir: Path) -> Dict[str, object]:
    split_totals = {"train": 0, "test": 0}
    by_split: Dict[str, Dict[str, int]] = {"train": {}, "test": {}}
    corrupted: List[str] = []

    min_w, min_h = 10**9, 10**9
    max_w, max_h = 0, 0
    sizes: List[Tuple[int, int]] = []

    for split in ["train", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            imgs = sorted([p for p in class_dir.iterdir() if p.is_file() and is_image(p)])
            by_split[split][class_dir.name] = len(imgs)

            for img in imgs:
                try:
                    with Image.open(img) as im:
                        w, h = im.size
                    sizes.append((w, h))
                    min_w, min_h = min(min_w, w), min(min_h, h)
                    max_w, max_h = max(max_w, w), max(max_h, h)
                    split_totals[split] += 1
                except Exception:
                    corrupted.append(str(img))

    classes = sorted(set(by_split["train"]) | set(by_split["test"]))
    class_totals = {c: by_split["train"].get(c, 0) + by_split["test"].get(c, 0) for c in classes}

    total = split_totals["train"] + split_totals["test"]
    expected = {"train": TRAIN_RATIO, "test": TEST_RATIO}
    actual = {
        "train": split_totals["train"] / total if total else 0.0,
        "test": split_totals["test"] / total if total else 0.0,
    }
    split_deviation = {k: actual[k] - expected[k] for k in expected}

    class_drift: Dict[str, Dict[str, float]] = {}
    max_class_drift = 0.0
    for cls in classes:
        n = class_totals[cls]
        if n == 0:
            continue
        train_r = by_split["train"].get(cls, 0) / n
        test_r = by_split["test"].get(cls, 0) / n
        drift = {"train": train_r - TRAIN_RATIO, "test": test_r - TEST_RATIO}
        class_drift[cls] = drift
        max_class_drift = max(max_class_drift, abs(drift["train"]), abs(drift["test"]))

    if sizes:
        avg_w = sum(x for x, _ in sizes) / len(sizes)
        avg_h = sum(y for _, y in sizes) / len(sizes)
        min_res = [min_w, min_h]
        max_res = [max_w, max_h]
    else:
        avg_w, avg_h = 0.0, 0.0
        min_res, max_res = None, None

    imbalance_ratio = (
        max(class_totals.values()) / min(class_totals.values())
        if class_totals and min(class_totals.values()) > 0
        else None
    )

    return {
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_policy": "70/30 train-test",
        "total_images": total,
        "total_classes": len(classes),
        "classes": classes,
        "split_totals": split_totals,
        "by_split": by_split,
        "class_totals": class_totals,
        "expected_split_ratio": expected,
        "actual_split_ratio": actual,
        "split_deviation": split_deviation,
        "class_ratio_deviation": class_drift,
        "max_abs_class_drift": max_class_drift,
        "imbalance_ratio": imbalance_ratio,
        "image_stats": {
            "min_resolution": min_res,
            "max_resolution": max_res,
            "avg_resolution": [avg_w, avg_h],
            "corrupted_images": corrupted,
        },
    }


def leakage_check(dataset_dir: Path) -> Dict[str, object]:
    hash_map: Dict[str, List[str]] = defaultdict(list)

    for split in ["train", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for img in split_dir.rglob("*"):
            if not img.is_file() or not is_image(img):
                continue
            h = sha1_file(img)
            hash_map[h].append(str(img.relative_to(dataset_dir)))

    duplicate_groups = [v for v in hash_map.values() if len(v) > 1]
    cross_split = []
    for group in duplicate_groups:
        involved = sorted({x.split("/")[0] for x in group})
        if len(involved) > 1:
            cross_split.append({"splits": involved, "files": group})

    return {
        "unique_hashes": len(hash_map),
        "duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicates_preview": cross_split[:10],
    }


def plot_validation(viz_dir: Path, report: Dict[str, object]) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    split_totals = report["split_totals"]
    splits = ["train", "test"]
    values = [split_totals[s] for s in splits]

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=splits, y=values, palette=["#2E86AB", "#C73E1D"])
    ax.set_title("Train/Test Distribution (70/30)")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(viz_dir / "train_test_distribution.png", dpi=150)
    plt.close()

    top_classes = sorted(report["class_totals"].items(), key=lambda x: x[1], reverse=True)[:20]
    if top_classes:
        names = [x[0] for x in top_classes]
        counts = [x[1] for x in top_classes]
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=counts, y=names, palette="viridis")
        ax.set_title("Top 20 Classes by Sample Count")
        ax.set_xlabel("Image Count")
        ax.set_ylabel("Class")
        plt.tight_layout()
        plt.savefig(viz_dir / "class_distribution_top20.png", dpi=150)
        plt.close()

    classes = report["classes"]
    if classes:
        matrix = [
            [report["by_split"]["train"].get(c, 0) for c in classes],
            [report["by_split"]["test"].get(c, 0) for c in classes],
        ]
        width = max(10, min(24, len(classes) * 0.35))
        plt.figure(figsize=(width, 4.2))
        ax = sns.heatmap(
            matrix,
            cmap="YlGnBu",
            annot=False,
            cbar_kws={"label": "Image Count"},
            xticklabels=classes,
            yticklabels=["train", "test"],
        )
        ax.set_title("Split-Class Matrix (Confusion Matrix Style)")
        ax.set_xlabel("Class")
        ax.set_ylabel("Split")
        plt.xticks(rotation=75, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(viz_dir / "split_class_matrix_heatmap.png", dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    source_dirs = [p.resolve() for p in args.source_dirs]

    paths = ensure_dirs(root)

    print("\n=== CURATE + VALIDATE PHASE ===")
    print(f"Project root: {root}")
    print(f"Source datasets: {len(source_dirs)}")

    curation_summary = curate_sources(source_dirs, paths["curated_original"], args.karnataka_crops)
    split_counts = split_train_test(paths["curated_original"], paths["dataset_dir"])

    with paths["summary_file"].open("w", encoding="utf-8") as f:
        json.dump({"curation": curation_summary, "split_counts": split_counts}, f, indent=2)
    print(f"[save] curation summary: {paths['summary_file']}")

    report = gather_stats(paths["dataset_dir"])
    leakage = leakage_check(paths["dataset_dir"])
    report["leakage"] = leakage

    with paths["report_file"].open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[save] validation report: {paths['report_file']}")

    if not args.skip_viz:
        plot_validation(paths["viz_dir"], report)
        print(f"[save] visualizations: {paths['viz_dir']}")

    total = report["total_images"] or 1
    train_n = report["split_totals"]["train"]
    test_n = report["split_totals"]["test"]
    train_pct = train_n / total * 100
    test_pct = test_n / total * 100

    print("\n=== SUMMARY ===")
    print(f"Total images: {report['total_images']}")
    print(f"Total classes: {report['total_classes']}")
    print(f"Train/Test: {train_n}/{test_n} ({train_pct:.2f}% / {test_pct:.2f}%)")
    print(f"Corrupted images: {len(report['image_stats']['corrupted_images'])}")
    print(f"Cross-split duplicate groups: {leakage['cross_split_duplicate_groups']}")


if __name__ == "__main__":
    main()
