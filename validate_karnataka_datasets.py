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
import numpy as np
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
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=128,
        help="Minimum image resolution used by cleaning stage.",
    )
    parser.add_argument(
        "--imbalance-warning-ratio",
        type=float,
        default=10.0,
        help="Warn when max/min class ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--imbalance-severe-ratio",
        type=float,
        default=20.0,
        help="Mark severe imbalance when max/min class ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--near-dup-hamming-threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for near-duplicate image detection.",
    )
    parser.add_argument(
        "--pre-split-near-dup-threshold",
        type=int,
        default=4,
        help="Hamming distance threshold to remove perceptual near-duplicates before split.",
    )
    parser.add_argument(
        "--near-dup-scope",
        choices=["class", "crop", "global"],
        default="class",
        help="Scope for near-duplicate matching: class, crop, or global.",
    )
    parser.add_argument(
        "--fail-on-near-dup",
        action="store_true",
        help="Exit with non-zero code if near-duplicate matches exceed threshold.",
    )
    parser.add_argument(
        "--near-dup-max-matches",
        type=int,
        default=0,
        help="Allowed near-duplicate matches across split before failure when --fail-on-near-dup is set.",
    )
    parser.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="Exit with non-zero code if any cross-split exact duplicates are found.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    curated_root = root / "data" / "karnataka_curated"
    paths = {
        "curated_root": curated_root,
        "curated_original": curated_root / "original",
        "curated_cleaned": curated_root / "cleaned",
        "curated_ready": curated_root / "ready_for_split",
        "dataset_dir": curated_root / "dataset",
        "train_dir": curated_root / "dataset" / "train",
        "test_dir": curated_root / "dataset" / "test",
        "metadata_csv": curated_root / "dataset_metadata.csv",
        "class_weights_file": curated_root / "class_weights.json",
        "training_readiness_file": curated_root / "training_readiness.json",
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


def split_train_test(cleaned_dir: Path, dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    clear_dir(dataset_dir)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    split_counts: Dict[str, Dict[str, int]] = {}

    for class_dir in sorted(cleaned_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [p for p in sorted(class_dir.iterdir()) if p.is_file() and is_image(p)]
        if not images:
            continue

        groups: Dict[int, List[Path]] = defaultdict(list)
        for img in images:
            try:
                ah = average_hash(img)
                groups[ah >> 48].append(img)
            except Exception:
                # Keep unreadable hash edge-cases in singleton groups.
                groups[hash(img.name)].append(img)

        group_items = list(groups.items())
        random.shuffle(group_items)

        target_train = int(len(images) * TRAIN_RATIO)
        train_imgs: List[Path] = []
        test_imgs: List[Path] = []

        for _, g_images in group_items:
            if len(train_imgs) < target_train:
                train_imgs.extend(g_images)
            else:
                test_imgs.extend(g_images)

        # Ensure both splits are non-empty when class has enough samples.
        if len(images) > 1 and not test_imgs:
            moved = train_imgs.pop()
            test_imgs.append(moved)
        if len(images) > 1 and not train_imgs:
            moved = test_imgs.pop()
            train_imgs.append(moved)

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


def average_hash(path: Path, hash_size: int = 8) -> int:
    """Compute an average hash (aHash) as a 64-bit integer."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        arr = np.asarray(gray, dtype=np.float32)
    mean = arr.mean()
    bits = (arr >= mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def reduce_perceptual_near_duplicates(
    cleaned_dir: Path,
    ready_dir: Path,
    threshold: int,
) -> Dict[str, object]:
    """Remove near-duplicate images within each class prior to splitting."""
    clear_dir(ready_dir)
    ready_dir.mkdir(parents=True, exist_ok=True)

    removed_total = 0
    kept_total = 0
    by_class: Dict[str, Dict[str, int]] = {}

    for class_dir in sorted(cleaned_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [p for p in sorted(class_dir.iterdir()) if p.is_file() and is_image(p)]
        out_class = ready_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        # Prefix bucketing keeps comparisons tractable while still catching many lookalikes.
        kept_hashes_by_prefix: Dict[int, List[int]] = defaultdict(list)
        class_kept = 0
        class_removed = 0

        for img_path in images:
            try:
                ah = average_hash(img_path)
            except Exception:
                class_removed += 1
                removed_total += 1
                continue

            prefix = ah >> 48
            bucket = kept_hashes_by_prefix[prefix]
            is_near_dup = any(hamming_distance(ah, existing) <= threshold for existing in bucket)

            if is_near_dup:
                class_removed += 1
                removed_total += 1
                continue

            shutil.copy2(img_path, out_class / img_path.name)
            bucket.append(ah)
            class_kept += 1
            kept_total += 1

        by_class[class_dir.name] = {
            "kept": class_kept,
            "removed": class_removed,
            "input": len(images),
        }

    return {
        "method": "average_hash_prefix_bucket",
        "hamming_threshold": threshold,
        "total_kept": kept_total,
        "total_removed": removed_total,
        "by_class": by_class,
    }


def split_key_for_scope(rel_path: str, scope: str) -> str:
    """Return grouping key for near-duplicate checks based on configured scope."""
    parts = rel_path.split("/")
    class_name = parts[1] if len(parts) > 2 else "unknown"
    if scope == "global":
        return "global"
    if scope == "class":
        return class_name
    # scope == "crop"
    return class_name.split("___")[0].lower()


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


def build_imbalance_alerts(
    class_totals: Dict[str, int], warning_ratio: float, severe_ratio: float
) -> Dict[str, object]:
    if not class_totals:
        return {
            "status": "unknown",
            "warning_ratio_threshold": warning_ratio,
            "severe_ratio_threshold": severe_ratio,
            "imbalance_ratio": None,
            "majority_class": None,
            "minority_class": None,
            "minority_under_500": [],
            "notes": ["No classes found while building imbalance alerts."],
        }

    max_class, max_count = max(class_totals.items(), key=lambda x: x[1])
    min_class, min_count = min(class_totals.items(), key=lambda x: x[1])
    ratio = max_count / min_count if min_count > 0 else None

    if ratio is None:
        status = "unknown"
    elif ratio >= severe_ratio:
        status = "severe"
    elif ratio >= warning_ratio:
        status = "warning"
    else:
        status = "ok"

    minority_under_500 = [
        {"class_name": cls, "count": count}
        for cls, count in sorted(class_totals.items(), key=lambda x: x[1])
        if count < 500
    ]

    notes = []
    if status in {"warning", "severe"}:
        notes.append("Consider weighted sampling/loss and targeted augmentation for minority classes.")
    if minority_under_500:
        notes.append("Some classes have fewer than 500 samples and may underperform.")

    return {
        "status": status,
        "warning_ratio_threshold": warning_ratio,
        "severe_ratio_threshold": severe_ratio,
        "imbalance_ratio": ratio,
        "majority_class": {"class_name": max_class, "count": max_count},
        "minority_class": {"class_name": min_class, "count": min_count},
        "minority_under_500": minority_under_500,
        "notes": notes,
    }


def compute_class_weights(class_totals: Dict[str, int]) -> Dict[str, float]:
    """Compute simple inverse-frequency weights normalized around 1.0 mean."""
    if not class_totals:
        return {}
    counts = list(class_totals.values())
    mean_count = sum(counts) / len(counts)
    weights = {
        cls: round(mean_count / count, 6) if count > 0 else 0.0
        for cls, count in class_totals.items()
    }
    return dict(sorted(weights.items()))


def build_training_readiness(
    report: Dict[str, object],
    leakage: Dict[str, object],
    min_samples_per_class: int = 500,
) -> Dict[str, object]:
    class_totals: Dict[str, int] = report.get("class_totals", {})
    low_sample_classes = [
        {"class_name": cls, "count": count}
        for cls, count in sorted(class_totals.items(), key=lambda x: x[1])
        if count < min_samples_per_class
    ]
    exact_ok = leakage.get("cross_split_duplicate_groups", 0) == 0
    near_ok = leakage.get("near_duplicate", {}).get("cross_split_near_duplicate_matches", 0) == 0
    imbalance_status = report.get("imbalance_alerts", {}).get("status", "unknown")
    balanced_enough = imbalance_status == "ok"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {
            "exact_cross_split_leakage_pass": exact_ok,
            "near_duplicate_leakage_pass": near_ok,
            "class_imbalance_ok": balanced_enough,
            "min_samples_per_class": min_samples_per_class,
            "all_classes_above_min_samples": len(low_sample_classes) == 0,
        },
        "low_sample_classes": low_sample_classes,
        "ready_for_training": exact_ok and near_ok,
        "notes": [
            "Use class_weights.json in training to mitigate imbalance.",
            "If near-duplicate leakage is non-zero, keep strict fail gate for model benchmarking.",
            "Class imbalance and low-sample classes are warnings, not hard blockers.",
        ],
    }


def leakage_check(
    dataset_dir: Path,
    near_dup_hamming_threshold: int = 5,
    near_dup_scope: str = "class",
    max_pair_checks: int = 200000,
) -> Dict[str, object]:
    hash_map: Dict[str, List[str]] = defaultdict(list)
    split_files: Dict[str, List[Path]] = {"train": [], "test": []}

    for split in ["train", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for img in split_dir.rglob("*"):
            if not img.is_file() or not is_image(img):
                continue
            h = sha1_file(img)
            hash_map[h].append(str(img.relative_to(dataset_dir)))
            split_files[split].append(img)

    duplicate_groups = [v for v in hash_map.values() if len(v) > 1]
    cross_split = []
    for group in duplicate_groups:
        involved = sorted({x.split("/")[0] for x in group})
        if len(involved) > 1:
            cross_split.append({"splits": involved, "files": group})

    # Near-duplicate check via aHash with scope-aware grouping and prefix bucketing.
    train_hashes = []
    test_hashes = []
    for p in split_files["train"]:
        try:
            rel = str(p.relative_to(dataset_dir))
            train_hashes.append((rel, average_hash(p), split_key_for_scope(rel, near_dup_scope)))
        except Exception:
            continue
    for p in split_files["test"]:
        try:
            rel = str(p.relative_to(dataset_dir))
            test_hashes.append((rel, average_hash(p), split_key_for_scope(rel, near_dup_scope)))
        except Exception:
            continue

    train_buckets: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    test_buckets: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    for rel_path, ah, scope_key in train_hashes:
        prefix = ah >> 48
        train_buckets[(scope_key, prefix)].append((rel_path, ah))
    for rel_path, ah, scope_key in test_hashes:
        prefix = ah >> 48
        test_buckets[(scope_key, prefix)].append((rel_path, ah))

    near_dups_preview = []
    near_dup_matches = 0
    pair_checks = 0
    truncated = False

    for key, t_items in train_buckets.items():
        s_items = test_buckets.get(key)
        if not s_items:
            continue
        for train_rel, train_hash in t_items:
            best_match = None
            best_dist = None
            for test_rel, test_hash in s_items:
                pair_checks += 1
                if pair_checks > max_pair_checks:
                    truncated = True
                    break
                dist = hamming_distance(train_hash, test_hash)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_match = test_rel
            if truncated:
                break
            if best_dist is not None and best_dist <= near_dup_hamming_threshold:
                near_dup_matches += 1
                if len(near_dups_preview) < 10:
                    near_dups_preview.append(
                        {
                            "train_file": train_rel,
                            "test_file": best_match,
                            "hamming_distance": best_dist,
                            "scope_key": key[0],
                            "ahash_prefix": key[1],
                        }
                    )
        if truncated:
            break

    return {
        "unique_hashes": len(hash_map),
        "duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicates_preview": cross_split[:10],
        "near_duplicate": {
            "method": "average_hash",
            "scope": near_dup_scope,
            "hamming_threshold": near_dup_hamming_threshold,
            "pair_checks": pair_checks,
            "max_pair_checks": max_pair_checks,
            "truncated": truncated,
            "cross_split_near_duplicate_matches": near_dup_matches,
            "cross_split_near_duplicates_preview": near_dups_preview,
        },
    }


def plot_validation(viz_dir: Path, report: Dict[str, object]) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    split_totals = report["split_totals"]
    splits = ["train", "test"]
    values = [split_totals[s] for s in splits]

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=splits, y=values, hue=splits, palette=["#2E86AB", "#C73E1D"], legend=False)
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
        ax = sns.barplot(x=counts, y=names, hue=names, palette="viridis", legend=False)
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
    clear_dir(paths["curated_cleaned"])
    cleaning_stats = edp.clean_dataset(
        source_dirs=[paths["curated_original"]],
        output_dir=paths["curated_cleaned"],
        min_resolution=args.min_resolution,
        crops=None,
    )
    near_dup_cleaning = reduce_perceptual_near_duplicates(
        cleaned_dir=paths["curated_cleaned"],
        ready_dir=paths["curated_ready"],
        threshold=args.pre_split_near_dup_threshold,
    )
    split_counts = split_train_test(paths["curated_ready"], paths["dataset_dir"])

    with paths["summary_file"].open("w", encoding="utf-8") as f:
        json.dump(
            {
                "curation": curation_summary,
                "cleaning": cleaning_stats,
                "pre_split_near_duplicate_cleanup": near_dup_cleaning,
                "split_counts": split_counts,
            },
            f,
            indent=2,
        )
    print(f"[save] curation summary: {paths['summary_file']}")

    report = gather_stats(paths["dataset_dir"])
    leakage = leakage_check(
        paths["dataset_dir"],
        near_dup_hamming_threshold=args.near_dup_hamming_threshold,
        near_dup_scope=args.near_dup_scope,
    )
    imbalance_alerts = build_imbalance_alerts(
        report["class_totals"],
        warning_ratio=args.imbalance_warning_ratio,
        severe_ratio=args.imbalance_severe_ratio,
    )
    report["leakage"] = leakage
    report["imbalance_alerts"] = imbalance_alerts

    class_weights = compute_class_weights(report["class_totals"])
    with paths["class_weights_file"].open("w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    _ = edp.generate_metadata(paths["dataset_dir"], paths["metadata_csv"])

    readiness = build_training_readiness(report, leakage)
    with paths["training_readiness_file"].open("w", encoding="utf-8") as f:
        json.dump(readiness, f, indent=2)

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
    print(f"Near-duplicate matches (cross-split): {leakage['near_duplicate']['cross_split_near_duplicate_matches']}")
    print(f"Class imbalance status: {imbalance_alerts['status']}")
    print(f"Pre-split near-duplicate removals: {near_dup_cleaning['total_removed']}")
    print(f"Class weights file: {paths['class_weights_file']}")
    print(f"Metadata file: {paths['metadata_csv']}")
    print(f"Training readiness file: {paths['training_readiness_file']}")
    print(f"Ready for training: {readiness['ready_for_training']}")

    if args.fail_on_leakage and leakage["cross_split_duplicate_groups"] > 0:
        raise SystemExit("Leakage check failed: cross-split exact duplicates detected.")
    if args.fail_on_near_dup and (
        leakage["near_duplicate"]["cross_split_near_duplicate_matches"] > args.near_dup_max_matches
    ):
        raise SystemExit(
            "Near-duplicate leakage check failed: "
            f"{leakage['near_duplicate']['cross_split_near_duplicate_matches']} matches exceed "
            f"allowed threshold {args.near_dup_max_matches}."
        )


if __name__ == "__main__":
    main()
