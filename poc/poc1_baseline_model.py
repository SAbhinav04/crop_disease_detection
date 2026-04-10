#!/usr/bin/env python3
"""PoC baseline: evaluate pretrained ResNet50 on Karnataka curated dataset.

This script performs inference only (no training) and computes a crop-level
accuracy proxy by mapping ImageNet predictions to crop names.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = repo_root / "data" / "karnataka_curated" / "dataset" / "test"
    default_output_file = repo_root / "poc" / "poc1_results.txt"

    parser = argparse.ArgumentParser(description="Evaluate pretrained ResNet50 on curated test set")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="ImageFolder test directory")
    parser.add_argument("--output", type=Path, default=default_output_file, help="Result text file path")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Max number of test images to evaluate (use 0 or negative for all)",
    )
    return parser.parse_args()


def target_class_to_crop(class_name: str) -> str:
    if class_name.startswith("Corn_(maize)"):
        return "corn"
    if class_name.startswith("Grape"):
        return "grape"
    if class_name.startswith("Pepper,_bell"):
        return "pepper"
    if class_name.startswith("Potato"):
        return "potato"
    if class_name.startswith("Tomato"):
        return "tomato"
    return "unknown"


def imagenet_label_to_crop(label: str) -> str:
    l = label.lower()
    if "corn" in l or "maize" in l:
        return "corn"
    if "grape" in l:
        return "grape"
    if "bell pepper" in l or "pepper" in l:
        return "pepper"
    if "potato" in l:
        return "potato"
    if "tomato" in l:
        return "tomato"
    return "unknown"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.data_dir}")

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    dataset = datasets.ImageFolder(root=str(args.data_dir), transform=transform)

    if args.max_samples > 0:
        n = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
    total_samples = len(dataset)

    if total_samples == 0:
        raise ValueError("No images found in the dataset.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = models.resnet50(weights=weights)
    model.eval()
    model.to(device)

    class_to_idx = dataset.dataset.class_to_idx if isinstance(dataset, Subset) else dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    categories = weights.meta["categories"]

    correct = 0
    unknown_pred = 0
    start = time.time()

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(images)
            pred_indices = outputs.argmax(dim=1)

            for pred_i, target_i in zip(pred_indices.tolist(), targets.tolist()):
                pred_crop = imagenet_label_to_crop(categories[pred_i])
                true_crop = target_class_to_crop(idx_to_class[target_i])
                if pred_crop == "unknown":
                    unknown_pred += 1
                if pred_crop == true_crop:
                    correct += 1

    elapsed = time.time() - start
    accuracy = 100.0 * correct / total_samples

    result_lines = [
        "PoC 1 Baseline Evaluation (Pretrained ResNet50, no training)",
        f"Dataset path: {args.data_dir}",
        f"Samples evaluated: {total_samples}",
        f"Batch size: {args.batch_size}",
        f"Device: {device}",
        "Metric: crop-level top-1 accuracy proxy (ImageNet label -> crop mapping)",
        f"Correct predictions: {correct}",
        f"Unknown crop predictions: {unknown_pred}",
        f"Accuracy: {accuracy:.2f}%",
        f"Elapsed time: {elapsed:.2f} sec",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    for line in result_lines:
        print(line)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()