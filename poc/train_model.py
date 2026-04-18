#!/usr/bin/env python3
"""
Fine-tune ResNet50 on PlantVillage dataset using two-phase transfer learning.

Phase 1 - Head training:
    Freeze the entire ResNet50 backbone, train only the new fc layer.
    Fast convergence, prevents destroying pretrained features.

Phase 2 - Fine-tuning:
    Unfreeze layer3, layer4 and fc. Train with a much lower lr.
    Adapts high-level features to plant disease patterns.

Usage:
    python poc/train_model.py

    Or with custom data path:
    python poc/train_model.py --data-dir data/plantvillage/PlantVillage/PlantVillage

Outputs saved to poc/:
    best_model.pth     - Best model weights (by val accuracy)
    class_names.json   - Ordered list of class name strings
    training_log.csv   - Per-epoch loss and accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "data" / "plantvillage" / "PlantVillage" / "PlantVillage"
    default_out  = repo_root / "poc"

    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet50 on PlantVillage (transfer learning)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data,
        help="ImageFolder root — each subdirectory is one class",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_out,
        help="Where to save best_model.pth, class_names.json, training_log.csv",
    )
    parser.add_argument("--phase1-epochs", type=int,   default=5,   help="Epochs for head-only training (phase 1)")
    parser.add_argument("--phase2-epochs", type=int,   default=10,  help="Epochs for fine-tuning (phase 2)")
    parser.add_argument("--batch-size",    type=int,   default=32,  help="Batch size (lower to 16 if you run out of memory)")
    parser.add_argument("--num-workers",   type=int,   default=2,   help="DataLoader worker processes")
    parser.add_argument("--val-split",     type=float, default=0.2, help="Fraction of data held out for validation")
    parser.add_argument("--phase1-lr",     type=float, default=1e-3, help="Learning rate for phase 1")
    parser.add_argument("--phase2-lr",     type=float, default=1e-4, help="Learning rate for phase 2")
    parser.add_argument("--resume",        action="store_true",     help="Load best_model.pth checkpoint and continue training")
    parser.add_argument("--skip-phase1",   action="store_true",     help="Skip Phase 1 head-only training (use with --resume)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_transforms():
    """Returns (train_transform, val_transform)."""
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def set_trainable(model: nn.Module, phase: int) -> None:
    """Phase 1: only fc. Phase 2: layer3 + layer4 + fc."""
    for name, param in model.named_parameters():
        if phase == 1:
            param.requires_grad = name.startswith("fc.")
        else:
            param.requires_grad = any(
                name.startswith(p) for p in ("layer3.", "layer4.", "fc.")
            )


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """One pass over loader. Returns (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if is_train else torch.inference_mode()
    with ctx:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += inputs.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def train_phase(
    *,
    label: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    epochs: int,
    best_acc: float,
    save_path: Path,
    log: list[dict],
) -> float:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    n_params = sum(p.numel() for p in trainable)
    print(f"\n{'─'*60}")
    print(f"  {label}  |  epochs={epochs}  lr={lr}  params={n_params:,}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device)

        scheduler.step(vl_acc)
        elapsed = time.time() - t0

        marker = ""
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            marker = "  ✓ saved"

        print(
            f"  epoch {epoch:>2}/{epochs}  "
            f"train {tr_loss:.4f}/{tr_acc:.4f}  "
            f"val {vl_loss:.4f}/{vl_acc:.4f}  "
            f"({elapsed:.1f}s){marker}"
        )
        log.append({
            "phase": label, "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss":   round(vl_loss, 6), "val_acc":   round(vl_acc, 6),
        })

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Validate data dir ──────────────────────────────────────────────────
    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {args.data_dir}\n"
            "Make sure you extracted PlantVillage to:\n"
            "  data/plantvillage/PlantVillage/PlantVillage/\n"
            "Each subfolder should be one disease class (e.g. Tomato_Early_blight/)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path      = args.output_dir / "best_model.pth"
    class_names_path = args.output_dir / "class_names.json"
    log_path        = args.output_dir / "training_log.csv"

    # ── Device ────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda")  if torch.cuda.is_available()  else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"\nDevice : {device}")
    if device.type == "cpu":
        print("⚠  CPU detected — training will take 2–4 hours.")
        print("   To do a quick smoke-test first, pass: --phase1-epochs 1 --phase2-epochs 1")

    # ── Dataset ───────────────────────────────────────────────────────────
    train_tf, val_tf = make_transforms()

    full_ds      = datasets.ImageFolder(root=str(args.data_dir), transform=train_tf)
    class_names  = full_ds.classes
    num_classes  = len(class_names)

    print(f"Classes: {num_classes}  |  Total images: {len(full_ds)}")
    for i, c in enumerate(class_names):
        print(f"  [{i:>2}] {c}")

    # Save class names immediately (backend can use them even before training ends)
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"\nSaved class names → {class_names_path}")

    # 80/20 split
    val_n   = int(len(full_ds) * args.val_split)
    train_n = len(full_ds) - val_n
    train_ds, val_idx_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )

    # Give the val subset its own no-augmentation transform
    val_ds = Subset(
        datasets.ImageFolder(root=str(args.data_dir), transform=val_tf),
        val_idx_ds.indices,
    )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    print(f"Train: {train_n}  |  Val: {val_n}  |  Batch: {args.batch_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    log: list[dict] = []
    best_acc = 0.0

    # ── Resume from checkpoint ─────────────────────────────────────────────
    if args.resume:
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"\n✓ Loaded checkpoint: {model_path}")
            print("  Evaluating checkpoint accuracy on val set before resuming...")
            model.eval()
            correct = total = 0
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs).argmax(1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
            best_acc = correct / total
            print(f"  Checkpoint val accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        else:
            print("\n⚠  --resume specified but no checkpoint found — starting from scratch.")

    # ── Phase 1: head only ─────────────────────────────────────────────────
    if not args.skip_phase1:
        set_trainable(model, phase=1)
        best_acc = train_phase(
            label="Phase1-HeadOnly",
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, device=device,
            lr=args.phase1_lr, epochs=args.phase1_epochs,
            best_acc=best_acc, save_path=model_path, log=log,
        )
    else:
        print("\n⏭  Skipping Phase 1 (--skip-phase1 set).")

    # ── Phase 2: fine-tune top layers ──────────────────────────────────────
    set_trainable(model, phase=2)
    best_acc = train_phase(
        label="Phase2-FineTune",
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device,
        lr=args.phase2_lr, epochs=args.phase2_epochs,
        best_acc=best_acc, save_path=model_path, log=log,
    )

    # ── Save training log (append if resuming, overwrite otherwise) ────────
    log_mode = "a" if (args.resume and log_path.exists()) else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["phase","epoch","train_loss","train_acc","val_loss","val_acc"]
        )
        if log_mode == "w":
            writer.writeheader()
        writer.writerows(log)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"  Model saved       → {model_path}")
    print(f"  Class names saved → {class_names_path}")
    print(f"  Training log      → {log_path}")
    print(f"{'='*60}")
    print("\nNext: restart the backend — it will load best_model.pth automatically.")


if __name__ == "__main__":
    main()
