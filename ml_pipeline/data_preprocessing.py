#!/usr/bin/env python3
"""Data preprocessing pipeline for ML image datasets."""

import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from torchvision import transforms
from tqdm import tqdm

def is_image(path: Path) -> bool:
    """Check if file is a supported image format."""
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def normalize_label(class_name: str) -> str:
    """Normalize class label: lowercase, replace special chars with underscores."""
    return re.sub(r'[^a-z0-9 ]+', '_', class_name.lower()).strip('_ ')

def parse_crop_disease(class_name: str) -> Tuple[str, str]:
    """Parse class_name into crop_type and disease."""
    parts = class_name.split('_')
    crop = parts[0]
    disease = '_'.join(parts[1:]) if len(parts) > 1 else 'healthy'
    return crop, disease

def md5_hash_image(image_path: Path) -> str:
    """Compute MD5 hash of image file for duplicate detection."""
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_dataset(source_dirs: List[Path], output_dir: Path, min_res: int = 128) -> Dict[str, int]:
    """Clean dataset: remove duplicates, corrupted, low-res images, normalize labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hash_seen = {}
    stats = defaultdict(int)
    
    total_processed = 0
    
    print("Scanning images for cleaning...")
    for src_dir in source_dirs:
        for file_path in src_dir.rglob('*'):
            if not file_path.is_file() or not is_image(file_path):
                continue
                
            total_processed += 1
            
            # Check for duplicates
            img_hash = md5_hash_image(file_path)
            if img_hash in hash_seen:
                stats['duplicates_removed'] += 1
                continue
            hash_seen[img_hash] = str(file_path)
            
            # Check corruption and resolution
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    if w < min_res or h < min_res:
                        stats['low_res_removed'] += 1
                        continue
            except Exception:
                stats['corrupted_removed'] += 1
                continue
            
            # Normalize class name and copy
            class_name = normalize_label(file_path.parent.name)
            out_class_dir = output_dir / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            out_path = out_class_dir / file_path.name
            shutil.copy2(file_path, out_path)
    
    stats['total_processed'] = total_processed
    print(f"Cleaning summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return dict(stats)

def print_class_distribution(dataset_dir: Path) -> Dict[str, float]:
    """Print class distribution and return class weights."""
    class_counts = defaultdict(int)
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*')))
            class_counts[class_dir.name] = count
    
    total_images = sum(class_counts.values())
    
    print("\nClass distribution:")
    print("-" * 50)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_images * 100 if total_images > 0 else 0
        print(f"{class_name:25s}: {count:4d} ({percentage:5.1f}%)")
    
    # Compute class weights
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    total = sum(counts)
    class_weights = {cls: total / count for cls, count in zip(classes, counts)}
    
    print(f"\nTotal classes: {len(classes)}, Total images: {total_images}")
    return class_weights

def strict_split(dataset_dir: Path, train_dir: Path, test_dir: Path, train_ratio: float = 0.7) -> None:
    """Class-wise split with hash verification to prevent leakage."""
    random.seed(42)
    
    train_hash_set = set()
    
    print("Performing strict train/test split...")
    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        random.shuffle(images)
        
        train_count = int(len(images) * train_ratio)
        if len(images) > 0 and train_count == 0:
            train_count = 1
        train_images = images[:train_count]
        test_images = images[train_count:]
        
        # Verify no hash collisions
        for img in train_images:
            h = md5_hash_image(img)
            if h in train_hash_set:
                print(f"Warning: Hash collision in {class_dir.name}, reshuffling...")
                random.shuffle(images)
                train_count = int(len(images) * train_ratio)
                train_images = images[:train_count]
                test_images = images[train_count:]
                break
            train_hash_set.add(h)
        
        # Copy to train
        train_class_dir = train_dir / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        # Copy to test
        test_class_dir = test_dir / class_dir.name
        test_class_dir.mkdir(parents=True, exist_ok=True)
        for img in test_images:
            shutil.copy2(img, test_class_dir / img.name)
        
        print(f"{class_dir.name}: train={len(train_images)}, test={len(test_images)}")
    
    print("Split complete.")

def augment_training_set(train_dir: Path, aug_factor: int = 1) -> int:
    """Generate augmented images using torchvision transforms."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    
    aug_count = 0
    
    print("Applying data augmentation...")
    for class_dir in tqdm(list(train_dir.iterdir())):
        if not class_dir.is_dir():
            continue
            
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        
        for orig_img_path in images:
            orig_img = Image.open(orig_img_path).convert('RGB')
            
            for i in range(aug_factor):
                aug_img = transform(orig_img)
                aug_pil = transforms.ToPILImage()(aug_img)
                
                aug_path = class_dir / f"{orig_img_path.stem}_aug{i+1}{orig_img_path.suffix}"
                aug_pil.save(aug_path)
                aug_count += 1
    
    print(f"Generated {aug_count} augmented images.")
    return aug_count

def generate_metadata(dataset_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Generate metadata CSV with image info."""
    rows = []
    
    print("Generating metadata...")
    for split_dir in dataset_dir.iterdir():
        if not split_dir.is_dir():
            continue
            
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or not is_image(img_path):
                    continue
                    
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    
                    rel_path = img_path.relative_to(dataset_dir)
                    crop_type, disease = parse_crop_disease(class_dir.name)
                    
                    rows.append({
                        'image_path': str(rel_path),
                        'class_name': class_dir.name,
                        'crop_type': crop_type,
                        'disease': disease,
                        'width': w,
                        'height': h
                    })
                except Exception:
                    continue
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv} ({len(df)} rows).")
    return df

def eval_metrics(y_true: list, y_pred: list, class_names: List[str]):
    """Compute and print evaluation metrics."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return cm

if __name__ == '__main__':
    print("Data preprocessing module ready.")

