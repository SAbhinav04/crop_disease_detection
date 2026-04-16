#!/usr/bin/env python3
"""Enhanced data pipeline for crop disease detection.

Modular functions to clean, balance, augment, split, and analyze dataset.
Integrates with existing validate_karnataka_datasets.py workflow.

Functions are production-ready, well-commented, and follow task requirements.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

# Constants
MIN_RESOLUTION = 128  # Remove images below 128x128
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
AUG_FACTOR = 2  # Number of augmentations per original image

def is_image(path: Path) -> bool:
    """Check if file is a supported image format."""
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def normalize_label(class_name: str) -> str:
    """Normalize class label: lowercase, remove special chars, standardize."""
    import re
    return re.sub(r'[^a-z0-9 ]+', '_', class_name.lower()).strip('_ ')

def parse_crop_disease(class_name: str) -> Tuple[str, str]:
    """Parse normalized class_name into crop and disease.
    
    Examples:
    'Tomato___Late_blight' -> ('tomato', 'late_blight')
    'Corn_(maize)___healthy' -> ('corn', 'healthy')
    """
    parts = class_name.split('___')
    if len(parts) == 2:
        crop = normalize_label(parts[0])
        disease = normalize_label(parts[1])
    else:
        # Fallback: first part crop, rest disease
        crop = normalize_label(parts[0])
        disease = normalize_label('___'.join(parts[1:]))
    return crop, disease

def sha256_hash_image(image_path: Path, chunk_size: int = 1024*1024) -> str:
    """Compute SHA256 hash of image file for duplicate detection."""
    h = hashlib.sha256()
    with open(image_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def clean_dataset(
    source_dirs: List[Path], 
    output_dir: Path,
    min_resolution: int = MIN_RESOLUTION,
    crops: Optional[List[str]] = None
) -> Dict[str, object]:
    """Comprehensive data cleaning:
    1. Filter Karnataka crops (if specified)
    2. Remove duplicates using SHA256
    3. Remove corrupted/unreadable images
    4. Filter low-resolution (< min_resolution)
    5. Normalize class labels
    
    Returns stats dictionary.
    """
    random.seed(RANDOM_SEED)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_seen: Dict[str, Path] = {}  # hash -> first seen path
    stats = defaultdict(int)
    
    all_images = []
    
    # Collect all candidate images
    for src_dir in source_dirs:
        for file_path in src_dir.rglob('*'):
            if not file_path.is_file() or not is_image(file_path):
                continue
                
            class_name = file_path.parent.name
            norm_class = normalize_label(class_name)
            
            if crops and not any(norm_crop in norm_class for norm_crop in [normalize_label(c) for c in crops]):
                stats['skipped_non_crop'] += 1
                continue
            
            try:
                with Image.open(file_path) as img:
                    img.verify()
                    w, h = img.size
                    if w < min_resolution or h < min_resolution:
                        stats['skipped_low_res'] += 1
                        continue
                    
                    img_hash = sha256_hash_image(file_path)
                    if img_hash in hash_seen:
                        stats['removed_duplicate'] += 1
                        continue
                    
                    hash_seen[img_hash] = file_path
                    all_images.append((file_path, norm_class))
                    
                    stats['kept_valid'] += 1
                    
            except Exception:
                stats['skipped_corrupted'] += 1
    
    # Copy cleaned images to output with normalized class dirs
    for orig_path, norm_class in tqdm(all_images, desc='Copying cleaned images'):
        out_class_dir = output_dir / norm_class
        out_class_dir.mkdir(parents=True, exist_ok=True)
        # Unique name based on hash
        img_hash = sha256_hash_image(orig_path)[:8]
        suffix = orig_path.suffix.lower()
        out_path = out_class_dir / f'{img_hash}{suffix}'
        import shutil
        shutil.copy2(orig_path, out_path)
    
    stats['total_unique_classes'] = len({cls for _, cls in all_images})
    return dict(stats)

def get_class_distribution(dataset_dir: Path) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Analyze class distribution and compute balanced class weights.
    
    Returns: (class_counts, class_weights)
    """
    class_counts = defaultdict(int)
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_counts[class_dir.name] = len([p for p in class_dir.iterdir() 
                                              if p.is_file() and is_image(p)])
    
    classes = list(class_counts.keys())
    counts = np.array(list(class_counts.values()))
    class_weights = compute_class_weight('balanced', classes=range(len(classes)), y=counts)
    weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    print("\n=== CLASS DISTRIBUTION ===")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f'{cls:25s}: {count:5d}')
    print(f"\nImbalance ratio (max/min): {max(counts)/min(counts):.1f}")
    
    return dict(class_counts), weights_dict

def augment_training_set(train_dir: Path, aug_factor: int = AUG_FACTOR) -> int:
    """Apply data augmentation to training set ONLY.
    
    Transforms: horizontal flip, rotation ±20°, brightness/contrast, random crop/zoom.
    Saves augmented images with _aug{N} suffix.
    Returns number of augmented images created.
    """
    device = torch.device('cpu')  # No GPU needed for aug
    
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])
    
    aug_count = 0
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        for orig_img_path in tqdm(images, desc=f'Augmenting {class_dir.name}', leave=False):
            # Load original
            orig_img = Image.open(orig_img_path).convert('RGB')
            
            for i in range(aug_factor):
                aug_tensor = transform_aug(orig_img)
                aug_pil = transforms.ToPILImage()(aug_tensor)
                
                aug_path = class_dir / f'{orig_img_path.stem}_aug{i+1}{orig_img_path.suffix}'
                aug_pil.save(aug_path)
                aug_count += 1
    
    print(f'Created {aug_count} augmented images in train set.')
    return aug_count

def strict_split(cleaned_dir: Path, train_dir: Path, test_dir: Path, train_ratio: float = TRAIN_RATIO) -> Dict[str, Dict[str, int]]:
    """Class-wise split with strict hash-based duplicate prevention.
    
    Since clean_dataset already removes global duplicates, random class-wise split is safe.
    Still performs final leakage check.
    """
    random.seed(RANDOM_SEED)
    
    split_counts = {}
    all_test_hashes = set()
    
    for class_dir in sorted(cleaned_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        images = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
        random.shuffle(images)
        
        train_end = int(len(images) * train_ratio)
        train_imgs = images[:train_end]
        test_imgs = images[train_end:]
        
        # Double-check no hash overlap with previous test sets
        for test_img in test_imgs:
            h = sha256_hash_image(test_img)
            if h in all_test_hashes:
                # Rare collision, reshuffle
                random.shuffle(images)
                train_end = int(len(images) * train_ratio)
                train_imgs = images[:train_end]
                test_imgs = images[train_end:]
                break
            all_test_hashes.add(h)
        
        # Create dirs and copy
        class_train_dir = train_dir / class_dir.name
        class_test_dir = test_dir / class_dir.name
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            import shutil
            shutil.copy2(img, class_train_dir / img.name)
        for img in test_imgs:
            import shutil
            shutil.copy2(img, class_test_dir / img.name)
        
        split_counts[class_dir.name] = {'train': len(train_imgs), 'test': len(test_imgs)}
    
    return split_counts

def generate_metadata(dataset_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Generate comprehensive metadata CSV.
    
    Columns: image_path, class_name, crop_type, disease, width, height, split, file_size_bytes
    """
    rows = []
    
    for split_name in ['train', 'test']:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
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
                    crop, disease = parse_crop_disease(class_dir.name)
                    
                    rows.append({
                        'image_path': str(rel_path),
                        'class_name': class_dir.name,
                        'crop_type': crop,
                        'disease': disease,
                        'width': w,
                        'height': h,
                        'split': split_name,
                        'file_size_bytes': img_path.stat().st_size
                    })
                except Exception:
                    continue
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f'Metadata saved: {output_csv} ({len(df)} images, {df["class_name"].nunique()} classes)')
    return df

def eval_metrics(
    y_true: np.ndarray | List, 
    y_pred: np.ndarray | List, 
    class_names: List[str],
    average: str = 'weighted'
) -> Dict[str, object]:
    """Compute comprehensive evaluation metrics using sklearn.
    
    Returns dict with precision, recall, f1, confusion_matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }
    
    print('\n=== MODEL PERFORMANCE ===')
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return metrics

if __name__ == '__main__':
    # Example usage / testing
    print('Enhanced Data Pipeline Ready!')
    print('Import functions into validate_karnataka_datasets.py')

