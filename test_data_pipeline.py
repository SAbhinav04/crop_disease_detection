#!/usr/bin/env python3
"""Test script for data_preprocessing.py pipeline."""

import shutil
from pathlib import Path

from PIL import Image
import data_preprocessing as dp

def count_images(dir_path: Path) -> int:
    """Count image files in directory recursively."""
    return len([p for p in dir_path.rglob('*') if p.is_file() and dp.is_image(p)])

def verify_no_duplicates(dir_path: Path) -> bool:
    """Verify no duplicate images using hash."""
    hashes = set()
    for img_path in dir_path.rglob('*'):
        if not img_path.is_file() or not dp.is_image(img_path):
            continue
        h = dp.md5_hash_image(img_path)
        if h in hashes:
            return False
        hashes.add(h)
    return True

def verify_train_test_no_overlap(train_dir: Path, test_dir: Path) -> bool:
    """Verify no duplicate images between train and test."""
    train_hashes = {dp.md5_hash_image(p) for p in train_dir.rglob('*') if dp.is_image(p)}
    test_hashes = {dp.md5_hash_image(p) for p in test_dir.rglob('*') if dp.is_image(p)}
    return len(train_hashes & test_hashes) == 0

def main():
    project_root = Path(__file__).parent
    sample_data = project_root / "data" / "sample"
    cleaned_data = project_root / "data" / "cleaned"
    train_dir = project_root / "data" / "train"
    test_dir = project_root / "data" / "test"
    metadata_csv = project_root / "data" / "metadata.csv"
    
    print("=== DATA PREPROCESSING PIPELINE TEST ===\n")
    
    # Setup: Create dummy sample data if not exists
    sample_data.mkdir(parents=True, exist_ok=True)
    classes = ['tomato_healthy', 'tomato_disease', 'potato_healthy']
    for cls in classes:
        cls_dir = sample_data / cls
        cls_dir.mkdir(exist_ok=True)
        # Create dummy images with random colors (different hashes)
        import random
        color = tuple(random.randint(0,255) for _ in range(3))
        img = Image.new('RGB', (200, 200), color=color)
        img.save(cls_dir / f'{cls}.jpg')
    print("Created sample dataset with 3 classes, 1 image each")
    
    print("1. CLEANING...")
    print(f"Images before cleaning: {count_images(sample_data)}")
    
    cleaning_stats = dp.clean_dataset([sample_data], cleaned_data)
    print("✅ CLEANING SUCCESS")
    
    print(f"Images after cleaning: {count_images(cleaned_data)}")
    print(f"Cleaning stats: {cleaning_stats}")
    assert verify_no_duplicates(cleaned_data), "FAIL: Duplicates found after cleaning"
    
    print("\n2. CLASS DISTRIBUTION...")
    weights = dp.print_class_distribution(cleaned_data)
    print("✅ DISTRIBUTION ANALYSIS SUCCESS")
    
    print("\n3. SPLITTING...")
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    dp.strict_split(cleaned_data, train_dir, test_dir)
    print("✅ SPLITTING SUCCESS")
    
    print(f"Train images: {count_images(train_dir)}")
    print(f"Test images: {count_images(test_dir)}")
    assert verify_train_test_no_overlap(train_dir, test_dir), "FAIL: Overlap between train/test"
    assert train_dir.exists() and test_dir.exists(), "FAIL: Split directories not created"
    
    print("\n4. AUGMENTATION...")
    pre_aug_count = count_images(train_dir)
    aug_count = dp.augment_training_set(train_dir)
    post_aug_count = count_images(train_dir)
    
    # Check _aug files created
    aug_files = len([p for p in train_dir.rglob('*_aug*') if dp.is_image(p)])
    
    print(f"Augmented: {aug_count}, Images before: {pre_aug_count}, after: {post_aug_count}")
    print(f"AUG files created: {aug_files}")
    if pre_aug_count > 0:
        assert post_aug_count > pre_aug_count, "FAIL: No augmentation occurred"
        assert aug_files > 0, "FAIL: No _aug files created"
    print("✅ AUGMENTATION SUCCESS")
    
    print("\n5. METADATA...")
    dataset_root = project_root / "data"
    metadata_df = dp.generate_metadata(dataset_root, metadata_csv)
    print("✅ METADATA SUCCESS")
    
    # Validate metadata
    print("\nMetadata sample:")
    print(metadata_df.head())
    print(f"Columns: {list(metadata_df.columns)}")
    
    expected_cols = ['image_path', 'class_name', 'crop_type', 'disease', 'width', 'height']
    assert list(metadata_df.columns) == expected_cols, "FAIL: Wrong columns"
    
    # Verify paths exist
    missing_paths = metadata_df[~metadata_df['image_path'].apply(lambda p: (dataset_root / p).exists())]
    assert len(missing_paths) == 0, f"FAIL: {len(missing_paths)} missing image paths"
    
    print("\n🎉 ALL TESTS PASSED! Pipeline works correctly.")
    print(f"Final outputs:")
    print(f"  Cleaned: {cleaned_data}")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    print(f"  Metadata: {metadata_csv}")

if __name__ == '__main__':
    main()
