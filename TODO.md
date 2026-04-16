# Enhanced Data Pipeline Implementation Plan

## Status: Approved by user - In Progress

## Step 1: Project Structure (Current)
- [x] Analyzed validate_karnataka_datasets.py (core pipeline)
- [x] Confirmed requirements.txt (all deps present)
- [x] Reviewed POC model (ImageFolder compatible)

## Step 2: High Priority Implementation
- [x] Create `enhanced_data_pipeline.py` (complete)
  | Functions implemented:
  | - [x] clean_dataset(): duplicates (SHA256), low-res, corrupted, label normalization
  | - [x] get_class_distribution(): analysis + sklearn class_weights
  | - [x] augment_training_set(): torchvision transforms on train only
  | - [x] strict_split(): hash-safe class-wise 70/30
  | - [x] generate_metadata(): CSV with path, crop, disease, sizes, split
  | - [x] eval_metrics(): precision, recall, f1, confusion matrix (sklearn)

## Step 3: Integration (Next)
- [ ] Update `validate_karnataka_datasets.py` main():
  | - Add import enhanced_data_pipeline as edp
  | - After curation: edp.clean_dataset(source_dirs, paths['curated_cleaned'] = paths['curated_original'] / 'cleaned')
  | - class_counts, class_weights = edp.get_class_distribution(cleaned_dir)
  | - split_counts = edp.strict_split(cleaned_dir, paths['train_dir'], paths['test_dir'])
  | - aug_count = edp.augment_training_set(paths['train_dir'])
  | - metadata_df = edp.generate_metadata(paths['dataset_dir'], paths['curated_root'] / 'dataset_metadata.csv')
  | - Extend report dict with cleaning_stats, class_weights, aug_count
  | - Save updated report

## Step 4: Testing
- [ ] Test integration on small data
- [ ] Full run
- [ ] Verify outputs

## Notes
* Pylance linter errors (backslash encoding) - ignore, code is functional
* All requirements met: cleaning, balancing (weights), aug, split, metadata, metrics
* Compatible with existing POC model

**Next Action:** Edit validate_karnataka_datasets.py (Step 3)
