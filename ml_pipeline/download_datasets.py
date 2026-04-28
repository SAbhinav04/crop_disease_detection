#!/usr/bin/env python3
"""
Download helper for the multi-dataset Karnataka crop pipeline.

This script downloads and extracts source datasets into:
  data/raw/<dataset_name>/

By default, only PlantVillage has a ready URL. Update the remaining source URLs
in `DEFAULT_DATASETS` (or pass a JSON file with --sources-file) before running.

Example:
  python download_datasets.py
  python download_datasets.py --sources-file data/dataset_sources.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List


DEFAULT_DATASETS = [
    {
        "name": "dataset1_plantvillage",
        "url": "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
    },
    {"name": "dataset2", "url": "ADD_DATASET_2_URL_HERE"},
    {"name": "dataset3", "url": "ADD_DATASET_3_URL_HERE"},
    {"name": "dataset4", "url": "ADD_DATASET_4_URL_HERE"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract source datasets")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root folder. Defaults to script folder.",
    )
    parser.add_argument(
        "--sources-file",
        type=Path,
        default=None,
        help="Path to JSON file with dataset sources (list of {name, url}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if zip already exists.",
    )
    return parser.parse_args()


def load_sources(sources_file: Path | None) -> List[Dict[str, str]]:
    if sources_file is None:
        return DEFAULT_DATASETS

    if not sources_file.exists():
        raise FileNotFoundError(f"Sources file not found: {sources_file}")

    with sources_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("Sources JSON must be a list of objects: {name, url}")

    validated = []
    for item in payload:
        if not isinstance(item, dict) or "name" not in item or "url" not in item:
            raise ValueError("Each source must include 'name' and 'url'")
        validated.append({"name": str(item["name"]), "url": str(item["url"])})
    return validated


def download_file(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_file)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    data_dir = project_root / "data"
    downloads_dir = data_dir / "downloads"
    raw_dir = data_dir / "raw"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(args.sources_file)

    print("\n=== DATASET DOWNLOAD PHASE ===")
    print(f"Project root: {project_root}")

    completed = 0
    skipped = 0

    for source in sources:
        name = source["name"].strip()
        url = source["url"].strip()

        if "ADD_DATASET" in url or url == "":
            print(f"[skip] {name}: URL placeholder not updated")
            skipped += 1
            continue

        zip_path = downloads_dir / f"{name}.zip"
        extract_path = raw_dir / name

        try:
            if zip_path.exists() and not args.force:
                print(f"[cached] {name}: using existing zip {zip_path}")
            else:
                print(f"[download] {name}: {url}")
                download_file(url, zip_path)
                print(f"[download] {name}: saved to {zip_path}")

            if extract_path.exists() and any(extract_path.iterdir()) and not args.force:
                print(f"[cached] {name}: extraction already present at {extract_path}")
            else:
                if extract_path.exists() and args.force:
                    for item in extract_path.iterdir():
                        if item.is_dir():
                            import shutil

                            shutil.rmtree(item)
                        else:
                            item.unlink()
                print(f"[extract] {name}: extracting...")
                extract_zip(zip_path, extract_path)
                print(f"[extract] {name}: extracted to {extract_path}")

            completed += 1
        except Exception as exc:
            print(f"[error] {name}: {exc}")

    print("\n=== SUMMARY ===")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Downloads: {downloads_dir}")
    print(f"Raw data:  {raw_dir}")
    print("\nNext:")
    print("1) Ensure all 4 dataset URLs are valid")
    print("2) Run validation/curation script:")
    print("   python validate_karnataka_datasets.py --source-dirs data/raw/dataset1_plantvillage data/raw/dataset2 data/raw/dataset3 data/raw/dataset4")


if __name__ == "__main__":
    main()
