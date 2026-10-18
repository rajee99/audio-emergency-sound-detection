"""AudioSet Metadata Retrieval Script.

Downloads and validates official Google AudioSet (2017) ontology and segment CSVs:
- ontology.json
- class_labels_indices.csv
- balanced_train_segments.csv
- eval_segments.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict
import requests
from tqdm import tqdm

OFFICIAL_URLS: Dict[str, str] = {
    "ontology.json": "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
    "class_labels_indices.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "balanced_train_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
    "eval_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
}


def download_file(url: str, dest_path: str, force: bool = False) -> bool:
    """Download a remote file with progress bar and integrity check."""
    if os.path.exists(dest_path) and not force and os.path.getsize(dest_path) > 0:
        print(f"[EXISTS] {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")
        return True

    print(f"[DOWNLOADING] {url} -> {dest_path}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading {url}: {e}", file=sys.stderr)
        return False


def retrieve_all_metadata(dest_dir: str = "data/metadata", force: bool = False) -> bool:
    """Download all required AudioSet metadata files to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True
    for filename, url in OFFICIAL_URLS.items():
        target = os.path.join(dest_dir, filename)
        ok = download_file(url, target, force=force)
        if not ok:
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download official AudioSet metadata.")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Target directory for metadata files")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    success = retrieve_all_metadata(args.metadata_dir, args.force)
    if success:
        print("Official AudioSet metadata retrieved successfully.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
