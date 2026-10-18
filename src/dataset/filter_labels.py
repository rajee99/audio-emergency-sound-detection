"""Filter and map AudioSet segments to target emergency and background classes.

Reads official AudioSet segment CSVs and maps AudioSet MIDs into the 8 target classes:
  0: screaming
  1: siren
  2: alarm
  3: glass_shatter
  4: smash_crash
  5: explosion
  6: gunshot
  7: background

Preserves multi-label annotations and outputs filtered CSV manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Set, Tuple
import yaml
import pandas as pd


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mid_to_class_map(config: dict) -> Tuple[Dict[str, List[int]], List[str]]:
    """Build mapping from AudioSet MID -> List of target class indices."""
    mid_to_classes: Dict[str, List[int]] = {}
    class_names: List[str] = []

    for idx, cls_info in enumerate(config["classes"]["target_classes"]):
        class_names.append(cls_info["name"])
        for mid in cls_info["mids"]:
            if mid not in mid_to_classes:
                mid_to_classes[mid] = []
            mid_to_classes[mid].append(idx)

    return mid_to_classes, class_names


def parse_audioset_segments_file(filepath: str) -> List[dict]:
    """Parse raw AudioSet segment CSV file handling headers and quotes."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            ytid = parts[0]
            start_sec = float(parts[1])
            end_sec = float(parts[2])
            labels_raw = ",".join(parts[3:]).strip('" ')
            mids = [l.strip() for l in labels_raw.split(",") if l.strip()]

            rows.append({
                "ytid": ytid,
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "mids": mids,
            })
    return rows


def filter_segments(
    raw_segments: List[dict],
    mid_to_classes: Dict[str, List[int]],
    class_names: List[str],
    max_per_class: int = -1,
) -> pd.DataFrame:
    """Filter raw segments matching at least one target class and generate multi-hot vectors."""
    num_classes = len(class_names)
    matched_records = []
    class_counts = {c: 0 for c in range(num_classes)}

    for seg in raw_segments:
        assigned_classes: Set[int] = set()
        for mid in seg["mids"]:
            if mid in mid_to_classes:
                for c in mid_to_classes[mid]:
                    assigned_classes.add(c)

        if not assigned_classes:
            continue

        # In subset mode, limit sampling per class while respecting diversity
        if max_per_class > 0:
            should_include = False
            for c in assigned_classes:
                if class_counts[c] < max_per_class:
                    should_include = True
                    break
            if not should_include:
                continue

        # Multi-hot binary vector
        multi_hot = [0] * num_classes
        for c in assigned_classes:
            multi_hot[c] = 1
            class_counts[c] += 1

        matched_records.append({
            "ytid": seg["ytid"],
            "start_seconds": seg["start_seconds"],
            "end_seconds": seg["end_seconds"],
            "mids": ";".join(seg["mids"]),
            "target_classes": ";".join([class_names[c] for c in sorted(assigned_classes)]),
            "multi_hot_labels": ",".join(map(str, multi_hot)),
            "is_emergency": int(any(c < 7 for c in assigned_classes)), # Index 7 is background
            "num_target_labels": len(assigned_classes),
        })

    df = pd.DataFrame(matched_records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Filter AudioSet metadata for target emergency sounds.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--subset", action="store_true", help="Enable small subset mode for rapid testing")
    parser.add_argument("--max-per-class", type=int, default=50, help="Max clips per class in subset mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mid_to_classes, class_names = build_mid_to_class_map(cfg)
    meta_dir = cfg["paths"]["metadata_dir"]

    max_clips = args.max_per_class if args.subset else -1
    print(f"Filtering AudioSet segments (subset_mode={args.subset}, max_per_class={max_clips})...")

    # 1. Balanced Train Split
    bal_raw = parse_audioset_segments_file(os.path.join(meta_dir, "balanced_train_segments.csv"))
    bal_df = filter_segments(bal_raw, mid_to_classes, class_names, max_per_class=max_clips)
    bal_out = os.path.join(meta_dir, "filtered_balanced_train.csv")
    bal_df.to_csv(bal_out, index=False)
    print(f"Saved {len(bal_df)} filtered balanced train segments to {bal_out}")

    # 2. Evaluation Split
    eval_raw = parse_audioset_segments_file(os.path.join(meta_dir, "eval_segments.csv"))
    eval_df = filter_segments(eval_raw, mid_to_classes, class_names, max_per_class=max_clips)
    eval_out = os.path.join(meta_dir, "filtered_eval.csv")
    eval_df.to_csv(eval_out, index=False)
    print(f"Saved {len(eval_df)} filtered evaluation segments to {eval_out}")

    # Class summary statistics
    print("\n--- Filtered Target Class Distribution (Balanced Train) ---")
    for idx, cname in enumerate(class_names):
        count = bal_df["target_classes"].apply(lambda s: cname in s.split(";")).sum()
        print(f"  [{idx}] {cname:<18}: {count} clips")

    multi_label_count = (bal_df["num_target_labels"] > 1).sum()
    print(f"\nMulti-label clips count: {multi_label_count} / {len(bal_df)} ({multi_label_count / len(bal_df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
