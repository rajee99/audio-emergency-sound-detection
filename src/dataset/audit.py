"""Dataset Audit and Integrity Verification Module.

Generates a verified, empirical report of the project dataset:
1. Discovered candidate clips from AudioSet.
2. Successfully retrieved genuine AudioSet clips vs unavailable/corrupted clips.
3. Synthetic test fixtures isolation.
4. Class distribution across genuine splits.
5. Verification of feature tensors alignment.
6. Zero-leakage verification across unique YouTube video IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

import pandas as pd
import soundfile as sf
import yaml


def run_dataset_audit(
    config_path: str = "configs/default.yaml",
    manifest_path: str = "data/manifests/balanced_train_manifest.csv",
    output_report_path: str = "reports/dataset_audit_report.json",
) -> Dict[str, any]:
    """Execute rigorous audit of genuine vs synthetic audio, manifests, and splits."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    config_file = root_dir / config_path
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = [c["name"] for c in cfg["classes"]["target_classes"]]
    manifest_file = root_dir / manifest_path

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    df = pd.read_csv(manifest_file)
    total_clips = len(df)

    # 1. Available vs Synthetic status
    status_counts = df["status"].value_counts().to_dict()
    num_available = int(status_counts.get("available", 0))
    num_synthetic = int(status_counts.get("synthetic_fallback", 0))

    # 2. Disk file verification
    missing_files = []
    corrupted_files = []
    valid_available_clips = []

    for idx, row in df.iterrows():
        fpath = root_dir / Path(row["local_filepath"])
        if not fpath.exists():
            missing_files.append(str(fpath))
        else:
            try:
                info = sf.info(str(fpath))
                if info.duration < 0.5:
                    corrupted_files.append({"path": str(fpath), "issue": "duration_too_short"})
                elif row["status"] == "available":
                    valid_available_clips.append(row["clip_id"])
            except Exception as e:
                corrupted_files.append({"path": str(fpath), "issue": str(e)})

    # 3. Genuine Available subset
    genuine_df = df[df["status"] == "available"].copy()

    # Per-class counts in genuine audio
    genuine_per_class = {}
    for c in classes:
        genuine_per_class[c] = int(genuine_df["target_classes"].apply(lambda s: c in s.split(";")).sum())

    # 4. Verify splits
    manifests_dir = root_dir / "data/manifests"
    train_file = manifests_dir / "train_split.csv"
    val_file = manifests_dir / "val_split.csv"
    test_file = manifests_dir / "test_split.csv"

    split_stats = {}
    for s_name, s_file in [("train", train_file), ("val", val_file), ("test", test_file)]:
        if s_file.exists():
            s_df = pd.read_csv(s_file)
            split_stats[s_name] = {
                "total_samples": len(s_df),
                "genuine_samples": int((s_df["status"] == "available").sum()) if "status" in s_df.columns else len(s_df),
                "synthetic_samples": int((s_df["status"] == "synthetic_fallback").sum()) if "status" in s_df.columns else 0,
                "unique_ytids": int(s_df["ytid"].nunique()),
                "per_class": {c: int(s_df["target_classes"].apply(lambda s: c in s.split(";")).sum()) for c in classes},
            }

    # 5. Data leakage check
    leakage = {"train_val": 0, "train_test": 0, "val_test": 0}
    if train_file.exists() and val_file.exists() and test_file.exists():
        tr_y = set(pd.read_csv(train_file)["ytid"])
        va_y = set(pd.read_csv(val_file)["ytid"])
        te_y = set(pd.read_csv(test_file)["ytid"])
        leakage["train_val"] = len(tr_y.intersection(va_y))
        leakage["train_test"] = len(tr_y.intersection(te_y))
        leakage["val_test"] = len(va_y.intersection(te_y))

    # 6. Feature tensor verification
    features_dir = root_dir / "data/processed/features"
    missing_features = []
    for _, row in genuine_df.iterrows():
        feat_p = features_dir / f"{row['clip_id']}_mel.pt"
        if not feat_p.exists():
            missing_features.append(str(feat_p))

    # 7. Conflicting / Ambiguous labels check
    conflicts = []
    for _, row in df.iterrows():
        tcs = row["target_classes"].split(";")
        emergency_classes = [c for c in tcs if c in classes[:6]]
        if emergency_classes and "background" in tcs:
            conflicts.append(row["clip_id"])

    audit_report = {
        "dataset_name": "AudioSet 2017 Emergency Detection Subset",
        "configured_classes": classes,
        "num_classes": len(classes),
        "total_clips_cataloged": total_clips,
        "retrieval_status": {
            "genuine_available_audio": num_available,
            "synthetic_fallback_fixtures": num_synthetic,
            "missing_audio_files": len(missing_files),
            "corrupted_audio_files": len(corrupted_files),
        },
        "genuine_per_class_representation": genuine_per_class,
        "splits_integrity": split_stats,
        "leakage_verification": leakage,
        "feature_cache_alignment": {
            "missing_feature_tensors": len(missing_features),
            "feature_tensors_dir": str(features_dir),
        },
        "label_hygiene": {
            "conflicting_emergency_and_background_clips": len(conflicts),
        },
    }

    out_path = root_dir / output_report_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n================ DATASET AUDIT REPORT ================")
    print(f"Total cataloged clips:            {total_clips}")
    print(f"Genuine AudioSet retrieved clips: {num_available}")
    print(f"Synthetic fallback fixtures:      {num_synthetic}")
    print(f"Missing audio files on disk:      {len(missing_files)}")
    print(f"Corrupted audio files:            {len(corrupted_files)}")
    print(f"Conflicting label tags:           {len(conflicts)}")
    print(f"Missing feature tensors:          {len(missing_features)}")
    print(f"YTID Leakage (Train-Test):        {leakage['train_test']}")
    print("\nGenuine Class Representation:")
    for c, cnt in genuine_per_class.items():
        print(f"  {c:<16}: {cnt} genuine clips")
    print(f"\nAudit report saved to: {out_path}")
    print("======================================================")

    return audit_report


if __name__ == "__main__":
    run_dataset_audit()
