"""Audio Preprocessing, Feature Extraction & Leakage-Free Dataset Splitter.

Splits data strictly by YouTube Video ID (YTID) to prevent data leakage.
Computes log-mel spectrogram features and pre-calculates positive class weights
for multi-label BCE loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from src.utils.audio_utils import compute_log_mel_spectrogram, load_audio


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_leakage_free_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, val, and test strictly grouped by YouTube ID (ytid)."""
    # GroupShuffleSplit ensuring no ytid leaks across splits
    gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(manifest_df, groups=manifest_df["ytid"]))

    train_df = manifest_df.iloc[train_idx].copy()
    temp_df = manifest_df.iloc[temp_idx].copy()

    # Split remaining into val and test
    val_rel_ratio = val_ratio / (val_ratio + test_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, train_size=val_rel_ratio, random_state=random_state)
    val_rel_idx, test_rel_idx = next(gss_val.split(temp_df, groups=temp_df["ytid"]))

    val_df = temp_df.iloc[val_rel_idx].copy()
    test_df = temp_df.iloc[test_rel_idx].copy()

    # Verify zero overlap in ytids
    train_ytids = set(train_df["ytid"])
    val_ytids = set(val_df["ytid"])
    test_ytids = set(test_df["ytid"])

    assert len(train_ytids.intersection(val_ytids)) == 0, "Data leakage detected: train and val share YTIDs!"
    assert len(train_ytids.intersection(test_ytids)) == 0, "Data leakage detected: train and test share YTIDs!"
    assert len(val_ytids.intersection(test_ytids)) == 0, "Data leakage detected: val and test share YTIDs!"

    return train_df, val_df, test_df


def extract_and_cache_features(
    df: pd.DataFrame,
    features_dir: str,
    cfg: dict,
    split_name: str,
) -> pd.DataFrame:
    """Precompute and cache log-mel spectrograms for each audio clip."""
    os.makedirs(features_dir, exist_ok=True)
    audio_cfg = cfg["audio"]
    target_sr = audio_cfg["sample_rate"]
    target_dur = audio_cfg["target_length_sec"]
    n_mels = audio_cfg["n_mels"]

    valid_records = []
    print(f"Precomputing features for {split_name} split ({len(df)} samples)...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        wav_path = row["local_filepath"]
        if not wav_path or not os.path.exists(wav_path):
            continue

        try:
            waveform, sr = load_audio(
                wav_path, target_sr=target_sr, mono=True, target_duration=target_dur
            )
            # Compute log-mel spectrogram: shape (n_mels, time_frames)
            mel_spec = compute_log_mel_spectrogram(
                waveform,
                sr=sr,
                n_mels=n_mels,
                n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"],
                win_length=audio_cfg["win_length"],
            )

            feat_filename = f"{row['clip_id']}_mel.pt"
            feat_path = os.path.join(features_dir, feat_filename)
            torch.save(mel_spec, feat_path)

            record = row.to_dict()
            record["feature_path"] = feat_path
            record["split"] = split_name
            valid_records.append(record)

        except Exception as e:
            print(f"[WARNING] Feature extraction failed for {wav_path}: {e}", file=sys.stderr)

    return pd.DataFrame(valid_records)


def compute_class_weights(df: pd.DataFrame, num_classes: int) -> List[float]:
    """Calculate positive class weights for BCEWithLogitsLoss: pos_weight = (N - P) / P."""
    labels_matrix = []
    for raw_label in df["multi_hot_labels"]:
        vec = [int(x) for x in raw_label.split(",")]
        labels_matrix.append(vec)

    labels_array = np.array(labels_matrix)
    total_samples = len(labels_array)
    pos_counts = np.sum(labels_array, axis=0)

    pos_weights = []
    for c in range(num_classes):
        p = pos_counts[c]
        if p == 0:
            pos_weights.append(1.0)
        else:
            neg = total_samples - p
            weight = float(max(1.0, neg / p))
            pos_weights.append(round(weight, 4))

    return pos_weights


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio and generate leakage-free dataset splits.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--manifest", default="data/manifests/balanced_train_manifest.csv", help="Input manifest CSV")
    parser.add_argument("--features-dir", default="data/processed/features", help="Output directory for cached features")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_classes = len(cfg["classes"]["target_classes"])
    class_names = [c["name"] for c in cfg["classes"]["target_classes"]]

    manifest_df = pd.read_csv(args.manifest)
    valid_df = manifest_df[manifest_df["status"].isin(["available", "synthetic_fallback"])].copy()
    print(f"Loaded manifest with {len(valid_df)} valid audio clips.")

    if len(valid_df) == 0:
        print("[ERROR] No valid audio clips found in manifest.", file=sys.stderr)
        sys.exit(1)

    # 1. Leakage-free splits
    train_ratio, val_ratio, test_ratio = cfg["data_pipeline"]["train_val_test_ratio"]
    train_df, val_df, test_df = prepare_leakage_free_splits(
        valid_df, train_ratio, val_ratio, test_ratio, random_state=cfg["project"]["seed"]
    )
    print(f"Splits generated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Extract and cache features
    train_processed = extract_and_cache_features(train_df, args.features_dir, cfg, "train")
    val_processed = extract_and_cache_features(val_df, args.features_dir, cfg, "val")
    test_processed = extract_and_cache_features(test_df, args.features_dir, cfg, "test")

    # 3. Save split manifests
    manifests_dir = cfg["paths"]["manifests_dir"]
    train_processed.to_csv(os.path.join(manifests_dir, "train_split.csv"), index=False)
    val_processed.to_csv(os.path.join(manifests_dir, "val_split.csv"), index=False)
    test_processed.to_csv(os.path.join(manifests_dir, "test_split.csv"), index=False)

    # 4. Calculate positive class weights
    pos_weights = compute_class_weights(train_processed, num_classes)
    stats = {
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": len(train_processed),
        "val_samples": len(val_processed),
        "test_samples": len(test_processed),
        "pos_weights": pos_weights,
    }
    stats_path = os.path.join(cfg["paths"]["processed_dir"], "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n--- Preprocessing & Split Statistics ---")
    print(f"Train samples: {len(train_processed)}")
    print(f"Val samples:   {len(val_processed)}")
    print(f"Test samples:  {len(test_processed)}")
    print("Positive class weights for BCE loss:")
    for idx, cname in enumerate(class_names):
        print(f"  [{idx}] {cname:<16}: {pos_weights[idx]}")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
