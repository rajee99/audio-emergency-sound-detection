"""Balanced Dataset Builder for Enhanced Emergency & Gunshot Sound Detection.

Constructs a balanced, high-representation dataset across the 7 active target classes:
  0: screaming
  1: alarm
  2: glass_shatter
  3: smash_crash
  4: explosion
  5: gunshot (significantly enhanced with diverse acoustic samples)
  6: background (negative environmental audio)
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from src.dataset.acquire_audio import (
    download_and_slice_segment,
    generate_synthetic_acoustic_sample,
    get_ffmpeg_executable,
    validate_audio_file,
)
from src.dataset.filter_labels import (
    build_mid_to_class_map,
    load_config,
    parse_audioset_segments_file,
)
from src.utils.audio_utils import save_wav


def build_enhanced_balanced_dataset(
    config_path: str = "configs/default.yaml",
    target_samples_per_class: int = 25,
    output_dir: str = "data/raw/balanced_train",
    manifest_path: str = "data/manifests/balanced_train_manifest.csv",
) -> pd.DataFrame:
    """Build a balanced dataset ensuring each class, especially gunshot, has strong representation."""
    cfg = load_config(config_path)
    mid_to_classes, class_names = build_mid_to_class_map(cfg)
    num_classes = len(class_names)
    print(f"Target classes ({num_classes}): {class_names}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    ffmpeg_exe = get_ffmpeg_executable()

    raw_segs = parse_audioset_segments_file("data/metadata/balanced_train_segments.csv")
    random.seed(42)
    random.shuffle(raw_segs)

    # Class-wise segment pools
    class_pools: Dict[int, List[dict]] = {c: [] for c in range(num_classes)}
    for seg in raw_segs:
        assigned = set()
        for mid in seg["mids"]:
            if mid in mid_to_classes:
                for c in mid_to_classes[mid]:
                    assigned.add(c)
        if assigned:
            for c in assigned:
                class_pools[c].append({
                    "ytid": seg["ytid"],
                    "start_seconds": seg["start_seconds"],
                    "end_seconds": seg["end_seconds"],
                    "mids": seg["mids"],
                    "assigned": sorted(list(assigned)),
                })

    print("\n--- AudioSet Segments Pool Available ---")
    for c, cname in enumerate(class_names):
        print(f"  [{c}] {cname:<16}: {len(class_pools[c])} candidates")

    # Select balanced quota per class (prioritizing gunshot)
    selected_segments = []
    seen_clip_ids = set()

    for c in range(num_classes):
        quota = target_samples_per_class
        if class_names[c] == "gunshot":
            quota = target_samples_per_class + 10  # Enhanced quota for gunshot
        elif class_names[c] == "background":
            quota = target_samples_per_class + 15  # More background negative examples

        pool = class_pools[c]
        added = 0
        for seg in pool:
            clip_id = f"{seg['ytid']}_{int(seg['start_seconds'])}_{int(seg['end_seconds'])}"
            if clip_id not in seen_clip_ids:
                seen_clip_ids.add(clip_id)
                selected_segments.append(seg)
                added += 1
                if added >= quota:
                    break

    print(f"\nTotal unique segments selected for balanced dataset: {len(selected_segments)}")

    # Acquire and validate audio files
    records = []
    print("Verifying and acquiring audio clips...")

    for seg in tqdm(selected_segments, desc="Acquiring balanced set"):
        ytid = seg["ytid"]
        start_sec = float(seg["start_seconds"])
        end_sec = float(seg["end_seconds"])
        assigned_indices = seg["assigned"]
        assigned_names = [class_names[i] for i in assigned_indices]

        clip_id = f"{ytid}_{int(start_sec)}_{int(end_sec)}"
        clip_filename = f"{clip_id}.wav"
        clip_path = os.path.join(output_dir, clip_filename)

        status = "pending"
        err_msg = ""
        duration = 0.0

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            valid, dur, msg = validate_audio_file(clip_path)
            if valid:
                status = "available"
                duration = dur
            else:
                status = "corrupted"
                err_msg = msg
        else:
            # Download via yt-dlp
            success, msg = download_and_slice_segment(
                ytid, start_sec, end_sec, clip_path, ffmpeg_exe, timeout_sec=20
            )
            if success:
                valid, dur, vmsg = validate_audio_file(clip_path)
                if valid:
                    status = "available"
                    duration = dur
                else:
                    status = "corrupted"
                    err_msg = vmsg
            else:
                status = "synthetic_fallback"
                primary_class = assigned_names[0]
                synth_wav = generate_synthetic_acoustic_sample(primary_class, duration_sec=10.0)
                save_wav(clip_path, synth_wav, sample_rate=16000)
                duration = 10.0
                err_msg = f"Acoustic prototype fallback ({msg[:50]})"

        # Ensure mutually exclusive emergency vs background labeling:
        # If any emergency sound is present, background is 0 and target_classes contains the emergency events.
        # Only when NO emergency is present is background labeled 1.
        is_emergency = int(any(c < (num_classes - 1) for c in assigned_indices))
        if is_emergency:
            clean_indices = [idx for idx in assigned_indices if idx < (num_classes - 1)]
            clean_names = [class_names[idx] for idx in clean_indices]
        else:
            clean_indices = [num_classes - 1]
            clean_names = ["background"]

        multi_hot = [0] * num_classes
        for idx in clean_indices:
            multi_hot[idx] = 1

        primary_class = clean_names[0]

        records.append({
            "clip_id": clip_id,
            "ytid": ytid,
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "target_classes": ";".join(clean_names),
            "primary_class": primary_class,
            "multi_hot_labels": ",".join(map(str, multi_hot)),
            "is_emergency": is_emergency,
            "local_filepath": clip_path,
            "duration_sec": duration,
            "status": status,
            "notes": err_msg,
        })

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nSaved updated manifest to {manifest_path} ({len(manifest_df)} clips).")

    # Class summary
    print("\n--- Final Balanced Dataset Representation ---")
    for c, cname in enumerate(class_names):
        count = sum(c in seg["assigned"] for seg in selected_segments)
        print(f"  [{c}] {cname:<16}: {count} clips")

    return manifest_df


if __name__ == "__main__":
    build_enhanced_balanced_dataset()
