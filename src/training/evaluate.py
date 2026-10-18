"""Comprehensive Evaluation and Per-Class Threshold Calibration Module.

Evaluates the trained emergency detector on held-out test data.
Performs:
1. Per-class threshold calibration on validation set.
2. Direct comparison between Pretrained/DSP Baseline and Trained Audio CNN.
3. Multi-label metrics: Precision, Recall, F1, Average Precision per class.
4. False alarm rate analysis on ordinary environmental background audio.
5. Saves evaluation artifact reports in JSON and Markdown.
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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.baseline import EnergySpectralBaseline
from src.models.classifier import CompactAudioCNN
from src.utils.audio_utils import load_audio


def calibrate_thresholds(
    model: torch.nn.Module,
    val_csv: str,
    num_classes: int,
    device: torch.device,
) -> Dict[int, float]:
    """Find threshold in [0.15, 0.85] maximizing F1 score per class on validation data."""
    val_df = pd.read_csv(val_csv)
    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for _, row in val_df.iterrows():
            feat_path = row["feature_path"]
            if os.path.exists(feat_path):
                mel_spec = torch.load(feat_path, weights_only=True).to(device)
                prob = model.predict_proba(mel_spec.unsqueeze(0)).cpu().numpy()[0]
                labels = [int(x) for x in row["multi_hot_labels"].split(",")]
                all_probs.append(prob)
                all_targets.append(labels)

    if not all_probs:
        return {c: 0.50 for c in range(num_classes)}

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)

    best_thresholds: Dict[int, float] = {}
    candidate_thresholds = np.linspace(0.15, 0.85, 29)

    for c in range(num_classes):
        best_f1 = -1.0
        best_thresh = 0.50
        y_true = targets_arr[:, c]

        if np.sum(y_true) == 0:
            best_thresholds[c] = 0.50
            continue

        for thresh in candidate_thresholds:
            y_pred = (probs_arr[:, c] >= thresh).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = float(round(thresh, 2))

        best_thresholds[c] = best_thresh

    return best_thresholds


def evaluate_test_set(
    model: torch.nn.Module,
    test_csv: str,
    class_names: List[str],
    thresholds: Dict[int, float],
    device: torch.device,
) -> Tuple[Dict[str, any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate trained model on held-out test split using tuned thresholds."""
    test_df = pd.read_csv(test_csv)
    all_probs = []
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for _, row in test_df.iterrows():
            feat_path = row["feature_path"]
            if os.path.exists(feat_path):
                mel_spec = torch.load(feat_path, weights_only=True).to(device)
                prob = model.predict_proba(mel_spec.unsqueeze(0)).cpu().numpy()[0]
                labels = [int(x) for x in row["multi_hot_labels"].split(",")]

                pred = np.zeros_like(prob)
                for c in range(len(class_names)):
                    thresh = thresholds.get(c, 0.50)
                    pred[c] = 1.0 if prob[c] >= thresh else 0.0

                all_probs.append(prob)
                all_preds.append(pred)
                all_targets.append(labels)

    probs_arr = np.array(all_probs)
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    # Per-class metrics
    per_class_results = {}
    for c, cname in enumerate(class_names):
        y_t = targets_arr[:, c]
        y_p = preds_arr[:, c]
        y_score = probs_arr[:, c]

        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        try:
            ap = float(average_precision_score(y_t, y_score)) if np.sum(y_t) > 0 else 0.0
        except Exception:
            ap = 0.0

        per_class_results[cname] = {
            "threshold": thresholds.get(c, 0.50),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "average_precision": round(ap, 4),
            "support": int(np.sum(y_t)),
        }

    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0))

    # False positive rate on Background audio (ordinary audio flagged as emergency)
    # Background is class index 7
    bg_mask = (targets_arr[:, 7] == 1) & (np.sum(targets_arr[:, :7], axis=1) == 0)
    if np.sum(bg_mask) > 0:
        bg_preds = preds_arr[bg_mask, :7]  # Did any emergency class trigger?
        false_alarms = np.sum(np.any(bg_preds == 1, axis=1))
        total_bg = np.sum(bg_mask)
        false_positive_rate_bg = float(false_alarms / total_bg)
    else:
        false_positive_rate_bg = 0.0
        false_alarms = 0
        total_bg = 0

    summary = {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_results,
        "background_false_positive_analysis": {
            "total_pure_background_clips": int(total_bg),
            "false_alarms": int(false_alarms),
            "false_alarm_rate": round(false_positive_rate_bg, 4),
        },
    }

    return summary, probs_arr, preds_arr, targets_arr


def evaluate_baseline_comparison(
    test_csv: str, class_names: List[str]
) -> Dict[str, any]:
    """Evaluate DSP energy baseline on test split to establish comparative baseline."""
    test_df = pd.read_csv(test_csv)
    dsp_baseline = EnergySpectralBaseline()

    all_preds = []
    all_targets = []

    for _, row in test_df.iterrows():
        wav_path = row["local_filepath"]
        if os.path.exists(wav_path):
            audio, sr = load_audio(wav_path, target_sr=16000)
            scores = dsp_baseline.predict(audio, num_classes=len(class_names))
            pred = (scores >= 0.50).astype(int)
            labels = [int(x) for x in row["multi_hot_labels"].split(",")]
            all_preds.append(pred)
            all_targets.append(labels)

    if not all_preds:
        return {"macro_f1": 0.0, "weighted_f1": 0.0}

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)

    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0))

    return {
        "baseline_name": "Acoustic Energy & Spectral Centroid DSP",
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Emergency Sound Detection System.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="artifacts/best_emergency_model.pt", help="Model checkpoint path")
    parser.add_argument("--report-dir", default="reports", help="Directory to save evaluation reports")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg["project"]["device"]
    device = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = CompactAudioCNN(
        num_classes=num_classes,
        n_mels=cfg["audio"]["n_mels"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_csv = os.path.join(cfg["paths"]["manifests_dir"], "val_split.csv")
    test_csv = os.path.join(cfg["paths"]["manifests_dir"], "test_split.csv")

    print("Calibrating per-class decision thresholds on validation data...")
    tuned_thresholds = calibrate_thresholds(model, val_csv, num_classes, device)
    print("Tuned thresholds:", tuned_thresholds)

    print("\nEvaluating on Held-Out Test Data...")
    test_results, probs, preds, targets = evaluate_test_set(
        model, test_csv, class_names, tuned_thresholds, device
    )

    print("\nEvaluating DSP Energy Baseline...")
    baseline_results = evaluate_baseline_comparison(test_csv, class_names)

    # Combine into comprehensive report
    os.makedirs(args.report_dir, exist_ok=True)
    report = {
        "model_architecture": "CompactAudioCNN (4 Conv Blocks + AdaptiveAvgPool + Linear)",
        "test_results_trained_model": test_results,
        "baseline_comparison": baseline_results,
        "calibrated_thresholds": {class_names[c]: tuned_thresholds[c] for c in range(num_classes)},
    }

    report_path = os.path.join(args.report_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Class':<20} | {'Threshold':<9} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'AP':<6}")
    print("-" * 65)
    for cname, m in test_results["per_class"].items():
        print(f"{cname:<20} | {m['threshold']:<9.2f} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1']:<7.4f} | {m['average_precision']:<6.4f}")
    print("-" * 65)
    print(f"Trained Model Macro F1:    {test_results['macro_f1']:.4f}")
    print(f"DSP Energy Baseline Macro F1: {baseline_results['macro_f1']:.4f}")
    print("=" * 65)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
