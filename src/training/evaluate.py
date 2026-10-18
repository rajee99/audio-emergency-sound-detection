"""Comprehensive Evaluation and Per-Class Threshold Calibration Module.

Evaluates the trained emergency detector on held-out test data.
Performs:
1. Per-class threshold calibration on validation set with guards against low-support overfitting.
2. Direct comparison between Pretrained/DSP Baseline and Trained Audio CNN.
3. Multi-label metrics: Accuracy, Precision, Recall, F1, Average Precision per class.
4. Mathematical confidence intervals (Wilson 95% CI) reflecting test sample size bounds.
5. Explicit test size disclosure (N=24 clips) for scientific integrity.
6. Saves evaluation artifact reports in JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.baseline import EnergySpectralBaseline
from src.models.classifier import CompactAudioCNN
from src.utils.audio_utils import load_audio


def wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion.

    Accurate for small sample sizes where normal approximation fails.
    """
    if n == 0:
        return 0.0, 0.0
    z = 1.95996 if confidence == 0.95 else 1.64485
    p = k / n
    denominator = 1.0 + (z**2) / n
    centre_adjusted_probability = p + (z**2) / (2.0 * n)
    adjusted_standard_deviation = math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)

    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

    return max(0.0, float(round(lower_bound, 4))), min(1.0, float(round(upper_bound, 4)))


def calibrate_thresholds(
    model: torch.nn.Module,
    val_csv: str,
    num_classes: int,
    class_names: List[str],
    device: torch.device,
    min_support_for_tuning: int = 2,
) -> Tuple[Dict[int, float], Dict[str, any]]:
    """Calibrate decision thresholds on validation data with small-sample guards."""
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
        return {c: 0.50 for c in range(num_classes)}, {"status": "val_empty_using_default"}

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)

    best_thresholds: Dict[int, float] = {}
    calibration_metadata: Dict[str, any] = {}
    candidate_thresholds = np.linspace(0.40, 0.65, 51)

    for c in range(num_classes):
        cname = class_names[c]
        y_true = targets_arr[:, c]
        pos_cnt = int(np.sum(y_true == 1))
        neg_cnt = int(np.sum(y_true == 0))

        # Check guard: If validation set has too few positive or negative examples, avoid overfitting
        if pos_cnt < min_support_for_tuning or neg_cnt < min_support_for_tuning:
            best_thresholds[c] = 0.50
            calibration_metadata[cname] = {
                "threshold": 0.50,
                "tuned": False,
                "reason": f"Insufficient validation support (positives={pos_cnt}, negatives={neg_cnt}, minimum required={min_support_for_tuning}). Retained default 0.50.",
            }
            continue

        best_score = -1.0
        best_thresh = 0.50
        for thresh in candidate_thresholds:
            y_pred = (probs_arr[:, c] >= thresh).astype(int)
            tpr = np.sum((y_true == 1) & (y_pred == 1)) / max(1, pos_cnt)
            tnr = np.sum((y_true == 0) & (y_pred == 0)) / max(1, neg_cnt)
            bal_acc = 0.5 * (tpr + tnr)
            acc = accuracy_score(y_true, y_pred)
            score = bal_acc + 0.1 * acc
            if score > best_score:
                best_score = score
                best_thresh = float(round(thresh, 2))

        best_thresholds[c] = best_thresh
        calibration_metadata[cname] = {
            "threshold": best_thresh,
            "tuned": True,
            "validation_positives": pos_cnt,
            "validation_negatives": neg_cnt,
            "validation_score": round(best_score, 4),
        }

    return best_thresholds, calibration_metadata


def evaluate_test_set(
    model: torch.nn.Module,
    test_csv: str,
    class_names: List[str],
    thresholds: Dict[int, float],
    device: torch.device,
) -> Tuple[Dict[str, any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate trained model on held-out test split using tuned thresholds and compute Wilson CIs."""
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
    total_test_samples = len(targets_arr)

    # Per-class metrics
    per_class_results = {}
    for c, cname in enumerate(class_names):
        y_t = targets_arr[:, c]
        y_p = preds_arr[:, c]
        y_score = probs_arr[:, c]

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        support = int(np.sum(y_t))

        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        try:
            ap = float(average_precision_score(y_t, y_score)) if support > 0 else 0.0
        except Exception:
            ap = 0.0

        # Confidence intervals (95% Wilson)
        acc_ci = wilson_score_interval(tp + tn, total_test_samples, 0.95)
        rec_ci = wilson_score_interval(tp, support, 0.95) if support > 0 else (0.0, 0.0)
        prec_ci = wilson_score_interval(tp, tp + fp, 0.95) if (tp + fp) > 0 else (0.0, 0.0)

        per_class_results[cname] = {
            "threshold": thresholds.get(c, 0.50),
            "support": support,
            "total_samples": total_test_samples,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "accuracy": round(acc, 4),
            "accuracy_95ci": list(acc_ci),
            "precision": round(prec, 4),
            "precision_95ci": list(prec_ci),
            "recall": round(rec, 4),
            "recall_95ci": list(rec_ci),
            "f1": round(f1, 4),
            "average_precision": round(ap, 4),
        }

    macro_acc = float(np.mean([per_class_results[cname]["accuracy"] for cname in class_names]))
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds_arr, average="weighted", zero_division=0))

    # False positive rate on Background audio (pure background audio flagged as emergency)
    bg_idx = class_names.index("background") if "background" in class_names else len(class_names) - 1
    bg_mask = (targets_arr[:, bg_idx] == 1) & (np.sum(targets_arr[:, :bg_idx], axis=1) == 0)
    if np.sum(bg_mask) > 0:
        bg_preds = preds_arr[bg_mask, :bg_idx]  # Did any emergency class trigger?
        false_alarms = int(np.sum(np.any(bg_preds == 1, axis=1)))
        total_bg = int(np.sum(bg_mask))
        false_positive_rate_bg = float(false_alarms / total_bg)
        fpr_ci = list(wilson_score_interval(false_alarms, total_bg, 0.95))
    else:
        false_positive_rate_bg = 0.0
        false_alarms = 0
        total_bg = 0
        fpr_ci = [0.0, 0.0]

    summary = {
        "macro_accuracy": round(macro_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_test_samples": total_test_samples,
        "per_class": per_class_results,
        "background_false_positive_analysis": {
            "total_pure_background_clips": total_bg,
            "false_alarms": false_alarms,
            "false_alarm_rate": round(false_positive_rate_bg, 4),
            "false_alarm_rate_95ci": fpr_ci,
        },
        "sample_size_disclosure": (
            f"The held-out test split currently contains N={total_test_samples} clips. "
            "Because this sample size is limited, individual class metrics (especially for rare classes) "
            "exhibit wide 95% Wilson confidence intervals. Results should be interpreted as proof-of-concept "
            "engineering indicators and threshold calibration validation, rather than conclusive population claims."
        ),
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
    tuned_thresholds, calib_meta = calibrate_thresholds(
        model, val_csv, num_classes, class_names, device, min_support_for_tuning=2
    )
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
        "configured_classes": class_names,
        "num_classes": num_classes,
        "calibration_status": calib_meta,
        "calibrated_thresholds": {class_names[c]: tuned_thresholds[c] for c in range(num_classes)},
        "test_results_trained_model": test_results,
        "baseline_comparison": baseline_results,
        "test_size_disclosure": test_results["sample_size_disclosure"],
    }

    report_path = os.path.join(args.report_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Class':<16} | {'Thresh':<6} | {'Supp':<4} | {'Prec (95% CI)':<17} | {'Recall (95% CI)':<17} | {'F1':<6} | {'AP':<6}")
    print("-" * 80)
    for cname, m in test_results["per_class"].items():
        prec_str = f"{m['precision']:.2f} [{m['precision_95ci'][0]:.2f},{m['precision_95ci'][1]:.2f}]"
        rec_str = f"{m['recall']:.2f} [{m['recall_95ci'][0]:.2f},{m['recall_95ci'][1]:.2f}]"
        print(f"{cname:<16} | {m['threshold']:<6.2f} | {m['support']:<4} | {prec_str:<17} | {rec_str:<17} | {m['f1']:<6.2f} | {m['average_precision']:<6.2f}")
    print("-" * 80)
    print(f"Trained Model Macro F1:       {test_results['macro_f1']:.4f}")
    print(f"DSP Energy Baseline Macro F1:    {baseline_results['macro_f1']:.4f}")
    print(f"Pure Background False Alarm Rate: {test_results['background_false_positive_analysis']['false_alarm_rate']:.2%}")
    print("=" * 80)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
