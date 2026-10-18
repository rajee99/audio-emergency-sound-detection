"""Streaming & Sliding-Window Emergency Sound Detection Engine.

Processes continuous audio or file streams in short overlapping windows (~0.96s).
Applies:
1. Frame-level feature extraction (Log-Mel Spectrogram).
2. Model probability inference.
3. Class-specific decision thresholding.
4. Temporal smoothing (rolling window median/mean).
5. Persistence filtering (consecutive detections requirement).
6. Debounce cooldown to prevent rapid duplicate alerts.
7. Clear architectural separation: Classification -> Emergency Scoring -> Alert Generation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import yaml

from src.models.classifier import CompactAudioCNN
from src.utils.audio_utils import compute_log_mel_spectrogram, load_audio, slice_into_windows


@dataclass
class WindowPrediction:
    """Prediction output for a single sliding window."""

    start_sec: float
    end_sec: float
    probabilities: Dict[str, float]
    detected_classes: List[str]
    emergency_score: float


@dataclass
class EmergencyAlert:
    """Alert emitted when an emergency acoustic event meets persistence & debounce criteria."""

    timestamp_sec: float
    end_timestamp_sec: float
    event_category: str
    confidence_score: float
    detection_threshold: float
    persistence_count: int
    alert_level: str  # 'WARNING'
    message: str


class StreamEmergencyDetector:
    """Real-time sliding window emergency sound detector."""

    def __init__(
        self,
        checkpoint_path: str = "artifacts/best_emergency_model.pt",
        config_path: str = "configs/default.yaml",
        device: Optional[str] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        if device is None:
            self.device = torch.device(
                "cuda" if (self.cfg["project"]["device"] != "cpu" and torch.cuda.is_available()) else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.class_names: List[str] = ckpt["class_names"]
        self.num_classes = len(self.class_names)

        self.model = CompactAudioCNN(
            num_classes=self.num_classes,
            n_mels=self.cfg["audio"]["n_mels"],
            embedding_dim=self.cfg["model"]["embedding_dim"],
            hidden_dim=self.cfg["model"]["hidden_dim"],
            dropout=self.cfg["model"]["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Class configuration & thresholds
        self.emergency_classes = set()
        self.thresholds: Dict[str, float] = {}
        self.persistence_requirements: Dict[str, int] = {}

        for cls_info in self.cfg["classes"]["target_classes"]:
            cname = cls_info["name"]
            if cls_info.get("is_emergency", False):
                self.emergency_classes.add(cname)
            self.thresholds[cname] = cls_info.get("default_threshold", 0.50)
            self.persistence_requirements[cname] = cls_info.get("persistence_frames", 2)

        # Override with custom/calibrated thresholds if provided
        if custom_thresholds:
            for k, v in custom_thresholds.items():
                if k in self.thresholds:
                    self.thresholds[k] = v

        self.debounce_seconds = self.cfg["detection"].get("debounce_seconds", 3.0)
        self.warning_message = self.cfg["detection"].get(
            "alert_message", "Potential acoustic warning detected — human verification required."
        )

        # State tracking
        self.persistence_counters: Dict[str, int] = {c: 0 for c in self.class_names}
        self.last_alert_time: Dict[str, float] = {c: -999.0 for c in self.class_names}
        self.smoothing_history = deque(maxlen=3)

    def reset_state(self) -> None:
        """Reset temporal state counters and smoothing history."""
        self.persistence_counters = {c: 0 for c in self.class_names}
        self.last_alert_time = {c: -999.0 for c in self.class_names}
        self.smoothing_history.clear()

    @torch.no_grad()
    def process_window(
        self, window_waveform: np.ndarray, start_sec: float, end_sec: float
    ) -> Tuple[WindowPrediction, List[EmergencyAlert]]:
        """Process a single ~0.96s window and generate window predictions and alerts."""
        sr = self.cfg["audio"]["sample_rate"]
        mel_spec = compute_log_mel_spectrogram(
            window_waveform,
            sr=sr,
            n_mels=self.cfg["audio"]["n_mels"],
            n_fft=self.cfg["audio"]["n_fft"],
            hop_length=self.cfg["audio"]["hop_length"],
            win_length=self.cfg["audio"]["win_length"],
        ).to(self.device)

        # Model forward
        probs = self.model.predict_proba(mel_spec.unsqueeze(0)).cpu().numpy()[0]
        self.smoothing_history.append(probs)

        # Smoothed probabilities across recent rolling window
        smoothed_probs = np.mean(np.array(self.smoothing_history), axis=0)

        prob_dict = {
            self.class_names[i]: float(round(smoothed_probs[i], 4))
            for i in range(self.num_classes)
        }

        # Level 1: Acoustic event classification
        detected_classes = []
        for cname, score in prob_dict.items():
            if score >= self.thresholds[cname]:
                detected_classes.append(cname)

        # Level 2: Potential-emergency scoring (max score across emergency classes only)
        emergency_scores = [
            prob_dict[c] for c in self.class_names if c in self.emergency_classes
        ]
        max_emergency_score = max(emergency_scores) if emergency_scores else 0.0

        win_pred = WindowPrediction(
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            probabilities=prob_dict,
            detected_classes=detected_classes,
            emergency_score=round(max_emergency_score, 4),
        )

        # Level 3: Alert generation with persistence and debounce filters
        alerts: List[EmergencyAlert] = []
        for cname in self.emergency_classes:
            score = prob_dict[cname]
            thresh = self.thresholds[cname]

            if score >= thresh:
                self.persistence_counters[cname] += 1
                # Check persistence
                if self.persistence_counters[cname] >= self.persistence_requirements[cname]:
                    # Check debounce cooldown
                    if (start_sec - self.last_alert_time[cname]) >= self.debounce_seconds:
                        self.last_alert_time[cname] = start_sec
                        alerts.append(
                            EmergencyAlert(
                                timestamp_sec=round(start_sec, 2),
                                end_timestamp_sec=round(end_sec, 2),
                                event_category=cname,
                                confidence_score=score,
                                detection_threshold=thresh,
                                persistence_count=self.persistence_counters[cname],
                                alert_level="WARNING",
                                message=f"{self.warning_message} [Event: {cname.upper()}]",
                            )
                        )
            else:
                # Sound subsided: reset persistence counter
                self.persistence_counters[cname] = 0

        return win_pred, alerts

    def process_file(
        self, audio_file_path: str
    ) -> Tuple[List[WindowPrediction], List[EmergencyAlert], Dict[str, any]]:
        """Run sliding-window detection across an entire audio file."""
        self.reset_state()
        sr = self.cfg["audio"]["sample_rate"]
        win_sec = self.cfg["audio"]["window_duration_sec"]
        hop_sec = self.cfg["audio"]["hop_duration_sec"]

        waveform, _ = load_audio(audio_file_path, target_sr=sr, mono=True, normalize=True)
        windows = slice_into_windows(waveform, sr=sr, window_sec=win_sec, hop_sec=hop_sec)

        predictions: List[WindowPrediction] = []
        all_alerts: List[EmergencyAlert] = []

        for start_sec, end_sec, chunk in windows:
            pred, alerts = self.process_window(chunk, start_sec, end_sec)
            predictions.append(pred)
            all_alerts.extend(alerts)

        summary = {
            "file": os.path.basename(audio_file_path),
            "total_duration_sec": round(len(waveform) / sr, 2),
            "total_windows_processed": len(windows),
            "total_alerts_emitted": len(all_alerts),
            "detected_emergency_types": list(set(a.event_category for a in all_alerts)),
            "potential_warning_flag": len(all_alerts) > 0,
        }

        return predictions, all_alerts, summary
