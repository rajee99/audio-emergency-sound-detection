"""Audio File Prediction CLI for Emergency Sound Detection.

Performs sliding-window inference on any WAV audio file.
Outputs detected acoustic events, confidence timeline, and potential emergency warnings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

from src.inference.stream_detector import StreamEmergencyDetector


def analyze_audio_file(
    audio_path: str,
    checkpoint_path: str = "artifacts/best_emergency_model.pt",
    config_path: str = "configs/default.yaml",
    output_report_path: str = "",
) -> None:
    """Analyze a single audio file and display structured prediction results."""
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Emergency Sound Detector from {checkpoint_path}...")
    detector = StreamEmergencyDetector(
        checkpoint_path=checkpoint_path, config_path=config_path
    )

    print(f"Processing audio: {audio_path}...")
    predictions, alerts, summary = detector.process_file(audio_path)

    print("\n" + "=" * 70)
    print("           ACOUSTIC EMERGENCY SOUND DETECTION REPORT")
    print("=" * 70)
    print(f"Input Audio File:        {summary['file']}")
    print(f"Duration:                {summary['total_duration_sec']} seconds")
    print(f"Sliding Windows Analyzed:{summary['total_windows_processed']}")
    print(f"Alerts Emitted:          {summary['total_alerts_emitted']}")

    if summary["potential_warning_flag"]:
        print("\n>>> STATUS: [POTENTIAL ACOUSTIC WARNING DETECTED]")
        print(">>> NOTICE: Human verification required — audio pattern is suspicious.")
        print(">>> SUSPECTED CATEGORIES:", ", ".join(summary["detected_emergency_types"]).upper())
    else:
        print("\n>>> STATUS: [NORMAL / NO EMERGENCY ACOUSTIC EVENT DETECTED]")
        print(">>> Ordinary environmental background sound pattern.")

    print("\n--- Event Detection Timeline ---")
    if alerts:
        alert_rows = []
        for a in alerts:
            alert_rows.append([
                f"{a.timestamp_sec:.2f}s - {a.end_timestamp_sec:.2f}s",
                a.event_category.upper(),
                f"{a.confidence_score * 100:.1f}%",
                f"{a.detection_threshold * 100:.1f}%",
                a.persistence_count,
                a.alert_level,
            ])
        print(pd.DataFrame(alert_rows, columns=["Time Interval", "Category", "Confidence", "Threshold", "Persistence", "Level"]).to_string(index=False))
    else:
        print("No threshold-exceeding emergency acoustic events detected.")

    print("=" * 70)

    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        report_data = {
            "summary": summary,
            "alerts": [a.__dict__ for a in alerts],
            "timeline": [p.__dict__ for p in predictions],
        }
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Detailed detection report saved to {output_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze an audio file for potential emergency sounds.")
    parser.add_argument("--audio", required=True, help="Path to input audio file (WAV)")
    parser.add_argument("--checkpoint", default="artifacts/best_emergency_model.pt", help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--output", default="", help="Optional JSON report output file")
    args = parser.parse_args()

    analyze_audio_file(args.audio, args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
