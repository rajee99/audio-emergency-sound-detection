"""Interactive FYDP Streamlit Application: Audio-Only Emergency Sound Detector.

Features:
- WAV file upload and built-in benchmark audio samples.
- Waveform & Log-Mel Spectrogram visualization.
- Multi-label sliding-window inference with persistence & debouncing.
- Ethical disclaimer banner and human verification alert indicator.
- Interactive timeline of detected acoustic events.
- Exportable CSV/JSON detection report.
"""

from __future__ import annotations

import io
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

from src.inference.stream_detector import StreamEmergencyDetector
from src.utils.audio_utils import compute_log_mel_spectrogram, load_audio

# Page setup
st.set_page_config(
    page_title="Audio Emergency Sound Detector (FYDP)",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 5px solid #0284C7;
        padding: 0.9rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #334155;
    }
    .alert-box {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .safe-box {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_detector(config_path: str = "configs/default.yaml", checkpoint_path: str = "artifacts/best_emergency_model.pt"):
    """Load detector once and cache in memory."""
    return StreamEmergencyDetector(
        checkpoint_path=checkpoint_path, config_path=config_path
    )


def main():
    st.markdown('<div class="main-header">🚨 Audio-Only Emergency Sound Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Final Year Design Project (FYDP) | Machine Learning & Audio Signal Processing</div>',
        unsafe_allow_html=True,
    )

    # Ethical and Technical Boundary Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>🛡️ Ethical & Operational Guardrail:</b> This system operates exclusively on acoustic environmental waveforms. 
            It contains <b>no speech-to-text, keyword spotting, or spoken-word transcription</b>. 
            Detections represent suspicious acoustic event patterns requiring human verification; 
            the system does not determine criminal intent or conclude that an assault or kidnapping has occurred.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.header("⚙️ System Configuration")
    st.sidebar.markdown("**Model Version:** `CompactAudioCNN v1.0`")
    st.sidebar.markdown("**Dataset:** `Google AudioSet (2017)`")
    st.sidebar.markdown("**Sample Rate:** `16,000 Hz Mono`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ Decision Thresholds")
    screaming_th = st.sidebar.slider("Screaming Threshold", 0.10, 0.90, 0.45, 0.05)
    alarm_th = st.sidebar.slider("Alarm Threshold", 0.10, 0.90, 0.50, 0.05)
    glass_th = st.sidebar.slider("Glass Shatter Threshold", 0.10, 0.90, 0.45, 0.05)
    crash_th = st.sidebar.slider("Smash / Crash Threshold", 0.10, 0.90, 0.45, 0.05)
    explosion_th = st.sidebar.slider("Explosion Threshold", 0.10, 0.90, 0.45, 0.05)
    gunshot_th = st.sidebar.slider("Gunshot Threshold", 0.10, 0.90, 0.50, 0.05)

    custom_thresholds = {
        "screaming": screaming_th,
        "alarm": alarm_th,
        "glass_shatter": glass_th,
        "smash_crash": crash_th,
        "explosion": explosion_th,
        "gunshot": gunshot_th,
        "background": 0.50,
    }

    # Audio Input Selection
    st.subheader("1. Audio Source Selection")
    input_mode = st.radio(
        "Choose audio source:",
        ["Select Built-in Test Sample", "Upload Custom WAV File"],
        horizontal=True,
    )

    audio_bytes = None
    audio_path = None
    temp_path = "artifacts/temp_upload.wav"

    sample_dir = "data/raw/test_sample"
    built_in_samples = {}
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith(".wav"):
                built_in_samples[f] = os.path.join(sample_dir, f)

    if input_mode == "Select Built-in Test Sample":
        if built_in_samples:
            selected_name = st.selectbox("Choose sample audio:", list(built_in_samples.keys()))
            audio_path = built_in_samples[selected_name]
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        else:
            st.info("No built-in samples found in test_sample directory.")
    else:
        uploaded_file = st.file_uploader("Upload WAV audio file (16 kHz recommended):", type=["wav", "flac", "ogg"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            audio_path = temp_path

    if audio_path and os.path.exists(audio_path):
        # Explicit Audio Playback
        st.subheader("2. Audio Playback & Inspection")
        col_play, col_info = st.columns([1, 2])
        with col_play:
            st.audio(audio_bytes, format="audio/wav")
        with col_info:
            waveform, sr = load_audio(audio_path, target_sr=16000)
            st.write(f"**Sample Rate:** {sr} Hz | **Channels:** Mono | **Duration:** {len(waveform)/sr:.2f}s")

        # Visualizations (Waveform + Mel Spectrogram)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        times = np.linspace(0, len(waveform) / sr, len(waveform))
        ax1.plot(times, waveform, color="#0284c7", linewidth=0.8)
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Acoustic Waveform", fontsize=10)
        ax1.grid(True, alpha=0.2)

        mel_spec = compute_log_mel_spectrogram(waveform, sr=sr, n_mels=64)
        im = ax2.imshow(
            mel_spec.numpy(),
            aspect="auto",
            origin="lower",
            extent=[0, len(waveform) / sr, 50, 8000],
            cmap="magma",
        )
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title("Log-Mel Spectrogram (64 Mel Filters)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Run Inference
        st.subheader("3. Emergency Sound Detection Analysis")
        detector = get_detector()
        detector.thresholds.update(custom_thresholds)

        with st.spinner("Processing audio with sliding-window detector (~0.96s windows)..."):
            predictions, alerts, summary = detector.process_file(audio_path)

        # Alert Banner
        if summary["potential_warning_flag"]:
            categories_str = ", ".join(summary["detected_emergency_types"]).upper()
            st.markdown(
                f"""
                <div class="alert-box">
                    <h3 style="color: #DC2626; margin:0 0 0.5rem 0;">⚠️ Potential Acoustic Warning Detected</h3>
                    <p style="margin: 0; font-size: 1.05rem;"><b>Status:</b> Human verification required — acoustic patterns suggest potential emergency events.</p>
                    <p style="margin: 0.3rem 0 0 0;"><b>Suspected Categories:</b> <span style="color:#B91C1C; font-weight:bold;">{categories_str}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="safe-box">
                    <h3 style="color: #16A34A; margin:0 0 0.5rem 0;">✅ Normal Environmental Audio</h3>
                    <p style="margin: 0; font-size: 1.05rem;">No threshold-exceeding emergency acoustic events detected. Sound profile matches ordinary background activity.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Average Probabilities Breakdown Chart
        st.subheader("4. Category Confidence Distribution")
        prob_matrix = np.array([[p.probabilities[c] for c in detector.class_names] for p in predictions])
        mean_probs = np.mean(prob_matrix, axis=0)

        chart_df = pd.DataFrame({
            "Acoustic Class": detector.class_names,
            "Mean Confidence": mean_probs,
            "Threshold": [detector.thresholds[c] for c in detector.class_names],
        })
        st.bar_chart(chart_df.set_index("Acoustic Class")["Mean Confidence"])

        # Timeline of Detections
        st.subheader("5. Acoustic Event Timeline")
        if alerts:
            alert_rows = []
            for a in alerts:
                alert_rows.append({
                    "Start Time (s)": f"{a.timestamp_sec:.2f}",
                    "End Time (s)": f"{a.end_timestamp_sec:.2f}",
                    "Category": a.event_category.upper(),
                    "Confidence": f"{a.confidence_score * 100:.1f}%",
                    "Threshold": f"{a.detection_threshold * 100:.1f}%",
                    "Persistence Frames": a.persistence_count,
                    "Alert Level": a.alert_level,
                })
            df_alerts = pd.DataFrame(alert_rows)
            st.dataframe(df_alerts, use_container_width=True)
        else:
            st.info("No discrete emergency events triggered the persistence and threshold criteria.")

        # Export Report
        st.subheader("6. Export Event Report")
        report_data = {
            "metadata": {
                "system": "Audio-Only Emergency Sound Detector",
                "model_version": "CompactAudioCNN v1.0",
                "file_analyzed": summary["file"],
                "total_duration_sec": summary["total_duration_sec"],
                "potential_warning_flag": summary["potential_warning_flag"],
                "detected_emergency_types": summary["detected_emergency_types"],
            },
            "alerts": [a.__dict__ for a in alerts],
            "calibrated_thresholds": custom_thresholds,
        }
        json_str = json.dumps(report_data, indent=2)

        st.download_button(
            label="📥 Download Structured JSON Report",
            data=json_str,
            file_name=f"emergency_detection_report_{os.path.splitext(summary['file'])[0]}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
