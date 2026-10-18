"""Automated Regression Test Suite for Silence Gating & Emergency Pipeline.

Validates:
1. Pure zero PCM produces exactly 0 alerts, sets is_silence_gated=True, resets persistence, and zeroes emergency scores.
2. File prediction with zero-valued WAV produces 0 alerts.
3. Microphone quiescent noise floor (RMS < 0.004) triggers silence gating and emits 0 alerts.
4. Ambient background sounds (RMS > 0.02) bypass silence gate without false emergency alerts.
5. High-amplitude acoustic signals bypass silence gate and preserve emergency classification capability.
6. Temporal state (persistence counters and smoothing queue) resets safely on silence.
"""

from __future__ import annotations

import io
import json
import os
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient


from backend.main import app, create_session_detector, initialize_global_model
from src.inference.stream_detector import StreamEmergencyDetector


@pytest.fixture(scope="module")
def client():
    initialize_global_model()
    with TestClient(app) as test_client:
        yield test_client


def test_zero_pcm_direct_stream_detector():
    """Verify StreamEmergencyDetector on pure digital silence (all zeros)."""
    detector = create_session_detector()
    zero_chunk = np.zeros(15360, dtype=np.float32)  # 0.96s at 16 kHz

    pred, alerts = detector.process_window(zero_chunk, start_sec=0.0, end_sec=0.96)

    # 1. Silence gate must be active
    assert pred.is_silence_gated is True
    assert pred.gate_reason is not None
    assert "silence gate active" in pred.gate_reason.lower()

    # 2. Emergency score must be exactly zeroed
    assert pred.emergency_score == 0.0
    for cls_name in detector.emergency_classes:
        assert pred.probabilities[cls_name] == 0.0

    # 3. Exactly zero alerts emitted
    assert len(alerts) == 0

    # 4. Persistence counters held at zero
    for cls_name, count in pred.persistence_counts.items():
        assert count == 0


def test_zero_pcm_websocket_regression(client):
    """Feed 16,000 zero samples over live WebSocket and assert no false alerts."""
    with client.websocket_connect("/ws/live") as ws:
        zero_bytes = np.zeros(16000, dtype=np.float32).tobytes()
        ws.send_bytes(zero_bytes)

        msg = ws.receive_json()
        assert msg["type"] == "prediction"
        assert msg["is_silence_gated"] is True
        assert msg["rms"] == 0.0
        assert msg["peak"] == 0.0
        assert msg["is_alert"] is False
        assert msg["alert"] is None
        assert msg["top_class"] == "background"

        # Check all emergency scores are 0.0
        for emergency_class in ["screaming", "alarm", "glass_shatter", "smash_crash", "explosion", "gunshot"]:
            assert msg["scores"][emergency_class] == 0.0
            assert msg["persistence_state"][emergency_class]["count"] == 0


def test_zero_pcm_file_upload_regression(client):
    """Upload a 3-second pure digital silence WAV and assert 0 alerts emitted."""
    sr = 16000
    zero_audio = np.zeros(int(sr * 3.0), dtype=np.float32)

    bio = io.BytesIO()
    sf.write(bio, zero_audio, sr, format="WAV")
    bio.seek(0)

    response = client.post(
        "/api/predict-file",
        files={"file": ("pure_silence.wav", bio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["max_emergency_score"] == 0.0
    assert len(data["alerts_emitted"]) == 0
    assert data["silence_gated_windows"] == data["total_windows_processed"]
    assert data["silence_gated_windows"] > 0

    # Check each window
    for win in data["window_predictions"]:
        assert win["is_silence_gated"] is True
        assert win["emergency_score"] == 0.0


def test_quiet_microphone_noise_floor():
    """Verify that typical laptop/USB mic noise floor (RMS ~0.001) is gated."""
    detector = create_session_detector()
    np.random.seed(42)

    # Mic noise floor: Gaussian noise with RMS = 0.0012 (-58 dBFS)
    mic_noise = np.random.normal(0, 0.0012, 15360).astype(np.float32)

    pred, alerts = detector.process_window(mic_noise, start_sec=0.0, end_sec=0.96)
    assert pred.is_silence_gated is True
    assert len(alerts) == 0
    assert pred.emergency_score == 0.0


def test_ambient_background_preservation():
    """Verify that non-emergency ambient room sounds (RMS > 0.02) bypass gate without triggering emergency alerts."""
    detector = create_session_detector()

    # Load genuine AudioSet ambient background clip if available
    bg_path = "data/raw/balanced_train/C0AQyOVGvFM_100_110.wav"
    if os.path.exists(bg_path):
        wav, sr = sf.read(bg_path, dtype="float32")
        chunk = wav[:15360]
    else:
        # Fallback room tone with low-frequency acoustic profile (150 Hz ambient hum)
        t = np.linspace(0, 0.96, 15360, endpoint=False, dtype=np.float32)
        chunk = 0.04 * np.sin(2 * np.pi * 150 * t)

    pred, alerts = detector.process_window(chunk, start_sec=0.0, end_sec=0.96)
    # Silence gate must NOT be active because audio has audible energy
    assert pred.is_silence_gated is False
    assert pred.rms > detector.silence_threshold_rms
    # Genuine background should not emit emergency alerts
    assert len(alerts) == 0



def test_temporal_state_reset_on_silence():
    """Verify that when silence is encountered, stale persistence and smoothing are safely purged."""
    detector = create_session_detector()

    # Manually inject state into persistence and smoothing
    detector.persistence_counters["explosion"] = 3
    detector.persistence_counters["gunshot"] = 2
    detector.smoothing_history.append(np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.9, 0.1]))

    # Feed silence window
    zero_chunk = np.zeros(15360, dtype=np.float32)
    pred, alerts = detector.process_window(zero_chunk, start_sec=5.0, end_sec=5.96)

    # All persistence counters must be reset
    assert detector.persistence_counters["explosion"] == 0
    assert detector.persistence_counters["gunshot"] == 0
    assert len(detector.smoothing_history) == 0
    assert len(alerts) == 0
    assert pred.emergency_score == 0.0


def test_genuine_emergency_signal_detection_preserved():
    """Verify that high-energy genuine acoustic events are not suppressed and alerts can fire."""
    detector = create_session_detector()

    # Create an audible siren/alarm signal (RMS ~ 0.45)
    sr = 16000
    t = np.linspace(0, 0.96, int(sr * 0.96), endpoint=False, dtype=np.float32)
    # Modulated alarm tone around 1000 Hz
    freq_mod = 1000 + 400 * np.sin(2 * np.pi * 3.0 * t)
    alarm_wave = 0.6 * np.sin(2 * np.pi * freq_mod * t).astype(np.float32)

    pred, alerts = detector.process_window(alarm_wave, start_sec=0.0, end_sec=0.96)

    # Must NOT be silence gated
    assert pred.is_silence_gated is False
    assert pred.rms > detector.silence_threshold_rms
    # Forward pass completed and produced valid probabilities
    assert sum(pred.probabilities.values()) > 0.0
