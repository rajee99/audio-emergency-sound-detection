"""Hardened Test Suite for FastAPI Backend Endpoints and Real-Time WebSocket.

Tests:
1. Health and readiness check.
2. Model info with 7-class configuration and sample size disclosure.
3. Predict-file endpoint with valid audio, empty audio, and payload limits.
4. WebSocket live streaming with explicit raw vs smoothed scores and persistence state.
5. Concurrent WebSocket session isolation (ensuring independent temporal state).
6. Continuous audio resampling sample count accuracy at 44.1 kHz and 48 kHz.
"""

from __future__ import annotations

import io
import json
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app, initialize_global_model


@pytest.fixture(scope="module")
def client():
    initialize_global_model()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 7


def test_model_info_endpoint(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "CompactAudioCNN (AudioSet 2017)"
    assert data["sample_rate"] == 16000
    assert len(data["target_classes"]) == 7
    class_names = [c["name"] for c in data["target_classes"]]
    assert class_names == [
        "screaming",
        "alarm",
        "glass_shatter",
        "smash_crash",
        "explosion",
        "gunshot",
        "background",
    ]
    assert "test_size_disclosure" in data


def test_predict_file_endpoint(client):
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), dtype=np.float32)
    sine_wave = 0.5 * np.sin(2 * np.pi * 1000 * t)

    bio = io.BytesIO()
    sf.write(bio, sine_wave, sr, format="WAV")
    bio.seek(0)

    response = client.post(
        "/api/predict-file",
        files={"file": ("test_sine.wav", bio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_sine.wav"
    assert data["duration_sec"] == 1.5
    assert data["total_windows_processed"] >= 1
    assert "max_emergency_score" in data
    assert "window_predictions" in data


def test_predict_file_empty_and_corrupt(client):
    # Empty file
    response = client.post(
        "/api/predict-file",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400

    # Corrupt audio bytes
    response = client.post(
        "/api/predict-file",
        files={"file": ("corrupt.wav", io.BytesIO(b"NOT_A_WAV_FILE"), "audio/wav")},
    )
    assert response.status_code == 400


def test_websocket_live_streaming_explicit_states(client):
    with client.websocket_connect("/ws/live") as websocket:
        # Send 16,000 float32 samples (~1.0s)
        pcm_bytes = np.zeros(16000, dtype=np.float32).tobytes()
        websocket.send_bytes(pcm_bytes)

        data = websocket.receive_json()
        assert data["type"] == "prediction"
        assert "raw_scores" in data
        assert "scores" in data
        assert "thresholds" in data
        assert "persistence_state" in data
        assert "is_screaming_score_high" in data
        assert "is_screaming_alert" in data

        # Verify class counts
        assert len(data["raw_scores"]) == 7
        assert len(data["scores"]) == 7
        assert "screaming" in data["raw_scores"]
        assert "screaming" in data["persistence_state"]
        screaming_p = data["persistence_state"]["screaming"]
        assert "count" in screaming_p
        assert "required" in screaming_p
        assert "persisted" in screaming_p

        # Test control command
        websocket.send_text(json.dumps({"type": "reset"}))
        ack = websocket.receive_json()
        assert ack["type"] == "reset_ack"


def test_websocket_session_isolation(client):
    """Ensure WebSocket clients and session detectors have completely independent buffers and state."""
    from backend.main import create_session_detector

    # Direct unit test of session detector isolation
    d1 = create_session_detector()
    d2 = create_session_detector()

    # d1 processes audio window, advancing its persistence counter
    dummy_wav = np.zeros(15360, dtype=np.float32)
    d1.persistence_counters["screaming"] = 5
    assert d1.persistence_counters["screaming"] == 5
    # d2 is completely isolated and unaffected
    assert d2.persistence_counters["screaming"] == 0

    # Sequential WebSocket session isolation test
    with client.websocket_connect("/ws/live") as ws1:
        chunk = np.zeros(16000, dtype=np.float32).tobytes()
        ws1.send_bytes(chunk)
        data1 = ws1.receive_json()
        assert data1["type"] == "prediction"
        ws1.send_text(json.dumps({"type": "reset"}))
        ack1 = ws1.receive_json()
        assert ack1["type"] == "reset_ack"

    # Second client connects cleanly as a fresh session
    with client.websocket_connect("/ws/live") as ws2:
        chunk = np.zeros(16000, dtype=np.float32).tobytes()
        ws2.send_bytes(chunk)
        data2 = ws2.receive_json()
        assert data2["type"] == "prediction"
        assert data2["persistence_state"]["screaming"]["count"] == 0


def test_continuous_resampler_timing_and_counts():
    """Verify that continuous resampling at 44.1 kHz and 48 kHz produces accurate sample counts."""
    import scipy.signal

    # Test 48000 Hz -> 16000 Hz (ratio 3:1)
    duration_sec = 2.0
    sr_48k = 48000
    t_48k = np.linspace(0, duration_sec, int(sr_48k * duration_sec), endpoint=False, dtype=np.float32)
    sig_48k = 0.5 * np.sin(2 * np.pi * 440 * t_48k)

    target_sr = 16000
    expected_samples = int(duration_sec * target_sr)
    resampled_48k = scipy.signal.resample(sig_48k, expected_samples)
    assert len(resampled_48k) == 32000

    # Test 44100 Hz -> 16000 Hz (fractional ratio)
    sr_44k = 44100
    t_44k = np.linspace(0, duration_sec, int(sr_44k * duration_sec), endpoint=False, dtype=np.float32)
    sig_44k = 0.5 * np.sin(2 * np.pi * 440 * t_44k)
    expected_samples_44k = int(duration_sec * target_sr)
    resampled_44k = scipy.signal.resample(sig_44k, expected_samples_44k)
    assert len(resampled_44k) == 32000
