"""Test suite for FastAPI backend endpoints and real-time WebSocket."""

import io
import json
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app, startup_event


@pytest.fixture(scope="module")
def client():
    # Trigger startup event to load detector & warm up model
    startup_event()
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
    assert "screaming" in class_names
    assert "alarm" in class_names
    assert "background" in class_names
    assert "siren" not in class_names


def test_predict_file_endpoint(client):
    # Generate 1.5 seconds of synthetic mono sine wave at 16000 Hz
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


def test_websocket_live_streaming(client):
    with client.websocket_connect("/ws/live") as websocket:
        # Send 16,000 samples of float32 zeros (~1 second of audio)
        pcm_bytes = np.zeros(16000, dtype=np.float32).tobytes()
        websocket.send_bytes(pcm_bytes)

        # Receive real-time prediction frame from server
        data = websocket.receive_json()
        assert data["type"] == "prediction"
        assert "scores" in data
        assert len(data["scores"]) == 7
        assert "screaming" in data["scores"]
        assert "inference_latency_ms" in data
        assert data["inference_latency_ms"] > 0
        assert "rms" in data
        assert data["top_class"] in data["scores"]

        # Test sending control reset message
        websocket.send_text(json.dumps({"type": "reset"}))
        ack = websocket.receive_json()
        assert ack["type"] == "reset_ack"
