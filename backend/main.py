"""FastAPI Backend Server for Real-Time Emergency Sound Detection.

Provides:
1. REST endpoints: /api/health, /api/model-info, /api/predict-file
2. Real-time WebSocket: /ws/live for streaming microphone PCM audio
3. Model loading, feature extraction, temporal smoothing, and alert emission
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import torch
import yaml

from src.inference.stream_detector import StreamEmergencyDetector
from src.utils.audio_utils import compute_log_mel_spectrogram, load_audio, slice_into_windows

app = FastAPI(
    title="Audio Emergency Sound Detector API",
    description="Real-Time Audio-Only Environmental Emergency Sound Detection using AudioSet 2017",
    version="2.0.0",
)

# Enable CORS for browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[StreamEmergencyDetector] = None
model_metadata: Dict[str, any] = {}


def load_calibrated_thresholds() -> Dict[str, float]:
    """Load calibrated thresholds from evaluation report if present."""
    rep_path = ROOT_DIR / "reports/evaluation_report.json"
    if rep_path.exists():
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("calibrated_thresholds", {})
        except Exception as e:
            print(f"[WARNING] Could not load calibrated thresholds: {e}")
    return {}


@app.on_event("startup")
def startup_event():
    """Initialize detector, load checkpoint, and warm up model on server startup."""
    global detector, model_metadata
    cfg_path = str(ROOT_DIR / "configs/default.yaml")
    ckpt_path = str(ROOT_DIR / "artifacts/best_emergency_model.pt")

    thresholds = load_calibrated_thresholds()
    print(f"[STARTUP] Initializing detector with checkpoint: {ckpt_path}")
    print(f"[STARTUP] Loaded calibrated thresholds: {thresholds}")

    detector = StreamEmergencyDetector(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        custom_thresholds=thresholds,
    )

    # Warm up forward pass with dummy tensor
    dummy_wav = np.zeros(int(16000 * 0.96), dtype=np.float32)
    t0 = time.perf_counter()
    detector.process_window(dummy_wav, 0.0, 0.96)
    warmup_time = (time.perf_counter() - t0) * 1000
    print(f"[STARTUP] Model warmup successful (Latency: {warmup_time:.2f} ms)")

    # Cache metadata
    eval_path = ROOT_DIR / "reports/evaluation_report.json"
    eval_metrics = {}
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    model_metadata = {
        "model_name": "CompactAudioCNN (AudioSet 2017)",
        "version": "2.0",
        "sample_rate": detector.cfg["audio"]["sample_rate"],
        "target_classes": [
            {
                "name": c["name"],
                "display_name": c.get("display_name", c["name"]),
                "is_emergency": c.get("is_emergency", False),
                "threshold": detector.thresholds.get(c["name"], 0.50),
            }
            for c in detector.cfg["classes"]["target_classes"]
        ],
        "class_names": detector.class_names,
        "window_duration_sec": detector.cfg["audio"].get("window_duration_sec", 0.96),
        "hop_duration_sec": detector.cfg["audio"].get("hop_duration_sec", 0.25),
        "evaluation_metrics": eval_metrics.get("test_results_trained_model", {}),
    }


@app.get("/api/health")
def health_check():
    """Health status and model readiness check."""
    return {
        "status": "online",
        "model_loaded": detector is not None,
        "device": str(detector.device) if detector else "uninitialized",
        "num_classes": len(detector.class_names) if detector else 0,
        "timestamp": time.time(),
    }


@app.get("/api/model-info")
def get_model_info():
    """Return model architecture, configured classes, thresholds, and performance metrics."""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized.")
    return model_metadata


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """Analyze an uploaded WAV file, returning sliding-window predictions and alerts."""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized.")

    contents = await file.read()
    try:
        with io.BytesIO(contents) as bio:
            audio, sr = sf.read(bio)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # Mono
            audio = audio.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio format: {e}")

    target_sr = detector.cfg["audio"]["sample_rate"]
    if sr != target_sr:
        import scipy.signal
        num_target_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_target_samples).astype(np.float32)

    duration_sec = float(len(audio) / target_sr)
    window_sec = float(detector.cfg["audio"].get("window_duration_sec", 0.96))
    hop_sec = float(detector.cfg["audio"].get("hop_duration_sec", 0.48))

    windows = slice_into_windows(audio, sr=target_sr, window_sec=window_sec, hop_sec=hop_sec)
    predictions = []
    all_alerts = []
    overall_detected: set[str] = set()

    detector.reset_state()
    for start_sec, end_sec, chunk in windows:
        pred, alerts = detector.process_window(chunk, start_sec, end_sec)
        predictions.append({
            "start_sec": pred.start_sec,
            "end_sec": pred.end_sec,
            "probabilities": pred.probabilities,
            "detected_classes": pred.detected_classes,
            "emergency_score": pred.emergency_score,
        })
        for c in pred.detected_classes:
            overall_detected.add(c)
        for a in alerts:
            all_alerts.append({
                "timestamp_sec": a.timestamp_sec,
                "event_category": a.event_category,
                "confidence_score": a.confidence_score,
                "detection_threshold": a.detection_threshold,
                "message": a.message,
            })

    # Find highest emergency score
    max_emergency_score = max([p["emergency_score"] for p in predictions]) if predictions else 0.0

    return {
        "filename": file.filename,
        "duration_sec": round(duration_sec, 2),
        "total_windows_processed": len(windows),
        "max_emergency_score": round(max_emergency_score, 4),
        "overall_detected_events": sorted(list(overall_detected)),
        "alerts_emitted": all_alerts,
        "window_predictions": predictions,
    }


@app.websocket("/ws/live")
async def websocket_live_inference(websocket: WebSocket):
    """Real-time bidirectional WebSocket endpoint for streaming microphone audio.
    
    Receives: Raw Float32 PCM audio chunks from browser microphone.
    Processes: Maintains a bounded rolling buffer of 16,000 samples (~1.0s).
    Sends: Live model predictions, smoothed scores, thresholds, and detection alerts.
    """
    await websocket.accept()
    if not detector:
        await websocket.send_json({"type": "error", "message": "Model not loaded"})
        await websocket.close()
        return

    detector.reset_state()
    target_sr = detector.cfg["audio"]["sample_rate"]
    window_samples = int(target_sr * detector.cfg["audio"].get("window_duration_sec", 0.96))
    audio_buffer = np.zeros(0, dtype=np.float32)

    session_start_time = time.time()
    last_inference_time = 0.0
    inference_interval_sec = 0.25  # Run inference every 250 ms

    try:
        while True:
            # Receive either binary Float32 bytes or text control commands
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                # Convert raw Float32 buffer
                pcm_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                if len(pcm_chunk) == 0:
                    continue

                # Append to rolling buffer (bounded to max 32,000 samples = 2 seconds)
                audio_buffer = np.concatenate([audio_buffer, pcm_chunk])
                if len(audio_buffer) > target_sr * 2:
                    audio_buffer = audio_buffer[-target_sr * 2:]

                current_time = time.time()
                elapsed_session_sec = current_time - session_start_time

                # Check if buffer has enough samples for an inference window
                if len(audio_buffer) >= window_samples and (current_time - last_inference_time) >= inference_interval_sec:
                    window_audio = audio_buffer[-window_samples:]
                    last_inference_time = current_time

                    # Run inference and measure latency
                    t_start = time.perf_counter()
                    pred, alerts = detector.process_window(
                        window_audio,
                        max(0.0, elapsed_session_sec - (window_samples / target_sr)),
                        elapsed_session_sec,
                    )
                    latency_ms = (time.perf_counter() - t_start) * 1000

                    # Calculate live audio RMS level
                    rms = float(np.sqrt(np.mean(window_audio ** 2)))

                    # Determine top prediction
                    probs = pred.probabilities
                    top_class = max(probs, key=probs.get)
                    top_score = probs[top_class]

                    # Specifically check screaming detection
                    is_screaming = probs.get("screaming", 0.0) >= detector.thresholds.get("screaming", 0.48)

                    # Build alert payload if qualifying event met persistence & debounce
                    alert_info = None
                    if alerts:
                        alert = alerts[0]
                        alert_info = {
                            "event": alert.event_category,
                            "score": alert.confidence_score,
                            "threshold": alert.detection_threshold,
                            "timestamp": alert.timestamp_sec,
                            "message": alert.message,
                        }

                    # Stream result to frontend
                    await websocket.send_json({
                        "type": "prediction",
                        "timestamp": round(elapsed_session_sec, 2),
                        "scores": probs,
                        "thresholds": detector.thresholds,
                        "top_class": top_class,
                        "top_score": top_score,
                        "is_emergency": top_class != "background" and top_score >= detector.thresholds.get(top_class, 0.50),
                        "is_alert": len(alerts) > 0,
                        "is_screaming_detected": is_screaming,
                        "alert": alert_info,
                        "inference_latency_ms": round(latency_ms, 2),
                        "rms": round(rms, 4),
                        "buffer_samples": len(audio_buffer),
                    })

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    cmd = payload.get("type")
                    if cmd == "reset":
                        detector.reset_state()
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        session_start_time = time.time()
                        await websocket.send_json({"type": "reset_ack", "timestamp": time.time()})
                    elif cmd == "set_thresholds":
                        new_th = payload.get("thresholds", {})
                        for k, v in new_th.items():
                            if k in detector.thresholds:
                                detector.thresholds[k] = float(v)
                        await websocket.send_json({"type": "config_ack", "thresholds": detector.thresholds})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        print("[WEBSOCKET] Client disconnected gracefully.")
    except Exception as e:
        print(f"[WEBSOCKET ERROR] {e}")
    finally:
        detector.reset_state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
