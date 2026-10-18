"""FastAPI Backend Server for Real-Time Emergency Sound Detection.

Hardened Features:
1. REST endpoints: /api/health, /api/model-info, /api/predict-file
2. Real-time WebSocket: /ws/live with session-isolated temporal smoothing & buffer
3. Model loaded ONCE into memory on startup; sessions share the neural weights safely
4. Explicit detection state: raw scores, smoothed scores, thresholds, persistence counters, and alert states
5. Hardened CORS, payload size limits (25 MB file, 64 KB WS binary chunk), and finite sample validation
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import torch
import yaml

from src.models.classifier import CompactAudioCNN
from src.inference.stream_detector import StreamEmergencyDetector
from src.utils.audio_utils import compute_log_mel_spectrogram, load_audio, slice_into_windows

# Configurable limits
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB max upload
MAX_WS_BYTES = 65536               # 64 KB max per WebSocket audio frame (~2.0s at 16k float32)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
).split(",")

# Global single-instance model and configuration
global_model: Optional[CompactAudioCNN] = None
global_cfg: Dict[str, any] = {}
global_thresholds: Dict[str, float] = {}
global_device: torch.device = torch.device("cpu")
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


def initialize_global_model():
    """Load the neural network checkpoint once into memory on application startup."""
    global global_model, global_cfg, global_thresholds, global_device, model_metadata
    cfg_path = ROOT_DIR / "configs/default.yaml"
    ckpt_path = ROOT_DIR / "artifacts/best_emergency_model.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        global_cfg = yaml.safe_load(f)

    device_str = global_cfg.get("project", {}).get("device", "auto")
    global_device = torch.device(
        "cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu"
    )

    global_thresholds = load_calibrated_thresholds()
    print(f"[STARTUP] Initializing global CompactAudioCNN from: {ckpt_path} on {global_device}")
    print(f"[STARTUP] Calibrated class thresholds: {global_thresholds}")

    ckpt = torch.load(ckpt_path, map_location=global_device, weights_only=False)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    global_model = CompactAudioCNN(
        num_classes=num_classes,
        n_mels=global_cfg["audio"]["n_mels"],
        embedding_dim=global_cfg["model"]["embedding_dim"],
        hidden_dim=global_cfg["model"]["hidden_dim"],
        dropout=global_cfg["model"]["dropout"],
    ).to(global_device)
    global_model.load_state_dict(ckpt["model_state_dict"])
    global_model.eval()

    # Model warmup forward pass
    dummy_wav = np.zeros(int(16000 * 0.96), dtype=np.float32)
    t0 = time.perf_counter()
    mel_spec = compute_log_mel_spectrogram(
        dummy_wav,
        sr=global_cfg["audio"]["sample_rate"],
        n_mels=global_cfg["audio"]["n_mels"],
        n_fft=global_cfg["audio"]["n_fft"],
        hop_length=global_cfg["audio"]["hop_length"],
        win_length=global_cfg["audio"]["win_length"],
    ).to(global_device)
    with torch.no_grad():
        global_model.predict_proba(mel_spec.unsqueeze(0))
    warmup_time = (time.perf_counter() - t0) * 1000
    print(f"[STARTUP] Model warmup successful (Latency: {warmup_time:.2f} ms)")

    # Cache metadata
    eval_path = ROOT_DIR / "reports/evaluation_report.json"
    eval_metrics = {}
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    target_classes_cfg = global_cfg["classes"]["target_classes"]
    model_metadata = {
        "model_name": "CompactAudioCNN (AudioSet 2017)",
        "version": "2.0",
        "sample_rate": global_cfg["audio"]["sample_rate"],
        "target_classes": [
            {
                "name": c["name"],
                "display_name": c.get("display_name", c["name"]),
                "is_emergency": c.get("is_emergency", False),
                "threshold": global_thresholds.get(c["name"], float(c.get("default_threshold", 0.50))),
                "persistence_frames": int(c.get("persistence_frames", 2)),
            }
            for c in target_classes_cfg
        ],
        "class_names": class_names,
        "window_duration_sec": global_cfg["audio"].get("window_duration_sec", 0.96),
        "hop_duration_sec": global_cfg["audio"].get("hop_duration_sec", 0.25),
        "evaluation_metrics": eval_metrics.get("test_results_trained_model", {}),
        "test_size_disclosure": eval_metrics.get("test_size_disclosure", "Test split contains N=24 clips."),
        "silence_gate": {
            "enabled": global_cfg.get("detection", {}).get("silence_gate_enabled", True),
            "threshold_rms": global_cfg.get("detection", {}).get("silence_threshold_rms", 0.004),
            "threshold_peak": global_cfg.get("detection", {}).get("silence_threshold_peak", 0.025),
            "sustained_frames": global_cfg.get("detection", {}).get("sustained_silence_frames", 1),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    initialize_global_model()
    yield
    print("[SHUTDOWN] Releasing backend resources.")


app = FastAPI(
    title="Audio Emergency Sound Detector API",
    description="Real-Time Audio-Only Environmental Emergency Sound Detection using AudioSet 2017",
    version="2.0.0",
    lifespan=lifespan,
)

# Hardened CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def create_session_detector() -> StreamEmergencyDetector:
    """Instantiate an isolated session detector sharing the global in-memory model."""
    if global_model is None:
        initialize_global_model()

    cfg_path = str(ROOT_DIR / "configs/default.yaml")
    ckpt_path = str(ROOT_DIR / "artifacts/best_emergency_model.pt")

    return StreamEmergencyDetector(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        custom_thresholds=global_thresholds,
        shared_model=global_model,
    )


@app.get("/api/health")
def health_check():
    """Health status and model readiness check."""
    return {
        "status": "online",
        "model_loaded": global_model is not None,
        "device": str(global_device),
        "num_classes": len(model_metadata.get("class_names", [])),
        "timestamp": time.time(),
    }


@app.get("/api/model-info")
def get_model_info():
    """Return model architecture, configured classes, thresholds, and performance metrics."""
    if global_model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized or checkpoint unavailable.")
    return model_metadata


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """Analyze an uploaded WAV file with size limits and robust audio validation."""
    if global_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size ({MAX_FILE_BYTES // (1024*1024)} MB).")

    try:
        with io.BytesIO(contents) as bio:
            audio, sr = sf.read(bio)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # Convert to mono
            audio = audio.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted audio format: {e}")

    # Validate finite values
    if not np.all(np.isfinite(audio)):
        raise HTTPException(status_code=400, detail="Audio contains invalid NaN or Infinite sample values.")

    target_sr = global_cfg["audio"]["sample_rate"]
    if sr != target_sr:
        import scipy.signal
        num_target_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_target_samples).astype(np.float32)

    duration_sec = float(len(audio) / target_sr)
    # Log audio file properties
    file_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
    file_peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    print(f"[API PREDICT-FILE] Filename: {file.filename} | Duration: {duration_sec:.2f}s | SR: {target_sr}Hz | RMS: {file_rms:.5f} | Peak: {file_peak:.5f}")

    window_sec = float(global_cfg["audio"].get("window_duration_sec", 0.96))
    hop_sec = float(global_cfg["audio"].get("hop_duration_sec", 0.48))

    windows = slice_into_windows(audio, sr=target_sr, window_sec=window_sec, hop_sec=hop_sec)
    predictions = []
    all_alerts = []
    overall_detected: set[str] = set()

    # Use an isolated detector for file evaluation
    detector = create_session_detector()

    for start_sec, end_sec, chunk in windows:
        pred, alerts = detector.process_window(chunk, start_sec, end_sec)
        predictions.append({
            "start_sec": pred.start_sec,
            "end_sec": pred.end_sec,
            "raw_probabilities": pred.raw_probabilities,
            "probabilities": pred.probabilities,
            "persistence_counts": pred.persistence_counts,
            "detected_classes": pred.detected_classes,
            "emergency_score": pred.emergency_score,
            "rms": pred.rms,
            "peak": pred.peak,
            "is_silence_gated": pred.is_silence_gated,
            "gate_reason": pred.gate_reason,
        })
        for c in pred.detected_classes:
            overall_detected.add(c)
        for a in alerts:
            all_alerts.append({
                "timestamp_sec": a.timestamp_sec,
                "event_category": a.event_category,
                "confidence_score": a.confidence_score,
                "detection_threshold": a.detection_threshold,
                "persistence_count": a.persistence_count,
                "message": a.message,
                "trigger_reason": a.trigger_reason,
            })

    max_emergency_score = max([p["emergency_score"] for p in predictions]) if predictions else 0.0
    gated_windows_count = sum(1 for p in predictions if p["is_silence_gated"])

    return {
        "filename": file.filename,
        "duration_sec": round(duration_sec, 2),
        "total_windows_processed": len(windows),
        "silence_gated_windows": gated_windows_count,
        "silence_gate_enabled": detector.silence_gate_enabled,
        "silence_threshold_rms": detector.silence_threshold_rms,
        "max_emergency_score": round(max_emergency_score, 4),
        "overall_detected_events": sorted(list(overall_detected)),
        "alerts_emitted": all_alerts,
        "window_predictions": predictions,
    }



@app.websocket("/ws/live")
async def websocket_live_inference(websocket: WebSocket):
    """Real-time bidirectional WebSocket endpoint for streaming microphone audio.

    Isolates temporal smoothing and audio buffers per client session.
    Distinguishes raw scores, smoothed scores, persistence states, and alerts.
    """
    await websocket.accept()
    if global_model is None:
        await websocket.send_json({"type": "error", "message": "Model not loaded"})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Dedicated isolated session detector
    session_detector = create_session_detector()
    target_sr = global_cfg["audio"]["sample_rate"]
    window_samples = int(target_sr * global_cfg["audio"].get("window_duration_sec", 0.96))
    audio_buffer = np.zeros(0, dtype=np.float32)

    session_start_time = time.time()
    last_inference_time = 0.0
    inference_interval_sec = 0.25  # Run inference every 250 ms
    last_log_time = 0.0
    last_silence_gated = False

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                if len(chunk_bytes) > MAX_WS_BYTES:
                    # Drop oversized binary payloads to protect server memory
                    continue

                pcm_chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                if len(pcm_chunk) == 0:
                    continue

                # Validate finite float values
                if not np.all(np.isfinite(pcm_chunk)):
                    continue

                pcm_chunk = np.clip(pcm_chunk, -1.0, 1.0)

                # Append to rolling buffer (bounded to max 32,000 samples = 2.0 seconds)
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
                    pred, alerts = session_detector.process_window(
                        window_audio,
                        max(0.0, elapsed_session_sec - (window_samples / target_sr)),
                        elapsed_session_sec,
                    )
                    latency_ms = (time.perf_counter() - t_start) * 1000

                    rms = float(pred.rms)
                    peak = float(pred.peak)
                    is_silence_gated = pred.is_silence_gated
                    gate_reason = pred.gate_reason

                    raw_probs = pred.raw_probabilities
                    smoothed_probs = pred.probabilities
                    top_class = max(smoothed_probs, key=smoothed_probs.get)
                    top_score = smoothed_probs[top_class]
                    top_raw_score = raw_probs.get(top_class, 0.0)

                    # Distinct Screaming State:
                    # 1. High Score: raw screaming probability >= threshold
                    screaming_thresh = session_detector.thresholds.get("screaming", 0.46)
                    is_screaming_score_high = raw_probs.get("screaming", 0.0) >= screaming_thresh
                    # 2. Screaming Alert: satisfies full persistence rule
                    is_screaming_alert = any(a.event_category == "screaming" for a in alerts)

                    # Persistence state breakdown per class
                    persistence_state = {
                        c: {
                            "count": pred.persistence_counts.get(c, 0),
                            "required": pred.persistence_requirements.get(c, 2),
                            "persisted": pred.persistence_counts.get(c, 0) >= pred.persistence_requirements.get(c, 2),
                        }
                        for c in session_detector.class_names
                    }

                    # Alert payload
                    alert_info = None
                    if alerts:
                        alert = alerts[0]
                        alert_info = {
                            "event": alert.event_category,
                            "score": alert.confidence_score,
                            "threshold": alert.detection_threshold,
                            "persistence_count": alert.persistence_count,
                            "timestamp": alert.timestamp_sec,
                            "message": alert.message,
                            "trigger_reason": alert.trigger_reason,
                        }

                    # Server-side periodic audio ingest logging
                    if (current_time - last_log_time) >= 2.0 or is_silence_gated != last_silence_gated:
                        last_log_time = current_time
                        last_silence_gated = is_silence_gated
                        print(
                            f"[WS AUDIO INGEST] SR: {target_sr}Hz | Buf: {len(audio_buffer)} smp | "
                            f"RMS: {rms:.5f} | Peak: {peak:.5f} | Gated: {is_silence_gated} | Top: {top_class} ({top_score:.3f})"
                        )

                    # Send structured state to client
                    await websocket.send_json({
                        "type": "prediction",
                        "timestamp": round(elapsed_session_sec, 2),
                        "raw_scores": raw_probs,
                        "scores": smoothed_probs,
                        "thresholds": session_detector.thresholds,
                        "persistence_state": persistence_state,
                        "top_class": top_class,
                        "top_score": top_score,
                        "top_raw_score": top_raw_score,
                        "is_emergency": top_class != "background" and top_score >= session_detector.thresholds.get(top_class, 0.50),
                        "is_alert": len(alerts) > 0,
                        "is_screaming_score_high": is_screaming_score_high,
                        "is_screaming_alert": is_screaming_alert,
                        "alert": alert_info,
                        "inference_latency_ms": round(latency_ms, 2),
                        "rms": round(rms, 6),
                        "peak": round(peak, 6),
                        "is_silence_gated": is_silence_gated,
                        "gate_reason": gate_reason,
                        "silence_threshold_rms": session_detector.silence_threshold_rms,
                        "buffer_samples": len(audio_buffer),
                    })

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    cmd = payload.get("type")
                    if cmd == "reset":
                        session_detector.reset_state()
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        session_start_time = time.time()
                        await websocket.send_json({"type": "reset_ack", "timestamp": time.time()})
                    elif cmd == "set_thresholds":
                        new_th = payload.get("thresholds", {})
                        for k, v in new_th.items():
                            if k in session_detector.thresholds:
                                session_detector.thresholds[k] = float(v)
                        await websocket.send_json({"type": "config_ack", "thresholds": session_detector.thresholds})
                    elif cmd == "set_silence_gate":
                        if "silence_threshold_rms" in payload:
                            session_detector.silence_threshold_rms = float(payload["silence_threshold_rms"])
                        if "silence_threshold_peak" in payload:
                            session_detector.silence_threshold_peak = float(payload["silence_threshold_peak"])
                        if "enabled" in payload:
                            session_detector.silence_gate_enabled = bool(payload["enabled"])
                        await websocket.send_json({
                            "type": "silence_gate_ack",
                            "silence_gate_enabled": session_detector.silence_gate_enabled,
                            "silence_threshold_rms": session_detector.silence_threshold_rms,
                            "silence_threshold_peak": session_detector.silence_threshold_peak,
                        })
                except json.JSONDecodeError:
                    pass


    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WEBSOCKET ERROR] {e}")
    finally:
        session_detector.reset_state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
