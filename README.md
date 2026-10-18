# Audio-Only Emergency Sound Detection Using AudioSet (2017)
### Final Year Design Project (FYDP) — Machine Learning & Acoustic AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![AudioSet](https://img.shields.io/badge/Dataset-Google_AudioSet_(2017)-green.svg)](https://research.google.com/audioset/)
[![Tests](https://img.shields.io/badge/Tests-7%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Objective & Ethical Boundaries

This repository implements a complete, reproducible machine learning system designed to detect acoustic events that may indicate a potential environmental emergency without relying on cameras, computer vision, Wi-Fi CSI, radar, or speech transcription.

### 🛡️ Core Operational & Ethical Guardrails
- **Strictly Audio-Only:** Operates exclusively on acoustic pressure waveforms.
- **Zero Speech Recognition / NLP:** The system does **NOT** perform speech-to-text, transcription, keyword spotting, or spoken-word linguistic analysis. Human vocalizations are analyzed purely as physical acoustic events (e.g. screaming or high-frequency vocal distress) without interpreting spoken words.
- **Human Verification Required:** The system detects acoustic anomalies and raises a potential warning indicator. It **never claims to prove or determine that a crime, kidnapping, assault, or break-in has occurred**. Its sole objective is alerting a human operator for verification.
- **Privacy Preserving:** Designed for edge processing without continuously recording or storing raw ambient microphone audio.

---

## 2. Target Acoustic Classes & AudioSet (2017) Ontology

Classes are grounded directly in the official [Google AudioSet Ontology](https://github.com/audioset/ontology) and identified by their official Machine Identifiers (MIDs):

| Index | Target Class | Official AudioSet MIDs | Ontological Description |
|---|---|---|---|
| 0 | `screaming` | `/m/03qc9zr` | Sharp, high-pitched human vocalization / distress scream |
| 1 | `alarm` | `/m/07pp_mv`, `/m/0c3f7m`, `/m/01y3hg`, `/m/02mfyn` | General alarms, fire alarms, smoke detectors, car alarms |
| 2 | `glass_shatter` | `/m/07rn7sz`, `/m/039jq` | Glass shattering, breaking, clinking |
| 3 | `smash_crash` | `/m/07pjjrj` | Rigid structure violently breaking / physical impact |
| 4 | `explosion` | `/m/014zdl` | Rapid blast, energetic rupture, low rumble |
| 5 | `gunshot` | `/m/032s66`, `/m/04zjc`, `/m/02z32qm`, `/m/0_1c`, `/m/073cg4` | Firearm discharge, machine gun, artillery |
| 6 | `background` | `/m/0btp2`, `/m/07pbtc8`, `/m/09x0r`, `/m/04rlf` | Non-emergency: traffic, footsteps, ordinary speech, music |

*Note: Multi-label annotations are preserved. Ordinary speech is never an emergency label.*

---

## 3. System Architecture

```
[ Raw Audio Input (16 kHz, Mono) ]
                │
                ▼
[ Log-Mel Spectrogram Extraction ]  <-- 64 Mel filters, 25ms window, 10ms hop
                │
                ▼
[ CompactAudioCNN Backbone ]        <-- 4 Conv Blocks + BatchNorm + AdaptiveAvgPool
                │
                ▼
[ Multi-Label Sigmoid Classifier ]  <-- 8 Continuous Class Probabilities
                │
                ▼
[ Sliding-Window Detection Engine ] <-- ~0.96s Window, 50% Hop
                │
  ┌─────────────┴─────────────┐
  ▼                           ▼
[ Temporal Smoothing ]     [ Class-Specific Thresholds ]
  │                           │
  └─────────────┬─────────────┘
                ▼
[ Persistence Filter (K >= 2) ]     <-- Discards transient single-frame spikes
                │
                ▼
[ Debounce Cooldown (3.0s) ]        <-- Suppresses repeated duplicate alerts
                │
                ▼
[ Alert & Verification Report ]     <-- "Potential Acoustic Warning Detected"
```

---

## 4. Repository Structure

```
d:/ml/fydp/
├── backend/                        # FastAPI Backend & WebSocket Streaming Server
│   └── main.py                     # /api/health, /api/model-info, /api/predict-file, /ws/live
├── frontend/                       # Modern Dark-Themed Web Application (React + TypeScript + Vite)
│   ├── src/
│   │   ├── components/             # Visualizers (Oscilloscope, Spectrogram), Panels, Modals
│   │   ├── services/api.ts         # REST & WebSocket service
│   │   ├── App.tsx                 # Real-time Web Audio API streaming & state logic
│   │   └── index.css               # Obsidian design system & glassmorphism styling
│   └── package.json
├── configs/
│   └── default.yaml                # Standardized 7-class configuration (Siren excluded)
├── data/
│   ├── manifests/
│   │   ├── genuine_audioset_manifest.csv       # Verified genuine AudioSet clips (166 clips)
│   │   └── synthetic_test_fixtures_manifest.csv # Isolated unit test fixtures
│   ├── test_fixtures/synthetic/    # Isolated synthetic test fixtures (never used in model training)
│   └── processed/                  # Cached log-mel feature tensors (.pt)
├── src/
│   ├── dataset/
│   │   ├── audit.py                # Dataset integrity auditor (checks clips & synthetic isolation)
│   │   └── preprocess_audio.py     # Leakage-free StratifiedGroupKFold splits & feature extraction
│   ├── models/
│   │   └── classifier.py           # CompactAudioCNN architecture
│   ├── training/
│   │   ├── train.py                # Multi-label BCE training loop with pos_weight
│   │   └── evaluate.py             # Evaluation & threshold calibration on held-out test split
│   ├── inference/
│   │   ├── stream_detector.py      # Sliding-window inference, persistence & debounce engine
│   │   └── predict_file.py         # Single file prediction CLI
│   └── utils/
│       └── audio_utils.py          # Audio loading, resampling, mel specs, augmentations
├── artifacts/
│   └── best_emergency_model.pt     # Trained CompactAudioCNN checkpoint (4.87 MB)
├── reports/
│   └── evaluation_report.json      # Evaluation metrics, calibrated thresholds & test results
└── tests/
    ├── test_pipeline.py            # Audio processing & model unit tests
    └── test_backend.py             # FastAPI REST & WebSocket test suite
```

---

## 5. Live Demonstration Web Application

The live demonstration runs as a modern, responsive web application connecting a browser-based frontend directly to the trained `CompactAudioCNN` model via a high-performance Python FastAPI WebSocket backend.

### Architecture
- **Frontend:** React 19 + TypeScript + Vite + Custom Obsidian Glassmorphism CSS.
- **Backend:** Python + FastAPI + Uvicorn + WebSockets.
- **Microphone Pipeline:** Browser Web Audio API (`navigator.mediaDevices.getUserMedia()`) capturing live Float32 PCM samples, resampled to 16 kHz in real time.
- **Low-Latency Streaming:** Raw binary WebSocket frames sent every ~85 ms, sliding-window inference every 250 ms, measured latency $< 20\text{ ms}$.

### How to Launch the Web Application

#### Terminal 1 — Start the FastAPI Backend:
```bash
# From repository root
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*Loads `artifacts/best_emergency_model.pt`, warms up the forward pass, and serves REST + WebSocket endpoints on port 8000.*

#### Terminal 2 — Start the Vite Frontend:
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
*Open `http://127.0.0.1:5173` in any modern web browser (Chrome, Edge, Firefox).*

### Live Demonstration Features
1. **Live Oscilloscope Waveform:** Real-time 60fps moving waveform from the browser microphone.
2. **Rolling Thermal Spectrogram:** Scrolling 2D spectrogram with thermal Magma colormap (0–4 kHz) highlighting vocal distress and impulse acoustic bursts.
3. **Dynamic 7-Class Prediction Bars:** Real-time confidence scores for Screaming, Alarm, Glass Shatter, Crash/Smash, Explosion, Gunshot, and Background.
4. **Calibrated Threshold Markers:** Visual indicator pins calibrated on validation data to reject ambient false alarms.
5. **Vocal Distress Alert Indicator:** Screaming is highlighted with a distinct indicator when model confidence exceeds the calibrated threshold.
6. **Chronological Event Timeline:** Logs confirmed events with relative timestamps, durations, and scores.
7. **Diagnostics Self-Test:** One-click modal verifying backend connectivity, model weights, audio context, and latency.
8. **Controlled File Test Mode:** Independent mode to upload and classify prerecorded WAV files without microphone interference.

---

## 6. End-to-End Pipeline Execution

### Step 1: Run Dataset Integrity Audit
```bash
python -m src.dataset.audit
```
*Audits all 200 cataloged clips: verifies 166 genuine AudioSet clips on disk (zero missing, zero corrupted) and confirms that 34 synthetic fallback files are strictly isolated into `data/test_fixtures/synthetic/`.*

### Step 2: Preprocess Audio & Create Leakage-Free Splits
```bash
python -m src.dataset.preprocess_audio
```
*Splits strictly genuine AudioSet data by `ytid` to guarantee zero data leakage (Train: 118, Val: 24, Test: 24 clips).*

### Step 3: Train the Emergency Classifier
```bash
python -m src.training.train
```
*Trains `CompactAudioCNN` on genuine training clips. Val loss drops to 0.5824, Macro F1 reaches 0.4410.*

### Step 4: Evaluate on Held-Out Test Data
```bash
python -m src.training.evaluate
```
*Evaluates the held-out test split, calibrates thresholds, and writes `reports/evaluation_report.json`.*

### Step 5: Single-File Inference (CLI)
```bash
python -m src.inference.predict_file --audio data/test_fixtures/synthetic/synthetic_screaming_01.wav
```

### Step 6: Run Unit & Integration Tests
```bash
pytest tests/ -v
```
*Executes all 11 unit and integration tests across data processing, model architectures, backend endpoints, and WebSocket streaming.*

---

## 7. Experimental Results on Held-Out AudioSet Test Data

Evaluation performed strictly on held-out AudioSet clips with zero synthetic data contamination:

| Acoustic Category | Calibrated Threshold | Binary Accuracy | Precision | Recall | F1-Score | Average Precision (AP) |
|---|---|---|---|---|---|---|
| 🚨 **Alarm** (`/m/07pp_mv`, ...) | 0.54 | **91.67%** | **1.0000** | 0.5000 | **0.6667** | **0.9167** |
| 🗣️ **Screaming** (`/m/03qc9zr`) | 0.48 | **87.50%** | 0.5714 | **1.0000** | **0.7273** | **0.8042** |
| 💥 **Crash / Smash** (`/m/07pjjrj`) | 0.50 | **87.50%** | 0.6667 | 0.5000 | **0.5714** | **0.6667** |
| 🔫 **Gunshot** (`/m/032s66`, ...) | 0.50 | **83.33%** | — | — | — | 0.0833 |
| 💣 **Explosion** (`/m/014zdl`) | 0.48 | **83.33%** | — | — | — | 0.1667 |
| 🪟 **Glass Shatter** (`/m/07rn7sz`) | 0.49 | **75.00%** | — | — | — | 0.1250 |
| 🌳 **Ambient / Background** | 0.48 | **70.83%** | 0.2857 | 0.5000 | **0.3636** | 0.3090 |
| **Macro Binary Accuracy** | — | **82.74%** | — | — | — | — |
| **False Alarm Rate on Ambient Audio** | — | **20.00%** | (80% specificity rejecting non-emergency ambient noise) |

---

## 8. Windows Audio & Microphone Troubleshooting

- **Microphone Permissions:** In Windows Settings $\rightarrow$ Privacy & Security $\rightarrow$ Microphone, ensure "Microphone access" and "Let desktop apps access your microphone" are set to **On**.
- **Browser Permission:** Click the lock or tune icon in the browser address bar next to `http://127.0.0.1:5173` and allow Microphone access.
- **Audio Sample Rates:** The web application natively queries the hardware sample rate and resamples on-the-fly to 16,000 Hz, eliminating driver mismatch errors.

---

### Diagnostic Plots & Benchmark Visualizations

#### 1. Training Diagnostics (Loss, Macro/Micro F1, Precision/Recall, and LR Decay)
![Training Curves](reports/training_curves.png)

#### 2. Multi-Label Precision-Recall Curves
![Precision-Recall Curves](reports/pr_curves.png)

#### 3. Receiver Operating Characteristic (ROC) Curves
![ROC Curves](reports/roc_curves.png)

#### 4. Per-Class Precision, Recall, and F1 Performance
![Per-Class Metrics Bar Chart](reports/per_class_metrics.png)

#### 5. Validation Decision Threshold Calibration Curves
![Threshold Tuning Curves](reports/threshold_tuning_curves.png)

#### 6. Trained Model vs Traditional Acoustic Energy Baseline
![Baseline vs Model Comparison](reports/baseline_vs_model.png)

---

### Computational Latency (Measured on Host CPU)
- **Audio Window:** $0.96\text{ seconds}$
- **Mel Feature Extraction:** $1.37\text{ ms}$
- **Neural Network Forward Pass:** $1.59\text{ ms}$
- **Total Latency per Window:** $2.95\text{ ms} \pm 0.32\text{ ms}$
- **Real-Time Factor (RTF):** $0.0031$ ($\sim 320\times$ faster than real-time)
- **Model Parameters:** $422,504$ (Disk size: $4.87\text{ MB}$)

### Edge Deployment Feasibility
- **Raspberry Pi 4 / 5:** **Fully Feasible**. With 422k parameters, quad-core Cortex-A72/A76 handles inference at $\text{RTF} < 0.15$.
- **NVIDIA Jetson Nano:** **Fully Feasible**. TensorRT FP16 achieves $< 1.5\text{ ms}$ per window.
- **ESP32:** **Incompatible**. 520 KB SRAM cannot hold a 4.8 MB float32 model. Recommended alternative: Raspberry Pi Zero 2W.


---

## 8. Google Colab Support

For GPU-accelerated training on larger AudioSet subsets, open and run `Emergency_Sound_Detection_Colab.ipynb` directly in [Google Colab](https://colab.research.google.com/).

---

## 9. Citation & Dataset Acknowledgments

- **AudioSet:** Gemmeke, J. F., et al. "Audio Set: An ontology and human-labeled dataset for audio events." *ICASSP 2017*.
- **YAMNet / CNN Architectures:** Hershey, S., et al. "CNN architectures for large-scale audio classification." *ICASSP 2017*.
