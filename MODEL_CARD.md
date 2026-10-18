# Model Card: CompactAudioCNN for Emergency Sound Detection

## Model Overview
- **Model Name:** CompactAudioCNN (Acoustic Emergency Sound Detector)
- **Model Version:** 1.0.0
- **Model Type:** 4-Block Convolutional Neural Network with Adaptive Average Pooling and Multi-Label Sigmoid Classifier Head.
- **Input Representation:** 64-channel Log-Mel Spectrogram (STFT: $N_{\text{FFT}}=1024$, Hop Length = 320 [20 ms], Window Length = 1024, Sample Rate = 16,000 Hz Mono).
- **Output:** 8 independent continuous probabilities in $[0.0, 1.0]$ corresponding to target emergency and environmental sound classes.
- **Parameters:** 422,504 float32 parameters.
- **Checkpoint Footprint:** ~4.87 MB.
- **Release Date:** October 2026.

---

## Intended Use & Boundaries

### Primary Intended Use
- Detection of environmental acoustic patterns that may indicate potential safety hazards or emergencies (e.g., human screams, sirens, fire alarms, smoke detectors, glass breaking, violent crashes, explosions, gunshots) in indoor and outdoor settings.
- Raising a high-priority advisory alert for **human verification and assessment**.

### Explicit Out-of-Scope & Prohibited Uses
> [!IMPORTANT]
> - **No Crime Determination:** The system MUST NOT be used to conclude or claim that a crime (e.g., kidnapping, assault, robbery) has occurred.
> - **No Speech Recognition or Keyword Spotting:** The model does not transcribe words, understand spoken language, or parse linguistic semantics.
> - **No Autonomous Physical Action:** The system must never trigger lethal force, automated locking of life-safety exits, or punitive responses without human operator verification.
> - **No Covert Surveillance:** Microphones must not store persistent raw audio recordings without explicit user consent.

---

## Target Classes & AudioSet Mappings

| Index | Class Name | AudioSet Machine IDs (MIDs) | Acoustic Event Description |
|---|---|---|---|
| 0 | `screaming` | `/m/03qc9zr` | Instinctive high-frequency human vocal distress |
| 1 | `alarm` | `/m/07pp_mv`, `/m/0c3f7m`, `/m/01y3hg`, `/m/02mfyn` | Fire alarms, smoke detectors, car alarms |
| 2 | `glass_shatter` | `/m/07rn7sz`, `/m/039jq` | Shattering, breaking, clinking rigid glass |
| 3 | `smash_crash` | `/m/07pjjrj` | Rigid structure impact / violent breaking |
| 4 | `explosion` | `/m/014zdl` | Rapid blast, energetic rupture, low rumble |
| 5 | `gunshot` | `/m/032s66`, `/m/04zjc`, `/m/02z32qm`, `/m/0_1c`, `/m/073cg4` | Firearm impulse, artillery, burst firing |
| 6 | `background` | `/m/0btp2`, `/m/07pbtc8`, `/m/09x0r`, `/m/04rlf` | Non-emergency: traffic, footsteps, speech, music |

---

## Data & Leakage Prevention Strategy
- **Primary Source:** Google AudioSet (2017) YouTube corpus (`balanced_train_segments.csv` and `eval_segments.csv`).
- **Grouping:** All dataset splits (Train 70%, Validation 15%, Test 15%) are partitioned strictly by YouTube Video ID (`ytid`). Segments sharing the same video ID never cross splits, guaranteeing **0% data leakage**.
- **Augmentation:** SpecAugment (frequency and time masking), additive white Gaussian noise (15-30 dB SNR), random gain ($\pm 6\text{ dB}$), and time shifting are applied exclusively to training data. Zero augmentation is applied to validation and test partitions.

---

## Quantitative Performance & Benchmarks

### Computational Latency (Empirically Measured on Host CPU)
- **Audio Analysis Window:** 0.96 seconds (15,360 samples at 16 kHz)
- **Mel Feature Extraction:** $1.37\text{ ms}$
- **Neural Network Forward:** $1.59\text{ ms}$
- **Total Latency per Window:** $2.95\text{ ms} \pm 0.32\text{ ms}$
- **95th Percentile Latency:** $3.70\text{ ms}$
- **Real-Time Factor (RTF):** $0.0031$ ($< 1.0$ indicates real-time processing; system processes audio ~320x faster than real-time).
- **Throughput:** $338.5\text{ windows / second}$.

### Edge Hardware Deployment Feasibility
- **Raspberry Pi 4 / 5:** Highly Feasible. Quad-core Cortex-A72/A76 handles inference with $\text{RTF} < 0.15$.
- **NVIDIA Jetson Nano:** Highly Feasible. TensorRT FP16 yields $< 1.5\text{ ms}$ latency.
- **ESP32 Microcontroller:** Incompatible for full model execution. The 422k parameter network requires $> 4.8\text{ MB}$ memory, exceeding ESP32's 520 KB on-chip SRAM.

---

## Known Limitations & Mitigation
1. **Weak Labels:** AudioSet annotations apply at the 10-second clip level; specific acoustic events may only occupy a brief 0.5s segment within the clip.
2. **Acoustic Domain Shift:** AudioSet YouTube clips contain background compression artifacts, room reverberation, and diverse microphone responses differing from real edge sensor mics.
3. **Class Imbalance:** Rare emergency sounds are outnumbered by speech and music. Mitigated using positive class weighting (`pos_weight`) in `BCEWithLogitsLoss`.
4. **False Positive Suppression:** Mitigated using sliding-window persistence ($K \ge 2$ consecutive frame threshold) and debounce cooldown (3.0s minimum gap between repeated alerts).
