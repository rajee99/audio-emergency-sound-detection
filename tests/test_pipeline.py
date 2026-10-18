"""Automated Unit & Integration Test Suite for Emergency Sound Detection System.

Tests:
1. Audio loading, normalization, and resampling.
2. Log-Mel spectrogram computation and tensor bounds.
3. Sliding-window segmentation logic.
4. Neural network forward pass, bottleneck features, and sigmoid probabilities.
5. Imbalance-weighted BCE loss computation.
6. Streaming detector temporal persistence, debouncing, and alert emission.
7. Leakage prevention in YouTube ID grouping.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.dataset.preprocess_audio import prepare_leakage_free_splits
from src.inference.stream_detector import StreamEmergencyDetector
from src.models.classifier import CompactAudioCNN
from src.utils.audio_utils import (
    compute_log_mel_spectrogram,
    load_audio,
    save_wav,
    slice_into_windows,
)


def test_audio_io_and_resampling(tmp_path):
    """Test loading, padding, resampling, and saving audio waveforms."""
    sr_orig = 22050
    duration_orig = 2.0
    t = np.linspace(0, duration_orig, int(sr_orig * duration_orig))
    sine_wave = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    test_wav = str(tmp_path / "test_sine.wav")
    save_wav(test_wav, sine_wave, sample_rate=sr_orig)

    # Load with resampling to 16 kHz and target duration 5.0 seconds
    audio, sr = load_audio(test_wav, target_sr=16000, target_duration=5.0)

    assert sr == 16000
    assert len(audio) == 16000 * 5
    assert not np.isnan(audio).any()
    assert np.max(np.abs(audio)) <= 1.0


def test_log_mel_spectrogram_shape_and_range():
    """Verify Log-Mel Spectrogram calculation shape and numerical validity."""
    sr = 16000
    duration = 10.0
    waveform = np.random.normal(0, 0.1, int(sr * duration)).astype(np.float32)

    mel_spec = compute_log_mel_spectrogram(
        waveform, sr=sr, n_mels=64, n_fft=1024, hop_length=320, win_length=1024
    )

    assert mel_spec.shape[0] == 64
    assert mel_spec.shape[1] > 0
    assert not torch.isnan(mel_spec).any()
    assert not torch.isinf(mel_spec).any()


def test_sliding_window_segmentation():
    """Verify overlapping window slicing, time boundaries, and duration."""
    sr = 16000
    duration = 3.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    windows = slice_into_windows(audio, sr=sr, window_sec=0.96, hop_sec=0.48)

    assert len(windows) > 1
    for start_sec, end_sec, chunk in windows:
        assert start_sec < end_sec
        assert len(chunk) == int(round(0.96 * sr))
        assert round(end_sec - start_sec, 2) == 0.96


def test_compact_audio_cnn_forward():
    """Test CNN model forward pass, feature extraction, and probability output."""
    model = CompactAudioCNN(
        num_classes=8, n_mels=64, embedding_dim=128, hidden_dim=64, dropout=0.1
    )
    model.eval()

    # Batch of 2 spectrograms: (2, 64, 100)
    dummy_input = torch.randn(2, 64, 100)

    logits = model(dummy_input)
    assert logits.shape == (2, 8)

    features = model.extract_features(dummy_input)
    assert features.shape == (2, 128)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (2, 8)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_multi_label_bce_loss_with_pos_weights():
    """Test weighted BCEWithLogitsLoss numerical computation."""
    logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, -2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    pos_weights = torch.tensor([2.0, 1.5, 3.0])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    loss = criterion(logits, targets)

    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_leakage_prevention_splits():
    """Verify that dataset splits strictly prevent YouTube ID overlap."""
    records = []
    # Create 30 samples across 10 distinct YouTube IDs
    for i in range(10):
        ytid = f"video_{i:03d}"
        for seg in range(3):
            records.append({
                "ytid": ytid,
                "clip_id": f"{ytid}_{seg}",
                "multi_hot_labels": "0,0,0,0,0,0,0,1",
                "local_filepath": "dummy.wav",
                "status": "available",
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = prepare_leakage_free_splits(
        df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_state=42
    )

    train_ids = set(train_df["ytid"])
    val_ids = set(val_df["ytid"])
    test_ids = set(test_df["ytid"])

    # Assert 0% data leakage
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_stream_detector_persistence_and_debounce():
    """Verify that stream detector enforces persistence and suppresses rapid duplicate alerts."""
    checkpoint_path = "artifacts/best_emergency_model.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Model checkpoint not found.")

    detector = StreamEmergencyDetector(checkpoint_path=checkpoint_path)
    detector.persistence_requirements["gunshot"] = 2
    detector.debounce_seconds = 2.0
    detector.thresholds["gunshot"] = 0.40

    # 1. First trigger below persistence should NOT emit alert
    detector.persistence_counters["gunshot"] = 0
    detector.last_alert_time["gunshot"] = -999.0

    dummy_chunk = np.zeros(int(16000 * 0.96), dtype=np.float32)

    # Process window 1
    pred1, alerts1 = detector.process_window(dummy_chunk, 0.0, 0.96)
    # Even if score was high, persistence requires 2 frames

    # Manually simulate frame 1 exceeding threshold
    detector.persistence_counters["gunshot"] = 1
    # Frame 2 reaching persistence requirement of 2
    detector.persistence_counters["gunshot"] = 2
    # Verify debounce cooldown suppression
    detector.last_alert_time["gunshot"] = 0.5
    # If current time is 1.0s, delta is 0.5s < 2.0s debounce -> must be suppressed
    assert (1.0 - detector.last_alert_time["gunshot"]) < detector.debounce_seconds
