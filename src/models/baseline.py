"""Pretrained Baseline & Transfer Learning Module for Emergency Sound Detection.

Implements:
1. ZeroShotPretrainedBaseline: Uses pretrained AudioSet event mapping.
2. EnergySpectralBaseline: Traditional acoustic DSP baseline (RMS + spectral centroid threshold).
3. EmbeddingFeatureExtractor: Backbone for extracting deep embeddings from waveforms.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from src.utils.audio_utils import compute_log_mel_spectrogram, load_audio


class EnergySpectralBaseline:
    """Traditional acoustic DSP baseline.

    Uses frame-level Root Mean Square (RMS) energy and Spectral Centroid
    to highlight why simple loudness/transient detectors fail on ordinary environmental audio.
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_len: int = 1024,
        hop_len: int = 512,
        energy_threshold: float = 0.15,
        centroid_threshold: float = 2200.0,
    ):
        self.sr = sr
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.energy_threshold = energy_threshold
        self.centroid_threshold = centroid_threshold

    def extract_features(self, waveform: np.ndarray) -> Tuple[float, float]:
        """Compute peak RMS energy and mean spectral centroid."""
        num_frames = (len(waveform) - self.frame_len) // self.hop_len + 1
        if num_frames <= 0:
            return float(np.sqrt(np.mean(waveform**2))), 1000.0

        rms_values = []
        centroids = []
        fft_freqs = np.fft.rfftfreq(self.frame_len, d=1.0 / self.sr)

        for i in range(num_frames):
            frame = waveform[i * self.hop_len : i * self.hop_len + self.frame_len]
            rms = np.sqrt(np.mean(frame**2))
            rms_values.append(rms)

            fft_mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            sum_mag = np.sum(fft_mag)
            if sum_mag > 1e-6:
                centroid = np.sum(fft_freqs * fft_mag) / sum_mag
            else:
                centroid = 0.0
            centroids.append(centroid)

        peak_rms = float(np.max(rms_values)) if rms_values else 0.0
        mean_centroid = float(np.mean(centroids)) if centroids else 0.0
        return peak_rms, mean_centroid

    def predict(self, waveform: np.ndarray, num_classes: int = 7) -> np.ndarray:
        """Heuristic baseline: if energy & centroid exceed threshold, trigger emergency classes."""
        peak_rms, centroid = self.extract_features(waveform)
        scores = np.zeros(num_classes, dtype=np.float32)

        # 0: screaming, 1: alarm, 2: glass_shatter, 3: smash_crash, 4: explosion, 5: gunshot, 6: background
        bg_idx = num_classes - 1
        if peak_rms > self.energy_threshold:
            if centroid > self.centroid_threshold:
                if num_classes > 0:
                    scores[0] = 0.65  # Screaming
                if num_classes > 1:
                    scores[1] = 0.60  # Alarm
            else:
                if num_classes > 5:
                    scores[5] = 0.70  # Gunshot
                if num_classes > 3:
                    scores[3] = 0.60  # Smash / Crash
        else:
            if bg_idx >= 0 and bg_idx < num_classes:
                scores[bg_idx] = 0.85  # Background

        return scores


class PretrainedAudioEmbeddingBackbone(nn.Module):
    """Deep convolutional embedding extractor for transfer learning."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """mel_spec shape: (batch, 1, n_mels, time_frames) -> (batch, embedding_dim)."""
        if mel_spec.ndim == 3:
            mel_spec = mel_spec.unsqueeze(1)
        x = self.features(mel_spec)
        return x.flatten(1)
