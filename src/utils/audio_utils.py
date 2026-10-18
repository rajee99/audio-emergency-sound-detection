"""Audio Utilities Module for Emergency Sound Detection System.

Provides robust audio I/O, resampling, normalization, log-mel spectrogram
computation, data augmentation, and sliding-window segmentation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.signal
import soundfile as sf
import torch


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    target_duration: Optional[float] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, resample to target_sr, and optionally

    pad/truncate to target_duration.

    Args:
        file_path: Path to the audio file (WAV, FLAC, OGG, etc.)
        target_sr: Desired sample rate (default 16000 Hz)
        mono: If True, multi-channel audio is averaged to single channel
        target_duration: If provided, pads or truncates audio to exactly this
          many seconds
        normalize: If True, scales amplitude to peak 0.95 to avoid clipping

    Returns:
        waveform: 1D float32 numpy array
        sr: Sample rate (equal to target_sr)
    """
    data, sr = sf.read(file_path, dtype="float32")

    # Multi-channel to mono
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rates do not match
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = target_sr

    # Pad or truncate to target duration if requested
    if target_duration is not None:
        target_samples = int(round(target_duration * target_sr))
        current_samples = len(data)
        if current_samples < target_samples:
            pad_width = target_samples - current_samples
            data = np.pad(data, (0, pad_width), mode="constant")
        elif current_samples > target_samples:
            data = data[:target_samples]

    # Peak normalization
    if normalize:
        max_val = np.max(np.abs(data))
        if max_val > 1e-6:
            data = (data / max_val) * 0.95

    return data.astype(np.float32), sr


def save_wav(file_path: str, waveform: np.ndarray, sample_rate: int = 16000) -> None:
    """Save a 1D float32 numpy array to WAV format."""
    clipped = np.clip(waveform, -1.0, 1.0)
    sf.write(file_path, clipped, sample_rate, subtype="PCM_16")


def compute_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 64,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Create a Mel frequency filterbank matrix using triangular filters."""
    # Convert Hz to Mel
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    # Convert Mel to Hz
    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                filterbank[m - 1, k] = (k - bin_points[m - 1]) / (
                    bin_points[m] - bin_points[m - 1]
                )
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                filterbank[m - 1, k] = (bin_points[m + 1] - k) / (
                    bin_points[m + 1] - bin_points[m]
                )

    return filterbank.astype(np.float32)


def compute_log_mel_spectrogram(
    waveform: Union[np.ndarray, torch.Tensor],
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 1024,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Compute Log-Mel Spectrogram from raw audio waveform.

    Output shape: (n_mels, time_frames)
    """
    if isinstance(waveform, np.ndarray):
        waveform_tensor = torch.from_numpy(waveform).float()
    else:
        waveform_tensor = waveform.float()

    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)  # Shape (1, samples)

    window = torch.hann_window(win_length)
    stft = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Power spectrogram
    power_spec = torch.abs(stft) ** 2.0  # Shape (1, freq_bins, time_frames)

    # Mel filterbank
    fb = compute_mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
    fb_tensor = torch.from_numpy(fb).float()  # Shape (n_mels, freq_bins)

    # Matrix multiplication: (n_mels, freq_bins) x (1, freq_bins, time_frames) -> (1, n_mels, time_frames)
    mel_spec = torch.matmul(fb_tensor, power_spec.squeeze(0))  # Shape (n_mels, time_frames)

    # Log-amplitude compression (dB)
    log_mel = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-6))
    return log_mel


# --- Audio Augmentation Functions (Training Only) ---


def add_gaussian_noise(
    waveform: np.ndarray,
    snr_db_range: Tuple[float, float] = (15.0, 30.0),
) -> np.ndarray:
    """Add additive white Gaussian noise at a random Signal-to-Noise Ratio (SNR)."""
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    sig_power = np.mean(waveform**2)
    if sig_power < 1e-8:
        return waveform
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(waveform)).astype(
        np.float32
    )
    augmented = waveform + noise
    return np.clip(augmented, -1.0, 1.0)


def apply_gain(
    waveform: np.ndarray,
    gain_db_range: Tuple[float, float] = (-6.0, 6.0),
) -> np.ndarray:
    """Apply random gain variation within [gain_min_db, gain_max_db]."""
    gain_db = np.random.uniform(gain_db_range[0], gain_db_range[1])
    gain_linear = 10.0 ** (gain_db / 20.0)
    augmented = waveform * gain_linear
    return np.clip(augmented, -1.0, 1.0)


def time_shift(
    waveform: np.ndarray,
    max_shift_sec: float = 0.5,
    sr: int = 16000,
) -> np.ndarray:
    """Random circular time-shift of the audio waveform."""
    max_samples = int(max_shift_sec * sr)
    shift = np.random.randint(-max_samples, max_samples + 1)
    return np.roll(waveform, shift)


def apply_spec_augment(
    mel_spec: torch.Tensor,
    freq_mask_max: int = 8,
    time_mask_max: int = 16,
) -> torch.Tensor:
    """Apply SpecAugment frequency and time masking to log-mel spectrogram tensor.

    mel_spec shape: (n_mels, time_frames)
    """
    augmented = mel_spec.clone()
    n_mels, time_frames = augmented.shape

    # Frequency masking
    f_width = np.random.randint(0, freq_mask_max + 1)
    if f_width > 0:
        f_start = np.random.randint(0, max(1, n_mels - f_width))
        augmented[f_start : f_start + f_width, :] = -80.0  # mask with silence dB

    # Time masking
    t_width = np.random.randint(0, time_mask_max + 1)
    if t_width > 0:
        t_start = np.random.randint(0, max(1, time_frames - t_width))
        augmented[:, t_start : t_start + t_width] = -80.0

    return augmented


# --- Sliding Window Segmentation ---


def slice_into_windows(
    waveform: np.ndarray,
    sr: int = 16000,
    window_sec: float = 0.96,
    hop_sec: float = 0.48,
) -> List[Tuple[float, float, np.ndarray]]:
    """Segment an arbitrary-length audio waveform into overlapping windows.

    Returns:
        List of tuples: (start_time_sec, end_time_sec, window_waveform)
    """
    win_len = int(round(window_sec * sr))
    hop_len = int(round(hop_sec * sr))
    total_len = len(waveform)

    if total_len < win_len:
        # Pad short file to full window length
        padded = np.pad(waveform, (0, win_len - total_len), mode="constant")
        return [(0.0, window_sec, padded)]

    windows = []
    start_idx = 0
    while start_idx + win_len <= total_len:
        end_idx = start_idx + win_len
        start_sec = start_idx / sr
        end_sec = end_idx / sr
        chunk = waveform[start_idx:end_idx]
        windows.append((start_sec, end_sec, chunk))
        start_idx += hop_len

    # Handle remaining tail if non-empty
    if start_idx < total_len and total_len - start_idx >= (win_len // 4):
        tail = waveform[total_len - win_len :]
        windows.append(((total_len - win_len) / sr, total_len / sr, tail))

    return windows
