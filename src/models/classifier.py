"""Neural Network Architectures for Emergency Sound Detection.

Implements:
1. CompactAudioCNN: Lightweight Convolutional Neural Network operating on Log-Mel Spectrograms.
2. AudioMLP: Dense embedding classifier head.
3. PretrainedAudioBackbone: Feature extractor for transfer learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block: Conv2D -> BatchNorm -> ReLU -> MaxPool2D -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: tuple[int, int] = (2, 2),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CompactAudioCNN(nn.Module):
    """Compact 4-layer CNN for multi-label acoustic event classification.

    Input shape:  (batch, 1, n_mels, time_frames)
    Output shape: (batch, num_classes) raw logits (unnormalized for BCEWithLogitsLoss)
    """

    def __init__(
        self,
        num_classes: int = 8,
        n_mels: int = 64,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels

        # Feature extraction backbone
        self.block1 = ConvBlock(1, 32, pool_size=(2, 2), dropout=0.15)
        self.block2 = ConvBlock(32, 64, pool_size=(2, 2), dropout=0.20)
        self.block3 = ConvBlock(64, 128, pool_size=(2, 2), dropout=0.25)
        self.block4 = ConvBlock(128, embedding_dim, pool_size=(2, 2), dropout=dropout)

        # Global pooling: aggregates across both frequency and time dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-label classification head
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck embedding vector."""
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (batch, 1, n_mels, time_frames)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)  # (batch, embedding_dim, 1, 1)
        return x.flatten(1)  # (batch, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (for numerical stability in BCEWithLogitsLoss)."""
        feat = self.extract_features(x)
        h = self.fc1(feat)
        if h.shape[0] > 1:
            h = self.bn_fc(h)
        h = self.relu(h)
        h = self.drop_fc(h)
        logits = self.classifier(h)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid multi-label probabilities in [0.0, 1.0]."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)


class AudioMLP(nn.Module):
    """Dense classifier for precomputed audio embeddings."""

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)
