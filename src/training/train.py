"""Training Pipeline for Audio-Only Emergency Sound Detection.

Implements multi-label classification using BCEWithLogitsLoss, positive class
imbalance weighting, train-only audio/spectrogram augmentation, checkpointing,
early stopping, and evaluation metric logging.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.models.classifier import CompactAudioCNN
from src.utils.audio_utils import apply_spec_augment


def set_seed(seed: int = 42) -> None:
    """Enforce reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AudioDataset(Dataset):
    """PyTorch Dataset loading precomputed log-mel spectrogram features."""

    def __init__(
        self,
        manifest_csv: str,
        num_classes: int = 8,
        is_training: bool = False,
        augment_spec: bool = True,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.num_classes = num_classes
        self.is_training = is_training
        self.augment_spec = augment_spec

        # Filter valid records
        self.records = []
        for _, row in self.df.iterrows():
            feat_path = row["feature_path"]
            if os.path.exists(feat_path):
                labels = [int(x) for x in row["multi_hot_labels"].split(",")]
                self.records.append((feat_path, labels))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_path, labels = self.records[idx]
        mel_spec = torch.load(feat_path, weights_only=True)  # (n_mels, time_frames)

        # Apply SpecAugment ONLY during training
        if self.is_training and self.augment_spec:
            mel_spec = apply_spec_augment(mel_spec, freq_mask_max=8, time_mask_max=16)

        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return mel_spec, label_tensor


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0
    for batch_feats, batch_labels in dataloader:
        batch_feats = batch_feats.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_feats)
        loss = criterion(logits, batch_labels)
        loss.backward()
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * len(batch_labels)

    return running_loss / max(1, len(dataloader.dataset))


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> Tuple[float, float, float, float, float]:
    """Evaluate validation loss, macro F1, micro F1, precision, and recall."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_feats, batch_labels in dataloader:
        batch_feats = batch_feats.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_feats)
        loss = criterion(logits, batch_labels)
        running_loss += loss.item() * len(batch_labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float().cpu().numpy()
        targets = batch_labels.cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets)

    val_loss = running_loss / max(1, len(dataloader.dataset))
    if not all_preds:
        return val_loss, 0.0, 0.0, 0.0, 0.0

    all_preds_arr = np.vstack(all_preds)
    all_targets_arr = np.vstack(all_targets)

    macro_f1 = float(f1_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0))
    micro_f1 = float(f1_score(all_targets_arr, all_preds_arr, average="micro", zero_division=0))
    precision = float(precision_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0))
    recall = float(recall_score(all_targets_arr, all_preds_arr, average="macro", zero_division=0))

    return val_loss, macro_f1, micro_f1, precision, recall


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str = "reports/training_curves.png",
) -> None:
    """Generate and save publication-quality 4-panel training/validation diagnostics."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    min_val_loss = history["val_loss"][best_epoch - 1]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: BCE Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], label="Train BCE Loss", color="#1f77b4", linewidth=2.2)
    ax1.plot(epochs, history["val_loss"], label="Val BCE Loss", color="#ff7f0e", linewidth=2.2, linestyle="--")
    ax1.scatter([best_epoch], [min_val_loss], color="#d62728", s=100, zorder=5, label=f"Best Checkpoint (Ep {best_epoch}: {min_val_loss:.3f})")
    ax1.set_title("A. Multi-Label BCE Loss Progression", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Macro & Micro F1 Scores
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="#2ca02c", linewidth=2.2)
    ax2.plot(epochs, history["val_micro_f1"], label="Val Micro F1", color="#9467bd", linewidth=2.0, linestyle=":")
    max_f1_idx = int(np.argmax(history["val_macro_f1"]))
    max_f1 = history["val_macro_f1"][max_f1_idx]
    ax2.scatter([max_f1_idx + 1], [max_f1], color="#2ca02c", s=100, zorder=5, label=f"Peak Macro F1 ({max_f1:.3f})")
    ax2.set_title("B. Validation Classification F1-Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Precision vs Recall
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["val_precision"], label="Val Macro Precision", color="#0284c7", linewidth=2.0)
    ax3.plot(epochs, history["val_recall"], label="Val Macro Recall", color="#e11d48", linewidth=2.0, linestyle="--")
    ax3.set_title("C. Validation Precision & Recall Trade-Off", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score")
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Learning Rate Decay
    ax4 = axes[1, 1]
    ax4.plot(epochs, history["learning_rate"], label="Learning Rate (AdamW)", color="#d97706", linewidth=2.0)
    ax4.set_title("D. Learning Rate Adaptation Schedule", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    ax4.set_yscale("log")
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("CompactAudioCNN Model Training Diagnostics (AudioSet 2017)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Publication-grade training curves saved to {output_path}")


def train_model(config_path: str = "configs/default.yaml") -> Dict[str, any]:
    """Full model training execution."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    device_str = cfg["project"]["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training using compute device: {device}")

    # Load stats & splits
    manifests_dir = cfg["paths"]["manifests_dir"]
    train_csv = os.path.join(manifests_dir, "train_split.csv")
    val_csv = os.path.join(manifests_dir, "val_split.csv")

    stats_path = os.path.join(cfg["paths"]["processed_dir"], "dataset_stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    num_classes = stats["num_classes"]
    class_names = stats["class_names"]
    pos_weights = stats["pos_weights"]

    # Datasets and Loaders
    batch_size = cfg["training"]["batch_size"]
    train_dataset = AudioDataset(train_csv, num_classes=num_classes, is_training=True)
    val_dataset = AudioDataset(val_csv, num_classes=num_classes, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Initialize Compact Audio CNN Model
    model = CompactAudioCNN(
        num_classes=num_classes,
        n_mels=cfg["audio"]["n_mels"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Imbalance-weighted Loss Function
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(artifacts_dir, "best_emergency_model.pt")

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
        "val_precision": [],
        "val_recall": [],
        "learning_rate": [],
    }

    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_macro_f1, val_micro_f1, val_prec, val_rec = evaluate_epoch(
            model, val_loader, criterion, device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))
        history["val_micro_f1"].append(round(val_micro_f1, 4))
        history["val_precision"].append(round(val_prec, 4))
        history["val_recall"].append(round(val_rec, 4))
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Macro F1: {val_macro_f1:.4f} | "
            f"Micro F1: {val_micro_f1:.4f} | "
            f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "val_micro_f1": val_micro_f1,
                    "class_names": class_names,
                    "config": cfg,
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history and plot curves
    plot_training_history(
        history, output_path=os.path.join(cfg["paths"]["reports_dir"], "training_curves.png")
    )
    with open(os.path.join(cfg["paths"]["reports_dir"], "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "best_checkpoint": best_checkpoint_path,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Audio-Only Emergency Sound Detector.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
