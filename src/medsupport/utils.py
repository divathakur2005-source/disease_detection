"""Shared helpers: seeding, metrics, plots, json I/O."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None,
                       labels: Sequence[str] | None = None) -> dict:
    """Accuracy / macro-P / macro-R / macro-F1 (+ macro AUC when probabilities given)."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }
    if y_prob is not None:
        try:
            n_cls = y_prob.shape[1]
            if n_cls == 2:
                out["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                out["auc_macro_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                                                           labels=list(range(n_cls))))
        except ValueError:
            pass
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if labels is not None:
        out["labels"] = list(labels)
    return out


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str], thr: float = 0.5) -> dict:
    """Per-label and macro metrics for a multi-label (sigmoid) head."""
    y_pred = (y_prob >= thr).astype(int)
    per = {}
    aucs = []
    for j, name in enumerate(labels):
        t, p, pr = y_true[:, j], y_pred[:, j], y_prob[:, j]
        row = {
            "support": int(t.sum()),
            "precision": float(precision_score(t, p, zero_division=0)),
            "recall": float(recall_score(t, p, zero_division=0)),
            "f1": float(f1_score(t, p, zero_division=0)),
        }
        if 0 < t.sum() < len(t):
            row["auc"] = float(roc_auc_score(t, pr))
            aucs.append(row["auc"])
        per[name] = row
    return {
        "threshold": thr,
        "n": int(len(y_true)),
        "exact_match_accuracy": float((y_pred == y_true).all(axis=1).mean()),
        "f1_macro": float(np.mean([v["f1"] for v in per.values()])),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "auc_macro": float(np.mean(aucs)) if aucs else None,
        "per_label": per,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion(cm: np.ndarray | list, labels: Sequence[str], path: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * n + 2), max(4, 0.45 * n + 1.5)))
    sns.heatmap(cm, annot=n <= 25, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax,
                cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(history: dict, path: Path, title: str = "") -> None:
    """history = {"train_loss": [...], "val_loss": [...], "val_metric": [...]}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    for k, v in history.items():
        if k.startswith("val_") and k != "val_loss":
            axes[1].plot(v, label=k)
    axes[1].set_title("Validation metric")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best, self.bad, self.stop = None, 0, False

    def step(self, value: float) -> bool:
        """Returns True when `value` is a new best."""
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if better:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
            self.stop = self.bad >= self.patience
        return better
