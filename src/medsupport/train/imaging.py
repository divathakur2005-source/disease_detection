"""Open-I trainer: image-only, clinical-only and cross-modal fusion.

    python -m medsupport.train.imaging image                      # DenseNet-121 baseline
    python -m medsupport.train.imaging clinical                   # SymptomEncoder (pre-trained on DDXPlus) baseline
    python -m medsupport.train.imaging fusion                     # cross-modal attention fusion
    python -m medsupport.train.imaging fusion --no-pretrain       # ablation: fusion without DDXPlus pre-training

All three predict the same 8 multi-label thoracic findings on the same
patient split, so their metrics are directly comparable
(reports/openi_<mode>_metrics.json, reports/openi_<mode>_test_probs.npy).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from medsupport.config import (DATA_PROCESSED, FIGURES_DIR, IMG_SIZE, MODELS_DIR, REPORTS_DIR, SEED,
                               THORACIC_LABEL_NAMES)
from medsupport.models.clinical import SymptomEncoderClassifier
from medsupport.models.fusion import CrossModalFusionNet
from medsupport.models.imaging import ImageEncoder
from medsupport.utils import EarlyStopping, device, load_json, multilabel_metrics, plot_curves, save_json, seed_everything

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomRotation(7),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])

# On Windows, DataLoader workers are *spawned*, so each one re-imports torch and
# re-reserves commit charge for the CUDA DLLs. On a 16 GB laptop that exhausts the
# pagefile and torch dies with "[WinError 1455] The paging file is too small"
# before training starts. Loading batches in the main process avoids the extra
# torch imports entirely; the GPU stays the bottleneck, so throughput is similar.
DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4


class OpenIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: list[str], train: bool, with_image: bool = True):
        self.paths = df["image_path"].tolist()
        self.sym = df[vocab].to_numpy().astype(np.float32)
        self.y = df[[f"y_{n}" for n in THORACIC_LABEL_NAMES]].to_numpy().astype(np.float32)
        self.tf = TRAIN_TF if train else EVAL_TF
        self.with_image = with_image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.with_image:
            img = Image.open(DATA_PROCESSED / self.paths[i]).convert("RGB")
            img = self.tf(img)
        else:
            img = torch.zeros(1)
        return img, torch.from_numpy(self.sym[i]), torch.from_numpy(self.y[i])


def build_model(mode: str, vocab_size: int, n_labels: int, pretrain: bool, init_image: str | None):
    if mode == "image":
        return ImageEncoder(n_classes=n_labels, pretrained=True)
    if mode == "clinical":
        m = SymptomEncoderClassifier(vocab_size, n_labels, dim=256, dropout=0.3)
        if pretrain:
            ck = torch.load(MODELS_DIR / "symptom_encoder.pt", map_location="cpu", weights_only=False)
            enc = {k[len("encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")}
            m.encoder.load_state_dict(enc)
            print("  loaded DDXPlus-pretrained SymptomEncoder")
        return m
    if mode == "fusion":
        m = CrossModalFusionNet(vocab_size, n_labels, dim=256, pretrained=True, freeze_until=None)
        if pretrain:
            ck = torch.load(MODELS_DIR / "symptom_encoder.pt", map_location="cpu", weights_only=False)
            enc = {k[len("encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")}
            m.clinical.load_state_dict(enc)
            print("  loaded DDXPlus-pretrained SymptomEncoder")
        if init_image:
            ck = torch.load(init_image, map_location="cpu", weights_only=False)
            feat = {k[len("features."):]: v for k, v in ck["state_dict"].items() if k.startswith("features.")}
            m.image.features.load_state_dict(feat)
            head = {k[len("head."):]: v for k, v in ck["state_dict"].items() if k.startswith("head.")}
            m.img_head.load_state_dict(head)
            print("  initialised image branch from", init_image)
        return m
    raise ValueError(mode)


def forward_logits(model, mode, img, sym):
    if mode == "image":
        return model(img), None
    if mode == "clinical":
        return model(sym), None
    out = model(img, sym)
    return out["logits"], out


@torch.no_grad()
def predict(model, mode, loader, dev):
    model.eval()
    probs, ys = [], []
    for img, sym, y in loader:
        img, sym = img.to(dev, non_blocking=True), sym.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits, _ = forward_logits(model, mode, img, sym)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def run(mode: str, epochs: int, bs: int, lr: float, lr_backbone: float, pretrain: bool, init_image: str | None,
        aux_weight: float, tag: str, workers: int) -> dict:
    seed_everything(SEED)
    dev = device()
    meta = load_json(DATA_PROCESSED / "openi_meta.json")
    vocab = meta["vocab"]
    n_labels = len(THORACIC_LABEL_NAMES)
    parts = {p: pd.read_parquet(DATA_PROCESSED / f"openi_{p}.parquet") for p in ("train", "val", "test")}
    with_image = mode != "clinical"
    dl = {}
    for p, df in parts.items():
        ds = OpenIDataset(df, vocab, train=(p == "train"), with_image=with_image)
        dl[p] = DataLoader(ds, batch_size=bs, shuffle=(p == "train"), num_workers=workers if with_image else 0,
                           pin_memory=dev.type == "cuda", persistent_workers=with_image and workers > 0,
                           drop_last=(p == "train"))
    name = f"openi_{mode}{tag}"
    print(f"[{name}] train {len(parts['train'])} val {len(parts['val'])} test {len(parts['test'])} "
          f"labels {n_labels} vocab {len(vocab)} device {dev}")

    model = build_model(mode, len(vocab), n_labels, pretrain, init_image).to(dev)

    # class imbalance: pos_weight = neg/pos per label, capped
    ytr = parts["train"][[f"y_{n}" for n in THORACIC_LABEL_NAMES]].to_numpy()
    pos = ytr.sum(0)
    pos_weight = torch.tensor(np.clip((len(ytr) - pos) / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    backbone_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if ("features" in n) else other_params).append(p)
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": lr_backbone}, {"params": other_params, "lr": lr}],
                            weight_decay=1e-4)
    steps = epochs * max(1, len(dl["train"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_backbone, lr], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "val_auc_macro": [], "val_f1_macro": []}
    stopper = EarlyStopping(patience=5, mode="max")
    best_state, t0 = None, time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for img, sym, y in dl["train"]:
            img, sym, y = img.to(dev, non_blocking=True), sym.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits, out = forward_logits(model, mode, img, sym)
                loss = criterion(logits.float(), y)
                if out is not None and aux_weight > 0:
                    loss = loss + aux_weight * (criterion(out["img_logits"].float(), y) +
                                                criterion(out["cli_logits"].float(), y))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        p_va, y_va = predict(model, mode, dl["val"], dev)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(p_va).clamp(1e-6, 1 - 1e-6),
                                                            torch.from_numpy(y_va)))
        m = multilabel_metrics(y_va, p_va, THORACIC_LABEL_NAMES)
        auc = m["auc_macro"] or 0.0
        history["train_loss"].append(tot / n)
        history["val_loss"].append(val_loss)
        history["val_auc_macro"].append(auc)
        history["val_f1_macro"].append(m["f1_macro"])
        if stopper.step(auc):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep + 1:02d} loss {tot / n:.4f} val_loss {val_loss:.4f} val_auc {auc:.4f} "
              f"val_f1 {m['f1_macro']:.4f} ({time.time() - t0:.0f}s) {'*' if stopper.bad == 0 else ''}")
        if stopper.stop:
            print("  early stop")
            break

    model.load_state_dict(best_state)
    p_te, y_te = predict(model, mode, dl["test"], dev)
    # pick per-label thresholds on the validation set (maximise F1), report both 0.5 and tuned
    p_va, y_va = predict(model, mode, dl["val"], dev)
    thr = tune_thresholds(y_va, p_va)
    metrics = multilabel_metrics(y_te, p_te, THORACIC_LABEL_NAMES)
    metrics_tuned = multilabel_metrics_thr(y_te, p_te, THORACIC_LABEL_NAMES, thr)
    result = {
        "mode": mode, "tag": tag, "pretrain": pretrain, "init_image": init_image, "epochs_run": len(history["train_loss"]),
        "best_val_auc_macro": stopper.best, "train_seconds": round(time.time() - t0, 1),
        "test_at_0.5": metrics, "test_tuned": metrics_tuned, "thresholds": dict(zip(THORACIC_LABEL_NAMES, thr.tolist())),
        "lr": lr, "lr_backbone": lr_backbone, "batch_size": bs, "aux_weight": aux_weight,
    }
    save_json(result, REPORTS_DIR / f"{name}_metrics.json")
    np.save(REPORTS_DIR / f"{name}_test_probs.npy", p_te)
    np.save(REPORTS_DIR / "openi_test_labels.npy", y_te)
    torch.save({"state_dict": model.state_dict(), "mode": mode, "vocab": vocab, "labels": THORACIC_LABEL_NAMES,
                "thresholds": thr.tolist(), "pretrain": pretrain}, MODELS_DIR / f"{name}.pt")
    plot_curves(history, FIGURES_DIR / f"{name}_curves.png", title=name)
    print(f"[{name}] TEST auc_macro {metrics['auc_macro']:.4f} f1_macro@0.5 {metrics['f1_macro']:.4f} "
          f"f1_macro@tuned {metrics_tuned['f1_macro']:.4f}")
    for lab, row in metrics["per_label"].items():
        print(f"    {lab:18s} auc {row.get('auc', float('nan')):.3f} f1 {row['f1']:.3f} support {row['support']}")
    return result


def tune_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 19)
    thr = np.full(y.shape[1], 0.5)
    for j in range(y.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            f1s = [f1_score(y[:, j], (p[:, j] >= t).astype(int), zero_division=0) for t in grid]
            thr[j] = grid[int(np.argmax(f1s))]
    return thr


def multilabel_metrics_thr(y, p, labels, thr):
    from sklearn.metrics import f1_score, precision_score, recall_score
    pred = (p >= thr[None, :]).astype(int)
    per = {}
    for j, n in enumerate(labels):
        per[n] = {"precision": float(precision_score(y[:, j], pred[:, j], zero_division=0)),
                  "recall": float(recall_score(y[:, j], pred[:, j], zero_division=0)),
                  "f1": float(f1_score(y[:, j], pred[:, j], zero_division=0)), "threshold": float(thr[j])}
    return {"f1_macro": float(np.mean([v["f1"] for v in per.values()])),
            "f1_micro": float(f1_score(y, pred, average="micro", zero_division=0)), "per_label": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["image", "clinical", "fusion"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--no-pretrain", action="store_true", help="do not load the DDXPlus-pretrained SymptomEncoder")
    ap.add_argument("--init-image", type=str, default=None, help="checkpoint of the image-only model to init fusion")
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="DataLoader worker processes (default 0 on Windows: spawned workers "
                         "re-import torch and can exhaust the pagefile)")
    a = ap.parse_args()
    run(a.mode, a.epochs, a.bs, a.lr, a.lr_backbone, not a.no_pretrain, a.init_image, a.aux_weight, a.tag, a.workers)


if __name__ == "__main__":
    main()
