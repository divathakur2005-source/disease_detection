"""Generic trainer for the tabular (clinical) MLPs.

    python -m medsupport.train.tabular diabetes
    python -m medsupport.train.tabular ddxplus
    python -m medsupport.train.tabular ddxplus_concepts   # pretrains the SymptomEncoder vocabulary model

Writes models/<name>.pt and reports/<name>_metrics.json (+ confusion matrix /
training curves under reports/figures/).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medsupport.config import DATA_PROCESSED, EXTRA_EVIDENCES, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from medsupport.models.clinical import ClinicalMLP, SymptomEncoderClassifier
from medsupport.utils import (EarlyStopping, device, load_json, multiclass_metrics, plot_confusion,
                              plot_curves, save_json, seed_everything)

TASKS = {
    # name: dict(prefix=parquet prefix, target=target column, meta=meta json, weighted=class weighting,
    #            arch="mlp" | "encoder", drop=columns not fed to the model)
    "diabetes": dict(prefix="diabetes", target="diabetes", meta="diabetes_meta.json", weighted=True, arch="mlp",
                     drop=()),
    "ddxplus": dict(prefix="ddxplus", target="disease_id", meta="ddxplus_meta.json", weighted=False, arch="mlp",
                    drop=()),
    # Pre-training of the SymptomEncoder used by the fusion model: evidence multi-hot only (no age/sex),
    # vocabulary padded with the Open-I-only extra tokens so the embedding table matches the fusion vocab.
    "symptom_encoder": dict(prefix="ddxplus", target="disease_id", meta="ddxplus_meta.json", weighted=False,
                            arch="encoder", drop=("age_norm", "sex_male")),
}


def load_split(prefix: str, part: str, target: str, drop: tuple[str, ...] = (), pad_extras: bool = False):
    df = pd.read_parquet(DATA_PROCESSED / f"{prefix}_{part}.parquet")
    y = df[target].to_numpy().astype(np.int64)
    df = df.drop(columns=[target, *drop])
    X = df.to_numpy().astype(np.float32)
    feats = list(df.columns)
    if pad_extras:
        X = np.concatenate([X, np.zeros((len(X), len(EXTRA_EVIDENCES)), dtype=np.float32)], axis=1)
        feats += list(EXTRA_EVIDENCES)
    return X, y, feats


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, bs: int = 4096) -> np.ndarray:
    model.eval()
    dev = device()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def run(task: str, epochs: int, bs: int, lr: float, hidden: tuple[int, ...], dropout: float) -> dict:
    seed_everything(SEED)
    cfg = TASKS[task]
    prefix, target, meta_name, weighted, arch, drop = (cfg["prefix"], cfg["target"], cfg["meta"], cfg["weighted"],
                                                       cfg["arch"], cfg["drop"])
    pad = arch == "encoder"
    meta = load_json(DATA_PROCESSED / meta_name)
    X_tr, y_tr, feats = load_split(prefix, "train", target, drop, pad)
    X_va, y_va, _ = load_split(prefix, "val", target, drop, pad)
    X_te, y_te, _ = load_split(prefix, "test", target, drop, pad)
    n_classes = int(max(y_tr.max(), y_va.max(), y_te.max()) + 1)
    class_names = meta.get("classes", [str(i) for i in range(n_classes)])
    dev = device()
    print(f"[{task}] train {X_tr.shape} val {X_va.shape} test {X_te.shape} classes {n_classes} arch {arch} device {dev}")

    if arch == "encoder":
        model = SymptomEncoderClassifier(X_tr.shape[1], n_classes, dim=hidden[0], dropout=dropout).to(dev)
    else:
        model = ClinicalMLP(X_tr.shape[1], n_classes, hidden=hidden, dropout=dropout).to(dev)
    weight = None
    if weighted:
        counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        weight = torch.tensor(counts.sum() / (n_classes * counts), device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=bs,
                        shuffle=True, drop_last=False)

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    stopper = EarlyStopping(patience=6, mode="max")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        p_va = predict(model, X_va)
        with torch.no_grad():
            val_loss = float(criterion(torch.log(torch.from_numpy(p_va).to(dev) + 1e-9), torch.from_numpy(y_va).to(dev)))
        m = multiclass_metrics(y_va, p_va.argmax(1))
        history["train_loss"].append(tot / n)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(m["f1_macro"])
        if stopper.step(m["f1_macro"]):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep + 1:02d} loss {tot / n:.4f} val_loss {val_loss:.4f} val_acc {m['accuracy']:.4f} "
              f"val_f1 {m['f1_macro']:.4f} {'*' if stopper.bad == 0 else ''}")
        if stopper.stop:
            print("  early stop")
            break

    model.load_state_dict(best_state)
    p_te = predict(model, X_te)
    metrics = multiclass_metrics(y_te, p_te.argmax(1), p_te, labels=class_names)
    metrics.update({"task": task, "features": feats, "n_features": len(feats), "n_classes": n_classes,
                    "epochs_run": len(history["train_loss"]), "train_seconds": round(time.time() - t0, 1),
                    "hidden": list(hidden), "dropout": dropout, "lr": lr, "batch_size": bs})
    if n_classes > 2:
        from sklearn.metrics import classification_report
        rep = classification_report(y_te, p_te.argmax(1), target_names=class_names, output_dict=True, zero_division=0)
        metrics["per_class"] = {k: v for k, v in rep.items() if k in class_names}
        # top-k accuracy is the clinically relevant number for a differential diagnosis
        topk = np.argsort(-p_te, axis=1)[:, :3]
        metrics["top3_accuracy"] = float((topk == y_te[:, None]).any(1).mean())

    torch.save({"state_dict": model.state_dict(), "in_dim": X_tr.shape[1], "n_classes": n_classes,
                "hidden": list(hidden), "dropout": dropout, "features": feats, "classes": class_names,
                "arch": arch},
               MODELS_DIR / f"{task}.pt")
    save_json(metrics, REPORTS_DIR / f"{task}_metrics.json")
    plot_confusion(metrics["confusion_matrix"], class_names, FIGURES_DIR / f"{task}_confusion.png",
                   title=f"{task} - test confusion matrix")
    plot_curves(history, FIGURES_DIR / f"{task}_curves.png", title=task)
    print(f"[{task}] TEST acc {metrics['accuracy']:.4f} f1_macro {metrics['f1_macro']:.4f} "
          f"auc {metrics.get('auc', metrics.get('auc_macro_ovr', float('nan'))):.4f}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--dropout", type=float, default=0.3)
    a = ap.parse_args()
    run(a.task, a.epochs, a.bs, a.lr, tuple(int(h) for h in a.hidden.split(",")), a.dropout)


if __name__ == "__main__":
    main()
