"""Clinical-only baseline on Open-I: can the parsed symptoms alone predict the
radiologist's findings?

This runs without any images, so it can be evaluated before the radiographs
finish downloading. It is also the honest lower bound the fusion model must
beat: if symptoms alone already reached the fusion model's accuracy, the image
branch would be adding nothing.

    python scripts/baseline_clinical_openi.py

Writes reports/openi_clinical_textonly_metrics.json and the probabilities, in
the same format scripts/compare_openi.py expects.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SEED, THORACIC_LABEL_NAMES  # noqa: E402
from medsupport.models.clinical import SymptomEncoderClassifier  # noqa: E402
from medsupport.train.imaging import multilabel_metrics_thr, tune_thresholds  # noqa: E402
from medsupport.utils import EarlyStopping, device, load_json, multilabel_metrics, save_json, seed_everything  # noqa: E402

PREFIX = "openi_noimg"


def main(epochs: int = 60, lr: float = 1e-3, bs: int = 128, pretrain: bool = True) -> dict:
    seed_everything(SEED)
    dev = device()
    meta = load_json(DATA_PROCESSED / f"{PREFIX}_meta.json")
    vocab = meta["vocab"]
    parts = {p: pd.read_parquet(DATA_PROCESSED / f"{PREFIX}_{p}.parquet") for p in ("train", "val", "test")}
    ycols = [f"y_{n}" for n in THORACIC_LABEL_NAMES]
    data = {p: (torch.tensor(df[vocab].to_numpy(), dtype=torch.float32),
                torch.tensor(df[ycols].to_numpy(), dtype=torch.float32)) for p, df in parts.items()}
    print(f"[clinical-textonly] train {len(parts['train'])} val {len(parts['val'])} test {len(parts['test'])} "
          f"vocab {len(vocab)} device {dev}")

    model = SymptomEncoderClassifier(len(vocab), len(THORACIC_LABEL_NAMES), dim=256, dropout=0.3).to(dev)
    if pretrain:
        ck = torch.load(MODELS_DIR / "symptom_encoder.pt", map_location="cpu", weights_only=False)
        enc = {k[len("encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")}
        # the DDXPlus encoder was trained on 231 codes; this vocabulary has a few more extras
        emb = enc["symptom_emb.weight"]
        if emb.shape[0] < len(vocab):
            pad = torch.zeros(len(vocab) - emb.shape[0], emb.shape[1])
            nn.init.trunc_normal_(pad, std=0.02)
            enc["symptom_emb.weight"] = torch.cat([emb, pad], 0)
            enc["mlp.0.weight"] = torch.cat(
                [enc["mlp.0.weight"], torch.zeros(enc["mlp.0.weight"].shape[0], len(vocab) - emb.shape[0])], 1)
        model.encoder.load_state_dict(enc)
        print(f"  loaded DDXPlus-pretrained encoder (padded {emb.shape[0]} -> {len(vocab)} codes)")

    ytr = data["train"][1].numpy()
    pos = ytr.sum(0)
    pos_weight = torch.tensor(np.clip((len(ytr) - pos) / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr, Ytr = (t.to(dev) for t in data["train"])
    stopper, best_state, t0 = EarlyStopping(patience=10, mode="max"), None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(data["val"][0].to(dev))).cpu().numpy()
        m = multilabel_metrics(data["val"][1].numpy(), pv, THORACIC_LABEL_NAMES)
        auc = m["auc_macro"] or 0.0
        if stopper.step(auc):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep + 1:02d} loss {tot / len(Xtr):.4f} val_auc {auc:.4f} f1 {m['f1_macro']:.4f}")
        if stopper.stop:
            print("  early stop")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = torch.sigmoid(model(data["test"][0].to(dev))).cpu().numpy()
        pva = torch.sigmoid(model(data["val"][0].to(dev))).cpu().numpy()
    yte, yva = data["test"][1].numpy(), data["val"][1].numpy()
    thr = tune_thresholds(yva, pva)
    at05 = multilabel_metrics(yte, pte, THORACIC_LABEL_NAMES)
    tuned = multilabel_metrics_thr(yte, pte, THORACIC_LABEL_NAMES, thr)
    result = {"mode": "clinical_textonly", "pretrain": pretrain, "test_at_0.5": at05, "test_tuned": tuned,
              "best_val_auc_macro": stopper.best, "train_seconds": round(time.time() - t0, 1),
              "thresholds": dict(zip(THORACIC_LABEL_NAMES, thr.tolist())),
              "note": "symptoms parsed from the referral indication only - no image"}
    save_json(result, REPORTS_DIR / "openi_clinicaltext_metrics.json")
    np.save(REPORTS_DIR / "openi_clinicaltext_test_probs.npy", pte)
    np.save(REPORTS_DIR / "openi_noimg_test_labels.npy", yte)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "labels": THORACIC_LABEL_NAMES,
                "thresholds": thr.tolist()}, MODELS_DIR / "openi_clinicaltext.pt")
    print(f"\n[clinical-textonly] TEST macro-AUC {at05['auc_macro']:.4f}  macro-F1(tuned) {tuned['f1_macro']:.4f}")
    for lab, row in at05["per_label"].items():
        print(f"    {lab:18s} auc {row.get('auc', float('nan')):.3f}  f1 {row['f1']:.3f}  support {row['support']}")
    return result


if __name__ == "__main__":
    main()
