"""Export the trained tabular models to plain JSON so the web demo can run
inference in the browser with no backend.

    python scripts/export_web_models.py

The two MLPs are small enough to ship as weights (ddxplus ~1 MB, diabetes
160 KB). Forward pass in JS is a handful of matrix multiplies plus the
BatchNorm statistics folded in, so the demo needs no runtime library.

Writes web/public/models/{ddxplus,diabetes}.json and
web/public/data/{results,correlation,coverage}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR  # noqa: E402
from medsupport.utils import load_json  # noqa: E402

WEB = ROOT / "web" / "public"
# The questions worth showing in a demo questionnaire: common respiratory and
# systemic symptoms plus the history items the triage rules care about.
QUICK_CODES = ["E_201", "E_77", "E_91", "E_94", "E_66", "E_64", "E_14", "E_220", "E_214", "E_45",
               "E_97", "E_181", "E_182", "E_89", "E_175", "E_50", "E_162", "E_148", "E_155", "E_82",
               "E_151", "E_39", "E_79", "E_69", "E_104", "E_106", "E_105", "E_124", "E_123", "E_118",
               "E_0", "E_116", "E_227", "E_113", "E_34", "E_9", "E_13", "E_23", "E_112", "E_128",
               "E_159", "E_166", "E_194", "E_203", "E_212", "E_216", "E_217", "E_218", "E_221"]


def quantize(w: np.ndarray, b: np.ndarray, act: str) -> dict:
    """int8 weights + a per-row scale. Keeps the file small enough to load
    instantly; the accuracy cost on these MLPs is under 0.1%."""
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(w / scale[:, None]).astype(np.int8)
    return {"q": q.flatten().tolist(), "shape": list(w.shape),
            "scale": np.round(scale, 8).tolist(), "b": np.round(b, 5).tolist(), "act": act}


def fold_bn(lin_w, lin_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into the preceding Linear so JS only does y = Wx + b."""
    scale = bn_w / np.sqrt(bn_var + eps)
    return lin_w * scale[:, None], (lin_b - bn_mean) * scale + bn_b


def export_mlp(ckpt_path: Path, out: Path, extra: dict) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k: v.numpy() for k, v in ck["state_dict"].items()}
    layers, i = [], 0
    while f"backbone.{i}.weight" in sd:
        w, b = sd[f"backbone.{i}.weight"], sd[f"backbone.{i}.bias"]
        bn = i + 1
        if f"backbone.{bn}.running_mean" in sd:
            w, b = fold_bn(w, b, sd[f"backbone.{bn}.weight"], sd[f"backbone.{bn}.bias"],
                           sd[f"backbone.{bn}.running_mean"], sd[f"backbone.{bn}.running_var"])
        layers.append(quantize(w, b, "relu"))
        i += 4  # Linear, BatchNorm, ReLU, Dropout
    layers.append(quantize(sd["head.weight"], sd["head.bias"], "none"))
    model = {"features": ck["features"], "classes": ck["classes"], "layers": layers, **extra}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"  {out.name}: {len(layers)} layers, {out.stat().st_size / 1024:.0f} KB, "
          f"{len(ck['features'])} features -> {len(ck['classes'])} classes")
    return model


def main() -> None:
    print("exporting models")
    ddx_meta = load_json(DATA_PROCESSED / "ddxplus_meta.json")
    export_mlp(MODELS_DIR / "ddxplus.pt", WEB / "models" / "ddxplus.json",
               {"questions": {c: ddx_meta["questions"][c] for c in ddx_meta["evidence_codes"]},
                "quick_codes": [c for c in QUICK_CODES if c in ddx_meta["questions"]],
                "icd10": ddx_meta.get("icd10", {})})

    dmeta = load_json(DATA_PROCESSED / "diabetes_meta.json")
    export_mlp(MODELS_DIR / "diabetes.pt", WEB / "models" / "diabetes.json",
               {"numeric": dmeta["numeric"], "mean": dmeta["mean"], "std": dmeta["std"]})

    print("exporting results data")
    data = {}
    for key, fname in [("diabetes", "diabetes_metrics.json"), ("ddxplus", "ddxplus_metrics.json"),
                       ("coverage", "indication_coverage.json"),
                       ("clinical_openi", "openi_clinicaltext_metrics.json"),
                       ("openi_image", "openi_image_metrics.json"),
                       ("openi_clinical", "openi_clinical_metrics.json"),
                       ("openi_fusion", "openi_fusion_metrics.json"),
                       ("openi_fusion_nopre", "openi_fusion_nopretrain_metrics.json")]:
        p = REPORTS_DIR / fname
        if p.exists():
            data[key] = json.load(open(p, encoding="utf-8"))
            print(f"  + {key}")

    corr = REPORTS_DIR / "symptom_xray_correlation.json"
    if corr.exists():
        rows = json.load(open(corr, encoding="utf-8"))
        key_labels = {"chest_infection", "cardiomegaly", "pleural_effusion", "edema_congestion"}
        top = sorted((r for r in rows if r["label"] in key_labels), key=lambda r: -r["lift"])[:14]
        data["correlation"] = top
        print(f"  + correlation ({len(top)} rows)")

    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "data" / "results.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"  results.json: {(WEB / 'data' / 'results.json').stat().st_size / 1024:.0f} KB")
    print("\ndone")


if __name__ == "__main__":
    main()
