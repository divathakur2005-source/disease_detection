"""Compile every reports/*_metrics.json into reports/RESULTS.md (the table the
README and the presentation quote).

    python scripts/make_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import REPORTS_DIR, THORACIC_LABEL_NAMES  # noqa: E402


def load(name: str) -> dict | None:
    p = REPORTS_DIR / f"{name}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    out = ["# Results\n", "All numbers are on held-out **test** splits never used for training or model selection.\n"]

    # ---- tabular models -------------------------------------------------
    out.append("## 1. Clinical (tabular) models\n")
    out.append("| model | data | test rows | accuracy | macro-F1 | AUC | notes |")
    out.append("|---|---|---|---|---|---|---|")
    d = load("diabetes")
    if d:
        out.append(f"| Diabetes-risk MLP | Kaggle diabetes (100k) | {d['n']:,} | {d['accuracy']:.4f} | {d['f1_macro']:.4f} | "
                   f"{d['auc']:.4f} | class-weighted; recall(diabetic) = "
                   f"{d['confusion_matrix'][1][1] / max(1, sum(d['confusion_matrix'][1])):.3f} |")
    d = load("ddxplus")
    if d:
        out.append(f"| Symptom→diagnosis MLP | DDXPlus (49 dx) | {d['n']:,} | {d['accuracy']:.4f} | {d['f1_macro']:.4f} | "
                   f"{d['auc_macro_ovr']:.4f} | top-3 acc {d.get('top3_accuracy', float('nan')):.4f} |")
        pc = d.get("per_class", {})
        rows = [(k, v) for k, v in pc.items() if k in ("Pneumonia", "Bronchitis", "URTI", "Influenza", "Tuberculosis",
                                                       "Acute COPD exacerbation / infection", "Viral pharyngitis")]
        if rows:
            out.append("\nRespiratory classes of the symptom model (test):\n")
            out.append("| diagnosis | precision | recall | F1 | support |")
            out.append("|---|---|---|---|---|")
            for k, v in rows:
                out.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |")
    d = load("symptom_encoder")
    if d:
        out.append(f"\nSymptomEncoder pre-training (token encoder used by the fusion model) on DDXPlus: "
                   f"test accuracy {d['accuracy']:.4f}, macro-F1 {d['f1_macro']:.4f}.\n")

    # ---- Open-I ----------------------------------------------------------
    out.append("\n## 2. Chest X-ray models (Open-I, 8 thoracic labels, same patient split)\n")
    runs = []
    for p in sorted(REPORTS_DIR.glob("openi_*_metrics.json")):
        runs.append((p.stem.replace("_metrics", ""), json.load(open(p))))
    if runs:
        out.append("| model | macro AUC | macro-F1 (tuned thr) | " + " | ".join(f"AUC {l}" for l in THORACIC_LABEL_NAMES) + " |")
        out.append("|---|---|---|" + "---|" * len(THORACIC_LABEL_NAMES))
        for name, m in sorted(runs, key=lambda r: -(r[1]["test_at_0.5"]["auc_macro"] or 0)):
            per = m["test_at_0.5"]["per_label"]
            out.append(f"| {name.replace('openi_', '')} | {m['test_at_0.5']['auc_macro']:.3f} | {m['test_tuned']['f1_macro']:.3f} | "
                       + " | ".join(f"{per[l]['auc']:.3f}" if "auc" in per[l] else "-" for l in THORACIC_LABEL_NAMES) + " |")
        n = runs[0][1]["test_at_0.5"]["n"]
        sup = {l: runs[0][1]["test_at_0.5"]["per_label"][l]["support"] for l in THORACIC_LABEL_NAMES}
        out.append(f"\nTest set: {n} patients; positives per label: " + ", ".join(f"{l} {v}" for l, v in sup.items()) + ".\n")
    corr = REPORTS_DIR / "symptom_xray_correlation.md"
    if corr.exists():
        out.append("\n## 3. Symptom ↔ X-ray correlation (paired Open-I data)\n")
        out.append("Lift = P(X-ray label | symptom parsed from the indication) / P(X-ray label). Support n ≥ 20.\n")
        out.append(corr.read_text(encoding="utf-8"))
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
