"""Side-by-side comparison of the Open-I models (image-only vs clinical-only vs
fusion [+ ablations]). Reads reports/openi_*_metrics.json and the saved test
probabilities, writes reports/openi_comparison.json / .md and ROC figures.

    python scripts/compare_openi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import FIGURES_DIR, REPORTS_DIR, THORACIC_LABEL_NAMES  # noqa: E402


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.load(REPORTS_DIR / "openi_test_labels.npy")

    # Only runs scored on the CURRENT test split are comparable. Stale artifacts
    # from an earlier split (e.g. before the indication vocabulary was expanded)
    # still match the glob, so skip any whose row count disagrees with the labels
    # instead of letting a non-comparable AUC into the table.
    runs, skipped = {}, []
    for p in sorted(REPORTS_DIR.glob("openi_*_metrics.json")):
        name = p.stem.replace("_metrics", "")
        probs = REPORTS_DIR / f"{name}_test_probs.npy"
        if not probs.exists():
            continue
        pr = np.load(probs)
        if pr.shape[0] != y.shape[0]:
            skipped.append(f"{name} ({pr.shape[0]} rows vs {y.shape[0]} in current split)")
            continue
        runs[name] = {"metrics": json.load(open(p)), "probs": pr}
    if skipped:
        print("  [skipped stale run(s) from a different split]")
        for s in skipped:
            print(f"    - {s}")
    if not runs:
        print("no runs found")
        return

    rows = []
    for name, r in runs.items():
        m = r["metrics"]
        row = {"model": name, "auc_macro": m["test_at_0.5"]["auc_macro"], "f1_macro_tuned": m["test_tuned"]["f1_macro"],
               "f1_micro_tuned": m["test_tuned"]["f1_micro"]}
        for lab in THORACIC_LABEL_NAMES:
            row[f"auc_{lab}"] = m["test_at_0.5"]["per_label"][lab].get("auc")
        rows.append(row)
    rows.sort(key=lambda r: -(r["auc_macro"] or 0))

    md = ["| model | AUC (macro) | F1 macro (tuned thr) | " + " | ".join(f"AUC {l}" for l in THORACIC_LABEL_NAMES) + " |",
          "|---|---|---|" + "---|" * len(THORACIC_LABEL_NAMES)]
    for r in rows:
        md.append(f"| {r['model']} | {r['auc_macro']:.3f} | {r['f1_macro_tuned']:.3f} | "
                  + " | ".join(f"{r[f'auc_{l}']:.3f}" if r[f"auc_{l}"] is not None else "-" for l in THORACIC_LABEL_NAMES) + " |")
    (REPORTS_DIR / "openi_comparison.md").write_text("\n".join(md), encoding="utf-8")
    json.dump(rows, open(REPORTS_DIR / "openi_comparison.json", "w"), indent=2)
    print("\n".join(md))

    # ROC curves for the clinically central labels
    for lab in ("chest_infection", "normal", "cardiomegaly", "pleural_effusion"):
        j = THORACIC_LABEL_NAMES.index(lab)
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        for name, r in runs.items():
            fpr, tpr, _ = roc_curve(y[:, j], r["probs"][:, j])
            ax.plot(fpr, tpr, label=f"{name.replace('openi_', '')} (AUC {roc_auc_score(y[:, j], r['probs'][:, j]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC - {lab}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"roc_{lab}.png", dpi=160)
        plt.close(fig)

    # bar chart of macro AUC
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    names = [r["model"].replace("openi_", "") for r in rows]
    ax.barh(names, [r["auc_macro"] for r in rows], color="#4C78A8")
    for i, r in enumerate(rows):
        ax.text(r["auc_macro"] + 0.003, i, f"{r['auc_macro']:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Macro AUC (test, 8 thoracic labels)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "openi_auc_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
