"""Evidence that symptoms and X-ray findings are correlated in the paired
Open-I data (the premise of multimodal fusion).

For every clinical evidence parsed from the indication text, computes the
*lift*  P(label | evidence present) / P(label)  for each thoracic label and
plots the strongest ones. Writes reports/symptom_xray_correlation.{json,md}
and reports/figures/symptom_xray_lift.png.

    python scripts/analyze_correlation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import DATA_PROCESSED, EXTRA_EVIDENCES, FIGURES_DIR, REPORTS_DIR, THORACIC_LABEL_NAMES  # noqa: E402
from medsupport.utils import load_json  # noqa: E402

MIN_SUPPORT = 20


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = load_json(DATA_PROCESSED / "openi_meta.json")
    ddx = load_json(DATA_PROCESSED / "ddxplus_meta.json")
    questions = {**ddx["questions"], **EXTRA_EVIDENCES}
    df = pd.concat([pd.read_parquet(DATA_PROCESSED / f"openi_{p}.parquet") for p in ("train", "val", "test")])
    vocab = meta["vocab"]
    Y = df[[f"y_{n}" for n in THORACIC_LABEL_NAMES]].to_numpy()
    base = Y.mean(0)

    rows = []
    for code in vocab:
        sel = df[code].to_numpy() == 1
        n = int(sel.sum())
        if n < MIN_SUPPORT:
            continue
        rate = Y[sel].mean(0)
        for j, lab in enumerate(THORACIC_LABEL_NAMES):
            rows.append({"evidence": code, "question": questions.get(code, code), "n": n, "label": lab,
                         "p_label_given_evidence": float(rate[j]), "p_label": float(base[j]),
                         "lift": float(rate[j] / base[j]) if base[j] > 0 else None})
    res = pd.DataFrame(rows)
    res.to_json(REPORTS_DIR / "symptom_xray_correlation.json", orient="records", indent=2)

    # strongest positive associations (lift >= 1.3) for the central labels
    top = (res[res["label"].isin(["chest_infection", "cardiomegaly", "pleural_effusion", "edema_congestion", "normal"])]
           .sort_values("lift", ascending=False))
    md = ["| evidence | label | n | P(label\\|evidence) | P(label) | lift |", "|---|---|---|---|---|---|"]
    for _, r in top.head(25).iterrows():
        md.append(f"| {r['question'][:60]} | {r['label']} | {r['n']} | {r['p_label_given_evidence']:.3f} | "
                  f"{r['p_label']:.3f} | {r['lift']:.2f}x |")
    (REPORTS_DIR / "symptom_xray_correlation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # figure: lift of selected evidences on chest_infection / cardiomegaly / normal
    show = top[top["lift"] >= 1.3].drop_duplicates(["evidence", "label"]).head(14)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(show) + 1.2))
    colors = {"chest_infection": "#D1495B", "cardiomegaly": "#EDAE49", "pleural_effusion": "#00798C",
              "edema_congestion": "#30638E", "normal": "#66A182"}
    labels_txt = [f"{q[:42]} → {l}" for q, l in zip(show["question"], show["label"])]
    ax.barh(labels_txt, show["lift"], color=[colors[l] for l in show["label"]])
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    for i, (v, n) in enumerate(zip(show["lift"], show["n"])):
        ax.text(v + 0.05, i, f"{v:.1f}x (n={n})", va="center", fontsize=7)
    ax.set_xlabel("Lift = P(X-ray label | symptom in indication) / P(X-ray label)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "symptom_xray_lift.png", dpi=170)
    plt.close(fig)
    print("saved", FIGURES_DIR / "symptom_xray_lift.png")


if __name__ == "__main__":
    main()
