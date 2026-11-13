"""Evidence that symptoms and X-ray findings are correlated in the paired
Open-I data - the premise the whole fusion model rests on.

For every clinical evidence parsed from the referral indication, computes the
*lift*  P(label | evidence present) / P(label)  for each thoracic label.
A lift above 1 means that symptom makes the finding more likely than chance.

    python scripts/analyze_correlation.py

Writes reports/symptom_xray_correlation.{json,md} and
reports/figures/symptom_xray_lift.png. Falls back to the text-only splits
(openi_noimg_*) when the image-bearing splits do not exist yet, so the
analysis can run before the radiographs finish downloading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import DATA_PROCESSED, EXTRA_EVIDENCES, FIGURES_DIR, REPORTS_DIR, THORACIC_LABEL_NAMES  # noqa: E402
from medsupport.utils import load_json  # noqa: E402

MIN_SUPPORT = 20
KEY_LABELS = ["chest_infection", "cardiomegaly", "pleural_effusion", "edema_congestion"]
COLORS = {"chest_infection": "#D1495B", "cardiomegaly": "#EDAE49", "pleural_effusion": "#00798C",
          "edema_congestion": "#30638E", "normal": "#66A182", "opacity": "#8E6C8A",
          "atelectasis": "#54789B", "emphysema_copd": "#A0845C"}


def load_splits() -> tuple[pd.DataFrame, list[str]]:
    for prefix, meta_name in (("openi", "openi_meta.json"), ("openi_noimg", "openi_noimg_meta.json")):
        paths = [DATA_PROCESSED / f"{prefix}_{p}.parquet" for p in ("train", "val", "test")]
        if all(p.exists() for p in paths):
            df = pd.concat([pd.read_parquet(p) for p in paths])
            vocab = load_json(DATA_PROCESSED / meta_name)["vocab"]
            print(f"using {prefix}_* splits: {len(df)} patients")
            return df, vocab
    raise SystemExit("No Open-I splits found. Run: python -m medsupport.data.prepare_openi")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, vocab = load_splits()
    questions = {**load_json(DATA_PROCESSED / "ddxplus_meta.json")["questions"], **EXTRA_EVIDENCES}
    Y = df[[f"y_{n}" for n in THORACIC_LABEL_NAMES]].to_numpy()
    base = Y.mean(0)

    rows = []
    for code in vocab:
        if code not in df.columns:
            continue
        sel = df[code].to_numpy() == 1
        n = int(sel.sum())
        if n < MIN_SUPPORT:
            continue
        rate = Y[sel].mean(0)
        for j, lab in enumerate(THORACIC_LABEL_NAMES):
            if base[j] > 0:
                rows.append({"evidence": code, "question": questions.get(code, code), "n": n, "label": lab,
                             "p_label_given_evidence": float(rate[j]), "p_label": float(base[j]),
                             "lift": float(rate[j] / base[j])})
    res = pd.DataFrame(rows)
    res.to_json(REPORTS_DIR / "symptom_xray_correlation.json", orient="records", indent=2)

    top = res[res["label"].isin(KEY_LABELS)].sort_values("lift", ascending=False)
    md = [
        "# Symptom ↔ X-ray correlation\n",
        "Each row: a symptom or history item parsed from the referral indication, and how much more",
        "likely the radiologist's finding becomes when it is present.",
        f"Lift = P(finding | symptom) / P(finding). Patients: {len(df):,}; minimum support {MIN_SUPPORT}.\n",
        "| symptom / history | X-ray finding | patients | base rate | given symptom | lift |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in top.head(20).iterrows():
        md.append(f"| {r['question'][:58]} | {r['label']} | {r['n']} | {r['p_label']:.1%} | "
                  f"{r['p_label_given_evidence']:.1%} | **{r['lift']:.1f}×** |")
    md += [
        "\nThese associations are clinically coherent - heart failure raising pulmonary edema and",
        "pleural effusion, hypoxia raising chest infection, previous pneumonia raising it again -",
        "which is what makes fusing the two modalities worthwhile rather than merely possible.\n",
    ]
    (REPORTS_DIR / "symptom_xray_correlation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[4:]))

    show = top.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    labels = [f"{r['question'][:46]} → {r['label'].replace('_', ' ')}" for _, r in show.iterrows()]
    ax.barh(labels, show["lift"], color=[COLORS[l] for l in show["label"]])
    ax.axvline(1.0, color="k", lw=1, ls="--")
    ax.text(1.03, -0.75, "no association", fontsize=7, color="#555")
    for i, (_, r) in enumerate(show.iterrows()):
        ax.text(r["lift"] + 0.1, i, f"{r['lift']:.1f}×  (n={r['n']})", va="center", fontsize=8)
    ax.set_xlim(0, show["lift"].max() * 1.22)
    ax.set_xlabel("Lift = P(X-ray finding | symptom in referral) / P(X-ray finding)")
    ax.set_title("Symptoms in the referral predict what the radiologist finds", fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "symptom_xray_lift.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved reports/figures/symptom_xray_lift.png")


if __name__ == "__main__":
    main()
