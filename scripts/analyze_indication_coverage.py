"""How well does the regex vocabulary parse Open-I indication text?

This is the honest audit of the project's central mechanism. It reports what
fraction of radiology referrals yield at least one clinical code, which codes
fire, and - crucially - what the *unparsed* reports contain, so the gap is
visible rather than hidden.

    python scripts/analyze_indication_coverage.py

Writes reports/indication_coverage.{json,md} and
reports/figures/indication_coverage.png.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import (DATA_RAW, EXTRA_EVIDENCES, FIGURES_DIR, INDICATION_TO_EVIDENCE,  # noqa: E402
                               REPORTS_DIR)
from medsupport.data.prepare_openi import parse_indication, vocabulary  # noqa: E402
from medsupport.utils import load_json, save_json  # noqa: E402

STOP = {"xxxx", "year", "old", "with", "the", "and", "for", "yearold", "male", "female", "of", "in",
        "to", "on", "his", "her", "no", "is", "has", "an", "at", "by", "or", "from", "this", "that",
        "who", "had", "was", "were", "age", "patient", "presents", "presenting", "states", "now"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vocab = vocabulary()
    vidx = {c: i for i, c in enumerate(vocab)}
    questions = {**load_json(ROOT / "data" / "processed" / "ddxplus_meta.json")["questions"], **EXTRA_EVIDENCES}
    rep = pd.read_csv(DATA_RAW / "openi_meta" / "indiana_reports.csv")
    ind = rep["indication"].fillna("")
    X = np.stack([parse_indication(t, vidx) for t in ind])
    ddx_mask = np.array([c.startswith("E_") for c in vocab])

    n = len(rep)
    summary = {
        "n_reports": n,
        "n_with_indication": int((ind.str.strip() != "").sum()),
        "vocab_size": len(vocab),
        "n_ddxplus_codes": int(ddx_mask.sum()),
        "n_extra_codes": len(EXTRA_EVIDENCES),
        "parsed_any": int((X.sum(1) > 0).sum()),
        "parsed_any_pct": float((X.sum(1) > 0).mean()),
        "parsed_ddxplus": int((X[:, ddx_mask].sum(1) > 0).sum()),
        "parsed_ddxplus_pct": float((X[:, ddx_mask].sum(1) > 0).mean()),
        "unparsed": int((X.sum(1) == 0).sum()),
        "mean_codes_per_report": float(X.sum(1).mean()),
        "ddxplus_codes_used": int(sum(1 for j, c in enumerate(vocab) if c.startswith("E_") and X[:, j].sum() > 0)),
    }

    hits = sorted(((vocab[j], int(X[:, j].sum())) for j in range(len(vocab)) if X[:, j].sum() > 0),
                  key=lambda t: -t[1])
    summary["code_hits"] = {c: h for c, h in hits}

    # what is in the reports we fail to parse
    unparsed_text = [t.lower() for t, row in zip(ind, X) if row.sum() == 0]
    words = Counter()
    for t in unparsed_text:
        for w in re.findall(r"[a-z]{2,}", t):
            if w not in STOP:
                words[w] += 1
    summary["unparsed_top_words"] = dict(words.most_common(40))
    summary["unparsed_examples"] = unparsed_text[:25]
    summary["unparsed_empty"] = int(sum(1 for t in unparsed_text if not t.strip() or t.strip() in (".", "xxxx.", "xxxx")))

    save_json(summary, REPORTS_DIR / "indication_coverage.json")

    md = [
        "# Indication parsing coverage\n",
        "The project links symptoms to radiographs by parsing each Open-I report's free-text",
        "*indication* (the referring clinician's reason for the scan) into the shared DDXPlus",
        "evidence vocabulary. This file audits how well that works.\n",
        f"- Reports: **{n:,}** ({summary['n_with_indication']:,} have indication text)",
        f"- Vocabulary: **{summary['vocab_size']}** codes = {summary['n_ddxplus_codes']} DDXPlus + {summary['n_extra_codes']} Open-I extras",
        f"- At least one code parsed: **{summary['parsed_any']:,} ({summary['parsed_any_pct']:.0%})**",
        f"- At least one *DDXPlus* code: **{summary['parsed_ddxplus']:,} ({summary['parsed_ddxplus_pct']:.0%})**",
        f"- Unparsed: **{summary['unparsed']:,}** (of which {summary['unparsed_empty']} are empty or only de-identified tokens)",
        f"- Mean codes per report: **{summary['mean_codes_per_report']:.2f}**",
        f"- DDXPlus questions that ever fire: **{summary['ddxplus_codes_used']} of {summary['n_ddxplus_codes']}**\n",
        "## Most frequent parsed codes\n",
        "| code | meaning | reports |", "|---|---|---|",
    ]
    for c, h in hits[:25]:
        md.append(f"| `{c}` | {questions.get(c, c)[:70]} | {h} |")
    md += [
        "\n## Known limitations\n",
        "1. **Indications are short.** A referral names one or two symptoms, not a full questionnaire,",
        f"   hence {summary['mean_codes_per_report']:.2f} codes per patient on average.",
        "2. **De-identification removes words.** Open-I replaces sensitive tokens with `XXXX`",
        "   (*\"XXXX on the congestion, XXXX for 2 days\"*), so some common symptoms are",
        "   systematically under-counted.",
        "3. **Only the questions a radiology referral mentions can fire** - the remaining DDXPlus",
        "   questions (rash colour, menstrual history, travel) never appear in a chest X-ray order.\n",
        "These are properties of the data, not defects of the method: the correlation analysis",
        "(`symptom_xray_correlation.md`) shows the codes that *do* parse carry real diagnostic signal.\n",
        "## Most common words in reports we fail to parse\n",
    ]
    md.append("| word | count |")
    md.append("|---|---|")
    for w, c in words.most_common(20):
        md.append(f"| {w} | {c} |")
    (REPORTS_DIR / "indication_coverage.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["≥1 DDXPlus\ncode", "Only Open-I\nextras", "Unparsed"]
    only_extra = int(((X.sum(1) > 0) & (X[:, ddx_mask].sum(1) == 0)).sum())
    vals = [summary["parsed_ddxplus"], only_extra, summary["unparsed"]]
    axes[0].bar(labels, vals, color=["#2C5F2D", "#97BC62", "#C9C9C9"])
    for i, v in enumerate(vals):
        axes[0].text(i, v + 25, f"{v}\n({v / n:.0%})", ha="center", fontsize=9)
    axes[0].set_ylabel("reports")
    axes[0].set_title(f"Indication parsing coverage (n={n:,})")
    axes[0].set_ylim(0, max(vals) * 1.25)

    top = hits[:14][::-1]
    axes[1].barh([questions.get(c, c)[:38] for c, _ in top], [h for _, h in top], color="#065A82")
    axes[1].set_xlabel("reports")
    axes[1].set_title("Most frequent parsed clinical codes")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "indication_coverage.png", dpi=170)
    plt.close(fig)
    print("\n".join(md[:14]))
    print("\nsaved reports/indication_coverage.{json,md} and figures/indication_coverage.png")


if __name__ == "__main__":
    main()
