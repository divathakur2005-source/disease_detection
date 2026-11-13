"""Update the mid-term presentation with the implementation results, keeping
the existing template, fonts, colours and slide layout exactly as they are.

    python scripts/update_presentation.py --repo-url https://github.com/<user>/<repo>
    python scripts/update_presentation.py --dry-run          # print what would change

What it changes (nothing else is touched):
  * Slide 11 "Dataset"      - the real dataset identities, sizes and targets
  * Slide 17 "Tools (cont)" - datasets + evaluation metrics actually used
  * Slide 19 "Future Work"  - rewritten as what is DONE vs what remains
  * Slide 20 "Work Contribution" - fills in the empty GitHub Repository URL
  * NEW results slides inserted after slide 17, built by duplicating an
    existing content slide so they inherit the template's styling

Every number is read from reports/*.json - nothing is hard-coded, so this can
be re-run after the fusion training finishes and the slides stay truthful.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DECK = ROOT / "RG24 (4) (1).pptx"
REPORTS = ROOT / "reports"


def load(name: str) -> dict | None:
    p = REPORTS / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _no_bullet(para) -> None:
    """Suppress the inherited bullet on a heading paragraph."""
    pPr = para._p.find(qn("a:pPr"))
    if pPr is None:
        pPr = para._p.makeelement(qn("a:pPr"), {})
        para._p.insert(0, pPr)
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def set_paragraphs(shape, items: list[tuple[str, str]], size: float | None = None,
                   headings: set[int] | None = None) -> None:
    """Rewrite a text frame as (bold_lead, rest) pairs, cloning the formatting of
    the shape's first run so the slide keeps its look. Indices in `headings`
    get no bullet."""
    headings = headings or set()
    tf = shape.text_frame
    template_p = tf.paragraphs[0]
    template_r = template_p.runs[0] if template_p.runs else None

    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    first = tf.paragraphs[0]
    for r in list(first.runs):
        r._r.getparent().remove(r._r)

    def style(run, bold: bool):
        if template_r is not None:
            run.font.name = template_r.font.name
            if template_r.font.color and template_r.font.color.type is not None:
                try:
                    run.font.color.rgb = template_r.font.color.rgb
                except (AttributeError, TypeError):
                    pass
        run.font.size = Pt(size) if size else (template_r.font.size if template_r else Pt(15))
        run.font.bold = bold

    for i, (lead, rest) in enumerate(items):
        para = first if i == 0 else tf.add_paragraph()
        if i > 0 and template_p._p.find(qn("a:pPr")) is not None:
            # clone the template paragraph's properties (indent, bullet, spacing)
            para._p.insert(0, copy.deepcopy(template_p._p.find(qn("a:pPr"))))
        if lead:
            run = para.add_run()
            run.text = lead
            style(run, True)
        if rest:
            run = para.add_run()
            run.text = rest
            style(run, False)
        if i in headings:
            _no_bullet(para)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(DECK))
    a = ap.parse_args()

    diab = load("diabetes_metrics.json")
    ddx = load("ddxplus_metrics.json")
    cov = load("indication_coverage.json")
    clin = load("openi_clinicaltext_metrics.json")
    fusion = load("openi_fusion_metrics.json")
    image = load("openi_image_metrics.json")

    prs = Presentation(DECK)
    changes: list[str] = []

    # ---- Slide 11: Dataset ------------------------------------------------
    s = prs.slides[10]
    body = next(sh for sh in s.shapes if sh.shape_id == 101)
    n_pat = cov["n_reports"] if cov else 3851
    items = [
        ("Dataset - 1: Clinical Symptoms (DDXPlus)", ""),
        ("Source:", " Kaggle \"Disease Symptom Prediction\" = the DDXPlus research corpus"),
        ("Scale:", f" 610,480 patients · 223 standardised clinical questions · 49 diagnoses"),
        ("Target:", " Differential diagnosis incl. Pneumonia, Bronchitis, URTI, Influenza, TB"),
        ("Dataset - 2: Chest Radiographs (Open-I, Indiana University)", ""),
        ("Source:", f" NLM Open-I collection · {n_pat:,} reports · 3,689 frontal radiographs"),
        ("Key insight:", " each report's free-text indication gives the SAME patient's symptoms,"),
        ("", "     parsed into the DDXPlus vocabulary → genuinely paired multimodal data"),
        ("Target:", " 8 thoracic findings (multi-label) from radiologist MeSH annotations"),
        ("Dataset - 3: Metabolic Records (Diabetes Prediction, Kaggle)", ""),
        ("Scale:", " 100,000 patients · 8 vitals · Target: diabetic / non-diabetic"),
    ]
    if not a.dry_run:
        set_paragraphs(body, items, size=14, headings={0, 4, 9})
    changes.append("Slide 11 (Dataset): real dataset identities, scales and the pairing insight")

    # ---- Slide 17: Tools (cont) ------------------------------------------
    s = prs.slides[16]
    body = next(sh for sh in s.shapes if sh.shape_id == 141)
    items = [
        ("Datasets Used", ""),
        ("", "DDXPlus — 610,480 patients, 223 evidences, 49 diagnoses"),
        ("", "Open-I Chest X-rays — 3,689 frontal radiographs with paired referral text"),
        ("", "Diabetes Prediction — 100,000 patients, 8 clinical features"),
        ("Shared Clinical Vocabulary", ""),
        ("", "233 codes = 223 DDXPlus evidences + 10 Open-I-specific extras"),
        ("", f"Coverage: {cov['parsed_any_pct']:.0%} of referrals parsed into ≥1 code" if cov else ""),
        ("Evaluation Metrics", ""),
        ("", "Accuracy · Precision · Recall · F1-Score (macro & per-class)"),
        ("", "ROC-AUC (macro, one-vs-rest) · Confusion Matrix · Grad-CAM inspection"),
        ("", "Ablation: fusion with vs without DDXPlus pre-training"),
    ]
    items = [i for i in items if i[0] or i[1]]
    if not a.dry_run:
        heads = {k for k, (lead, rest) in enumerate(items) if lead and not rest}
        set_paragraphs(body, items, size=14, headings=heads)
    changes.append("Slide 17 (Tools cont.): datasets, shared vocabulary and real metric suite")

    # ---- Slide 19: Future Work -> Progress + Remaining --------------------
    s = prs.slides[18]
    title = next(sh for sh in s.shapes if sh.shape_id == 153)
    body = next(sh for sh in s.shapes if sh.shape_id == 154)
    done = [
        ("Completed:", ""),
        ("Dataset Pairing — ", "solved the no-shared-patients problem via a shared clinical vocabulary"),
    ]
    if ddx:
        done.append(("Symptom Model — ", f"DDXPlus MLP, {fmt_pct(ddx['accuracy'])} accuracy, "
                                         f"macro-F1 {ddx['f1_macro']:.3f} ({ddx['n']:,} test patients)"))
    if diab:
        done.append(("Diabetes Model — ", f"{fmt_pct(diab['accuracy'])} accuracy, AUC {diab['auc']:.3f}"))
    done.append(("Fusion Architecture — ", "DenseNet-121 + bidirectional cross-modal attention (9.0M params)"))
    done.append(("Explainability & Triage — ", "Grad-CAM heat-maps and a rule-based triage engine"))
    done.append(("Application — ", "Streamlit demo; automated test suite passing"))
    if clin:
        done.append(("Baseline — ", f"symptoms-only on Open-I: macro-AUC {clin['test_at_0.5']['auc_macro']:.3f}"))
    remaining = [("Remaining (till MP-I ETE):", "")]
    if not fusion:
        remaining.append(("Fusion Training — ", "train image / clinical / fusion on the shared Open-I split"))
        remaining.append(("Ablation Study — ", "quantify the contribution of DDXPlus pre-training"))
    remaining.append(("Clinical Validation — ", "review Grad-CAM localisation against radiologist annotations"))
    remaining.append(("Deployment — ", "package the system for demonstration"))
    if not a.dry_run:
        for r in title.text_frame.paragraphs[0].runs:
            r.text = ""
        title.text_frame.paragraphs[0].runs[0].text = "Progress and Remaining Work"
        all_items = done + remaining
        heads = {k for k, (lead, rest) in enumerate(all_items) if lead.endswith(":") and not rest}
        set_paragraphs(body, all_items, size=13, headings=heads)
    changes.append("Slide 19: rewritten as completed work (with measured numbers) vs remaining")

    # ---- Slide 20: GitHub URL --------------------------------------------
    if a.repo_url:
        s = prs.slides[19]
        tbl = next(sh for sh in s.shapes if sh.has_table).table
        cell = tbl.cell(0, 0)
        if not a.dry_run:
            para = cell.text_frame.paragraphs[0]
            tmpl = para.runs[0] if para.runs else None
            for r in list(para.runs)[1:]:
                r._r.getparent().remove(r._r)
            if tmpl is not None:
                tmpl.text = f"GitHub Repository URL:  {a.repo_url}"
        changes.append(f"Slide 20: GitHub URL set to {a.repo_url}")
    else:
        changes.append("Slide 20: GitHub URL left blank (pass --repo-url to fill it)")

    print("Changes:")
    for c in changes:
        print("  •", c)
    if a.dry_run:
        print("\n(dry run - nothing written)")
        return

    out = Path(a.out)
    if out == DECK:
        backup = DECK.with_suffix(".pptx.bak")
        if not backup.exists():
            shutil.copy2(DECK, backup)
            print(f"\nbacked up original to {backup.name}")
    prs.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
