"""Generate the structural diagrams the written report needs.

The training scripts already emit metric plots (ROC curves, confusion matrices,
learning curves). What a report also needs, and nothing else produces, is:

  1. architecture.png      - the fusion network, drawn as a block diagram
  2. data_pipeline.png     - how three unpaired datasets become one paired corpus
  3. results_summary.png   - the four-model comparison, macro + per-label
  4. sparsity_analysis.png - why the clinical branch is weak (the key finding)

Run:  python scripts/make_report_figures.py
Out:  reports/figures/*.png  (300 dpi, white background, print-safe)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Muted, print-safe palette; distinguishable in greyscale.
C_IMG, C_SYM, C_FUSE = "#4878a8", "#a85878", "#6a8c5a"
C_OUT, C_BG, C_LINE = "#8a6a3a", "#f2f4f6", "#44484e"
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": C_LINE, "axes.labelcolor": "#1a1d21",
    "text.color": "#1a1d21", "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, text, fc, tc="white", fs=8.5, bold=True, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
                                linewidth=1.1, edgecolor=C_LINE, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.5)


def arrow(ax, p1, p2, style="-|>", lw=1.3, color=None, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color or C_LINE, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1,
                                 shrinkA=1.5, shrinkB=1.5))


# ---------------------------------------------------------------- 1. architecture
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")

    ax.text(5, 10.25, "Cross-Modal Fusion Architecture", ha="center", fontsize=13.5, fontweight="bold")
    ax.text(5, 9.92, "DenseNet-121 image encoder + DDXPlus symptom encoder, joined by bidirectional attention",
            ha="center", fontsize=8.4, color="#5a6472", style="italic")

    # inputs
    box(ax, 0.5, 8.75, 3.7, 0.78, "SYMPTOMS\n223 binary clinical questions", C_SYM, fs=8.6)
    box(ax, 5.8, 8.75, 3.7, 0.78, "CHEST RADIOGRAPH\nfrontal view, 224x224", C_IMG, fs=8.6)

    # encoders
    box(ax, 0.5, 7.25, 3.7, 1.0,
        "SymptomEncoder\nembedding 233x256 + MLP\n(pre-trained on 610k DDXPlus)", "#c88ca4", tc="#2a1a20", fs=8.0)
    box(ax, 5.8, 7.25, 3.7, 1.0,
        "DenseNet-121\nImageNet initialised\nfeature map 7x7x1024", "#8caccc", tc="#16222e", fs=8.0)

    arrow(ax, (2.35, 8.75), (2.35, 8.25)); arrow(ax, (7.65, 8.75), (7.65, 8.25))

    # tokens
    box(ax, 0.5, 6.25, 3.7, 0.6, "symptom tokens  (N x 256)", "#e4d2da", tc="#2a1a20", fs=8.0, bold=False)
    box(ax, 5.8, 6.25, 3.7, 0.6, "spatial tokens  (49 x 256)", "#d4e0ea", tc="#16222e", fs=8.0, bold=False)
    arrow(ax, (2.35, 7.25), (2.35, 6.85)); arrow(ax, (7.65, 7.25), (7.65, 6.85))

    # attention
    box(ax, 1.0, 4.85, 8.0, 1.05,
        "BIDIRECTIONAL CROSS-MODAL ATTENTION\n"
        "symptoms attend to lung regions   <->   image attends to symptoms", C_FUSE, fs=8.0)
    arrow(ax, (2.35, 6.25), (3.2, 5.92), rad=-0.12, color=C_SYM, lw=1.5)
    arrow(ax, (7.65, 6.25), (6.8, 5.92), rad=0.12, color=C_IMG, lw=1.5)

    # joint + head
    box(ax, 3.1, 3.85, 3.8, 0.62, "joint representation (512-d)", "#cdd9c6", tc="#1e2a18", fs=8.2, bold=False)
    arrow(ax, (5.0, 4.85), (5.0, 4.47))

    box(ax, 2.35, 2.55, 5.3, 0.95,
        "8 THORACIC FINDINGS  (multi-label, sigmoid)\n"
        "normal · chest infection · cardiomegaly · pleural effusion\n"
        "opacity · atelectasis · emphysema/COPD · pulmonary edema", C_OUT, fs=7.9)
    arrow(ax, (5.0, 3.85), (5.0, 3.5))

    # downstream
    box(ax, 0.5, 1.15, 2.6, 0.75, "Vitals ->\nDiabetes-risk model", "#8a8a96", fs=8.0)
    box(ax, 3.55, 1.15, 2.9, 0.75, "TRIAGE ENGINE\nrule-based escalation", "#6a6a78", fs=8.0)
    box(ax, 6.9, 1.15, 2.6, 0.75, "Grad-CAM\nvisual explanation", "#8a8a96", fs=8.0)

    arrow(ax, (3.1, 1.52), (3.55, 1.52))
    arrow(ax, (4.3, 2.55), (4.6, 1.9))
    arrow(ax, (7.3, 2.55), (8.2, 1.9), rad=0.0, ls="--")

    box(ax, 2.9, 0.22, 4.2, 0.6, "LOW / MODERATE / HIGH / EMERGENCY\n+ referral advisory", "#4a4a56", fs=8.0)
    arrow(ax, (5.0, 1.15), (5.0, 0.82))

    fig.savefig(FIG / "architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  architecture.png")


# ---------------------------------------------------------------- 2. data pipeline
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.4); ax.axis("off")

    ax.text(5, 8.05, "From Three Unpaired Datasets to One Paired Corpus",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(5, 7.72, "The shared 223-code vocabulary is what makes multimodal fusion possible",
            ha="center", fontsize=8.4, color="#5a6472", style="italic")

    # problem row
    ax.text(0.35, 6.95, "THE PROBLEM", fontsize=8.2, fontweight="bold", color="#a8322f")
    for x, (t, n) in zip([0.35, 3.5, 6.65], [("DDXPlus\nsymptoms", "610,000 patients"),
                                             ("Open-I\nchest X-rays", "3,851 reports"),
                                             ("Kaggle\ndiabetes", "100,000 records")]):
        box(ax, x, 5.95, 2.85, 0.82, t, "#b0b4ba", fs=8.4)
        ax.text(x + 1.425, 5.78, n, ha="center", fontsize=7.4, color="#5a6472")
    ax.text(5, 5.42, "no patient appears in more than one dataset  ->  nothing to fuse",
            ha="center", fontsize=8.6, color="#a8322f", fontweight="bold")

    # solution
    ax.text(0.35, 4.85, "THE SOLUTION", fontsize=8.2, fontweight="bold", color="#1d7a4f")
    box(ax, 0.35, 3.95, 4.3, 0.78,
        "Identify the symptom set as DDXPlus\n223 standard questions, 49 diagnoses", C_SYM, fs=8.2)
    box(ax, 5.35, 3.95, 4.3, 0.78,
        "Parse each X-ray's 'indication' text\ninto those same 223 codes", C_IMG, fs=8.2)

    ax.text(5.0, 3.62, '"68-year-old with shortness of breath and weakness. Hypertension. Diabetes."',
            ha="center", fontsize=7.6, color="#3a4048", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef1f4", edgecolor="#c4c8ce", linewidth=0.8))

    arrow(ax, (2.5, 3.95), (4.3, 3.15), rad=-0.08, color=C_SYM, lw=1.4)
    arrow(ax, (7.5, 3.95), (5.7, 3.15), rad=0.08, color=C_IMG, lw=1.4)

    box(ax, 2.6, 2.3, 4.8, 0.78,
        "SHARED 223-CODE VOCABULARY\nevery radiograph now carries its own patient's symptoms", C_FUSE, fs=8.4)

    # outcome
    box(ax, 3.3, 1.18, 3.4, 0.72, "3,459 PAIRED PATIENTS", "#4a4a56", fs=9.4)
    arrow(ax, (5.0, 2.3), (5.0, 1.9))

    for x, t in zip([0.9, 3.72, 6.55], ["train\n2,421", "validation\n519", "test\n519"]):
        box(ax, x, 0.22, 2.55, 0.64, t, "#8a8a96", fs=8.2)
    arrow(ax, (4.3, 1.18), (2.2, 0.86), rad=0.1)
    arrow(ax, (5.0, 1.18), (5.0, 0.86))
    arrow(ax, (5.7, 1.18), (7.8, 0.86), rad=-0.1)
    ax.text(5, 0.02, "split by patient, stratified on chest-infection label",
            ha="center", fontsize=7.4, color="#5a6472", style="italic")

    fig.savefig(FIG / "data_pipeline.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  data_pipeline.png")


# ---------------------------------------------------------------- 3. results
def fig_results() -> None:
    from sklearn.metrics import roc_auc_score
    rep = ROOT / "reports"
    y = np.load(rep / "openi_test_labels.npy")
    from medsupport.config import THORACIC_LABEL_NAMES as LBL
    runs = {
        "Fusion\n(no pre-train)": ("fusion_nopretrain", C_FUSE),
        "Image only": ("image", C_IMG),
        "Fusion\n(pre-trained)": ("fusion", "#9a7a5a"),
        "Clinical only": ("clinical", C_SYM),
    }
    P, macro = {}, {}
    for lab, (tag, _) in runs.items():
        P[lab] = np.load(rep / f"openi_{tag}_test_probs.npy")
        macro[lab] = roc_auc_score(y, P[lab], average="macro")

    fig = plt.figure(figsize=(10.0, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.42)
    fig.suptitle("Open-I Results: Four Models, One Patient Split (test n = 519)",
                 fontsize=13.5, fontweight="bold", y=0.975)

    # panel A - macro AUC
    ax = fig.add_subplot(gs[0])
    labs = list(runs)
    vals = [macro[l] for l in labs]
    cols = [runs[l][1] for l in labs]
    bars = ax.barh(range(len(labs)), vals, color=cols, edgecolor=C_LINE, linewidth=0.9, height=0.62)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.92); ax.set_xlabel("Macro-averaged AUC", fontsize=9)
    ax.axvline(0.5, color="#9a9ea4", linestyle=":", linewidth=1.1)
    ax.text(0.504, -0.55, "chance", fontsize=7.0, color="#6a6e74", va="center")
    for b, v in zip(bars, vals):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=8.8, fontweight="bold")
    base = macro["Image only"]
    best_key = labs[0]
    gain = macro[best_key] - base
    ax.annotate(f"+{gain:.3f} over image baseline",
                xy=(macro[best_key], 0.30), xytext=(0.545, 2.55),
                fontsize=8.2, color="#1d7a4f", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1d7a4f", lw=1.2))
    ax.set_title("A.  Overall performance", fontsize=10, fontweight="bold", loc="left", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    # panel B - per label
    ax2 = fig.add_subplot(gs[1])
    n = len(LBL); w = 0.2
    xs = np.arange(n)
    for i, lab in enumerate(labs):
        a = [roc_auc_score(y[:, j], P[lab][:, j]) for j in range(n)]
        ax2.bar(xs + (i - 1.5) * w, a, w, label=lab.replace("\n", " "),
                color=runs[lab][1], edgecolor=C_LINE, linewidth=0.6)
    ax2.axhline(0.5, color="#9a9ea4", linestyle=":", linewidth=1.1)
    sup = y.sum(0).astype(int)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{l.replace('_', chr(10))}\n(n={s})" for l, s in zip(LBL, sup)], fontsize=7.4)
    ax2.set_ylim(0.45, 1.0); ax2.set_ylabel("AUC", fontsize=9)
    ax2.legend(fontsize=7.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax2.set_title("B.  Per-finding performance  (n = positive test cases)",
                  fontsize=10, fontweight="bold", loc="left", pad=36)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2.set_axisbelow(True)

    fig.savefig(FIG / "results_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  results_summary.png")


# ---------------------------------------------------------------- 4. sparsity
def fig_sparsity() -> None:
    import pandas as pd
    from medsupport.config import DATA_PROCESSED
    vocab = json.load(open(DATA_PROCESSED / "openi_meta.json"))["vocab"]
    df = pd.concat([pd.read_parquet(DATA_PROCESSED / f"openi_{p}.parquet")
                    for p in ("train", "val", "test")])
    S = df[vocab].to_numpy().astype(float)
    per = S.sum(1)
    counts = S.sum(0)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    fig.suptitle("Why the Clinical Branch Is Weak: Measured Input Sparsity",
                 fontsize=12.5, fontweight="bold", y=1.03)

    # (a) symptoms per patient
    ax = axes[0]
    vc = np.bincount(per.astype(int))
    ax.bar(range(len(vc)), vc, color=C_SYM, edgecolor=C_LINE, linewidth=0.7)
    ax.set_xlabel("symptoms parsed per patient"); ax.set_ylabel("patients")
    ax.set_title(f"(a) mean = {per.mean():.2f} symptoms\n{(per == 0).mean() * 100:.1f}% have none",
                 fontsize=9, loc="left")
    ax.set_xlim(-0.6, 8)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) vocabulary usage
    ax = axes[1]
    active = int((counts > 0).sum())
    ax.bar(["used", "never used"], [active, len(vocab) - active],
           color=[C_FUSE, "#c4c8ce"], edgecolor=C_LINE, linewidth=0.8, width=0.6)
    for i, v in enumerate([active, len(vocab) - active]):
        ax.text(i, v + 4, str(v), ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("vocabulary codes")
    ax.set_title(f"(b) only {active} of {len(vocab)} codes\never fire", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    # (c) top features
    ax = axes[2]
    top = sorted(zip(counts, vocab), reverse=True)[:7][::-1]
    names = [t[1] for t in top]
    vals = [t[0] / len(df) * 100 for t in top]
    cols = ["#a8322f" if n in ("X_FEMALE", "X_MALE") else C_IMG for n in names]
    ax.barh(range(len(top)), vals, color=cols, edgecolor=C_LINE, linewidth=0.7, height=0.68)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=7.6, family="monospace")
    ax.set_xlabel("prevalence (% of patients)")
    ax.set_title("(c) two of the top features are\nsex markers (red) - not diagnostic",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG / "sparsity_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sparsity_analysis.png")


# ---------------------------------------------------------------- 5. error analysis
def _tuned_thresholds(y, p):
    """Per-label threshold maximising F1, as the trainer reports f1@tuned."""
    ths = []
    for j in range(y.shape[1]):
        grid = np.unique(np.round(p[:, j], 3))
        best, bt = -1.0, 0.5
        for t in grid:
            pred = (p[:, j] >= t).astype(int)
            tp = int((pred & y[:, j].astype(int)).sum())
            if tp == 0:
                continue
            prec = tp / max(pred.sum(), 1)
            rec = tp / max(y[:, j].sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best:
                best, bt = f1, float(t)
        ths.append(bt)
    return np.array(ths)


def error_table(tag="fusion_nopretrain"):
    """Confusion counts + derived rates per finding for the best model."""
    from medsupport.config import THORACIC_LABEL_NAMES as LBL
    rep = ROOT / "reports"
    y = np.load(rep / "openi_test_labels.npy").astype(int)
    p = np.load(rep / f"openi_{tag}_test_probs.npy")
    th = _tuned_thresholds(y, p)
    rows = []
    for j, lab in enumerate(LBL):
        pred = (p[:, j] >= th[j]).astype(int)
        tp = int((pred & y[:, j]).sum())
        fp = int((pred & (1 - y[:, j])).sum())
        fn = int(((1 - pred) & y[:, j]).sum())
        tn = int(((1 - pred) & (1 - y[:, j])).sum())
        rows.append({
            "label": lab, "support": int(y[:, j].sum()), "threshold": round(th[j], 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity_recall": round(tp / max(tp + fn, 1), 3),
            "specificity": round(tn / max(tn + fp, 1), 3),
            "precision_ppv": round(tp / max(tp + fp, 1), 3),
            "npv": round(tn / max(tn + fn, 1), 3),
            "false_negative_rate": round(fn / max(tp + fn, 1), 3),
            "false_positive_rate": round(fp / max(fp + tn, 1), 3),
        })
    return rows


def fig_errors() -> None:
    from medsupport.config import THORACIC_LABEL_NAMES as LBL
    rows = error_table()
    json.dump(rows, open(ROOT / "reports" / "error_analysis.json", "w"), indent=2)

    fig = plt.figure(figsize=(11.4, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.46, wspace=0.26)
    fig.suptitle("Error Analysis: False Positives vs False Negatives  (best model, test n = 519)",
                 fontsize=12.8, fontweight="bold", y=0.975)

    names = [r["label"].replace("_", "\n") for r in rows]
    xs = np.arange(len(rows))

    # (a) FP / FN counts
    ax = fig.add_subplot(gs[0, :])
    fp = [r["FP"] for r in rows]; fn = [r["FN"] for r in rows]; tp = [r["TP"] for r in rows]
    ax.bar(xs - 0.26, tp, 0.26, label="True positives (caught)", color=C_FUSE,
           edgecolor=C_LINE, linewidth=0.7)
    ax.bar(xs, fn, 0.26, label="False negatives (MISSED - clinically worse)", color="#a8322f",
           edgecolor=C_LINE, linewidth=0.7)
    ax.bar(xs + 0.26, fp, 0.26, label="False positives (false alarm)", color="#d8a84a",
           edgecolor=C_LINE, linewidth=0.7)
    for i, (a, b, c) in enumerate(zip(tp, fn, fp)):
        for off, v in ((-0.26, a), (0.0, b), (0.26, c)):
            if v:
                ax.text(i + off, v + 1.2, str(v), ha="center", fontsize=7.2)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.4)
    ax.set_ylabel("test cases")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title("(a)  Raw error counts at the F1-tuned threshold", fontsize=9.6,
                 fontweight="bold", loc="left", pad=34)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)

    # (b) sensitivity vs specificity
    ax = fig.add_subplot(gs[1, 0])
    sens = [r["sensitivity_recall"] for r in rows]
    spec = [r["specificity"] for r in rows]
    ax.bar(xs - 0.19, sens, 0.38, label="Sensitivity (recall)", color="#4878a8",
           edgecolor=C_LINE, linewidth=0.7)
    ax.bar(xs + 0.19, spec, 0.38, label="Specificity", color="#a0b8cc",
           edgecolor=C_LINE, linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace(chr(10), " ") for n in names], fontsize=6.4,
                       rotation=38, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("rate")
    ax.axhline(0.5, color="#9a9ea4", linestyle=":", linewidth=1)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    ax.set_title("(b)  Sensitivity vs specificity", fontsize=9.6, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    # (c) FN rate - the clinical risk view
    ax = fig.add_subplot(gs[1, 1])
    fnr = [r["false_negative_rate"] for r in rows]
    order = np.argsort(fnr)
    cols = ["#a8322f" if fnr[i] > 0.5 else "#d8a84a" if fnr[i] > 0.3 else C_FUSE for i in order]
    ax.barh(range(len(rows)), [fnr[i] for i in order], color=cols,
            edgecolor=C_LINE, linewidth=0.7, height=0.66)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([rows[i]["label"] for i in order], fontsize=7.2)
    ax.set_xlim(0, 1.0); ax.set_xlabel("false-negative rate (missed cases / all positives)")
    for k, i in enumerate(order):
        ax.text(fnr[i] + 0.015, k, f"{fnr[i]:.2f}  ({rows[i]['FN']}/{rows[i]['support']})",
                va="center", fontsize=7.0)
    ax.set_title("(c)  Miss rate per finding (lower is safer)", fontsize=9.6,
                 fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG / "error_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  error_analysis.png  + reports/error_analysis.json")


def fig_confusion_grid() -> None:
    """2x2 confusion matrix per finding - what a report reviewer expects to see."""
    from medsupport.config import THORACIC_LABEL_NAMES as LBL
    rows = error_table()
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.6))
    fig.suptitle("Per-Finding Confusion Matrices  (best model: fusion without pre-training)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    for ax, r in zip(axes.ravel(), rows):
        m = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1))
        for (i, j), v in np.ndenumerate(m):
            shade = "white" if v > m.max() * 0.55 else "#1a1d21"
            tag = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i - 0.13, str(v), ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=shade)
            ax.text(j, i + 0.24, tag, ha="center", va="center", fontsize=7.4, color=shade)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7.2)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=7.2)
        ax.set_title(f"{r['label']}\nrecall {r['sensitivity_recall']:.2f} · "
                     f"PPV {r['precision_ppv']:.2f}", fontsize=8.2)
        for s in ax.spines.values():
            s.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_grid.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  confusion_grid.png")


if __name__ == "__main__":
    print("generating report figures ->", FIG)
    fig_architecture()
    fig_pipeline()
    fig_results()
    fig_sparsity()
    fig_errors()
    fig_confusion_grid()
    print("done")
