# Why the clinical-only baseline scores 0.581

Diagnostic investigation of the weak clinical branch, run after the four-model
comparison showed fusion failing to beat the image-only baseline.

## Finding

The clinical baseline is not under-trained or mis-loaded — it is **starved of
input**. The symptom vectors parsed from Open-I indication text are far too
sparse to support 8-way thoracic prediction.

| Measure | Value |
|---|---|
| Vocabulary columns | 233 |
| Columns ever active | **53** (182 never fire) |
| Mean symptoms per patient | **1.28** |
| Patients with an empty symptom vector | **554 / 3,105 (17.8%)** |

## Where the missing symptoms went

| Cause | Patients | Fixable? |
|---|---|---|
| Indication blank | 67 | No |
| Indication fully redacted to `XXXX` | 191 | No |
| Real words the parser missed | **296** | **Yes** |

**8.3% of all patients are irreducible** — Open-I de-identification replaced
clinical terms with `XXXX` before publication (`"XXXX-year-old XXXX with XXXX."`).
No parser improvement recovers these.

The remaining 296 are genuine parser gaps, e.g.
`"Elevated blood pressure and thoracic pain"`, `"left arm numbness for 2 weeks"`,
`"History of nodule"`, `"headache."`, `"Routine medical exam"`.

## The deeper problem: the frequent features are uninformative

Most common parsed features across 3,105 patients:

| Feature | Count | Share |
|---|---|---|
| E_14 | 743 | 23.9% |
| **X_FEMALE** | 547 | 17.6% |
| E_66 | 528 | 17.0% |
| **X_MALE** | 375 | 12.1% |
| E_34 | 171 | 5.5% |
| X_TB_SCREEN | 85 | 2.7% |

Two of the top four are **sex markers**, which carry almost no information about
thoracic findings. The clinically meaningful features (TB screening, pain,
trauma) appear in under 4% of patients each. So the branch's most-available
signal is its least diagnostic.

## Consequence for the fusion result

With a near-chance clinical branch (0.581 vs 0.5 chance), cross-modal attention
has little to attend to, which is why:

- fusion (0.813) does not beat image-only (0.826)
- fusion *without* DDXPlus pre-training (0.825) ties image-only

This is consistent with the measured input sparsity, not evidence that
cross-modal fusion is unsound in principle.

## Reconciling this with the §2 lift analysis

The lift table (hypoxia → chest infection, 5.2x) is **not contradicted**: those
lifts are real but computed on the *subsets where the symptom is present* — 33
patients for hypoxia, 30 for heart disease. A feature can be strongly
predictive when present and still be useless to a classifier that sees it in
1% of cases.

## Honest ceiling

Fixing all 296 recoverable cases raises density from ~1.28 to roughly ~1.4
symptoms/patient and empty rows from 17.8% to ~8.3%. That is a real
improvement worth making, but it does not turn 1.4 sparse, sex-dominated
features into a competitive 8-label predictor. **The paired-symptom signal in
Open-I is intrinsically thin** — that is the finding, and it is a legitimate
result to report.
