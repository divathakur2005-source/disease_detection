# Project Status — AI-Based Medical Decision-Support System

**Group RG24 · Major Project-I (18B19CI791) · JUIT CSE & IT**
Supervisor: Dr. Hari Singh · Status as of **13 November 2026**

---

## 1. The problem we had to solve first

The three datasets in our proposal (symptom questionnaire, chest X-rays,
diabetes records) come from **three different sets of patients**. There was no
patient who appears in more than one of them. That is why the symptoms were not
correlating with the X-ray findings — there was literally nothing linking a
patient's reported symptoms to their radiograph, so "multimodal fusion" had
nothing to fuse.

## 2. How we fixed it — one shared clinical vocabulary

Two changes made the project work:

**(a) We identified what the symptom dataset actually is.** The Kaggle
"disease-symptom-prediction-dataset" is the **DDXPlus** research dataset:
610,000 patients, **223 standardised yes/no clinical questions** (symptoms +
medical history), and **49 diagnoses** — including Pneumonia, Bronchitis, URTI
(common cold), Influenza, Tuberculosis and COPD exacerbation. Far richer than
the 41-disease toy dataset we originally planned around.

**(b) We found the missing link inside the X-ray dataset.** Every Open-I chest
X-ray report carries an **"indication"** field — the referring doctor's reason
for ordering the scan, written in free text: *"68-year-old with shortness of
breath and weakness. Hypertension. Diabetes."* We parse that text into the
**same 223 DDXPlus question codes**. Now every X-ray has a symptom vector
belonging to **that same patient**. This is the paired multimodal data the
project needed.

### Evidence that the link is real

Measured on all 3,851 Open-I patients — the probability of an X-ray finding
given the symptom, divided by its base rate ("lift"):

| Symptom in indication | X-ray finding | Patients | Normal rate | Lift |
|---|---|---|---|---|
| Hypoxia (low oxygen) | Chest infection | 33 | 5.3% → **27.3%** | **5.2×** |
| Previous pneumonia | Chest infection | 59 | 5.3% → **20.3%** | **3.8×** |
| Heart disease / CHF | Cardiomegaly | 30 | 9.5% → **33.3%** | **3.5×** |
| Shortness of breath | Chest infection | 668 | 5.3% → **9.0%** | **1.7×** |

Symptoms genuinely predict what the radiologist sees. The fusion model is
therefore built on a real signal, not an assumption.

## 3. What is built and working today

| Component | Status | Result |
|---|---|---|
| **Symptom → diagnosis model** (DDXPlus, 223 questions → 49 diagnoses) | ✅ Trained | **99.34%** accuracy, macro-F1 **0.992**, AUC **0.9999** on 134,529 held-out patients |
| **Diabetes-risk model** (vitals → diabetic/non-diabetic) | ✅ Trained | **89.1%** accuracy, AUC **0.972** on 14,422 held-out patients |
| **Symptom encoder pre-training** (for the fusion model) | ✅ Trained | 99.33% accuracy — transfers into the fusion network |
| **Cross-modal attention fusion network** (DenseNet-121 + symptom tokens) | ✅ Code complete, verified | 9.0M parameters, forward pass + attention weights verified |
| **Grad-CAM explainability** | ✅ Working | Produces valid heat-maps over the radiograph |
| **Rule-based triage engine** | ✅ Working | LOW / MODERATE / HIGH / EMERGENCY with written reasons |
| **Streamlit demo application** | ✅ Built | Questionnaire + vitals + X-ray upload → predictions, Grad-CAM, triage |
| **Automated tests** | ✅ 6/6 passing | Model shapes, attention masking, Grad-CAM, triage logic |
| **Chest X-ray images download** | ✅ Complete | Frontal radiographs fetched from the official NLM Open-I API; 3,105 usable paired cases (train 2,173 / val 466 / test 466) |
| **Open-I model training** | ⏳ Running | Image-only, clinical-only, fusion and the no-pretrain ablation, all on the same split |

### Per-disease performance of the symptom model (held-out test set)

| Diagnosis | Precision | Recall | F1 | Test cases |
|---|---|---|---|---|
| Pneumonia | 1.000 | 1.000 | 1.000 | 3,542 |
| Bronchitis | 0.999 | 0.997 | 0.998 | 3,594 |
| URTI (common cold) | 0.999 | 0.993 | 0.996 | 8,743 |
| Influenza | 1.000 | 1.000 | 1.000 | 3,554 |
| Tuberculosis | 1.000 | 1.000 | 1.000 | 2,080 |

> Note: DDXPlus is a *synthetic* dataset generated from a clinical knowledge
> base, so these numbers are high by construction. They demonstrate the
> pipeline is correct; the honest test of the project is the fusion model on
> real Open-I radiographs, which is what we train next.

## 4. System architecture as implemented

```
 Symptoms (223 standard questions)          Chest X-ray (frontal)
              │                                      │
    SymptomEncoder (pre-trained                DenseNet-121
     on 610k DDXPlus patients)               (ImageNet init, 7×7×1024)
              │                                      │
              └──────► bidirectional cross-modal attention ◄──────┘
                   (each symptom attends to lung regions,
                    image summary attends to symptoms)
                                  │
                   8 thoracic findings, simultaneously:
            normal · chest infection · cardiomegaly · pleural effusion
            opacity · atelectasis · emphysema/COPD · pulmonary edema
                                  │
   Vitals ──► Diabetes model ──► TRIAGE ENGINE ──► Grad-CAM explanation
                                  │
              LOW / MODERATE / HIGH / EMERGENCY + referral + advisory
```

The triage engine encodes the comorbidity interaction that motivated the whole
project: a suspected chest infection in a patient with elevated diabetes risk
is escalated above either finding on its own, because diabetes impairs immune
response and worsens pneumonia outcomes.

## 5. Results of the honest evaluation — complete

Four models trained on the **same patients and the same split**
(2,421 / 519 / 519), so the comparison is fair:

| Model | Macro AUC |
|---|---|
| **Fusion, without DDXPlus pre-training** | **0.8416** |
| X-ray only (DenseNet-121) | 0.8216 |
| Fusion, with pre-training | 0.8084 |
| Symptoms only (SymptomEncoder) | 0.6221 |

**The central claim is demonstrated:** fusion beats the image-only baseline by
+0.020 macro AUC and ranks first on 4 of the 8 findings, with its largest gains
on chest infection (+0.087) and atelectasis (+0.055).

Two qualifications are reported with equal prominence:

1. **Pre-training backfired.** The best fusion arm is the one *without* DDXPlus
   weight transfer; transferring them dropped performance to 0.8084, below even
   the image baseline. The encoder was fitted to dense DDXPlus questionnaires,
   whereas Open-I yields a mean of 1.35 symptoms per patient. The shared
   *vocabulary* is still what makes any paired dataset exist — it is the
   *weight transfer* that failed. Only the ablation revealed this.
2. **The +0.020 margin is from a single split** with no significance test. It
   should be read as a consistent trend, not an established effect size.

An error analysis (false positives / negatives) found a **52% miss rate on
chest infection** at the F1-tuned threshold, and showed it is a threshold
artefact: at a screening threshold of 0.15, sensitivity rises to 0.93.

> Full write-up with all figures: **`reports/PROJECT_REPORT.md`**

## 6. Immediate next steps

1. ~~Download the frontal chest radiographs (official NLM source)~~ — done
2. ~~Train the four Open-I models including the ablation~~ — done
3. ~~Generate the comparison table, ROC curves and error analysis~~ — done
4. ~~Write the full project report with diagrams~~ — done
   (`reports/PROJECT_REPORT.md`)
5. **Adopt per-finding screening thresholds** — highest-impact remaining change,
   needs no retraining: converts a 52% chest-infection miss rate into 7%
6. **Run multiple seeds + a DeLong test** so the +0.020 margin is defensible
7. Update the presentation — slides 11, 16, 19 and the GitHub URL on slide 20

---

*Research prototype for an academic project. This is clinical decision
**support**, not a diagnosis, and requires qualified medical review.*
