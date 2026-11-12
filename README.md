# AI-Based Medical Decision-Support System

**Major Project - I (18B19CI791), Group RG24, JUIT - Department of CSE & IT**
Supervisor: Dr. Hari Singh

A multimodal deep-learning system that combines **patient-reported symptoms /
history (tabular)** with a **chest radiograph (image)** to predict several
conditions at once, explain the image decision with Grad-CAM, and produce a
rule-based triage recommendation.

> Research prototype for an academic project. It is decision *support*, not a
> diagnosis, and must not be used without qualified medical review.

---

## 1. The core idea: one clinical vocabulary for every data source

The three public datasets in the proposal do not share patients, so there was
no natural way to "fuse" them. The fix is a **shared clinical vocabulary**:

* **DDXPlus** (Kaggle *disease-symptom-prediction-dataset*): 609k synthetic
  patients, 223 yes/no evidence questions (symptoms + antecedents), 49
  diagnoses incl. *Pneumonia, Bronchitis, URTI (common cold), Influenza,
  Tuberculosis, COPD exacerbation*. Its evidence codes `E_xx` become the
  vocabulary.
* **Open-I / Indiana University chest X-rays**: 3,851 reports with the
  radiologist's MeSH labels **and** the referring physician's *indication*
  (why the X-ray was ordered: "shortness of breath, fever, diabetic"). The
  indication is parsed with regular expressions into the same `E_xx` codes, so
  every X-ray comes with a symptom vector **from the same patient**. This is
  the paired multimodal data.
* **Diabetes Prediction dataset** (Kaggle, 100k): vitals -> diabetes risk, used
  by the triage engine as the metabolic comorbidity signal.

```
                 DDXPlus (609k patients)                 Open-I (3.8k X-rays + indications)
                 ┌───────────────────────┐              ┌──────────────┐  ┌──────────────────┐
 symptoms  ───►  │  MLP: 223 evid.+age+  │              │ DenseNet-121 │  │ SymptomEncoder   │
 (E_xx codes)    │  sex -> 49 diagnoses  │              │ (7x7x1024)   │  │ (pre-trained on  │
                 └───────────┬───────────┘              └──────┬───────┘  │  DDXPlus)        │
                             │ pre-trained SymptomEncoder      │          └────────┬─────────┘
                             └──────────────────────────────────────────────────────┘
                                                                 ▼
                                               bidirectional cross-modal attention
                                               (symptoms ⇄ image patches) + harmonisation
                                                                 ▼
                                   8 thoracic labels: normal, chest_infection, cardiomegaly,
                                   pleural_effusion, opacity, atelectasis, emphysema/COPD, edema
                                                                 ▼
 vitals ───► diabetes MLP ───►  rule-based TRIAGE (LOW / MODERATE / HIGH / EMERGENCY) + Grad-CAM
```

## 2. Repository layout

```
src/medsupport/
  config.py              paths, thoracic label definitions, indication->evidence vocabulary
  data/prepare_ddxplus.py   DDXPlus csv -> parquet splits + question metadata
  data/prepare_diabetes.py  diabetes csv -> scaled parquet splits
  data/prepare_openi.py     Open-I reports -> labels + evidence vectors; extracts frontal images
  models/clinical.py     ClinicalMLP, SymptomEncoder (symptom tokens), SymptomEncoderClassifier
  models/imaging.py      DenseNet-121 feature extractor (ImageNet init)
  models/fusion.py       CrossModalFusionNet - bidirectional cross-attention fusion
  explain/gradcam.py     Grad-CAM on the DenseNet feature map
  triage.py              transparent rule-based triage engine
  train/tabular.py       trainer for diabetes / ddxplus / symptom_encoder
  train/imaging.py       trainer for image-only / clinical-only / fusion on Open-I
scripts/
  download_data.py       fetches all datasets into data/raw
  analyze_correlation.py symptom <-> X-ray label lift analysis (evidence for fusion)
  compare_openi.py       comparison table + ROC curves across the Open-I models
app/app.py               Streamlit demo
reports/                 metrics json, comparison tables, figures (committed)
models/                  trained weights (not committed, see .gitignore)
```

## 3. Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # ~2.5 GB, mostly X-ray images
set PYTHONPATH=src                              # (export on Linux/macOS)

python -m medsupport.data.prepare_diabetes
python -m medsupport.data.prepare_ddxplus
python -m medsupport.data.prepare_openi

python -m medsupport.train.tabular diabetes
python -m medsupport.train.tabular ddxplus --bs 2048 --lr 2e-3 --hidden 512,256
python -m medsupport.train.tabular symptom_encoder --bs 2048 --lr 2e-3 --hidden 256   # pre-training for fusion

python -m medsupport.train.imaging image
python -m medsupport.train.imaging clinical
python -m medsupport.train.imaging fusion --init-image models/openi_image.pt
python -m medsupport.train.imaging fusion --no-pretrain --tag _nopretrain            # ablation

python scripts/analyze_correlation.py
python scripts/compare_openi.py
streamlit run app/app.py
```

## 4. Results

See `reports/` (filled in by the scripts above; summary table maintained in
[reports/RESULTS.md](reports/RESULTS.md)).

## 5. Datasets & credits

| # | Dataset | Source | Used for |
|---|---|---|---|
| 1 | DDXPlus (Kaggle mirror "Disease Symptom Prediction") | https://www.kaggle.com/datasets/muherezaalouzious/disease-symptom-prediction-dataset ; question texts from https://huggingface.co/datasets/aai530-group6/ddxplus | symptom -> diagnosis model; SymptomEncoder pre-training |
| 2 | Chest X-rays (Indiana University, Open-I) | https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university (images via https://huggingface.co/datasets/ykumards/open-i) | image branch, fusion, Grad-CAM |
| 3 | Diabetes Prediction Dataset | https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset | diabetes-risk model |
| 4 | Disease Symptom Description (41 diseases) | https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset | precaution / advisory text in the app |

Open-I: Demner-Fushman et al., *Preparing a collection of radiology
examinations for distribution and retrieval*, JAMIA 2016. DDXPlus: Fansi
Tchango et al., *DDXPlus: A New Dataset for Automatic Medical Diagnosis*,
NeurIPS Datasets & Benchmarks 2022.
