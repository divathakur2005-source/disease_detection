"""Central paths, constants and the shared clinical vocabulary.

The clinical vocabulary is the key design decision of this project: every
symptom / antecedent is expressed as a *DDXPlus evidence code* (E_xx) so that

  * the symptom model trained on DDXPlus (475k patients, 49 diagnoses),
  * the free-text indication of an Open-I chest X-ray report, and
  * the check-boxes a user ticks in the app

all live in the same feature space, and one SymptomEncoder can be pre-trained
on DDXPlus and then fused with the X-ray branch on Open-I.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MEDSUPPORT_ROOT", Path(__file__).resolve().parents[2]))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES_DIR = DATA_PROCESSED / "images"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"          # metrics json, plots, confusion matrices
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_PROCESSED, IMAGES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 224
IMG_STORE_SIZE = 320       # images are cached at this resolution, random-cropped/resized to IMG_SIZE

# --------------------------------------------------------------------------- #
# Thoracic labels derived from Open-I MeSH annotations (multi-label).
# "chest_infection" groups the findings a radiologist associates with an
# infective / inflammatory lower-respiratory process (pneumonia, infiltrate,
# consolidation, airspace disease).
# --------------------------------------------------------------------------- #
THORACIC_LABELS = {
    "normal": r"^normal$",
    "chest_infection": r"pneumonia|infiltrat|consolidat|airspace disease|bronchitis|bronchiectasis",
    "cardiomegaly": r"cardiomegaly|cardiac shadow/enlarged|heart/enlarged",
    "pleural_effusion": r"pleural effusion|costophrenic angle.*blunted",
    "opacity": r"opacity|nodule|\bmass\b|granuloma",
    "atelectasis": r"atelectasis",
    "emphysema_copd": r"emphysema|pulmonary disease, chronic obstructive|hyperdistention|hyperinflation",
    "edema_congestion": r"pulmonary edema|pulmonary congestion",
}
THORACIC_LABEL_NAMES = list(THORACIC_LABELS)

# --------------------------------------------------------------------------- #
# Open-I-only vocabulary extras: concepts that appear in X-ray indications but
# have no DDXPlus evidence question. They get their own embedding rows which are
# trained only during fusion.
# --------------------------------------------------------------------------- #
EXTRA_EVIDENCES = {
    "X_HYPOXIA": "Hypoxia / low oxygen saturation",
    "X_TB_SCREEN": "Tuberculosis screening (positive TB test / PPD)",
    "X_PREOP": "Pre-operative / clearance screening",
    "X_TRAUMA": "Trauma, fall or accident",
    "X_PAIN_OTHER": "Pain outside the chest (back, abdomen, shoulder, neck)",
    "X_DEVICE": "Line / tube / pacemaker placement check",
    "X_FEMALE": "Female patient",
    "X_MALE": "Male patient",
}

# Regular expression over the lower-cased indication text  ->  evidence codes.
# Note: Open-I reports are de-identified with "XXXX" tokens, which removes some
# words (e.g. most occurrences of "cough"); the patterns below are the ones that
# survive de-identification.
INDICATION_TO_EVIDENCE: list[tuple[str, list[str]]] = [
    (r"\bcough",                                                 ["E_201"]),
    (r"productive|sputum",                                       ["E_201", "E_77"]),
    (r"\bfever|febrile|pyrexia",                                 ["E_91"]),
    (r"\bchill|rigor",                                           ["E_94"]),
    (r"short(ness)? of breath|\bsob\b|dyspn|difficulty breathing|respiratory distress|trouble breathing",
                                                                 ["E_66"]),
    (r"chest pain|chest tightness|chest pressure|chest discomfort|chest wall pain", ["E_14"]),
    (r"pleuritic|pain (with|on) (deep )?(breath|inspiration)",   ["E_14", "E_220"]),
    (r"wheez",                                                   ["E_214"]),
    (r"hemoptysis|haemoptysis|coughing (up )?blood|blood in sputum", ["E_45"]),
    (r"sore throat|pharyngitis",                                 ["E_97"]),
    (r"congestion|runny nose|rhinorrhea|sinus",                  ["E_181"]),
    (r"sinusitis",                                               ["E_119"]),
    (r"fatigue|tired",                                           ["E_89"]),
    (r"weakness|malaise|generalized|body ache|myalgia",          ["E_175"]),
    (r"night sweat|diaphoresis|sweat",                           ["E_50"]),
    (r"weight loss",                                             ["E_162"]),
    (r"nausea|vomit|emesis",                                     ["E_148"]),
    (r"palpitation|tachycard|arrhythm|\bafib\b|atrial fib|\ba-?fib\b|irregular heart", ["E_155"]),
    (r"syncope|faint|passed out|loss of consciousness",          ["E_159"]),
    (r"dizz|lightheaded|light-headed|vertigo",                   ["E_82"]),
    (r"\bedema|oedema|swelling|swollen",                         ["E_151"]),
    (r"confus|altered mental|disorient",                         ["E_39"]),
    (r"smok|tobacco",                                            ["E_79"]),
    (r"diabet|\bdm\b|hyperglyc",                                 ["E_69"]),
    (r"hypertens|\bhtn\b|high blood pressure",                   ["E_104"]),
    (r"\bchf\b|heart failure|cardiomyopathy",                    ["E_106"]),
    (r"coronary|\bcad\b|myocardial|angina|heart attack|\bmi\b|stent|bypass|cabg", ["E_105"]),
    (r"heart valve|valvular|murmur|heart defect",                ["E_139"]),
    (r"asthma",                                                  ["E_124"]),
    (r"\bcopd\b|emphysema|chronic obstructive",                  ["E_123"]),
    (r"pneumonia",                                               ["E_118"]),
    (r"\bflu\b|influenza|viral|\bcold\b|\buri\b|upper respiratory", ["E_0"]),
    (r"\bhiv\b|aids\b",                                          ["E_2"]),
    (r"cancer|malignan|carcinoma|neoplasm|lymphoma|leukemia|metasta|tumor|\bmass\b", ["E_34"]),
    (r"chemo|immunosupp|transplant|immunocomp",                  ["E_227"]),
    (r"renal|kidney|dialysis|\besrd\b|\bckd\b",                  ["E_113"]),
    (r"sleep apnea|\bosa\b",                                     ["E_186"]),
    (r"\bdvt\b|thrombo|embol|\bpe\b",                            ["E_109"]),
    (r"stroke|\bcva\b|\btia\b",                                  ["E_107"]),
    (r"anemia|anaemia",                                          ["E_24"]),
    (r"reflux|\bgerd\b|heartburn",                               ["E_125"]),
    (r"alcohol|etoh",                                            ["E_78"]),
    (r"\biv ?drug|ivdu|intravenous drug",                        ["E_61"]),
    (r"lymph",                                                   ["E_9"]),
    (r"recent surgery|post-?op|s/p surgery|surgery (last|within)", ["E_196"]),
    # Open-I-only extras
    (r"hypox|low (oxygen|o2|sat)|desat|o2 sat",                  ["X_HYPOXIA"]),
    (r"\btb\b|tuberc|\bppd\b|quantiferon",                       ["X_TB_SCREEN"]),
    (r"pre-?op|preoperative|clearance|prior to surgery|pre-?surg", ["X_PREOP"]),
    (r"trauma|\bfall\b|fell\b|accident|\bmvc\b|\bmva\b|injur|assault|collision|rollover", ["X_TRAUMA"]),
    (r"back pain|abdominal|\babd\b|epigastric|shoulder|flank|neck pain|rib pain|arm pain|hip pain", ["X_PAIN_OTHER"]),
    (r"\bline\b|catheter|pacemaker|\bpicc\b|port\b|\btube\b|placement|\bicd\b|defibrillator", ["X_DEVICE"]),
    (r"female|woman|\bshe\b|\bher\b",                            ["X_FEMALE"]),
    (r"\bmale\b|\bman\b|\bhe\b|\bhis\b",                         ["X_MALE"]),
]
