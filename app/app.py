"""AI-Based Medical Decision-Support System - Streamlit demo.

    streamlit run app/app.py

Flow:  symptoms & history (questionnaire)  +  vitals  +  chest X-ray
       -> symptom model (DDXPlus, 49 diagnoses)
       -> diabetes-risk model
       -> cross-modal fusion model (X-ray + symptoms -> 8 thoracic findings) with Grad-CAM
       -> rule-based triage + advisory
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medsupport.config import DATA_PROCESSED, DATA_RAW, EXTRA_EVIDENCES, MODELS_DIR, THORACIC_LABEL_NAMES  # noqa: E402
from medsupport.data.prepare_diabetes import FEATURES as DIAB_FEATURES, SMOKING  # noqa: E402
from medsupport.explain import GradCAM, overlay_heatmap  # noqa: E402
from medsupport.models import ClinicalMLP, CrossModalFusionNet, ImageEncoder  # noqa: E402
from medsupport.train.imaging import EVAL_TF  # noqa: E402
from medsupport.triage import triage  # noqa: E402
from medsupport.utils import load_json  # noqa: E402

st.set_page_config(page_title="AI Medical Decision Support", page_icon="🩺", layout="wide")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evidence codes shown as quick check-boxes (common respiratory / systemic symptoms & history)
QUICK = ["E_201", "E_77", "E_91", "E_94", "E_66", "E_64", "E_14", "E_220", "E_214", "E_45", "E_97", "E_181",
         "E_182", "E_89", "E_175", "E_50", "E_162", "E_148", "E_155", "E_82", "E_151", "E_39",
         "E_79", "E_69", "E_104", "E_106", "E_105", "E_124", "E_123", "E_118", "E_0", "E_116", "E_227", "E_113"]
COLD_LIKE = {"URTI", "Viral pharyngitis", "Acute rhinosinusitis", "Allergic sinusitis", "Acute laryngitis", "Influenza"}
CHEST_INFECTION_LIKE = {"Pneumonia", "Bronchitis", "Acute COPD exacerbation / infection", "Bronchiectasis", "Tuberculosis"}


@st.cache_resource
def load_models():
    out = {}
    ddx_meta = load_json(DATA_PROCESSED / "ddxplus_meta.json")
    out["ddx_meta"] = ddx_meta
    ck = torch.load(MODELS_DIR / "ddxplus.pt", map_location="cpu", weights_only=False)
    m = ClinicalMLP(ck["in_dim"], ck["n_classes"], hidden=tuple(ck["hidden"]), dropout=ck["dropout"])
    m.load_state_dict(ck["state_dict"])
    out["ddx"] = (m.to(DEV).eval(), ck["features"], ck["classes"])

    ck = torch.load(MODELS_DIR / "diabetes.pt", map_location="cpu", weights_only=False)
    m = ClinicalMLP(ck["in_dim"], ck["n_classes"], hidden=tuple(ck["hidden"]), dropout=ck["dropout"])
    m.load_state_dict(ck["state_dict"])
    out["diab"] = (m.to(DEV).eval(), load_json(DATA_PROCESSED / "diabetes_meta.json"))

    fusion_ck = MODELS_DIR / "openi_fusion.pt"
    if fusion_ck.exists():
        ck = torch.load(fusion_ck, map_location="cpu", weights_only=False)
        m = CrossModalFusionNet(len(ck["vocab"]), len(ck["labels"]), pretrained=False, freeze_until=None)
        m.load_state_dict(ck["state_dict"])
        out["fusion"] = (m.to(DEV).eval(), ck["vocab"], ck["labels"], np.array(ck["thresholds"]))
    image_ck = MODELS_DIR / "openi_image.pt"
    if image_ck.exists():
        ck = torch.load(image_ck, map_location="cpu", weights_only=False)
        m = ImageEncoder(n_classes=len(ck["labels"]), pretrained=False)
        m.load_state_dict(ck["state_dict"])
        out["image"] = (m.to(DEV).eval(), ck["labels"], np.array(ck["thresholds"]))
    desc = pd.read_csv(DATA_RAW / "symptom_desc" / "symptom_precaution.csv")
    out["precautions"] = {r["Disease"].strip().lower(): [str(r[f"Precaution_{i}"]) for i in range(1, 5) if pd.notna(r[f"Precaution_{i}"])]
                          for _, r in desc.iterrows()}
    return out


M = load_models()
ddx_model, ddx_feats, ddx_classes = M["ddx"]
questions = M["ddx_meta"]["questions"]

st.title("🩺 AI-Based Medical Decision-Support System")
st.caption("Multimodal deep learning: patient-reported symptoms + chest X-ray → multi-disease prediction, "
           "Grad-CAM explanation and triage. **Decision support only - not a diagnosis.**")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("1. Patient & symptoms")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 0, 110, 45)
    sex = c2.selectbox("Sex", ["Female", "Male"])
    st.markdown("**Tick everything that applies**")
    ticked: set[str] = set()
    cols = st.columns(2)
    for i, code in enumerate(QUICK):
        if cols[i % 2].checkbox(questions[code], key=code):
            ticked.add(code)
    with st.expander("More questions (full DDXPlus questionnaire)"):
        others = [c for c in ddx_feats if c.startswith("E_") and c not in QUICK]
        more = st.multiselect("Select additional evidences", others, format_func=lambda c: f"{c}: {questions[c]}")
        ticked |= set(more)

    st.subheader("2. Vitals (diabetes risk)")
    v1, v2, v3 = st.columns(3)
    bmi = v1.number_input("BMI", 10.0, 80.0, 26.0)
    hba1c = v2.number_input("HbA1c (%)", 3.0, 15.0, 5.6)
    glucose = v3.number_input("Blood glucose (mg/dL)", 50, 400, 110)
    v4, v5, v6 = st.columns(3)
    hyp = v4.checkbox("Hypertension (diagnosed)", value="E_104" in ticked)
    hd = v5.checkbox("Heart disease (diagnosed)", value="E_105" in ticked or "E_106" in ticked)
    smoking = v6.selectbox("Smoking history", SMOKING, index=0)

with right:
    st.subheader("3. Chest X-ray (optional)")
    up = st.file_uploader("Upload a frontal chest radiograph (PNG/JPG)", type=["png", "jpg", "jpeg"])
    sample = None
    sample_codes: set[str] = set()
    if up is None:
        test = pd.read_parquet(DATA_PROCESSED / "openi_test.parquet") if (DATA_PROCESSED / "openi_test.parquet").exists() else None
        if test is not None:
            # label -> positional index, so duplicate indication texts cannot pick the wrong row
            options = {f"[{i}] uid {u}: {str(ind)[:55]}": i
                       for i, (u, ind) in enumerate(zip(test["uid"], test["indication"]))}
            pick = st.selectbox("...or pick a sample from the Open-I test set", ["(none)"] + list(options))
            if pick != "(none)":
                sample = test.iloc[options[pick]]
    img_pil = None
    if up is not None:
        img_pil = Image.open(up).convert("RGB")
    elif sample is not None:
        img_pil = Image.open(DATA_PROCESSED / sample["image_path"]).convert("RGB")
        st.info(f"Indication: *{sample['indication']}*  \nRadiologist MeSH: `{sample['mesh']}`")
        # evidences parsed from this patient's own indication; merged into `ticked`
        # below (the quick check-boxes have already been rendered at this point)
        for code in (M["fusion"][1] if "fusion" in M else []):
            if code in sample.index and sample[code] == 1:
                sample_codes.add(code)
    if img_pil is not None:
        st.image(img_pil, caption="Input radiograph", use_container_width=True)

if sample_codes:
    ticked |= sample_codes
    st.caption("Symptoms parsed from the selected patient's referral: "
               + ", ".join(sorted(questions.get(c, EXTRA_EVIDENCES.get(c, c))[:40] for c in sample_codes)))

run = st.button("Run decision support", type="primary", use_container_width=True)

if run:
    # ---------------- symptom model ----------------
    x = np.zeros(len(ddx_feats), dtype=np.float32)
    for code in ticked:
        if code in ddx_feats:
            x[ddx_feats.index(code)] = 1
    x[ddx_feats.index("age_norm")] = age / 100.0
    x[ddx_feats.index("sex_male")] = 1.0 if sex == "Male" else 0.0
    with torch.no_grad():
        p_ddx = torch.softmax(ddx_model(torch.from_numpy(x)[None].to(DEV)), 1)[0].cpu().numpy()
    order = np.argsort(-p_ddx)[:5]
    p_cold = float(sum(p_ddx[i] for i, c in enumerate(ddx_classes) if c in COLD_LIKE))
    p_ci_sym = float(sum(p_ddx[i] for i, c in enumerate(ddx_classes) if c in CHEST_INFECTION_LIKE))

    # ---------------- diabetes model ----------------
    dm_model, dm_meta = M["diab"]
    row = {"age": age, "hypertension": int(hyp), "heart_disease": int(hd), "bmi": bmi, "HbA1c_level": hba1c,
           "blood_glucose_level": glucose, "gender_male": 1.0 if sex == "Male" else 0.0}
    for s in SMOKING:
        row[f"smoking_{s.replace(' ', '_')}"] = 1.0 if smoking == s else 0.0
    for c in dm_meta["numeric"]:
        row[c] = (row[c] - dm_meta["mean"][c]) / dm_meta["std"][c]
    xd = np.array([row[c] for c in DIAB_FEATURES], dtype=np.float32)
    with torch.no_grad():
        p_dm = float(torch.softmax(dm_model(torch.from_numpy(xd)[None].to(DEV)), 1)[0, 1])

    # ---------------- imaging / fusion ----------------
    thoracic = {}
    cam_img = None
    attn_rows = None
    if img_pil is not None and "fusion" in M:
        fmodel, vocab, labels, thr = M["fusion"]
        sym = np.zeros(len(vocab), dtype=np.float32)
        for code in ticked:
            if code in vocab:
                sym[vocab.index(code)] = 1
        sym[vocab.index("X_FEMALE" if sex == "Female" else "X_MALE")] = 1
        it = EVAL_TF(img_pil)[None].to(DEV)
        st_ = torch.from_numpy(sym)[None].to(DEV)
        with torch.no_grad():
            out = fmodel(it, st_, return_attn=True)
            probs = torch.sigmoid(out["logits"])[0].cpu().numpy()
            p_img_only = torch.sigmoid(out["img_logits"])[0].cpu().numpy()
            a = out["attn_i2c"][0, 0].cpu().numpy()        # (1+S,) attention of image summary over symptoms
        thoracic = dict(zip(labels, probs.tolist()))
        present = [(vocab[i - 1], a[i]) for i in range(1, len(a)) if sym[i - 1] == 1]
        if present:
            tot = sum(w for _, w in present) or 1.0
            attn_rows = sorted([(questions.get(c, EXTRA_EVIDENCES.get(c, c)), w / tot) for c, w in present],
                               key=lambda t: -t[1])
        j = labels.index("chest_infection")
        top_j = int(np.argmax(probs[1:]) + 1) if probs[1:].max() > probs[0] else j
        cam = GradCAM(fmodel, fmodel.image.features)
        heat = cam(lambda: fmodel(it, st_)["logits"], top_j, (224, 224))
        cam.remove()
        base = np.array(img_pil.resize((224, 224)))
        cam_img = overlay_heatmap(base, heat)
        cam_label = labels[top_j]

    p_ci = thoracic.get("chest_infection", p_ci_sym)
    tri = triage(p_ci, p_dm, p_cold,
                 symptoms={"breathlessness" if "E_66" in ticked else "", "chest_pain" if "E_14" in ticked else "",
                           "hemoptysis" if "E_45" in ticked else "", "syncope" if "E_82" in ticked else ""} - {""},
                 thoracic_probs=thoracic, age=age)

    st.divider()
    st.subheader("Results")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Symptom model (DDXPlus, 49 diagnoses)**")
        for i in order:
            st.progress(float(p_ddx[i]), text=f"{ddx_classes[i]}  {p_ddx[i]:.1%}")
        st.metric("Common-cold-like (URTI/pharyngitis/...)", f"{p_cold:.0%}")
    with r2:
        st.markdown("**Diabetes-risk model**")
        st.metric("P(diabetes)", f"{p_dm:.0%}")
        if thoracic:
            st.markdown("**X-ray + symptoms (fusion)**")
            for lab in labels:
                flag = "🔴" if thoracic[lab] >= thr[labels.index(lab)] and lab != "normal" else ("🟢" if lab == "normal" and thoracic[lab] >= thr[0] else "⚪")
                st.write(f"{flag} {lab.replace('_', ' ')}: **{thoracic[lab]:.0%}**  "
                         f"<span style='color:gray'>(image-only {p_img_only[labels.index(lab)]:.0%})</span>", unsafe_allow_html=True)
    with r3:
        color = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "EMERGENCY": "red"}[tri.level]
        st.markdown(f"**Triage: :{color}[{tri.level}]**  (score {tri.score}/100)")
        st.write(tri.referral)
        for r in tri.reasons:
            st.write("• " + r)
        if tri.advisory:
            st.markdown("**Advisory**")
            for a_ in tri.advisory:
                st.write("• " + a_)
        top_dx = ddx_classes[order[0]].lower()
        for k, v in M["precautions"].items():
            if k in top_dx or top_dx in k:
                st.write("Precautions (" + k + "): " + ", ".join(v))

    if cam_img is not None:
        st.subheader("Explainability")
        e1, e2 = st.columns(2)
        e1.image(cam_img, caption=f"Grad-CAM for '{cam_label}' (fusion model)", use_container_width=True)
        if attn_rows:
            e2.markdown("**Which reported symptoms the image representation attended to**")
            e2.bar_chart(pd.DataFrame(attn_rows, columns=["symptom", "attention"]).set_index("symptom"))
    st.caption("Models: MLP on DDXPlus (symptoms→diagnosis), MLP on Kaggle diabetes data, DenseNet-121 + cross-modal "
               "attention fusion on Open-I chest X-rays. Research prototype - requires qualified medical review.")
