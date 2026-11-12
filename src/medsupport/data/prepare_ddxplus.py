"""Dataset-1: DDXPlus (Kaggle mirror "disease-symptom-prediction-dataset").

475k train / 134k test synthetic patients, 223 binary evidence questions
(symptoms + antecedents), AGE, SEX, 49 diagnoses.

Produces data/processed/ddxplus_{train,val,test}.parquet with
    E_0 ... E_227  (int8 multi-hot, canonical DDXPlus codes)
    age_norm       (AGE / 100)
    sex_male       (0/1)
    disease_id     (0..48)
and ddxplus_meta.json with class names, evidence codes and their English
question text (used by the app to render the questionnaire).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from medsupport.config import DATA_PROCESSED, DATA_RAW, SEED
from medsupport.utils import save_json

RAW = DATA_RAW / "symptom"
META = DATA_RAW / "ddxplus_meta"


def canonical_columns(cols: list[str]) -> dict[str, str]:
    """'E_116_have_you_had_a_cold' -> 'E_116'."""
    out = {}
    for c in cols:
        m = re.match(r"^(E_\d+)", c)
        if m:
            out[c] = m.group(1)
    return out


def load(part: str, evidence_codes: list[str]) -> pd.DataFrame:
    path = RAW / f"{part}.csv"
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    rename = canonical_columns(cols)
    dtypes = {c: "int8" for c in rename}
    df = pd.read_csv(path, dtype=dtypes)
    df = df.rename(columns=rename)
    missing = [e for e in evidence_codes if e not in df.columns]
    for e in missing:
        df[e] = np.int8(0)
    out = df[evidence_codes].copy()
    out["age_norm"] = (df["AGE"].astype(float) / 100.0).astype("float32")
    out["sex_male"] = (df["SEX"] == "M").astype("int8")
    out["disease"] = df["DISEASE"].str.strip()
    return out


def main() -> None:
    evid = json.load(open(META / "release_evidences.json", encoding="utf-8"))
    cond = json.load(open(META / "release_conditions.json", encoding="utf-8"))
    evidence_codes = sorted(evid, key=lambda k: int(k.split("_")[1]))

    train_full = load("train", evidence_codes)
    test = load("test", evidence_codes)
    classes = sorted(set(train_full["disease"]) | set(test["disease"]))
    cls_id = {c: i for i, c in enumerate(classes)}
    for df in (train_full, test):
        df["disease_id"] = df["disease"].map(cls_id).astype("int16")
        df.drop(columns=["disease"], inplace=True)

    train, val = train_test_split(train_full, test_size=0.1, stratify=train_full["disease_id"], random_state=SEED)
    for name, df in {"train": train, "val": val, "test": test}.items():
        df.to_parquet(DATA_PROCESSED / f"ddxplus_{name}.parquet", index=False)
        print(name, df.shape)

    icd = {c["condition_name"].strip(): c.get("icd10-id", "") for c in cond.values()}
    meta = {
        "classes": classes,
        "icd10": {c: icd.get(c, "") for c in classes},
        "evidence_codes": evidence_codes,
        "questions": {k: evid[k]["question_en"] for k in evidence_codes},
        "is_antecedent": {k: bool(evid[k]["is_antecedent"]) for k in evidence_codes},
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "class_counts_train": {c: int((train["disease_id"] == i).sum()) for c, i in cls_id.items()},
    }
    save_json(meta, DATA_PROCESSED / "ddxplus_meta.json")
    print("classes", len(classes), "evidences", len(evidence_codes))


if __name__ == "__main__":
    main()
