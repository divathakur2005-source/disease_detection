"""Dataset-3: Diabetes Prediction Dataset (Kaggle, 100k rows).

Produces data/processed/diabetes_{train,val,test}.parquet with numeric
features and a `diabetes` target, plus diabetes_meta.json (feature names,
scaler statistics) so the app can preprocess a single patient identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from medsupport.config import DATA_PROCESSED, DATA_RAW, SEED
from medsupport.utils import save_json

RAW = DATA_RAW / "diabetes" / "diabetes_prediction_dataset.csv"
SMOKING = ["never", "former", "current", "not current", "ever", "No Info"]
FEATURES = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level",
            "gender_male"] + [f"smoking_{s.replace(' ', '_')}" for s in SMOKING]
NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]:
        out[c] = df[c].astype(float)
    out["gender_male"] = (df["gender"] == "Male").astype(float)
    for s in SMOKING:
        out[f"smoking_{s.replace(' ', '_')}"] = (df["smoking_history"] == s).astype(float)
    return out[FEATURES]


def main() -> None:
    df = pd.read_csv(RAW)
    df = df.drop_duplicates().reset_index(drop=True)
    X = encode(df)
    y = df["diabetes"].astype(int)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    mean = X_tr[NUMERIC].mean()
    std = X_tr[NUMERIC].std().replace(0, 1.0)
    for part, (Xp, yp) in {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)}.items():
        Xp = Xp.copy()
        Xp[NUMERIC] = (Xp[NUMERIC] - mean) / std
        Xp["diabetes"] = yp.values
        Xp.to_parquet(DATA_PROCESSED / f"diabetes_{part}.parquet", index=False)
        print(part, Xp.shape, "positives", int(yp.sum()))

    save_json({"features": FEATURES, "numeric": NUMERIC, "mean": mean.to_dict(), "std": std.to_dict(),
               "n_rows": int(len(df)), "positive_rate": float(y.mean())},
              DATA_PROCESSED / "diabetes_meta.json")


if __name__ == "__main__":
    main()
