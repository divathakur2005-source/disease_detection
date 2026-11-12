"""Dataset-2: Open-I / Indiana University chest X-rays (3,851 reports).

Three things are produced:
  1. Multi-label thoracic targets from the MeSH annotations (config.THORACIC_LABELS).
  2. A multi-hot clinical vector from the free-text *indication* (the referring
     physician's reason for the X-ray: symptoms + history) parsed into the
     shared DDXPlus evidence vocabulary (config.INDICATION_TO_EVIDENCE).
  3. Frontal images extracted from the Hugging Face parquet mirror
     (ykumards/open-i) and cached as 320px PNGs in data/processed/images/.

Output: data/processed/openi_{train,val,test}.parquet, openi_meta.json.
Split is by patient (uid), stratified on the chest_infection label.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from medsupport.config import (DATA_PROCESSED, DATA_RAW, EXTRA_EVIDENCES, IMAGES_DIR, IMG_STORE_SIZE,
                               INDICATION_TO_EVIDENCE, SEED, THORACIC_LABELS)
from medsupport.utils import load_json, save_json

REPORTS = DATA_RAW / "openi_meta" / "indiana_reports.csv"
PARQUET_DIR = DATA_RAW / "openi"


def vocabulary() -> list[str]:
    meta = load_json(DATA_PROCESSED / "ddxplus_meta.json")
    return list(meta["evidence_codes"]) + list(EXTRA_EVIDENCES)


def parse_indication(text: str, vocab_index: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(vocab_index), dtype=np.int8)
    t = (text or "").lower()
    for pattern, codes in INDICATION_TO_EVIDENCE:
        if re.search(pattern, t):
            for c in codes:
                v[vocab_index[c]] = 1
    # "female" also matches "male" patterns via \bmale\b? No - but "woman"/"man": guard explicitly.
    if v[vocab_index["X_FEMALE"]] and re.search(r"female|woman|\bshe\b|\bher\b", t) and not re.search(r"\bmale\b|\bman\b|\bhe\b|\bhis\b", t):
        v[vocab_index["X_MALE"]] = 0
    return v


def labels_from_mesh(mesh: str) -> np.ndarray:
    m = (mesh or "").lower()
    return np.array([1 if re.search(p, m) else 0 for p in THORACIC_LABELS.values()], dtype=np.int8)


def extract_images() -> dict[int, str]:
    """Reads every parquet shard present, saves the frontal image of each uid as PNG.
    Returns {uid: relative path}. Shards that are missing/incomplete are skipped
    (so the script can be re-run while the download is still in progress)."""
    import pyarrow.parquet as pq

    found: dict[int, str] = {}
    shards = sorted(PARQUET_DIR.glob("*.parquet"))
    if not shards:
        print("!! no parquet shards in", PARQUET_DIR, "- images skipped")
        return found
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        try:
            pf = pq.ParquetFile(shard)
        except Exception as e:  # incomplete download
            print(f"!! skipping {shard.name}: {e}")
            continue
        for batch in tqdm(pf.iter_batches(batch_size=16, columns=["uid", "img_frontal"]), desc=shard.name,
                          total=(pf.metadata.num_rows + 15) // 16):
            uids = batch.column("uid").to_pylist()
            imgs = batch.column("img_frontal").to_pylist()
            for uid, raw in zip(uids, imgs):
                if raw is None:
                    continue
                out = IMAGES_DIR / f"{uid}.png"
                if not out.exists():
                    im = Image.open(io.BytesIO(raw)).convert("L")
                    im.thumbnail((IMG_STORE_SIZE * 2, IMG_STORE_SIZE * 2))
                    w, h = im.size
                    s = IMG_STORE_SIZE / min(w, h)
                    im = im.resize((max(IMG_STORE_SIZE, round(w * s)), max(IMG_STORE_SIZE, round(h * s))),
                                   Image.BILINEAR)
                    im.save(out, optimize=True)
                found[int(uid)] = str(out.relative_to(DATA_PROCESSED))
    return found


def main() -> None:
    vocab = vocabulary()
    vidx = {c: i for i, c in enumerate(vocab)}
    rep = pd.read_csv(REPORTS)
    rep["uid"] = rep["uid"].astype(int)

    Y = np.stack([labels_from_mesh(m) for m in rep["MeSH"].fillna("")])
    X = np.stack([parse_indication(t, vidx) for t in rep["indication"].fillna("")])
    label_names = list(THORACIC_LABELS)
    print("label positives:", dict(zip(label_names, Y.sum(0).tolist())))
    hits = X.sum(0)
    top = sorted(zip(vocab, hits.tolist()), key=lambda x: -x[1])[:25]
    print("most frequent parsed evidences:", top)
    print("reports with >=1 parsed evidence:", int((X.sum(1) > 0).sum()), "/", len(rep))

    images = extract_images()
    rep["image_path"] = rep["uid"].map(images)
    has_img = rep["image_path"].notna()
    print("reports with frontal image:", int(has_img.sum()))
    if has_img.sum() == 0:
        print("!! no images available yet - download the parquet shards (scripts/download_data.py) and re-run")
        return

    df = pd.DataFrame(X, columns=vocab)
    for j, n in enumerate(label_names):
        df[f"y_{n}"] = Y[:, j]
    df.insert(0, "uid", rep["uid"].values)
    df["image_path"] = rep["image_path"].values
    df["indication"] = rep["indication"].fillna("").values
    df["mesh"] = rep["MeSH"].fillna("").values
    df["impression"] = rep["impression"].fillna("").values
    df = df[has_img.values].reset_index(drop=True)

    strat = df["y_chest_infection"].astype(str) + "_" + df["y_normal"].astype(str)
    tr, tmp = train_test_split(df, test_size=0.3, stratify=strat, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=strat.loc[tmp.index], random_state=SEED)
    for name, part in {"train": tr, "val": va, "test": te}.items():
        part.to_parquet(DATA_PROCESSED / f"openi_{name}.parquet", index=False)
        print(name, part.shape, "chest_infection+", int(part["y_chest_infection"].sum()))

    save_json({"vocab": vocab, "extra_evidences": EXTRA_EVIDENCES, "label_names": label_names,
               "label_positives": dict(zip(label_names, Y[has_img.values].sum(0).tolist())),
               "n_reports": int(len(rep)), "n_with_image": int(has_img.sum()),
               "evidence_hits": dict(zip(vocab, X[has_img.values].sum(0).tolist()))},
              DATA_PROCESSED / "openi_meta.json")


if __name__ == "__main__":
    main()
