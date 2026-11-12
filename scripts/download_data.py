"""Download the three public datasets used by the project into data/raw/.

    python scripts/download_data.py            # everything (~2.5 GB, mostly X-ray images)
    python scripts/download_data.py --no-images

Kaggle's public download endpoint works without an account for these datasets.
The chest X-ray images come from the Hugging Face mirror (ykumards/open-i) of
the Kaggle Open-I dataset because the original 14 GB Kaggle archive is
impractical; the mirror stores the same images inside parquet shards.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"
ITEMS = {
    "symptom": [KAGGLE + "muherezaalouzious/disease-symptom-prediction-dataset"],
    "symptom_desc": [KAGGLE + "itachi9604/disease-symptom-description-dataset"],
    "diabetes": [KAGGLE + "iammustafatz/diabetes-prediction-dataset"],
    "openi_meta": [KAGGLE + "raddar/chest-xrays-indiana-university/indiana_reports.csv"],
}
DDX_META = "https://huggingface.co/datasets/aai530-group6/ddxplus/resolve/main/"
HF_OPENI = "https://huggingface.co/datasets/ykumards/open-i"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("  exists", dest.name)
        return
    print("  downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "medsupport-downloader"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()

    for folder, urls in ITEMS.items():
        print(folder)
        for u in urls:
            z = RAW / folder / "_zip" / (u.rsplit("/", 1)[-1] + ".zip")
            fetch(u, z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(RAW / folder)

    print("ddxplus_meta")
    for n in ("release_evidences.json", "release_conditions.json"):
        fetch(DDX_META + n, RAW / "ddxplus_meta" / n)

    if a.no_images:
        return
    print("openi images (parquet shards)")
    with urllib.request.urlopen(HF_OPENI.replace("huggingface.co/datasets", "huggingface.co/api/datasets") + "/tree/main/data") as r:
        files = json.load(r)
    for f in files:
        fetch(f"{HF_OPENI}/resolve/main/{f['path']}", RAW / "openi" / Path(f["path"]).name)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
