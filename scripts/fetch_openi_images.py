"""Fetch the Open-I (Indiana University) chest X-rays directly from the NLM
Open-I service at 512 px - the official source, and ~50x smaller than the raw
2048 px Kaggle archive.

    python scripts/fetch_openi_images.py              # frontal images only (~3.8k)
    python scripts/fetch_openi_images.py --all        # frontal + lateral (7,470)

Output: data/raw/openi_nlm/listing.json  (uid, image paths, MeSH, Problems)
        data/raw/openi_nlm/images/<filename>.png

Frontal/lateral is decided with the Kaggle `indiana_projections.csv`
(same filenames without the "CXR" prefix / ".dcm" suffix).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = RAW / "openi_nlm"
IMG = OUT / "images"
API = "https://openi.nlm.nih.gov/api/search?coll=cxr&it=xg&m={m}&n={n}"
BASE = "https://openi.nlm.nih.gov"
HEADERS = {"User-Agent": "medsupport-research-downloader"}


def get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"failed {url}: {last}")


def listing(page: int) -> list[dict]:
    path = OUT / "listing.json"
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    # first call tells us the total
    first = json.loads(get(API.format(m=1, n=page)))
    total = int(first["total"])
    rows = list(first["list"])
    m = page + 1
    while m <= total:
        n = min(m + page - 1, total)
        try:
            d = json.loads(get(API.format(m=m, n=n)))
        except Exception as e:  # noqa: BLE001
            print("  page", m, n, "failed:", e, "- retrying with smaller page")
            page = max(10, page // 2)
            continue
        rows.extend(d["list"])
        print(f"  listed {len(rows)}/{total}", end="\r")
        m = n + 1
    print()
    keep = [{"uid": r["uid"], "pmcid": r.get("pmcid"), "imgLarge": r["imgLarge"], "MeSH": r.get("MeSH"),
             "Problems": r.get("Problems")} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(keep, open(path, "w", encoding="utf-8"), indent=0)
    return keep


def kaggle_name(img_large: str) -> str:
    """'/imgs/512/198/1000/CXR1000_IM-0003-1001.png' -> '1000_IM-0003-1001.dcm.png'."""
    fn = img_large.rsplit("/", 1)[-1]
    return fn.replace("CXR", "", 1).replace(".png", ".dcm.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    rows = listing(a.page)
    print("images listed:", len(rows))
    proj = pd.read_csv(RAW / "openi_meta" / "indiana_projections.csv")
    frontal = set(proj.loc[proj["projection"] == "Frontal", "filename"])
    todo = [r for r in rows if a.all or kaggle_name(r["imgLarge"]) in frontal]
    print("to download:", len(todo))
    IMG.mkdir(parents=True, exist_ok=True)

    def work(r):
        dest = IMG / r["imgLarge"].rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size > 1000:
            return dest.name, "exists"
        data = get(BASE + r["imgLarge"])
        if not data.startswith(b"\x89PNG"):
            return dest.name, "not-png"
        dest.write_bytes(data)
        return dest.name, "ok"

    t0 = time.time()
    counts = {"ok": 0, "exists": 0, "not-png": 0, "error": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                _, status = f.result()
            except Exception as e:  # noqa: BLE001
                status = "error"
                print("  !!", e)
            counts[status] += 1
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} {counts} {time.time() - t0:.0f}s", end="\r")
    print("\ndone", counts)


if __name__ == "__main__":
    sys.exit(main())
