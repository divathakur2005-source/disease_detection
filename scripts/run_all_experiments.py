"""Run the full Open-I experiment matrix in one go, then compile the results.

    python scripts/run_all_experiments.py              # all four runs
    python scripts/run_all_experiments.py --quick      # fewer epochs, for a smoke test

The four runs share one patient split, so their metrics are directly
comparable:

    image        DenseNet-121 on the radiograph only            (baseline A)
    clinical     SymptomEncoder on the parsed symptoms only     (baseline B)
    fusion       both + bidirectional cross-modal attention     (proposed)
    fusion_nopre same, without DDXPlus pre-training             (ablation)

The ablation isolates how much the shared DDXPlus vocabulary actually
contributes, which is the project's central claim.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(args: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    env = {**__import__("os").environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-u", "-W", "ignore", *args], cwd=ROOT, env=env)
    ok = p.returncode == 0
    print(f"<<< {label}: {'OK' if ok else 'FAILED (exit %d)' % p.returncode} in {time.time() - t0:.0f}s", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    a = ap.parse_args()
    ep = "4" if a.quick else "20"

    if not a.skip_prepare:
        if not run(["-m", "medsupport.data.prepare_openi"], "prepare Open-I (labels, evidences, images)"):
            print("\nprepare_openi failed - have the images been downloaded? "
                  "Run: python scripts/fetch_openi_images.py")
            sys.exit(1)

    results = {}
    results["image"] = run(["-m", "medsupport.train.imaging", "image", "--epochs", ep], "baseline A: image only")
    results["clinical"] = run(["-m", "medsupport.train.imaging", "clinical", "--epochs", ep],
                              "baseline B: clinical only")
    results["fusion"] = run(["-m", "medsupport.train.imaging", "fusion", "--epochs", ep,
                             "--init-image", str(ROOT / "models" / "openi_image.pt")], "proposed: cross-modal fusion")
    results["fusion_nopre"] = run(["-m", "medsupport.train.imaging", "fusion", "--epochs", ep, "--no-pretrain",
                                   "--tag", "_nopretrain"], "ablation: fusion without DDXPlus pre-training")

    run(["scripts/analyze_correlation.py"], "symptom <-> X-ray correlation analysis")
    run(["scripts/compare_openi.py"], "comparison table + ROC curves")
    run(["scripts/make_results.py"], "compile reports/RESULTS.md")

    print("\n" + "=" * 70)
    for k, v in results.items():
        print(f"  {k:14s} {'OK' if v else 'FAILED'}")
    print("=" * 70)
    print("\nSee reports/RESULTS.md, reports/openi_comparison.md and reports/figures/")


if __name__ == "__main__":
    main()
