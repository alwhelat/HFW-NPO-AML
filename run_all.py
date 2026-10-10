"""
Run the complete HFW-NPO-AML study.

    python run_all.py                  # full experiments (hours)
    python run_all.py --quick          # small settings, end-to-end check (minutes)
    python run_all.py --only prognosis --jobs 4
    python run_all.py --skip-prepare   # reuse data/processed
    python run_all.py --quick --tag v2 # write to results_quick_v2/ and figures_quick_v2/

Steps: 1 prepare data | 2 diagnosis | 3 prognosis | 4 chemoresistance
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

STEPS = {
    "prepare": "01_prepare_data.py",
    "diagnosis": "02_run_diagnosis.py",
    "prognosis": "03_run_prognosis.py",
    "chemoresistance": "04_run_chemoresistance.py",
}


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script[:-3], ROOT / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small settings for a fast check")
    ap.add_argument("--only", default="", help="comma list: prepare,diagnosis,prognosis,chemoresistance")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--jobs", type=int, default=None, help="parallel CV folds")
    ap.add_argument("--methods", default="", help="comma list of method names to run")
    ap.add_argument("--tag", default="", help="suffix for separate output folders, "
                    "e.g. --tag v2 -> results_v2/ and figures_v2/")
    args = ap.parse_args()

    if args.quick:
        config.apply_quick_mode()
        config.RESULTS_DIR = config.PROJECT_ROOT / "results_quick"
        config.FIGURES_DIR = config.PROJECT_ROOT / "figures_quick"
    if args.tag:
        config.RESULTS_DIR = config.RESULTS_DIR.with_name(f"{config.RESULTS_DIR.name}_{args.tag}")
        config.FIGURES_DIR = config.FIGURES_DIR.with_name(f"{config.FIGURES_DIR.name}_{args.tag}")
    config.LOG_FILE = config.RESULTS_DIR / "progress.log"
    if args.jobs:
        config.N_JOBS = args.jobs
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or None

    steps = [s.strip() for s in args.only.split(",") if s.strip()] or list(STEPS)
    if args.skip_prepare and "prepare" in steps:
        steps.remove("prepare")

    from hfwnpo_aml.progress import log
    log(f"run started: steps={steps} quick={args.quick} jobs={config.N_JOBS} "
        f"log file={config.LOG_FILE}")
    t0 = time.time()
    for step in steps:
        print(f"\n{'=' * 78}\n  STEP: {step}\n{'=' * 78}")
        log(f"STEP {step} started")
        mod = _load(STEPS[step])
        if step == "prepare":
            mod.main()
        else:
            mod.main(methods)
    log(f"run finished in {(time.time() - t0) / 60:.1f} min")
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. "
          f"Results: {config.RESULTS_DIR} | Figures: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
