"""Small, explicitly labelled GeoSR futility screen.

This helper is deliberately separate from the frozen decision runner.  It
does not alter :mod:`run_geosr` or any scientific constant; it only reuses the
already materialised deterministic teacher/selection caches and asks whether
there is any reason to spend the remaining compute.  The screen trains the
registered subject-balanced ERM and GeoSR methods on one outer fold.  It is
not a final claim and its output is never merged into the full decision tree.

The source cache is copied (never moved) into an isolated artifact root so a
futility run cannot overwrite a pending full-protocol run.  Cache metadata are
checked by the canonical runner's existing fingerprint checks.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

import run_geosr_decision as d


METHODS = ("SUBJECT_BALANCED_ERM", "GEOSR")


def copy_cache(source_root: Path, dest_root: Path) -> None:
    """Copy compact deterministic caches, preserving atomic cache files."""
    src = source_root / "runtime" / "seed-0" / "cache"
    dst = dest_root / "runtime" / "seed-0" / "cache"
    if not src.is_dir():
        raise FileNotFoundError(f"source cache directory missing: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        # Existing destination files are left intact; the canonical loader
        # will reject stale metadata and recompute them safely.
        if not out.exists():
            shutil.copy2(path, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # Configure the canonical runner in-process.  Only the scope is narrowed;
    # all model, split, optimizer, cross-fit and descriptor constants remain
    # imported unchanged from run_geosr_decision/run_geosr.
    d.DATASETS = ("OpenBMI",)
    d.FOLDS = (0,)
    d.METHODS = METHODS
    d.ROOT = args.root.resolve()
    d.RESULTS = d.ROOT / "results"
    d.RUNTIME = d.ROOT / "runtime"
    d.ROOT.mkdir(parents=True, exist_ok=True)
    copy_cache(args.source_root.resolve(), d.ROOT)
    print("FUTILITY_SCOPE OpenBMI fold=0 methods=SUBJECT_BALANCED_ERM,GEOSR", flush=True)
    print("FUTILITY_ONLY no final claim; outcome labels remain unopened until an explicit lock", flush=True)
    d.preflight(torch.device(args.device))


if __name__ == "__main__":
    main()
