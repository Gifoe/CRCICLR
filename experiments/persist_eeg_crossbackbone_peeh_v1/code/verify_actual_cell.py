"""One-cell server-side equivalence check for the repaired PEEH ridge path.

This diagnostic is read-only: it loads a frozen checkpoint and cached data,
constructs the real representation/geometry, and compares fast scores with the
slow locked-protocol reference.  It never writes a runtime cell.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import run_crossbackbone_peeh as peeh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=peeh.MODELS, default="EEGNet")
    parser.add_argument("--task", choices=peeh.TASKS, default="OpenBMI_MI")
    parser.add_argument("--fold", type=int, choices=peeh.FOLDS, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row = next(
        x for x in peeh.all_checkpoint_rows()
        if x["Model"] == args.model and x["Task"] == args.task and x["fold"] == args.fold
    )
    data = peeh.load_fold_arrays(args.task, args.fold)
    x, y, sub, sess, _, _, _ = peeh.prepare_capped(data, args.task, args.fold)
    net, head = peeh.build_model(row, device)
    h = peeh.representations(net, head, x, args.model, device)
    spec = peeh.spectrum(h, y, sub, sess, args.task, args.model, args.fold)
    spec["classes"] = data["classes"]

    subjects = peeh.natural_subjects(sub)
    vals = np.asarray(subjects, dtype=object)
    vals = vals[np.random.default_rng(
        peeh.stable_seed("half", args.model, args.task, args.fold, 0)
    ).permutation(len(vals))]
    n = max(1, min(len(vals) - 1, len(vals) // 2))
    fit, eva = set(vals[:n]), set(vals[n:])
    fi = np.flatnonzero(np.isin(sub, list(fit)))
    ei = np.flatnonzero(np.isin(sub, list(eva)))

    started = time.perf_counter()
    engine = peeh.FixedStandardizerKernelRidge(
        h[fi], y[fi], h[ei], data["classes"], spec
    )
    init_seconds = time.perf_counter() - started
    checks = [()] + [tuple(b) for b in spec["blocks"][:3]]
    rng = np.random.default_rng(peeh.stable_seed("actual-equivalence", args.model, args.task, args.fold))
    checks.extend(tuple(rng.choice(spec["rank"], min(4, spec["rank"]), replace=False)) for _ in range(3))
    rows = []
    for dims in checks:
        pred, scores = engine.predict(dims)
        ref_pred, ref_scores = peeh.ridge_fixed_direct(
            h[fi], y[fi], h[ei], data["classes"], spec, dims,
            (engine.mu, engine.sd),
        )
        err = float(np.max(np.abs(scores - ref_scores)))
        mismatch = int(np.count_nonzero(pred != ref_pred))
        if not np.allclose(scores, ref_scores, rtol=3e-4, atol=3e-5) or mismatch:
            raise AssertionError({"dims": list(map(int, dims)), "max_abs_score_error": err,
                                  "prediction_mismatches": mismatch})
        rows.append({"dims": list(map(int, dims)), "max_abs_score_error": err,
                     "prediction_mismatches": mismatch})
    print(json.dumps({
        "status": "PASS", "model": args.model, "task": args.task,
        "fold": args.fold, "representation_dim": int(h.shape[1]),
        "fit_rows": len(fi), "evaluation_rows": len(ei), "engine_mode": engine.mode,
        "engine_init_seconds": init_seconds, "checks": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
