"""Hash-checked, TRAIN-only review of the preregistered reliability gates.

This review deliberately does not open OUTER or select a pooling method.  A
failed premise is reported as a failure, not used to change the score or null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fold(runtime: Path, fold: int) -> tuple[dict, np.ndarray]:
    directory = runtime / f"fold{fold}" / "reliability_v1"
    seal_path = directory / "TRAIN_RELIABILITY_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("fold"), seal.get("status"), seal.get("formal_final_heldout_eeg_reads")) != (
        fold, "COMPLETE_TRAIN_RELIABILITY_ONLY", 0
    ):
        raise RuntimeError(f"fold {fold}: invalid TRAIN-only reliability seal")
    if set(seal["files"]) != {
        "TOKEN_RELIABILITY.csv", "TOKEN_RELIABILITY_NULL.csv",
        "TOKEN_RELIABILITY_REPRODUCIBILITY.csv", "TOKEN_STRUCTURE_DECOMPOSITION.csv"
    }:
        raise RuntimeError(f"fold {fold}: unexpected reliability file inventory")
    for name, expected in seal["files"].items():
        if sha(directory / name) != expected:
            raise RuntimeError(f"fold {fold}: {name} SHA mismatch")
    extraction_path = runtime / f"fold{fold}" / "train" / "EXTRACTION_PROVENANCE.json"
    if sha(extraction_path) != seal["source_extraction_sha256"]:
        raise RuntimeError(f"fold {fold}: source extraction provenance SHA mismatch")

    scores = pd.read_csv(directory / "TOKEN_RELIABILITY.csv")
    null = pd.read_csv(directory / "TOKEN_RELIABILITY_NULL.csv")
    halves = pd.read_csv(directory / "TOKEN_RELIABILITY_REPRODUCIBILITY.csv")
    if (len(scores), set(scores["token_index"]), len(null), len(halves)) != (
        248, set(range(248)), 1000, 200
    ):
        raise RuntimeError(f"fold {fold}: TRAIN reliability row inventory mismatch")
    if null.groupby("null").size().to_dict() != {
        "SESSION2_WITHIN_SUBJECT_LABEL": 500, "SESSION_SUBJECT_PAIRING": 500
    }:
        raise RuntimeError(f"fold {fold}: null type/draw inventory mismatch")
    if scores["fold"].nunique() != 1 or int(scores["fold"].iloc[0]) != fold:
        raise RuntimeError(f"fold {fold}: score fold identity mismatch")
    for frame in (scores, null, halves):
        numeric = frame.select_dtypes(include="number").to_numpy()
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"fold {fold}: nonfinite TRAIN metric")

    r = scores.sort_values("token_index")["R_cos"].to_numpy(dtype=np.float64)
    real = seal["real"]
    if not np.isclose(r.std(), real["std"], atol=1e-10):
        raise RuntimeError(f"fold {fold}: real spread does not match sealed scores")
    pairing = null.loc[null["null"] == "SESSION_SUBJECT_PAIRING"]
    labels = null.loc[null["null"] == "SESSION2_WITHIN_SUBJECT_LABEL"]
    pairing_std_p95 = float(np.quantile(pairing["std"], .95))
    paired_p = float((1 + (pairing["std"] >= real["std"]).sum()) / 501)
    label_p = float((1 + (labels["std"] >= real["std"]).sum()) / 501)
    half_spearman = float(halves["spearman"].mean())
    half_jaccard = float(halves["top_quartile_jaccard"].mean())
    row = {
        "fold": fold,
        "train_subject_count": int(seal["train_subject_count"]),
        "real_reliability_std": float(real["std"]),
        "real_top_bottom_quartile_gap": float(real["top_bottom_quartile_gap"]),
        "pairing_null_std_p95": pairing_std_p95,
        "pairing_null_spread_empirical_p": paired_p,
        "label_null_spread_empirical_p": label_p,
        "split_half_mean_spearman": half_spearman,
        "split_half_mean_top_quartile_jaccard": half_jaccard,
        "gate_A_fold": bool(real["top_bottom_quartile_gap"] >= .20 and real["std"] > pairing_std_p95),
        "reliability_seal_sha256": sha(seal_path),
    }
    return row, r


def audit(runtime: Path) -> dict:
    folds = [read_fold(runtime, fold) for fold in range(5)]
    rows = [row for row, _ in folds]
    ranks = []
    for i in range(5):
        for j in range(i + 1, 5):
            ranks.append({"fold_a": i, "fold_b": j,
                          "spearman_R": float(spearmanr(folds[i][1], folds[j][1]).statistic)})
    mean_spearman = float(np.mean([row["split_half_mean_spearman"] for row in rows]))
    mean_jaccard = float(np.mean([row["split_half_mean_top_quartile_jaccard"] for row in rows]))
    gate_a = sum(row["gate_A_fold"] for row in rows) >= 4
    gate_b = mean_spearman >= .50 and mean_jaccard >= .40
    return {
        "schema": "TOKEN_RELIABILITY_TRAIN_GATES_A_B_REVIEW_V1",
        "status": "COMPLETE_TRAIN_ONLY_GATE_REVIEW",
        "folds": rows,
        "crossfold_reliability_rank_spearman": ranks,
        "split_half_mean_spearman_across_folds": mean_spearman,
        "split_half_mean_top_quartile_jaccard_across_folds": mean_jaccard,
        "gate_A_pass": bool(gate_a),
        "gate_B_numeric_pass": bool(gate_b),
        "gate_B_crossfold_consistency": "DESCRIPTIVE_ONLY_NO_POST_HOC_THRESHOLD",
        "interpretation_if_stopped_now": (
            "NO_MEANINGFUL_TOKEN_RELIABILITY_HETEROGENEITY" if not gate_a
            else "TOKEN_RELIABILITY_NOT_REPRODUCIBLE" if not gate_b else "UNDETERMINED"
        ),
        "outer_development_eeg_reads": 0,
        "formal_final_heldout_eeg_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path(os.environ["TOKEN_AUDIT_RUNTIME"]))
    args = parser.parse_args()
    result = audit(args.runtime.resolve())
    output = args.runtime.resolve() / "train_gate_review_v1"
    output.mkdir(exist_ok=False)
    path = output / "TRAIN_GATE_A_B_REVIEW.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate_A_pass": result["gate_A_pass"],
                      "gate_B_numeric_pass": result["gate_B_numeric_pass"],
                      "sha256": sha(path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
