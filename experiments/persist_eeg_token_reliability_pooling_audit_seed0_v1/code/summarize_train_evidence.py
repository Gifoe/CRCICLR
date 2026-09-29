"""Compact TRAIN-only token evidence with strict upstream hash validation.

This is descriptive analysis, not pooling selection or OUTER evaluation.
Outputs are created in a new runtime directory and never overwrite a result.
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


def load_checked(runtime: Path, fold: int, stage: str, seal_name: str,
                 status: str) -> tuple[Path, dict, str]:
    directory = runtime / f"fold{fold}" / stage
    seal_path = directory / seal_name
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal.get("fold"), seal.get("status"), seal.get("formal_final_heldout_eeg_reads")) != (fold, status, 0):
        raise RuntimeError(f"fold {fold}: invalid {stage} seal")
    for name, expected in seal["files"].items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise RuntimeError(f"fold {fold}: invalid {stage} file/hash: {name}")
    extraction = runtime / f"fold{fold}" / "train" / "EXTRACTION_PROVENANCE.json"
    if sha(extraction) != seal["source_extraction_sha256"]:
        raise RuntimeError(f"fold {fold}: {stage} does not match TRAIN extraction")
    return directory, seal, sha(seal_path)


def rank_set(score: np.ndarray, fraction: float) -> set[int]:
    count = max(1, int(np.ceil(fraction * len(score))))
    return set(np.argsort(-score, kind="stable")[:count].tolist())


def build(runtime: Path, output: Path) -> dict:
    reviews = runtime / "train_gate_review_v1" / "TRAIN_GATE_A_B_REVIEW.json"
    review = json.loads(reviews.read_text(encoding="utf-8"))
    if review.get("status") != "COMPLETE_TRAIN_ONLY_GATE_REVIEW" or review.get("outer_development_eeg_reads") != 0:
        raise RuntimeError("invalid TRAIN gate review")
    if review["gate_A_pass"] or review["gate_B_numeric_pass"]:
        raise RuntimeError("this early-negative TRAIN summary is inapplicable if Gates A/B pass")
    all_reliability, all_null, all_halves, all_utility, all_groups, all_structure = [], [], [], [], [], []
    fold_provenance, score_vectors, joint_sets = [], [], []
    for fold in range(5):
        rdir, rseal, rsha = load_checked(runtime, fold, "reliability_v1",
                                         "TRAIN_RELIABILITY_SEAL.json", "COMPLETE_TRAIN_RELIABILITY_ONLY")
        udir, useal, usha = load_checked(runtime, fold, "utility_v1",
                                         "TRAIN_UTILITY_SEAL.json", "COMPLETE_TRAIN_UTILITY")
        if rseal["source_extraction_sha256"] != useal["source_extraction_sha256"]:
            raise RuntimeError(f"fold {fold}: reliability and utility use different TRAIN tokens")
        reliability = pd.read_csv(rdir / "TOKEN_RELIABILITY.csv").sort_values("token_index")
        utility = pd.read_csv(udir / "TOKEN_UTILITY.csv").sort_values("token_index")
        if (len(reliability), len(utility), set(reliability.token_index), set(utility.token_index)) != (
            248, 248, set(range(248)), set(range(248))
        ):
            raise RuntimeError(f"fold {fold}: token index inventory mismatch")
        if (reliability.fold.nunique(), utility.fold.nunique(), int(reliability.fold.iloc[0]),
            int(utility.fold.iloc[0])) != (1, 1, fold, fold):
            raise RuntimeError(f"fold {fold}: mixed fold identities")
        for name, frame, expected_count in (
            ("null", pd.read_csv(rdir / "TOKEN_RELIABILITY_NULL.csv"), 1000),
            ("halves", pd.read_csv(rdir / "TOKEN_RELIABILITY_REPRODUCIBILITY.csv"), 200),
            ("structure", pd.read_csv(rdir / "TOKEN_STRUCTURE_DECOMPOSITION.csv"), 1),
        ):
            if len(frame) != expected_count or set(frame.fold) != {fold}:
                raise RuntimeError(f"fold {fold}: {name} inventory mismatch")
            {"null": all_null, "halves": all_halves, "structure": all_structure}[name].append(frame)
        r = reliability.R_cos.to_numpy(dtype=np.float64)
        u = utility.U_probe.to_numpy(dtype=np.float64)
        if not np.isfinite(r).all() or not np.isfinite(u).all() or r.std() <= 0 or u.std() <= 0:
            raise RuntimeError(f"fold {fold}: nonfinite/constant TRAIN token score")
        r_hi, u_hi = r >= np.median(r), u >= np.median(u)
        labels = np.select(
            (r_hi & u_hi, r_hi & ~u_hi, ~r_hi & u_hi, ~r_hi & ~u_hi),
            ("RELIABLE_USEFUL", "RELIABLE_LOWUTILITY", "FRAGILE_USEFUL", "FRAGILE_LOWUTILITY"),
            default="UNASSIGNED",
        )
        if (labels == "UNASSIGNED").any():
            raise RuntimeError("TRAIN-only RU group assignment incomplete")
        joint = (r - r.mean()) / r.std() + (u - u.mean()) / u.std()
        group = pd.DataFrame({"fold": fold, "token_index": np.arange(248), "RU_group": labels,
                              "R_cos": r, "U_probe": u, "RU_joint_source_score": joint,
                              "R_threshold_train_median": float(np.median(r)),
                              "U_threshold_train_median": float(np.median(u))})
        all_groups.append(group)
        all_reliability.append(reliability)
        all_utility.append(utility)
        score_vectors.append(r)
        joint_sets.append(rank_set(joint, .25))
        top_r = rank_set(r, .25)
        bottom_r = rank_set(-r, .25)
        fold_provenance.append({
            "fold": fold, "train_subject_count": int(rseal["train_subject_count"]),
            "reliability_seal_sha256": rsha, "utility_seal_sha256": usha,
            "source_extraction_sha256": rseal["source_extraction_sha256"],
            "R_U_probe_spearman": float(spearmanr(r, u).statistic),
            "R_U_erase_spearman_descriptive_convergence_warning": float(
                spearmanr(r, utility.U_erase.to_numpy(dtype=np.float64)).statistic),
            "top_R_quartile_mean_U_probe": float(u[list(top_r)].mean()),
            "bottom_R_quartile_mean_U_probe": float(u[list(bottom_r)].mean()),
            "RU_group_counts": group.RU_group.value_counts().to_dict(),
        })
    cross = []
    for i in range(5):
        for j in range(i + 1, 5):
            ri, rj = rank_set(score_vectors[i], .25), rank_set(score_vectors[j], .25)
            si, sj = joint_sets[i], joint_sets[j]
            cross.append({"fold_a": i, "fold_b": j,
                          "R_spearman": float(spearmanr(score_vectors[i], score_vectors[j]).statistic),
                          "R_top_quartile_jaccard": len(ri & rj) / len(ri | rj),
                          "RU_top_quartile_jaccard_descriptive": len(si & sj) / len(si | sj)})

    output.mkdir(exist_ok=False)
    tables = {
        "TOKEN_RELIABILITY.csv": pd.concat(all_reliability, ignore_index=True),
        "TOKEN_RELIABILITY_NULL.csv": pd.concat(all_null, ignore_index=True),
        "TOKEN_RELIABILITY_REPRODUCIBILITY.csv": pd.concat(all_halves, ignore_index=True),
        "TOKEN_UTILITY.csv": pd.concat(all_utility, ignore_index=True),
        "TOKEN_RU_GROUPS.csv": pd.concat(all_groups, ignore_index=True),
        "TOKEN_STRUCTURE_DECOMPOSITION.csv": pd.concat(all_structure, ignore_index=True),
        "CROSSFOLD_TOKEN_STABILITY.csv": pd.DataFrame(cross),
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    utility_log = runtime / "utility_queue_v1.log"
    log_text = utility_log.read_text(encoding="utf-8", errors="replace")
    result = {
        "schema": "TOKEN_RELIABILITY_TRAIN_EVIDENCE_V1",
        "status": "COMPLETE_TRAIN_DESCRIPTIVE_NO_POOLING_SELECTION",
        "train_gate_review_sha256": sha(reviews),
        "gate_A_pass": False, "gate_B_numeric_pass": False,
        "folds": fold_provenance,
        "utility_erasure_convergence_warning_count": log_text.count("ConvergenceWarning"),
        "utility_erase_interpretation": "PROVISIONAL_HIGH_DIMENSIONAL_LIBLINEAR_CONVERGENCE_WARNING",
        "primary_utility_for_selection": "U_probe_only",
        "outer_development_eeg_reads": 0, "formal_final_heldout_eeg_reads": 0,
        "files": {name: sha(output / name) for name in tables},
    }
    (output / "TRAIN_EVIDENCE_PROVENANCE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    result = build(runtime, runtime / "train_evidence_v1")
    print(json.dumps({"status": result["status"], "folds": len(result["folds"]),
                      "files": len(result["files"]),
                      "provenance_sha256": sha(runtime / "train_evidence_v1" / "TRAIN_EVIDENCE_PROVENANCE.json")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
