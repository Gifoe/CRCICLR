from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "persist_eeg_p2p3"
P3 = SOURCE / "p3"
OUT = ROOT / "outputs" / "persist_eeg_p3closure_p4" / "p3_closure"
TASKS = ("mi", "erp", "ssvep")
ROLES = ("epoch0", "epoch1", "early", "middle", "late", "final", "best")
ERASURE_ROLES = {"epoch0", "early", "middle", "best"}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_rank(spectrum: list[float] | np.ndarray) -> float:
    values = np.asarray(spectrum, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return float("nan")
    probabilities = values / values.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def load_spectra() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = sorted(P3.glob("seed_*/fold-*/analysis/epoch-*/COMPLETE.json"))
    if len(complete) != 171:
        raise RuntimeError(f"P3 closure requires 171 analyses, found {len(complete)}")
    for marker in complete:
        profile = json.loads((marker.parent / "rank_profile.json").read_text(encoding="utf-8"))
        for role in profile["roles"]:
            rows.append(
                {
                    "seed": int(profile["seed"]),
                    "fold": int(profile["fold"]),
                    "epoch": int(profile["epoch"]),
                    "role": role,
                    "U_L_spectrum": json.dumps(profile["U_L_spectrum"]),
                    "U_L_effective_rank": effective_rank(profile["U_L_spectrum"]),
                    "U_M_spectrum": json.dumps(profile["U_M_spectrum"]),
                    "U_M_effective_rank": effective_rank(profile["U_M_spectrum"]),
                    "U_L_orthonormality_error": float(profile["orthonormality_U_L"]),
                    "U_M_orthonormality_error": float(profile["orthonormality_U_M"]),
                    "finite": bool(profile["finite"]),
                    "test_subjects_used_for_fit": bool(profile["test_subjects_used_for_fit"]),
                }
            )
    return pd.DataFrame(rows)


def complete_trajectory() -> pd.DataFrame:
    event = pd.read_csv(P3 / "trajectory_metrics.csv")
    verification = pd.read_csv(P3 / "trajectory_verification.csv")
    rank = pd.read_csv(P3 / "trajectory_rank_profile.csv")
    spectra = load_spectra()

    keys = ["seed", "fold", "epoch", "role"]
    raw = (
        event[event.condition == "raw"]
        .set_index(keys + ["task"])["balanced_accuracy"]
        .rename("task_BA")
        .reset_index()
    )
    erased = (
        event[event.condition == "erase_UL"]
        .set_index(keys + ["task"])["balanced_accuracy"]
        .rename("erase_UL_BA")
        .reset_index()
    )
    long = verification[
        (verification.kind == "cross_session")
        & (verification.source == verification.target)
    ][keys + ["source", "auroc"]].rename(columns={"source": "task", "auroc": "long_AUROC"})
    frame = raw.merge(erased, how="left", on=keys + ["task"])
    frame["erase_UL_minus_raw_BA"] = frame.erase_UL_BA - frame.task_BA
    frame = frame.merge(long, how="left", on=keys + ["task"])
    frame = frame.merge(rank, how="left", on=keys)
    frame = frame.merge(spectra, how="left", on=keys)
    selected_epochs = (
        rank.assign(_prelocked=rank.role.isin(ERASURE_ROLES))
        .groupby(["seed", "fold", "epoch"], as_index=False)._prelocked.max()
        .rename(columns={"_prelocked": "erasure_computed_for_checkpoint"})
    )
    frame = frame.merge(selected_epochs, how="left", on=["seed", "fold", "epoch"])
    frame["erasure_prelocked_for_role"] = frame.role.isin(ERASURE_ROLES)
    frame["missing_erasure_reason"] = np.where(
        frame.erase_UL_BA.isna(),
        "checkpoint_epoch_not_selected_for_prelocked_erasure",
        "",
    )
    frame = frame.sort_values(["seed", "fold", "role_order", "task"]).reset_index(drop=True)
    expected = 5 * 5 * len(ROLES) * len(TASKS)
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} trajectory rows, found {len(frame)}")
    if frame[frame.erasure_computed_for_checkpoint].erase_UL_BA.isna().any():
        raise RuntimeError("Missing a prelocked erasure result")
    if frame[~frame.erasure_computed_for_checkpoint].erase_UL_BA.notna().any():
        raise RuntimeError("Unexpected erasure result in a non-prelocked role")
    if frame.test_subjects_used_for_fit.any():
        raise RuntimeError("Detected test subject use while fitting a trajectory subspace")
    return frame


def paired_seed_emergence(frame: pd.DataFrame) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed in range(5):
        block = frame[frame.seed == seed]
        for task in TASKS:
            task_block = block[block.task == task]
            means = task_block.groupby("role").agg(
                task_BA=("task_BA", "mean"),
                long_AUROC=("long_AUROC", "mean"),
                utility=("erase_UL_minus_raw_BA", "mean"),
            )
            records.append(
                {
                    "seed": seed,
                    "task": task,
                    "epoch0": clean(means.loc["epoch0"].to_dict()),
                    "best": clean(means.loc["best"].to_dict()),
                    "delta_task_BA": float(means.loc["best", "task_BA"] - means.loc["epoch0", "task_BA"]),
                    "delta_long_AUROC": float(means.loc["best", "long_AUROC"] - means.loc["epoch0", "long_AUROC"]),
                    "delta_utility": float(means.loc["best", "utility"] - means.loc["epoch0", "utility"]),
                }
            )
    return {
        "definition": "fold means within each seed; deltas are best minus epoch0",
        "test_data_used_for_adaptation": False,
        "records": records,
    }


def relation_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    usable = frame[frame.role.isin(ERASURE_ROLES)].copy()
    run = (
        usable.groupby(["seed", "role_order", "role", "task"], as_index=False)
        .agg(long_AUROC=("long_AUROC", "mean"), utility=("erase_UL_minus_raw_BA", "mean"))
    )
    within: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []
    for task in TASKS:
        task_run = run[run.task == task].copy()
        for seed, group in task_run.groupby("seed"):
            rho, p = spearmanr(group.long_AUROC, group.utility)
            within.append(
                {
                    "seed": int(seed),
                    "task": task,
                    "n_checkpoint_roles": int(len(group)),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                }
            )
        task_run["long_seed_demeaned"] = task_run.long_AUROC - task_run.groupby("seed").long_AUROC.transform("mean")
        task_run["utility_seed_demeaned"] = task_run.utility - task_run.groupby("seed").utility.transform("mean")
        rho, p = spearmanr(task_run.long_seed_demeaned, task_run.utility_seed_demeaned)
        pooled.append(
            {
                "task": task,
                "n_seed_role_points": int(len(task_run)),
                "spearman_rho": float(rho),
                "p_value": float(p),
            }
        )
    return {
        "status": "EXPLORATORY_ONLY",
        "unit": "seed x prelocked checkpoint role after fold averaging",
        "roles": sorted(ERASURE_ROLES),
        "within_seed": within,
        "seed_demeaned_pooled": pooled,
        "warning": "Four checkpoint roles per seed provide low power; this is not confirmatory evidence.",
    }


def covariance(values: np.ndarray, indices: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(values.shape[1], dtype=np.float64)
    n = 0
    for start in range(0, len(indices), batch_size):
        block = np.asarray(values[indices[start : start + batch_size]], dtype=np.float64)
        total += block.sum(axis=0)
        n += len(block)
    mean = total / n
    scatter = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        block = np.asarray(values[indices[start : start + batch_size]], dtype=np.float64) - mean
        scatter += block.T @ block
    return mean, scatter / max(n - 1, 1)


def geometry() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in range(5):
        for fold in range(5):
            base = P3 / f"seed_{seed}" / f"fold-{fold}"
            selection = json.loads((base / "CHECKPOINT_SELECTION.json").read_text(encoding="utf-8"))
            epoch = int(selection["best_epoch"])
            analysis = base / "analysis" / f"epoch-{epoch:03d}"
            embedding_path = base / "embeddings" / f"epoch-{epoch:03d}" / "embeddings.npy"
            metadata_path = base / "embeddings" / f"epoch-{epoch:03d}" / "metadata.parquet"
            audit_path = SOURCE / "p2" / f"seed_{seed}" / f"fold-{fold}" / "fold_local_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("test_subjects_used_for_fit", False):
                raise RuntimeError(f"Invalid P2 fold-local audit: seed {seed} fold {fold}")
            metadata = pd.read_parquet(metadata_path)
            train_subjects = set(map(str, audit["outer_train_subjects"]))
            train_indices = np.flatnonzero(metadata.subject_id.astype(str).isin(train_subjects).to_numpy())
            test_indices = np.flatnonzero(metadata.subject_id.astype(str).isin(set(map(str, audit["test_subjects"]))).to_numpy())
            if np.intersect1d(train_indices, test_indices).size:
                raise RuntimeError("Train/test index overlap in geometry analysis")
            values = np.load(embedding_path, mmap_mode="r")
            if len(metadata) != len(values):
                raise RuntimeError("Embedding/metadata length mismatch")
            _, cov = covariance(values, train_indices)
            eigval, eigvec = np.linalg.eigh(cov)
            order = np.argsort(eigval)[::-1]
            eigval, eigvec = eigval[order], eigvec[:, order]
            spaces = np.load(analysis / "subspaces.npz")
            ul = np.asarray(spaces["U_L"], dtype=np.float64)
            rank = int(ul.shape[1])
            upca = eigvec[:, :rank]
            singular = np.linalg.svd(ul.T @ upca, compute_uv=False)
            singular = np.clip(singular, 0.0, 1.0)
            angles = np.degrees(np.arccos(singular))
            overlap = float(np.square(singular).sum() / max(rank, 1))
            total_variance = float(np.trace(cov))
            ul_variance = float(np.trace(ul.T @ cov @ ul) / total_variance)
            pca_variance = float(eigval[:rank].sum() / total_variance)
            rng = np.random.default_rng(91_000_000 + seed * 1000 + fold)
            random_variance: list[float] = []
            for _ in range(100):
                q, _ = np.linalg.qr(rng.normal(size=(cov.shape[0], rank)), mode="reduced")
                random_variance.append(float(np.trace(q.T @ cov @ q) / total_variance))
            random_array = np.asarray(random_variance)
            runs.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": epoch,
                    "rank": rank,
                    "outer_train_subjects": sorted(train_subjects, key=int),
                    "test_subject_count": int(len(audit["test_subjects"])),
                    "test_subjects_used_for_fit": False,
                    "embedding_source": str(embedding_path.relative_to(ROOT)).replace("\\", "/"),
                    "normalized_overlap_UL_PCA": overlap,
                    "principal_angles_degrees": angles.tolist(),
                    "UL_captured_variance_ratio": ul_variance,
                    "PCA_captured_variance_ratio": pca_variance,
                    "random_same_rank_captured_variance": {
                        "draws": 100,
                        "mean": float(random_array.mean()),
                        "std": float(random_array.std(ddof=1)),
                        "q2_5": float(np.quantile(random_array, 0.025)),
                        "q50": float(np.quantile(random_array, 0.5)),
                        "q97_5": float(np.quantile(random_array, 0.975)),
                        "UL_percentile": float(100.0 * np.mean(random_array <= ul_variance)),
                    },
                    "UL_orthonormality_error": float(np.max(np.abs(ul.T @ ul - np.eye(rank)))),
                    "PCA_orthonormality_error": float(np.max(np.abs(upca.T @ upca - np.eye(rank)))),
                }
            )
            print(f"[geometry] seed={seed} fold={fold} epoch={epoch} rank={rank}", flush=True)
    numeric = [
        "normalized_overlap_UL_PCA",
        "UL_captured_variance_ratio",
        "PCA_captured_variance_ratio",
        "UL_orthonormality_error",
        "PCA_orthonormality_error",
    ]
    summary = {
        key: {
            "mean": float(np.mean([run[key] for run in runs])),
            "std": float(np.std([run[key] for run in runs], ddof=1)),
            "min": float(np.min([run[key] for run in runs])),
            "max": float(np.max([run[key] for run in runs])),
        }
        for key in numeric
    }
    summary["mean_principal_angles_degrees"] = np.mean(
        [np.pad(run["principal_angles_degrees"], (0, 16 - len(run["principal_angles_degrees"])), constant_values=np.nan) for run in runs],
        axis=0,
    )[np.isfinite(np.mean(
        [np.pad(run["principal_angles_degrees"], (0, 16 - len(run["principal_angles_degrees"])), constant_values=np.nan) for run in runs],
        axis=0,
    ))].tolist()
    summary["random_same_rank_mean_captured_variance"] = float(
        np.mean([run["random_same_rank_captured_variance"]["mean"] for run in runs])
    )
    return {
        "analysis_population": "P3 best checkpoint, each seed/fold independently",
        "fit_population": "outer-train subjects only",
        "covariance_definition": "global centered covariance of raw best-checkpoint embeddings on outer-train subjects",
        "pca_definition": "top eigenvectors of the same fold-local covariance; rank matched to U_L",
        "random_control": "100 Haar-like QR bases per seed/fold at matched rank",
        "test_subjects_used_for_fit": False,
        "runs": runs,
        "summary": summary,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p2_report = json.loads((SOURCE / "P2_FINAL_REPORT.json").read_text(encoding="utf-8"))
    p3_old = json.loads((SOURCE / "P3_FINAL_REPORT.json").read_text(encoding="utf-8"))
    if p2_report["status"] != "P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY":
        raise RuntimeError("P2 is not in the required pass state")
    if p3_old["status"] != "P3_TRAJECTORY_CLAIM_NOT_SUPPORTED":
        raise RuntimeError("Existing P3 decision differs from the audited expected state")

    frame = complete_trajectory()
    csv_path = OUT / "P3_TRAJECTORY_COMPLETE.csv"
    frame.to_csv(csv_path, index=False)
    write_json(
        OUT / "P3_TRAJECTORY_COMPLETE.json",
        {
            "rows": int(len(frame)),
            "seeds": sorted(frame.seed.unique().tolist()),
            "folds": sorted(frame.fold.unique().tolist()),
            "roles": list(ROLES),
            "tasks": list(TASKS),
            "erasure_roles": sorted(ERASURE_ROLES),
            "non_erasure_roles": {
                "roles": ["epoch1", "late", "final"],
                "policy": "missing unless its physical epoch aliases a prelocked erasure role",
            },
            "records": frame.to_dict(orient="records"),
        },
    )
    emergence = paired_seed_emergence(frame)
    emergence["persistence_strength_vs_utility"] = relation_analysis(frame)
    write_json(OUT / "P3_EMERGENCE_ANALYSIS.json", emergence)
    geom = geometry()
    write_json(OUT / "P3_UL_PCA_GEOMETRY.json", geom)

    report = {
        "status": "P3_CLOSED_AND_FROZEN",
        "p2_status": p2_report["status"],
        "selective_compression": "NOT_SUPPORTED",
        "long_medium_independence": "NOT_SUPPORTED",
        "MEDIUM_AS_INDEPENDENT_CORE_SCALE": "NOT_SUPPORTED",
        "task_relevant_persistence_emergence": "EXPLORATORY_SUPPORT",
        "formal_p4_representation": "PERSISTENT_PLUS_COMPLEMENTARY_FAST",
        "forbidden_formalization": "zL_plus_zM_plus_zF",
        "p3_checkpoint_runs": int(p3_old["checkpoint_runs"]),
        "headline": {
            "MI_epoch0_to_best_BA": p3_old["statistics"]["mi_event_epoch0_to_best"],
            "SSVEP_epoch0_to_best_BA": p3_old["statistics"]["ssvep_event_epoch0_to_best"],
            "Long_epoch0_to_best_AUROC": p3_old["statistics"]["long_epoch0_to_best"],
            "best_Long_AUROC": p3_old["best_cross_session_long_auroc"],
            "mean_UL_rank": p3_old["mean_UL_rank"],
            "compression_negative_seeds": p3_old["compression_negative_seeds"],
            "UL_PCA_geometry_summary": geom["summary"],
        },
        "interpretation": [
            "Long persistence increased rather than compressed during training; the selective-compression hypothesis failed.",
            "Medium persistence is weak and geometrically redundant with Long persistence, so it is not an independent core scale.",
            "The P4 implication is a two-part decomposition: an explicitly persistent low-rank subspace plus its orthogonal complementary/fast component.",
            "Trajectory relations are exploratory and may not be used as confirmatory evidence or to tune P4 on outer-test subjects.",
        ],
        "test_driven_adaptation": False,
    }
    write_json(OUT / "P3_FINAL_REPORT_V2.json", report)
    markdown = f"""# PERSIST-EEG P3 Closure Report V2

Decision: `P3_CLOSED_AND_FROZEN`

## Required conclusions

- P2: `{report['p2_status']}`
- Selective compression: `NOT_SUPPORTED`
- Long/Medium independence: `NOT_SUPPORTED`
- `MEDIUM_AS_INDEPENDENT_CORE_SCALE = NOT_SUPPORTED`
- Task-relevant persistence emergence: `EXPLORATORY_SUPPORT`
- P4 formalization: Persistent + Complementary/Fast; do not use `zL + zM + zF`.

## Evidence

- MI epoch0 to best BA change: {p3_old['statistics']['mi_event_epoch0_to_best']['mean']:.6f}, 95% CI {p3_old['statistics']['mi_event_epoch0_to_best']['ci95']}.
- SSVEP epoch0 to best BA change: {p3_old['statistics']['ssvep_event_epoch0_to_best']['mean']:.6f}, 95% CI {p3_old['statistics']['ssvep_event_epoch0_to_best']['ci95']}.
- Cross-session Long AUROC epoch0 to best change: {p3_old['statistics']['long_epoch0_to_best']['mean']:.6f}, 95% CI {p3_old['statistics']['long_epoch0_to_best']['ci95']}.
- Seeds showing the preregistered compression direction: {p3_old['compression_negative_seeds']}/5.
- Best-checkpoint Long AUROC: {p3_old['best_cross_session_long_auroc']:.6f}.
- Mean U_L rank: epoch0 {p3_old['mean_UL_rank']['epoch0']:.3f}, best {p3_old['mean_UL_rank']['best']:.3f}.
- Mean normalized U_L/PCA overlap: {geom['summary']['normalized_overlap_UL_PCA']['mean']:.6f}.
- Mean variance captured: U_L {geom['summary']['UL_captured_variance_ratio']['mean']:.6f}; PCA {geom['summary']['PCA_captured_variance_ratio']['mean']:.6f}; random same-rank {geom['summary']['random_same_rank_mean_captured_variance']:.6f}.

The trajectory table contains all 5 seeds, 5 folds, 7 locked checkpoint roles, and 3 tasks. U_L erasure was prelocked only for epoch0/early/middle/best. Epoch1/late/final are missing unless their physical checkpoint aliases a prelocked role; no missing intervention was fabricated.

No outer-test data were used to fit U_L, PCA, random controls, or to adapt the P4 method.
"""
    (OUT / "P3_FINAL_REPORT_V2.md").write_text(markdown, encoding="utf-8")
    file_hashes = {
        path.name: sha256(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "P3_FROZEN.json"
    }
    write_json(
        OUT / "P3_FROZEN.json",
        {
            "status": "P3_FROZEN",
            "source_status": p3_old["status"],
            "files_sha256": file_hashes,
            "modification_after_freeze_prohibited": True,
        },
    )
    print(json.dumps({"status": report["status"], "files": len(file_hashes)}, indent=2))


if __name__ == "__main__":
    main()
