"""Seed-0 GeoSR decision run (two folds, three pre-registered methods).

This is a screening/continuation run, not the complete frozen experiment.  It
reuses the exact helpers, splits, cross-fitting, model, optimizer, and outcome
sealing of :mod:`run_geosr`, while writing into an isolated ``decision_seed0``
tree.  The fixed screening rule is written before outcome data are opened.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import audit_primitives as ap
import run_geosr as g


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "decision_seed0"
RESULTS = ROOT / "results"
RUNTIME = ROOT / "runtime"
DATASETS = ("OpenBMI", "WBCIC")
FOLDS = (0, 1)
METHODS = ("SUBJECT_BALANCED_ERM", "RANDOM_RANK", "GEOSR")
SEED = 0

# This rule is deliberately fixed before outcome labels are loaded.  It is a
# compute-continuation screen only; the full six-gate seed-0 decision remains
# defined by FROZEN_PROTOCOL.json and requires all five folds/methods.
SCREEN_RULE: dict[str, Any] = {
    "both_dataset_mean_delta_min_pp": 0.10,
    "one_dataset_mean_delta_min_pp": 0.30,
    "folds_required_nonnegative": 2,
    "nonnegative_subject_fraction_min": 0.60,
    "material_harm_fraction_max": 0.15,
    "bottom25_delta_min_pp": 0.0,
    "require_geosr_not_worse_than_random": True,
    "paired_bootstrap_upper_ci_min_pp": 0.0,
}


def jclean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jclean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jclean(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return jclean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(jclean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def role_hash(role: Mapping[str, list[str]]) -> str:
    return g.role_hash(role)


def protocol_lock_path() -> Path:
    return ROOT / "PRE_OUTCOME_GEOSR_DECISION_LOCK.json"


def write_screen_protocol() -> None:
    payload = {
        "schema": "PERSIST_EEG_GEOSR_DECISION_SCREEN_V1",
        "seed": SEED,
        "datasets": list(DATASETS),
        "folds": list(FOLDS),
        "methods": list(METHODS),
        "screening_only": True,
        "complete_protocol_required_for_final_claim": True,
        "scientific_definition_changed": False,
        "rule": SCREEN_RULE,
        "outcome_labels_read": False,
        "outer_sealed_access": False,
        "code_fingerprint": g.code_fingerprint(),
    }
    write_json(ROOT / "DECISION_SCREEN_PROTOCOL.json", payload)
    (ROOT / "DECISION_SCREEN_PROTOCOL.md").write_text(
        "# GeoSR seed-0 decision screen\n\n"
        "This file is written before any outcome labels are opened.  It only "
        "decides whether to spend compute on folds 2--4; it cannot support a "
        "final scientific claim.\n\n"
        "Fixed rule (both OpenBMI and WBCIC): mean paired GeoSR−"
        "SUBJECT_BALANCED_ERM >= 0.10 pp; both decision folds non-negative; "
        "at least 60% of pooled biological subjects non-negative; material "
        "harm <=15%; fixed bottom-25% delta >=0; GeoSR not worse than "
        "RANDOM_RANK; paired-bootstrap upper CI >0.  At least one dataset "
        "must have mean delta >=0.30 pp.\n",
        encoding="utf-8",
    )


def preflight() -> None:
    g.seed_everything(SEED)
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_screen_protocol()
    chosen_cap, support_lock = ap.choose_descriptor_cap()
    if int(chosen_cap) != g.CAP:
        raise RuntimeError(f"descriptor support audit failed: expected {g.CAP}, got {chosen_cap}")
    write_json(ROOT / "DATA_SUPPORT_LOCK.json", {**support_lock, "seed": SEED, "chosen_cap": g.CAP, "decision_run": True})
    roles_by_dataset: dict[str, list[dict[str, list[str]]]] = {}
    role_hashes: dict[str, list[str]] = {}
    for d in DATASETS:
        roles, _, _ = ap.load_roles(d)
        roles_by_dataset[d] = roles
        role_hashes[d] = [role_hash(r) for r in roles]
    write_json(ROOT / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_DECISION_DATA_LEGALITY_V1", "seed": SEED,
        "decision_run": True, "folds": list(FOLDS), "methods": list(METHODS),
        "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False,
        "outcome_labels_read_before_lock": False, "outcome_labels_read_after_lock": False,
        "role_hashes": role_hashes, "descriptor_cap": g.CAP,
    })
    all_crossfit: list[dict[str, Any]] = []
    all_teachers: list[dict[str, Any]] = []
    all_risk: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    state_hashes: dict[str, Any] = {}
    fold_manifests: dict[str, Any] = {}
    cache_root = RUNTIME / f"seed-{SEED}" / "cache"
    for dataset in DATASETS:
        for fold in FOLDS:
            role = roles_by_dataset[dataset][fold]
            source = g.subj_sort(role["model_fit"])
            source_all = g.subj_sort(set(role["model_fit"]) | set(role["discovery"]))
            cache = g.FoldCache(dataset, source_all, SEED, fold)
            fit_rows = cache.rows(source, g.sessions_for(dataset))
            discovery_rows = cache.rows(role["discovery"], (g.SESSION_DISCOVERY[dataset],))
            refit_rows = cache.rows(source_all, g.sessions_for(dataset))
            fit_mean, fit_std = cache.normalizer(fit_rows)
            cache.normalize(fit_mean, fit_std)
            print(f"[decision] start {dataset} fold={fold}", flush=True)
            risk1, a1, t1 = g.crossfit_scalars(cache, source, dataset, fold, SEED, "initial_selection", device,
                                               cache_root=cache_root)
            weights1, wa1 = g.source_weights(risk1, dataset, fold, SEED, methods=METHODS)
            risk2, a2, t2 = g.crossfit_scalars(cache, source_all, dataset, fold, SEED, "final_refit", device,
                                               cache_root=cache_root)
            weights2, wa2 = g.source_weights(risk2, dataset, fold, SEED, methods=METHODS)
            refit_mean, refit_std = cache.normalizer(refit_rows)
            cache.normalize(refit_mean, refit_std)
            all_crossfit.extend(a1 + a2); all_teachers.extend(t1 + t2)
            for _, row in risk1.iterrows(): all_risk.append({**row.to_dict(), "seed": SEED})
            for _, row in risk2.iterrows(): all_risk.append({**row.to_dict(), "seed": SEED})
            for row in wa1["rows"]: all_weights.append({**row, "stage": "initial_selection"})
            for row in wa2["rows"]: all_weights.append({**row, "stage": "final_refit"})
            state, init_seed, init_sha = g.initial_state(cache, dataset, fold, SEED, "student")
            key = f"{dataset}/fold-{fold}/seed-{SEED}"
            state_hashes[key] = {"initial_state_sha256": init_sha, "initial_seed": init_seed}
            selected: dict[str, int] = {}
            histories: dict[str, Any] = {}
            for method in METHODS:
                wvec = g.weight_vector(cache, fit_rows, weights1[method], method)
                sel_path = cache_root / dataset / f"fold-{fold}" / "student_initial_selection" / f"{method}.pt"
                ep, hist, sel_hit = g.select_epoch_cached(
                    cache, fit_rows, discovery_rows, fit_mean, fit_std, wvec, state,
                    dataset, fold, SEED, "student-common", device, path=sel_path,
                    expected_extra={"method": method, "stage": "initial_selection"})
                selected[method] = ep; histories[method] = hist
                training_rows.append({"dataset": dataset, "fold": fold, "seed": SEED, "stage": "initial_selection",
                                      "method": method, "selected_epoch": ep, "initial_state_sha256": init_sha,
                                      "normalizer_mean_sha256": g.bytes_sha(fit_mean.tobytes()),
                                      "normalizer_std_sha256": g.bytes_sha(fit_std.tobytes()),
                                      "training_subjects": len(source), "discovery_subjects": len(role["discovery"]),
                                      "weight_mean": float(wvec.mean()), "weight_min": float(wvec.min()), "weight_max": float(wvec.max()),
                                      "selection_sec": float(sum(float(h.get("sec", 0.0)) for h in hist)), "cache_hit": bool(sel_hit)})
            ckpt_info: dict[str, Any] = {}
            for method in METHODS:
                wvec = g.weight_vector(cache, refit_rows, weights2[method], method)
                ck = RUNTIME / f"seed-{SEED}" / dataset / f"fold-{fold}" / f"{method}.pt"
                ck_expected = {"schema": g.CACHE_SCHEMA_VERSION, "code_fingerprint": g.code_fingerprint(),
                               "dataset": dataset, "fold": int(fold), "seed": SEED, "method": method,
                               "stage": "final_refit", "selected_epoch": int(selected[method]),
                               "initial_state_sha256": init_sha, "rows_sha256": g.array_sha(refit_rows),
                               "weights_sha256": g.array_sha(np.asarray(wvec, dtype=np.float32)),
                               "mean_sha256": g.bytes_sha(np.asarray(refit_mean).tobytes()),
                               "std_sha256": g.bytes_sha(np.asarray(refit_std).tobytes())}
                timing: dict[str, Any] = {}
                ck_hit = g.checkpoint_cache_valid(ck, ck_expected)
                if ck_hit:
                    ck_sha = g.file_sha(ck)
                    print(f"[cache] checkpoint hit {dataset} fold={fold} method={method}", flush=True)
                else:
                    model = g.fit_exact(cache, refit_rows, refit_mean, refit_std, wvec, state, dataset, fold, SEED,
                                        "student-common", selected[method], device, timing=timing)
                    ck_sha = g.save_checkpoint(ck, model, refit_mean, refit_std, dataset, fold, SEED, method,
                                               selected[method], init_sha, cache_meta=ck_expected)
                    del model
                ckpt_info[method] = {"path": str(ck), "sha256": ck_sha, "selected_epoch": selected[method]}
                training_rows.append({"dataset": dataset, "fold": fold, "seed": SEED, "stage": "final_refit",
                                      "method": method, "selected_epoch": selected[method], "initial_state_sha256": init_sha,
                                      "normalizer_mean_sha256": g.bytes_sha(refit_mean.tobytes()),
                                      "normalizer_std_sha256": g.bytes_sha(refit_std.tobytes()),
                                      "training_subjects": len(source_all), "discovery_subjects": len(role["discovery"]),
                                      "weight_mean": float(wvec.mean()), "weight_min": float(wvec.min()), "weight_max": float(wvec.max()),
                                      "checkpoint_sha256": ck_sha, "fit_sec": float(timing.get("sec", 0.0)),
                                      "fit_sec_per_epoch": float(timing.get("sec_per_epoch", 0.0)), "cache_hit": bool(ck_hit)})
            fold_manifests[key] = {
                "dataset": dataset, "fold": fold, "seed": SEED, "model_fit_subjects": source,
                "discovery_subjects": g.subj_sort(role["discovery"]),
                "outcome_subjects_hash": g.bytes_sha("|".join(g.subj_sort(role["outcome"])).encode()),
                "role_hash": role_hash(role), "selected_epochs": selected, "checkpoints": ckpt_info,
                "initial_normalizer_mean_sha256": g.bytes_sha(fit_mean.tobytes()),
                "initial_normalizer_std_sha256": g.bytes_sha(fit_std.tobytes()),
                "refit_normalizer_mean_sha256": g.bytes_sha(refit_mean.tobytes()),
                "refit_normalizer_std_sha256": g.bytes_sha(refit_std.tobytes()),
                "source_initial_weight_lock": wa1["lock"], "source_final_weight_lock": wa2["lock"],
                "initial_histories": histories, "methods": list(METHODS), "decision_screen": True,
            }
            write_json(RUNTIME / f"seed-{SEED}" / dataset / f"fold-{fold}" / "FOLD_PROGRESS.json", fold_manifests[key])
            print(f"[decision] complete {dataset} fold={fold}", flush=True)
            del cache
    write_csv(RESULTS / "CROSS_FIT_ASSIGNMENTS.csv", all_crossfit)
    write_csv(RESULTS / "CROSSFIT_TEACHER_AUDIT.csv", all_teachers)
    write_csv(RESULTS / "SOURCE_GEOMETRY_RISK.csv", all_risk)
    write_csv(RESULTS / "SOURCE_WEIGHT_AUDIT.csv", all_weights)
    write_csv(RESULTS / "TRAINING_SUMMARY.csv", training_rows)
    write_json(RESULTS / "INITIAL_STATE_HASHES.json", state_hashes)
    lock = {
        "schema": "PERSIST_EEG_GEOSR_DECISION_PRE_OUTCOME_LOCK_V1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED,
        "datasets": list(DATASETS), "folds": list(FOLDS), "backbone": "EEGNet", "methods": list(METHODS),
        "inner_crossfit_k": g.INNER_K, "descriptor_cap": g.CAP,
        "formula": {"N_geo": "1 - .5*(cos(v_s,t2,c_t1)+cos(v_s,t1,c_t2))",
                     "N_loss": "balanced held-out descriptor NLL", "ranks": "scipy rankdata(method=average)",
                     "GeoSR": "0.5+0.5*r_geo+0.5*r_loss", "weight_range": [0.5, 1.5]},
        "training": {"architecture": "canonical VanillaEEGNet F1=8 D=2 F2=16 temporal=64 pool=4,8 dropout=.25 embedding=64",
                     "optimizer": "AdamW lr=3e-4 wd=5e-4 batch=64 grad_clip=5", "max_epochs": g.MAX_EPOCHS,
                     "min_epochs": g.MIN_EPOCHS, "patience": g.PATIENCE, "epoch_selection": "discovery mean subject BA; lower NLL; earlier epoch",
                     "fair_initial_state_per_outer_fold": True, "same_minibatch_order": True},
        "role_hashes": {d: [role_hashes[d][f] for f in FOLDS] for d in DATASETS},
        "fold_manifests": fold_manifests,
        "outcome_labels_read": False, "canonical_outcome_indices_materialized": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "decision_screen": True, "complete_protocol_required_for_final_claim": True,
        "code_sha256": g.file_sha(Path(g.__file__)), "audit_primitives_sha256": g.file_sha(Path(g.ap.__file__)),
    }
    write_json(protocol_lock_path(), lock)
    write_json(ROOT / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_DECISION_DATA_LEGALITY_V1", "seed": SEED,
        "decision_run": True, "folds": list(FOLDS), "methods": list(METHODS),
        "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False,
        "outcome_labels_read_before_lock": False, "outcome_labels_read_after_lock": False,
        "lock_sha256": g.file_sha(protocol_lock_path()), "role_hashes": role_hashes, "descriptor_cap": g.CAP,
    })
    write_json(RUNTIME / f"seed-{SEED}" / "PREFLIGHT_MANIFEST.json", fold_manifests)
    print("DECISION_PRE_OUTCOME_LOCKED", flush=True)


def evaluate_outcome(device) -> dict[str, Any]:
    lock = json.loads(protocol_lock_path().read_text(encoding="utf-8"))
    if lock.get("outcome_labels_read") is not False or lock.get("decision_screen") is not True:
        raise RuntimeError("invalid decision pre-outcome lock")
    manifest = json.loads((RUNTIME / f"seed-{SEED}" / "PREFLIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        roles, _, _ = ap.load_roles(dataset)
        for fold in FOLDS:
            role = roles[fold]
            # First and only materialization of outcome-subject labels occurs
            # after the decision lock has been atomically written.
            data = ap.load_ab_data(dataset, set(role["outcome"]))
            key = f"{dataset}/fold-{fold}/seed-{SEED}"
            for method in METHODS:
                ck = Path(manifest[key]["checkpoints"][method]["path"])
                all_rows.extend([{**r, "method": method} for r in g.eval_checkpoint(data, ck, role["outcome"], dataset, fold, SEED, device)])
                data = ap.load_ab_data(dataset, set(role["outcome"]))
            del data
    frame = pd.DataFrame(all_rows)
    write_csv(RESULTS / "OUTCOME_PER_SUBJECT.csv", frame)
    fold_summary = frame.groupby(["dataset", "fold", "seed", "method"], as_index=False).agg(
        mean_subject_BA=("BA", "mean"), mean_accuracy=("accuracy", "mean"), mean_macro_F1=("macro_F1", "mean"),
        mean_NLL=("NLL", "mean"), n_subjects=("subject_id", "nunique"))
    write_csv(RESULTS / "OUTCOME_PER_FOLD.csv", fold_summary)
    subj = frame.groupby(["dataset", "method", "subject_id"], as_index=False).agg(
        BA=("BA", "mean"), accuracy=("accuracy", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"))
    perf = subj.groupby(["dataset", "method"], as_index=False).agg(
        mean_subject_BA=("BA", "mean"), mean_accuracy=("accuracy", "mean"), mean_macro_F1=("macro_F1", "mean"),
        mean_NLL=("NLL", "mean"), n_subjects=("subject_id", "nunique"))
    write_csv(RESULTS / "PERFORMANCE_SUMMARY.csv", perf)
    metrics: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        f = subj[subj.dataset == dataset]
        base = f[f.method == "SUBJECT_BALANCED_ERM"].set_index("subject_id").BA
        geo = f[f.method == "GEOSR"].set_index("subject_id").reindex(base.index)
        random = f[f.method == "RANDOM_RANK"].set_index("subject_id").reindex(base.index)
        delta = geo.BA - base
        bottom_ids = base.sort_values().index[:max(1, int(math.ceil(.25 * len(base))))]
        boot_sb = g.bootstrap_pair(geo.BA.to_numpy(), base.to_numpy(), dataset, "SUBJECT_BALANCED_ERM_DECISION", SEED)
        boot_random = g.bootstrap_pair(geo.BA.to_numpy(), random.BA.to_numpy(), dataset, "RANDOM_RANK_DECISION", SEED)
        fold_geo = fold_summary[(fold_summary.dataset == dataset) & (fold_summary.method == "GEOSR")].set_index("fold").mean_subject_BA
        fold_sb = fold_summary[(fold_summary.dataset == dataset) & (fold_summary.method == "SUBJECT_BALANCED_ERM")].set_index("fold").mean_subject_BA
        bottom_delta = float((geo.loc[bottom_ids].BA.mean() - base.loc[bottom_ids].mean()) * 100.0)
        row = {
            "dataset": dataset, "n_subjects": int(len(delta)), "mean_delta_pp": float(delta.mean() * 100.0),
            "bottom25_delta_pp": bottom_delta, "nonnegative_subject_fraction": float(np.mean(delta >= 0)),
            "material_harm_fraction": float(np.mean(delta <= -.02)), "fold_nonnegative_count": int(np.sum(fold_geo.to_numpy() >= fold_sb.to_numpy())),
            "fold_count": len(FOLDS), "geosr_mean_BA": float(geo.BA.mean()), "sb_erm_mean_BA": float(base.mean()),
            "random_rank_mean_BA": float(random.BA.mean()), "geosr_not_worse_random": bool(geo.BA.mean() >= random.BA.mean()),
            "bootstrap_sb": boot_sb, "bootstrap_random": boot_random,
        }
        metrics[dataset] = row
        comparison_rows.extend([
            {"dataset": dataset, "method": "GEOSR", "comparator": "SUBJECT_BALANCED_ERM", **boot_sb},
            {"dataset": dataset, "method": "GEOSR", "comparator": "RANDOM_RANK", **boot_random},
        ])
    write_csv(RESULTS / "CONTROL_COMPARISON.csv", comparison_rows)
    screen: dict[str, bool] = {}
    for d, m in metrics.items():
        screen[d] = bool(
            m["mean_delta_pp"] >= SCREEN_RULE["both_dataset_mean_delta_min_pp"]
            and m["bottom25_delta_pp"] >= SCREEN_RULE["bottom25_delta_min_pp"]
            and m["fold_nonnegative_count"] >= SCREEN_RULE["folds_required_nonnegative"]
            and m["nonnegative_subject_fraction"] >= SCREEN_RULE["nonnegative_subject_fraction_min"]
            and m["material_harm_fraction"] <= SCREEN_RULE["material_harm_fraction_max"]
            and m["geosr_not_worse_random"]
            and m["bootstrap_sb"]["CI95_high_pp"] > SCREEN_RULE["paired_bootstrap_upper_ci_min_pp"]
        )
    continue_folds = bool(all(screen.values()) and any(metrics[d]["mean_delta_pp"] >= SCREEN_RULE["one_dataset_mean_delta_min_pp"] for d in DATASETS))
    terminal = "DECISION_RUN_POSITIVE_SIGNAL_CONTINUE_FOLDS_2_4" if continue_folds else "DECISION_RUN_NO_STABLE_POSITIVE_SIGNAL_STOP"
    decision = {"schema": "PERSIST_EEG_GEOSR_DECISION_RESULT_V1", "seed": SEED, "terminal": terminal,
                "continue_folds_2_4": continue_folds, "screen_by_dataset": screen, "metrics": metrics,
                "screen_rule": SCREEN_RULE, "outer_sealed_access": False, "outcome_after_lock": True,
                "scientific_definition_changed": False, "final_claim_authorized": False}
    write_json(RESULTS / "DECISION_SCREENING.json", decision)
    write_json(RESULTS / "FINAL_DECISION.json", decision)
    report = ["# GeoSR seed-0 decision screen", "", f"Terminal: `{terminal}`", "",
              "This is a two-fold/three-method continuation screen, not the complete six-method/five-fold claim.", "",
              "|Dataset|GeoSR−SB-ERM (pp)|bottom25 (pp)|nonnegative|material harm|folds nonnegative|GeoSR≥Random|", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        m = metrics[d]
        report.append(f"|{d}|{m['mean_delta_pp']:.3f}|{m['bottom25_delta_pp']:.3f}|{m['nonnegative_subject_fraction']:.3f}|{m['material_harm_fraction']:.3f}|{m['fold_nonnegative_count']}/{m['fold_count']}|{m['geosr_not_worse_random']}|")
    report += ["", "Fixed screen:", "", *[f"- {k}: `{v}`" for k, v in SCREEN_RULE.items()], "",
               f"Continuation decision: `{continue_folds}`", "", "No outer sealed data or scientific rescue was used.", ""]
    (RESULTS / "DECISION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    legality = json.loads((ROOT / "DATA_LEGALITY_AUDIT.json").read_text(encoding="utf-8"))
    legality.update({"canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": True,
                     "outcome_labels_read_after_lock": True, "outcome_evaluated_after_lock": True,
                     "lock_sha256": g.file_sha(protocol_lock_path())})
    write_json(ROOT / "DATA_LEGALITY_AUDIT.json", legality)
    write_json(RESULTS / "VALIDATION.json", {"decision_screen_only": True, "outcome_after_lock": True,
                                               "outer_sealed_closed": True, "scientific_definition_changed": False,
                                               "pass": True})
    print(terminal, flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "outcome", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = __import__("torch").device(args.device)
    if args.phase in ("preflight", "all"):
        preflight()
    if args.phase in ("outcome", "all"):
        evaluate_outcome(device)


if __name__ == "__main__":
    main()
