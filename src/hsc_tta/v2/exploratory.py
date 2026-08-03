from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score

from hsc_tta.actions import T3A
from hsc_tta.actions_v2 import RobustResidualAdapter
from hsc_tta.prediction_sets import evaluate_prediction_sets
from hsc_tta.v2.benefit_predictor import fit_benefit_predictor
from hsc_tta.v2.development_surfaces import ACTIONS, ALPHAS, LAMBDAS, _features, _labels, _outputs, _tokens, load_source_model
from hsc_tta.v2.joint_certificate import estimate_scales, finite_sample_quantile, joint_bounds, subject_joint_scores
from hsc_tta.v2.risk_predictor import fit_risk_predictor
from hsc_tta.v2.selector_v2 import select_joint_action


IDENTIFIERS = {"dataset", "seed", "subject_id", "alpha", "action_available"}


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def _context_state(root: Path, token_dataset: str, model_dataset: str, seed: int, subject: str,
                   indices: np.ndarray, device: str, state_path: Path) -> list[dict[str, object]]:
    model, _ = load_source_model(root, model_dataset, seed, device)
    context = _tokens(root, token_dataset, subject, indices)
    logits, hidden = _outputs(model, context, device)
    source = softmax(logits.astype(np.float64), axis=1)
    probabilities = {"no_tta": source}
    diagnostics: dict[str, dict[str, object]] = {"no_tta": {"status": "ok"}}
    t3a = T3A(model.classifier.weight.detach().cpu().numpy(), filter_k=20, confidence=None)
    initial = t3a.prototypes.copy()
    t3a.adapt(hidden, logits)
    probabilities["official_t3a"] = t3a.predict_proba(hidden)
    diagnostics["official_t3a"] = {"status": "ok", "prototype_shift": float(np.linalg.norm(t3a.prototypes - initial)),
                                     "support_count": len(t3a.supports)}
    robust = RobustResidualAdapter(model, steps=3, learning_rate=5e-5, beta=.5, gamma=.1, eta=1e-3,
                                   reliability_quantile=.2, device=device)
    robust.adapt_on_context(context)
    robust_status = robust.failure_status()
    diagnostics["robust_residual_adapter"] = {"status": robust_status, **robust.diagnostics()}
    robust_state = None
    if robust_status == "ok":
        probabilities["robust_residual_adapter"] = robust.predict_context(context)
        robust_state = robust.freeze_state()
    else:
        probabilities["robust_residual_adapter"] = source
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"t3a_prototypes": t3a.prototypes, "robust_status": robust_status,
                "robust_state": robust_state}, state_path)
    rows = []
    for action in ACTIONS:
        row = _features(subject, action, source, probabilities[action], hidden, diagnostics[action])
        row.update({"dataset": token_dataset, "seed": seed, "action_available": diagnostics[action]["status"] == "ok",
                    "u_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()})
        rows.append(row)
    return rows


def _future_outcomes(root: Path, token_dataset: str, model_dataset: str, seed: int, subject: str,
                     indices: np.ndarray, device: str, state_path: Path) -> list[dict[str, object]]:
    model, _ = load_source_model(root, model_dataset, seed, device)
    future = _tokens(root, token_dataset, subject, indices)
    labels = _labels(root, token_dataset, subject, indices)
    logits, hidden = _outputs(model, future, device)
    source = softmax(logits.astype(np.float64), axis=1)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    probabilities = {"no_tta": source, "official_t3a": softmax(hidden @ state["t3a_prototypes"], axis=1)}
    if state["robust_status"] == "ok":
        robust = RobustResidualAdapter(model, steps=3, learning_rate=5e-5, beta=.5, gamma=.1, eta=1e-3,
                                       reliability_quantile=.2, device=device)
        robust._status = "ok"
        robust._frozen = state["robust_state"]
        probabilities["robust_residual_adapter"] = robust.predict_future(future)
    else:
        probabilities["robust_residual_adapter"] = source
    no_error = float(np.mean(source.argmax(1) != labels))
    rows: list[dict[str, object]] = []
    for action in ACTIONS:
        curve = evaluate_prediction_sets(probabilities[action], labels, LAMBDAS)
        error = float(curve[0]["argmax_error"])
        for alpha in ALPHAS:
            critical = next((int(row["lambda_index"]) for row in curve if row["future_risk"] <= alpha), 20)
            selected = curve[critical]
            result = {"dataset": token_dataset, "seed": seed, "subject_id": subject, "action": action, "alpha": alpha,
                      "true_critical_index": critical, "true_benefit": no_error - error, "argmax_error": error,
                      "future_risk": selected["future_risk"], "future_average_set_size": selected["average_set_size"],
                      "future_singleton_rate": selected["singleton_rate"], "macro_f1": selected["macro_f1"],
                      "balanced_accuracy": selected["balanced_accuracy"],
                      "cohen_kappa": cohen_kappa_score(labels, probabilities[action].argmax(1)),
                      "action_available": state["robust_status"] == "ok" if action == "robust_residual_adapter" else True}
            for index, curve_row in enumerate(curve):
                result[f"risk_j{index}"] = curve_row["future_risk"]
                result[f"set_size_j{index}"] = curve_row["average_set_size"]
                result[f"singleton_j{index}"] = curve_row["singleton_rate"]
            rows.append(result)
    return rows


def _fit_bundle(features: pd.DataFrame, outcomes: pd.DataFrame, meta_ids: set[str], alpha: float,
                feature_columns: list[str]):
    meta_b = features[features.subject_id.isin(meta_ids)].merge(
        outcomes[np.isclose(outcomes.alpha, .1) & outcomes.subject_id.isin(meta_ids)][["subject_id", "action", "true_benefit"]],
        on=["subject_id", "action"], validate="one_to_one").rename(columns={"true_benefit": "benefit_target"})
    benefit = fit_benefit_predictor(meta_b, feature_columns)
    chosen_b = benefit.oof[benefit.oof.model == benefit.model_name].rename(
        columns={"true_gain": "true_benefit", "predicted_gain": "predicted_benefit"})
    meta_r = features[features.subject_id.isin(meta_ids)].merge(
        outcomes[np.isclose(outcomes.alpha, alpha) & outcomes.subject_id.isin(meta_ids)][["subject_id", "action", "true_critical_index"]],
        on=["subject_id", "action"], validate="one_to_one")
    meta_r["alpha"] = alpha
    risk = fit_risk_predictor(meta_r, feature_columns)
    chosen_r = risk.oof[risk.oof.model == risk.model_name]
    scale_frame = chosen_r.merge(chosen_b[["subject_id", "action", "true_benefit", "predicted_benefit"]],
                                 on=["subject_id", "action"], validate="one_to_one")
    scale_frame.loc[scale_frame.action == "no_tta", ["true_benefit", "predicted_benefit"]] = 0.0
    return risk.model, benefit.model, estimate_scales(scale_frame)


def _freeze_decisions(features: pd.DataFrame, cal_outcomes: pd.DataFrame, eval_ids: set[str], cal_ids: set[str],
                      risk_model, benefit_model, scales, alpha: float, feature_columns: list[str], path: Path) -> pd.DataFrame:
    cal = features[features.subject_id.isin(cal_ids)].merge(
        cal_outcomes[np.isclose(cal_outcomes.alpha, alpha) & cal_outcomes.subject_id.isin(cal_ids)][
            ["subject_id", "action", "true_critical_index", "true_benefit"]], on=["subject_id", "action"], validate="one_to_one")
    cal["predicted_critical_index"] = risk_model.predict(cal[feature_columns])
    cal["predicted_benefit"] = benefit_model.predict(cal[feature_columns])
    cal.loc[cal.action == "no_tta", ["true_benefit", "predicted_benefit"]] = 0.0
    q, _ = finite_sample_quantile(subject_joint_scores(cal, scales).joint_score, .1)
    ev = features[features.subject_id.isin(eval_ids)].copy()
    ev["predicted_critical_index"] = risk_model.predict(ev[feature_columns])
    ev["predicted_benefit"] = benefit_model.predict(ev[feature_columns])
    ev.loc[ev.action == "no_tta", "predicted_benefit"] = 0.0
    upper, lower = joint_bounds(ev.predicted_critical_index, ev.predicted_benefit, q, scales, 20)
    ev["certified_critical_index"], ev["benefit_lower"] = upper, lower
    decisions = []
    for subject, group in ev.groupby("subject_id"):
        candidates = []
        for row in group.itertuples(index=False):
            index = int(row.certified_critical_index)
            candidates.append({"action": row.action, "available": row.action_available,
                               "certified_critical_index": index, "benefit_lower": row.benefit_lower,
                               "context_average_set_size": getattr(row, f"context_set_size_j{index}"),
                               "adaptation_cost": row.action_cost})
        decisions.append({"subject_id": subject, "alpha": alpha, "q": q,
                          **select_joint_action(pd.DataFrame(candidates), sentinel_index=20)})
    frame = pd.DataFrame(decisions)
    _atomic(frame, path)
    freeze = {"decision_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "V_opened": False,
              "label": "exploratory_tainted_final"}
    path.with_suffix(".freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return frame


def _summarize(decisions: pd.DataFrame, outcomes: pd.DataFrame, dataset: str, seed: int) -> list[dict[str, object]]:
    rows = []
    for alpha, decision in decisions.groupby("alpha"):
        selected = decision.merge(outcomes[np.isclose(outcomes.alpha, alpha)], left_on=["subject_id", "selected_action"],
                                  right_on=["subject_id", "action"], validate="one_to_one")
        risks = np.asarray([r[f"risk_j{int(r.certified_critical_index)}"] for _, r in selected.iterrows()])
        tta = selected.selected_action != "no_tta"
        values = {"violation": float(np.mean(risks > alpha)), "csr": float(np.mean(selected.certified_critical_index < 20)),
                  "full_set_fallback": float(np.mean(selected.certified_critical_index == 20)),
                  "average_set_size": float(np.mean([r[f"set_size_j{int(r.certified_critical_index)}"] for _, r in selected.iterrows()])),
                  "argmax_error": float(selected.argmax_error.mean()), "macro_f1": float(selected.macro_f1.mean()),
                  "balanced_accuracy": float(selected.balanced_accuracy.mean()), "cohen_kappa": float(selected.cohen_kappa.mean()),
                  "gain_vs_no_tta": float(selected.true_benefit.mean()), "tta_selection_rate": float(tta.mean()),
                  "selected_tta_ppv": float(np.mean(selected.loc[tta, "true_benefit"] > 0)) if tta.any() else np.nan}
        rows.extend({"dataset": dataset, "seed": seed, "alpha": alpha, "policy": "joint_hsc_tta_v2",
                     "label": "exploratory_tainted_external" if dataset == "cap" else "exploratory_tainted_final",
                     "metric": metric, "value": value} for metric, value in values.items())
    return rows


def run_exploratory_replication(root: str | Path, device: str = "cuda", resume: bool = True) -> dict[str, pd.DataFrame]:
    root = Path(root)
    base = root / "outputs/v2_joint_certified"
    if not (base / "freeze/V2_METHOD_FREEZE.json").exists():
        raise PermissionError("V2_METHOD_FREEZE.json is required before exploratory final access")
    development_features = pd.read_parquet(base / "actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    development_outcomes = pd.read_parquet(base / "actions/DEVELOPMENT_ACTION_SURFACE.parquet")
    feature_columns = [c for c in development_features.columns if c not in IDENTIFIERS and c != "u_state_sha256"]
    out = base / "exploratory"
    out.mkdir(exist_ok=True)
    result_rows: dict[str, list[dict[str, object]]] = {"hmc": [], "eegmmidb": [], "cap": []}
    for seed in range(5):
        hmc_bundles = {}
        for dataset in ("hmc", "eegmmidb"):
            split = json.loads((root / "data/splits" / dataset / f"seed_{seed}.json").read_text())
            roles = split["roles"]
            test_ids = set(roles["final_test"])
            meta_ids, cal_ids = set(roles["meta_risk_train"]), set(roles["conformal_calibration"])
            episodes = pd.read_parquet(root / "data/episodes_main120" / dataset / f"seed_{seed}.parquet").set_index("subject_id")
            state_dir = out / "u_states" / dataset / f"seed_{seed}"
            feature_path = out / "context_features" / dataset / f"seed_{seed}.parquet"
            test_features = pd.read_parquet(feature_path) if resume and feature_path.exists() else pd.DataFrame()
            completed = set(test_features.subject_id.unique()) if len(test_features) else set()
            records = test_features.to_dict("records")
            for subject in sorted(test_ids - completed):
                records.extend(_context_state(root, dataset, dataset, seed, subject, np.asarray(episodes.loc[subject].context_indices, int),
                                              device, state_dir / f"{subject.replace(':', '_')}.pt"))
                _atomic(pd.DataFrame(records), feature_path)
            test_features = pd.DataFrame(records)
            train_features = development_features[(development_features.dataset == dataset) & (development_features.seed == seed)]
            train_outcomes = development_outcomes[(development_outcomes.dataset == dataset) & (development_outcomes.seed == seed)]
            all_features = pd.concat([train_features, test_features], ignore_index=True)
            decisions = []
            for alpha in ALPHAS:
                bundle = _fit_bundle(train_features, train_outcomes, meta_ids, alpha, feature_columns)
                if dataset == "hmc": hmc_bundles[alpha] = bundle
                decisions.append(_freeze_decisions(all_features, train_outcomes, test_ids, cal_ids, *bundle, alpha,
                                                   feature_columns, out / "decisions" / dataset / f"seed_{seed}_alpha_{alpha:.2f}.parquet"))
            decisions_frame = pd.concat(decisions, ignore_index=True)
            future_path = out / "counterfactuals" / dataset / f"seed_{seed}.parquet"
            future = pd.read_parquet(future_path) if resume and future_path.exists() else pd.DataFrame()
            completed_future = set(future.subject_id.unique()) if len(future) else set()
            future_records = future.to_dict("records")
            for subject in sorted(test_ids - completed_future):
                future_records.extend(_future_outcomes(root, dataset, dataset, seed, subject, np.asarray(episodes.loc[subject].future_indices, int),
                                                       device, state_dir / f"{subject.replace(':', '_')}.pt"))
                _atomic(pd.DataFrame(future_records), future_path)
            for freeze_path in (out / "decisions" / dataset).glob(f"seed_{seed}_*.freeze.json"):
                payload = json.loads(freeze_path.read_text()); payload["V_opened"] = True
                freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            result_rows[dataset].extend(_summarize(decisions_frame, pd.DataFrame(future_records), dataset, seed))

        dataset = "cap"
        split = json.loads((root / "data/splits/cap" / f"seed_{seed}.json").read_text())
        roles = split["roles"]; cal_ids, test_ids = set(roles["target_site_calibration"]), set(roles["external_final_test"])
        all_ids = cal_ids | test_ids
        episodes = pd.read_parquet(root / "data/episodes_main120/cap" / f"seed_{seed}.parquet").set_index("subject_id")
        state_dir = out / "u_states/cap" / f"seed_{seed}"
        feature_path = out / "context_features/cap" / f"seed_{seed}.parquet"
        cap_features = pd.read_parquet(feature_path) if resume and feature_path.exists() else pd.DataFrame()
        completed = set(cap_features.subject_id.unique()) if len(cap_features) else set(); records = cap_features.to_dict("records")
        for subject in sorted(all_ids - completed):
            records.extend(_context_state(root, "cap", "hmc", seed, subject, np.asarray(episodes.loc[subject].context_indices, int),
                                          device, state_dir / f"{subject.replace(':', '_')}.pt")); _atomic(pd.DataFrame(records), feature_path)
        cap_features = pd.DataFrame(records)
        cal_path = out / "counterfactuals/cap" / f"seed_{seed}_calibration.parquet"
        cal_outcomes = pd.read_parquet(cal_path) if resume and cal_path.exists() else pd.DataFrame()
        completed_cal = set(cal_outcomes.subject_id.unique()) if len(cal_outcomes) else set(); cal_records = cal_outcomes.to_dict("records")
        for subject in sorted(cal_ids - completed_cal):
            cal_records.extend(_future_outcomes(root, "cap", "hmc", seed, subject, np.asarray(episodes.loc[subject].future_indices, int),
                                                device, state_dir / f"{subject.replace(':', '_')}.pt")); _atomic(pd.DataFrame(cal_records), cal_path)
        decisions = []
        for alpha in ALPHAS:
            decisions.append(_freeze_decisions(cap_features, pd.DataFrame(cal_records), test_ids, cal_ids, *hmc_bundles[alpha], alpha,
                                               feature_columns, out / "decisions/cap" / f"seed_{seed}_alpha_{alpha:.2f}.parquet"))
        decisions_frame = pd.concat(decisions, ignore_index=True)
        test_path = out / "counterfactuals/cap" / f"seed_{seed}_test.parquet"
        test_outcomes = pd.read_parquet(test_path) if resume and test_path.exists() else pd.DataFrame()
        completed_test = set(test_outcomes.subject_id.unique()) if len(test_outcomes) else set(); test_records = test_outcomes.to_dict("records")
        for subject in sorted(test_ids - completed_test):
            test_records.extend(_future_outcomes(root, "cap", "hmc", seed, subject, np.asarray(episodes.loc[subject].future_indices, int),
                                                 device, state_dir / f"{subject.replace(':', '_')}.pt")); _atomic(pd.DataFrame(test_records), test_path)
        for freeze_path in (out / "decisions/cap").glob(f"seed_{seed}_*.freeze.json"):
            payload = json.loads(freeze_path.read_text()); payload["V_opened"] = True
            freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result_rows["cap"].extend(_summarize(decisions_frame, pd.DataFrame(test_records), "cap", seed))
    outputs = {}
    for dataset, rows in result_rows.items():
        frame = pd.DataFrame(rows); frame.to_csv(out / f"EXPLORATORY_{dataset.upper()}_RESULTS.csv", index=False); outputs[dataset] = frame
    (out / "EXPLORATORY_RESULTS_NOT_CONFIRMATORY.md").write_text(
        "# Exploratory replication only\n\nHMC, EEGMMIDB, and CAP outcomes were inspected during v1. These post-freeze reruns are labeled tainted exploratory replications and cannot be used for method revision or confirmatory claims. U-derived action states and decisions were written and hashed before each corresponding test V was opened.\n",
        encoding="utf-8")
    return outputs
