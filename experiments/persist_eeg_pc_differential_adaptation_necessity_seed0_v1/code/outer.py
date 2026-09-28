"""Read OUTER_DEVELOPMENT only after TRAIN selection; frozen-classifier evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ANALYSIS_PATH = Path(os.environ.get("PCDA_ANALYSIS_CODE", str(Path(__file__).with_name("analysis.py"))))
spec = importlib.util.spec_from_file_location("pcda_analysis", ANALYSIS_PATH)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
core = analysis.core


def outer_embeddings(model_name: str, task: str, fold: int, train_proof: dict):
    ids, sessions, split_sha, cache_name = core.SDG.split_role(task, fold)
    if split_sha != train_proof["split_sha256"] or {k: list(v) for k, v in ids.items()} != train_proof["role_subjects"]:
        raise RuntimeError("frozen population role/split drift")
    inner = core.SDG.module(f"pcda_outer_inner_{fold}", core.SDG.SEVEN_CODE / "tech_recipe_selection.py")
    source, sy, ss, mapping = inner._openbmi_rows(core.SDG.CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[0],), cache_name)
    mean, std, norm_sha = core.SDG.historical_normalizer(source)
    del source, sy, ss
    if norm_sha != train_proof["normalizer_sha256"] or core.clean(mapping) != train_proof["class_mapping"]:
        raise RuntimeError("frozen preprocessing drift")
    model, head, proof = core.SDG.checkpoint(model_name, task, fold, split_sha, norm_sha,
                                             (int(train_proof.get("channels") or 62), int(train_proof.get("samples") or (250 if task == "OpenBMI_ERP" else 1000))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    before = core.SDG.state_sha(model)
    if before != train_proof["model_state_sha256_before"] or proof["checkpoint_sha256"] != train_proof["checkpoint_sha256"]:
        raise RuntimeError("frozen model/checkpoint drift")
    outer = core.SDG.population_features(inner, model, head, ids, task, fold, "OUTER_DEVELOPMENT",
                                          sessions, cache_name, mapping, mean, std, device)
    after = core.SDG.state_sha(model)
    if after != before: raise RuntimeError("model parameter or native buffer mutated")
    proof.update({"model_state_sha256_before": before, "model_state_sha256_after": after,
                  "split_sha256": split_sha, "normalizer_sha256": norm_sha,
                  "outer_subject_ids": list(ids["OUTER_DEVELOPMENT"]),
                  "checkpoint_validation_subject_ids": list(ids["CHECKPOINT_VALIDATION"]),
                  "formal_final_heldout_eeg_reads": 0})
    if set(outer["subject"]) != set(ids["OUTER_DEVELOPMENT"]) or set(outer["subject"]) & set(ids["TRAIN_GEOMETRY"]):
        raise RuntimeError("OUTER subject inventory or overlap failure")
    return outer, proof


def subject_episodes(outer: dict, train: dict, task: str, fold: int, budget: int, lam: float):
    h, y, sub, sess = (outer[k] for k in ("h", "y", "subject", "session"))
    source = np.asarray(train["h"][train["session"] == 1], np.float64)
    mu, sd = analysis.source_stats(source)
    w = np.asarray(train["head_weight"], np.float64)
    bias = np.asarray(train["head_bias"], np.float64)
    result = {}
    for subject in sorted(set(sub), key=int):
        context = (sub == subject) & (sess == 1)
        evaluated = (sub == subject) & (sess == 2)
        ch = np.asarray(h[context], np.float64)
        eh = np.asarray(h[evaluated], np.float64)
        ey = np.asarray(y[evaluated], np.int64)
        # OUTER context is selected without reading its labels.  The frozen
        # OpenBMI ERP cache order contains complete 12-stimulus sequences;
        # that grouping was checked using only TRAIN pseudo-target rows.
        if task == "OpenBMI_ERP":
            if len(ch) < 12 * budget:
                raise RuntimeError(f"ERP OUTER context shorter than {budget} full sequences")
            ix = np.arange(12 * budget)
            ctx = {"unit": "fixed_12_stimulus_sequence_cache_order", "units": budget,
                   "epochs": 12 * budget, "label_blind": True}
        else:
            if len(ch) < budget: raise RuntimeError(f"MI OUTER context shorter than {budget} trials")
            ix = np.arange(budget)
            ctx = {"unit": "trial", "units": budget, "epochs": budget, "label_blind": True}
        delta = analysis.delta_affine(eh, ch[ix], mu, sd, lam)
        result[str(subject)] = {"subject": str(subject), "h": eh, "y": ey,
                                "base": eh @ w.T + bias, "delta": delta, "w": w,
                                "context": ctx, "source_mu": mu, "source_std": sd,
                                "eval_rows": len(ey)}
    return result


def final_bases(train: dict, model: str, task: str, fold: int):
    p = np.asarray(train["p_final"], np.float64)
    r = p.shape[1]
    h = np.asarray(train["h"], np.float64)
    y, sub, sess = (train[k] for k in ("y", "subject", "session"))
    bases = {"PROTECTED": p, "PERSIST_ALL": np.asarray(train["persist_final"], np.float64),
             "PCA_R": PCA(n_components=r, svd_solver="full").fit(h).components_.T if r else np.empty((h.shape[1], 0)),
             "SUPERVISED_DECISION_R": analysis.supervised_basis(h, y, sub, sess, r, task)}
    bases["RANDOM_R"] = [analysis.random_basis(h.shape[1], r, model, task, fold, -1, draw) for draw in range(100)]
    for name, q in bases.items():
        if name == "RANDOM_R": continue
        if q.shape[1] and not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-6):
            raise RuntimeError(f"nonorthonormal final partition: {name}")
    return bases


def method_rows(episodes: dict, q: np.ndarray, pair: tuple[float, float], method: str, partition: str, draw: int = -1):
    rows = []
    for subject, ep in episodes.items():
        score, _, _, _ = analysis.scores(ep, q, pair)
        rows.append({"subject": subject, "method": method, "partition": partition,
                     "draw": draw, "alpha_P": pair[0], "alpha_C": pair[1],
                     "context_units": ep["context"]["units"], "context_epochs": ep["context"]["epochs"],
                     "eval_rows": ep["eval_rows"], **score})
    return rows


def evaluate(model: str, task: str, fold: int) -> None:
    name = core.cell_id(model, task, fold)
    train_path = core.RUNTIME / "train" / f"{name}.npz"
    train_audit_path = core.RUNTIME / "train" / f"{name}.json"
    selection_path = core.RUNTIME / "selection" / f"{name}.json"
    target = core.RUNTIME / "outer" / f"{name}.json"
    if target.exists(): raise FileExistsError(target)
    train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    if selected["protocol_sha256"] != core.sha(core.EXP / "protocol/PROTOCOL_LOCK.json") or selected["train_cache_sha256"] != core.sha(train_path) or selected["train_audit_sha256"] != core.sha(train_audit_path):
        raise RuntimeError("selection/protocol/TRAIN hash mismatch")
    if selected["outer_development_accessed"] or selected["final_heldout_eeg_reads"]:
        raise RuntimeError("selection has forbidden outcome access")
    with np.load(train_path, allow_pickle=False) as loaded:
        train = {key: loaded[key] for key in loaded.files}
    print("OUTER_EXTRACTION_START", name, flush=True)
    outer, proof = outer_embeddings(model, task, fold, train_audit["proof"])
    if core.SDG.digest_array(train["head_weight"], train["head_bias"]) != train_audit["proof"]["head_weight_sha256"]:
        raise RuntimeError("frozen head hash mismatch")
    budget = int(selected["source_only_selected_base"]["budget"])
    lam = float(selected["source_only_selected_base"]["lambda"])
    episodes = subject_episodes(outer, train, task, fold, budget, lam)
    bases = final_bases(train, model, task, fold)
    if int(bases["PROTECTED"].shape[1]) != train_audit["full"]["rank_protected"]:
        raise RuntimeError("final P rank mismatch")
    dim = train["h"].shape[1]
    empty = np.empty((dim, 0), np.float64)
    chosen = selected["partition_selections"]["PROTECTED"]["best"]
    uniform = selected["partition_selections"]["PROTECTED"]["uniform"]
    diff_pair = (float(chosen["alpha_P"]), float(chosen["alpha_C"]))
    uniform_alpha = float(uniform["alpha_P"])
    norm_alpha = float(selected["normmatched_uniform_alpha"])
    methods = [
        ("NO_ADAPT", "PROTECTED", (0.0, 0.0), bases["PROTECTED"]),
        ("FULL_GENERIC", "PROTECTED", (1.0, 1.0), bases["PROTECTED"]),
        ("DIFFERENTIAL_P_C", "PROTECTED", diff_pair, bases["PROTECTED"]),
        ("BEST_UNIFORM", "PROTECTED", (uniform_alpha, uniform_alpha), bases["PROTECTED"]),
        ("NORM_MATCHED_UNIFORM", "PROTECTED", (norm_alpha, norm_alpha), bases["PROTECTED"]),
        ("STRICT_P_PROTECT", "PROTECTED", (0.0, 1.0), bases["PROTECTED"]),
        ("P_ONLY_ADAPT", "PROTECTED", (1.0, 0.0), bases["PROTECTED"]),
    ]
    for family in ("PERSIST_ALL", "PCA_R", "SUPERVISED_DECISION_R"):
        row = selected["partition_selections"][family]["best"]
        methods.append((f"{family}_DIFFERENTIAL", family,
                        (float(row["alpha_P"]), float(row["alpha_C"])), bases[family]))
    method = []
    for method_name, family, pair, q in methods:
        method.extend(method_rows(episodes, q, pair, method_name, family))
    random = []
    for row in selected["random_partition_selections"]:
        draw = int(row["draw"])
        best = row["best"]
        pair = (float(best["alpha_P"]), float(best["alpha_C"]))
        random.extend(method_rows(episodes, bases["RANDOM_R"][draw], pair,
                                  "RANDOM_R_DIFFERENTIAL", "RANDOM_R", draw))
    print("OUTER_RANDOM_COMPLETE", name, len(random), flush=True)
    tolerance = []
    for axis in ("P", "C"):
        for a in analysis.ALPHAS:
            pair = (a, diff_pair[1]) if axis == "P" else (diff_pair[0], a)
            tolerance.extend({**r, "axis": axis, "swept_alpha": a} for r in
                             method_rows(episodes, bases["PROTECTED"], pair, f"{axis}_SWEEP", "PROTECTED"))
    rollback = []
    for component in ("P", "C"):
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            pair = (beta, 1.0) if component == "P" else (1.0, beta)
            rollback.extend({**r, "component_rolled_back": component, "remaining_strength": beta} for r in
                            method_rows(episodes, bases["PROTECTED"], pair, f"{component}_ROLLBACK", "PROTECTED"))
    context = []
    budgets = core.PROTOCOL["erp_context_budgets_sequences" if task == "OpenBMI_ERP" else "mi_context_budgets_trials"]
    for b in budgets:
        other = subject_episodes(outer, train, task, fold, int(b), lam)
        context.extend({**r, "budget": int(b)} for r in
                       method_rows(other, bases["PROTECTED"], diff_pair, "DIFFERENTIAL_P_C", "PROTECTED"))
    q = bases["PROTECTED"]
    w = np.asarray(train["head_weight"], np.float64)
    wp = w @ q @ q.T if q.shape[1] else np.zeros_like(w)
    wc = w - wp
    margin = w[1] - w[0]
    pm = margin @ q @ q.T if q.shape[1] else np.zeros_like(margin)
    cm = margin - pm
    leverage = []
    for subject, ep in episodes.items():
        dp = analysis.projector(ep["delta"], q); dc = ep["delta"] - dp
        leverage.append({"subject": subject, "W_P_frobenius": float(np.linalg.norm(wp)),
                         "W_C_frobenius": float(np.linalg.norm(wc)),
                         "P_sensitivity_fraction": float(np.linalg.norm(wp) ** 2 / max(np.linalg.norm(w) ** 2, analysis.EPS)),
                         "C_sensitivity_fraction": float(np.linalg.norm(wc) ** 2 / max(np.linalg.norm(w) ** 2, analysis.EPS)),
                         "P_margin_operator_norm": float(np.linalg.norm(pm)),
                         "C_margin_operator_norm": float(np.linalg.norm(cm)),
                         "realized_P_logit_delta_rms": float(np.sqrt(np.mean((dp @ w.T) ** 2))),
                         "realized_C_logit_delta_rms": float(np.sqrt(np.mean((dc @ w.T) ** 2)))})
    oracle = []
    for subject, ep in episodes.items():
        candidates = []
        for pair in analysis.PAIRS:
            sc, _, _, _ = analysis.scores(ep, q, pair)
            candidates.append({"alpha_P": pair[0], "alpha_C": pair[1], **sc})
        best = analysis.choose(candidates)
        diagonal = analysis.choose(candidates, uniform_only=True)
        oracle.append({"subject": subject, "label_assisted_evaluation_only": True,
                       "best_offdiagonal_or_diagonal_alpha_P": best["alpha_P"],
                       "best_offdiagonal_or_diagonal_alpha_C": best["alpha_C"],
                       "best_BA": best["BA"], "best_uniform_BA": diagonal["BA"],
                       "oracle_offdiagonal_headroom_BA": best["BA"] - diagonal["BA"]})
    result = {"schema": "PCDA_OUTER_FOLD_V1", "model": model, "task": task, "fold": fold,
              "status": selected["status"], "proof": proof,
              "protocol_sha256": selected["protocol_sha256"], "selection_sha256": core.sha(selection_path),
              "train_cache_sha256": core.sha(train_path), "train_audit_sha256": core.sha(train_audit_path),
              "rank_P": int(q.shape[1]), "rank_C": int(dim - q.shape[1]),
              "rank_PERSIST_ALL": int(bases["PERSIST_ALL"].shape[1]),
              "source_only_budget": budget, "source_only_lambda": lam,
              "selected_alpha_P": diff_pair[0], "selected_alpha_C": diff_pair[1],
              "selected_uniform_alpha": uniform_alpha, "normmatched_uniform_alpha": norm_alpha,
              "outer_subject_count": len(episodes), "outer_subject_ids": sorted(episodes, key=int),
              "method_subject_rows": method, "random_subject_rows": random,
              "tolerance_subject_rows": tolerance, "rollback_subject_rows": rollback,
              "context_subject_rows": context, "leverage_subject_rows": leverage,
              "evaluation_only_oracle_subject_rows": oracle,
              "formal_final_heldout_eeg_reads": 0, "outer_used_for_selection": False}
    core.write_json_new(target, result)
    print("OUTER_COMPLETE", name, core.sha(target), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.task, args.fold)


if __name__ == "__main__": main()
