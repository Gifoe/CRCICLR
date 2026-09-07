"""Task-only selection and prospective H1--H5 audit for B1--B4."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common import (
    BACKBONES,
    BLOCKS,
    BOOTSTRAP_DRAWS,
    EPS,
    MODEL,
    OUT,
    PRIMARY_SEED,
    PROTOCOL,
    RANDOM_DRAWS,
    REPO_ROOT,
    RESULTS,
    audit_roles,
    balanced_accuracy_score,
    bootstrap_mean,
    ce_rows,
    centered_rms,
    clean,
    device_from_argument,
    discovered_basis,
    exact_matched_delta,
    get_or_train,
    holm,
    infer,
    local_binary_dependence,
    macro_f1_score,
    persistence_values,
    project_rows,
    random_bases,
    require_development_protocol,
    save_npz,
    sha256_file,
    sha_lines,
    signflip_p,
    softmax,
    stable_seed,
    true_margin,
    write_csv,
    write_json,
    write_once,
)
from freeze_protocol import CONFIGS


def candidate_configs(backbone: str) -> list[dict[str, Any]]:
    path = PROTOCOL / f"BACKBONE_{backbone.upper()}_TASK_SEARCH_LOCK.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    configs = [dict(value) for value in lock["configs"]]
    if lock.get("PERSIST_metrics_forbidden_during_selection") is not True or configs != CONFIGS[backbone]:
        raise RuntimeError(f"Task-search lock mismatch for {backbone}")
    return configs


def competence(backbone: str, device: torch.device, workers: int) -> dict[str, Any]:
    scope, _ = require_development_protocol()
    subjects = list(map(str, scope["allowed_subjects"]))
    folds = {int(key[1:]): list(map(str, value)) for key, value in scope["folds"].items()}
    configs = candidate_configs(backbone)
    subject_rows, checkpoint_rows = [], []
    started = time.time()
    for config in configs:
        for fold in range(5):
            outcome = folds[fold]
            train_subjects = [subject for subject in subjects if subject not in set(outcome)]
            checkpoint = MODEL / backbone / "competence" / f"{config['id']}_fold-{fold}.pt"
            model, payload = get_or_train(
                backbone,
                checkpoint,
                train_subjects,
                config,
                f"competence-fold-{fold}",
                device,
                workers,
            )
            arrays = infer(model, outcome, [2], device, workers)
            prediction = arrays["logits"].argmax(1)
            ce = ce_rows(arrays["logits"], arrays["labels"])
            for index, subject in enumerate(outcome):
                mask = arrays["subject_index"] == index
                subject_rows.append(
                    {
                        "backbone": backbone,
                        "config": config["id"],
                        "fold": fold,
                        "subject": subject,
                        "n_S3_trials": int(mask.sum()),
                        "balanced_accuracy": balanced_accuracy_score(arrays["labels"][mask], prediction[mask]),
                        "macro_f1": macro_f1_score(arrays["labels"][mask], prediction[mask]),
                        "cross_entropy": float(ce[mask].mean()),
                    }
                )
            checkpoint_rows.append(
                {
                    "backbone": backbone,
                    "config": config["id"],
                    "fold": fold,
                    "checkpoint": str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "model_state_sha256": payload["model_state_sha256"],
                    "train_subjects_hash": payload["train_subjects_hash"],
                    "representation_dim": payload["representation_dim"],
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_rows, aggregate = [], {}
    for config in configs:
        rows = [row for row in subject_rows if row["config"] == config["id"]]
        ba = np.asarray([row["balanced_accuracy"] for row in rows], np.float64)
        nll = np.asarray([row["cross_entropy"] for row in rows], np.float64)
        f1 = np.asarray([row["macro_f1"] for row in rows], np.float64)
        if len(ba) != 41:
            raise RuntimeError(f"Task search is incomplete for {backbone}/{config['id']}")
        mean, lcb, ucb = bootstrap_mean(ba, stable_seed("competence", backbone, config["id"]))
        fraction = float(np.mean(ba > 0.5))
        row = {
            "backbone": backbone,
            "config": config["id"],
            "subject_count": len(ba),
            "mean_subject_BA": mean,
            "median_subject_BA": float(np.median(ba)),
            "subject_bootstrap_CI95_L": lcb,
            "subject_bootstrap_CI95_U": ucb,
            "fraction_subject_BA_gt_0p5": fraction,
            "mean_subject_macro_F1": float(f1.mean()),
            "mean_subject_NLL": float(nll.mean()),
            "worst20_subject_BA": float(np.mean(np.sort(ba)[: max(1, math.ceil(0.2 * len(ba)))])),
            "competence_gate_pass": bool(mean >= 0.60 and lcb > 0.55 and fraction >= 0.70),
        }
        aggregate_rows.append(row)
        aggregate[config["id"]] = row
    winner_config = sorted(
        configs,
        key=lambda value: (
            -aggregate[value["id"]]["mean_subject_BA"],
            aggregate[value["id"]]["mean_subject_NLL"],
            value["id"],
        ),
    )[0]
    winner = aggregate[winner_config["id"]]
    gate_pass = bool(winner["competence_gate_pass"])
    write_csv(RESULTS / f"BACKBONE_{backbone.upper()}_TASK_SUBJECT_RESULTS.csv", subject_rows)
    write_csv(RESULTS / f"BACKBONE_{backbone.upper()}_TASK_RESULTS.csv", aggregate_rows)
    result = {
        "terminal_state": f"BACKBONE_{backbone.upper()}_COMPETENCE_{'PASS' if gate_pass else 'FAIL'}",
        "backbone": backbone,
        "selection_basis": "task-only five-fold development unseen-subject S3 mean subject BA",
        "winner": winner_config["id"],
        "winner_metrics": winner,
        "gate_pass": gate_pass,
        "outer_test_used": False,
        "actionability_used_for_selection": False,
        "elapsed_seconds": time.time() - started,
        "checkpoints": checkpoint_rows,
    }
    write_json(RESULTS / f"BACKBONE_{backbone.upper()}_COMPETENCE_RESULT.json", result)
    frozen = {
        "status": "BACKBONE_REPRESENTATION_FROZEN" if gate_pass else "BACKBONE_COMPETENCE_FAIL_FROZEN",
        "backbone": backbone,
        "config": winner_config,
        "selection_metrics": winner,
        "selection_is_task_only": True,
        "competence_gate_pass": gate_pass,
        "competence_checkpoint_set": [
            value for value in checkpoint_rows if value["config"] == winner_config["id"]
        ],
        "representation_layer": "forward_features output immediately before .head",
        "classifier_head": "single frozen linear layer",
        "actionability_based_reselection_forbidden": True,
        "outer_test_state": "OUTER_TEST_LOCKED",
    }
    write_once(PROTOCOL / f"BACKBONE_{backbone.upper()}_FROZEN.json", frozen)
    print(json.dumps(clean(result), indent=2), flush=True)
    return result


def per_subject_finite(
    base_logits: np.ndarray,
    candidate_logits: np.ndarray,
    random_logits: Sequence[np.ndarray],
    subject_index: np.ndarray,
    count: int,
) -> np.ndarray:
    output = np.empty(count, np.float64)
    for index in range(count):
        mask = subject_index == index
        candidate = centered_rms(candidate_logits[mask] - base_logits[mask])
        control = float(np.mean([centered_rms(value[mask] - base_logits[mask]) for value in random_logits]))
        output[index] = candidate / max(control, EPS)
    return output


def aggregate_local_null(
    backbone: str, block: str, fold_values: Sequence[Mapping[str, float]], rank: int
) -> dict[str, float]:
    """Convolve the exact per-fold Haar beta nulls using frozen Sobol QMC."""
    from scipy.stats import beta, qmc

    count = 2**20
    engine = qmc.Sobol(d=len(fold_values), scramble=True, seed=stable_seed("local-sobol", backbone, block))
    uniform = np.clip(engine.random_base2(m=20), 1e-12, 1 - 1e-12)
    null = np.zeros(count, np.float64)
    candidate = []
    for index, value in enumerate(fold_values):
        dim = int(value["dim"])
        total = float(value["total"])
        null += beta.ppf(uniform[:, index], rank / 2.0, (dim - rank) / 2.0) * total / rank
        candidate.append(float(value["candidate"]))
    null /= len(fold_values)
    observed = float(np.mean(candidate))
    mean_null = float(null.mean())
    q025, q975 = map(float, np.quantile(null, (0.025, 0.975)))
    return {
        "candidate": observed,
        "random_mean": mean_null,
        "ratio": observed / max(mean_null, EPS),
        "ratio_CI95_L": observed / max(q975, EPS),
        "ratio_CI95_U": observed / max(q025, EPS),
        "p_raw": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "qmc_draws": count,
    }


def audit(backbone: str, device: torch.device, workers: int) -> dict[str, Any]:
    scope, _ = require_development_protocol()
    frozen_path = PROTOCOL / f"BACKBONE_{backbone.upper()}_FROZEN.json"
    if not frozen_path.is_file():
        raise RuntimeError(f"Run task-only competence first: {backbone}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not frozen.get("competence_gate_pass"):
        result = {
            "backbone": backbone,
            "terminal_state": f"BACKBONE_{backbone.upper()}_COMPETENCE_FAIL",
            "H1_H5_not_run": True,
            "outer_test_used": False,
        }
        write_json(RESULTS / f"BACKBONE_{backbone.upper()}_AUDIT_RESULT.json", result)
        return result
    config = dict(frozen["config"])
    blocks = [name for name, _, _ in BLOCKS]
    h1_subject = {name: [] for name in blocks}
    finite_subject = {name: [] for name in blocks}
    outcome_values = {
        name: {key: [] for key in ("subject", "fold", "u_abs", "u_spec", "ba_delta", "ba_random", "ba_specific")}
        for name in blocks
    }
    local_fold_values = {name: [] for name in blocks}
    persistence_subject_rows, subject_rows, random_rows = [], [], []
    baseline_rows, basis_rows, checkpoint_rows = [], [], []
    started = time.time()
    for fold in range(5):
        outcome, discovery, model_fit = audit_roles(scope, fold)
        checkpoint = MODEL / backbone / "audit" / f"fold-{fold}.pt"
        model, payload = get_or_train(
            backbone,
            checkpoint,
            model_fit,
            config,
            f"audit-fold-{fold}",
            device,
            workers,
        )
        checkpoint_rows.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "model_state_sha256": payload["model_state_sha256"],
                "model_fit_subjects_hash": sha_lines(model_fit),
            }
        )
        discovery_arrays = infer(model, discovery, [0, 1], device, workers)
        outcome_arrays = infer(model, outcome, [2], device, workers)
        embedding_path = OUT / "cache" / backbone / "fold_embeddings" / f"fold-{fold}.npz"
        save_npz(
            embedding_path,
            discovery_embeddings=discovery_arrays["embeddings"].astype(np.float16),
            discovery_logits=discovery_arrays["logits"],
            discovery_labels=discovery_arrays["labels"],
            discovery_subject_index=discovery_arrays["subject_index"],
            discovery_session=discovery_arrays["session"],
            discovery_subjects=discovery_arrays["subjects"],
            outcome_embeddings=outcome_arrays["embeddings"].astype(np.float16),
            outcome_logits=outcome_arrays["logits"],
            outcome_labels=outcome_arrays["labels"],
            outcome_subject_index=outcome_arrays["subject_index"],
            outcome_session=outcome_arrays["session"],
            outcome_subjects=outcome_arrays["subjects"],
        )
        basis_all, eigenvalues, center, session_means, basis_diag = discovered_basis(discovery_arrays)
        basis_path = MODEL / backbone / "audit" / f"persistent_basis_fold-{fold}.npz"
        save_npz(
            basis_path,
            basis=basis_all.astype(np.float32),
            eigenvalues=eigenvalues.astype(np.float64),
            center=center.astype(np.float32),
            session_means=np.stack([session_means[0], session_means[1]]).astype(np.float32),
            discovery_subjects=np.asarray(discovery),
        )
        weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
        outcome_h = outcome_arrays["embeddings"].astype(np.float64)
        outcome_z = outcome_arrays["logits"].astype(np.float64)
        outcome_y = outcome_arrays["labels"].astype(int)
        outcome_sid = outcome_arrays["subject_index"].astype(int)
        discovery_h = discovery_arrays["embeddings"].astype(np.float64)
        discovery_z = discovery_arrays["logits"].astype(np.float64)
        discovery_sid = discovery_arrays["subject_index"].astype(int)
        base_ce = ce_rows(outcome_z, outcome_y)
        base_pred, base_prob = outcome_z.argmax(1), softmax(outcome_z)
        for index, subject in enumerate(outcome):
            mask = outcome_sid == index
            baseline_rows.append(
                {
                    "backbone": backbone,
                    "fold": fold,
                    "subject": subject,
                    "n_S3_trials": int(mask.sum()),
                    "baseline_BA": balanced_accuracy_score(outcome_y[mask], base_pred[mask]),
                    "baseline_CE": float(base_ce[mask].mean()),
                }
            )

        dim = outcome_h.shape[1]
        for block_name, start, end in BLOCKS:
            block, rank = basis_all[:, start:end], end - start
            random = random_bases(dim, rank, backbone, fold, block_name)
            candidate_persistence, random_persistence = persistence_values(
                discovery_arrays, block, random, session_means
            )
            persistence_specific = candidate_persistence - random_persistence.mean(axis=0)
            h1_subject[block_name].extend(persistence_specific.tolist())
            for subject_index, subject in enumerate(discovery):
                persistence_subject_rows.append(
                    {
                        "backbone": backbone,
                        "fold": fold,
                        "block": block_name,
                        "subject": subject,
                        "candidate_persistence": candidate_persistence[subject_index],
                        "random_persistence_mean": random_persistence[:, subject_index].mean(),
                        "persistence_specific": persistence_specific[subject_index],
                    }
                )

            discovery_delta = project_rows(discovery_h, block)
            discovery_candidate_z = discovery_z - discovery_delta @ weight.T
            discovery_random_z = [
                discovery_z - exact_matched_delta(discovery_h, discovery_delta, value) @ weight.T
                for value in random
            ]
            finite = per_subject_finite(
                discovery_z, discovery_candidate_z, discovery_random_z, discovery_sid, len(discovery)
            )
            finite_subject[block_name].extend(finite.tolist())
            local = local_binary_dependence(weight, block)
            local_fold_values[block_name].append(
                {
                    "fold": fold,
                    "candidate": local["candidate"],
                    "total": float(np.sum((weight[1] - weight[0]) ** 2)),
                    "dim": dim,
                    "fold_ratio": local["ratio"],
                    "fold_p": local["p_raw"],
                }
            )

            target_delta = project_rows(outcome_h, block)
            candidate_z = outcome_z - target_delta @ weight.T
            candidate_ce = ce_rows(candidate_z, outcome_y)
            candidate_pred, candidate_prob = candidate_z.argmax(1), softmax(candidate_z)
            random_z = [
                outcome_z - exact_matched_delta(outcome_h, target_delta, value) @ weight.T
                for value in random
            ]
            random_ce = np.stack([ce_rows(value, outcome_y) for value in random_z])
            for subject_index, subject in enumerate(outcome):
                mask = outcome_sid == subject_index
                base_ba = balanced_accuracy_score(outcome_y[mask], base_pred[mask])
                candidate_ba = balanced_accuracy_score(outcome_y[mask], candidate_pred[mask])
                random_ba = np.asarray(
                    [balanced_accuracy_score(outcome_y[mask], value.argmax(1)[mask]) for value in random_z]
                )
                u_abs = float(np.mean(candidate_ce[mask] - base_ce[mask]))
                u_random = float(np.mean(random_ce[:, mask] - base_ce[None, mask]))
                ba_delta = candidate_ba - base_ba
                ba_random = float(random_ba.mean() - base_ba)
                values = outcome_values[block_name]
                values["subject"].append(subject)
                values["fold"].append(fold)
                values["u_abs"].append(u_abs)
                values["u_spec"].append(u_abs - u_random)
                values["ba_delta"].append(ba_delta)
                values["ba_random"].append(ba_random)
                values["ba_specific"].append(ba_delta - ba_random)
                subject_rows.append(
                    {
                        "backbone": backbone,
                        "fold": fold,
                        "block": block_name,
                        "subject": subject,
                        "base_BA": base_ba,
                        "candidate_BA": candidate_ba,
                        "random_BA_mean": float(random_ba.mean()),
                        "u_abs": u_abs,
                        "u_random": u_random,
                        "u_spec": u_abs - u_random,
                        "delta_BA": ba_delta,
                        "delta_BA_random": ba_random,
                        "delta_BA_specific": ba_delta - ba_random,
                        "logit_RMS": centered_rms(candidate_z[mask] - outcome_z[mask]),
                        "margin_displacement": float(np.mean(np.abs(true_margin(candidate_z[mask], outcome_y[mask]) - true_margin(outcome_z[mask], outcome_y[mask])))),
                        "prediction_flip_rate": float(np.mean(candidate_pred[mask] != base_pred[mask])),
                        "total_variation": float(np.mean(0.5 * np.sum(np.abs(candidate_prob[mask] - base_prob[mask]), axis=1))),
                    }
                )
                for draw in range(RANDOM_DRAWS):
                    random_rows.append(
                        {
                            "backbone": backbone,
                            "fold": fold,
                            "block": block_name,
                            "subject": subject,
                            "draw": draw,
                            "u_random": float(np.mean(random_ce[draw, mask] - base_ce[mask])),
                            "delta_BA_random": float(random_ba[draw] - base_ba),
                            "finite_logit_RMS": centered_rms(random_z[draw][mask] - outcome_z[mask]),
                        }
                    )
            basis_rows.append(
                {
                    "backbone": backbone,
                    "fold": fold,
                    "block": block_name,
                    "rank": rank,
                    "eigenvalue_sum": float(eigenvalues[start:end].sum()),
                    "minimum_eigenvalue": float(eigenvalues[start:end].min()),
                    "basis_sha256": sha256_file(basis_path),
                    "embedding_sha256": sha256_file(embedding_path),
                    **basis_diag,
                }
            )
            print(f"[audit {backbone} fold={fold}] {block_name} complete", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(len(value) != 41 for value in h1_subject.values()) or any(
        len(value) != 41 for value in finite_subject.values()
    ) or any(len(value["subject"]) != 41 for value in outcome_values.values()):
        raise RuntimeError(f"Incomplete subject cross-fitting for {backbone}")

    p_raw = {family: {} for family in ("H1", "H2", "H3_finite", "H4", "protected")}
    persistence_rows, utility_rows, decision_rows, actionability_rows = [], [], [], []
    block_cache = {}
    for block_name, start, end in BLOCKS:
        persistence = np.asarray(h1_subject[block_name], np.float64)
        finite = np.asarray(finite_subject[block_name], np.float64)
        outcome = outcome_values[block_name]
        u_abs = np.asarray(outcome["u_abs"], np.float64)
        u_spec = np.asarray(outcome["u_spec"], np.float64)
        ba_delta = np.asarray(outcome["ba_delta"], np.float64)
        ba_random = np.asarray(outcome["ba_random"], np.float64)
        ba_specific = np.asarray(outcome["ba_specific"], np.float64)
        summaries = {
            "persistence": bootstrap_mean(persistence, stable_seed(backbone, block_name, "H1")),
            "u_abs": bootstrap_mean(u_abs, stable_seed(backbone, block_name, "u_abs")),
            "u": bootstrap_mean(u_spec, stable_seed(backbone, block_name, "H2")),
            "finite": bootstrap_mean(finite, stable_seed(backbone, block_name, "H3")),
            "ba": bootstrap_mean(ba_specific, stable_seed(backbone, block_name, "H4")),
        }
        p_raw["H1"][block_name] = signflip_p(persistence, "positive")
        p_raw["H2"][block_name] = signflip_p(u_spec, "negative")
        p_raw["protected"][block_name] = signflip_p(u_spec, "positive")
        p_raw["H3_finite"][block_name] = signflip_p(np.log(np.maximum(finite, EPS)), "positive")
        p_raw["H4"][block_name] = signflip_p(ba_specific, "positive")
        local = aggregate_local_null(backbone, block_name, local_fold_values[block_name], end - start)
        folds = np.asarray(outcome["fold"], int)
        loso = [float(np.delete(ba_specific, index).mean()) for index in range(len(ba_specific))]
        lofo = [float(ba_specific[folds != fold].mean()) for fold in range(5)]
        nonnegative = float(np.mean(ba_specific >= 0))
        stability = bool(min(loso) > 0 and min(lofo) > 0 and nonnegative >= 0.60)
        persistence_rows.append(
            {"backbone": backbone, "block": block_name, "rank": end - start, "mean_specific_advantage": summaries["persistence"][0], "CI95_L": summaries["persistence"][1], "CI95_U": summaries["persistence"][2], "p_raw": p_raw["H1"][block_name]}
        )
        utility_rows.append(
            {"backbone": backbone, "block": block_name, "rank": end - start, "u_abs_mean": summaries["u_abs"][0], "u_abs_CI95_L": summaries["u_abs"][1], "u_abs_CI95_U": summaries["u_abs"][2], "u_spec_mean": summaries["u"][0], "u_spec_CI95_L": summaries["u"][1], "u_spec_CI95_U": summaries["u"][2], "p_raw_harmful": p_raw["H2"][block_name], "p_raw_protected": p_raw["protected"][block_name]}
        )
        decision_rows.append(
            {"backbone": backbone, "block": block_name, "rank": end - start, "local_energy": local["candidate"], "local_random_mean": local["random_mean"], "local_ratio": local["ratio"], "local_ratio_CI95_L": local["ratio_CI95_L"], "local_ratio_CI95_U": local["ratio_CI95_U"], "local_p_raw": local["p_raw"], "local_qmc_draws": local["qmc_draws"], "finite_ratio_mean": summaries["finite"][0], "finite_ratio_CI95_L": summaries["finite"][1], "finite_ratio_CI95_U": summaries["finite"][2], "finite_p_raw": p_raw["H3_finite"][block_name]}
        )
        actionability_rows.append(
            {"backbone": backbone, "block": block_name, "rank": end - start, "delta_BA_mean": float(ba_delta.mean()), "delta_BA_random_mean": float(ba_random.mean()), "delta_BA_specific_mean": summaries["ba"][0], "delta_BA_specific_CI95_L": summaries["ba"][1], "delta_BA_specific_CI95_U": summaries["ba"][2], "p_raw": p_raw["H4"][block_name], "minimum_LOSO_mean": min(loso), "minimum_leave_one_fold_out_mean": min(lofo), "nonnegative_subject_fraction": nonnegative, "median_specific": float(np.median(ba_specific)), "worst_subject_specific": float(np.min(ba_specific)), "worst20_subject_specific": float(np.mean(np.sort(ba_specific)[: math.ceil(0.2 * len(ba_specific))])), "stability_preliminary": stability}
        )
        block_cache[block_name] = {**summaries, "local": local, "stability": stability}

    adjusted = {family: holm(values) for family, values in p_raw.items()}
    for row in persistence_rows:
        row["p_holm"] = adjusted["H1"][row["block"]]
    for row in utility_rows:
        row["p_holm_harmful"] = adjusted["H2"][row["block"]]
        row["p_holm_protected"] = adjusted["protected"][row["block"]]
    for row in decision_rows:
        row["finite_p_holm"] = adjusted["H3_finite"][row["block"]]
    for row in actionability_rows:
        row["p_holm"] = adjusted["H4"][row["block"]]

    assignments = []
    for block_name, start, end in BLOCKS:
        value = block_cache[block_name]
        h1 = bool(value["persistence"][1] > 0 and adjusted["H1"][block_name] < 0.05)
        h2 = bool(value["u"][2] < 0 and adjusted["H2"][block_name] < 0.05)
        h3 = bool(value["local"]["ratio_CI95_L"] > 1 and value["local"]["p_raw"] < 0.05 and value["finite"][1] > 1 and adjusted["H3_finite"][block_name] < 0.05)
        h4 = bool(value["ba"][1] > 0 and value["ba"][0] >= 0.005 and adjusted["H4"][block_name] < 0.05)
        h5 = bool(value["stability"])
        protected = bool(h1 and h3 and value["u"][1] > 0 and adjusted["protected"][block_name] < 0.05)
        preliminary = bool(h1 and h2 and h3 and h4 and h5)
        if preliminary:
            assignment, action = "PRELIMINARY_ACTIONABLE-HARMFUL", "AWAIT_GLOBAL_MULTIPLICITY"
        elif protected:
            assignment, action = "PROTECTED", "PRESERVE"
        elif h1 and not h3:
            assignment, action = "DECISION-NULL / WEAKLY ACTIVE", "NO_OP"
        elif h1 and h3:
            assignment, action = "DECISION-ACTIVE NON-ACTIONABLE", "NO_OP"
        else:
            assignment, action = "UNCERTAIN", "NO_OP"
        assignments.append(
            {
                "backbone": backbone,
                "block": block_name,
                "rank": end - start,
                "H1": h1,
                "H2": h2,
                "H3": h3,
                "H4": h4,
                "H5": h5,
                "preliminary_all_H1_H5": preliminary,
                "protected_utility_gate": protected,
                "p_H1_raw": p_raw["H1"][block_name],
                "p_H2_raw": p_raw["H2"][block_name],
                "p_H3_local_raw": value["local"]["p_raw"],
                "p_H3_finite_raw": p_raw["H3_finite"][block_name],
                "p_H4_raw": p_raw["H4"][block_name],
                "p_joint_raw": max(p_raw["H1"][block_name], p_raw["H2"][block_name], value["local"]["p_raw"], p_raw["H3_finite"][block_name], p_raw["H4"][block_name]),
                "assignment": assignment,
                "action": action,
            }
        )

    prefix = f"BACKBONE_{backbone.upper()}"
    write_csv(RESULTS / f"{prefix}_PERSISTENCE_RESULTS.csv", persistence_rows)
    write_csv(RESULTS / f"{prefix}_SIGNED_UTILITY_RESULTS.csv", utility_rows)
    write_csv(RESULTS / f"{prefix}_DECISION_DEPENDENCE_RESULTS.csv", decision_rows)
    write_csv(RESULTS / f"{prefix}_ACTIONABILITY_RESULTS.csv", actionability_rows)
    write_csv(RESULTS / f"{prefix}_BLOCK_ASSIGNMENTS.csv", assignments)
    write_csv(RESULTS / f"{prefix}_AUDIT_SUBJECT_RESULTS.csv", subject_rows)
    write_csv(RESULTS / f"{prefix}_AUDIT_RANDOM_SUBJECT_RESULTS.csv", random_rows)
    write_csv(RESULTS / f"{prefix}_PERSISTENCE_SUBJECT_RESULTS.csv", persistence_subject_rows)
    write_csv(RESULTS / f"{prefix}_AUDIT_BASELINE_SUBJECT_RESULTS.csv", baseline_rows)
    write_csv(RESULTS / f"{prefix}_BASIS_RESULTS.csv", basis_rows)
    result = {
        "backbone": backbone,
        "terminal_state": f"BACKBONE_{backbone.upper()}_AUDIT_COMPLETE_AWAIT_GLOBAL_MULTIPLICITY",
        "competence": frozen["selection_metrics"],
        "baseline_mean_subject_BA": float(np.mean([row["baseline_BA"] for row in baseline_rows])),
        "preliminary_H1_H5_blocks": [row["block"] for row in assignments if row["preliminary_all_H1_H5"]],
        "protected_blocks": [row["block"] for row in assignments if row["protected_utility_gate"]],
        "assignments": assignments,
        "outer_test_used": False,
        "checkpoints": checkpoint_rows,
        "elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS / f"{prefix}_AUDIT_RESULT.json", result)
    print(json.dumps(clean(result), indent=2), flush=True)
    return result


def run_all(device: torch.device, workers: int) -> None:
    for backbone in BACKBONES:
        result = competence(backbone, device, workers)
        if result["gate_pass"]:
            audit(backbone, device, workers)
        else:
            audit(backbone, device, workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("competence", "audit", "all"))
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    device = device_from_argument(args.device)
    if args.stage == "all":
        if args.backbone:
            result = competence(args.backbone, device, args.workers)
            audit(args.backbone, device, args.workers)
        else:
            run_all(device, args.workers)
        return
    if not args.backbone:
        parser.error("--backbone is required for competence/audit")
    if args.stage == "competence":
        competence(args.backbone, device, args.workers)
    else:
        audit(args.backbone, device, args.workers)


if __name__ == "__main__":
    main()
