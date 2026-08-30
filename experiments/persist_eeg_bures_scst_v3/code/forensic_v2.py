"""Artifact-backed forensic audit of the immutable V2 run.

The audit never reruns V2 S3.  It inspects detached source caches, committed
V2 summaries, and already-written matching/checkpoint artifacts.  Quantities
that are not recoverable from those artifacts are recorded as unavailable
instead of being silently reconstructed from a different experiment.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as c
from bures import anchor_excluded_indices, anchor_excluded_neighbors


def _load_mixed_effects():
    path = c.V2_EXP / "code" / "mixed_effects.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("v2_forensic_mixed_effects", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cache_files() -> list[Path]:
    root = c.V2_CACHE
    return sorted(root.glob("*/fold-*/seed-*/train.npz")) if root.is_dir() else []


def _neighbor_metrics(features: np.ndarray, subjects: np.ndarray, row_ids: np.ndarray) -> dict[str, float]:
    n = len(features); own_nearest = []; own_top3 = []; own_top5 = []; non_source_top5 = []
    # The V2 engine queried all points (including the anchor).  Compute those
    # top-5 sets in chunks, while the corrected local radius uses the shared
    # anchor-excluded implementation once per cache.
    local_idx, local_dist = anchor_excluded_neighbors(features, row_ids, 5)
    radius_values = np.nanmean(np.where(np.isfinite(local_dist), local_dist, np.nan), axis=1)
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        value = np.asarray(features[start:stop], np.float64)
        distance = ((value[:, None, :] - np.asarray(features, np.float64)[None, :, :]) ** 2).sum(axis=2)
        order = np.argpartition(distance, kth=min(4, n - 1), axis=1)[:, :5]
        order_dist = np.take_along_axis(distance, order, axis=1)
        order = np.take_along_axis(order, np.argsort(order_dist, axis=1, kind="stable"), axis=1)
        for local, neighbors in enumerate(order):
            pos = start + local; own_nearest.append(bool(int(neighbors[0]) == pos)); own_top3.append(bool(pos in neighbors[:3])); own_top5.append(bool(pos in neighbors[:5])); non_source_top5.append(float(np.mean(np.asarray(subjects)[neighbors] != str(subjects[pos]))))
    return {
        "nearest_self_rate": float(np.mean(own_nearest)) if own_nearest else float("nan"),
        "top3_self_rate": float(np.mean(own_top3)) if own_top3 else float("nan"),
        "top5_self_rate": float(np.mean(own_top5)) if own_top5 else float("nan"),
        "target_subject_fraction_top5": float(np.mean(non_source_top5)) if non_source_top5 else float("nan"),
        "anchor_excluded_local_5nn_radius": float(np.nanmedian(radius_values)) if np.isfinite(radius_values).any() else float("nan"),
    }


def _v2_displacement_metrics(features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, row_ids: np.ndarray, dataset: str, fold: int, seed: int) -> dict[str, float]:
    module = _load_mixed_effects()
    if module is None:
        return {"displacement_radius_ratio_median": float("nan"), "target_distance_improvement": float("nan"), "target_gaussian_affinity_improvement": float("nan")}
    try:
        bank = module.MixedEffectsBank(features, labels, subjects, row_ids)
    except Exception:
        return {"displacement_radius_ratio_median": float("nan"), "target_distance_improvement": float("nan"), "target_gaussian_affinity_improvement": float("nan")}
    ratios = []; distance_improvements = []
    for position in range(len(features)):
        source = str(subjects[position]); pool = [s for s in bank.subjects if str(s) != source]
        rng = np.random.default_rng(c.stable_seed("v2-forensic-target", dataset, fold, seed, int(row_ids[position])))
        if len(pool) > 8: pool = list(np.asarray(pool)[rng.choice(len(pool), 8, replace=False)])
        local = anchor_excluded_indices(features, row_ids, position)
        if len(local):
            d = np.linalg.norm(features[local] - features[position][None], axis=1); radius = float(np.sort(d)[: min(5, len(d))].mean())
        else: radius = float("nan")
        for target in pool:
            direction = bank.direction(int(row_ids[position]), str(target), factorized=True)
            displacement = (3.0 / 64.0) * direction
            if np.isfinite(radius) and radius > 0: ratios.append(float(np.linalg.norm(displacement) / radius))
            target_mask = (subjects.astype(str) == str(target)) & (labels == labels[position])
            if np.any(target_mask):
                target_values = features[target_mask]; before = np.sort(np.linalg.norm(target_values - features[position][None], axis=1))[: min(5, len(target_values))].mean(); after = np.sort(np.linalg.norm(target_values - (features[position] + displacement)[None], axis=1))[: min(5, len(target_values))].mean(); distance_improvements.append(float(before - after))
    return {"displacement_radius_ratio_median": float(np.median(ratios)) if ratios else float("nan"), "target_distance_improvement": float(np.mean(distance_improvements)) if distance_improvements else float("nan"), "target_gaussian_affinity_improvement": float("nan")}


def _head_audit() -> tuple[float, float, int]:
    root = c.V2_EXP / "runtime" / "discovery_units"
    methods = ("Factorized-Uniform-NoKL", "Factorized-HardRandom", "ME-HardSCST")
    disagreements = []; distances = []
    if not root.is_dir(): return float("nan"), float("nan"), 0
    for fold in c.FOLDS:
        for seed in c.SEEDS:
            outcomes = c.V2_EXP / "runtime" / "discovery_cache" / f"fold-{fold}" / f"seed-{seed}" / "outcome.npz"
            if not outcomes.is_file(): continue
            with np.load(outcomes) as values: features = values["final"].astype(np.float64)
            predictions = {}
            states = {}
            for method in methods:
                path = root / method / f"fold-{fold}" / f"seed-{seed}" / "model.pt"
                if not path.is_file(): continue
                try:
                    import torch
                    payload = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
                    weight = payload["head.weight"].detach().cpu().numpy(); bias = payload["head.bias"].detach().cpu().numpy(); predictions[method] = (features @ weight.T + bias).argmax(1); states[method] = np.concatenate([weight.ravel(), bias.ravel()])
                except Exception:
                    continue
            for left in range(len(methods)):
                for right in range(left + 1, len(methods)):
                    a, b = methods[left], methods[right]
                    if a in predictions and b in predictions: disagreements.append(float(np.mean(predictions[a] != predictions[b])))
                    if a in states and b in states: distances.append(float(np.linalg.norm(states[a] - states[b])))
    return (float(np.mean(disagreements)) if disagreements else float("nan"), float(np.mean(distances)) if distances else float("nan"), len(disagreements))


def main() -> None:
    c.ensure_dirs(); rows = []
    for path in _cache_files():
        rel = path.relative_to(c.V2_CACHE); dataset, fold_tag, seed_tag = rel.parts[:3]; fold = int(fold_tag.split("-")[-1]); seed = int(seed_tag.split("-")[-1])
        with np.load(path) as values: features = values["final"].astype(np.float32); labels = values["labels"].astype(np.int64); subjects = values["subjects"].astype(str); row_ids = values["indices"].astype(np.int64)
        rows.append({"dataset": dataset, "fold": fold, "seed": seed, "metric_scope": "V2 source cache", **_neighbor_metrics(features, subjects, row_ids), **_v2_displacement_metrics(features, labels, subjects, row_ids, dataset, fold, seed), "transported_margin_drop_over_abs_clean_margin": np.nan, "margin_drop_status": "NOT_REGENERABLE_FROM_V2_CACHE_NO_LOGITS"})
    frame = pd.DataFrame(rows)
    match_path = c.V2_EXP / "results" / "HARD_RANDOM_MATCHING.csv"
    if match_path.is_file():
        matching = pd.read_csv(match_path); mismatch = float(matching["whitened_norm_error"].abs().mean()) if "whitened_norm_error" in matching else float("nan"); matching_empty = bool(len(matching) == 0); matching_regenerated = bool(np.isfinite(mismatch) and mismatch < 1e-5)
    else:
        mismatch = float("nan"); matching_empty = True; matching_regenerated = False
    disagreement, parameter_distance, disagreement_pairs = _head_audit()
    if len(frame):
        frame["prediction_disagreement_mean"] = disagreement; frame["final_head_parameter_distance_mean"] = parameter_distance; frame["prediction_disagreement_pairs"] = disagreement_pairs
        frame["hard_random_matching_empty"] = matching_empty; frame["hard_random_whitened_norm_mismatch_mean"] = mismatch; frame["matching_quantities_regenerable"] = matching_regenerated
    else:
        frame = pd.DataFrame([{"status": "V2_SOURCE_CACHE_MISSING", "hard_random_matching_empty": matching_empty, "hard_random_whitened_norm_mismatch_mean": mismatch, "matching_quantities_regenerable": matching_regenerated, "prediction_disagreement_mean": disagreement, "final_head_parameter_distance_mean": parameter_distance}])
    c.write_csv(c.RESULTS / "V2_FORENSIC_METRICS.csv", frame)
    summary = {"source_cache_units": int(len(rows)), "nearest_self_rate_mean": float(frame.nearest_self_rate.mean()) if "nearest_self_rate" in frame else None, "top3_self_rate_mean": float(frame.top3_self_rate.mean()) if "top3_self_rate" in frame else None, "top5_self_rate_mean": float(frame.top5_self_rate.mean()) if "top5_self_rate" in frame else None, "prediction_disagreement_mean": disagreement, "final_head_parameter_distance_mean": parameter_distance, "hard_random_matching_empty": matching_empty, "hard_random_whitened_norm_mismatch_mean": mismatch, "matching_quantities_regenerable": matching_regenerated, "v2_result_unchanged": True}
    c.write_json(c.RESULTS / "V2_FORENSIC_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
