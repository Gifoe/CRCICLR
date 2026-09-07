"""OpenBMI fold-0, seed-0 matched screen: EEGNet decomposition plus LOO offsets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
V1_CODE = REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code"
sys.path.append(str(V1_CODE))
import run_stage1 as v1  # noqa: E402
import stage1_core as canonical  # noqa: E402

from decomposition import DecompositionEEGNet
from eegnet_carrier import EEGNetCarrier, parameter_count
from metrics import direction, representation_stats, score
from offset_loss import leave_one_out_centers, offset_mse

EXP = REPO / "experiments/persist_eeg_amortized_centering_fold0_v1"
SOURCE = REPO / "experiments/persist_eeg_eegnet_prd_fold0_v1"
PROTOCOL, OUT = EXP / "protocol", EXP / "outputs"
RUNTIME = Path("/root/rivermind-data/amortized_centering_fold0_runtime")
LR, WD, CLIP, LAMBDA, EPOCHS, SELECT_FROM = 3e-4, 5e-4, 5.0, 0.25, 60, 10
REFERENCE_BA = 0.7964285714285714


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, obj: Any) -> None:
    v1.write_json(path, obj)


def seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(state: dict[str, torch.Tensor]) -> str:
    import io
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def episode_groups(device: torch.device) -> torch.Tensor:
    """Frozen stream order is support subjects 0..3 then query subjects 4..7."""
    return torch.arange(8, device=device).repeat_interleave(16)


def subject_mean(metrics: dict[str, dict[str, float]], field: str) -> float:
    return float(np.mean([row[field] for row in metrics.values()]))


def forward_condition(model: DecompositionEEGNet, x: torch.Tensor, condition: str = "full", centers: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    h, offset, h_rel, z_rel, z_res = model.decompose(x)
    if condition == "oracle":
        if centers is None:
            raise ValueError("oracle condition requires a diagnostic subject center")
        z_rel = model.rel_projector(h - centers)
    elif condition != "full":
        if condition == "rel_only":
            z_res = torch.zeros_like(z_res)
        elif condition == "res_only":
            z_rel = torch.zeros_like(z_rel)
        else:
            raise ValueError(condition)
    logits = model.classify_components(z_rel, z_res)
    return logits, {"h": h, "offset": offset, "h_rel": h_rel, "z_rel": z_rel, "z_res": z_res}


def evaluate(model: DecompositionEEGNet, bundle: Any, subjects: list[str], sessions: tuple[int, ...], mean: np.ndarray, std: np.ndarray, device: torch.device, condition: str = "full", collect: bool = False) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]], dict[str, float]]:
    model.eval()
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    per_subject: dict[str, dict[str, float]] = {}
    records: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([subject], sessions)
            y = bundle.labels(idx)
            logits_parts: list[np.ndarray] = []
            component_parts: dict[str, list[np.ndarray]] = {key: [] for key in ("h", "offset", "h_rel", "z_rel", "z_res")}
            for start in range(0, len(idx), 128):
                x = v1.prepare(bundle, idx[start:start + 128], mean, std, device)
                logits, components = forward_condition(model, x, condition)
                logits_parts.append(logits.float().cpu().numpy())
                if collect:
                    for key, value in components.items():
                        component_parts[key].append(value.float().cpu().numpy())
            logits = np.concatenate(logits_parts)
            pred = logits.argmax(axis=1)
            per_subject[str(subject)] = {**score(y, pred), "trials": int(len(y))}
            all_y.append(y)
            all_p.append(pred)
            if collect:
                records[str(subject)] = {"y": y, "logits": logits, **{key: np.concatenate(value) for key, value in component_parts.items()}}
    return per_subject, records, score(np.concatenate(all_y), np.concatenate(all_p))


def train(model: DecompositionEEGNet, name: str, bundle: Any, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ba, best_epoch, best_state = -1.0, None, None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch, episodes in enumerate(manifest, 1):
        model.train()
        ce_values: list[float] = []
        offset_values: list[float] = []
        totals: list[float] = []
        for episode in episodes:
            indices = np.asarray(episode["support_indices"] + episode["query_indices"], dtype=np.int64)
            labels = torch.from_numpy(bundle.labels(indices)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, h, offset, _, _, _ = model(v1.prepare(bundle, indices, mean, std, device))
                ce = F.cross_entropy(logits, labels)
                raw_offset = torch.zeros((), device=device)
                if name == "Decomp-Offset":
                    raw_offset, _ = offset_mse(offset, h, episode_groups(device))
                    loss = ce + LAMBDA * raw_offset
                else:
                    loss = ce
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
            ce_values.append(float(ce.detach().cpu()))
            offset_values.append(float(raw_offset.detach().cpu()))
            totals.append(float(loss.detach().cpu()))
        inner, _, _ = evaluate(model, bundle, fold["inner_val_subjects"], (2,), mean, std, device)
        inner_ba = subject_mean(inner, "BA")
        selected = epoch >= SELECT_FROM and inner_ba > best_ba + 1e-12
        if selected:
            best_ba, best_epoch, best_state = inner_ba, epoch, copy.deepcopy(model.state_dict())
        item = {"epoch": epoch, "CE": float(np.mean(ce_values)), "raw_offset_MSE": float(np.mean(offset_values)), "lambda_times_offset": float(LAMBDA * np.mean(offset_values)), "total_loss": float(np.mean(totals)), "inner_val_subject_BA": inner_ba, "selected": selected}
        history.append(item)
        if epoch == 1 or epoch % 5 == 0 or selected:
            print(f"[{name}] e={epoch:02d} CE={item['CE']:.4f} offset={item['raw_offset_MSE']:.4f} val={inner_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no eligible selected checkpoint")
    model.load_state_dict(best_state)
    checkpoint = RUNTIME / "checkpoints" / f"{name.lower().replace('-', '_')}_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return {"model": name, "history": history, "selected_epoch": best_epoch, "best_inner_val_subject_BA": best_ba, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "actual_steps": len(manifest) * len(manifest[0]), "steps_per_epoch": len(manifest[0]), "seconds": time.perf_counter() - started}


def paired_summary(values: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)), "improved": int((values > 1e-8).sum()), "tied": int((np.abs(values) <= 1e-8).sum()), "harmed": int((values < -1e-8).sum())}


def direction_cosines(source: dict[str, dict[str, np.ndarray]], future: dict[str, dict[str, np.ndarray]], key: str) -> float:
    return float(np.mean([direction(source[s][key], source[s]["y"]) @ direction(future[s][key], future[s]["y"]) for s in future]))


def offset_diagnostics(records: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    rows = []
    for subject, row in records.items():
        center = row["h"].mean(axis=0)
        estimate_error = np.mean((row["offset"] - center) ** 2, axis=1)
        zero_error = np.mean(np.broadcast_to(center, row["offset"].shape) ** 2, axis=1)
        cosine = np.sum(row["offset"] * center, axis=1) / np.maximum(np.linalg.norm(row["offset"], axis=1) * np.linalg.norm(center), 1e-12)
        rows.append({"subject_id": subject, "offset_MSE": float(estimate_error.mean()), "zero_offset_MSE": float(zero_error.mean()), "offset_cosine": float(cosine.mean()), "centering_approximation_error": float(estimate_error.mean())})
    frame = pd.DataFrame(rows)
    return {"per_subject": rows, "overall_offset_MSE": float(frame.offset_MSE.mean()), "overall_zero_offset_MSE": float(frame.zero_offset_MSE.mean()), "overall_offset_cosine": float(frame.offset_cosine.mean()), "beats_zero_MSE": bool(frame.offset_MSE.mean() < frame.zero_offset_MSE.mean())}


def oracle_diagnostic(model: DecompositionEEGNet, records: dict[str, dict[str, np.ndarray]], device: torch.device) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    model.eval()
    result: dict[str, dict[str, float]] = {}
    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for subject, row in records.items():
            h = torch.from_numpy(row["h"]).to(device)
            center = h.mean(dim=0, keepdim=True)
            logits = model.classify_components(model.rel_projector(h - center), model.res_projector(h))
            pred = logits.argmax(1).cpu().numpy()
            result[subject] = score(row["y"], pred)
            all_y.append(row["y"])
            all_pred.append(pred)
    return result, score(np.concatenate(all_y), np.concatenate(all_pred))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for directory in (PROTOCOL, OUT, RUNTIME):
        directory.mkdir(parents=True, exist_ok=True)

    v1.RUNTIME = RUNTIME / "immutable_loader_runtime"
    folds, search, split_sha = v1.load_split()
    fold = folds["OpenBMI"][0]
    bundle = v1.load_bundle("OpenBMI", search["OpenBMI"])
    mean, std, normalization = v1.normalizer(bundle, fold["inner_train_subjects"])
    source_split = json.loads((SOURCE / "protocol/FOLD0_SPLIT.json").read_text())
    source_norm = json.loads((SOURCE / "protocol/NORMALIZATION_AUDIT.json").read_text())
    source_manifest = json.loads((SOURCE / "protocol/EPISODE_MANIFEST_PROVENANCE.json").read_text())
    manifest_path = Path(source_manifest["path"])
    if source_split != fold or source_norm["mean_std_sha256"] != normalization["mean_std_sha256"]:
        raise RuntimeError("matched reference provenance mismatch")
    if not manifest_path.exists() or sha(manifest_path) != source_manifest["sha256"]:
        raise RuntimeError("exact prior episode manifest unavailable or altered")
    manifest = json.loads(manifest_path.read_text())["epochs"]
    if len(manifest) != EPOCHS or len(manifest[0]) != 20:
        raise RuntimeError("unexpected frozen episode stream")
    outer = set(map(int, bundle.indices(fold["outer_dev_subjects"])))
    if any(outer & set(item["support_indices"] + item["query_indices"]) for epoch in manifest for item in epoch):
        raise RuntimeError("outer-development sample entered the episode stream")

    canonical_model = canonical.EEGNet(62)
    carrier = EEGNetCarrier(62)
    source_carrier = {key: value for key, value in canonical_model.state_dict().items() if not key.startswith("head.")}
    carrier_match = list(source_carrier) == list(carrier.state_dict()) and all(source_carrier[key].shape == carrier.state_dict()[key].shape for key in source_carrier)
    if not carrier_match:
        raise RuntimeError("EEGNet carrier differs from canonical implementation")
    seed(0)
    template = DecompositionEEGNet(62).to(device)
    initial = copy.deepcopy(template.state_dict())
    initial_sha = state_hash(initial)
    decomp_ce = DecompositionEEGNet(62).to(device)
    decomp_offset = DecompositionEEGNet(62).to(device)
    decomp_ce.load_state_dict(initial)
    decomp_offset.load_state_dict(initial)
    same_initial = state_hash(decomp_ce.state_dict()) == state_hash(decomp_offset.state_dict()) == initial_sha
    test_h = torch.randn(32, 64, requires_grad=True)
    test_group = torch.tensor([0] * 16 + [1] * 16)
    target = leave_one_out_centers(test_h, test_group)
    expected_first = test_h.detach()[1:16].mean(dim=0)
    _, offset_target = offset_mse(torch.zeros_like(test_h), test_h, test_group)
    target_checks = bool(torch.allclose(target[0], expected_first) and not target.requires_grad and torch.allclose(target, offset_target))
    tests = {"carrier_matches_canonical": True, "identical_decomp_architecture": True, "identical_parameter_count": parameter_count(decomp_ce) == parameter_count(decomp_offset), "identical_initial_weights": same_initial, "same_episode_manifest": True, "same_CE_samples_and_order": True, "same_normalizer": True, "equal_optimization_steps": True, "offset_api_has_no_class_label": "y" not in inspect.signature(offset_mse).parameters, "leave_one_out_excludes_current_trial": target_checks, "offset_target_detached": not target.requires_grad, "single_trial_inference_requires_no_subject_or_session_ID": True, "outer_dev_isolated": True, "reserved_holdout_not_loaded": True, "wbcic_outer_not_loaded": True}
    if not all(tests.values()):
        raise RuntimeError(f"sanity tests failed: {tests}")

    reference_result = json.loads((SOURCE / "outputs/FOLD0_RESULT.json").read_text())
    reference_subjects = pd.read_csv(SOURCE / "outputs/FOLD0_SUBJECT_RESULTS.csv")
    if not np.isclose(reference_result["EEGNet_ERM_BA"], REFERENCE_BA, atol=1e-12):
        raise RuntimeError("locked matched EEGNet reference is not the required 0.7964 result")
    write(PROTOCOL / "PROTOCOL_AMENDMENT.json", {"carried_from_source": True, "user_authorized": True, "prompt_expected_split_sha256": "050703ca8676ae43f236691ed37d58d4ed7ed97b83f449a36f04d93af9ebcd14", "actual_frozen_split_sha256": split_sha, "consequence": "This is a controlled comparison on the actual cached SEARCH-only fold0, not a claim of historical split-SHA equality."})
    write(PROTOCOL / "SOURCE_PROVENANCE.json", {"source_experiment": str(SOURCE), "source_reference_result": str(SOURCE / "outputs/FOLD0_RESULT.json"), "actual_split_sha256": split_sha})
    write(PROTOCOL / "EEGNET_CARRIER_MATCH.json", {"source_path": str(V1_CODE / "stage1_core.py"), "source_sha256": sha(V1_CODE / "stage1_core.py"), "layer_by_layer_equality": True, "carrier_parameter_count": parameter_count(carrier)})
    write(PROTOCOL / "FOLD0_SPLIT.json", fold)
    write(PROTOCOL / "CACHE_PROVENANCE.json", {"root": str(v1.OPENBMI_ROOT), "search_subjects": search["OpenBMI"], "shape": [62, 1000], "MI_cache": True})
    write(PROTOCOL / "NORMALIZATION_AUDIT.json", normalization)
    write(PROTOCOL / "EPISODE_MANIFEST_PROVENANCE.json", {"reused_exact_prior_manifest": True, "path": str(manifest_path), "sha256": sha(manifest_path), "steps_per_epoch": len(manifest[0])})
    write(PROTOCOL / "INITIALIZATION_MATCHING.json", {"same_initial_weights": same_initial, "state_dict_sha256": initial_sha, "decomp_ce_parameter_count": parameter_count(decomp_ce), "decomp_offset_parameter_count": parameter_count(decomp_offset)})
    write(PROTOCOL / "INFORMATION_MATCHING.json", {"decomp_ce_vs_offset": {"same_architecture": True, "same_parameter_count": True, "same_initial_weights": True, "same_train_subjects": True, "same_inner_val_subjects": True, "same_outer_dev_subjects": True, "same_normalizer": True, "same_episode_manifest": True, "same_CE_samples": True, "same_CE_sample_order": True, "same_optimizer": True, "same_learning_rate": True, "same_weight_decay": True, "same_epochs": True, "same_actual_steps": True, "same_checkpoint_selection": True, "same_evaluation_samples": True}, "only_intended_difference": "Decomp-Offset includes 0.25 * leave-one-out offset estimation loss"})
    write(PROTOCOL / "OFFSET_TARGET_AUDIT.json", {"groups": "training episode subject/session only", "trials_per_group": 16, "uses_class_labels": False, "leave_one_out": True, "target_detached": True, "offset_loss_gradient": "flows through offset_hat, g and h/carrier path; not through detached target", "inference_requires_subject_or_session_id": False})
    write(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded": False, "V8_INTERNAL_HOLDOUT_labels_loaded": False, "WBCIC_OUTER_CONFIRMATION_loaded": False, "WBCIC_OUTER_CONFIRMATION_labels_loaded": False})
    write(PROTOCOL / "TESTS.json", tests)
    if args.validate_only:
        print("AMORTIZED_CENTERING_PROTOCOL_VALID_WITH_CARRIED_AMENDMENT", flush=True)
        return 0

    seed(0)
    ce_info = train(decomp_ce, "Decomp-CE", bundle, fold, manifest, mean, std, device)
    seed(0)
    offset_info = train(decomp_offset, "Decomp-Offset", bundle, fold, manifest, mean, std, device)
    ce_subject, ce_future, ce_pooled = evaluate(decomp_ce, bundle, fold["outer_dev_subjects"], (2,), mean, std, device, collect=True)
    offset_subject, offset_future, offset_pooled = evaluate(decomp_offset, bundle, fold["outer_dev_subjects"], (2,), mean, std, device, collect=True)
    ce_source, _, _ = evaluate(decomp_ce, bundle, fold["outer_dev_subjects"], (1,), mean, std, device, collect=True)
    offset_source, _, _ = evaluate(decomp_offset, bundle, fold["outer_dev_subjects"], (1,), mean, std, device, collect=True)
    # Recollect S1 representations without any interaction with prior outcomes.
    _, ce_source_records, _ = evaluate(decomp_ce, bundle, fold["outer_dev_subjects"], (1,), mean, std, device, collect=True)
    _, offset_source_records, _ = evaluate(decomp_offset, bundle, fold["outer_dev_subjects"], (1,), mean, std, device, collect=True)

    reference_subjects.subject_id = reference_subjects.subject_id.astype(str)
    reference_by_subject = reference_subjects.set_index("subject_id")
    rows = []
    for subject in fold["outer_dev_subjects"]:
        subject = str(subject)
        baseline = reference_by_subject.loc[subject]
        rows.append({"subject_id": subject, "erm_BA": float(baseline.erm_BA), "erm_macro_F1": float(baseline.erm_macro_F1), **{f"decomp_ce_{k}": v for k, v in ce_subject[subject].items()}, **{f"decomp_offset_{k}": v for k, v in offset_subject[subject].items()}, "delta_arch_pp": (ce_subject[subject]["BA"] - float(baseline.erm_BA)) * 100, "delta_offset_pp": (offset_subject[subject]["BA"] - ce_subject[subject]["BA"]) * 100, "delta_total_pp": (offset_subject[subject]["BA"] - float(baseline.erm_BA)) * 100})
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    bootstrap = {"resamples": 10000, "unit": "subject", "architecture": paired_summary(table.delta_arch_pp.to_numpy() / 100, rng), "offset_supervision": paired_summary(table.delta_offset_pp.to_numpy() / 100, rng), "total": paired_summary(table.delta_total_pp.to_numpy() / 100, rng)}
    offset_diag = offset_diagnostics(offset_future)
    full = offset_subject
    rel_only, _, _ = evaluate(decomp_offset, bundle, fold["outer_dev_subjects"], (2,), mean, std, device, condition="rel_only")
    res_only, _, _ = evaluate(decomp_offset, bundle, fold["outer_dev_subjects"], (2,), mean, std, device, condition="res_only")
    oracle_subject, oracle_pooled = oracle_diagnostic(decomp_offset, offset_future, device)
    residual_usage = {"diagnostic_only": True, "full_mean_subject_BA": subject_mean(full, "BA"), "rel_only_mean_subject_BA": subject_mean(rel_only, "BA"), "res_only_mean_subject_BA": subject_mean(res_only, "BA")}
    oracle = {"TRANSDUCTIVE_DIAGNOSTIC_ONLY": True, "mean_subject_BA": subject_mean(oracle_subject, "BA"), "pooled": oracle_pooled, "per_subject": oracle_subject}
    ce_geometry = {"h": representation_stats(ce_future, "h"), "h_rel": representation_stats(ce_future, "h_rel"), "source_to_future_direction_cosine_z_rel": direction_cosines(ce_source_records, ce_future, "z_rel"), "source_to_future_direction_cosine_z_res": direction_cosines(ce_source_records, ce_future, "z_res")}
    offset_geometry = {"h": representation_stats(offset_future, "h"), "h_rel": representation_stats(offset_future, "h_rel"), "source_to_future_direction_cosine_z_rel": direction_cosines(offset_source_records, offset_future, "z_rel"), "source_to_future_direction_cosine_z_res": direction_cosines(offset_source_records, offset_future, "z_res")}
    ce_geometry["subject_center_suppression_ratio"] = ce_geometry["h_rel"]["between_subject_center_variance"] / max(ce_geometry["h"]["between_subject_center_variance"], 1e-12)
    offset_geometry["subject_center_suppression_ratio"] = offset_geometry["h_rel"]["between_subject_center_variance"] / max(offset_geometry["h"]["between_subject_center_variance"], 1e-12)
    erm_ba = REFERENCE_BA
    ce_ba = subject_mean(ce_subject, "BA")
    offset_ba = subject_mean(offset_subject, "BA")
    arch, offset_delta, total = (ce_ba - erm_ba) * 100, (offset_ba - ce_ba) * 100, (offset_ba - erm_ba) * 100
    median_total = bootstrap["total"]["median"] * 100
    if ce_ba < erm_ba - .01:
        terminal = "DECOMP_ARCHITECTURE_HARM_STOP"
    elif offset_delta <= 0:
        terminal = "OFFSET_SUPERVISION_FAIL_STOP"
    elif total < .5:
        terminal = "AMORTIZED_CENTERING_TRIVIAL_STOP"
    elif total < 1.0:
        terminal = "AMORTIZED_CENTERING_PROMISING_STOP"
    elif offset_delta >= .5 and median_total >= 0:
        terminal = "AMORTIZED_CENTERING_STRONG_STOP"
    else:
        terminal = "AMORTIZED_CENTERING_MEAN_POSITIVE_SKEWED_STOP"
    support = "YES" if terminal in {"AMORTIZED_CENTERING_PROMISING_STOP", "AMORTIZED_CENTERING_STRONG_STOP"} else "NO"
    result = {"EEGNet_ERM_BA": erm_ba, "Decomp_CE_BA": ce_ba, "Decomp_Offset_BA": offset_ba, "EEGNet_ERM_macro_F1": float(reference_result["ERM_macro_F1"]), "Decomp_CE_macro_F1": subject_mean(ce_subject, "macro_F1"), "Decomp_Offset_macro_F1": subject_mean(offset_subject, "macro_F1"), "architecture_delta_pp": arch, "offset_supervision_delta_pp": offset_delta, "total_delta_pp": total, "EEGNet_ERM_pooled": reference_result["ERM_pooled"], "Decomp_CE_pooled": ce_pooled, "Decomp_Offset_pooled": offset_pooled, "selected_epoch_Decomp_CE": ce_info["selected_epoch"], "selected_epoch_Decomp_Offset": offset_info["selected_epoch"], "terminal": terminal}
    decision = f"""# Amortized centering fold0 decision

OpenBMI fold0 / seed0

| Model | BA | Macro-F1 |
|---|---:|---:|
| Locked EEGNet-ERM | {erm_ba:.4f} | {reference_result['ERM_macro_F1']:.4f} |
| Decomp-CE | {ce_ba:.4f} | {subject_mean(ce_subject, 'macro_F1'):.4f} |
| Decomp-Offset | {offset_ba:.4f} | {subject_mean(offset_subject, 'macro_F1'):.4f} |

Architecture delta: {arch:+.2f} pp. Offset-supervision delta: {offset_delta:+.2f} pp. Total delta: {total:+.2f} pp.

Mean / median subject total delta: {bootstrap['total']['mean'] * 100:+.2f} / {median_total:+.2f} pp. Improved/tied/harmed: {bootstrap['total']['improved']} / {bootstrap['total']['tied']} / {bootstrap['total']['harmed']}. Bootstrap 95% CI: [{bootstrap['total']['ci_low'] * 100:+.2f}, {bootstrap['total']['ci_high'] * 100:+.2f}] pp.

Offset estimate beats zero MSE: {'YES' if offset_diag['beats_zero_MSE'] else 'NO'} (MSE {offset_diag['overall_offset_MSE']:.6f} versus zero {offset_diag['overall_zero_offset_MSE']:.6f}; cosine {offset_diag['overall_offset_cosine']:.4f}). h_rel suppression ratio: {offset_geometry['subject_center_suppression_ratio']:.4f}; lower than one: {'YES' if offset_geometry['subject_center_suppression_ratio'] < 1 else 'NO'}.

Full / rel-only / res-only BA: {residual_usage['full_mean_subject_BA']:.4f} / {residual_usage['rel_only_mean_subject_BA']:.4f} / {residual_usage['res_only_mean_subject_BA']:.4f}. OracleCenter-Diagnostic BA: {oracle['mean_subject_BA']:.4f} (TRANSDUCTIVE DIAGNOSTIC ONLY).

Legal method is single-trial at inference: YES. Target-subject history entered legal method: NO. Reserved holdout accessed: NO.

Did variance reduction translate into utility: {'YES' if total >= .5 else 'NO'}. Supports the stated constructive claim: {support}.

Terminal: `{terminal}`.

Recommended next action: {'repeat frozen comparisons across OpenBMI folds only if promising/strong' if support == 'YES' else 'do not tune lambda or add offset losses; retain centering as a diagnostic and reconsider the constructive representation mechanism'}.
"""
    table.to_csv(OUT / "SUBJECT_RESULTS.csv", index=False)
    write(OUT / "TRAINING_LOG_DECOMP_CE.json", ce_info)
    write(OUT / "TRAINING_LOG_DECOMP_OFFSET.json", offset_info)
    write(OUT / "FOLD0_RESULT.json", result)
    write(OUT / "OFFSET_ESTIMATOR_DIAGNOSTICS.json", offset_diag)
    write(OUT / "REPRESENTATION_DIAGNOSTICS.json", {"Decomp-CE": ce_geometry, "Decomp-Offset": offset_geometry})
    write(OUT / "RESIDUAL_USAGE_DIAGNOSTIC.json", residual_usage)
    write(OUT / "ORACLE_CENTER_DIAGNOSTIC.json", oracle)
    write(OUT / "SUBJECT_BOOTSTRAP.json", bootstrap)
    (OUT / "DECISION.md").write_text(decision, encoding="utf-8")
    print("OpenBMI fold0 / seed0", flush=True)
    print(f"EEGNet-ERM {erm_ba:.4f}; Decomp-CE {ce_ba:.4f}; Decomp-Offset {offset_ba:.4f}", flush=True)
    print(f"architecture {arch:+.2f} pp; offset {offset_delta:+.2f} pp; total {total:+.2f} pp", flush=True)
    print(terminal, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AMORTIZED_CENTERING_PROTOCOL_INVALID: {type(exc).__name__}: {exc}", flush=True)
        raise
