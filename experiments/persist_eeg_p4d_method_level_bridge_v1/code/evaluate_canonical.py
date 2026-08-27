from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

import p4d_common as c


sys.path.insert(0, str(c.P4A / "code"))
import common as p4a_common  # noqa: E402


def subject_metrics(evaluated: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = evaluated["subjects"].astype(str)
    labels = evaluated["labels"].astype(np.int64)
    logits = evaluated["logits"].astype(np.float64)
    predictions = logits.argmax(1)
    ce = p4a_common.numpy_cross_entropy(logits, labels)
    for subject in p4a_common.subject_sort(np.unique(subjects)):
        mask = subjects == subject
        rows.append(
            {
                "subject_id": subject,
                "BA": float(balanced_accuracy_score(labels[mask], predictions[mask])),
                "F1": float(f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)),
                "CE": float(ce[mask].mean()),
                "rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def load_model(setting: str, fold: int, seed: int, method: str, lam: float, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    complete = c.read_json(c.require_file(c.source_complete(setting, fold, seed, method, lam)))
    checkpoint_path = c.require_file(c.checkpoint_path(setting, fold, seed, method, lam))
    if c.sha256(checkpoint_path) != complete["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = p4a_common.build_model(setting, p4a_common.stable_seed("P4A-init", setting, fold, seed))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device), complete


def selected_configs() -> list[tuple[str, float]]:
    payload = c.read_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json")
    return [
        (str(row["method"]), float(row["lambda_star"]))
        for row in payload["methods"]
        if row["status"] == "IDENTITY_MANIPULATION_COMPETENT"
    ]


def main() -> None:
    prefreeze_path = c.EXP / "P4D_PRE_TASK_OUTCOME_FREEZE.json"
    freeze = c.read_json(c.require_file(prefreeze_path))
    if freeze.get("pass") is not True or freeze.get("future_method_task_outcome_access_count_before_freeze") != 0:
        raise RuntimeError("P4D pre-task-outcome freeze missing or invalid")
    configs = selected_configs()
    identity = pd.read_csv(c.RESULTS / "canonical_identity_manipulation_source_only.csv")
    burden = pd.read_csv(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("prospective evaluation requires the server GPU")
    run_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for setting in ("S4", "S6"):
        bundle = p4a_common.load_data(setting)
        raw = torch.from_numpy(np.asarray(bundle.x)).to(device=device, non_blocking=False)
        print(f"[{setting}] prospective evaluation on {torch.cuda.get_device_name(0)}", flush=True)
        for fold in c.FOLDS:
            roles = p4a_common.roles_for(setting, fold)
            outcome_indices = p4a_common.row_indices(bundle.metadata, roles["outcome"], (bundle.future_session,))
            unit = p4a_common.run_dir(setting, fold, 0).parent
            for seed in c.SEEDS:
                normalizer_path = p4a_common.run_dir(setting, fold, seed) / "normalizer.npz"
                normalizer = np.load(c.require_file(normalizer_path), allow_pickle=False)
                mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=device)
                std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=device)
                erm_model, erm_complete = load_model(setting, fold, seed, "ERM", 0.0, device)
                erm_eval = p4a_common.evaluate_model(erm_model, raw, bundle.metadata, outcome_indices, mean, std, batch_size=512)
                erm_subject = subject_metrics(erm_eval).set_index("subject_id")
                del erm_model, erm_eval
                for method, lam in configs:
                    model, complete = load_model(setting, fold, seed, method, lam, device)
                    evaluated = p4a_common.evaluate_model(model, raw, bundle.metadata, outcome_indices, mean, std, batch_size=512)
                    method_subject = subject_metrics(evaluated).set_index("subject_id")
                    if list(method_subject.index) != list(erm_subject.index):
                        raise RuntimeError(f"subject mismatch {setting}/{fold}/{seed}/{method}")
                    for subject in method_subject.index:
                        subject_rows.append(
                            {
                                "setting_id": setting,
                                "fold": fold,
                                "seed": seed,
                                "method": method,
                                "lambda": lam,
                                "subject_id": subject,
                                "BA_ERM": float(erm_subject.loc[subject, "BA"]),
                                "BA_method": float(method_subject.loc[subject, "BA"]),
                                "DeltaG_BA": float(method_subject.loc[subject, "BA"] - erm_subject.loc[subject, "BA"]),
                                "F1_ERM": float(erm_subject.loc[subject, "F1"]),
                                "F1_method": float(method_subject.loc[subject, "F1"]),
                                "DeltaG_F1": float(method_subject.loc[subject, "F1"] - erm_subject.loc[subject, "F1"]),
                                "CE_ERM": float(erm_subject.loc[subject, "CE"]),
                                "CE_method": float(method_subject.loc[subject, "CE"]),
                                "DeltaG_CE": float(erm_subject.loc[subject, "CE"] - method_subject.loc[subject, "CE"]),
                            }
                        )
                    id_row = identity[(identity.setting_id == setting) & (identity.fold == fold) & (identity.seed == seed) & (identity.method == method)].iloc[0]
                    burden_row = burden[(burden.setting_id == setting) & (burden.fold == fold) & (burden.seed == seed)].iloc[0]
                    run_rows.append(
                        {
                            "setting_id": setting,
                            "dataset": c.SETTINGS[setting]["dataset"],
                            "task": c.SETTINGS[setting]["task"],
                            "backbone": c.SETTINGS[setting]["backbone"],
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "lambda": lam,
                            "subjects": len(method_subject),
                            "BA_ERM": float(erm_subject.BA.mean()),
                            "BA_method": float(method_subject.BA.mean()),
                            "DeltaG_BA": float(method_subject.BA.mean() - erm_subject.BA.mean()),
                            "F1_ERM": float(erm_subject.F1.mean()),
                            "F1_method": float(method_subject.F1.mean()),
                            "DeltaG_F1": float(method_subject.F1.mean() - erm_subject.F1.mean()),
                            "CE_ERM": float(erm_subject.CE.mean()),
                            "CE_method": float(method_subject.CE.mean()),
                            "DeltaG_CE": float(erm_subject.CE.mean() - method_subject.CE.mean()),
                            "I_ERM": float(id_row.I_ERM),
                            "I_method": float(id_row.I_method),
                            "S_I_abs": float(id_row.S_I_abs),
                            "S_I_rel": float(id_row.S_I_rel),
                            "z_SI": float(id_row.z_SI),
                            "R_unsafe": float(burden_row.R_unsafe),
                            "R_admissible": float(burden_row.R_admissible),
                            "R_unsafe_mass": float(burden_row.R_unsafe_mass),
                            "method_checkpoint_sha256": complete["checkpoint_sha256"],
                            "erm_checkpoint_sha256": erm_complete["checkpoint_sha256"],
                            "outcome_scope": "frozen P4A development future role; subject-first",
                        }
                    )
                    del model, evaluated
                torch.cuda.empty_cache()
        del raw
        torch.cuda.empty_cache()
    run_frame = pd.DataFrame(run_rows)
    subject_frame = pd.DataFrame(subject_rows)
    expected = 30 * len(configs)
    if len(run_frame) != expected or run_frame[["setting_id", "fold", "seed", "method"]].duplicated().any():
        raise RuntimeError(f"future outcome matrix is not balanced: {len(run_frame)} vs {expected}")
    c.write_csv(c.RESULTS / "canonical_method_future_outcomes.csv", run_frame)
    c.write_csv(c.RESULTS / "canonical_method_future_outcomes_subject.csv", subject_frame)
    complete_payload = {
        "schema": "PERSIST_EEG_P4D_METHOD_OUTCOME_EVALUATION_COMPLETE_V1",
        "pass": True,
        "timestamp_utc": c.now_utc(),
        "run_rows": len(run_frame),
        "subject_rows": len(subject_frame),
        "settings": ["S4", "S6"],
        "configs": [{"method": method, "lambda": lam} for method, lam in configs],
        "subject_first": True,
        "partial_outcome_retuning": False,
        "pre_task_outcome_freeze_sha256": c.sha256(prefreeze_path),
        "run_csv_sha256": c.sha256(c.RESULTS / "canonical_method_future_outcomes.csv"),
        "subject_csv_sha256": c.sha256(c.RESULTS / "canonical_method_future_outcomes_subject.csv"),
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
    }
    c.write_json(c.RESULTS / "P4D_METHOD_OUTCOME_EVALUATION_COMPLETE.json", complete_payload)
    c.write_text(
        c.EXP / "P4D_METHOD_OUTCOME_ACCESS_LEDGER.md",
        f"""# P4D Method Outcome Access Ledger

Prospective canonical method outcomes were first accessed only after `P4D_PRE_TASK_OUTCOME_FREEZE.json` was written. The freeze SHA-256 was `{complete_payload['pre_task_outcome_freeze_sha256']}`. Evaluation used the exact P4A roles and subject-first BA/F1/CE for S4 and S6. No lambda, method, burden, threshold, normalization, or analysis specification was changed after access. OpenBMI sealed internal holdout and WBCIC outer 10 were untouched.
""",
    )
    print(json.dumps(complete_payload, indent=2))
    print("P4D_CANONICAL_METHOD_FUTURE_OUTCOME_EVALUATION_COMPLETE")


if __name__ == "__main__":
    main()
