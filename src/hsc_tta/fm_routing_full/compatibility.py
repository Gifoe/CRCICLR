from __future__ import annotations

from typing import Any


MODEL_ORDER = ["cbramod", "eegpt", "bendr", "brant", "eeg2rep", "neurogpt", "biot", "labram"]
DATASET_ORDER = ["hmc", "eegmmidb", "sleepedffull", "bcic2a"]
COMBINATION_ORDER = [
    ["hmc", "eegmmidb"],
    ["hmc", "bcic2a"],
    ["sleepedffull", "eegmmidb"],
    ["sleepedffull", "bcic2a"],
]


def compatible_channel_semantics(model: str, dataset: str) -> tuple[bool, str]:
    if dataset == "sleepedffull":
        return False, "dataset is not present; no subject/sample coverage"
    if model == "cbramod":
        return True, "ACPE has no fixed electrode lookup; actual channel labels are retained as generic signal channels"
    if dataset == "hmc":
        reasons = {
            "eegpt": "58-channel checkpoint channel identities cannot represent C3-M2/C4-M1",
            "bendr": "Deep1010 would treat bipolar derivations as scalp-electrode locations",
            "brant": "intracranial model release lacks an admissible scalp-EEG adapter/checkpoint",
            "eeg2rep": "spatial kernel is checkpoint channel-count/order dependent",
            "neurogpt": "checkpoint spatial convolution is channel-count/order dependent",
            "biot": "PREST tokens 10/14 mean C3-P3/C4-P4, not C3-M2/C4-M1",
            "labram": "referential C3/C4 position tokens cannot represent C3-M2/C4-M1",
        }
        return False, reasons[model]
    if model == "biot":
        if dataset == "eegmmidb":
            return True, "the exact 16 PREST bipolar montages can be derived from the 64 referential electrodes"
        return False, "BCIC2A lacks the electrodes required for the fixed PREST 16-montage set"
    if model == "labram":
        return True, "dataset contains named referential 10-20 electrodes covered by official standard_1020 identities"
    if model == "eegpt":
        return True, "official downstream code provides named-channel subset handling for EEGMMIDB/BCIC2A-like referential inputs"
    if model == "bendr":
        return True, "Deep1010 can map named referential scalp electrodes without inventing a bipolar derivation"
    if model == "neurogpt":
        return (dataset == "bcic2a", "official project targets BCIC2A, but exact frozen checkpoint compatibility still requires a loadable checkpoint audit")
    if model in {"brant", "eeg2rep"}:
        return False, "no loadable official pretrained checkpoint/config is available for an exact adapter audit"
    return False, "unsupported pair"


def make_matrix(model_registry: dict[str, dict[str, Any]], datasets: dict[str, dict[str, Any]], smoke: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_priority, model in enumerate(MODEL_ORDER):
        spec = model_registry[model]
        for dataset_priority, dataset in enumerate(DATASET_ORDER):
            data = datasets[dataset]
            semantic_ok, semantic_reason = compatible_channel_semantics(model, dataset)
            present = bool(data["present"])
            a1 = present and bool(data["unit_explicit"]) and bool(spec["unit_explicit"])
            a2 = present and bool(data["sampling_rate_hz"]) and bool(spec["sampling_rate_rule_explicit"])
            a3 = present and semantic_ok
            a4 = not (dataset == "hmc" and model in {"eegpt", "bendr", "biot", "labram"})
            a5 = semantic_ok
            a6 = present and semantic_ok and float(data["subject_coverage"]) >= 0.95 and float(data["sample_coverage"]) >= 0.95
            a7 = bool(smoke.get(model, {}).get(dataset, False))
            a8 = a7 and bool(spec["backbone_freezable"])
            a9 = bool(spec["target_supervised_exposure_absent"])
            a10 = True
            checks = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
            rows.append({
                "model": model, "model_priority": model_priority, "dataset": dataset,
                "dataset_priority": dataset_priority, "task_family": data["task_family"],
                "A_COMP_1": a1, "A_COMP_2": a2, "A_COMP_3": a3, "A_COMP_4": a4,
                "A_COMP_5": a5, "A_COMP_6": a6, "A_COMP_7": a7, "A_COMP_8": a8,
                "A_COMP_9": a9, "A_COMP_10": a10,
                "subject_coverage": data["subject_coverage"] if semantic_ok else 0.0,
                "sample_coverage": data["sample_coverage"] if semantic_ok else 0.0,
                "compatible": all(checks), "semantic_reason": semantic_reason,
                "checkpoint_status": spec["checkpoint_status"], "code_commit": spec["code_commit"],
                "checkpoint_sha256": spec.get("checkpoint_sha256"),
            })
    return rows


def choose_core(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for datasets in COMBINATION_ORDER:
        common = []
        for model in MODEL_ORDER:
            selected = [row for row in rows if row["model"] == model and row["dataset"] in datasets]
            if len(selected) == 2 and all(row["compatible"] for row in selected):
                common.append(model)
        if "cbramod" in common and len(common) >= 3:
            return {"datasets": datasets, "models": common[:3]}
    return None
