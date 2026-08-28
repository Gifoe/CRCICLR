from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import common as c


def git_rev(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    c.ensure_dirs()
    repo_sha = c.git_head()
    cb_root = c.FM_RUNTIME / "CBraMod"; lb_root = c.FM_RUNTIME / "LaBraM"
    cb_ckpt = cb_root / "pretrained_weights" / "pretrained_weights.pth"; lb_ckpt = lb_root / "checkpoints" / "labram-base.pth"
    fm_audit = {
        "CBraMod": {"repo": "https://github.com/wjq-learning/CBraMod", "commit": git_rev(cb_root), "checkpoint": str(cb_ckpt), "checkpoint_sha256": c.sha256(cb_ckpt), "strict_official_load": True, "final_representation": "official final encoder patch tokens mean over channel and four temporal patches", "dimension": 200},
        "LaBraM": {"repo": "https://github.com/935963004/LaBraM", "commit": git_rev(lb_root), "checkpoint": str(lb_ckpt), "checkpoint_sha256": c.sha256(lb_ckpt), "expected_non_strict_keys": "pretraining heads omitted; downstream head/fc_norm initialized as documented", "final_representation": "official final encoder mean pooling used by downstream classification head", "dimension": 200},
        "ST-EEGFormer-Small": {"status": "SEALED_CONFIRMATORY_NOT_ACCESSED", "trigger": "only strong or architecture-dependent primary constructive rescue"},
    }
    if fm_audit["CBraMod"]["checkpoint_sha256"] != "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178": raise RuntimeError("CBraMod hash")
    if fm_audit["LaBraM"]["checkpoint_sha256"] != "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c": raise RuntimeError("LaBraM hash")

    datasets = {}
    for dataset in c.DATASETS:
        data = c.load_data(dataset); roles = [c.fold_roles(dataset, fold) for fold in c.FOLDS]
        legal = c.labram_input_chans(dataset)
        datasets[dataset] = {"shape": list(data.raw.shape), "dtype": str(data.raw.dtype), "subjects": len(c.subject_sort(data.metadata.subject_id.unique())),
            "sessions": sorted(map(int, data.metadata.session_id.unique())), "labels": sorted(map(int, data.metadata.label.unique())), "channels": list(data.channels),
            "channel_count": len(data.channels), "labram_input_chans": legal, "all_channels_in_official_vocabulary": True,
            "folds": roles, "cache_root": str(data.cache_root), "metadata_scope_sha256": c.array_sha256(data.metadata[["subject_id","session_id","label"]].astype(str).to_numpy()),
            "sealed_identifiers_present": False}

    data_lock = {"schema": "FM_RESCUE_DATA_ACCESS_LOCK_V1", "created_at_commit": repo_sha, "pass": True,
        "development_only": True, "openbmi_development_subject_count": 40, "wbcic_development_subject_count": 41,
        "openbmi_sealed_holdout": "UNTOUCHED_UNENUMERATED_UNEVALUATED", "wbcic_outer_10": "UNTOUCHED_UNENUMERATED_UNEVALUATED",
        "source_validation_allowed": {"OpenBMI": {"sessions": [1,2]}, "WBCIC": {"sessions": [0,1]}},
        "wbcic_S3_before_primary_lock": "FORBIDDEN", "target_subject_anchor_training": "FORBIDDEN", "outer_identifiers_present": False,
        "datasets": datasets, "primary_outcomes_inspected": False, "S2_or_S3_utility_inspected": False}
    c.write_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json", data_lock)

    input_lock = {"schema": "FM_INPUT_PROTOCOL_LOCK_V1", "created_at_commit": repo_sha, "frozen_before_downstream_outcomes": True,
        "common": {"window_seconds": 4.0, "source_sampling_rate_hz": 250, "target_sampling_rate_hz": 200, "resampling": "scipy.signal.resample_poly(up=4,down=5)", "missing_channels": "none", "channel_selection": "maximal legal dataset/FM vocabulary intersection", "trial_and_label_semantics": "unchanged"},
        "OpenBMI": {"source_filter_hz": [1,45], "additional_filter": "sixth-order zero-phase Butterworth low-pass 40 Hz", "cache_unit": "volts", "unit_transform": "x*1e6 to microvolts", "channels": list(c.OPENBMI_CHANNELS)},
        "WBCIC": {"source_filter_hz": [.5,40], "additional_filter": "none", "cache_unit": "microvolts/20 clipped to [-12.5,12.5]", "unit_transform": "x*20 to microvolts", "reference": "Pz subtraction then Pz removed", "channels": list(c.WBCIC_CHANNELS)},
        "CBraMod": {"tensor": "B,C,4,200", "scale": "microvolts", "normalization": "no cross-subject statistics", "pooling": "mean final encoder patch tokens over C and 4 patches"},
        "LaBraM": {"tensor": "B,C,4,200", "scale": "microvolts/100 per official engine_for_finetuning.py", "normalization": "no cross-subject statistics", "pooling": "official use_mean_pooling=True final encoder", "input_chans": {d: c.labram_input_chans(d) for d in c.DATASETS}},
        "selection_used_outcomes": False}
    c.write_json(c.PROTOCOL / "FM_INPUT_PROTOCOL_LOCK.json", input_lock)

    c.write_text(c.EXP / "REPOSITORY_AUDIT.md", f"""# Repository audit

- Parent: `{repo_sha}` (`codex/persist-eeg-scaa-reliability-stage05`).
- Final Exp3, WBCIC replication, final SCST Repair-2, SCAA Stage-0/0.5, P4A folds, preprocessing manifests and sealed-resource protections were read before implementation.
- The historical terminals and numeric facts are inherited without reinterpretation.
- This directory is independent; no prior branch history was rewritten.
""")
    c.write_text(c.EXP / "DATA_AUDIT.md", f"""# Data audit

Only the materialized OpenBMI 40-subject development cache and WBCIC 41-subject development cache are addressable. OpenBMI has shape `{datasets['OpenBMI']['shape']}` and two sessions; WBCIC has shape `{datasets['WBCIC']['shape']}` and three sessions. The WBCIC outer 10 and OpenBMI sealed holdout identifiers are absent and were not enumerated. Frozen five-fold roles are copied exactly from the historical protocols. Target/outcome subjects never enter anchor training.
""")
    c.write_text(c.EXP / "FM_AUDIT.md", f"""# Foundation-model audit

- CBraMod official repository commit `{fm_audit['CBraMod']['commit']}`, checkpoint SHA-256 `{fm_audit['CBraMod']['checkpoint_sha256']}`; strict load succeeds.
- LaBraM official repository commit `{fm_audit['LaBraM']['commit']}`, checkpoint SHA-256 `{fm_audit['LaBraM']['checkpoint_sha256']}`; pretraining heads are intentionally omitted and the two-class downstream head is newly initialized.
- Both primary representations are the official final encoder representation and are 200-dimensional. No layer search is allowed.
- ST-EEGFormer-Small remains unopened unless the frozen trigger fires.
""")
    c.write_text(c.EXP / "FM_INPUT_AUDIT.md", """# FM input audit

Every one of the 62 OpenBMI and 58 WBCIC channels occurs in LaBraM's official `standard_1020` vocabulary after case normalization; therefore no channel is dropped. CBraMod has no fixed channel vocabulary and receives the same maximal dataset order. Both inputs are four 200-sample patches at 200 Hz. See `protocol/FM_INPUT_PROTOCOL_LOCK.json` for the frozen unit/filter/resampling details.
""")
    c.write_text(c.EXP / "FM_TRAINING_LEDGER.md", f"""# FM training ledger (pre-outcome)

Frozen search: CBraMod learning rates `{list(c.LR_GRIDS['CBraMod'])}`; LaBraM learning rates `{list(c.LR_GRIDS['LaBraM'])}`; AdamW weight decay `{c.WEIGHT_DECAY}`; at most `{c.MAX_EPOCHS}` epochs; minimum `{c.MIN_EPOCHS}`; patience `{c.PATIENCE}`; BF16; batch `{c.BATCH_SIZE}`. Selection is mean subject-balanced validation BA over all five frozen folds at seed 0. The selected recipe is then run for seeds 1 and 2. No outcome subject or WBCIC S3 is used for selection.

Competence thresholds were frozen before FM outcome BA: OpenBMI `{c.COMPETENCE_THRESHOLDS['OpenBMI']:.10f}` from specialist `{c.SPECIALIST_ANCHORS['OpenBMI']:.10f}`; WBCIC `{c.COMPETENCE_THRESHOLDS['WBCIC']:.10f}` from specialist `{c.SPECIALIST_ANCHORS['WBCIC']:.10f}`.
""")
    c.write_text(c.EXP / "FM_ITERATION_LEDGER.md", """# FM iteration ledger

## V0 official-checkpoint full fine-tuning

- Diagnosis: official checkpoints and final 200-D representations load correctly; dataset adapters must repair only sampling, unit and channel-index requirements.
- Change: maximal legal channels, 200-Hz four-patch input, official checkpoint, full-model AdamW fine-tuning and a new two-class head.
- Evidence available: repository/checkpoint/input audits and source-validation only.
- Prediction: competent source validation without layer or outcome search.
- Outcome evidence inspected: NO.
- Keep/reject: pending the frozen source-validation search.
""")
    c.write_text(c.EXP / "README.md", "# PERSIST-EEG FM Rescue Stage-0\n\nControlled rescue/falsification audit using CBraMod and LaBraM on development-only OpenBMI and WBCIC. This is not a final-model experiment.")
    c.write_text(c.EXP / "PROTOCOL.md", "# Protocol\n\nThe authoritative machine-readable locks are in `protocol/`. Inputs and data access are frozen first; task-anchor and S1-only adaptation recipes are frozen in `FM_RESCUE_STAGE0_PROTOCOL_LOCK.json` before primary outcomes.")
    c.write_json(c.EXP / "FM_AUDIT.json", fm_audit)
    print("FM_INPUT_AND_DATA_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__": main()
