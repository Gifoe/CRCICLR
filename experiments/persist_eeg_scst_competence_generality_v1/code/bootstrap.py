from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
PREV = REPO / "experiments" / "persist_eeg_fm_rescue_stage0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest(paths: list[Path]) -> tuple[list[dict[str, object]], str]:
    rows = []
    for path in sorted(paths):
        rows.append({
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(payload).hexdigest()


def main() -> None:
    for name in ("code", "protocol", "results", "figures", "runtime"):
        (EXP / name).mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    expected_branch = "codex/persist-eeg-scst-competence-generality-v1"
    if branch != expected_branch:
        raise RuntimeError(f"wrong branch: {branch}")

    anchors = list((PREV / "runtime" / "anchors" / "CBraMod").rglob("*.pt"))
    representations = list((PREV / "runtime" / "representations" / "CBraMod").rglob("*.npz"))
    preprocessing = [
        PREV / "runtime" / "OPENBMI_FM_INPUT_MASK.npy",
        PREV / "runtime" / "OPENBMI_FM_INPUT_UV_200HZ.npy",
        PREV / "runtime" / "WBCIC_FM_INPUT_MASK.npy",
        PREV / "runtime" / "WBCIC_FM_INPUT_UV_200HZ.npy",
    ]
    # OpenBMI has model_fit/validation/outcome (45 files); WBCIC additionally
    # has the separately sealed future-session outcome_all partition (60).
    if len(anchors) != 30 or len(representations) != 105 or not all(p.is_file() for p in preprocessing):
        raise RuntimeError((len(anchors), len(representations), [p.is_file() for p in preprocessing]))
    anchor_rows, anchor_hash = manifest(anchors)
    representation_rows, representation_hash = manifest(representations)
    preprocessing_rows, preprocessing_hash = manifest(preprocessing)

    final = json.loads((PREV / "results" / "FM_RESCUE_FINAL_REPORT.json").read_text(encoding="utf-8"))
    ratios = final["answers"]["27_30_SCST_ratios"]
    lock = {
        "schema": "CBRAMOD_GEOMETRY_PRESERVATION_LOCK_V1",
        "created_before_new_competence_outcomes": True,
        "base_commit": head,
        "branch": branch,
        "representation_definition": "frozen 200-D CBraMod task-projector penultimate output",
        "channel_mapping": {"OpenBMI": "62/62", "WBCIC": "58/58"},
        "folds": [0, 1, 2, 3, 4],
        "seeds": [0, 1, 2],
        "anchor_manifest_sha256": anchor_hash,
        "representation_manifest_sha256": representation_hash,
        "preprocessing_manifest_sha256": preprocessing_hash,
        "anchor_files": anchor_rows,
        "representation_files": representation_rows,
        "preprocessing_files": preprocessing_rows,
        "previous_scst_gates": {
            "OpenBMI": {
                "independent_session_3nn_ratio": ratios["OpenBMI/CBraMod"],
                "residual_stability": True,
                "subject_fidelity": True,
                "random_control_advantage": True,
                "class_fidelity": True,
                "manifold_valid": True,
            },
            "WBCIC": {
                "independent_session_3nn_ratio": ratios["WBCIC/CBraMod"],
                "residual_stability": True,
                "subject_fidelity": True,
                "random_control_advantage": True,
                "class_fidelity": True,
                "manifold_valid": True,
            },
        },
        "frozen_manifold_threshold": 1.25,
    }
    write_json(EXP / "protocol" / "CBRAMOD_GEOMETRY_PRESERVATION_LOCK.json", lock)
    write_json(EXP / "protocol" / "DATA_ACCESS_LOCK.json", {
        "schema": "SCST_COMPETENCE_DATA_ACCESS_LOCK_V1",
        "branch": branch,
        "base_commit": head,
        "development_datasets": ["OpenBMI", "WBCIC"],
        "reuse_exact_existing_folds": True,
        "WBCIC_outer_10": "UNTOUCHED_UNENUMERATED_UNEVALUATED",
        "OpenBMI_reserved_holdout": "UNTOUCHED_UNENUMERATED_UNEVALUATED",
        "sealed_resources_accessed": False,
    })
    write_json(EXP / "protocol" / "COMPETENCE_PROTOCOL_LOCK.json", {
        "schema": "SCST_COMPETENCE_PROTOCOL_LOCK_V1",
        "created_before_held_development_decoder_outcomes": True,
        "competence_thresholds": {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821},
        "decoder_input": "per-fold source-training z-score of frozen 200-D representation",
        "architectures": {
            "H0": "linear 200->2",
            "H1": "200->128->2 GELU",
            "H2": "200->256->64->2 GELU",
        },
        "learning_rates": [0.0003, 0.001, 0.003],
        "weight_decays": [0.0001, 0.001, 0.01],
        "dropouts": [0.0, 0.2, 0.4],
        "loss": "standard cross entropy",
        "selection": "dataset-global mean source-validation subject-balanced BA; NLL tie-break",
        "outcome_training_epochs": "dataset-global median selected source-validation best epoch",
        "future_session_used_for_selection": False,
        "subject_specific_tuning": False,
    })
    write_json(EXP / "protocol" / "SPECIALIST_SCREEN_LOCK.json", {
        "schema": "SCST_SPECIALIST_SCREEN_LOCK_V1",
        "created_before_specialist_outcomes": True,
        "models": ["FBCNet", "ATCNet", "EEGInceptionMI"],
        "backup": "EEGNeX only if none of the primary three is competent and admissible",
        "seeds": [0, 1, 2],
        "folds": [0, 1, 2, 3, 4],
        "representation": "final hidden representation immediately before classifier",
        "selection_data": "source model-fit/validation only",
        "competence_thresholds": {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821},
        "manifold_threshold": 1.25,
        "future_session_used_for_selection": False,
    })
    (EXP / "COMPETENCE_ITERATION_LEDGER.md").write_text(
        "# Competence iteration ledger\n\n"
        "All entries are recorded before any future-session SCST utility is inspected.\n\n"
        "## Iteration 1 — frozen-representation decoder repair\n\n"
        "- Hypothesis: the frozen CBraMod geometry contains nonlinear task information not recovered by the current linear head.\n"
        "- Available information: previous source-validation and held-development task competence; frozen SCST geometry audit.\n"
        "- Change: globally selected H0/H1/H2 decoder with frozen feature z-scoring.\n"
        "- Predicted effect: improve task BA without changing representation hashes or geometry.\n"
        "- Keep/reject: pending source-validation selection and held-development evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps({"branch": branch, "head": head, "anchors": len(anchors), "representations": len(representations), "lock": str(EXP / 'protocol' / 'CBRAMOD_GEOMETRY_PRESERVATION_LOCK.json')}, indent=2))


if __name__ == "__main__":
    main()
