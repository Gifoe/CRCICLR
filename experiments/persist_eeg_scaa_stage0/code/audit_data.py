from __future__ import annotations

from collections import Counter

import common as c


def main() -> None:
    c.ensure_dirs()
    data = c.load_data()
    subjects = c.subject_sort(data.metadata.subject_id.unique())
    expected = c.subject_sort(c.P3.frozen_subjects())
    if subjects != expected or len(subjects) != 41:
        raise RuntimeError("authorized WBCIC development scope mismatch")
    if set(data.metadata.session_id.unique()) != {0, 1, 2}:
        raise RuntimeError("WBCIC session scope mismatch")
    mapping = c.target_fold_map()
    folds = []
    anchors = []
    for fold in c.FOLDS:
        role = c.roles(fold)
        folds.append({"fold": fold, **{key: list(value) for key, value in role.items()}})
        for backbone in c.BACKBONES:
            for seed in c.SEEDS:
                unit, checkpoint, normalizer = c.anchor_paths(backbone, fold, seed)
                if not checkpoint.is_file() or not normalizer.is_file() or not (unit / "UNIT_PROTOCOL.json").is_file():
                    raise FileNotFoundError(f"missing anchor artifact {backbone}/{fold}/{seed}")
                unit_protocol = c.read_json(unit / "UNIT_PROTOCOL.json")
                unit_roles = unit_protocol["roles"]
                model_fit = set(map(str, unit_roles["model_fit"]))
                outcome = set(map(str, unit_roles["outcome"]))
                if model_fit & outcome or outcome != set(role["outcome"]):
                    raise RuntimeError(f"anchor subject leakage {backbone}/{fold}/{seed}")
                anchors.append(
                    {
                        "backbone": backbone,
                        "fold": fold,
                        "seed": seed,
                        "checkpoint_path": str(checkpoint.relative_to(c.REPO)),
                        "checkpoint_sha256": c.sha256(checkpoint),
                        "normalizer_sha256": c.sha256(normalizer),
                        "target_subjects": list(role["outcome"]),
                        "target_seen_by_anchor": False,
                        "S2_or_S3_labels_used_for_anchor_training_or_selection": False,
                    }
                )
    cells = data.metadata.groupby(["subject_id", "session_id", "label"]).size()
    session_counts = Counter(map(int, data.metadata.session_id))
    lock = {
        "schema": "PERSIST_EEG_SCAA_STAGE0_DATA_ACCESS_LOCK_V1",
        "created_at_commit": c.git_head(),
        "dataset": "WBCIC / NEMAR nm000348",
        "development_subject_count": 41,
        "development_subjects": subjects,
        "development_subject_hash": c.P3.protocol()["dataset"]["allowed_subjects_hash"],
        "sealed_outer": {
            "count": 10,
            "identifiers_present": False,
            "accessed": False,
            "enumerated": False,
            "preprocessed": False,
            "evaluated": False,
        },
        "sessions": {"S1": 0, "S2": 1, "S3": 2},
        "cache": {
            "shape": list(data.x.shape),
            "dtype": str(data.x.dtype),
            "rows": len(data.metadata),
            "session_rows": dict(sorted(session_counts.items())),
            "minimum_subject_session_class_rows": int(cells.min()),
            "maximum_subject_session_class_rows": int(cells.max()),
        },
        "folds": folds,
        "target_to_outcome_fold": mapping,
        "anchors": anchors,
        "anchor_checkpoint_count": len(anchors),
        "preprocessing": {
            "channels": 58,
            "samples": 1000,
            "sampling_rate_hz": 250,
            "reference": "Pz",
            "signal_scaling": "uV/20 with frozen clipping; stored float16",
            "normalization": "identity mean=0/std=1 as frozen WBCIC protocol",
        },
        "S2_or_S3_adaptation_utility_inspected": False,
        "pass": True,
    }
    c.write_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json", lock)
    lines = [
        "# SCAA Stage-0 data audit",
        "",
        "- Dataset: WBCIC / NEMAR nm000348 only.",
        f"- Authorized development subjects: 41; IDs: `{','.join(subjects)}`.",
        f"- Cache: `{tuple(data.x.shape)}` `{data.x.dtype}`; sessions S1/S2/S3 = 0/1/2.",
        "- Sealed outer: 10 subjects; identifiers absent and not enumerated, accessed, preprocessed, or evaluated.",
        "- Folds: the frozen five subject-disjoint folds. Each development subject is an outcome target exactly once.",
        "- Anchors: 30 competent ERM checkpoints (2 backbones x 5 folds x 3 seeds).",
        "- For each target, the checkpoint is taken from its outcome fold; target membership is disjoint from model-fit subjects.",
        "- EEGNet anchor source: P3 WBCIC independent replication; S1+S2 model-fit subjects, validation-discovery S3, held-target S3 competence only.",
        "- EEGConformer anchor source: P4A S4; S1+S2 model-fit subjects, S1+S2 validation subjects, held-target S3 competence only.",
        "- No target S2/S3 label is used by the new adaptation recipe, hyperparameter selection, or checkpoint selection.",
        "- No adaptation utility was inspected during this audit.",
    ]
    c.write_text(c.EXP / "DATA_AUDIT.md", "\n".join(lines))
    print("SCAA_STAGE0_DATA_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()

