from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


EXPECTED = {
    "hmc": {"pool": 55, "outer_eval": 11, "meta_fit": 30, "calibration": 14},
    "eegmmidb": {"pool": 45, "outer_eval": 9, "meta_fit": 24, "calibration": 12},
}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_v2_split(payload: dict[str, object]) -> None:
    dataset = str(payload["dataset"])
    if dataset not in EXPECTED:
        raise ValueError("CAP and unknown datasets are forbidden from v2 development splits")
    expected = EXPECTED[dataset]
    meta = set(payload["meta_fit_subjects"])
    calibration = set(payload["calibration_subjects"])
    evaluation = set(payload["outer_evaluation_subjects"])
    final = set(payload["excluded_old_final_subjects"])
    source = set(payload["source_task_head_subjects"])
    if meta & calibration or meta & evaluation or calibration & evaluation:
        raise RuntimeError("v2 outer split has subject overlap")
    if (meta | calibration | evaluation) & final:
        raise RuntimeError("old final-test subject leaked into v2 development")
    if (meta | calibration | evaluation) & source:
        raise RuntimeError("source task-head subject leaked into v2 development")
    for key, actual in (("meta_fit", len(meta)), ("calibration", len(calibration)),
                        ("outer_eval", len(evaluation))):
        if actual != expected[key]:
            raise RuntimeError(f"{dataset} {key} size {actual} != {expected[key]}")


def generate_v2_splits(root: str | Path, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, object]:
    root = Path(root)
    output = root / "data" / "splits_v2_dev"
    hashes: dict[str, str] = {}
    report_rows: list[dict[str, object]] = []
    for dataset, expected in EXPECTED.items():
        for seed in seeds:
            split = json.loads((root / "data" / "splits" / dataset / f"seed_{seed}.json").read_text())
            roles = split["roles"]
            development = sorted(set(roles["meta_risk_train"]) | set(roles["conformal_calibration"]))
            if len(development) != expected["pool"]:
                raise RuntimeError(f"{dataset} development pool is {len(development)}, expected {expected['pool']}")
            rng = np.random.default_rng(92000 + seed)
            shuffled = np.asarray(development, dtype=object)[rng.permutation(len(development))]
            folds = np.array_split(shuffled, 5)
            coverage: list[str] = []
            for fold_index, evaluation_values in enumerate(folds):
                evaluation = sorted(map(str, evaluation_values.tolist()))
                training = sorted(set(development) - set(evaluation))
                fold_rng = np.random.default_rng(93000 + seed * 10 + fold_index)
                calibration = sorted(map(str, np.asarray(training, object)[
                    fold_rng.permutation(len(training))[: expected["calibration"]]
                ].tolist()))
                meta = sorted(set(training) - set(calibration))
                payload = {
                    "version": "v2-nested-development-v1",
                    "dataset": dataset,
                    "original_seed": seed,
                    "outer_fold": fold_index,
                    "meta_fit_subjects": meta,
                    "calibration_subjects": calibration,
                    "outer_evaluation_subjects": evaluation,
                    "source_task_head_subjects": sorted(roles["task_head_train"]),
                    "excluded_old_final_subjects": sorted(roles["final_test"]),
                    "cap_development_forbidden": True,
                }
                validate_v2_split(payload)
                path = output / dataset / f"seed_{seed}" / f"outer_fold_{fold_index}.json"
                _atomic_json(path, payload)
                hashes[str(path)] = file_sha256(path)
                coverage.extend(evaluation)
                report_rows.append({"dataset": dataset, "seed": seed, "fold": fold_index,
                                    "meta_fit": len(meta), "calibration": len(calibration),
                                    "outer_evaluation": len(evaluation), "sha256": hashes[str(path)]})
            if sorted(coverage) != development or len(coverage) != len(set(coverage)):
                raise RuntimeError(f"{dataset} seed {seed} outer evaluation coverage is incomplete")
    provenance = root / "outputs" / "v2_joint_certified" / "provenance"
    _atomic_json(provenance / "V2_SPLIT_HASHES.json", {"files": hashes})
    lines = ["# V2 nested development split report", "", "All folds are subject-disjoint and exclude source-training and old final-test subjects.", "",
             "| dataset | seed | fold | meta fit | calibration | outer evaluation |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['dataset']} | {r['seed']} | {r['fold']} | {r['meta_fit']} | {r['calibration']} | {r['outer_evaluation']} |" for r in report_rows]
    (provenance / "V2_SPLIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"files": len(hashes), "rows": report_rows}
