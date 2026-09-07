from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = EXPERIMENT_ROOT / "outputs"
PROTOCOL = OUTPUTS / "protocol"
RESULTS = OUTPUTS / "results"
FIGURES = OUTPUTS / "figures"
V2_ROOT = REPO_ROOT / "experiments" / "persist_eeg_prospective_action_policy_v2"
V2_OUTPUTS = V2_ROOT / "outputs"
V2_CODE = V2_ROOT / "code"

# Import the frozen V2 implementation rather than reimplementing its policy.
# V2.1 has no modules named common/data/metrics/policies, so these names cannot
# shadow V2 internals in this process.
if str(V2_CODE) not in sys.path:
    sys.path.insert(0, str(V2_CODE))
import common as v2_common  # noqa: E402
import data as v2_data  # noqa: E402
import freeze as v2_freeze  # noqa: E402
import metrics as v2_metrics  # noqa: E402
import policies as v2_policies  # noqa: E402


FULL_MENU = ("amplify", "geometry", "erase")
SAFE_MENU = ("amplify", "geometry")
POOL_NAMES = ("EXPLORATION_POOL", "DEVELOPMENT_HOLDOUT")
POOL_LABELS = {"EXPLORATION_POOL": "exploration", "DEVELOPMENT_HOLDOUT": "holdout"}


def ensure_directories() -> None:
    for path in (OUTPUTS, PROTOCOL, RESULTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def load_pool(cache_root: Path, pool: str) -> pd.DataFrame:
    return v2_data.load_pool(cache_root, pool).frame.copy().reset_index(drop=True)


def selected_prediction(frame: pd.DataFrame, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction, effect = v2_metrics._realized(frame, selected)
    probability = frame.p1_noop.to_numpy(dtype=float).copy()
    for action in FULL_MENU:
        mask = selected == action
        probability[mask] = frame.loc[mask, f"p1_{action}"].to_numpy(dtype=float)
    return prediction.astype(int), probability, effect


def v2_policies_for_frame(frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    disagree = frame.other_run_base_disagrees.to_numpy(dtype=bool)
    selected = {
        "M0_KEEP": np.full(len(frame), "noop", dtype=object),
        "I003_CROSS_RUN_FULL": v2_metrics.select_actions(frame, disagree, FULL_MENU),
        "I003_CROSS_RUN_PROTECTED_SAFE": v2_metrics.select_actions(frame, disagree, SAFE_MENU),
        "ORACLE_FULL_MENU": v2_metrics.oracle_actions(frame, FULL_MENU),
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    for policy_id, actions in selected.items():
        prediction, probability, effect = selected_prediction(frame, actions)
        result[policy_id] = {
            "selected": actions,
            "prediction": prediction,
            "probability": probability,
            "effect": effect,
        }
    return result


def _sort_compare(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    tolerance: float = 1e-14,
) -> tuple[bool, float, str | None]:
    left = actual[keys + columns].copy()
    right = expected[keys + columns].copy()
    # CSV inference turns canonical numeric-looking subject strings into ints.
    # Normalize every identity key before sorting; otherwise ``1, 14, 4`` and
    # ``1, 4, 14`` create a false reconstruction failure.
    for key in keys:
        left[key] = left[key].astype(str)
        right[key] = right[key].astype(str)
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return False, float("inf"), f"row_count {len(left)} != {len(right)}"
    for key in keys:
        if not left[key].equals(right[key]):
            return False, float("inf"), f"key mismatch: {key}"
    maximum = 0.0
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            difference = np.abs(left[column].to_numpy(dtype=float) - right[column].to_numpy(dtype=float))
            finite = difference[np.isfinite(difference)]
            maximum = max(maximum, float(finite.max()) if len(finite) else 0.0)
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                atol=tolerance,
                rtol=0,
                equal_nan=True,
            ):
                return False, maximum, f"numeric mismatch: {column}"
        elif not left[column].fillna("<NA>").astype(str).equals(right[column].fillna("<NA>").astype(str)):
            return False, float("inf"), f"value mismatch: {column}"
    return True, maximum, None


def _policy_tables(frame: pd.DataFrame, policies: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects: list[pd.DataFrame] = []
    runs: list[pd.DataFrame] = []
    actions: list[pd.DataFrame] = []
    for policy_id, values in policies.items():
        subject, run, action = v2_metrics.policy_tables(frame, values["selected"])
        for table in (subject, run, action):
            table.insert(0, "policy_id", policy_id)
        subjects.append(subject)
        runs.append(run)
        actions.append(action)
    return (
        pd.concat(subjects, ignore_index=True),
        pd.concat(runs, ignore_index=True),
        pd.concat(actions, ignore_index=True),
    )


def _summary_metrics(frame: pd.DataFrame, policies: dict[str, dict[str, np.ndarray]], pool: str) -> pd.DataFrame:
    oracle_gain = v2_metrics.policy_metrics(
        frame,
        policies["ORACLE_FULL_MENU"]["selected"],
        bootstrap_repetitions=3000 if pool == "EXPLORATION_POOL" else 5000,
        seed_offset=90,
    )["mean_subject_delta_BA"]
    rows = []
    seed_offsets = {"M0_KEEP": 0, "I003_CROSS_RUN_FULL": 3, "I003_CROSS_RUN_PROTECTED_SAFE": 4, "ORACLE_FULL_MENU": 90}
    for policy_id, values in policies.items():
        repetitions = 5000 if pool == "DEVELOPMENT_HOLDOUT" and policy_id in ("M0_KEEP", "ORACLE_FULL_MENU") else 3000
        metrics = v2_metrics.policy_metrics(
            frame,
            values["selected"],
            oracle_gain=oracle_gain,
            bootstrap_repetitions=repetitions,
            seed_offset=seed_offsets[policy_id],
        )
        rows.append({"policy_id": policy_id, **metrics})
    return pd.DataFrame(rows)


def _identity_and_prediction_hashes(frame: pd.DataFrame, policies: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    identity_columns = ["fold_id", "seed_id", "router_fold_id", "manifest_index", "subject_id", "session_id"]
    identity = frame[identity_columns].sort_values(identity_columns).astype(str).to_dict(orient="records")
    policy_hashes = {}
    for policy_id, values in policies.items():
        rows = pd.DataFrame(
            {
                **{column: frame[column].astype(str) for column in identity_columns},
                "action": values["selected"].astype(str),
                "prediction": values["prediction"].astype(str),
                "effect": values["effect"].astype(str),
            }
        ).sort_values(identity_columns).to_dict(orient="records")
        policy_hashes[policy_id] = canonical_hash(rows)
    return {"identity_sha256": canonical_hash(identity), "policy_prediction_sha256": policy_hashes}


def write_analysis_spec() -> dict[str, Any]:
    spec = {
        "status": "V2_1_ANALYSIS_PREDECLARED",
        "experiment_type": "post-V2 exploratory falsification audit",
        "baselines": {
            "B0_TARGET_KEEP": "target KEEP prediction",
            "B1_OTHER_RUN_HARD_MAJORITY": "class 1 iff leave-target-run KEEP vote fraction >= 0.5",
            "B2_ALL_RUN_HARD_MAJORITY": "class 1 iff all-run KEEP vote fraction >= 0.5; exact tie deterministically maps to class 1",
            "B3_OTHER_RUN_PROBABILITY_MEAN": "class 1 iff leave-target-run mean KEEP p(y=1) >= 0.5",
            "B4_ALL_RUN_PROBABILITY_MEAN": "class 1 iff all-run mean KEEP p(y=1) >= 0.5",
            "B5_OTHER_RUN_LOGIT_MEAN": "class 1 iff leave-target-run mean centered KEEP margin >= 0",
            "B6_ALL_RUN_LOGIT_MEAN": "class 1 iff all-run mean centered KEEP margin >= 0",
            "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE": "all-run p(y=1) mean weighted by max(abs(p-0.5),1e-12), threshold 0.5",
        },
        "controls": {
            "C1_GATED_DIRECT_CONSENSUS": "replace target prediction with other-run hard majority on disagreement",
            "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL": "same availability mask as frozen FULL; directly emit other-run majority",
            "C3_ACTION_MASKED_DIRECT_CONSENSUS_SAFE": "same AMPLIFY+GEOMETRY mask as frozen SAFE; directly emit other-run majority",
        },
        "best_ensemble_selection": {
            "pool": "EXPLORATION_POOL only",
            "criterion": "largest mean subject-balanced Delta BA versus B0 among B1-B7",
            "tie_break": "lexicographically smallest baseline ID within 1e-15",
            "threshold_tuning": False,
        },
        "confidence_weight_formula_frozen_before_outcomes": True,
        "paired_bootstrap_unit": "subject",
        "bootstrap_repetitions": 10000,
        "deployment_i003_aggregation": "DEPLOYMENT_OUTPUT_NOT_YET_DEFINED",
        "previous_holdout_is_not_sealed_for_v2_1": True,
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    spec["analysis_spec_hash"] = canonical_hash(spec)
    write_json(PROTOCOL / "V2_1_ANALYSIS_SPEC.json", spec)
    return spec


def reconstruct_v2(cache_root: Path) -> dict[str, Any]:
    ensure_directories()
    spec = write_analysis_spec()
    lock = v2_freeze.verify_lock()
    expected_lock = "e679c7a955ccf3745bb35ce6c86a61c57705557f3eed8917b724b0e5613b5fd4"
    checks: list[dict[str, Any]] = []
    pool_payload: dict[str, Any] = {}
    all_pass = lock["policy_lock_hash"] == expected_lock
    checks.append(
        {
            "check": "V2_POLICY_LOCK_HASH",
            "passed": all_pass,
            "expected": expected_lock,
            "actual": lock["policy_lock_hash"],
            "max_abs_difference": 0.0,
        }
    )

    split = json.loads((V2_OUTPUTS / "protocol" / "AUTONOMOUS_RESEARCH_SPLIT.json").read_text(encoding="utf-8"))
    expected_cache_hashes = {Path(path).name: value for path, value in split["source_sha256"].items()}
    for name in v2_data.FILES.values():
        actual_hash = sha256_file(cache_root / name)
        expected_hash = expected_cache_hashes.get(name)
        passed = expected_hash == actual_hash
        checks.append(
            {
                "check": f"source_cache_sha256:{name}",
                "passed": passed,
                "expected": expected_hash,
                "actual": actual_hash,
                "max_abs_difference": 0.0,
            }
        )
        all_pass &= passed

    for pool in POOL_NAMES:
        label = POOL_LABELS[pool]
        frame = load_pool(cache_root, pool)
        policies = v2_policies_for_frame(frame)
        subject, run, action = _policy_tables(frame, policies)
        summary = _summary_metrics(frame, policies, pool)
        if pool == "EXPLORATION_POOL":
            expected_subject = pd.read_csv(V2_OUTPUTS / "exploration" / "EXPLORATION_SUBJECT_RESULTS.csv")
            expected_run = pd.read_csv(V2_OUTPUTS / "exploration" / "EXPLORATION_RUN_RESULTS.csv")
            expected_action = pd.read_csv(V2_OUTPUTS / "exploration" / "EXPLORATION_ACTION_RESULTS.csv")
            expected_summary = pd.read_csv(V2_OUTPUTS / "exploration" / "EXPLORATION_POLICY_RESULTS.csv")
        else:
            expected_subject = pd.read_csv(V2_OUTPUTS / "holdout" / "DEVELOPMENT_HOLDOUT_SUBJECT_RESULTS.csv")
            expected_run = pd.read_csv(V2_OUTPUTS / "holdout" / "DEVELOPMENT_HOLDOUT_RUN_RESULTS.csv")
            expected_action = pd.read_csv(V2_OUTPUTS / "holdout" / "DEVELOPMENT_HOLDOUT_ACTION_RESULTS.csv")
            expected_summary = pd.read_csv(V2_OUTPUTS / "holdout" / "DEVELOPMENT_HOLDOUT_POLICY_RESULTS.csv")
        # Exploration persisted the oracle summary in EXPLORATION_DECISION,
        # but did not persist oracle subject/run/action rows.  Compare only
        # actually existing historical CSV rows here and audit the oracle
        # summary separately below.  Holdout persisted all four policies.
        historical_policy_ids = (
            ["M0_KEEP", "I003_CROSS_RUN_FULL", "I003_CROSS_RUN_PROTECTED_SAFE"]
            if pool == "EXPLORATION_POOL"
            else list(policies)
        )
        comparisons = [
            (
                "subject_delta",
                subject[subject.policy_id.isin(historical_policy_ids)],
                expected_subject[expected_subject.policy_id.isin(historical_policy_ids)],
                ["policy_id", "subject_id"],
                ["delta_BA", "available_runs"],
            ),
            (
                "run_delta",
                run[run.policy_id.isin(historical_policy_ids)],
                expected_run[expected_run.policy_id.isin(historical_policy_ids)],
                ["policy_id", "fold_id", "seed_id"],
                ["delta_BA", "subjects"],
            ),
            (
                "action_counts",
                action[action.policy_id.isin(historical_policy_ids)],
                expected_action[expected_action.policy_id.isin(historical_policy_ids)],
                ["policy_id", "action"],
                ["count", "fraction", "mean_effect", "rescue_count", "harm_count"],
            ),
            (
                "summary",
                summary[summary.policy_id.isin(historical_policy_ids)],
                expected_summary[expected_summary.policy_id.isin(historical_policy_ids)],
                ["policy_id"],
                [
                    "mean_subject_delta_BA",
                    "bootstrap_CI95_L",
                    "bootstrap_CI95_U",
                    "action_rate",
                    "unsafe_intervention_rate",
                    "rescue_precision",
                    "positive_run_fraction",
                ],
            ),
        ]
        for name, actual, expected, keys, columns in comparisons:
            passed, maximum, reason = _sort_compare(actual, expected, keys, columns)
            checks.append(
                {
                    "check": f"{label}:{name}",
                    "passed": passed,
                    "expected": "historical V2 CSV",
                    "actual": reason or "numerically identical",
                    "max_abs_difference": maximum,
                }
            )
            all_pass &= passed
        if pool == "EXPLORATION_POOL":
            decision = json.loads(
                (V2_OUTPUTS / "exploration" / "EXPLORATION_DECISION.json").read_text(encoding="utf-8")
            )
            oracle_actual = v2_metrics.policy_metrics(
                frame,
                policies["ORACLE_FULL_MENU"]["selected"],
                bootstrap_repetitions=3000,
                seed_offset=90,
            )
            oracle_expected = decision["oracle"]
            oracle_columns = sorted(set(oracle_actual) & set(oracle_expected))
            passed, maximum, reason = _sort_compare(
                pd.DataFrame([{**oracle_actual, "policy_id": "ORACLE_FULL_MENU"}]),
                pd.DataFrame([{**oracle_expected, "policy_id": "ORACLE_FULL_MENU"}]),
                ["policy_id"],
                oracle_columns,
            )
            checks.append(
                {
                    "check": "exploration:oracle_summary",
                    "passed": passed,
                    "expected": "historical EXPLORATION_DECISION.json",
                    "actual": reason or "numerically identical",
                    "max_abs_difference": maximum,
                }
            )
            all_pass &= passed
        hashes = _identity_and_prediction_hashes(frame, policies)
        pool_payload[label] = {
            "rows": len(frame),
            "subjects": int(frame.subject_id.nunique()),
            "manifest_trials": int(frame.manifest_index.nunique()),
            "runs": sorted(
                [f"fold-{fold}_seed-{seed}" for fold, seed in frame[["fold_id", "seed_id"]].drop_duplicates().itertuples(index=False)]
            ),
            **hashes,
        }

    payload = {
        "status": "V2_RECONSTRUCTION_PASS" if all_pass else "V2_RECONSTRUCTION_FAIL",
        "v2_policy_lock_hash": lock["policy_lock_hash"],
        "v2_split_assignment_hash": split["assignment_hash"],
        "checks": checks,
        "pools": pool_payload,
        "analysis_spec_hash": spec["analysis_spec_hash"],
        "numerical_tolerance": 1e-14,
        "historical_artifact_scope": {
            "exploration": "M0/FULL/SAFE subject-run-action-summary CSV plus ORACLE summary JSON",
            "holdout": "M0/FULL/SAFE/ORACLE subject-run-action-summary CSV",
            "note": "V2 did not persist exploration ORACLE subject/run/action rows; they are deterministically recomputed but cannot be compared to a nonexistent historical table.",
        },
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "V2_RECONSTRUCTION.json", payload)
    check_table = pd.DataFrame(checks)[["check", "passed", "max_abs_difference", "actual"]]
    md = f"""# Exact V2 reconstruction

`{payload['status']}`

- Frozen lock: `{lock['policy_lock_hash']}`
- Split hash: `{split['assignment_hash']}`
- Numerical tolerance: `1e-14`
- WBCIC outer accessed: `false`

The frozen V2 implementation was imported without modification. Subject,
run, action, rescue/harm, and summary tables were recomputed from the hashed
router caches. Identity and final prediction hashes cover every manifest/run
row and are stored in the JSON artifact.

{markdown_table(check_table)}
"""
    (PROTOCOL / "V2_RECONSTRUCTION.md").write_text(md, encoding="utf-8")
    if not all_pass:
        raise RuntimeError("V2 reconstruction failed; V2.1 analysis is prohibited")
    return payload
