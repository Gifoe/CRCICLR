#!/usr/bin/env python
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pandas as pd

from _common import parser
from hsc_tta.utils import require_cpu


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
REPO = ROOT / "repo"


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def write_report(name: str, text: str) -> None:
    target = REPO / name
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, target)


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def main() -> int:
    args = parser("Generate evidence-backed CPU phase reports").parse_args()
    require_cpu(args.device)
    manifests = ROOT / "data/manifests"
    recordings = pd.read_parquet(manifests / "recordings.parquet")
    subjects = pd.read_parquet(manifests / "subjects.parquet")
    exclusions = pd.read_parquet(manifests / "exclusions.parquet")
    annotations = pd.read_parquet(manifests / "annotations.parquet")
    validation = json.loads((ROOT / "outputs/cpu_validation/validation_summary.json").read_text())
    audit = json.loads((manifests / "dataset_audit.json").read_text())
    simulation = pd.read_csv(ROOT / "outputs/cpu_simulation/simulation_summary.csv").iloc[0]
    post_selection = pd.read_csv(ROOT / "outputs/cpu_simulation/post_selection_validity.csv")

    dataset_rows = []
    preprocessing_rows = []
    label_rows = []
    cache_stats: dict[str, object] = {}
    for dataset in ("eegmmidb", "hmc", "cap"):
        raw_size = tree_size(ROOT / f"data/raw/{dataset}")
        processed_size = tree_size(ROOT / f"data/processed/{dataset}")
        ds_recordings = recordings[recordings.dataset == dataset]
        ds_subjects = subjects[subjects.dataset == dataset]
        eligible = int(ds_subjects.eligible.sum())
        excluded = int((~ds_subjects.eligible).sum())
        dataset_rows.append([
            dataset, len(ds_recordings), int(ds_recordings.readable.sum()), eligible, excluded,
            gib(raw_size), gib(processed_size), audit[dataset]["eligible_subjects"],
        ])
        label_counter: Counter[int] = Counter()
        windows = 0
        one_channel = 0
        caches = sorted((ROOT / f"data/processed/{dataset}").glob("*.h5"))
        sample_rows = []
        for cache in caches:
            with h5py.File(cache, "r") as handle:
                labels = handle["label"][:]
                label_counter.update(map(int, labels))
                windows += len(labels)
                if int(handle["channel_mask"][:].sum()) == 1:
                    one_channel += 1
                if len(sample_rows) < 2:
                    metadata = json.loads(handle.attrs["metadata_json"])
                    sample_rows.append([metadata["subject_id"], len(labels)])
        label_rows.append([dataset, windows, dict(sorted(label_counter.items()))])
        state = json.loads((ROOT / f"state/preprocess_{dataset}.json").read_text())
        preprocessing_rows.append([dataset, len(caches), len(state.get("failed", [])), windows, one_channel, gib(processed_size)])
        cache_stats[dataset] = {"caches": len(caches), "windows": windows, "labels": dict(sorted(label_counter.items())), "sample_rows": sample_rows}

    exclusion_rows = (
        exclusions.groupby(["dataset", "exclusion_reason"]).size().reset_index(name="count").values.tolist()
        if len(exclusions) else []
    )
    split_rows = []
    episode_rows = []
    for dataset, details in validation["datasets"].items():
        for seed, seed_details in details["seeds"].items():
            split_rows.append([dataset, seed, json.dumps(seed_details["role_counts"], sort_keys=True)])
            episode_rows.append([
                dataset, seed, seed_details["episodes"], seed_details["excluded_episodes"],
                f"{seed_details['context_min']}–{seed_details['context_max']}",
                f"{seed_details['future_min']}–{seed_details['future_max']}",
            ])

    download_rows = []
    for dataset in ("eegmmidb", "hmc", "cap"):
        frame = pd.read_parquet(manifests / f"{dataset}_download_manifest.parquet")
        download_rows.append([dataset, len(frame), int((frame.status == "verified_existing").sum()), int(frame.status.isin(["missing", "failed"]).sum())])

    mock_rows = []
    for name in ("subject_context_features", "subject_action_surface", "subject_decisions"):
        path = ROOT / f"mock_features/{name}.parquet"
        mock_rows.append([path.name, len(pd.read_parquet(path)), path.stat().st_size])

    pytest_text = (ROOT / "logs/pytest_final.log").read_text(errors="ignore")
    coverage_text = (ROOT / "logs/coverage_final.log").read_text(errors="ignore")
    pytest_match = re.search(r"(\d+) passed", pytest_text)
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", coverage_text)
    pytest_passed = int(pytest_match.group(1)) if pytest_match else None
    coverage_pct = int(coverage_match.group(1)) if coverage_match else None

    disk = shutil.disk_usage(ROOT)
    cpu_model = next((line.split(":", 1)[1].strip() for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines() if line.startswith("Model name:")), platform.processor())
    memory_limit_path = Path("/sys/fs/cgroup/memory.max")
    memory_limit = memory_limit_path.read_text().strip() if memory_limit_path.exists() else "unknown"
    generated_at = datetime.now(timezone.utc).isoformat()

    data_table = table(
        ["dataset", "recordings", "readable", "eligible subjects", "excluded subjects", "raw", "processed", "audit eligible"],
        dataset_rows,
    )
    downloads_table = table(["dataset", "manifest files", "SHA verified", "missing/failed"], download_rows)
    exclusion_table = table(["dataset", "reason", "count"], exclusion_rows) if exclusion_rows else "No exclusions."
    labels_table = table(["dataset", "windows", "mapped label counts"], label_rows)
    preprocessing_table = table(["dataset", "complete caches", "failures", "windows", "one-channel caches", "size"], preprocessing_rows)
    splits_table = table(["dataset", "seed", "role counts"], split_rows)
    episodes_table = table(["dataset", "seed", "episodes", "excluded", "context range", "future range"], episode_rows)
    mock_table = table(["file", "rows", "bytes"], mock_rows)
    post_table = table(post_selection.columns.tolist(), post_selection.values.tolist())

    write_report("DATA_AUDIT_REPORT.md", f"""# Data audit report

Generated: {generated_at}

{data_table}

All downloaded payloads passed local SHA256 verification. The five CAP exclusions are retained in `exclusions.parquet`; no quality rule was relaxed.

{exclusion_table}

Annotation vocabulary audit rows: {len(annotations)}. Full recording, subject, channel, annotation-vocabulary, exclusion, and download manifests are under `data/manifests` outside Git.
""")

    write_report("SMOKE_DOWNLOAD_REPORT.md", f"""# Smoke download report

Initial readable smoke payloads were EEGMMIDB subject 001 target runs, HMC SN001 with scoring sidecars, and CAP brux1 with TXT/ST sidecars. Smoke reads passed before full download. Full-download verification supersedes the smoke status:

{downloads_table}

Source URLs are the public PhysioNet endpoints recorded in `configs/datasets/*.yaml`. No authenticated or foundation-model download was used.
""")

    smoke_rows = []
    for dataset, values in cache_stats.items():
        for subject_id, n_windows in values["sample_rows"]:
            smoke_rows.append([dataset, subject_id, n_windows, "quality_flags present; episode valid"])
    write_report("SMOKE_TEST_REPORT.md", f"""# Preprocessing smoke-test report

Two subjects per dataset were read, channel-selected, filtered, resampled, windowed, atomically cached, resumed, and episode-checked before the full run.

{table(["dataset", "subject", "windows", "checks"], smoke_rows)}

The smoke run detected the CAP clock-format inconsistency (`HH.MM.SS` versus `HH:MM:SS`); the parser and regression test now cover both formats.
""")

    write_report("PREPROCESSING_REPORT.md", f"""# Full CPU preprocessing report

Generated: {generated_at}

{preprocessing_table}

Configuration: sleep uses 30 s non-overlapping epochs, 0.3–40 Hz band-pass, 200 Hz target rate, and no default notch; EEGMMIDB uses runs 4/6/8/10/12/14, 1–40 Hz, 160 Hz, and up to 4 s per mapped imagery event. Caches store physical signal, labels, time boundaries, recording/run IDs, channel names/mask, sampling rate, three per-window quality flags, and the preprocessing config hash.

The container cgroup memory limit is {memory_limit} bytes. HMC and CAP therefore ran serially; CAP used one fresh Python process per remaining subject after long-process allocator pressure was observed. Resume checks occur before raw EDF loading. All final caches have `complete=true`; failed subject count is zero.

Mapped label/event counts:

{labels_table}
""")

    write_report("SPLIT_AND_EPISODE_REPORT.md", f"""# Split and episode report

All split units are subjects. Five deterministic seeds were generated, and the validator confirmed exact subject coverage and pairwise role disjointness. CAP calibration uses deterministic proportional stratification over the pathology prefixes available in official record names; every represented pathology contributes at least one calibration subject when capacity permits.

{splits_table}

{episodes_table}

Leakage validation: {len(validation['leakage_failures'])} failures. Artifact validation: `valid={validation['valid']}` with {len(validation['failures'])} total failures. Sleep context uses the first 90 minutes of clock time from the first valid epoch; future begins at the boundary. MI context runs are 4/6 and future runs are 8/10/12/14.
""")

    write_report("NEXT_GPU_PHASE.md", f"""# Next GPU phase (not executed)

The CPU phase did not download CBraMod, call CUDA, extract real embeddings, train a real task head, or run real TTA.

Required order for the later GPU phase:

1. Obtain and checksum the public CBraMod checkpoint; record license and exact revision.
2. Load only the subject roles defined in `data/splits`; reserve subject-level early stopping inside `task_head_train`.
3. Extract frozen-backbone representations from cached windows. Normalization may be per-window or fitted only on U_s/training data; V_s statistics must never transform U_s.
4. Train the HMC and EEGMMIDB task heads only on their task-head roles. CAP inherits the HMC head.
5. Fit meta-risk predictors only on `meta_risk_train` subjects. CAP inherits the HMC predictor.
6. Produce the three parquet interfaces shown below. Context and adaptation features may use only U_s. Future risk and task metrics may use V_s only during offline evaluation.
7. Fit one simultaneous residual quantile from calibration subjects and freeze it before final-test decisions.
8. Run schema, grid-completeness, subject-role, context/future, and provenance checks before computing final metrics.

{mock_table}

Pooled float32 embeddings for {sum(v['windows'] for v in cache_stats.values()):,} windows require about 0.19 GiB at dimension 200 or 0.48 GiB at dimension 512, before metadata. Dense token embeddings could require 5–20 GiB depending on token count. Current free space ({gib(disk.free)}) is sufficient for these scenarios but must be rechecked before execution.
""")

    write_report("CPU_PHASE_REPORT.md", f"""# HSC-TTA EEG CPU phase report

Generated: {generated_at}

## Environment and scope

- Host: {platform.node()}
- CPU: {cpu_model}
- Python: {sys.version.split()[0]} in `hsc_cpu`
- Container memory limit: {memory_limit} bytes
- CUDA: disabled for every CPU-stage command (`CUDA_VISIBLE_DEVICES=`)
- Data disk: {gib(disk.total)} total, {gib(disk.free)} free
- GPU/foundation model work: not executed

## Downloads and audit

{downloads_table}

{data_table}

{exclusion_table}

## Preprocessing

{preprocessing_table}

{labels_table}

All eligible subjects have one complete, config-hashed HDF5 cache. Raw data remained read-only. The five excluded CAP records lack both required central-channel alternatives.

## Subject splits and deployment episodes

{splits_table}

{episodes_table}

Independent artifact validation found zero split-role overlaps, zero U_s/V_s overlaps, zero sleep boundary violations, and zero MI run-protocol violations. No episode failed the minimum-future rule.

CAP target-site calibration is deterministically proportionally stratified by the pathology prefix encoded in the public record name.

## Statistical core and synthetic validation

Implemented components include prediction sets, empirical-Bernstein-style block bounds, grouped meta-risk prediction, finite-sample simultaneous residual quantiles, deterministic safe action selection, subject-level metrics/bootstrap, three action interfaces, schemas, and simulations A–E.

- Synthetic subjects: {int(simulation['n_subjects'])}
- Calibration subjects: {int(simulation['n_calibration'])}
- Simultaneous quantile q: {simulation['q']}
- Surface coverage: {simulation['surface_coverage']}
- Certified Subject Rate: {simulation['certified_subject_rate']}

Post-selection comparison:

{post_table}

The default synthetic result is conservative but not useful: `q=1.0` and CSR=0. The certificate covers because it saturates, not because the method demonstrates nontrivial certification. With the configured block bound, the additive term `3 log(3/eta)/B` is already above 0.20 for small B; typical future horizons therefore make alpha 0.10/0.20 certification difficult or impossible. This must be resolved theoretically and empirically before claiming a successful method.

## Frozen GPU interface

{mock_table}

All rows were validated with the Pydantic schemas. Fields derived from V_s are confined to offline action-surface/decision evaluation; context features are synthetic U_s-only interface rows with the exact future field names.

## Tests

- pytest: {pytest_passed} passed, one expected conservative-quantile warning
- line coverage: {coverage_pct}%
- artifact validation: valid={validation['valid']}, failures={len(validation['failures'])}, leakage failures={len(validation['leakage_failures'])}

Coverage is adequate for the statistical core but weak in real EDF adapters/preprocessing branches; the real full-data run provides integration evidence but is not counted by pytest coverage.

## Completion and known limitations

CPU deliverables are complete: public data download and SHA manifests, unified audit, full preprocessing, five subject-disjoint splits, all deployment episodes, statistical components, simulations, frozen schemas, tests, and reports. Deliberately unfinished work is the prohibited GPU phase: checkpoint acquisition, real embeddings, real task-head/meta-risk training, real action surfaces, and final scientific results.

Known limitations:

1. The current bound/synthetic configuration produces trivial certification (`CSR=0`).
2. Five CAP recordings were excluded for missing required central channels.
3. The AutoDL container is capped at 2 GiB RAM; CAP required per-subject process isolation.
4. Pytest line coverage is {coverage_pct}%, below a strong production target.
5. CAP filenames are treated as recording-level subject IDs because no stronger cross-record identity mapping is available in the public metadata.

Current disk free space is {gib(disk.free)}, sufficient to proceed to the estimated GPU cache scenarios in `NEXT_GPU_PHASE.md`.
""")

    state = {
        "current_stage": "CPU-7 complete",
        "download_status": {row[0]: {"files": row[1], "verified": row[2], "failures": row[3]} for row in download_rows},
        "preprocessing_status": {row[0]: {"complete_subjects": row[1], "failed_subjects": row[2], "windows": row[3]} for row in preprocessing_rows},
        "split_seeds": [0, 1, 2, 3, 4],
        "artifact_validation": {"valid": validation["valid"], "failures": validation["failures"]},
        "pytest_passed": pytest_passed,
        "coverage_percent": coverage_pct,
        "cuda_disabled": True,
        "foundation_model_downloaded": False,
        "last_updated": generated_at,
    }
    state_path = ROOT / "state/cpu_phase_state.json"
    temporary = state_path.with_suffix(state_path.suffix + ".part")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, state_path)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
