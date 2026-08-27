from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
RUNS = EXP / "runtime" / "runs"
LOGS = EXP / "runtime" / "logs"
SETTINGS = ("S4", "S5", "S6")
FOLDS = range(5)
SEEDS = range(3)
GRID_METHODS = {
    "dann": ("0.01", "0.10", "1.00"),
    "coral": ("0.01", "0.10", "1.00"),
    "mmd": ("0.01", "0.10", "1.00"),
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_hash(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True, encoding="utf-8", errors="replace"
    ).strip()


def candidate_id(path: Path) -> str:
    rel = path.relative_to(RUNS)
    setting, fold, seed = rel.parts[:3]
    return f"{setting}/{fold}/{seed}/{path.stem}"


def scheduler_state() -> dict[str, object]:
    script = (
        "$t=Get-ScheduledTask -TaskName 'PERSIST_EEG_P4A_PIPELINE';"
        "$i=Get-ScheduledTaskInfo -TaskName 'PERSIST_EEG_P4A_PIPELINE';"
        "[pscustomobject]@{State=[string]$t.State;"
        "LastRunTime=$i.LastRunTime.ToString('o');"
        "LastTaskResult=$i.LastTaskResult;"
        "NextRunTime=$i.NextRunTime.ToString('o')}|ConvertTo-Json -Compress"
    )
    raw = subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    return json.loads(raw)


def main() -> None:
    candidate_files = sorted(RUNS.glob("S*/fold-*/seed-*/candidates/*.json"))
    completed_ids = [candidate_id(path) for path in candidate_files]
    erm_ids = [item for item in completed_ids if item.endswith("erm__lambda-0.00")]
    grid_ids = [item for item in completed_ids if not item.endswith("erm__lambda-0.00")]

    expected_grid_ids = [
        f"{setting}/fold-{fold}/seed-{seed}/{method}__lambda-{value}"
        for setting in SETTINGS
        for fold in FOLDS
        for seed in SEEDS
        for method, values in GRID_METHODS.items()
        for value in values
    ]
    incomplete_grid_ids = sorted(set(expected_grid_ids) - set(grid_ids))

    counts_by_setting = {
        setting: sum(item.startswith(f"{setting}/") for item in grid_ids)
        for setting in SETTINGS
    }
    checkpoints: list[dict[str, object]] = []
    for path in candidate_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(row["checkpoint"])
        checkpoints.append(
            {
                "configuration_id": candidate_id(path),
                "path": str(checkpoint),
                "declared_sha256": row.get("checkpoint_sha256"),
                "exists": checkpoint.exists(),
            }
        )

    last_file = max(candidate_files, key=lambda item: item.stat().st_mtime)
    last_row = json.loads(last_file.read_text(encoding="utf-8"))
    protocol_files = [
        "P4A_PROTOCOL_FROZEN.json",
        "SETTING_MANIFEST.json",
        "SETTING_SOURCE_MANIFEST.json",
        "PROTOCOL_FREEZE_COMMIT.json",
        "OUTCOME_ACCESS_LEDGER.md",
    ]
    hashes = {name: sha256(EXP / name) for name in protocol_files}
    hashes["runtime/logs/grid_stdout.log"] = sha256(LOGS / "grid_stdout.log")
    hashes["completed_configuration_set"] = set_hash(completed_ids)
    hashes["incomplete_grid_configuration_set"] = set_hash(incomplete_grid_ids)

    snapshot = {
        "schema": "P4A_GRID_PAUSE_SNAPSHOT_V1",
        "label": "OPTIONAL_PARTIAL_INVARIANCE_GRID",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "dirty_status": git("status", "--short").splitlines(),
        },
        "progress": {
            "erm_completed": len(erm_ids),
            "erm_expected": 45,
            "grid_completed": len(grid_ids),
            "grid_expected": 405,
            "grid_counts_by_setting": counts_by_setting,
        },
        "pause_boundary": {
            "current_running_configuration": None,
            "current_training_process_count": 0,
            "current_atomic_job_allowed_to_finish": True,
            "last_completed_configuration": candidate_id(last_file),
            "last_completed_candidate_file": str(last_file),
            "last_completed_elapsed_seconds": last_row.get("elapsed_seconds"),
            "future_non_erm_launches_stopped_after_grid_count": len(grid_ids),
            "stop_rule": "Stop immediately after the in-flight candidate JSON was atomically written; no scientific outcome was consulted.",
        },
        "scheduler": scheduler_state(),
        "runtime_paths": {
            "runtime_root": str(EXP / "runtime"),
            "runs": str(RUNS),
            "logs": str(LOGS),
            "grid_stdout": str(LOGS / "grid_stdout.log"),
            "grid_stderr": str(LOGS / "grid_stderr.log"),
        },
        "completed_configuration_ids": sorted(completed_ids),
        "incomplete_grid_configuration_ids": incomplete_grid_ids,
        "checkpoints": checkpoints,
        "hashes": hashes,
        "purity": {
            "partial_grid_excluded_from_p4b_hypothesis_selection": True,
            "invariance_outcome_deltas_remain_sealed": True,
            "direction_level_future_utilities_remain_sealed": True,
            "openbmi_internal_holdout_untouched": True,
            "wbcic_outer_holdout_untouched_not_enumerated": True,
        },
    }

    json_path = EXP / "P4A_GRID_PAUSE_SNAPSHOT.json"
    json_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    md = f"""# P4A Grid Pause Snapshot

- Timestamp (UTC): `{snapshot['timestamp_utc']}`
- Label: `OPTIONAL_PARTIAL_INVARIANCE_GRID`
- Branch / commit: `{snapshot['git']['branch']}` / `{snapshot['git']['commit']}`
- Mandatory ERM: **{len(erm_ids)}/45 complete**
- Optional non-ERM grid: **{len(grid_ids)}/405 complete**
- Per-setting grid: S4={counts_by_setting['S4']}/135, S5={counts_by_setting['S5']}/135, S6={counts_by_setting['S6']}/135
- Last completed atomic configuration: `{candidate_id(last_file)}`
- Current training process: none
- Scheduler: `{snapshot['scheduler']['State']}`; next trigger `{snapshot['scheduler']['NextRunTime']}`

The in-flight candidate was allowed to finish and atomically write its candidate JSON, checkpoint, and source freeze. The launcher was then stopped before it could begin another non-ERM configuration. Existing runtime artifacts were retained without deletion or renaming.

This is a **computational-scope pause**, not a scientific-outcome decision. The partial invariance grid is excluded from P4B hypothesis, predictor, normalization, threshold, and setting-selection decisions. Invariance outcome deltas and direction-level future utilities remain sealed.

Exact completed/incomplete configuration IDs, checkpoint paths, scheduler state, dirty Git state, and hashes are recorded in `P4A_GRID_PAUSE_SNAPSHOT.json`.
"""
    (EXP / "P4A_GRID_PAUSE_SNAPSHOT.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
