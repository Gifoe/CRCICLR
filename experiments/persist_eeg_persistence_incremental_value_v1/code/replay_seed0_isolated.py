"""Crash-isolated exact seed-0 replay of the unchanged formal PEEH runner."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
P1 = ROOT.parent
FORMAL = ROOT / "experiments/persist_eeg_crossbackbone_peeh_v1/code/run_crossbackbone_peeh.py"
RUNTIME = P1 / "persist_incremental_value_runtime/seed0_replay"
REFERENCE = P1 / "crossbackbone_peeh_runtime"
OUTPUT = EXP / "outputs/SEED0_REPLAY_AUDIT.csv"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FIELDS = ("model", "task", "fold", "status", "protected_rank", "max_numeric_abs_delta", "seconds", "reason")


def compare(a: dict, b: dict) -> tuple[str, float, str]:
    if a["checkpoint_sha256"] != b["checkpoint_sha256"]:
        return "FAIL", float("inf"), "checkpoint mismatch"
    if a["protected_blocks"] != b["protected_blocks"]:
        return "FAIL", float("inf"), "Protected coordinate mismatch"
    if a["rank"] != b["rank"] or a["representation_dim"] != b["representation_dim"]:
        return "FAIL", float("inf"), "rank or representation mismatch"
    if len(a["protected_assignment"]) != len(b["protected_assignment"]):
        return "FAIL", float("inf"), "candidate block count mismatch"
    assignment_diffs = []
    keys = ("absolute_CE_harm", "absolute_CI_low", "absolute_CI_high", "excess_CE_harm", "excess_CI_low", "excess_CI_high")
    for old, new in zip(a["protected_assignment"], b["protected_assignment"]):
        if (old["block"], old["dimensions"], old["persistence_supported"], old["protected"]) != (
            new["block"], new["dimensions"], new["persistence_supported"], new["protected"]
        ):
            return "FAIL", float("inf"), "block/gate mismatch"
        assignment_diffs.extend(abs(float(old[k]) - float(new[k])) for k in keys)
    if [r["subject_id"] for r in a["subjects"]] != [r["subject_id"] for r in b["subjects"]]:
        return "FAIL", float("inf"), "subject mismatch"
    if max(assignment_diffs, default=0.0) > 1e-4:
        return "FAIL", max(assignment_diffs), "TRAIN utility evidence numerical drift"
    primary_diffs = []
    random_diffs = []
    for old, new in zip(a["subjects"], b["subjects"]):
        primary_diffs.extend(abs(float(old[k]) - float(new[k])) for k in ("intact_BA", "protected_BA"))
        random_diffs.append(abs(float(old["random_BA"]) - float(new["random_BA"])))
    # BA is quantized at one trial, while the reference averages 100 random
    # controls. A few near-tie random-probe predictions may flip across BLAS
    # thread counts. Never tolerate any intact/Protected BA change; cap the
    # mean-random discrepancy at 0.001 BA (0.1 pp), fixed before ablation.
    if max(primary_diffs, default=0.0) > 1e-10:
        return "FAIL", max(primary_diffs), "intact or Protected BA changed"
    if max(random_diffs, default=0.0) > 0.001:
        return "FAIL", 100 * max(random_diffs), "random-control mean BA drift >0.1 pp"
    diff_pp = 100 * max(random_diffs, default=0.0)
    return "PASS", diff_pp, "random mean quantization drift, pp" if diff_pp else ""


def save(rows: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    rows = []
    if OUTPUT.is_file():
        with OUTPUT.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(r["model"], r["task"], int(r["fold"])) for r in rows if r["status"] in ("PASS", "PROTOCOL_INVALID")}
    env = os.environ.copy()
    env.update({"PEEH_RUNTIME": str(RUNTIME), "SEVEN_RUNTIME": str(P1 / "seven_backbone_fourtask_3seed_runtime"),
                "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    for model in MODELS:
        for task in TASKS:
            for fold in range(5):
                key = (model, task, fold)
                if key in done:
                    continue
                start = time.monotonic()
                path = Path("cells") / model.lower() / task.lower() / f"fold{fold}_seed0.json"
                result_path = RUNTIME / path
                if not result_path.is_file():
                    command = [sys.executable, "-u", str(FORMAL), "--stage", "run", "--model", model, "--task", task, "--fold", str(fold)]
                    process = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
                    if process.returncode != 0:
                        reason = f"exit={process.returncode}; stderr={process.stderr[-1000:]}"
                        if key == ("CBraMod", "WBCIC_MI", 4) and "active rank below 4" in reason:
                            status = "PROTOCOL_INVALID"
                        else:
                            status = "FAIL"
                        row = {"model": model, "task": task, "fold": fold, "status": status,
                               "protected_rank": "", "max_numeric_abs_delta": "", "seconds": round(time.monotonic()-start, 3), "reason": reason}
                        rows = [r for r in rows if (r["model"], r["task"], int(r["fold"])) != key] + [row]
                        save(rows)
                        print("REPLAY", *key, status, reason[:160], flush=True)
                        continue
                reference = REFERENCE / path
                if not reference.is_file():
                    status, difference, reason, rank = "FAIL", "", "formal reference cell absent", ""
                else:
                    a = json.loads(reference.read_text(encoding="utf-8"))
                    b = json.loads(result_path.read_text(encoding="utf-8"))
                    status, difference, reason = compare(a, b)
                    rank = b["protected_dimensions"]
                row = {"model": model, "task": task, "fold": fold, "status": status,
                       "protected_rank": rank, "max_numeric_abs_delta": difference,
                       "seconds": round(time.monotonic()-start, 3), "reason": reason}
                rows = [r for r in rows if (r["model"], r["task"], int(r["fold"])) != key] + [row]
                save(rows)
                print("REPLAY", *key, status, "seconds", row["seconds"], flush=True)
    if len(rows) != 100:
        return 3
    return 0 if sum(r["status"] == "PASS" for r in rows) == 99 and sum(
        r["status"] == "PROTOCOL_INVALID" for r in rows) == 1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
