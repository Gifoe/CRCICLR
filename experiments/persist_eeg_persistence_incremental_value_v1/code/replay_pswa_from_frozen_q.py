"""Replay formal seed-0 PSWA probes from the exact frozen canonical arrays.

This audits the original random-coordinate draws and numeric session/subject
results. It does not reconstruct representations or change PEEH assignments.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
P1 = ROOT.parent
OLD = P1 / "crossbackbone_pswa_runtime"
EXT = P1 / "baseline_metrics_closure_v1_pswa_runtime"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
OUTPUT = EXP / "outputs/SEED0_PSWA_REPLAY_AUDIT.csv"
FIELDS = ("model", "task", "fold", "status", "rank", "protected_rank", "sessions", "max_BA_abs_delta", "reason")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def ridge_scores(x: np.ndarray, y: np.ndarray, z: np.ndarray, classes: int) -> np.ndarray:
    # Exact published_run_pswa_recovery.py ridge_scores implementation.
    mu = x.mean(0, dtype=np.float64); sd = x.std(0, dtype=np.float64); sd[sd < 1e-6] = 1.0
    train = ((x - mu) / sd).astype(np.float32); test = ((z - mu) / sd).astype(np.float32)
    targets = np.eye(classes, dtype=np.float64)[y]; target_mean = targets.mean(0); centered = targets - target_mean
    weight = np.linalg.solve(train.T @ train + 0.01 * np.eye(train.shape[1]), train.T @ centered)
    return test @ weight + target_mean


def scores(arrays: dict[str, np.ndarray], dims: list[int], classes: int) -> dict[tuple[str, str], float]:
    x, y = arrays["q_train"][:, dims], arrays["y_train"]
    result = {}
    for session in sorted(key.removeprefix("q_eval_") for key in arrays if key.startswith("q_eval_")):
        q = arrays[f"q_eval_{session}"][:, dims]
        labels = arrays[f"y_eval_{session}"]
        owners = arrays[f"subjects_eval_{session}"].astype(str)
        pred = ridge_scores(x, y, q, classes).argmax(1)
        for subject in sorted(set(owners)):
            mask = owners == subject
            result[(session, subject)] = float(balanced_accuracy_score(labels[mask], pred[mask]))
    return result


def write(rows: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    rows = []
    if OUTPUT.is_file():
        with OUTPUT.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(r["model"], r["task"], int(r["fold"])) for r in rows if r["status"] in ("PASS", "EMPTY", "PROTOCOL_INVALID")}
    for model in MODELS:
        root = EXT if model in ("ModernTCN", "Medformer") else OLD
        for task in TASKS:
            for fold in range(5):
                key = model, task, fold
                if key in done:
                    continue
                path = Path("cells") / model.lower() / task.lower() / f"fold{fold}_seed0.json"
                cell_path = root / path
                if not cell_path.is_file():
                    status, reason, rank, protected_rank, sessions, delta = "FAIL", "PSWA cell missing", "", "", "", ""
                else:
                    cell = json.loads(cell_path.read_text(encoding="utf-8"))
                    state = cell.get("status")
                    if state in ("EMPTY_PROTECTED", "EMPTY_UNDEFINED_PSWA"):
                        status, reason, rank, protected_rank, sessions, delta = "EMPTY", "k=0", cell.get("active_rank", ""), 0, "", ""
                    elif state in ("NO_PROTECTED_ASSIGNMENT", "PROTOCOL_INVALID") and key == ("CBraMod", "WBCIC_MI", 4):
                        status, reason, rank, protected_rank, sessions, delta = "PROTOCOL_INVALID", state, "", "", "", ""
                    elif state != "RECOVERY_OK":
                        status, reason, rank, protected_rank, sessions, delta = "FAIL", f"unexpected state {state}", "", "", "", ""
                    else:
                        cache = Path(cell["q_cache_path"])
                        if not cache.is_file() or sha(cache) != cell["q_cache_sha256"]:
                            status, reason, rank, protected_rank, sessions, delta = "FAIL", "q-cache SHA mismatch", "", "", "", ""
                        else:
                            with np.load(cache, allow_pickle=False) as payload:
                                arrays = {k: payload[k] for k in payload.files}
                            rank = int(arrays["q_train"].shape[1]); protected = list(map(int, cell["protected_indices"]))
                            protected_rank = len(protected)
                            controls = cell["random_controls"]
                            expected = [np.random.default_rng(stable_seed("final-random", model, task, fold, draw)).choice(
                                np.arange(rank), size=protected_rank, replace=False).astype(int).tolist() for draw in range(100)]
                            if controls != expected or any(len(x) != protected_rank for x in controls):
                                status, reason, sessions, delta = "FAIL", "random controls changed or not exact rank", "", ""
                            else:
                                classes = int(max(arrays["y_train"].max(), *(
                                    arrays[k].max() for k in arrays if k.startswith("y_eval_"))) + 1)
                                replay_p = scores(arrays, protected, classes)
                                reference_p = {(str(r["session"]), str(r["subject_id"])): float(r["BA"]) for r in cell["protected_session_results"]}
                                diffs = [abs(value - reference_p.get(key2, float("inf"))) for key2, value in replay_p.items()]
                                if len(replay_p) != len(reference_p): diffs.append(float("inf"))
                                reference_r = {(int(r["draw"]), str(r["session"]), str(r["subject_id"])): float(r["BA"])
                                               for r in cell["random_session_results"]}
                                count = 0
                                for draw, dims in enumerate(controls):
                                    for key2, value in scores(arrays, dims, classes).items():
                                        diffs.append(abs(value - reference_r.get((draw, *key2), float("inf"))))
                                        count += 1
                                if count != len(reference_r): diffs.append(float("inf"))
                                delta = max(diffs, default=0.0)
                                status = "PASS" if delta <= 1e-10 else "FAIL"
                                reason = "" if status == "PASS" else "PSWA session BA numeric mismatch"
                                sessions = ";".join(sorted({k[0] for k in replay_p}))
                row = {"model": model, "task": task, "fold": fold, "status": status, "rank": rank,
                       "protected_rank": protected_rank, "sessions": sessions, "max_BA_abs_delta": delta, "reason": reason}
                rows = [r for r in rows if (r["model"], r["task"], int(r["fold"])) != key] + [row]
                write(rows)
                print("PSWA_REPLAY", *key, status, flush=True)
                if status == "FAIL":
                    return 2
    return 0 if len(rows) == 100 and not any(r["status"] == "FAIL" for r in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
