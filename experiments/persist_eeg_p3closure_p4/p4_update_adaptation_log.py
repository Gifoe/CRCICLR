from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
P4 = ROOT / "outputs" / "persist_eeg_p3closure_p4" / "p4"


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-version", required=True, choices=["V0", "V1", "V2"])
    parser.add_argument("--to-version", required=True, choices=["V1", "V2", "V3"])
    args = parser.parse_args()
    source_path = P4 / "development" / args.from_version / "fold-0" / "seed-0" / "DEVELOPMENT_RESULT.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["held_out_test_used"]:
        raise RuntimeError("Cannot adapt from a test-informed result")
    failed = [key for key, value in source["checks"].items() if not value]
    transitions = {
        ("V0", "V1"): {
            "category": "OPTIMIZATION",
            "problem": "Auxiliary and budget objectives act before task representation stabilizes; MI validation performance drops and its persistent path has negligible utility.",
            "change": "Add 5-epoch task-only warmup, ramp persistence/order losses over 5 epochs, delay budget to epoch 10 and halve lambda_B; modestly strengthen matched persistence and ordering losses.",
            "reason": "V0 passed the macro semantic gap and exact-geometry checks, so the core decomposition is retained. The observed failures are performance interaction and ineffective MI usage, not absence of persistence semantics.",
        },
        ("V1", "V2"): {
            "category": "METHOD",
            "problem": "The concatenation readout allows the persistent contribution to be weak or scale-confounded despite a valid semantic subspace.",
            "change": "Use the prompt-authorized residual readout C_F(h_F) + C_P(g_t * z_P) while retaining the exact projector, pair relation, losses, rank, and curriculum.",
            "reason": "This directly addresses persistent-path bypass after optimization/curriculum has already been tested.",
        },
        ("V2", "V3"): {
            "category": "REPRESENTATION",
            "problem": "A rank-8 residual persistent branch remains unstable or semantically diffuse.",
            "change": "Reduce maximum rank from 8 to 4, strengthen persistence/order losses, and lengthen their ramp while weakening budget pressure.",
            "reason": "P2/P3 estimated U_L ranks near 4-5; rank reduction is attempted only after optimization and readout failures.",
        },
    }
    key = (args.from_version, args.to_version)
    if key not in transitions:
        raise RuntimeError(f"Unsupported transition: {key}")
    path = P4 / "P4_ADAPTATION_LOG.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "protocol": "maximum four major versions V0-V3; TRAIN/VALIDATION only",
        "held_out_test_used": False,
        "entries": [],
    }
    if any(entry["to_version"] == args.to_version for entry in payload["entries"]):
        print(f"Adaptation to {args.to_version} already recorded")
        return
    entry = {
        "version": args.from_version,
        "to_version": args.to_version,
        **transitions[key],
        "evidence": {
            "development_status": source["status"],
            "failed_checks": failed,
            "validation_task_BA": source["validation"]["task_BA"],
            "PERSIST_minus_historical_reference": source["validation"]["PERSIST_minus_historical_reference"],
            "semantic_macro_gap": source["semantic"]["macro_gap_zp_minus_hf"],
            "MI_learned_minus_zero_BA": source["gates"]["mi"]["learned_minus_zero_BA"],
            "gate_means": {task: value["mean"] for task, value in source["gates"].items()},
        },
        "data_used": ["TRAIN", "VALIDATION"],
        "held_out_test_used": False,
    }
    payload["entries"].append(entry)
    write(path, payload)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
