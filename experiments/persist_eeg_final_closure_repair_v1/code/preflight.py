"""Server-side preflight for the frozen closure repair."""
from __future__ import annotations

import json

from common import (
    EXP,
    RUNTIME,
    audit_frozen_tables,
    certificate_coordinate_map,
    core,
    diag,
    load_authorized_data,
    protocol,
    state_sha256,
    write_json,
)
from matched_aux import AuxNet


def main() -> None:
    cfg = protocol()
    data = load_authorized_data()
    roles = core.outer_folds(data.search_subjects)
    if len(roles) != 5 or any(len(role["outcome"]) != 8 or len(role["source"]) != 32 for role in roles):
        raise RuntimeError("frozen fold cardinality changed")
    tables = audit_frozen_tables()
    if not tables["pass"]:
        raise RuntimeError(f"frozen artifact audit failed: {tables['issues']}")
    mappings = {}
    for fold in range(5):
        for seed in range(3):
            mapping = certificate_coordinate_map(fold, seed)
            if len(mapping) == 0:
                raise RuntimeError(f"historical PUD basis is empty: fold={fold} seed={seed}")
            mappings[f"{fold}/{seed}"] = mapping

    config = diag.eegnet_f8_config()
    first = AuxNet(config)
    second = AuxNet(config)
    init_seed = core.stable_seed("closure-repair-preflight-init")
    core.deterministic_reinitialize(first, init_seed)
    core.deterministic_reinitialize(second, init_seed)
    exact_full = state_sha256(first) == state_sha256(second)
    exact_main = state_sha256(first, prefixes=("encoder.", "head.")) == state_sha256(second, prefixes=("encoder.", "head."))
    phase_a_source = (EXP / "code" / "phase_a_repair.py").read_text(encoding="utf-8")
    forbidden_phase_a = [token for token in ("optimizer.step(", "train_single(", "fit_certificate(", "adapt_single(", "adapt_dual(") if token in phase_a_source]
    payload = {
        "pass": bool(exact_full and exact_main and not forbidden_phase_a),
        "protocol_schema": cfg["schema"],
        "authorized_rows": len(data.metadata),
        "authorized_subjects": len(data.search_subjects),
        "folds": len(roles),
        "frozen_table_audit": tables,
        "certificate_coordinate_mappings": mappings,
        "same_seed_full_initialization_exact": exact_full,
        "same_seed_main_initialization_exact": exact_main,
        "phase_a_forbidden_calls": forbidden_phase_a,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_json(RUNTIME / "PREFLIGHT.json", payload)
    if not payload["pass"]:
        raise RuntimeError(f"preflight failed: {payload}")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
