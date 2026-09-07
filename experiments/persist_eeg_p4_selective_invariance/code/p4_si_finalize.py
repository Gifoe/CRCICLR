from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "persist_eeg_p4_selective_invariance"
VERSIONS = ("SI_V0", "SI_V1", "SI_V2", "SI_V3", "SI_V4")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    summaries: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for version in VERSIONS:
        path = OUT / f"P4_SI_DEVELOPMENT_SUMMARY_{version}.json"
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        summary = load_json(path)
        if summary.get("version") != version or summary.get("runs") != 6:
            raise RuntimeError(f"Invalid versioned summary provenance: {path}")
        if summary.get("outer_test_used") is not False:
            raise RuntimeError(f"Outer-test marker is not false: {path}")
        summaries[version] = summary
    if missing:
        raise RuntimeError(f"Missing versioned summaries: {missing}")

    adaptation_path = OUT / "protocol" / "P4_SI_ADAPTATION_LOG.json"
    adaptation = load_json(adaptation_path) if adaptation_path.exists() else []
    test_marker = OUT / "formal" / "TEST_ACCESS_STARTED.json"
    if test_marker.exists():
        raise RuntimeError("Formal test-access marker exists; closure must not claim test firewall")

    version_table = []
    for version in VERSIONS:
        summary = summaries[version]
        version_table.append(
            {
                "version": version,
                "decision": summary["decision"],
                "mean_task_delta": summary["mean_task_delta"],
                "gate_A_to_D_pass": summary["gate_A_to_D_pass"],
                "checks": summary["checks"],
                "outer_test_used": summary["outer_test_used"],
            }
        )

    final = summaries["SI_V4"]
    payload = {
        "status": "P4_SI_CLOSED",
        "decision": "P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED",
        "candidate_version": "SI_V4",
        "candidate_decision": final["decision"],
        "method": "PERSIST-SI Selective Persistence Invariance",
        "scientific_conclusion": (
            "Repeated-measure persistence and target relevance are detectable, and protected-vs-nuisance intervention "
            "audits are often valid, but the current selective representation intervention does not establish a stable "
            "validation generalization gain or the preregistered nuisance-suppression gate across the development panel."
        ),
        "version_table": version_table,
        "adaptation_entries": adaptation,
        "outer_test_used": False,
        "formal_evaluation_authorized": False,
        "method_lock_created": False,
        "missing_versioned_summaries": missing,
        "closure_reason": (
            "Five scientifically distinct development versions (SI-V0 through SI-V4) were evaluated on 3 folds x 2 seeds. "
            "No version met the preregistered lock/generalization requirements; further tuning would be unbounded search."
        ),
    }
    write_json(OUT / "P4_SI_FINAL_REPORT.json", payload)
    write_json(
        OUT / "P4_SI_LOCK_REFUSED.json",
        {
            "status": "LOCK_REFUSED",
            "decision": payload["decision"],
            "candidate_version": "SI_V4",
            "reason": payload["closure_reason"],
            "outer_test_used": False,
            "formal_evaluation_authorized": False,
        },
    )
    report = [
        "# PERSIST-SI P4 Closure",
        "",
        "- Decision: `P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED`",
        "- Formal method lock: `REFUSED`",
        "- Outer-test used: `false`",
        "- Formal outer evaluation authorized: `false`",
        "",
        "## Development versions",
        "",
        "| Version | MI ΔBA | ERP ΔBA | SSVEP ΔBA | A–D | Decision |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for version in VERSIONS:
        summary = summaries[version]
        delta = summary["mean_task_delta"]
        report.append(
            f"| {version} | {delta['mi']:+.4f} | {delta['erp']:+.4f} | {delta['ssvep']:+.4f} | "
            f"{str(summary['gate_A_to_D_pass']).lower()} | {summary['decision']} |"
        )
    report += [
        "",
        "## Closure",
        "",
        payload["scientific_conclusion"],
        "",
        payload["closure_reason"],
        "",
        "All method changes were fit/evaluated with TRAIN/VALIDATION only. No outer-test signal, label, embedding, or metric was accessed.",
    ]
    (OUT / "P4_SI_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        OUT / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "stage": "P4_SELECTIVE_INVARIANCE",
            "decision": payload["decision"],
            "outer_test_used": False,
            "formal_evaluation_authorized": False,
            "final_report": "P4_SI_FINAL_REPORT.json",
        },
    )
    print(json.dumps({"status": payload["status"], "decision": payload["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
