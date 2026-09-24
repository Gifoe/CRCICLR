import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "second_five"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> None:
    cells = []
    for fold in range(5):
        cell = ROOT / f"eegnet_openbmi_ssvep_fold{fold}_seed0"
        provenance_path = cell / "PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = set(provenance["output_sha256"])
        actual = {path.name for path in cell.iterdir() if path.is_file()}
        assert actual == expected | {"PROVENANCE.json", "MECHANISM_CLOSURE_REPORT.md"}
        for name, expected_sha in provenance["output_sha256"].items():
            path = cell / name
            assert sha256(path) == expected_sha, (fold, name, "sha256")
            assert csv_rows(path) == provenance["output_rows"][name], (fold, name, "rows")
        report = cell / "MECHANISM_CLOSURE_REPORT.md"
        assert sha256(report) == provenance["report_sha256"], (fold, "report_sha256")
        assert provenance["status"] == "CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT"
        assert provenance["model"] == "EEGNet"
        assert provenance["task"] == "OpenBMI_SSVEP"
        assert provenance["fold"] == fold
        assert provenance["seed"] == 0
        assert provenance["final_heldout_accessed"] is False
        assert provenance["regime_metadata"]["outer_used_for_selection"] is False
        cells.append({
            "fold": fold,
            "directory": cell.name,
            "file_count": len(actual),
            "provenance_sha256": sha256(provenance_path),
            "bytes": sum(path.stat().st_size for path in cell.iterdir() if path.is_file()),
            "validated_output_files": len(expected),
        })
    result = {
        "status": "PASS",
        "scope": "EEGNet/OpenBMI_SSVEP folds 0-4, seed 0; compact previews only",
        "cell_count": len(cells),
        "final_heldout_accessed": False,
        "outer_used_for_selection": False,
        "checks": [
            "exact compact file inventory",
            "all provenance-declared output SHA-256 digests",
            "all provenance-declared CSV row counts",
            "report SHA-256 digest",
            "cell identity and compact-preview status",
            "final-heldout exclusion and no outer-outcome selection",
        ],
        "cells": cells,
    }
    (ROOT / "VALIDATION_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
