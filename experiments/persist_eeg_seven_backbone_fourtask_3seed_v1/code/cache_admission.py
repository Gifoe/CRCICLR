"""Admit complete cache mirrors without opening labels, predictions, or outcomes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO = Path(os.environ.get("SEVEN_REPO", Path(__file__).resolve().parents[3])).resolve()
PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None); raise
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def require(paths: list[Path], label: str) -> list[str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"{label} missing {len(missing)} files, e.g. {missing[:3]}")
    return [str(path) for path in paths]


def main() -> int:
    partial = Path(os.environ["PERSIST_CACHE_ROOT"]).resolve()
    openbmi = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
    wbcic = Path(os.environ["FULL_WBCIC_CACHE"]).resolve()
    modern = load_module("seven_backbone_cache_modern", REPO / "experiments" / "persist_eeg_outcome_blind_modern_backbone_seed0_v1" / "code" / "modern_common.py")
    task_code = load_module("seven_backbone_cache_tasks", REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1" / "code" / "task_datasets.py")
    folds, search, split_sha = modern.load_split()
    task_search, _, task_reference, _ = task_code.split_reference()
    openbmi_subjects = sorted(search["OpenBMI"], key=lambda x:int(x)); wbcic_subjects = sorted(search["WBCIC"], key=lambda x:int(x.replace("sub-", "")))
    if openbmi_subjects != sorted(task_search, key=lambda x:int(x)): raise RuntimeError("MI and task-generalization OpenBMI search memberships disagree")
    checks: dict[str, list[Path]] = {"OpenBMI_MI": [], "OpenBMI_ERP": [], "OpenBMI_SSVEP": [], "WBCIC_MI": []}
    for subject in openbmi_subjects:
        for session in (1, 2):
            for task, cache_name in (("OpenBMI_MI", "mi"), ("OpenBMI_ERP", "erp"), ("OpenBMI_SSVEP", "ssvep")):
                base = openbmi / f"sub-{int(subject):02d}" / f"ses-{session}" / f"{cache_name}_1train"
                checks[task].extend([base.with_name(base.name + "_signals.npy"), base.with_name(base.name + "_codes.npy")])
    for subject in wbcic_subjects:
        for session in (0, 1, 2):
            checks["WBCIC_MI"].extend([wbcic / subject / f"ses-{session}_epochs.npy", wbcic / subject / f"ses-{session}_labels.npy"])
    # Byte-identical common signal files prove that the complete mirrors extend
    # the prior active cache rather than silently changing preprocessing.
    comparators = {
        "OpenBMI_MI_sub01_ses1": [partial / "openbmi" / "openbmi" / "sub-01" / "ses-1" / "mi_1train_signals.npy", openbmi / "sub-01" / "ses-1" / "mi_1train_signals.npy"],
        "WBCIC_MI_sub1_ses0": [partial / "wbcic" / "wbcic_epochs" / "sub-1" / "ses-0_epochs.npy", wbcic / "sub-1" / "ses-0_epochs.npy"],
    }
    comparisons = {}
    for label, paths in comparators.items():
        require(paths, label); hashes = [digest(path) for path in paths]
        comparisons[label] = {"paths":[str(path) for path in paths], "sha256":hashes, "identical":hashes[0] == hashes[1]}
    if not all(value["identical"] for value in comparisons.values()): raise RuntimeError("complete cache mirror fails byte-identity admission")
    record = {"pass":True, "access_scope":"filenames and signal-byte hashes only; no labels, predictions, scores, outer outcomes, or held-out outcomes were read",
              "frozen_split_sha256":split_sha, "task_split_sha256":task_reference["source_sha256"],
              "roots":{"partial":str(partial), "complete_openbmi":str(openbmi), "complete_wbcic":str(wbcic)},
              "search_subject_counts":{"OpenBMI":len(openbmi_subjects), "WBCIC":len(wbcic_subjects)},
              "required_file_counts":{task:len(require(paths, task)) for task, paths in checks.items()}, "identity_comparisons":comparisons,
              "outer_data_accessed":False, "fixed_heldout_accessed":False}
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    (PROTOCOL / "CACHE_SOURCE_REUSE_AUDIT.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CACHE_SOURCE_ADMISSION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
