from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def numeric_subject_key(value: str) -> int:
    return int(str(value).replace("sub-", ""))


def load_search(repo: Path) -> dict[str, list[str]]:
    source = repo / "experiments" / "persist_eeg_transfer_geometry_stage1_v1" / "protocol" / "STAGE1_SEARCH_CV_SPLIT.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("protocol") != "STAGE1_SEARCH_ONLY" or raw.get("seed") != 0:
        raise RuntimeError("unexpected source split protocol")
    out: dict[str, list[str]] = {}
    for dataset, expected in (("OpenBMI", 40), ("WBCIC", 31)):
        vals = sorted((str(x) for x in raw["datasets"][dataset]["search_subjects"]), key=numeric_subject_key)
        if len(vals) != expected or len(set(vals)) != expected:
            raise RuntimeError(f"{dataset} SEARCH subject audit failed")
        out[dataset] = vals
    return out


def build(search: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset, subjects in search.items():
        perm = np.random.default_rng(0).permutation(np.asarray(subjects, dtype=object)).tolist()
        outer_parts = [list(map(str, x.tolist())) for x in np.array_split(np.asarray(perm, dtype=object), 5)]
        folds: list[dict[str, Any]] = []
        for fold_idx, outer in enumerate(outer_parts):
            remaining = [s for s in subjects if s not in set(outer)]
            rng = np.random.default_rng(1000 + fold_idx)
            rem = rng.permutation(np.asarray(remaining, dtype=object)).tolist()
            n_val = max(1, int(round(0.20 * len(remaining))))
            inner_val = [str(x) for x in rem[:n_val]]
            inner_train = [str(x) for x in rem[n_val:]]
            folds.append({
                "fold_id": fold_idx,
                "fold_seed": 0,
                "inner_split_seed": 1000 + fold_idx,
                "inner_train_subjects": inner_train,
                "inner_val_subjects": inner_val,
                "outer_dev_subjects": outer,
            })
        result[dataset] = folds
    return result


def split_payload(repo: Path) -> dict[str, Any]:
    search = load_search(repo)
    folds = build(search)
    return {
        "protocol": "CARRIER_5FOLD_MULTISEED_STABILITY_V1",
        "split_seed": 0,
        "inner_val_seed_base": 1000,
        "partition": "numpy.random.default_rng(0) then numpy.array_split(permuted_subjects, 5)",
        "search_subjects": search,
        "folds": folds,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    repo = Path(os.environ.get("R2EEG_REPO", "/root/rivermind-data/CRCICLR_EEGNET_PRD_WORK")).resolve()
    exp = repo / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1"
    protocol = exp / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    payload = split_payload(repo)
    path = protocol / "FIVEFOLD_SPLIT.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (protocol / "SPLIT_HASH.json").write_text(json.dumps({"path": str(path), "sha256": sha256(path)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "subjects": {k: len(v) for k, v in payload["search_subjects"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
