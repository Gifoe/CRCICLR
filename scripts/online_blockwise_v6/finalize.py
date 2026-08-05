from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hsc_tta.online_blockwise.pipeline import atomic_csv, atomic_json, sha256_file, write_text


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    output = repo / "outputs/online_blockwise_v6"
    delivery = repo / "delivery/online_blockwise_v6"
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)

    atomic_csv(pd.DataFrame(columns=["job", "error"]), output / "FAILURES.csv")
    backbone = pd.read_csv(output / "results/BACKBONE_GATE.csv")
    block = pd.read_csv(output / "results/BLOCK_PROTOCOL_GATE.csv")
    gate_a = pd.read_csv(output / "results/GATE_A.csv")
    rows = []
    for frame, gate in ((backbone, "BACKBONE"), (block, "BLOCK_PROTOCOL"), (gate_a, "A")):
        for row in frame.to_dict(orient="records"):
            rows.append({"gate": gate, "dataset": row["dataset"], "pass": bool(row["pass"]),
                         "details_json": json.dumps(row, sort_keys=True)})
    atomic_csv(pd.DataFrame(rows), output / "results/GATE_SUMMARY.csv")

    write_text(provenance / "COMMANDS.txt",
               "/root/miniconda3/envs/hsc_gpu/bin/python scripts/online_blockwise_v6/run_all.py "
               "--repo-root /root/autodl-tmp/hsc_tta_eeg/repo --resume\n"
               "PYTHONPATH=src /root/miniconda3/envs/hsc_gpu/bin/python -m pytest -q")
    write_text(provenance / "ENVIRONMENT.txt",
               f"python={sys.version}\nplatform={platform.platform()}\n"
               f"torch={__import__('torch').__version__}\npandas={pd.__version__}\n"
               f"cuda_available={__import__('torch').cuda.is_available()}")

    decision_path = delivery / "V6_STAGE0_DECISION.json"
    decision = json.loads(decision_path.read_text())
    state = json.loads((output / "RUN_STATE.json").read_text())
    decision["completed_cache_jobs"] = len(state.get("completed_jobs", []))
    decision["failed_jobs"] = len(state.get("failed_jobs", []))
    decision["stopping_gate"] = "Gate A"
    decision["later_stages_not_run"] = ["B1", "B2", "B3", "Gate B", "B4", "Gate C", "method development"]
    decision["finalized_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(decision_path, decision)

    hashes = {}
    for root in (output, delivery):
        for path in root.rglob("*"):
            if path.is_file() and path.name not in {"HASHES.json", "DELIVERY_MANIFEST.json"}:
                hashes[str(path.relative_to(repo))] = sha256_file(path)
    atomic_json(provenance / "HASHES.json", hashes)
    files = []
    for root in (output, delivery):
        for path in root.rglob("*"):
            if path.is_file() and path.name != "DELIVERY_MANIFEST.json":
                files.append({"path": str(path.relative_to(repo)), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    atomic_json(delivery / "DELIVERY_MANIFEST.json",
                {"created_at": datetime.now(timezone.utc).isoformat(), "files": files})


if __name__ == "__main__":
    main()
