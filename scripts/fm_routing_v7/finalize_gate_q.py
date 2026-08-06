#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo = pathlib.Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "src"))
    from hsc_tta.fm_routing.models import load_biot, load_labram
    from hsc_tta.fm_routing.pipeline import Pipeline, atomic_json, utc_now, write_text

    pipeline = Pipeline(repo)
    state = json.loads((repo / "outputs/fm_routing_v7/RUN_STATE.json").read_text())
    if not state.get("terminal") or state.get("verdict") != "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE":
        raise SystemExit("finalizer requires the terminal Gate-Q scientific stop")
    passed = pipeline.qualify_models(transition_state=False)
    if passed:
        raise SystemExit("corrected Gate Q unexpectedly passes; do not mutate the terminal verdict")
    provenance_path = repo / "outputs/fm_routing_v7/audit/MODEL_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text())
    labram = load_labram(repo.parent, torch.device("cpu"))
    biot = load_biot(repo.parent, torch.device("cpu"))
    provenance["models"]["cbramod"]["frozen_encoder_parameters"] = provenance["models"]["cbramod"]["parameters_in_checkpoint"]
    provenance["models"]["labram"]["frozen_encoder_parameters"] = sum(value.numel() for value in labram.parameters())
    provenance["models"]["biot"]["frozen_encoder_parameters"] = sum(value.numel() for value in biot.parameters())
    atomic_json(provenance_path, provenance)
    write_text(repo / "delivery/fm_routing_v7/MODEL_PRETRAINING_PROVENANCE.md", pipeline._provenance_markdown(provenance))
    gate = pd.read_csv(repo / "outputs/fm_routing_v7/results/MODEL_QUALIFICATION_GATE.csv")
    pd.DataFrame([
        {"gate": "Gate Q", "status": "FAIL", "reason": "at least one frozen model failed qualification"},
        {"gate": "Gate A", "status": "NOT_RUN", "reason": "forbidden after Gate Q failure"},
    ]).to_csv(repo / "outputs/fm_routing_v7/results/GATE_SUMMARY.csv", index=False)
    atomic_json(repo / "outputs/fm_routing_v7/audit/QUALIFICATION_FINALIZATION.json", {
        "timestamp": utc_now(),
        "status": "corrected_without_new_scientific_jobs",
        "changes": [
            "inferred EEGMMIDB chance from its four frozen labels instead of a hard-coded class count",
            "computed dataset balanced accuracy from full OOF predictions per head seed instead of averaging fold scores",
            "checked predicted-class coverage and minimum per-seed nonconstant-subject coverage",
        ],
        "verdict_changed": False,
    })
    pipeline.build_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
