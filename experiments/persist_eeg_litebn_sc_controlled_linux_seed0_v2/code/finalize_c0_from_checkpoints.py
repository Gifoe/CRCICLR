"""Replay deterministic evaluations and regenerate the B0/C0/C1 report."""
import json
import sys
import torch

sys.path.insert(0, "experiments/persist_eeg_litebn_sc_controlled_linux_seed0_v2/code")
import run_c0_wbcic as runner


_, datasets, _ = runner.base.load_folds()
folds = {int(row["fold_id"]): row for row in datasets["WBCIC"]}
c0 = sorted(json.loads((runner.PROTOCOL / "C0_CHECKPOINT_PROVENANCE.json").read_text())["records"], key=lambda row: int(row["fold"]))
c1 = sorted(json.loads((runner.PROTOCOL / "C1_CHECKPOINT_PROVENANCE.json").read_text())["records"], key=lambda row: int(row["fold"]))
reference = sorted(
    json.loads((runner.shared.REFERENCE_EXP / "protocol" / "LITEBN_CHECKPOINT_PROVENANCE.json").read_text())["records"],
    key=lambda row: int(row["fold"]),
)
runner.evaluate_and_finalize(folds, c0, c1, reference, torch.device("cuda"))
