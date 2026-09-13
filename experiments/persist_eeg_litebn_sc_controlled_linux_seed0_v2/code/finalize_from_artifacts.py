"""Regenerate reporting-only artifacts without training or evaluation."""
import json
import sys

sys.path.insert(0, "experiments/persist_eeg_litebn_sc_controlled_linux_seed0_v2/code")
import run_c1_wbcic as runner


decision = json.loads((runner.OUT / "CONTINUATION_DECISION.json").read_text(encoding="utf-8"))
provenance = json.loads((runner.PROTOCOL / "C1_CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))
runner.finalize(provenance["records"], decision["outer"], decision["internal_heldout"])
