from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from validate_incremental_relation import validate


def test_compact_artifacts_recompute_locked_terminal():
    pilot = Path(__file__).resolve().parents[1]
    report = validate(pilot)
    assert report["pass"] is True
    assert report["terminal"] == "INCREMENTAL_RELATION_STOP_NO_CLEAR_GAIN"
    result = json.loads((pilot / "results/INCREMENTAL_RELATION_RESULT.json").read_text())
    assert result["final_claim_authorized"] is False
