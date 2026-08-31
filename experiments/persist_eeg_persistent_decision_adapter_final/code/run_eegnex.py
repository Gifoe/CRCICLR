"""Cross-backbone gate; EEGNeX is never opened before ATCNet success."""
from __future__ import annotations

import json
import pda_core as c


def main() -> None:
    gate = json.loads((c.RESULTS / "SOURCE_GATE.json").read_text(encoding="utf-8"))
    if gate.get("source_gate_pass") is not True:
        print("EEGNeX_SKIPPED_ATCNET_SOURCE_GATE_FAILED")
        return
    raise RuntimeError("EEGNeX S2 evaluation requires the ATCNet future-session lock")


if __name__ == "__main__":
    main()
