#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from hsc_tta.splits import make_internal_subject_split
from hsc_tta.utils import require_cpu


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")


def main() -> int:
    require_cpu("cpu")
    for dataset in ("hmc", "cap", "eegmmidb"):
        for seed in range(5):
            source = ROOT / f"data/splits/{dataset}/seed_{seed}.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            internal = make_internal_subject_split(payload["roles"], dataset, seed)
            output = ROOT / f"data/splits_internal/{dataset}/seed_{seed}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(internal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
